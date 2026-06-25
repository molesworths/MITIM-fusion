"""
boundary.py
-----------
Separatrix boundary-condition models for PORTALS-Edge.

Public interface::

    model = build_bc_model(bc_model, bc_model_options)
    model.get_boundary_conditions(powerstate, batch_idx=0)
    bc_dict = model.bc_dict   # {var: [value, rho_location]}

``bc_model`` may be a string (legacy) or a dict::

    {"y": "<YModel>", "aLy": "<ALyModel>"}

Available y models:    "Fixed", "TFTP"
Available aLy models:  "Fixed", "PeretSSF", "EichManz", "Ensemble"

New-style options::

    bc_model         = {"y": "TFTP",   "aLy": "Ensemble"}
    bc_model_options = {
        "y":        {"ne_target": 3.0, "te_target": 0.005, "fq_e": 0.2},
        "aLy":      {"G0": 1.25, "shear_ref": 5.0},
        "combined": {"max_iter": 100, "tol": 1e-3, "damping": 0.5},
    }

Legacy string names remain fully supported:
    "Fixed" / "FixedInitial"            → y=Fixed,  aLy=Fixed
    "TwoFluid_PeretSSF" / ...           → y=TFTP,   aLy=PeretSSF
    "TwoFluid_EichManz" / "Tftp_SepOS"  → y=TFTP,   aLy=EichManz

Variable name conventions (match powerstate.plasma keys)
---------------------------------------------------------
ne      : electron density           [1e19 m⁻³]
te      : electron temperature       [keV]
ti      : ion temperature            [keV]
ni      : main-ion density           [1e19 m⁻³]
aLne    : e-density gradient scale   [dimensionless] = a/L_ne
aLte    : e-temp gradient scale      [dimensionless] = a/L_Te
aLti    : ion-temp gradient scale    [dimensionless] = a/L_Ti
aLni    : ion-density gradient scale [dimensionless] = a/L_ni

``bc_dict`` values use MITIM plasma-dict units (keV, 1e19 m⁻³, dimensionless).
The ``rho_location`` value of 1.0 pins the variable at rho = 1.0.

GP-based sensitivity analysis
------------------------------
After ``get_boundary_conditions()`` the model exposes:

    ``bc_leaf_tensors``  : dict[str, torch.Tensor]
        Scalar leaf tensors (requires_grad=True) mirroring ``bc_dict`` values in
        MITIM plasma-dict units.  Each call to ``get_boundary_conditions()``
        creates a fresh set — zero the grads between passes with
        ``[t.grad.zero_() for t in model.bc_leaf_tensors.values() if t.grad is not None]``.

    ``get_y_grad_profiles(y_profiles)``
        For ``defined_on='aLy'`` spline parameterizations, returns y profiles as
        torch tensors carrying grad back to the y-BC leaf tensors via the identity
        ``y(ρ) = y_bc · f(ρ)`` where the normalized shape ``f(ρ)`` is a numpy scalar.
        See method docstring for usage example.

    For aLy-BC sensitivity, one finite-difference pass through the parameterizer
    (cheap numpy splines) followed by a GP evaluation is the practical approach.
"""

import numpy as np
import torch
from scipy.constants import e as e_J, u as _u_kg
from scipy.optimize import fsolve, brentq

from mitim_tools.misc_tools import PLASMAtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools.misc_tools.PLASMAtools import md_u as _MD_U

try:
    from mitim_tools.edge_tools.equilibrium import load_equilibrium
except Exception:
    load_equilibrium = None

# ---------------------------------------------------------------------------
# LCFS state container — extracts all scalars using torch interpolation
# ---------------------------------------------------------------------------

class _LCFSState:
    """
    All LCFS scalars extracted from one powerstate batch element.

    Internal units: m⁻³ for densities, eV for temperatures, SI for geometry.
    Torch-based linear interpolation throughout for consistency with STATEedge.
    """

    @staticmethod
    def _interp1d(x: torch.Tensor, y: torch.Tensor, x0: float) -> float:
        """Linear interpolation of 1-D torch tensors at scalar x0 → float."""
        x = x.to(dtype=torch.double)
        y = y.to(dtype=torch.double)
        if x.numel() == 0:
            return float("nan")
        if x.numel() == 1:
            return float(y[0].item())
        if x[0] > x[-1]:
            x = torch.flip(x, dims=(0,))
            y = torch.flip(y, dims=(0,))
        x0_t = torch.tensor(x0, dtype=x.dtype, device=x.device).clamp(min=x[0], max=x[-1])
        idx_hi = int(torch.searchsorted(x, x0_t, right=False).item())
        idx_hi = max(1, min(idx_hi, x.numel() - 1))
        idx_lo = idx_hi - 1
        denom = (x[idx_hi] - x[idx_lo]).clamp(min=torch.finfo(x.dtype).eps)
        w = (x0_t - x[idx_lo]) / denom
        return float((y[idx_lo] + w * (y[idx_hi] - y[idx_lo])).item())

    @classmethod
    def extract(
        cls,
        powerstate,
        b: int = 0,
        Zeff_override: float | None = None,
        mi_ref_u: float = _MD_U,
        Lpar_override: float | None = None,
    ) -> "_LCFSState":
        s = cls()
        s._fill(powerstate, b, Zeff_override, mi_ref_u, Lpar_override)
        return s

    def _fill(self, powerstate, b, Zeff_override, mi_ref_u, Lpar_override):
        I = self._interp1d
        d = powerstate.profiles.derived
        p_raw = powerstate.profiles.profiles
        p = powerstate.plasma

        rho_t = p.get("rho", None)
        rho_ref: torch.Tensor | None = None
        if isinstance(rho_t, torch.Tensor):
            rho_ref = (rho_t[b, :] if rho_t.dim() == 2 else rho_t).detach().to(torch.double)

        def _plasma_lcfs(tensor, species_idx=None) -> float:
            if not isinstance(tensor, torch.Tensor):
                return float(tensor)
            if tensor.dim() == 0:
                return float(tensor.item())
            if tensor.dim() == 1:
                if rho_ref is not None and tensor.shape[0] == rho_ref.shape[0]:
                    return I(rho_ref, tensor.detach(), 1.0)
                return float(tensor[b if tensor.shape[0] > b else -1].item())
            if tensor.dim() == 2:
                arr = tensor[b, :].detach()
                return I(rho_ref, arr, 1.0) if rho_ref is not None else float(arr[-1].item())
            if tensor.dim() == 3:
                s_idx = 0 if species_idx is None else int(species_idx)
                arr = tensor[b, :, s_idx].detach()
                return I(rho_ref, arr, 1.0) if rho_ref is not None else float(arr[-1].item())
            return float(tensor.reshape(-1)[-1].item())

        rho_gacode = torch.as_tensor(np.asarray(p_raw["rho(-)"], dtype=float), dtype=torch.double)

        def _derived_lcfs(arr, species_idx=None) -> float:
            a_t = torch.as_tensor(np.asarray(arr, dtype=float), dtype=torch.double)
            if a_t.ndim == 2:
                a_t = a_t[:, 0 if species_idx is None else int(species_idx)]
            return I(rho_gacode, a_t, 1.0)

        self.ne = _plasma_lcfs(p["ne"]) * 1e19
        self.te = _plasma_lcfs(p["te"]) * 1e3
        self.ti = _plasma_lcfs(p["ti"], species_idx=0) * 1e3
        self.ni = _plasma_lcfs(p["ni"], species_idx=0) * 1e19

        if Zeff_override is not None:
            self.Zeff = float(Zeff_override)
        elif "Zeff" in p:
            self.Zeff = _plasma_lcfs(p["Zeff"])
        elif "Zeff" in d:
            self.Zeff = _derived_lcfs(d["Zeff"])
        else:
            self.Zeff = 1.0

        self.a   = float(d["a"])
        eps_lcfs = float(d["eps"])
        self.R0  = float(np.asarray(d["R_LF"])[-1])
        self.BT  = abs(_derived_lcfs(d["B_ref"]))
        self.q   = _derived_lcfs(p_raw["q(-)"])

        polflux_t   = torch.as_tensor(np.asarray(p_raw["polflux(Wb/radian)"], dtype=float), dtype=torch.double)
        r_t         = torch.as_tensor(np.asarray(d["r"], dtype=float), dtype=torch.double)
        dpolflux_dr = torch.gradient(polflux_t, spacing=(r_t,), dim=0)[0]
        self.Bp = abs(float((self.R0**(-1) * dpolflux_dr)[-1].item()))

        self.kappa = float(d["kappa995"])
        self.delta = float(d["delta995"])
        kappa_hat  = np.sqrt((1 + self.kappa**2 * (1 + 2*self.delta**2 - 1.2*self.delta**3)) / 2)
        self.q_cyl = kappa_hat * abs(self.BT) / max(self.Bp / eps_lcfs, 1e-30)
        self.shear = _derived_lcfs(d["s_hat"]) if "s_hat" in d else 1.0

        self.aLne = _plasma_lcfs(p["aLne"]) if "aLne" in p else _derived_lcfs(d["aLne"])
        self.aLte = _plasma_lcfs(p["aLte"]) if "aLte" in p else _derived_lcfs(d["aLTe"])
        self.aLti = (
            _plasma_lcfs(p["aLti"], species_idx=0) if "aLti" in p
            else _derived_lcfs(d["aLTi"], species_idx=0)
        )
        if "aLni0" in p:
            self.aLni = _plasma_lcfs(p["aLni0"])
        elif "aLni" in p:
            self.aLni = _plasma_lcfs(p["aLni"], species_idx=0)
        else:
            self.aLni = _derived_lcfs(d["aLni"], species_idx=0)

        # Plasma current [MA] and Greenwald density n_GW [m^-3] for the SepOS
        # density feasibility guard (section 4.5). Prefer derived/raw current,
        # fall back to Ampere's law from the poloidal field at the LCFS.
        Ip_MA = None
        if "Ip" in d:
            try:
                Ip_MA = abs(float(np.asarray(d["Ip"]).reshape(-1)[-1]))
            except Exception:
                Ip_MA = None
        if Ip_MA is None and "current(MA)" in p_raw:
            try:
                Ip_MA = abs(float(np.asarray(p_raw["current(MA)"]).reshape(-1)[0]))
            except Exception:
                Ip_MA = None
        if Ip_MA is None:
            mu0 = 4.0e-7 * np.pi
            Ip_MA = abs(2.0 * np.pi * self.a * kappa_hat * self.Bp / mu0) / 1e6
        self.Ip = Ip_MA
        self.n_GW = Ip_MA / max(np.pi * self.a**2, 1e-30) * 1e20  # m^-3

        volp_lcfs = float(np.asarray(d["volp_geo"])[-1])
        self.A = volp_lcfs
        _Qe_key = "Qe_edgetargets" if "Qe_edgetargets" in p else "QeMWm2_fixedtargets"
        _Qi_key = "Qi_edgetargets" if "Qi_edgetargets" in p else "QiMWm2_fixedtargets"
        self.Pe = _plasma_lcfs(p[_Qe_key]) * volp_lcfs
        self.Pi = _plasma_lcfs(p[_Qi_key]) * volp_lcfs

        self.mi_ref = mi_ref_u
        self._Lpar_override = Lpar_override
        self._update_kinetics_derived()

    def _update_kinetics_derived(self):
        te_keV = self.te * 1e-3
        ne_20  = self.ne * 1e-20
        self.rho_s  = PLASMAtools.rho_s(te_keV, self.mi_ref, self.BT)
        self.c_s    = PLASMAtools.c_s(te_keV, self.mi_ref)
        self.LogLam = float(
            PLASMAtools.loglam(
                torch.tensor(te_keV, dtype=torch.double),
                torch.tensor(ne_20, dtype=torch.double),
            ).item()
        )
        me_kg = 9.1094e-31
        self.me_over_mi = me_kg / (self.mi_ref * _u_kg)

    def update_kinetics(self, ne: float, te: float, ti: float):
        """Update (ne [m⁻³], te [eV], ti [eV]) and recompute derived quantities."""
        self.ne = ne
        self.te = te
        self.ti = ti
        self.ni = ne / max(self.Zeff, 1e-30)
        self._update_kinetics_derived()

    @property
    def Lpar(self) -> float:
        if self._Lpar_override is not None:
            return float(self._Lpar_override)
        return np.pi * abs(self.q_cyl) * self.R0


# ---------------------------------------------------------------------------
# Y models — compute ne, te, ti at the LCFS
# (prefixed Y_ to avoid name collision with aLy models below)
# ---------------------------------------------------------------------------

class _YModel:
    """Abstract base for y-value LCFS models."""
    def solve(self, state: _LCFSState, lambda_q: float) -> tuple[float, float, float]:
        """Return (ne [m⁻³], te [eV], ti [eV]) at the LCFS."""
        raise NotImplementedError


class Y_Fixed(_YModel):
    """
    Freeze LCFS values. On the first call, values are read from ``options``
    (ne in 1e19 m⁻³, te/ti in keV, same as MITIM plasma-dict units) with
    fallback to the current powerstate. Subsequent calls are no-ops.
    """
    def __init__(self, options: dict):
        self._opts = dict(options)
        self._fixed_ne: float | None = None
        self._fixed_te: float | None = None
        self._fixed_ti: float | None = None

    def _init(self, state: _LCFSState):
        self._fixed_ne = float(self._opts["ne"]) * 1e19 if "ne" in self._opts else state.ne
        self._fixed_te = float(self._opts["te"]) * 1e3  if "te" in self._opts else state.te
        self._fixed_ti = float(self._opts["ti"]) * 1e3  if "ti" in self._opts else state.ti

    def solve(self, state: _LCFSState, lambda_q: float) -> tuple[float, float, float]:
        if self._fixed_ne is None:
            self._init(state)
        return self._fixed_ne, self._fixed_te, self._fixed_ti


class Y_TFTP(_YModel):
    """
    Two-fluid two-point model: given divertor target conditions and a
    heat-flux width lambda_q, back-calculate upstream (LCFS) values.

    Parameters
    ----------
    ne_target        : float  Target density [1e19 m⁻³], default 3.0
    te_target        : float  Target electron temperature [keV], default 0.005
    ti_target        : float or None  Target ion temperature [keV]; defaults to te_target
    fq_e             : float  Electron heat conduction fraction, default 0.2
    fq_i             : float  Ion heat conduction fraction, default 0.2
    fmom             : float  Momentum-loss factor (used for ne_u), default 0.5
    lambda_q_scalar  : float  Multiplicative correction to lambda_q, default 1.0
    """
    def __init__(self, options: dict):
        self.ne_target       = float(options.get("ne_target", 3.0)) * 1e19
        self.te_target       = float(options.get("te_target", 0.005)) * 1e3
        _ti_target           = options.get("ti_target", None)
        self.ti_target       = float(_ti_target) * 1e3 if _ti_target is not None else None
        self.fq_e            = float(options.get("fq_e", options.get("fmom", 0.5)))
        self.fq_i            = float(options.get("fq_i", options.get("fmom", 0.5)))
        self.fmom            = float(options.get("fmom", 0.5))
        self.lambda_q_scalar = float(options.get("lambda_q_scalar", 1.0))

    def solve(self, state: _LCFSState, lambda_q: float) -> tuple[float, float, float]:
        Ti_target = self.ti_target if self.ti_target is not None else self.te_target
        ne_t = self.ne_target
        Te_t = self.te_target
        Ti_t = Ti_target

        kappa_0e = 2600.0 / (0.672 + 0.076 * state.Zeff**0.5 + 0.252 * state.Zeff)
        kappa_0i = kappa_0e * state.me_over_mi**0.5

        Apar_sol = 4.0 * np.pi * state.R0 * self.lambda_q_scalar * lambda_q * (state.Bp / state.BT)
        qpar_e   = state.Pe * 1e6 / max(Apar_sol, 1e-30)
        qpar_i   = state.Pi * 1e6 / max(Apar_sol, 1e-30)
        Lpar     = state.Lpar

        Te_u = max(Te_t, (Te_t**3.5 + 3.5 * self.fq_e * qpar_e * Lpar / kappa_0e)**(2.0/7.0))
        Ti_u = max(Ti_t, (Ti_t**3.5 + 3.5 * self.fq_i * qpar_i * Lpar / kappa_0i)**(2.0/7.0))
        ne_u = (2.0 * ne_t * (Te_t + Ti_t)) / max(self.fmom * (Te_u + Ti_u / max(state.Zeff, 1e-30)), 1e-30)

        return ne_u, Te_u, Ti_u


# ---------------------------------------------------------------------------
# aLy models — compute gradient scale lengths and lambda_q at the LCFS
# (prefixed ALy_ to avoid name collision with y models above)
# ---------------------------------------------------------------------------

def _ssf_decay_lengths(
    state: "_LCFSState",
    G0: float,
    alpha_s: float,
    f_Delta: float = 1.0,
    Lambda: float | None = None,
) -> tuple[float, float, float, float, float]:
    """
    Peret 2025 SSF turbulent decay lengths (section 4.2).

    Temperature- and geometry-driven (independent of density). Solves the
    λ_p/ρ_s fixed point with **brentq** (bracketed root, robust) using the
    **natural log** sheath factor, then returns

        (λ_p, λ_n, λ_T, λ_q, γ)   [m, m, m, m, dimensionless].
    """
    te = state.te; ti = state.ti
    rho_s = state.rho_s
    R0 = state.R0; Lpar = state.Lpar

    if Lambda is None:
        Lambda = 0.5 * np.log(1.0 / (2.0 * np.pi * state.me_over_mi))

    gamma_0 = 2.5 * ti / te - 0.5 * np.log(2.0 * np.pi * state.me_over_mi * (1.0 + ti / te))
    gamma = 2.0 * gamma_0 / 3.0
    g = G0 * rho_s / R0

    def residual(lp: float) -> float:
        lp = max(float(lp), 1e-12 * rho_s)
        sqrt_gamma = np.sqrt(gamma)
        lT = sqrt_gamma / (sqrt_gamma - 1.0) * lp
        beta = f_Delta * (1.0 / Lambda + lp / lT)
        alpha_ExB = -0.43 * beta * Lambda * (rho_s / lp) ** 1.5 / g ** 0.5
        lhs = lp / rho_s
        rhs = (3.9 * g ** (3.0 / 11.0) * (2.0 * rho_s / Lpar) ** (-6.0 / 11.0)
               * gamma ** (-4.0 / 11.0) / (1.0 + (alpha_s + alpha_ExB) ** 2) ** (9.0 / 11.0))
        return lhs - rhs

    # Scan a log grid for a sign change, then bracket with brentq (section 4.2).
    grid = rho_s * np.logspace(np.log10(1.0), np.log10(1000.0), 64)
    fvals = np.array([residual(x) for x in grid])
    sign_change = np.where(np.sign(fvals[:-1]) != np.sign(fvals[1:]))[0]
    if sign_change.size == 0:
        raise RuntimeError("SSF: no bracketed root for lambda_p over [rho_s, 1000 rho_s]")
    i = int(sign_change[0])
    lambda_p = float(brentq(residual, grid[i], grid[i + 1], xtol=1e-6 * rho_s, rtol=1e-8))

    sqrt_gamma = np.sqrt(gamma)
    lambda_n = sqrt_gamma * lambda_p
    lambda_T = sqrt_gamma / (sqrt_gamma - 1.0) * lambda_p
    lambda_q = (2.0 / 7.0) * lambda_T
    return lambda_p, lambda_n, lambda_T, lambda_q, gamma


class _aLyModel:
    """Abstract base for aLy (gradient-scale-length) LCFS models."""
    def solve(self, state: _LCFSState) -> tuple[float, float, float, float, float]:
        """Return (aLne, aLte, aLti, aLni [dimensionless], lambda_q [m])."""
        raise NotImplementedError


class aLy_Fixed(_aLyModel):
    """
    Freeze aLy values from ``options`` (dimensionless) with fallback to
    the current powerstate on the first call.  lambda_q is inferred from
    aLte via the Spitzer-Härm relation lambda_q = (2/7) * a / aLte.
    """
    def __init__(self, options: dict):
        self._opts = dict(options)
        self._fixed: dict | None = None

    def _init(self, state: _LCFSState):
        aLte = float(self._opts.get("aLte", state.aLte))
        lambda_q = (2.0/7.0) * state.a / max(aLte, 1e-30)
        self._fixed = {
            "aLne":     float(self._opts.get("aLne", state.aLne)),
            "aLte":     aLte,
            "aLti":     float(self._opts.get("aLti", state.aLti)),
            "aLni":     float(self._opts.get("aLni", state.aLni)),
            "lambda_q": lambda_q,
        }

    def solve(self, state: _LCFSState) -> tuple[float, float, float, float, float]:
        if self._fixed is None:
            self._init(state)
        f = self._fixed
        return f["aLne"], f["aLte"], f["aLti"], f["aLni"], f["lambda_q"]


class aLy_PeretSSF(_aLyModel):
    """
    Peret 2025 SSF turbulent flux-decay-length model.

    Parameters
    ----------
    G0               : float  Curvature-drive geometry factor, default 1.25
    shear_ref        : float  Reference shear (+ for LSN, − for USN), default 5.0
    f_Delta          : float  Delta correction factor, default 1.0
    Lambda           : float or None  Sheath log factor; computed from me/mi if None
    gfile_path       : str or None  Path to g-file for equilibrium-derived G0/alpha_s
    eq_n_points      : int   Flux-surface discretization points, default 512
    eq_eddy_width_m  : float Eddy width for SSF geometry, default 0.005
    """
    def __init__(self, options: dict):
        self.G0               = float(options.get("G0", 1.25))
        self.shear_ref        = float(options.get("shear_ref", 5.0))
        self.f_Delta          = float(options.get("f_Delta", 1.0))
        self.Lambda           = options.get("Lambda", None)
        self._gfile_path      = options.get("gfile_path", options.get("gfile", None))
        self._eq              = None
        self._eq_n_points     = int(options.get("eq_n_points", 512))
        self._eq_eddy_width_m = float(options.get("eq_eddy_width_m", 0.005))
        self.verbose          = bool(options.get("verbose", False))

    def _load_equilibrium(self):
        if load_equilibrium is None or self._gfile_path is None:
            return None
        if self._eq is not None:
            return self._eq
        try:
            self._eq = load_equilibrium(
                self._gfile_path,
                n_points=self._eq_n_points,
                eddy_width_m=self._eq_eddy_width_m,
            )
        except Exception as exc:
            if self.verbose:
                print(f"[PeretSSF] equilibrium load failed ({exc}); using fallback.", typeMsg="w")
            self._gfile_path = None
        return self._eq

    def solve(self, state: _LCFSState) -> tuple[float, float, float, float, float]:
        te = state.te; ti = state.ti
        rho_s = state.rho_s
        R0 = state.R0; Lpar = state.Lpar; a = state.a

        Lambda = (
            0.5 * np.log(1.0 / (2.0 * np.pi * state.me_over_mi))
            if self.Lambda is None else float(self.Lambda)
        )

        eq = self._load_equilibrium()
        if eq is not None:
            G0      = float(eq.G0)
            alpha_s = float(eq.alpha_s)
        else:
            G0      = self.G0
            alpha_s = -self.shear_ref / max(abs(state.shear), 0.1)

        lambda_p, lambda_n, lambda_T, lambda_q, gamma = _ssf_decay_lengths(
            state, G0=G0, alpha_s=alpha_s, f_Delta=self.f_Delta, Lambda=Lambda,
        )

        aLne = a / max(lambda_n, 1e-30)
        aLte = a / max(lambda_T, 1e-30)
        aLti = (te / max(ti, 1e-30)) * aLte
        aLni = aLne

        return aLne, aLte, aLti, aLni, lambda_q


class aLy_EichManz(_aLyModel):
    """
    Eich–Manz SepOS (Separatrix Operating Space) scale-length scaling.
    """
    def __init__(self, options: dict):
        self.verbose = bool(options.get("verbose", False))

    def solve(self, state: _LCFSState) -> tuple[float, float, float, float, float]:
        ti = state.ti; te = state.te; ne = state.ne
        rho_s = state.rho_s; c_s = state.c_s
        R0 = state.R0; a = state.a

        rho_s_pol = rho_s * (state.BT / max(state.Bp, 1e-30))
        nuei_cgs  = 2.91e-6 * (ne * 1e-6) * state.LogLam / max(te**1.5, 1e-30)
        alpha_t   = ((1.0 + ti/te) * state.me_over_mi
                     * nuei_cgs * state.q_cyl**2 * R0 / max(c_s, 1e-30))

        lambda_n = 2.9 * (1.0 + 10.4 * alpha_t**2.5) * rho_s_pol
        lambda_T = 2.1 * (1.0 + 2.1  * alpha_t**1.7) * rho_s_pol
        lambda_q = (2.0/7.0) * lambda_T

        aLne = a / max(lambda_n, 1e-30)
        aLte = a / max(lambda_T, 1e-30)
        aLti = (te / max(ti, 1e-30)) * aLte
        aLni = aLne

        return aLne, aLte, aLti, aLni, lambda_q


class aLy_Ensemble(_aLyModel):
    """
    Arithmetic mean of PeretSSF and EichManz.  Falls back to whichever
    succeeds if one fails.  Per-model options may be nested under "PeretSSF"
    and "EichManz" keys within the options dict.
    """
    def __init__(self, options: dict):
        peret_opts = dict(options.get("PeretSSF", options))
        eich_opts  = dict(options.get("EichManz",  options))
        self._peret  = aLy_PeretSSF(peret_opts)
        self._eich   = aLy_EichManz(eich_opts)
        self.verbose = bool(options.get("verbose", False))

    def solve(self, state: _LCFSState) -> tuple[float, float, float, float, float]:
        results = []
        for name, model in (("PeretSSF", self._peret), ("EichManz", self._eich)):
            try:
                results.append(model.solve(state))
            except Exception as exc:
                if self.verbose:
                    print(f"[Ensemble] {name} failed: {exc}", typeMsg="w")
        if not results:
            raise RuntimeError("[Ensemble] Both PeretSSF and EichManz failed")
        n = len(results)
        return tuple(sum(r[i] for r in results) / n for i in range(5))


# ---------------------------------------------------------------------------
# Combined iterative solver
# ---------------------------------------------------------------------------

class CombinedBCModel:
    """
    Iterative solver coupling a y-model and an aLy-model.

    Each iteration:
      1. Update kinetic state with current (ne, te, ti).
      2. aLy-model: (ne, te, ti) → aLy values + lambda_q.
      3. y-model:   lambda_q     → new (ne, te, ti).
      4. Damp updates and check convergence.

    Fixed models settle immediately (convergence in 1 iteration).

    After ``get_boundary_conditions()`` the following attributes are populated:

    ``bc_dict``          : {key: [float_val_in_MITIM_units, rho_loc]}
    ``bc_leaf_tensors``  : {key: torch.Tensor, requires_grad=True}
        Scalar leaf tensors mirroring ``bc_dict``.  Use with
        ``get_y_grad_profiles()`` to build a differentiable path from y-BC
        values to downstream GP predictions.

    Parameters (via options / "combined" sub-dict)
    -----------------------------------------------
    max_iter  : int    Maximum iterations, default 100
    tol       : float  Convergence tolerance (relative change), default 1e-3
    damping   : float  Damping factor ∈ (0, 1], default 0.5
    verbose   : bool   Print warnings, default False
    Zeff      : float or None  Override Zeff from powerstate
    mi_ref_u  : float  Reference ion mass in atomic units, default md_u
    Lpar      : float or None  Override parallel connection length [m]
    """

    def __init__(self, y_model: _YModel, aly_model: _aLyModel, options: dict):
        self.y_model   = y_model
        self.aly_model = aly_model
        self.max_iter  = int(options.get("max_iter", 100))
        self.tol       = float(options.get("tol", 1e-3))
        self.damping   = float(options.get("damping", 0.5))
        self.verbose   = bool(options.get("verbose", False))
        self._Zeff_override = options.get("Zeff", None)
        self._mi_ref_u      = float(options.get("mi_ref_u", _MD_U))
        self._Lpar_override = options.get("Lpar", None)
        self.bc_dict: dict = {}
        self.bc_leaf_tensors: dict[str, torch.Tensor] = {}
        self.converged: bool = False

    def get_boundary_conditions(self, powerstate, batch_idx: int = 0) -> None:
        state = _LCFSState.extract(
            powerstate,
            b=batch_idx,
            Zeff_override=self._Zeff_override,
            mi_ref_u=self._mi_ref_u,
            Lpar_override=self._Lpar_override,
        )
        self._solve(state)

    def _solve(self, state: _LCFSState) -> None:
        ne, te, ti = state.ne, state.te, state.ti
        converged  = False

        for _ in range(self.max_iter):
            ne_old, te_old, ti_old = ne, te, ti
            state.update_kinetics(ne, te, ti)

            try:
                aLne, aLte, aLti, aLni, lambda_q = self.aly_model.solve(state)
            except Exception as exc:
                if self.verbose:
                    print(f"[CombinedBC] aLy model failed: {exc}", typeMsg="w")
                break

            try:
                ne_new, te_new, ti_new = self.y_model.solve(state, lambda_q)
            except Exception as exc:
                if self.verbose:
                    print(f"[CombinedBC] y model failed: {exc}", typeMsg="w")
                break

            d  = self.damping
            ne = (1.0 - d) * ne + d * ne_new
            te = (1.0 - d) * te + d * te_new
            ti = (1.0 - d) * ti + d * ti_new

            rel = max(
                abs((ne - ne_old) / max(ne_old, 1e-30)),
                abs((te - te_old) / max(te_old, 1e-30)),
                abs((ti - ti_old) / max(ti_old, 1e-30)),
            )
            if rel < self.tol:
                converged = True
                break

        if not converged and self.verbose:
            print("[CombinedBC] Did not converge within max_iter", typeMsg="w")
        self.converged = converged

        state.update_kinetics(ne, te, ti)
        try:
            aLne, aLte, aLti, aLni, _ = self.aly_model.solve(state)
        except Exception:
            aLne, aLte, aLti, aLni = state.aLne, state.aLte, state.aLti, state.aLni

        ni = ne / max(state.Zeff, 1e-30)

        self.bc_dict = {
            "ne":   [ne   * 1e-19, 1.0],
            "te":   [te   * 1e-3,  1.0],
            "ti":   [ti   * 1e-3,  1.0],
            "ni":   [ni   * 1e-19, 1.0],
            "aLne": [aLne,         1.0],
            "aLte": [aLte,         1.0],
            "aLti": [aLti,         1.0],
            "aLni": [aLni,         1.0],
        }

        # Leaf tensors for GP-based sensitivity analysis (requires_grad=True).
        # Same MITIM plasma-dict units as bc_dict.
        # Re-created each call so each analysis pass starts with fresh leaves.
        self.bc_leaf_tensors = {
            key: torch.tensor(float(val), dtype=torch.double, requires_grad=True)
            for key, (val, _) in self.bc_dict.items()
        }

    def get_y_grad_profiles(
        self,
        y_profiles: "dict[str, np.ndarray | torch.Tensor]",
        channels: "list[str] | None" = None,
    ) -> "dict[str, torch.Tensor]":
        """
        Reconstruct y profiles as torch tensors that carry gradient back to
        the y-BC leaf tensors in ``bc_leaf_tensors``.

        Valid for ``defined_on='aLy'`` parameterizations where the profile
        factorises as::

            y(ρ) = y_bc · exp(−∫ aLy dρ′)  ≡  y_bc · f(ρ, X)

        The normalised shape ``f(ρ) = y_np / float(y_bc)`` is computed without
        grad (it depends on the spline knots X, not the BC value).  Multiplying
        by the leaf tensor restores the differentiable connection::

            y_grad(ρ) = bc_leaf_tensor · torch.as_tensor(f(ρ))

        Parameters
        ----------
        y_profiles : dict
            Numpy or detached torch arrays keyed by plasma channel name (e.g.
            ``"te"``, ``"ne"``), in MITIM plasma-dict units (keV, 1e19 m⁻³).
            Typically obtained as
            ``{ch: powerstate.plasma[ch][b].detach().cpu().numpy()}``.
        channels : list[str] or None
            Channels to process; defaults to the intersection of y_profiles
            and bc_leaf_tensors keys.

        Returns
        -------
        dict[str, torch.Tensor]
            ``{channel: tensor(n_rho)}`` with grad path to bc_leaf_tensors.

        Example — sensitivity of GP objective to te_lcfs
        -------------------------------------------------
        >>> powerstate.calculateBoundaryConditions()
        >>> bc_model = powerstate._bc_model_instance
        >>> te_np = powerstate.plasma["te"][0].detach().numpy()
        >>> grad_profiles = bc_model.get_y_grad_profiles({"te": te_np})
        >>> te_grad = grad_profiles["te"]          # tensor, requires_grad via te_lcfs leaf
        >>> obj = gp_surrogate(te_grad, ...)        # differentiable GP call
        >>> obj.backward()
        >>> sens_te = bc_model.bc_leaf_tensors["te"].grad   # ∂obj/∂te_lcfs
        >>>
        >>> # For aLy-BC sensitivity use one finite-difference pass instead:
        >>> # aLy BCs pin the spline shape (not a multiplicative factor) so the
        >>> # algebraic shortcut does not apply.  Perturb bc_dict["aLte"] by δ,
        >>> # re-run parameterizer.update() (cheap numpy), evaluate GP, divide.
        """
        if not self.bc_leaf_tensors:
            raise RuntimeError(
                "bc_leaf_tensors empty — call get_boundary_conditions() first."
            )

        keys = channels if channels is not None else list(y_profiles.keys())
        out: dict[str, torch.Tensor] = {}

        for ch in keys:
            if ch not in y_profiles or ch not in self.bc_leaf_tensors:
                continue

            y_raw = y_profiles[ch]
            y_np = (
                y_raw.detach().cpu().numpy()
                if isinstance(y_raw, torch.Tensor)
                else np.asarray(y_raw, dtype=float)
            )

            bc_leaf  = self.bc_leaf_tensors[ch]
            bc_float = float(bc_leaf.detach())

            if abs(bc_float) < 1e-30:
                continue

            # f(ρ) = y(ρ) / y_bc: pure numpy, no grad needed.
            # Multiplying by the leaf tensor is the sole source of grad.
            shape_t = torch.as_tensor(y_np / bc_float, dtype=torch.double)
            out[ch] = bc_leaf * shape_t

        return out


class TwoFluidSynthesis:
    """
    Self-consistent LCFS model (lcfs_bc_model.md sections 4-5).

    Couples, at a 2-D fixed point in (T_e,u, T_i,u):
      - electron conduction (section 4.1, f_cond,e = 1, Spitzer-Harm 2/7),
      - SSF turbulent decay lengths (section 4.2, brentq, natural log),
      - a 0-D ion energy balance with e-i equipartition (section 4.3, R_th).
    Gradients follow section 4.4 (R_th-blended a/L_Ti with a finite sheath
    floor). The separatrix density is a *prescribed* Greenwald fraction
    (section 4.5, ``f_GW`` user-given, not a DV), clamped to the SepOS
    ideal-ballooning ceiling and L-mode density-limit floor.

    Interface mirrors ``CombinedBCModel`` so PORTALS-Edge can use it
    interchangeably; additionally exposes ``F_y`` (prior means for the
    soft-prior residuals) and ``sigma_y`` (per-channel stiffness, section 6).
    """

    _ME_KG = 9.1094e-31

    def __init__(self, options: dict):
        o = dict(options or {})
        self.te_target = float(o.get("te_target", 0.005)) * 1e3  # eV
        _ti_t = o.get("ti_target", None)
        self.ti_target = float(_ti_t) * 1e3 if _ti_t is not None else None
        self.f_GW = float(o.get("f_GW", 0.3))
        self.gamma_i = float(o.get("gamma_i_sheath", 2.5))
        self.G0 = float(o.get("G0", 1.25))
        self.shear_ref = float(o.get("shear_ref", 5.0))
        self.f_Delta = float(o.get("f_Delta", 1.0))
        self.Lambda = o.get("Lambda", None)
        self._gfile_path = o.get("gfile_path", o.get("gfile", None))
        self._eq_n_points = int(o.get("eq_n_points", 512))
        self._eq_eddy_width_m = float(o.get("eq_eddy_width_m", 0.005))
        self._eq = None
        self.max_iter = int(o.get("max_iter", 100))
        self.tol = float(o.get("tol", 1e-4))
        self.verbose = bool(o.get("verbose", False))
        self._Zeff_override = o.get("Zeff", None)
        self._mi_ref_u = float(o.get("mi_ref_u", _MD_U))
        self._Lpar_override = o.get("Lpar", None)
        self.sepos_clamp = bool(o.get("sepos_clamp", False))  # disconnected (section 4.5); kept for later
        # Per-channel soft-prior stiffness sigma_y (section 6): tight on the
        # conduction-set a/L_Te, looser on a/L_ne, loosest on the weakly-modeled
        # a/L_Ti. Consumed by powerstate_edge if present.
        self.sigma_y = {
            "aLte": float(o.get("sigma_aLte", 0.5)),
            "aLne": float(o.get("sigma_aLne", 0.5)),
            "aLti": float(o.get("sigma_aLti", 0.5)),
        }

        self.bc_dict: dict = {}
        self.bc_leaf_tensors: dict[str, torch.Tensor] = {}
        self.F_y: dict = {}
        self.converged: bool = False

    # ------------------------------------------------------------------
    def _geometry_drive(self, state: "_LCFSState") -> tuple[float, float]:
        """Return (G0, alpha_s) from the 2-D equilibrium if available, else heuristic."""
        if load_equilibrium is not None and self._gfile_path is not None:
            try:
                if self._eq is None:
                    self._eq = load_equilibrium(
                        self._gfile_path,
                        n_points=self._eq_n_points,
                        eddy_width_m=self._eq_eddy_width_m,
                    )
                return float(self._eq.G0), float(self._eq.alpha_s)
            except Exception as exc:
                if self.verbose:
                    print(f"[Synthesis] equilibrium load failed ({exc}); heuristic.", typeMsg="w")
                self._gfile_path = None
        return self.G0, -self.shear_ref / max(abs(state.shear), 0.1)

    @staticmethod
    def _kappa0e(Zeff: float) -> float:
        return 2600.0 / (0.672 + 0.076 * Zeff**0.5 + 0.252 * Zeff)

    def _kinetics(self, state, ne, Te, Ti, G0, alpha_s):
        """One pass of the coupled model: (ne,Te,Ti) -> predicted (Te,Ti) + diagnostics."""
        state.update_kinetics(ne, Te, Ti)
        lam_p, lam_n, lam_T, lam_q, gamma = _ssf_decay_lengths(
            state, G0=G0, alpha_s=alpha_s, f_Delta=self.f_Delta, Lambda=self.Lambda,
        )

        Apar = 4.0 * np.pi * state.R0 * lam_q * (state.Bp / state.BT)
        q_eu = state.Pe * 1e6 / max(Apar, 1e-30)
        q_iu = state.Pi * 1e6 / max(Apar, 1e-30)
        Lpar = state.Lpar

        # 4.1 electron conduction (f_cond,e = 1)
        kappa0e = self._kappa0e(state.Zeff)
        Te_new = max(self.te_target, (self.te_target**3.5 + 3.5 * q_eu * Lpar / kappa0e) ** (2.0 / 7.0))

        # 4.3 ion temperature: conduction-limited decoupled value (kappa_0i = kappa_0e sqrt(me/mi)),
        # blended to Te by e-i equipartition. The weak ion conductivity enforces Te <= Ti <= Ti_dec
        # with no ad hoc clamp; Ti_dec/Te ~ ((q_iu/q_eu)(mi/me)^0.5)^(2/7) ~ (mi/me)^(1/7) at equal
        # powers (~2 at q_iu = q_eu/5, ~5 for ion-heavy splits).
        Ti_t = self.ti_target if self.ti_target is not None else self.te_target
        mi_kg = self._mi_ref_u * _u_kg
        ne_cm3 = ne * 1e-6
        tau_e = 3.44e5 * Te_new**1.5 / max(ne_cm3 * state.LogLam, 1e-30)  # s (NRL e-i collision time)
        c_s = np.sqrt(max(e_J * Te_new, 1e-30) / mi_kg)                  # m/s

        kappa0i = kappa0e * np.sqrt(state.me_over_mi)                    # ion Spitzer-Harm
        Ti_dec = max(Ti_t, (Ti_t**3.5 + 3.5 * q_iu * Lpar / kappa0i)**(2.0/7.0))
        R_th = (Lpar / max(c_s, 1e-30)) * state.me_over_mi / max(tau_e, 1e-30)  # tau_par,i / tau_eq
        w = R_th / (1.0 + R_th)
        Ti_new = max(Ti_t, w * Te_new + (1.0 - w) * Ti_dec)

        diag = dict(lam_p=lam_p, lam_n=lam_n, lam_T=lam_T, lam_q=lam_q,
                    gamma=gamma, R_th=R_th)
        return Te_new, Ti_new, diag

    def get_boundary_conditions(self, powerstate, batch_idx: int = 0) -> None:
        state = _LCFSState.extract(
            powerstate, b=batch_idx,
            Zeff_override=self._Zeff_override,
            mi_ref_u=self._mi_ref_u, Lpar_override=self._Lpar_override,
        )
        G0, alpha_s = self._geometry_drive(state)

        # Density is a prescribed Greenwald fraction (fixed, section 4.5).
        ne_sep = self.f_GW * state.n_GW

        # 2-D fixed point on (Te, Ti) (section 5). Picard with under-relaxation;
        # robust because the temperature map is a contraction (section 2).
        Te, Ti = state.te, state.ti
        self.converged = False
        for _ in range(self.max_iter):
            Te_new, Ti_new, diag = self._kinetics(state, ne_sep, Te, Ti, G0, alpha_s)
            rel = max(abs(Te_new - Te) / max(Te, 1e-30), abs(Ti_new - Ti) / max(Ti, 1e-30))
            Te, Ti = Te_new, Ti_new
            if rel < self.tol:
                self.converged = True
                break
        if not self.converged and self.verbose:
            print("[Synthesis] (Te,Ti) fixed point did not converge", typeMsg="w")

        # Final diagnostics at the converged temperatures.
        _, _, diag = self._kinetics(state, ne_sep, Te, Ti, G0, alpha_s)
        a = state.a
        aLp = a / max(diag["lam_p"], 1e-30)
        aLne = a / max(diag["lam_n"], 1e-30)
        aLte = a / max(diag["lam_T"], 1e-30)

        # 4.4 R_th-blended ion gradient with finite sheath floor.
        gamma_i_blend = (2.0 / 3.0) * self.gamma_i
        sg_i = np.sqrt(gamma_i_blend)
        aLti_dec = (sg_i - 1.0) / sg_i * aLp
        w = diag["R_th"] / (1.0 + diag["R_th"])
        aLti = w * aLte + (1.0 - w) * aLti_dec
        aLni = aLne

        # 4.5 SepOS density feasibility clamp on the prescribed n_e,sep.
        # DISCONNECTED for now (kept for later use): n_e,sep is taken directly as
        # the prescribed Greenwald fraction without the SepOS ceiling/floor clamp.
        # Re-enable by setting sepos_clamp=True (calls self._sepos_clamp, which
        # together with self._lh_roots implements section 4.5).
        ne_final = ne_sep
        # if self.sepos_clamp:
        #     ne_final = self._sepos_clamp(state, ne_sep, Te, Ti, aLne, aLte, aLti, aLni)

        ni = ne_final / max(state.Zeff, 1e-30)

        self.bc_dict = {
            "ne":   [ne_final * 1e-19, 1.0],
            "te":   [Te * 1e-3, 1.0],
            "ti":   [Ti * 1e-3, 1.0],
            "ni":   [ni * 1e-19, 1.0],
            "aLne": [aLne, 1.0],
            "aLte": [aLte, 1.0],
            "aLti": [aLti, 1.0],
            "aLni": [aLni, 1.0],
        }
        self.bc_leaf_tensors = {
            key: torch.tensor(float(val), dtype=torch.double, requires_grad=True)
            for key, (val, _) in self.bc_dict.items()
        }
        # Prior means for the section-6 soft-prior residuals.
        self.F_y = {"aLne": aLne, "aLte": aLte, "aLti": aLti}

    def _lh_roots(self, state, Te, Ti, lam_pe, alpha_c, Lambda_pi):
        """
        Roots of the L-H criterion G(n_sep)=0 at fixed T_e,sep (section 4.5,
        Eq. 8 = H.10). G is U-shaped so there are generically two roots
        n_LH,low < n_LH,high (H-mode where G>0). The only n-dependence is via
        k_EM^2 ∝ n and α_t ∝ n. Returns (n_low, n_high) or (None, None).
        """
        mu0 = 4.0e-7 * np.pi
        mi_kg = self._mi_ref_u * _u_kg
        tau_i = Ti / max(Te, 1e-30)
        omega_B = 2.0 * lam_pe / max(state.R0, 1e-30)
        # n-linear coefficients: k_EM^2 = c_k*n, α_t = c_a*n.
        c_k = mu0 * e_J * Te * mi_kg / max(state.BT**2 * self._ME_KG, 1e-300)
        c_a = 3.13e-18 * state.R0 * state.q_cyl**2 * state.Zeff / max(Te**2, 1e-30)

        def G(n):
            n = max(float(n), 1e-30)
            k2 = c_k * n
            k = np.sqrt(max(k2, 1e-300))
            at = c_a * n
            lhs = alpha_c * k * tau_i * Lambda_pi / (1.0 + (at / max(alpha_c, 1e-30)) ** 2 * k2)
            rhs = at * (0.5 + k2) + 0.5 * (alpha_c / max(k2, 1e-300)) * np.sqrt(max(omega_B, 0.0)) * tau_i * Lambda_pi
            return lhs - rhs

        # Scan a log grid in units of n_GW for sign changes.
        grid = state.n_GW * np.logspace(-3.0, np.log10(5.0), 96)
        gv = np.array([G(x) for x in grid])
        idx = np.where(np.sign(gv[:-1]) != np.sign(gv[1:]))[0]
        roots = []
        for i in idx:
            try:
                roots.append(float(brentq(G, grid[i], grid[i + 1], xtol=1e-3 * state.n_GW, rtol=1e-8)))
            except Exception:
                continue
        if not roots:
            return None, None
        roots.sort()
        return roots[0], roots[-1]

    def _sepos_clamp(self, state, ne_sep, Te, Ti, aLne, aLte, aLti, aLni) -> float:
        """
        Clamp n_e,sep to the SepOS feasible band (section 4.5):

            n_sep,max = min(n_ball, n_LH,high)            ceiling
            n_sep,min = lower_envelope(n_DL, n_LH,low)    floor

        where n_LH,low/high are the two roots of the L-H criterion G (Eq. 8) and
        n_ball / n_DL are the closed-form ideal-ballooning ceiling and L-mode
        density-limit floor. Warns if the prescribed density is clipped.
        """
        mu0 = 4.0e-7 * np.pi
        kappa_hat = np.sqrt(
            (1.0 + state.kappa**2 * (1.0 + 2.0 * state.delta**2 - 1.2 * state.delta**3)) / 2.0
        )
        alpha_c = kappa_hat**1.2 * (1.0 + 1.5 * state.delta)
        lam_pe = state.a / max(aLne + aLte, 1e-30)
        # Λ_pi = λ_pe/λ_pi = (a/L_ni + a/L_Ti)/(a/L_ne + a/L_Te); paper reduces to 1.
        Lambda_pi = (aLni + aLti) / max(aLne + aLte, 1e-30)
        T_tot = Te + Ti  # eV

        # Closed-form ballooning ceiling and density-limit floor.
        n_ball = (alpha_c * lam_pe * state.BT**2
                  / max(2.0 * mu0 * state.R0 * state.q_cyl**2 * e_J * T_tot, 1e-30))
        n_DL = (state.n_GW * 0.11 * (np.sqrt(alpha_c) / max(kappa_hat**2, 1e-30))
                * np.sqrt(max(Te, 0.0) / max(state.Zeff, 1e-30))
                * lam_pe**0.25 * state.R0**0.25)

        # L-H roots feed both bounds: upper root caps the ceiling, lower root the floor.
        n_lh_low, n_lh_high = self._lh_roots(state, Te, Ti, lam_pe, alpha_c, Lambda_pi)

        n_max = n_ball if n_lh_high is None else min(n_ball, n_lh_high)
        n_min = n_DL if n_lh_low is None else min(n_DL, n_lh_low)

        lo, hi = min(n_min, n_max), max(n_min, n_max)
        ne_clamped = float(np.clip(ne_sep, lo, hi))
        if self.verbose and abs(ne_clamped - ne_sep) / max(ne_sep, 1e-30) > 1e-6:
            print(
                f"[Synthesis] n_e,sep={ne_sep:.3e} clamped to {ne_clamped:.3e} "
                f"(SepOS band [{lo:.3e},{hi:.3e}]; n_ball={n_ball:.2e}, n_DL={n_DL:.2e}, "
                f"n_LH=[{n_lh_low},{n_lh_high}]) — f_GW not consistent with predicted T_e,sep",
                typeMsg="w",
            )
        return ne_clamped


# ---------------------------------------------------------------------------
# Registry and factory
# ---------------------------------------------------------------------------

_Y_MODELS: dict = {
    "Fixed": Y_Fixed,
    "TFTP":  Y_TFTP,
}

_aLy_MODELS: dict = {
    "Fixed":    aLy_Fixed,
    "PeretSSF": aLy_PeretSSF,
    "EichManz": aLy_EichManz,
    "Ensemble": aLy_Ensemble,
}

# Legacy string name → (y_name, aly_name)
_LEGACY_MAP: dict = {
    "Fixed":                     ("Fixed", "Fixed"),
    "FixedInitial":              ("Fixed", "Fixed"),
    "TwoFluid_PeretSSF":         ("TFTP",  "PeretSSF"),
    "TwoFluidTwoPoint_PeretSSF": ("TFTP",  "PeretSSF"),
    "TwoFluid_EichManz":         ("TFTP",  "EichManz"),
    "Tftp_SepOS":                ("TFTP",  "EichManz"),
}


def build_bc_model(bc_model, bc_model_options: dict | None = None) -> CombinedBCModel:
    """
    Instantiate a ``CombinedBCModel`` by name or by specification dict.

    Parameters
    ----------
    bc_model : str or dict
        String: legacy name, e.g. ``"Fixed"``, ``"TwoFluid_PeretSSF"``.
        Dict:   ``{"y": <y_name>, "aLy": <aly_name>}`` with model names from
                ``_Y_MODELS`` and ``_ALY_MODELS``.

    bc_model_options : dict or None
        String bc_model  → flat dict forwarded to all three constructors.
        Dict bc_model    → may contain sub-dicts keyed "y", "aLy", "combined";
                           missing sub-dicts fall back to the flat dict.
    """
    if bc_model_options is None:
        bc_model_options = {}

    # Self-consistent section-4/5 synthesis model (its own coupled solver).
    _SYNTH_NAMES = {"Synthesis", "TwoFluidSynthesis", "TwoFluid_Synthesis", "TFTP_Synthesis"}
    _is_synth = (
        (isinstance(bc_model, str) and bc_model in _SYNTH_NAMES)
        or (isinstance(bc_model, dict) and (
            bc_model.get("aLy") in _SYNTH_NAMES or bc_model.get("model") in _SYNTH_NAMES
        ))
    )
    if _is_synth:
        flat = {k: v for k, v in bc_model_options.items() if k not in ("y", "aLy", "combined")}
        synth_opts = {**flat,
                      **dict(bc_model_options.get("aLy", {}) if isinstance(bc_model_options.get("aLy"), dict) else {}),
                      **dict(bc_model_options.get("combined", {}))}
        return TwoFluidSynthesis(synth_opts)

    if isinstance(bc_model, str):
        if bc_model not in _LEGACY_MAP:
            raise KeyError(
                f"Unknown BC model '{bc_model}'. "
                f"Available strings: {list(_LEGACY_MAP)}  "
                f"or pass a dict with 'y' and 'aLy' keys."
            )
        y_name, aly_name = _LEGACY_MAP[bc_model]
        y_opts        = dict(bc_model_options)
        aly_opts      = dict(bc_model_options)
        combined_opts = dict(bc_model_options)

    elif isinstance(bc_model, dict):
        y_name   = bc_model.get("y",   "Fixed")
        aly_name = bc_model.get("aLy", "Fixed")
        flat     = {k: v for k, v in bc_model_options.items() if k not in ("y", "aLy", "combined")}
        y_opts        = {**flat, **dict(bc_model_options.get("y",   {}))}
        aly_opts      = {**flat, **dict(bc_model_options.get("aLy", {}))}
        combined_opts = {**flat, **dict(bc_model_options.get("combined", {}))}

    else:
        raise TypeError(f"bc_model must be str or dict, got {type(bc_model).__name__}")

    if y_name not in _Y_MODELS:
        raise KeyError(f"Unknown y model '{y_name}'. Available: {list(_Y_MODELS)}")
    if aly_name not in _aLy_MODELS:
        raise KeyError(f"Unknown aLy model '{aly_name}'. Available: {list(_aLy_MODELS)}")

    return CombinedBCModel(
        y_model=_Y_MODELS[y_name](y_opts),
        aly_model=_aLy_MODELS[aly_name](aly_opts),
        options=combined_opts,
    )
