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
from scipy.optimize import fsolve

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

        gamma_0 = 2.5*ti/te - 0.5*np.log(2.0*np.pi*state.me_over_mi*(1.0 + ti/te))
        gamma   = 2.0*gamma_0/3.0
        g = G0 * rho_s / R0
        f_Delta = self.f_Delta

        def residual(lp_arr):
            lp = float(lp_arr[0])
            sqrt_gamma = np.sqrt(gamma)
            lT        = sqrt_gamma / (sqrt_gamma - 1.0) * lp
            beta      = f_Delta * (1.0/Lambda + lp/lT)
            alpha_ExB = -0.43 * beta * Lambda * (rho_s/lp)**1.5 / g**0.5
            lhs = lp / rho_s
            rhs = (3.9 * g**(3.0/11.0) * (2.0*rho_s/Lpar)**(-6.0/11.0)
                   * gamma**(-4.0/11.0) / (1.0 + (alpha_s + alpha_ExB)**2)**(9.0/11.0))
            return [lhs - rhs]

        res = fsolve(residual, [10.0*rho_s], full_output=True, xtol=1e-6)
        if res[2] != 1:
            raise RuntimeError("PeretSSF: fsolve did not converge")

        lambda_p   = float(res[0][0])
        sqrt_gamma = np.sqrt(gamma)
        lambda_n   = sqrt_gamma * lambda_p
        lambda_T   = sqrt_gamma / (sqrt_gamma - 1.0) * lambda_p
        lambda_q   = (2.0/7.0) * lambda_T

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
