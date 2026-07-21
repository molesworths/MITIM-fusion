"""
neutrals.py
-----------
Main-ion (D⁰) neutral density solver for PORTALS-Edge.

All models share the public interface::

    model = <Model>(options)
    model.solve(powerstate, batch_idx=0)

After ``solve()``, the following keys are written into ``powerstate.plasma``:

    plasma['n0']              : torch.Tensor (batch, rmin)  [1e19 m⁻³]
        Main-ion (D⁰) ground-state neutral density profile (gridded on rmin
        [meters]).  This is the key read by ``analytical_model_edge._evaluate_particle_fluxes``
        and ``_evaluate_ionization_loss``.

    plasma['S_ion_main']      : torch.Tensor (batch, rmin)  [1e19 m⁻³ s⁻¹]
        Volumetric ionisation source:  S = n₀ × ν_iz (gridded on rmin [meters]).
        Directly usable in particle-flux target and ionisation power-loss
        calculations without needing to recompute the rate coefficient.

    plasma['nu_ioniz_main']   : torch.Tensor (batch, rmin)  [s⁻¹]
        Effective D⁰ ionisation frequency:  ν_iz = n_e × ⟨σv⟩_iz(n_e, T_e).
        Evaluated from ADAS H SCD data when Aurora is available; falls back
        to an analytic Lyman-α-weighted fit otherwise.

    plasma['nu_cx_main']      : torch.Tensor (batch, rmin)  [s⁻¹]
        Effective D⁰-D⁺ charge-exchange frequency:  ν_cx = n_i × ⟨σv⟩_cx(T_i).
        Zero when ``include_cx`` is False.  Consumed by the edge target model
        to build the charge-exchange ion energy sink.

    plasma['Ge_core_reinject'] : torch.Tensor (batch, rmin)  [1e20 m⁻² s⁻¹]
        Electron flux from core-bound neutral escape (two-population model):
        hot neutrals that cross the inner boundary ionise inside it and, in
        steady state, re-emerge as an outward electron flux Φ_inner/volp.  Added
        to ``Ge1E20m2`` by the edge target postprocessing.  Zero for NullNeutrals
        and for the legacy single-population solver.

Two-population opacity model (default)
--------------------------------------
Wall-recycled neutrals enter cold (Franck-Condon, ``T_cold_eV`` ~ 3 eV) and
are lost to ionisation *and* charge exchange.  Each CX event converts a cold
neutral into a hot one at the local Ti, so the cold CX loss ``ν_cx·n_cold``
is the distributed birth source of a hot population that penetrates deeper and
is lost only to ionisation.  The cold/hot partition is not prescribed: all
influx is cold at the LCFS and the split emerges from the local ν_cx/ν_iz
competition (reported as ``f_hot`` when ``verbose``).  This fixes the
single-population (v_th = v_th(Ti)) tendency to under-estimate pedestal opacity.
Set ``two_population=False`` for the legacy single-population solver below.

Particle accounting is strict: the amplitude is fixed by the cold inward flux
across the LCFS = ``source_rate``.  In-domain ionisation is then < source_rate;
the balance is hot neutrals escaping across the inner boundary (recovered as an
outward electron flux ``Ge_core_reinject``) and across the LCFS to the SOL (a
genuine loss):  source_rate = ∫ν_iz·n0 dV + Φ_inner + Φ_outer.

Legacy single-population solver
-------------------------------
Solver selection is governed by the Knudsen number estimated over the
outermost ``Kn_eval_fraction`` of the radial domain:

    Kn(r) = λ_mfp(r) / L_ne(r)

    λ_mfp = v_th / ν_iz          (ionisation-limited mean free path)
    L_ne  = |n_e / (dn_e/dr)|    (electron density scale length)
    v_th  = √(2 k_B T_i / m_D)  (neutral thermal speed ≈ ion thermal speed)

  Kn_edge ≤ Kn_thresh  →  **diffusive** solver (collisional sub-regime)

      1-D slab steady-state diffusion with ionisation sink:

          d/dx [ D_n(x) dn₀/dx ] − ν_iz(x) n₀ = 0

      D_n = v_th² / (3 ν_total)   with ν_total = ν_iz + ν_cx

      Solved as a tridiagonal linear system via ``scipy.linalg.solve_banded``.
      Boundary conditions:
          n₀[0]  = 0  (inner, Dirichlet: fully absorbed / ionised)
          n₀[-1] = 1  (outer, Dirichlet, rescaled to source_rate)

  Kn_edge  > Kn_thresh  →  **kinetic** (free-streaming) solver

      Mono-energetic streaming with ionisation attenuation (Beer-Lambert):

          n₀(r) ∝ exp(−τ(r))
          τ(r) = ∫_r^{r_LCFS} ν_iz(r') / v_th(r') dr'

In both cases the amplitude is fixed by steady-state particle balance:

    ∫ n₀(r) × ν_iz(r) dV = source_rate   [s⁻¹]

where  dV = volp(r) dr  in GACODE convention (volp = dV/dr_min [m²]).

Charge-exchange
---------------
CX (D⁰ + D⁺ → D⁺ + D⁰) does not net-remove D⁰ atoms from the neutral
population, so it does not appear as a sink in the density equation.  However
it increases the total D⁰-D⁺ collision frequency, reducing the effective
mean free path.  When ``include_cx`` is True, ν_cx is added to ν_total when
computing D_n, using a simple analytic rate fit for self-CX:

    ⟨σv⟩_cx ≈ 2e-14 × (T_i_eV / 1000)^0.3 cm³/s  (D+D resonant CX, rough fit)

ν_cx is NOT added to the Kn estimate or the density-equation sink.

Integration with STATEedge / powerstate_edge
--------------------------------------------
``calculateNeutrals()`` in ``powerstate_edge`` calls this solver.  It is
invoked between ``calculateChargeStates()`` and ``calculateTargets()``, after
kinetic profiles are fully reconstructed.

Notes
-----
* KN1D (IDL-based) is NOT called.  This is a pure-Python analytic model for
  use inside every PORTALS iteration.
* ADAS H SCD rates are loaded once and cached in ``_ADAS_CACHE`` at module
  level; subsequent calls reuse the same table.
* The 1-D slab approximation is used for the diffusion equation.  The
  cylindrical correction of order (dr/r)² ≈ 1 % in the pedestal region
  is negligible.
"""

import numpy as np
import torch
from scipy.linalg import solve_banded
from scipy.constants import m_p, e as q_e

from mitim_tools.misc_tools.LOGtools import printMsg as print

try:
    import aurora as _aurora_pkg
    _AURORA_AVAILABLE = True
except ImportError:
    _aurora_pkg = None
    _AURORA_AVAILABLE = False


# ---------------------------------------------------------------------------
# ADAS data cache — loaded once on first use, shared across all instances
# ---------------------------------------------------------------------------

_ADAS_CACHE: dict = {}


def _load_h_ioniz_rate():
    """Return ADAS H SCD ionisation rate table, loading once and caching."""
    if "H_scd" not in _ADAS_CACHE:
        if not _AURORA_AVAILABLE:
            _ADAS_CACHE["H_scd"] = None
        else:
            try:
                ad = _aurora_pkg.atomic.get_atom_data("H", ["scd"])
                _ADAS_CACHE["H_scd"] = ad["scd"]
            except Exception as exc:
                print(
                    f"[AnalyticNeutrals] Could not load ADAS H SCD data: {exc}.  "
                    f"Falling back to analytic fit.",
                    typeMsg="w",
                )
                _ADAS_CACHE["H_scd"] = None
    return _ADAS_CACHE["H_scd"]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class NeutralModel:
    """Abstract base — subclasses implement ``solve(powerstate, batch_idx)``."""

    def __init__(self, options: dict):
        self.options = options

    def solve(self, powerstate, batch_idx: int = 0) -> None:
        """
        Populate ``powerstate.plasma`` with main-ion neutral density data.

        Must write:
          ``plasma['n0']``             (batch, rmin)  [1e19 m⁻³]
          ``plasma['S_ion_main']``     (batch, rmin)  [1e19 m⁻³ s⁻¹]
          ``plasma['nu_ioniz_main']``  (batch, rmin)  [s⁻¹]
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# NullNeutrals
# ---------------------------------------------------------------------------

class NullNeutrals(NeutralModel):
    """Zero neutral density — no source.  Useful for testing or for runs
    where the main-ion recycling is handled externally."""

    def solve(self, powerstate, batch_idx: int = 0) -> None:
        p     = powerstate.plasma
        batch = p["te"].shape[0]
        n_rho = p["te"].shape[1]
        _kw   = {"dtype": p["te"].dtype, "device": p["te"].device}

        if "n0" not in p or p["n0"].shape != (batch, n_rho):
            p["n0"]               = torch.zeros(batch, n_rho, **_kw)
            p["S_ion_main"]       = torch.zeros(batch, n_rho, **_kw)
            p["nu_ioniz_main"]    = torch.zeros(batch, n_rho, **_kw)
            p["nu_cx_main"]       = torch.zeros(batch, n_rho, **_kw)
            p["tau_n0"]           = torch.zeros(batch, n_rho, **_kw)
            p["Ge_core_reinject"] = torch.zeros(batch, n_rho, **_kw)


# ---------------------------------------------------------------------------
# AnalyticNeutrals — 1-D diffusive / kinetic steady-state solver
# ---------------------------------------------------------------------------

class AnalyticNeutrals(NeutralModel):
    """
    1-D steady-state D⁰ neutral density solver with automatic regime selection.

    Parameters
    ----------
    options : dict
        ``source_rate`` : float, default 1e21
            Number of D⁰ neutrals crossing the LCFS inward per second [s⁻¹].
            Sets the amplitude of n₀ via particle balance.
        ``mu_amu`` : float, default 2.014
            Atomic mass of the neutral species [amu].  2.014 for deuterium.
        ``Kn_thresh`` : float, default 0.3
            Knudsen number threshold for solver selection.  Above this value
            the kinetic (free-streaming) solver is used; at or below, the
            diffusive (collisional) solver is used.
        ``Kn_eval_fraction`` : float, default 0.3
            Fraction of the outer radial domain used to compute the
            representative Kn for solver selection.
        ``include_cx`` : bool, default True
            When True, an analytic charge-exchange rate is added to the
            total collision frequency when computing D_n.  CX is NOT added
            to the Knudsen number estimate or the ionisation sink.
        ``two_population`` : bool, default True
            Use the coupled cold (Franck-Condon) + hot (CX-generated) neutral
            model for opacity/penetration.  When False, the legacy single-
            population Knudsen-selected diffusive/kinetic solver is used
            (neutral speed = v_th(Ti), which tends to under-estimate opacity).
        ``T_cold_eV`` : float, default 3.0
            Franck-Condon temperature [eV] of the wall-recycled cold neutral
            population.  Sets the near-separatrix penetration depth.  The hot
            population temperature is the local Ti; the cold→hot partition is
            not prescribed — it emerges from the local ν_cx / ν_iz competition.
        ``verbose`` : bool, default False
    """

    def __init__(self, options: dict):
        super().__init__(options)
        self.source_rate      = options.get("source_rate",      1e21)
        self.mu_amu           = options.get("mu_amu",           2.014)
        self.Kn_thresh        = options.get("Kn_thresh",        0.3)
        self.Kn_eval_fraction = options.get("Kn_eval_fraction", 0.3)
        self.include_cx       = options.get("include_cx",       True)
        # Two-population (cold Franck-Condon + CX-generated hot) opacity model.
        self.two_population   = options.get("two_population",   True)
        self.T_cold_eV        = options.get("T_cold_eV",        3.0)
        self.verbose          = options.get("verbose",          False)

    # ------------------------------------------------------------------
    # Ionisation frequency
    # ------------------------------------------------------------------

    def _nu_ioniz(
        self, ne_cm3: np.ndarray, Te_eV: np.ndarray
    ) -> np.ndarray:
        """
        Return  ν_iz(r) [s⁻¹] = n_e × ⟨σv⟩_iz(n_e, T_e).

        Uses the ADAS H SCD effective ionisation rate coefficient when Aurora
        is available; otherwise falls back to the Voronov (1997) analytic fit
        for H ground-state ionisation (ADNDT 65, 1997, Table 1):

            U = 13.6 eV / T_e
            ⟨σv⟩_iz = 2.91×10⁻¹⁴ × U^0.39 × exp(−U) / (0.232 + U)  m³/s

        The Voronov fit is accurate to better than 10% for 5 ≤ T_e ≤ 5000 eV.
        Use ADAS for quantitative work.
        """
        scd = _load_h_ioniz_rate()
        if scd is not None:
            # interp_atom_prof: xprof = log10(ne [cm⁻³]), yprof = log10(Te [eV])
            # x_multiply=True → returns ne × S_cd [cm³/s × cm⁻³ = s⁻¹]
            # Input shape: (nt=1, nr)
            log_ne = np.log10(np.maximum(ne_cm3, 1e8))[np.newaxis, :]
            log_Te = np.log10(np.maximum(Te_eV,  0.1 ))[np.newaxis, :]
            # Returns (nt=1, nion=1, nr) for H (single ionisation stage)
            nu_grid = _aurora_pkg.atomic.interp_atom_prof(
                scd, log_ne, log_Te, x_multiply=True
            )
            return nu_grid[0, 0, :].copy()   # (nr,)  s⁻¹
        else:
            # Voronov (1997) fit: U = chi/T_e, chi = 13.6 eV (H ionisation energy)
            U = 13.6 / np.maximum(Te_eV, 0.1)
            sigma_v_m3 = (
                2.91e-14          # A = 2.91e-8 cm³/s converted to m³/s
                * U ** 0.39
                * np.exp(-U)
                / (0.232 + U)
            )  # m³/s  (P=0, so (1 + P√U) denominator is unity)
            return ne_cm3 * 1e6 * sigma_v_m3   # s⁻¹ (ne_cm3→m⁻³, σv in m³/s)

    # ------------------------------------------------------------------
    # Charge-exchange collision frequency (approximate, for D_n only)
    # ------------------------------------------------------------------

    def _nu_cx(
        self, ni_cm3: np.ndarray, Ti_eV: np.ndarray
    ) -> np.ndarray:
        """
        Analytic estimate of ν_cx = nᵢ × ⟨σv⟩_cx for D⁰ + D⁺ → D⁺ + D⁰.

        Fit to Freeman & Jones (1974) D resonant-CX cross-sections:
            ⟨σv⟩_cx ≈ 2e-14 × (T_i/1000)^0.3  m³/s   (T_i in eV)
        (≈ 2e-8 cm³/s at 1 keV — the correct order for resonant D-D CX).

        Returns ν_cx [s⁻¹].
        """
        sigma_v_cx = 2e-14 * (np.maximum(Ti_eV, 1.0) / 1000.0) ** 0.3  # m³/s
        # ni_cm3 → m⁻³ (×1e6) so that ν_cx = n_i[m⁻³] × ⟨σv⟩[m³/s] is in s⁻¹,
        # matching the ionisation-frequency convention in _nu_ioniz.
        return ni_cm3 * 1e6 * sigma_v_cx   # s⁻¹

    # ------------------------------------------------------------------
    # Thermal speed
    # ------------------------------------------------------------------

    def _v_th(self, Ti_eV: np.ndarray) -> np.ndarray:
        """Return v_th = √(2 k_B T_i / m_D) [m/s]."""
        return np.sqrt(
            2.0 * np.maximum(Ti_eV, 0.1) * q_e / (self.mu_amu * m_p)
        )

    # ------------------------------------------------------------------
    # Knudsen number
    # ------------------------------------------------------------------

    def _knudsen(
        self,
        r_m:    np.ndarray,
        ne_cm3: np.ndarray,
        Ti_eV:  np.ndarray,
        nu_iz:  np.ndarray,
    ) -> np.ndarray:
        """
        Kn(r) = λ_mfp(r) / L_ne(r).

        λ_mfp = v_th / ν_iz  (ionisation dominates for deeply penetrating
                               neutrals in the closed-flux region)
        L_ne  = |n_e / (∂n_e/∂r)|  (density gradient scale length)
        """
        v_th       = self._v_th(Ti_eV)
        lambda_mfp = v_th / np.maximum(nu_iz, 1.0)     # m

        dne_dr = np.gradient(ne_cm3, r_m)               # cm⁻³ m⁻¹
        L_ne   = np.where(
            np.abs(dne_dr) > 0,
            np.abs(ne_cm3 / np.maximum(np.abs(dne_dr), 1e-30)),
            1.0,
        )   # m  (cm⁻³ / (cm⁻³ m⁻¹) = m, consistent with λ_mfp)
        L_ne   = np.clip(L_ne, 1e-3, 1e3)

        return lambda_mfp / L_ne    # dimensionless

    # ------------------------------------------------------------------
    # Diffusive solver
    # ------------------------------------------------------------------

    def _solve_diffusive(
        self,
        r_m:    np.ndarray,
        nu_iz:  np.ndarray,
        nu_cx:  np.ndarray,
        Ti_eV:  np.ndarray,
        volp:   np.ndarray,
    ) -> np.ndarray:
        """
        Solve  d/dx [ D_n(x) dn₀/dx ] − ν_iz(x) n₀ = 0  (1-D slab).

        D_n = v_th² / (3 ν_total)  where  ν_total = ν_iz + ν_cx.

        Boundary conditions:
          n₀[0]  = 0   (inner Dirichlet — absorbing)
          n₀[-1] = 1   (outer Dirichlet — rescaled below)

        The shape solution is rescaled so that
          ∫ n₀ × ν_iz × dV = source_rate  [s⁻¹].

        Returns n₀ [m⁻³].
        """
        nr    = len(r_m)
        v_th  = self._v_th(Ti_eV)                             # (nr,) m/s
        nu_tot = nu_iz + nu_cx                                 # (nr,) s⁻¹
        D_n    = v_th**2 / (3.0 * np.maximum(nu_tot, 1e-10))  # (nr,) m²/s

        lo  = np.zeros(nr)
        di  = np.zeros(nr)
        hi  = np.zeros(nr)
        rhs = np.zeros(nr)

        # Inner Dirichlet: n₀[0] = 0
        di[0] = 1.0

        # Interior nodes
        for i in range(1, nr - 1):
            dr_up = r_m[i + 1] - r_m[i]
            dr_dn = r_m[i]     - r_m[i - 1]
            dr_c  = 0.5 * (dr_up + dr_dn)
            D_up  = 0.5 * (D_n[i] + D_n[i + 1])
            D_dn  = 0.5 * (D_n[i] + D_n[i - 1])
            hi[i] =  D_up / (dr_up * dr_c)
            lo[i] =  D_dn / (dr_dn * dr_c)
            di[i] = -(hi[i] + lo[i] + nu_iz[i])

        # Outer Dirichlet: n₀[-1] = 1 (rescaled after solve)
        di[-1]  = 1.0
        rhs[-1] = 1.0

        # scipy solve_banded format for (1, 1) banded matrix:
        #   ab[0, j] = superdiag at column j   (ab[0, 0] unused)
        #   ab[1, j] = diagonal  at column j
        #   ab[2, j] = subdiag   at column j   (ab[2, -1] unused)
        ab       = np.zeros((3, nr))
        ab[0, 1:] = hi[:-1]   # hi[i] connects row i → column i+1
        ab[1, :]  = di
        ab[2, :-1]= lo[1:]    # lo[i] connects row i → column i-1

        try:
            n0_shape = solve_banded((1, 1), ab, rhs)
        except Exception as exc:
            print(
                f"[AnalyticNeutrals] Tridiagonal diffusive solve failed: {exc}.  "
                f"Inserting zeros.",
                typeMsg="w",
            )
            return np.zeros(nr)

        n0_shape = np.maximum(n0_shape, 0.0)
        return self._rescale_to_source(n0_shape, nu_iz, volp, r_m)

    # ------------------------------------------------------------------
    # General 1-D diffusion solve with loss + distributed source
    # ------------------------------------------------------------------

    def _solve_diffusion_1d(
        self,
        r_m:      np.ndarray,
        D_n:      np.ndarray,
        nu_loss:  np.ndarray,
        source:   np.ndarray,
        inner_val: float = 0.0,
        outer_val: float = 1.0,
    ) -> np.ndarray:
        """
        Solve  d/dx [ D_n(x) dn/dx ] − ν_loss(x) n + source(x) = 0  (1-D slab)
        with Dirichlet boundaries  n[0] = inner_val,  n[-1] = outer_val.

        ``source`` [same units as ν_loss·n] is a distributed volumetric source
        (used to feed the CX-generated hot population from cold CX losses).
        Returns the (non-negative) profile; amplitude is set by the caller.
        """
        nr  = len(r_m)
        lo  = np.zeros(nr)
        di  = np.zeros(nr)
        hi  = np.zeros(nr)
        rhs = np.zeros(nr)

        di[0]  = 1.0
        rhs[0] = inner_val

        for i in range(1, nr - 1):
            dr_up = r_m[i + 1] - r_m[i]
            dr_dn = r_m[i]     - r_m[i - 1]
            dr_c  = 0.5 * (dr_up + dr_dn)
            D_up  = 0.5 * (D_n[i] + D_n[i + 1])
            D_dn  = 0.5 * (D_n[i] + D_n[i - 1])
            hi[i] =  D_up / (dr_up * dr_c)
            lo[i] =  D_dn / (dr_dn * dr_c)
            di[i] = -(hi[i] + lo[i] + nu_loss[i])
            rhs[i] = -source[i]

        di[-1]  = 1.0
        rhs[-1] = outer_val

        ab        = np.zeros((3, nr))
        ab[0, 1:] = hi[:-1]
        ab[1, :]  = di
        ab[2, :-1]= lo[1:]

        try:
            n_shape = solve_banded((1, 1), ab, rhs)
        except Exception as exc:
            print(
                f"[AnalyticNeutrals] 1-D diffusion solve failed: {exc}.  "
                f"Inserting zeros.",
                typeMsg="w",
            )
            return np.zeros(nr)

        return np.maximum(n_shape, 0.0)

    # ------------------------------------------------------------------
    # Kinetic hot-neutral density from a distributed birth source
    # ------------------------------------------------------------------

    def _solve_hot_kinetic(
        self,
        r_m:    np.ndarray,
        q_hot:  np.ndarray,
        nu_iz:  np.ndarray,
        v_hot:  np.ndarray,
        volp:   np.ndarray,
    ) -> tuple:
        """
        Free-streaming hot-neutral density from a distributed volumetric birth
        source ``q_hot`` [m⁻³ s⁻¹], attenuated by ionisation.

        Hot neutrals are born (by CX of cold neutrals) throughout the domain and
        stream in both radial directions, attenuating as ``exp(−|τ_h(r)−τ_h(r')|)``
        with the ionisation optical depth ``τ_h(r) = ∫_r^{LCFS} ν_iz/v_hot dr'``.
        For an isotropic 1-D-slab split (half inward, half outward):

            n_hot(r) = 1/(2 v_hot(r)) ∫ q_hot(r') exp(−|τ_h(r)−τ_h(r')|) dr'

        Neutrals reaching either boundary escape (no reflection).  A hot particle
        born at r' survives to the inner boundary (τ = τ_max) with probability
        exp(−(τ_max − τ(r'))) and to the LCFS (τ = 0) with exp(−τ(r')).  Half go
        each way, so the escaping *flows* [s⁻¹] are

            Φ_inner = ½ ∫ q_hot(r') exp(−(τ_max − τ(r'))) volp dr'   (→ core)
            Φ_outer = ½ ∫ q_hot(r') exp(−τ(r'))          volp dr'   (→ SOL, lost)

        Returns (n_hot, phi_inner, phi_outer), all in the amplitude units of q_hot.
        """
        nr = len(r_m)

        # Ionisation optical depth for the hot population, τ=0 at the LCFS.
        nu_over_v = nu_iz / np.maximum(v_hot, 1.0)
        tau_h = np.zeros(nr)
        for i in range(nr - 2, -1, -1):
            dr        = abs(r_m[i + 1] - r_m[i])
            tau_h[i]  = tau_h[i + 1] + 0.5 * (nu_over_v[i] + nu_over_v[i + 1]) * dr

        # Trapezoidal weights for the r' integral.
        w = np.zeros(nr)
        w[1:-1] = 0.5 * (r_m[2:] - r_m[:-2])
        w[0]    = 0.5 * (r_m[1] - r_m[0])
        w[-1]   = 0.5 * (r_m[-1] - r_m[-2])
        w = np.abs(w)

        # n_hot(r) = 1/(2 v_hot(r)) Σ_r' q_hot(r') exp(-|τ(r)-τ(r')|) w(r')
        kernel = np.exp(-np.abs(tau_h[:, None] - tau_h[None, :]))   # (nr, nr)
        n_hot  = (kernel * (q_hot * w)[None, :]).sum(axis=1) / (2.0 * np.maximum(v_hot, 1.0))

        # Boundary escape flows [s⁻¹]: inner (→ core, recovered) and outer (→ SOL, lost).
        tau_max   = tau_h.max()
        birth     = q_hot * volp * w                                # per-cell birth rate [s⁻¹]
        phi_inner = 0.5 * float(np.sum(birth * np.exp(-(tau_max - tau_h))))
        phi_outer = 0.5 * float(np.sum(birth * np.exp(-tau_h)))

        return np.maximum(n_hot, 0.0), phi_inner, phi_outer

    # ------------------------------------------------------------------
    # Two-population (cold Franck-Condon + CX-generated hot) solver
    # ------------------------------------------------------------------

    def _solve_two_population(
        self,
        r_m:    np.ndarray,
        nu_iz:  np.ndarray,
        nu_cx:  np.ndarray,
        Ti_eV:  np.ndarray,
        volp:   np.ndarray,
    ) -> tuple:
        """
        Coupled cold + hot neutral model.

        Cold (Franck-Condon, T = ``T_cold_eV``) is injected at the LCFS and
        lost to ionisation *and* charge exchange (sink ν_iz + ν_cx).  Each CX
        event moves a neutral from the cold to the hot population, so the cold
        CX loss ``ν_cx · n_cold`` is the distributed birth source of the hot
        population.  Hot (T = local Ti) is lost only to ionisation (CX of a hot
        neutral with a hot bulk ion re-randomises velocity but keeps it hot).

        The velocity that sets penetration differs between populations, but the
        *scattering* frequency controlling the diffusion coefficient is the
        total ν_iz + ν_cx for both.  Populations are solved at unit cold boundary
        amplitude, then a single amplitude ``A`` is fixed by strict particle
        accounting: the cold inward flux across the LCFS equals ``source_rate``
        (the system is linear).  In-domain ionisation is then < source_rate; the
        balance is hot neutrals escaping across the boundaries:

            source_rate = ∫ ν_iz n0 dV  +  Φ_inner  +  Φ_outer

        Φ_inner (hot neutrals crossing the inner boundary toward the core) is
        recovered as electron flux via ``Ge_core_reinject`` (see ``solve``);
        Φ_outer (hot leaking back to the SOL) is a genuine loss.

        Returns
        -------
        (n0_m3, tau_cold, f_hot, reinject_flux, in_domain_fraction)
          n0_m3         : total neutral density n_cold + n_hot  [m⁻³]
          tau_cold      : cold-population optical depth (opacity diagnostic)
          f_hot         : fraction of in-domain ionisation from hot neutrals
          reinject_flux : core-bound electron flux Φ_inner / volp  [m⁻² s⁻¹]
          in_domain_fraction : ∫ν_iz n0 dV / source_rate
        """
        nu_scat = np.maximum(nu_iz + nu_cx, 1e-10)

        T_cold  = np.full_like(Ti_eV, self.T_cold_eV)
        v_cold  = self._v_th(T_cold)                 # m/s
        v_hot   = self._v_th(Ti_eV)                  # m/s

        D_cold  = v_cold**2 / (3.0 * nu_scat)        # m²/s

        # Cold population: collisional (short mfp) → diffusive.  Boundary-driven,
        # lost to ionisation + CX.  Unit boundary amplitude (rescaled below).
        n_cold = self._solve_diffusion_1d(
            r_m, D_cold, nu_iz + nu_cx, np.zeros_like(nu_iz),
            inner_val=0.0, outer_val=1.0,
        )

        # Hot population: born by cold CX loss (q_hot = ν_cx·n_cold), lost only
        # to ionisation.  Its mean free path v_hot/ν_iz is typically >> the
        # pedestal width, so it is nearly free-streaming — a diffusive treatment
        # is invalid (it would drain to the boundaries).  Solve it kinetically.
        q_hot = nu_cx * n_cold
        n_hot, phi_inner_1, phi_outer_1 = self._solve_hot_kinetic(
            r_m, q_hot, nu_iz, v_hot, volp
        )

        n_tot_shape = n_cold + n_hot

        # Hot fraction of the in-domain ionisation source (scale-invariant).
        iz_hot = float(np.trapz(nu_iz * n_hot        * volp, r_m))
        iz_tot = float(np.trapz(nu_iz * n_tot_shape  * volp, r_m))
        f_hot  = iz_hot / iz_tot if iz_tot > 0.0 else 0.0

        # Cold optical depth (opacity diagnostic), integrated inward from LCFS.
        nu_over_v = (nu_iz + nu_cx) / np.maximum(v_cold, 1.0)
        tau_cold  = np.zeros(len(r_m))
        for i in range(len(r_m) - 2, -1, -1):
            dr          = abs(r_m[i + 1] - r_m[i])
            tau_cold[i] = tau_cold[i + 1] + 0.5 * (nu_over_v[i] + nu_over_v[i + 1]) * dr

        # Amplitude A: cold inward flux across the LCFS = source_rate (Fick's law).
        dr_edge   = r_m[-1] - r_m[-2]
        dncold_dr = (n_cold[-1] - n_cold[-2]) / dr_edge          # >0 (n falls inward)
        phi_in_1  = D_cold[-1] * dncold_dr * volp[-1]            # s⁻¹ per unit amplitude
        if phi_in_1 > 0.0:
            A = self.source_rate / phi_in_1
        else:
            # Degenerate cold gradient (e.g. ν≈0); fall back to source-matched amplitude.
            if self.verbose:
                print(
                    "[AnalyticNeutrals] LCFS cold influx <= 0; using in-domain "
                    "source normalisation for amplitude.",
                    typeMsg="w",
                )
            n0_tmp = self._rescale_to_source(n_tot_shape, nu_iz, volp, r_m)
            A = (n0_tmp[-1] / n_tot_shape[-1]) if n_tot_shape[-1] > 0 else 0.0

        n0_m3 = A * n_tot_shape

        # Core-bound escape recovered as electron flux (rarefied as Φ_inner/volp).
        phi_inner = A * phi_inner_1                              # s⁻¹
        reinject_flux = phi_inner / np.maximum(volp, 1e-30)     # m⁻² s⁻¹

        iz_in_domain = float(np.trapz(nu_iz * n0_m3 * volp, r_m))
        in_domain_fraction = iz_in_domain / max(self.source_rate, 1e-300)

        return n0_m3, tau_cold, f_hot, reinject_flux, in_domain_fraction

    # ------------------------------------------------------------------
    # Kinetic (free-streaming) solver
    # ------------------------------------------------------------------

    def _solve_kinetic(
        self,
        r_m:   np.ndarray,
        nu_iz: np.ndarray,
        Ti_eV: np.ndarray,
        volp:  np.ndarray,
    ) -> np.ndarray:
        """
        Free-streaming model with ionisation Beer-Lambert attenuation:

            n₀(r) ∝ exp(−τ(r))
            τ(r) = ∫_{r}^{r_LCFS} ν_iz(r') / v_th(r') dr'

        τ is integrated inward from the LCFS by trapezoidal rule.
        Amplitude is set by particle balance (see ``_rescale_to_source``).

        Returns n₀ [m⁻³].
        """
        nr         = len(r_m)
        v_th       = self._v_th(Ti_eV)                 # (nr,) m/s
        nu_over_v  = nu_iz / np.maximum(v_th, 1.0)     # (nr,) m⁻¹

        # Cumulative optical depth from outer boundary inward
        tau = np.zeros(nr)
        for i in range(nr - 2, -1, -1):
            dr     = abs(r_m[i + 1] - r_m[i])   # abs guards against non-monotone r_m
            tau[i] = tau[i + 1] + 0.5 * (nu_over_v[i] + nu_over_v[i + 1]) * dr

        n0_shape = np.exp(-tau)   # dimensionless; n0_shape[-1] = 1.0 by construction
        return self._rescale_to_source(n0_shape, nu_iz, volp, r_m), tau

    # ------------------------------------------------------------------
    # Shared particle-balance rescaling
    # ------------------------------------------------------------------

    def _rescale_to_source(
        self,
        n0_shape: np.ndarray,
        nu_iz:    np.ndarray,
        volp:     np.ndarray,
        r_m:      np.ndarray,
    ) -> np.ndarray:
        """
        Scale a dimensionless shape profile so that
          ∫ n₀(r) × ν_iz(r) dV = source_rate  [s⁻¹].

        Unit analysis:
          A [m⁻³] × ∫ n₀_shape [-] × ν_iz [s⁻¹] × volp [m²] dr [m] = source_rate [s⁻¹]
          → integral [m³/s], A = source_rate [s⁻¹] / integral [m³/s] = m⁻³  ✓

        Returns A × n₀_shape  in [m⁻³].
        """
        integral = np.trapz(n0_shape * nu_iz * volp, r_m)   # m³/s
        if integral <= 0.0:
            if self.verbose:
                print(
                    "[AnalyticNeutrals] Particle balance integral = 0 — "
                    "inserting zeros.",
                    typeMsg="w",
                )
            return np.zeros_like(n0_shape)

        A = self.source_rate / integral   # m⁻³
        return n0_shape * A               # m⁻³

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def solve(self, powerstate, batch_idx: int = 0) -> None:
        """
        Compute the D⁰ neutral density and ionisation source for batch element
        *batch_idx* and write results into ``powerstate.plasma``.
        """
        b     = batch_idx
        p     = powerstate.plasma
        batch = p["te"].shape[0]
        n_rho = p["te"].shape[1]
        dfT   = p["te"]
        _kw   = {"dtype": dfT.dtype, "device": dfT.device}

        # --- Extract 1-D profiles ------------------------------------------------
        rmin_m  = p["rmin"][b, :].cpu().numpy()              # m
        ne_raw  = p["ne"][b, :].cpu().numpy()                # expected: 1e19 m⁻³
        # Guard against silently wrong units: typical pedestal ne is 0.1–50 × 1e19 m⁻³
        if ne_raw.max() < 1e-3 or ne_raw.max() > 1e4:
            print(
                f"[AnalyticNeutrals] batch {b}: p['ne'] max = {ne_raw.max():.3e} — "
                f"expected units are 1e19 m\u207b\u00b3 (typical range 0.1\u201350).  "
                f"Check unit convention before proceeding.",
                typeMsg="w",
            )
        ne_cm3  = ne_raw * 1e13                              # 1e19 m⁻³ → cm⁻³
        Te_eV   = p["te"][b,  :].cpu().numpy() * 1e3         # keV → eV
        Ti_eV   = (
            p["ti"][b, :].cpu().numpy() * 1e3
            if p["ti"].dim() >= 2
            else Te_eV.copy()
        )   # keV → eV
        volp    = p["volp"][b, :].cpu().numpy()               # m² (dV/dr_min)

        ne_cm3 = np.maximum(ne_cm3, 1e8)
        Te_eV  = np.maximum(Te_eV,  0.1)
        Ti_eV  = np.maximum(Ti_eV,  0.1)

        # Main ion density for CX; fall back to ne if ni not available
        if "ni" in p and p["ni"].dim() == 3:
            ni_cm3 = p["ni"][b, :, 0].cpu().numpy() * 1e13
        else:
            ni_cm3 = ne_cm3.copy()
        ni_cm3 = np.maximum(ni_cm3, 1e8)

        # --- Rate coefficients ---------------------------------------------------
        nu_iz = self._nu_ioniz(ne_cm3, Te_eV)      # (nr,) s⁻¹
        # ν_cx is required by the two-population model (it generates the hot
        # population), so compute it whenever CX or two_population is active.
        nu_cx = (
            self._nu_cx(ni_cm3, Ti_eV)
            if (self.include_cx or self.two_population)
            else np.zeros_like(nu_iz)
        )

        # Core-bound re-injection flux [m⁻² s⁻¹]; nonzero only for two_population.
        reinject_flux = np.zeros_like(nu_iz)

        # --- Solve ---------------------------------------------------------------
        if self.two_population:
            n0_m3, tau_n0, f_hot, reinject_flux, in_dom = self._solve_two_population(
                rmin_m, nu_iz, nu_cx, Ti_eV, volp
            )
            if self.verbose:
                print(
                    f"[AnalyticNeutrals] batch {b}: two-population solver "
                    f"(T_cold = {self.T_cold_eV:.1f} eV) "
                    f"→ f_hot = {f_hot:.3f}, tau_cold_max = {tau_n0.max():.3f}, "
                    f"in-domain ionised fraction = {in_dom:.3f} "
                    f"(core re-inject + SOL loss = {1.0 - in_dom:.3f}).",
                    typeMsg="i",
                )
        else:
            # --- Legacy single-population Knudsen-selected solver ----------------
            Kn      = self._knudsen(rmin_m, ne_cm3, Ti_eV, nu_iz)
            n_outer = max(1, int(self.Kn_eval_fraction * n_rho))
            Kn_edge = float(np.median(Kn[-n_outer:]))

            use_kinetic = Kn_edge > self.Kn_thresh
            if self.verbose:
                solver_tag = "kinetic (free-streaming)" if use_kinetic else "diffusive (collisional)"
                print(
                    f"[AnalyticNeutrals] batch {b}: "
                    f"Kn_edge = {Kn_edge:.3f} (threshold = {self.Kn_thresh}) "
                    f"→ {solver_tag} solver.",
                    typeMsg="i",
                )

            if use_kinetic:
                # _solve_kinetic returns (n0, tau) — reuse tau to avoid duplication
                n0_m3, tau_n0 = self._solve_kinetic(rmin_m, nu_iz, Ti_eV, volp)
            else:
                n0_m3 = self._solve_diffusive(rmin_m, nu_iz, nu_cx, Ti_eV, volp)
                # Compute optical depth separately for the diffusive case
                v_th_arr  = self._v_th(Ti_eV)                            # (nr,) m/s
                nu_over_v = nu_iz / np.maximum(v_th_arr, 1.0)           # (nr,) m⁻¹
                tau_n0    = np.zeros(n_rho)
                for i in range(n_rho - 2, -1, -1):
                    dr        = abs(rmin_m[i + 1] - rmin_m[i])
                    tau_n0[i] = tau_n0[i + 1] + 0.5 * (nu_over_v[i] + nu_over_v[i + 1]) * dr

        n0_m3    = np.maximum(n0_m3, 0.0)

        S_ion_m3 = n0_m3 * nu_iz     # m⁻³ s⁻¹

        # --- Convert to powerstate units: 1e19 m⁻³ (as ne, ni) ------------------
        n0_1e19    = n0_m3  * 1e-19
        S_ion_1e19 = S_ion_m3 * 1e-19
        # Core-bound re-injection electron flux → 1E20 m⁻² s⁻¹ (as Ge1E20m2).
        reinject_1e20 = reinject_flux * 1e-20

        # n0/ne trace-neutral validity guard
        n0_over_ne = n0_m3 / np.maximum(ne_cm3 * 1e6, 1.0)  # both in m⁻³
        if n0_over_ne.max() > 0.1:
            print(
                f"[AnalyticNeutrals] batch {b}: max(n0/ne) = {n0_over_ne.max():.2f} "
                f"exceeds 0.1 — trace-neutral approximation may be breaking down.",
                typeMsg="w",
            )

        # Sanity check: integrated source should match source_rate within
        # floating-point precision.
        if self.verbose:
            check = float(np.trapz(S_ion_m3 * volp, rmin_m))
            print(
                f"[AnalyticNeutrals] batch {b}: "
                f"∫Sion dV = {check:.3e} s⁻¹  (target = {self.source_rate:.3e} s⁻¹).",
                typeMsg="i",
            )

        # --- Write to plasma -----------------------------------------------------
        if "n0" not in p or p["n0"].shape != (batch, n_rho):
            p["n0"]               = torch.zeros(batch, n_rho, **_kw)
            p["S_ion_main"]       = torch.zeros(batch, n_rho, **_kw)
            p["nu_ioniz_main"]    = torch.zeros(batch, n_rho, **_kw)
            p["nu_cx_main"]       = torch.zeros(batch, n_rho, **_kw)
            p["tau_n0"]           = torch.zeros(batch, n_rho, **_kw)
            p["Ge_core_reinject"] = torch.zeros(batch, n_rho, **_kw)

        p["n0"][b]               = torch.from_numpy(n0_1e19).to(dfT)
        p["S_ion_main"][b]       = torch.from_numpy(S_ion_1e19).to(dfT)
        p["nu_ioniz_main"][b]    = torch.from_numpy(nu_iz).to(dfT)
        p["nu_cx_main"][b]       = torch.from_numpy(nu_cx).to(dfT)
        p["tau_n0"][b]           = torch.from_numpy(tau_n0.astype(np.float64)).to(dfT)
        p["Ge_core_reinject"][b] = torch.from_numpy(reinject_1e20).to(dfT)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

NEUTRAL_MODELS: dict = {
    "Null":             NullNeutrals,
    "none":             NullNeutrals,
    "Analytic":         AnalyticNeutrals,
    "AnalyticNeutrals": AnalyticNeutrals,
}


def build_neutral_model(name: str, options: dict) -> NeutralModel:
    """
    Instantiate a neutral model by name.

    Parameters
    ----------
    name : str
        One of the keys in ``NEUTRAL_MODELS``.
    options : dict
        Model-specific options forwarded to the class constructor.

    Raises
    ------
    KeyError
        If ``name`` is not registered.
    """
    if name not in NEUTRAL_MODELS:
        raise KeyError(
            f"Unknown neutral model '{name}'.  "
            f"Available: {list(NEUTRAL_MODELS.keys())}"
        )
    return NEUTRAL_MODELS[name](options)
