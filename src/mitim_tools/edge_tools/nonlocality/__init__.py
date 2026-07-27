"""nonlocality -- nonlocal corrections to local quasilinear edge transport.

Post-processing module implementing the two mechanisms of the NONLOCAL-OFFLINE
design (adapted for in-loop use), applied in this order:

  1. **Nonlocal ExB quench** (exb.py): shear-free TGLF + external per-ky Waltz
     quench with the rms-smeared diamagnetic gamma_E (correlation.py). Order
     matters: shear suppresses the source, spreading redistributes what
     survives.
  2. **Turbulence spreading, tier A** (correlation.spread_matrix): linear
     radial redistribution of the *physical* turbulent fluxes across the rhoCP
     anchors, absorbing at both domain edges. In-loop this is honestly a
     small coupling matrix between anchors (sub-anchor structure is an
     interpolation model, declared).

Wiring contract (who sees what):
  * plasma["<var>_tr_turb"]      : ExB-QUENCHED local fluxes. These feed the
      OFs and hence the flux GPs; the quench is learnable per-surface via the
      deterministic feature plasma["gamma_exb_nl"] (a/c_s units).
  * plasma["<var>_tr_nl"]        : quenched + SPREAD totals (turb spread +
      neoc). calculateMetrics points at these -- the solver only ever sees
      nonlocal-adjusted residuals.
  * The GP path applies the SAME frozen spread matrix in
      portals_edge.scalarized_objective (spreading can never be a GP feature:
      it depends on the flux field the GP predicts).

The spread matrix is FROZEN at first evaluation (lambda_c from the then-current
rho_s): exact real/GP-path consistency beats tracking the ~10-20% rho_s drift,
which is subsumed by the lambda_c x{0.5,1,2} sensitivity anyway.

Options (edge_options):
    nonlocal_model         : "Analytic" | "Null"/None (skip)
    nonlocal_model_options : {
        "ExB": True, "Spreading": True,
        "lambda_c_mult": 8.0,       # C in lambda_c = C rho_s  (5-12 literature band)
        "kernel": "gauss",          # "lorentzian" = MinT-draft variant
        "alpha_e": 1.0, "p": 2.0,   # quench-rule constants (ledger)
        "spread_lambda_mult": None, # spreading kernel C (default: lambda_c_mult)
        "exb_envelope": None,       # None=rms | "gauss"/"lorentzian": pedestal-wide bump | "curvature": gamma_eff w/ flow-curvature term
        "exb_envelope_window": (0.85, 1.0),
        "nonlocal_exb_to_tglf": False,  # (unwired) inject gamma_E into TGLF VEXB_SHEAR instead
        "gamma_exb_nl_feature": False,  # opt in: add smeared shear as a GP input feature
    }

Zero-operator validation gate: lambda_c_mult -> 0 with ExB off must reproduce
the raw per-radius code output identically.
"""

import numpy as np
import torch

from mitim_tools.misc_tools.LOGtools import printMsg as print
from . import correlation, exb, spreading

__all__ = ["parse_options", "inject_run_settings", "calculateNonlocal",
           "compute_gamma_exb_nl", "correlation", "exb", "spreading"]

_TURB_KEYS = {          # plasma flux keys <-> quench-factor channels
    "QeMWm2":  "Qe",
    "QiMWm2":  "Qi",
    "Ge1E20m2": "Ge",
    "GZ1E20m2": "GZ",
    "MtJm2":   "Mt",
    "QieMWm3": "Qie",
}
_SPREAD_KEYS = ["QeMWm2", "QiMWm2", "Ge1E20m2", "GZ1E20m2", "MtJm2"]  # no volumetric exchange


def parse_options(opts):
    """Normalize edge_options nonlocal settings into a single dict (or None)."""
    name = str(opts.get("nonlocal_model", "Null") or "Null")
    if name.lower() in ("null", "none", "off"):
        return None
    if name.lower() != "analytic":
        raise ValueError(f"[nonlocality] unknown nonlocal_model {name!r}")
    o = dict(opts.get("nonlocal_model_options", {}) or {})
    o.setdefault("ExB", True)
    o.setdefault("Spreading", True)
    o.setdefault("lambda_c_mult", 8.0)
    o.setdefault("kernel", "gauss")
    o.setdefault("alpha_e", 1.0)
    o.setdefault("p", 2.0)
    o.setdefault("spread_lambda_mult", None)
    # Single pedestal-wide ExB envelope instead of the local/rms-smeared shear.
    # None -> rms smear (physical). "gauss"/"lorentzian" -> fit one smooth bump to
    # |gamma_E| over exb_envelope_window, removing the mid-pedestal notch so the
    # quench is monotone-in-radius (solver-friendly, over-suppresses the notch).
    o.setdefault("exb_envelope", None)
    o.setdefault("exb_envelope_window", (0.85, 1.0))
    # Inject the nonlocal shear into TGLF's VEXB_SHEAR instead of the external
    # per-ky quench (see inject_run_settings). NOT YET WIRED (needs per-surface
    # input.tglf write hook); guarded there.
    o.setdefault("nonlocal_exb_to_tglf", False)
    # Add the rms-smeared shear as a per-surface GP INPUT feature. OFF by default:
    # gamma_exb_nl is highly collinear with the driving gradients, so it usually burns
    # a feature dimension without adding information. The quench (which reads
    # plasma["gamma_exb_nl"] as a physical input) is unaffected by this flag.
    o.setdefault("gamma_exb_nl_feature", False)
    return o


def inject_run_settings(transport_options, nl_options):
    """Force the shear-free run settings the external quench requires.

    TGLF:   VEXB_SHEAR=0, ALPHA_QUENCH=0 (module owns ALL ExB physics).
    QLGYRO: ROTATION_FLAG=0 (its built-in treatment is a forced internal
            quench -- qlgyro_tglf_map.f90 sets tglf_alpha_quench_in=1.0).
    Overrides conflicting user extraOptions with a warning.
    """
    if not (nl_options and nl_options.get("ExB", True)):
        return
    if nl_options.get("nonlocal_exb_to_tglf", False):
        # Alternative mode: hand the nonlocal gamma_E to TGLF's *internal* quench
        # (VEXB_SHEAR) rather than applying the external per-ky quench here.
        # Not wired: VEXB_SHEAR is PER-SURFACE, so it cannot be forced through the
        # global extraOptions used here -- it needs the per-surface gamma_exb_nl
        # (computed later in calculateRotation) written into each input.tglf as it
        # is generated by the transport builder. Fail loudly rather than silently
        # using TGLF's LOCAL shear or double-counting with the external quench.
        raise NotImplementedError(
            "[nonlocality] nonlocal_exb_to_tglf=True is not yet wired: it requires "
            "writing per-surface gamma_exb_nl into VEXB_SHEAR at input.tglf build "
            "time (and disabling the external per-ky quench to avoid double count). "
            "Use the default external quench (nonlocal_exb_to_tglf=False) for now.")
    injections = {
        "tglf":   {"VEXB_SHEAR": 0.0, "ALPHA_QUENCH": 0.0},
        "qlgyro": {"ROTATION_FLAG": 0},
    }
    code_options = (transport_options or {}).get("options", {}) or {}
    for code, extra in injections.items():
        if code not in code_options:
            continue
        run = code_options[code].setdefault("run", {})
        user = run.get("extraOptions") or {}
        for k, v in extra.items():
            if k in user and user[k] != v:
                print(f"[nonlocality] overriding user extraOptions {k}={user[k]} "
                      f"-> {v} ({code}: external quench owns ExB)", typeMsg="w")
            user[k] = v
        run["extraOptions"] = user
        print(f"[nonlocality] {code} run settings injected: {extra}", typeMsg="i")


# --------------------------------------------------------------------------- #
# gamma_E,nl : rms-smeared diamagnetic ExB shear (analytic, differentiable)
# --------------------------------------------------------------------------- #
def compute_gamma_exb_nl(ps, rotation_fine, nl_options):
    """|gamma_E,eff|(r) on the plasma grid, in a/c_s units.

    RMS Gaussian kernel average of the fine-grid gamma_exb [1/s] over
    lambda_c = C rho_s, normalized by the local a/c_s. Fully analytic in the
    profiles -> serves simultaneously as the physical quench input and as the
    deterministic per-surface GP feature ("gamma_exb_nl").
    """
    from mitim_tools.edge_tools.rotation import cubic_interp

    roa_f = rotation_fine["roa"]                      # (n_f,) shared fine grid
    gexb_f = rotation_fine["gamma_exb"]               # (batch, n_f) [1/s]
    p = ps.plasma
    roa1d = p["roa"][0] if p["roa"].dim() > 1 else p["roa"]
    a = p["a"].reshape(-1)[0]                         # minor radius [m] (shared)

    # lambda_c on the fine grid from rho_s (plasma grid -> fine, batch 0:
    # kernel geometry is shared across the batch and detached from autograd).
    rho_s_f = cubic_interp(roa1d, p["rho_s"][:1], roa_f)[0]
    lam = correlation.lambda_c(rho_s_f, float(nl_options["lambda_c_mult"]))
    sigma = 0.5 * lam                                  # Gaussian sigma = lambda_c/2

    r_f = (a * roa_f).detach()
    env_form = nl_options.get("exb_envelope")
    if env_form == "curvature":
        # Curvature-augmented effective shear (local finite-eddy form): the
        # first-derivative gamma_E vanishes at the Er-well extremum, but an eddy
        # of width lambda_c feels the shear averaged over its extent; a Taylor
        # expansion of that average restores the missing flow-curvature term,
        #   gamma_eff = sqrt( gamma_E^2 + (lambda_c^2/12) (d gamma_E/dr)^2 ),
        # with d gamma_E/dr propto d^2 omega0/dr^2. Local (needs only gamma_E and
        # its gradient), fills the notch via curvature, reduces to |gamma_E| where
        # the shear is large. See Ch.6 (eq:rca-gamma-eff). CAVEAT: the derivative
        # is effectively a 3rd pressure derivative (omega0 ~ dp_i/dr) and the 1/12
        # coefficient is heuristic (gyrokinetic calibration pending).
        dg = torch.gradient(gexb_f, spacing=(r_f,), dim=-1)[0]   # (batch,n_f) [1/s/m]
        gexb_eff_f = torch.sqrt(gexb_f ** 2 + (lam ** 2 / 12.0) * dg ** 2)
    elif env_form:
        # Reduced single-envelope model: fit one smooth bump to |gamma_E| per
        # batch element (scipy, on the fine grid) -> replaces both the notch and
        # the local variation with a monotone-in-radius profile. DETACHED: the
        # fit is non-differentiable, which is safe here because gamma_exb_nl is
        # consumed detached (quench input; optional GP feature). Physical rms
        # path (default) stays fully differentiable.
        roa_np = roa_f.detach().cpu().numpy()
        g_np = gexb_f.detach().cpu().numpy()                     # (batch, n_f) [1/s]
        win = tuple(nl_options.get("exb_envelope_window", (0.85, 1.0)))
        env = np.stack([exb.fit_exb_envelope(roa_np, np.abs(g_np[b]),
                                             form=env_form, window=win)
                        for b in range(g_np.shape[0])])
        gexb_eff_f = torch.from_numpy(env).to(gexb_f)            # (batch, n_f) [1/s]
    else:
        gexb_eff_f = correlation.rms_smear(gexb_f, r_f, sigma,
                                           kernel=nl_options["kernel"])  # (batch, n_f) [1/s]

    # back to the plasma grid, normalized to a/c_s (dimensionless, TGLF units)
    gexb_eff = cubic_interp(roa_f, gexb_eff_f, roa1d).clamp(min=0.0)
    return gexb_eff * p["tau_norm"]


# --------------------------------------------------------------------------- #
# main entry point (real-evaluation path)
# --------------------------------------------------------------------------- #
def calculateNonlocal(ps, folder, nl_options):
    """Apply ExB quench + tier-A spreading to a just-evaluated powerstate_edge.

    Called from powerstate_edge.calculate() step 5b (after calculateTransport,
    before calculateElm). Operates on the lifted fine-grid plasma flux arrays;
    factors/operators are defined at the rhoCP anchors and interpolated
    (declared smoothness model, exact at the anchors).
    """
    from mitim_tools.misc_tools import PLASMAtools

    p = ps.plasma
    rho_cp = ps.rhoCP.detach().cpu().numpy()
    rho_fine = p["rho"][0].detach().cpu().numpy()
    diags = {"options": dict(nl_options)}

    # ------------------------------------------------------------------ #
    # 1. ExB quench (per-ky, from the shear-free run's own spectra)
    # ------------------------------------------------------------------ #
    if nl_options.get("ExB", True):
        turb_model = (ps.transport_options.get("evaluator_instance_attributes", {})
                      or {}).get("turbulence_model", "tglf")
        if turb_model != "tglf":
            raise NotImplementedError(
                f"[nonlocality] per-ky quench implemented for TGLF only "
                f"(got {turb_model!r}); QLGYRO needs its native QL-file parser "
                "(electrons-first, amplitude-normalized).")

        if "gamma_exb_nl" not in p:
            raise RuntimeError("[nonlocality] plasma['gamma_exb_nl'] missing -- "
                               "calculateRotation must run with nonlocal ExB enabled")
        gexb_nl_cp = ps._interp_tensor_from_rho_to_rhoCP(p["gamma_exb_nl"])[0]

        tglf_folder = folder / "base_tglf"
        imp_species = 1 + int(getattr(ps, "impurityPosition_transport",
                                      getattr(ps, "impurityPosition", 1))) + 1
        # species index in TGLF files: 1=e-, 2..=ions; gacode impurityPosition is
        # 0-based within ions -> TGLF species = impurityPosition + 2

        factors_cp = {ch: np.ones(len(rho_cp)) for ch in _TURB_KEYS.values()}
        diags["quench"] = {}
        for i, rho in enumerate(rho_cp):
            fac, d = exb.quench_factors(
                tglf_folder, float(rho), float(gexb_nl_cp[i]),
                alpha_e=float(nl_options["alpha_e"]), p=float(nl_options["p"]),
                impurity_species=imp_species)
            for ch, f in fac.items():
                factors_cp[ch][i] = f
            diags["quench"][float(rho)] = {"factors": fac,
                                           "gamma_e_eff": float(gexb_nl_cp[i]),
                                           **{k: d[k] for k in ("fully_quenched_kys",)}}

        print("[nonlocality] ExB quench factors (Qe/Qi per surface): "
              + ", ".join(f"{r:.3f}:({factors_cp['Qe'][i]:.2f}/{factors_cp['Qi'][i]:.2f})"
                          for i, r in enumerate(rho_cp)), typeMsg="i")

    # ------------------------------------------------------------------ #
    # 2. Apply quench + tier-A spreading AT THE ANCHORS, then lift to the fine
    #    grid with a SINGLE linear interpolation per output.
    #
    # Both operators are anchor-native: quench factors are per-anchor from the
    # spectra, and the spread matrix is anchor x anchor. Doing the arithmetic at
    # the anchors and lifting once keeps _tr_turb / _tr / _tr_nl piecewise-linear
    # between knots -- matching how the raw transport flux is represented.
    # The old path multiplied the already-lifted fine flux by a lifted fine
    # factor, i.e. (linear flux) x (linear factor) = spurious quadratic arcs
    # between knots (up to ~15% off the anchor chord for Qe). The anchor values
    # -- all the flux match ever sees (calculateMetrics re-samples _tr_nl at the
    # anchors) -- are identical; only the non-physical between-knot curvature is
    # removed. This also makes the real-eval path use the same anchor-space
    # arithmetic as the GP path in portals_edge.scalarized_objective.
    # ------------------------------------------------------------------ #
    def _to_cp(t):                                   # fine -> anchors (exact for lin.)
        return ps._interp_tensor_from_rho_to_rhoCP(t)

    def _to_fine(a_cp):                              # anchors -> fine (single lift)
        arr = a_cp.detach().cpu().numpy()
        out = np.stack([np.interp(rho_fine, rho_cp, arr[b]) for b in range(arr.shape[0])])
        return torch.from_numpy(out).to(ps.dfT)

    exb_on = nl_options.get("ExB", True)
    M = spreading.get_spread_matrix(ps, nl_options) if nl_options.get("Spreading", True) else None
    if M is not None:
        diags["spread_row_sums"] = M.sum(dim=-1).cpu().numpy()

    for key, ch in _TURB_KEYS.items():
        tk = f"{key}_tr_turb"
        if tk not in p:
            continue
        A_turb = _to_cp(p[tk])                                   # (batch, n_cp)
        fac = (torch.from_numpy(factors_cp[ch]).to(A_turb)
               if exb_on else torch.ones(A_turb.shape[-1]).to(A_turb))
        A_turb_q = A_turb * fac                                  # quench @ anchors
        p[tk] = _to_fine(A_turb_q)                               # single lift (GP-facing)
        sk = f"{key}_tr_turb_stds"
        if sk in p:
            p[sk] = _to_fine(_to_cp(p[sk]) * fac)

        if key in _SPREAD_KEYS:                                  # Qie has no total/spread
            A_neoc = _to_cp(p[f"{key}_tr_neoc"])
            p[f"{key}_tr"] = _to_fine(A_turb_q + A_neoc)         # quenched total
            if M is not None:
                A_spread = torch.einsum("ij,bj->bi", M.to(A_turb_q), A_turb_q)
                p[f"{key}_tr_nl"] = _to_fine(A_spread + A_neoc)  # + spread turb

    # Convective wrappers (Ce, CZ) from the freshly-lifted particle fluxes.
    for ckey, gkey in (("Ce", "Ge1E20m2"), ("CZ", "GZ1E20m2")):
        mult = 1.0 / ps.fImp_orig if ckey == "CZ" else 1.0
        for tt in ("", "_turb", "_neoc", "_nl"):
            src = f"{gkey}_tr{tt}"
            if src in p:
                p[f"{ckey}_tr{tt}"] = PLASMAtools.convective_flux(p["te"], p[src]) * mult

    ps._nonlocal_diags = diags
