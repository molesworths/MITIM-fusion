"""
edge_tools.uq.run
-----------------
Orchestration entry point: run the full uncertainty propagation once for a
flux-matched powerstate and return the risk-aware objective, residual error bars,
and a convergence flag.  Injects the propagated stds into the powerstate so the
rest of PORTALS-edge consumes them unchanged.

Intended call site: right after ``powerstate.flux_match(...)``, using
``FluxMatch_Xopt`` as the fixed DVs.  It operates on a deep copy and never
mutates the caller's solver state except for the explicit std injection into the
powerstate passed in ``inject_into`` (if given).

    from mitim_tools.edge_tools.uq import UQInputs
    from mitim_tools.edge_tools.uq.run import run_edge_uq

    uq = UQInputs()
    uq.add_lcfs("ne", 0.10); uq.add_lcfs("te", 0.10); uq.add_lcfs("ti", 0.15)
    uq.add_source("impurity_source_rate", 0.30)
    uq.source_nominal_fn = lambda ps, nm: ps._cs_model_options["source_rate"]

    res = run_edge_uq(powerstate, uq, X_dvs=powerstate.FluxMatch_Xopt, k_risk=2.0)
    solver_objective = res["robust_objective"]
"""

from typing import Optional

import torch

from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools.edge_tools.uq.driver import UQState, residual_factor
from mitim_tools.edge_tools.uq import objective as obj
from mitim_tools.edge_tools.uq.surrogates import read_transport_gp_stds
from mitim_tools.edge_tools.uq.propagate import std_from_factor


# All plasma quantities that experience the propagated input uncertainty and are
# worth storing (as "{key}{profile_std_suffix}") for later plotting.  Keys absent
# from a given run (e.g. nz_all with charge_state_model="Null") are skipped
# automatically by the scan.  Grouped only for readability.
OBSERVABLE_GROUPS = {
    # y-profiles and their gradient scale lengths (aLy)
    "profiles":   ("te", "ti", "ne", "ni", "aLte", "aLti", "aLne", "aLni"),
    # main-ion neutrals
    "neutrals":   ("n0", "S_ion_main", "nu_ioniz_main", "tau_n0"),
    # impurities / radiation
    "impurities": ("nz_all", "nZ", "aLnZ", "qrad_aurora", "qrad", "Zeff", "fZ"),
    # E x B rotation and shear
    "rotation":   ("w0", "Er", "E_rad", "vexb", "gamma_exb", "vexb_shear",
                   "gamma_p", "mach", "w0_n", "aLw0_n"),
}
DEFAULT_OBSERVABLE_KEYS = tuple(k for grp in OBSERVABLE_GROUPS.values() for k in grp)


def _residual_order(powerstate):
    """(channel, cp) order of the residual vector = predicted_channels x rhoCP."""
    return [(ch, cp) for ch in powerstate.predicted_channels
            for cp in range(len(powerstate.rhoCP))]


def _aly_model_name(powerstate):
    """Return the configured aLy BC-model name (handles dict or str)."""
    bm = getattr(powerstate, "_bc_model_name", None)
    if isinstance(bm, dict):
        return str(bm.get("aLy", ""))
    return str(bm or "")


def _read_aly_bc(powerstate, channels):
    """Per-channel LCFS aLy value after calculateBoundaryConditions."""
    out = {}
    par = getattr(powerstate, "parameterizer", None)
    for ch in channels:
        key = f"aL{ch}"
        val = None
        if par is not None and hasattr(par, "get_nearest_bc"):
            try:
                b = par.get_nearest_bc(key, 1.0)
                if b is not None:
                    val = float(b["val"])
            except Exception:
                val = None
        if val is None:
            bc = getattr(powerstate, "bc_dict", {}).get(key)
            if isinstance(bc, list) and bc:
                val = float(bc[0]["val"]) if isinstance(bc[0], dict) else float(bc[0])
            elif isinstance(bc, dict):
                val = float(bc["val"])
        if val is not None:
            out[ch] = val
    return out


def induced_aly_sigma(powerstate, channels, sigma_y):
    """
    PeretSSF mode: relative LCFS aLy uncertainty *induced* by the LCFS y (ne/Te/Ti)
    uncertainties, by finite-differencing the aLy BC model through
    ``calculateBoundaryConditions`` (cheap; no transport).  Each y channel is
    perturbed by its sigma and the resulting aLy change on every channel is summed
    in quadrature (PeretSSF couples channels), giving relative sigma per channel.
    """
    import copy
    base = _read_aly_bc(powerstate, channels)
    var = {ch: 0.0 for ch in channels}
    for pert_ch, sy in sigma_y.items():
        ps = copy.deepcopy(powerstate)
        opt = dict(getattr(ps, "_bc_model_options", {}) or {})
        if pert_ch not in opt:
            continue
        opt[pert_ch] = opt[pert_ch] * (1.0 + sy)
        ps._bc_model_options = opt
        ps._bc_model_instance = None
        ps.calculateBoundaryConditions()
        aly = _read_aly_bc(ps, channels)
        for ch in channels:
            if ch in aly and ch in base:
                var[ch] += (aly[ch] - base[ch]) ** 2
    return {ch: (var[ch] ** 0.5) / max(abs(base.get(ch, 0.0)), 1e-9)
            for ch in channels if ch in base}


def peret_aly_directions(powerstate, channels, sigma_te, sigma_ti):
    """
    Correlated aLy perturbation directions from PeretSSF, for use with a Fixed
    NOMINAL aLy: the aLy response to a +1-sigma Te and Ti perturbation, computed
    through a PeretSSF bc_model.  Returns [(source, {ch: delta_aLy})] -- a rank-2
    factor of the PeretSSF aLy covariance (correlations across aLne/aLte/aLti
    preserved because one Te/Ti perturbation moves them together).
    """
    from mitim_tools.edge_tools.boundary import build_bc_model
    opts = dict(getattr(powerstate, "_bc_model_options", {}) or {})

    def aly(o):
        m = build_bc_model({"y": "Fixed", "aLy": "PeretSSF"}, o)
        m.get_boundary_conditions(powerstate, batch_idx=0)
        return {ch: float(m.bc_dict[f"aL{ch}"][0]) for ch in channels
                if f"aL{ch}" in m.bc_dict}
    base = aly(opts)
    dirs = []
    for src, sig in (("te", sigma_te), ("ti", sigma_ti)):
        if src not in opts or not sig:
            continue
        o = dict(opts); o[src] = o[src] * (1.0 + sig)
        pert = aly(o)
        delta = {ch: pert.get(ch, base.get(ch, 0.0)) - base.get(ch, 0.0)
                 for ch in base}
        if any(abs(v) > 1e-9 for v in delta.values()):
            dirs.append((src, delta))
    return dirs


def augment_with_peret_aly(x0, L, layout, powerstate, channels,
                           sigma_te, sigma_ti):
    """Append correlated PeretSSF-projected aLy columns (Fixed-nominal case)."""
    from mitim_tools.edge_tools.uq.inputs import ScatterSpec
    dirs = peret_aly_directions(powerstate, channels, sigma_te, sigma_ti)
    if not dirs:
        return x0, L, layout, 0
    dtype = x0.dtype
    base = x0.numel()
    n_new = len(dirs)
    x0_aug = torch.cat([x0, torch.zeros(n_new, dtype=dtype, device=x0.device)])
    k_in = L.shape[1]
    L_aug = torch.zeros((x0_aug.numel(), k_in + n_new), dtype=dtype, device=x0.device)
    L_aug[:L.shape[0], :k_in] = L
    for i, (src, delta) in enumerate(dirs):
        slot = base + i
        L_aug[slot, k_in + i] = 1.0
        layout[f"peret_aly_{src}"] = {
            "slots": [slot], "sigma": torch.ones(1, dtype=dtype),
            "corr_factor": None,
            "scatter": ScatterSpec(kind="aly_cov", key=f"peret_aly_{src}",
                                   slots=[slot], space="absolute",
                                   meta={"delta": delta}),
        }
    print(f"[UQ] PeretSSF-projected aLy: added {n_new} correlated aLy columns "
          f"(Fixed nominal) -> {[d[0] for d in dirs]}", typeMsg="i")
    return x0_aug, L_aug, layout, n_new


def _build_sigma_prior(uq_inputs, powerstate, lcfs_aly_mode):
    """
    Per-channel {ch: {'y':σ, 'aLy':σ}} for the fit-error foot priors.
    y from add_lcfs; aLy from add_bc_option (fixed) or induced via PeretSSF.
    """
    sigma_prior = {}
    for pend in getattr(uq_inputs, "entries", []):
        nm, rs = pend.name, pend.rel_sigma
        if nm in ("ne", "te", "ti"):
            sigma_prior.setdefault(nm, {})["y"] = rs
        elif nm in ("aLne", "aLte", "aLti"):
            sigma_prior.setdefault(nm[2:], {})["aLy"] = rs

    is_peret = "peret" in _aly_model_name(powerstate).lower()
    if lcfs_aly_mode == "fixed" or (lcfs_aly_mode == "auto" and not is_peret):
        return sigma_prior

    # PeretSSF mode: aLy uncertainty is INDUCED from Te/Ti through the bc_model on
    # the ordinary add_lcfs scan columns (correlated aLne/aLte/aLti foot response).
    # Registering add_bc_option too would double-count it -> warn and drop the
    # explicit aLy so only the PeretSSF-induced contribution is used.
    explicit = [ch for ch, d in sigma_prior.items() if "aLy" in d]
    if explicit:
        print(f"[UQ] PeretSSF mode: dropping explicit add_bc_option aLy for "
              f"{explicit} to avoid double-counting the PeretSSF-induced aLy "
              f"(comes from add_lcfs Te/Ti through the bc_model).", typeMsg="w")
        for ch in explicit:
            sigma_prior[ch].pop("aLy", None)
    # PeretSSF / induced: fill aLy for channels that have y but no explicit aLy.
    # Use the explicit PeretSSF projection (works for Fixed OR PeretSSF nominal --
    # it always builds a PeretSSF model), quadrature-summing the Te/Ti direction
    # deltas into a relative sigma per channel.
    need = {ch: d["y"] for ch, d in sigma_prior.items() if "y" in d and "aLy" not in d}
    if need:
        try:
            s_te = sigma_prior.get("te", {}).get("y", 0.0)
            s_ti = sigma_prior.get("ti", {}).get("y", 0.0)
            dirs = peret_aly_directions(powerstate, list(need), s_te, s_ti)
            base_aly = _read_aly_bc(powerstate, list(need))
            induced = {}
            for ch in need:
                var = sum(delta.get(ch, 0.0) ** 2 for _, delta in dirs)
                if ch in base_aly and abs(base_aly[ch]) > 1e-9:
                    induced[ch] = var ** 0.5 / abs(base_aly[ch])
            for ch, s in induced.items():
                sigma_prior.setdefault(ch, {})["aLy"] = s
            print(f"[UQ] PeretSSF-induced aLy sigma: "
                  f"{ {c: round(s,3) for c,s in induced.items()} }", typeMsg="i")
        except Exception as exc:
            print(f"[UQ] induced aLy sigma failed ({exc}); aLy prior omitted",
                  typeMsg="w")
    return sigma_prior


def run_edge_uq(
    powerstate,
    uq_inputs,
    X_dvs: torch.Tensor,
    batch: int = 0,
    k_risk: float = 2.0,
    mode: str = "real_scan",
    do_blackbox: bool = True,
    fd_mode: str = "forward",
    rel_step: float = 1.0,
    rotation_proxy: bool = True,
    transport_proxy=None,
    propagate_fit_error: bool = True,
    fit_error_rel_cap: float = 0.10,
    fit_error_max_dirs=None,
    fit_error_step: float = 0.1,
    lcfs_aly_mode: str = "auto",
    outlier_factor: float = 8.0,
    use_surrogate_gp: bool = False,
    near_optimum: Optional[bool] = None,
    inject_into=None,
    inject_training_stds: bool = False,
    observable_keys=DEFAULT_OBSERVABLE_KEYS,
    profile_std_suffix: str = "_uq_std",
    alpha_conv: float = 0.05,
    n_samples: int = 20000,
):
    """
    Full in-loop UQ pass.

    Parameters
    ----------
    powerstate : flux-matched powerstate_edge.
    uq_inputs  : UQInputs with registered LCFS / source / vtor uncertainties.
    X_dvs      : fixed flux-match DVs (e.g. powerstate.FluxMatch_Xopt).
    k_risk     : risk-aversion multiplier for mu + k*sigma.
    mode       : "real_scan" (default) -- scan every uncertain input through the
        real TGLF/NEO/Aurora model (the use_scan_trick_for_stds analogue on the
        inputs); costs n_inputs extra real evals, baseline reused from the eval
        that just ran.  "linear" -- cheap jvp + FD, but requires a differentiable
        transport (transport_proxy); not for real TGLF/NEO.
    rel_step   : input-scan step in units of the input sigma (1.0 = 1-sigma corner).
    do_blackbox: include finite-difference black-box columns (source/D/V).
    use_surrogate_gp : harvest existing ``*_tr_turb_stds`` and re-add them.  Leave
        FALSE for the in-loop real-eval path: those fields already hold the TGLF/NEO
        model std for this evaluation, so we only want to quadrature-ADD the
        input-propagation contribution, not double-count the model std.  Set TRUE
        only when transport is a bare GP surrogate whose posterior std has not yet
        been written into those fields.
    near_optimum : force sampled objective distribution; if None, auto-detect
        (sampled when the delta sigma is a large fraction of mu).
    inject_into : powerstate to write propagated stds / diagnostics into (default:
        none; pass the live powerstate to store them for the objective layer and
        plotting).  Diagnostic profile/observable stds ("{key}_uq_std"),
        _edge_uq_summary and _edge_uq_breakdown are always stored; they never feed
        GP training.
    inject_training_stds : if True, ALSO quadrature-add the propagated input std
        into the OF flux "{key}_stds" fields that become the surrogate GP training
        noise (legacy behaviour).  Default FALSE: the GP maps deterministic
        features->flux with ~0 noise and stays informative; the input uncertainty
        is consumed at the objective level (sigma_J / robust_objective / Sigma_r),
        i.e. propagated THROUGH the trained map, not injected as training noise.

    Returns
    -------
    dict with keys: mu_J, sigma_J, robust_objective, residual_std, converged,
    chi2_stat, dof, distribution (if sampled), and the raw factors.
    """
    dtype = X_dvs.dtype

    # 1. input factor
    x0, layout = uq_inputs.build_x0(powerstate, batch=batch)
    L = uq_inputs.cholesky_columns(x0, layout)

    # 1b. Append mtanh fit-error columns (theta-covariance eigen-directions) so the
    # SplineMtanhAnalytic fit uncertainty rides the SAME real-model scan, across
    # all channels, into derived profiles / rotation / transport / targets.
    if propagate_fit_error:
        from mitim_tools.edge_tools.uq.fit_uq import augment_with_fit_error
        # Per-channel LCFS y/aLy uncertainties -> foot soft-priors that regularize
        # the degenerate mtanh fit directions.  y from add_lcfs(ne/te/ti), aLy from
        # add_bc_option(aLne/aLte/aLti) (or, in PeretSSF mode, its induced sigma).
        sigma_prior = _build_sigma_prior(uq_inputs, powerstate, lcfs_aly_mode)
        k_in = L.shape[1]
        x0, L, layout, n_fit = augment_with_fit_error(
            x0, L, layout, powerstate, powerstate.predicted_channels,
            rel_cap=fit_error_rel_cap, max_dirs_per_channel=fit_error_max_dirs,
            sigma_prior=sigma_prior)
        # fit-error columns get a small linear step (they are a linearization; a
        # full 1-sigma corner can jump a reconstruction regime discontinuously)
        fit_error_cols = list(range(k_in, k_in + n_fit))
    else:
        fit_error_cols = []
        sigma_prior = _build_sigma_prior(uq_inputs, powerstate, lcfs_aly_mode)

    # 1c. Fixed NOMINAL aLy + PeretSSF-PROJECTED aLy uncertainty: when the user
    # keeps aLy='Fixed' (nominal values they chose) but asks for peret projection,
    # Fixed aLy won't respond to Te/Ti in the scan -- so inject the correlated
    # PeretSSF aLy directions explicitly.  (If nominal aLy is already PeretSSF, the
    # add_lcfs Te/Ti columns carry it automatically; don't double-inject.)
    is_peret_nominal = "peret" in _aly_model_name(powerstate).lower()
    if lcfs_aly_mode == "peret" and not is_peret_nominal:
        s_te = sigma_prior.get("te", {}).get("y", 0.0)
        s_ti = sigma_prior.get("ti", {}).get("y", 0.0)
        if s_te or s_ti:
            x0, L, layout, _ = augment_with_peret_aly(
                x0, L, layout, powerstate, list(powerstate.predicted_channels),
                s_te, s_ti)

    # 2. propagate target + transport (stacked) -> residual factor
    st = UQState(powerstate, X_dvs=X_dvs, layout=layout, batch=batch, stacked=True,
                 rotation_proxy=rotation_proxy, transport_proxy=transport_proxy)
    st.outlier_factor = outlier_factor
    profile_std = None
    # The OF flux keys (Ge/GZ/Qe.../Mt...) are ALWAYS scanned so their native-unit
    # std can be injected into the exact keys the GP reads -- this is the correct,
    # unit-safe path (the old profile_map base wrote Ce/CZ for ne/nZ, which the GP
    # never reads).  User observable_keys are scanned alongside for plotting.
    flux_keys = obj.flux_observable_keys(powerstate.predicted_channels)
    scan_keys = list(dict.fromkeys(list(observable_keys) + list(flux_keys)))
    if mode == "real_scan":
        # Every input column scanned through the real TGLF/NEO/Aurora model
        # (use_scan_trick_for_stds analogue on the inputs).  Baseline reused from
        # the eval that just ran -> only n_inputs extra real evals.
        baseline = st.stacked_baseline_from_plasma(powerstate)
        # Adaptive fit-error step: full step (clean flux, no 1/step noise blow-up)
        # for directions where the reconstruction is linear; small step only for
        # the few that cross a regime boundary.  Cheap profile-only probes.
        col_steps = {}
        if fit_error_cols:
            probe_key = "aLne" if "aLne" in powerstate.plasma else \
                next((f"aL{c}" for c in powerstate.predicted_channels
                      if f"aL{c}" in powerstate.plasma), None)
            if probe_key:
                col_steps = st.fit_step_map(x0, L, fit_error_cols, key=probe_key,
                                            small_step=fit_error_step)
        y0, L_out, obs0, L_obs = st.scan_with_observables(
            x0, L, scan_keys, baseline=baseline, rel_step=rel_step,
            col_steps=col_steps)
    elif mode == "linear":
        # Cheap jvp (differentiable transport) + FD black-box columns.  Requires a
        # differentiable transport (transport_proxy) -- not for real TGLF/NEO.
        y0, L_out = residual_factor(st, x0, L, do_blackbox=do_blackbox, fd_mode=fd_mode)
        L_obs = {}
    else:
        raise ValueError(f"unknown mode {mode!r} (use 'real_scan' | 'linear')")
    tar0, tr0, L_tar, L_tr = obj.split_stacked(y0, L_out)
    source, L_r = obj.residual_from_split(tar0, tr0, L_tar, L_tr)

    # 3. residual covariance + error bars
    Sigma_r, residual_std = obj.residual_covariance(L_r)

    # 4. Store propagated stds.  The features (aLne/aLte/aLti/nuei/tite/betae) are
    #    deterministic functions of the profiles and TGLF/NEO are deterministic, so
    #    the GP maps features->flux with ~0 irreducible noise.  Profile/input
    #    uncertainty is NOT noise in that map -- it is uncertainty in WHERE in
    #    feature space we are, propagated THROUGH the (trained) map at evaluation.
    #    Hence it must NOT pollute the OF flux "{key}_stds" fields that become the
    #    surrogate GP TRAINING noise (doing so drives the GP to its prior when the
    #    propagated std approaches the flux magnitude).  It is retained as the
    #    objective-level products (L_r, Sigma_r, sigma_J, robust_objective) and as
    #    diagnostic profile/observable stds for plotting.
    #
    #    inject_training_stds=True restores the legacy behaviour (write into the
    #    GP-training "{flux}_stds"); leave FALSE for the clean-GP path.
    if inject_into is not None:
        if inject_training_stds:
            obj.inject_flux_stds(inject_into, L_obs, powerstate.predicted_channels)
        profile_std = {}
        flux_set = set(flux_keys)
        for k, Lk in L_obs.items():
            sig = std_from_factor(Lk).reshape(powerstate.plasma[k].shape)
            if k not in flux_set:                    # non-flux observable -> plot key
                inject_into.plasma[k + profile_std_suffix] = sig
            profile_std[k] = sig

    # 4b. Per-source uncertainty breakdown: each column of the observable factor
    # is one uncertainty source, so the per-source std contribution is available
    # for every flux/observable (variance is additive across sources).
    source_breakdown = None
    if L_obs:
        labels = obj.column_source_labels(layout, L)
        source_breakdown = {k: obj.variance_breakdown(Lk, labels)
                            for k, Lk in L_obs.items()}
        if inject_into is not None:
            inject_into._edge_uq_breakdown = source_breakdown

    # 5. objective sigma: delta method, escalate to sampling near the optimum
    mu_J, sigma_J = obj.objective_sigma_delta(source, L_r)
    distribution = None
    if near_optimum is None:
        near_optimum = bool(float(sigma_J) > 0.25 * float(mu_J.clamp_min(1e-30)))
    if near_optimum:
        distribution = obj.objective_sigma_sampling(source, L_r, n_samples=n_samples)
        mu_eff = torch.as_tensor(distribution["mean"], dtype=dtype)
        # upper-tail measure from the sampled (skewed) distribution
        robust = torch.as_tensor(distribution["quantiles"].get(0.95, distribution["mean"]),
                                 dtype=dtype)
    else:
        mu_eff = mu_J
        robust = obj.robust_objective(mu_J, sigma_J, k=k_risk)

    # 6. convergence
    converged, stat, dof, thr = obj.chi2_convergence(source, Sigma_r, alpha=alpha_conv)

    print(f"[UQ] mu_J={float(mu_eff):.4g}  sigma_J={float(sigma_J):.4g}  "
          f"robust={float(robust):.4g}  converged={converged}", typeMsg="i")

    # Persist a compact UQ summary ONTO the powerstate so it survives in the saved
    # powerstate.pkl and can be plotted later alongside the profile/flux stds:
    # residual bands (target/transport/combined) + objective + convergence.
    if inject_into is not None:
        inject_into._edge_uq_summary = {
            "order": _residual_order(powerstate),           # [(channel, cp), ...]
            "residual_mean": source.detach(),
            "residual_std": residual_std.detach(),
            "target_std": std_from_factor(L_tar).detach(),
            "transport_std": std_from_factor(L_tr).detach(),
            "mu_J": float(mu_eff), "sigma_J": float(sigma_J),
            "robust_objective": float(robust),
            "converged": converged, "chi2_stat": stat, "dof": dof,
            "chi2_threshold": thr, "near_optimum": near_optimum,
            "profile_std_keys": list((profile_std or {}).keys()),
            "profile_std_suffix": profile_std_suffix,
        }

    return {
        "mu_J": float(mu_eff),
        "sigma_J": float(sigma_J),
        "robust_objective": float(robust),
        "residual_mean": source.detach(),
        "residual_std": residual_std.detach(),
        "Sigma_r": Sigma_r.detach(),
        "L_r": L_r.detach(),
        "converged": converged,
        "chi2_stat": stat,
        "dof": dof,
        "chi2_threshold": thr,
        "near_optimum": near_optimum,
        "distribution": distribution,
        "profile_std": profile_std,   # {key: std tensor on plasma grid} (y & aLy)
        "source_breakdown": source_breakdown,  # {obs_key: {source: std contribution}}
        # Per-observable covariance FACTORS (m_key, k) with shared input columns:
        # the anchor-tier product the cheap GP-delta tier consumes
        # (gp_propagate.feature_factor_from_L_obs -> gp_flux_covariance).  Empty
        # in mode="linear".  Detached: these are a measured band, not a graph.
        "L_obs": {k: Lk.detach() for k, Lk in L_obs.items()},
    }
