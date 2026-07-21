"""
edge_tools.uq.fit_uq
--------------------
Propagate the SplineMtanhAnalytic mtanh-base fit uncertainty (the per-channel
parameter covariance ``parameterizer._fit_cov[prof]`` over theta={s,c,w1,r})
through the SAME real-model scan used for input uncertainty.

Each dominant eigen-direction of the (capped) fit covariance becomes an extra
scan column: it perturbs theta via ``parameterizer._theta_offset[prof]``, rebuilds
the profile, and re-runs the full chain (derived profiles -> rotation ->
transport/targets).  So the fit error reaches every observable and flux exactly
like an input, at one real eval per direction.

Conditioning: a poorly-constrained theta (e.g. w1 when A is tiny) can carry a
huge raw variance, but the profile is nearly insensitive to it, so it is largely
harmless -- still, we cap each theta's marginal fit std to ``rel_cap`` (relative)
before building directions, which only bites those low-impact directions.

Corrector projection: SplineMtanhAnalytic re-pins aLy at the interior knots to
the fixed input DVs independently of theta (the residual corrector), so a raw
theta perturbation is partly cancelled when the profile is rebuilt.  Ranking
directions by raw theta variance therefore (a) mis-orders them and (b) can spend
real evals on directions the corrector nearly annihilates.  We instead propagate
Sigma_theta through the *corrector-projected* profile Jacobian
``R = d aLy_corrected(x)/d theta`` (:func:`_corrector_projected_jac`) and take the
dominant modes of the profile-space covariance ``C_P = R Sigma_theta Rᵀ``.  By
construction R has ~zero rows at the knots, so those modes -- and hence the
propagated fit-error band -- pinch to zero at the knots and are nonzero only
between/beyond them, while being ordered by TRUE observable impact.
"""

import numpy as np
import torch

from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools.edge_tools.uq.inputs import ScatterSpec


def _foot_jacobians(par, theta, y_bc, aLy_bc, x_foot, eps=1e-4):
    """Finite-difference d y/d theta and d aLy/d theta of the BASE mtanh at the
    near-LCFS foot points x_foot.  (base = the fit target; the corrector overlay
    is a separate downstream term.)"""
    x_foot = np.asarray(x_foot, dtype=float)

    def eval_at(th):
        g, D0, delta, m, c, _ = par._theta_to_phys(th, y_bc, aLy_bc)
        y = par._y_mtanh(x_foot, g, D0, delta, m, c, y_bc)
        dy = par._dydx_mtanh(x_foot, g, D0, delta, m, c)
        aLy = -dy / np.where(np.abs(y) < 1e-12, 1e-12, y)
        return np.asarray(y, float), np.asarray(aLy, float)

    y0, a0 = eval_at(theta)
    n = len(theta)
    gy = np.zeros((len(x_foot), n))
    ga = np.zeros((len(x_foot), n))
    for j in range(n):
        tp = np.array(theta, dtype=float); tp[j] += eps
        yj, aj = eval_at(tp)
        gy[:, j] = (yj - y0) / eps
        ga[:, j] = (aj - a0) / eps
    return np.nan_to_num(gy), np.nan_to_num(ga)


def regularized_fit_cov(par, prof, sigma_y_rel, sigma_aLy_rel, foot_x=None,
                        n_foot=3):
    """
    Fit covariance with the LCFS y and aLy BC uncertainties folded in as soft
    priors on the near-LCFS foot shape.  Both BCs are hard-enforced in the
    reconstruction (zero Jacobian), so they are invisible to (JᵀJ) -- this adds
    them back as the information they carry:

        Cov = pinv( JᵀJ/sigma2  +  Σ_foot [ g_aLy g_aLyᵀ/σ_aLy²  +  g_y g_yᵀ/σ_y² ] )

    where σ_aLy = sigma_aLy_rel*|aLy_bc|, σ_y = sigma_y_rel*|y_bc| (absolute LCFS
    uncertainties).  Only bites the degenerate c/w/r directions that swing the
    under-resolved foot; leaves well-determined directions ~unchanged.  Falls
    back to the data-only cov if the pieces are missing.
    """
    data = getattr(par, "_fit_cov_data", {}).get(prof)
    if data is None or data.get("JtJ") is None:
        return getattr(par, "_fit_cov", {}).get(prof)
    JtJ = np.asarray(data["JtJ"], float)
    sig2 = max(float(data["sigma2"]), 1e-30)
    theta = np.asarray(data["theta"], float)
    y_bc, aLy_bc = float(data["y_bc"]), float(data["aLy_bc"])

    if foot_x is None:
        last_knot = float(np.max(par.knots))
        foot_x = np.linspace(0.5 * (last_knot + 1.0), 1.0 - 1e-3, n_foot)

    gy, ga = _foot_jacobians(par, theta, y_bc, aLy_bc, foot_x)
    s_a = max(sigma_aLy_rel * abs(aLy_bc), 1e-9)
    s_y = max(sigma_y_rel * abs(y_bc), 1e-9)

    prec = JtJ / sig2
    for i in range(len(foot_x)):
        prec = prec + np.outer(ga[i], ga[i]) / s_a ** 2
        prec = prec + np.outer(gy[i], gy[i]) / s_y ** 2
    return np.linalg.pinv(prec)


def cap_covariance(cov, theta, rel_cap=0.10, floor=1e-3):
    """Scale rows/cols so each theta's marginal std <= rel_cap*|theta| (floored),
    preserving correlations."""
    cov = np.asarray(cov, dtype=float)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    sig = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    cap = rel_cap * np.maximum(np.abs(theta), floor)
    scale = np.ones_like(sig)
    m = sig > cap
    scale[m] = cap[m] / np.maximum(sig[m], 1e-30)
    D = np.diag(scale)
    return D @ cov @ D


def _edge_grid(powerstate, par, n_fallback=200):
    """Transport-relevant evaluation grid restricted to the parameterizer domain
    x >= x0 (the scan evaluates profiles on powerstate.plasma['roa']).  Falls
    back to a dense linspace over [x0, 1) if roa is unavailable."""
    x0 = float(getattr(par, "x0", 0.85))
    roa = None
    plasma = getattr(powerstate, "plasma", None)
    if isinstance(plasma, dict) and plasma.get("roa") is not None:
        r = plasma["roa"]
        roa = (r.detach().cpu().numpy() if torch.is_tensor(r) else np.asarray(r)).ravel()
    if roa is None or roa.size == 0:
        return np.linspace(x0, 1.0 - 1e-3, n_fallback)
    g = np.unique(roa[np.isfinite(roa)])
    g = g[g >= x0 - 1e-9]
    return g if g.size >= 4 else np.linspace(x0, 1.0 - 1e-3, n_fallback)


def _corrector_projected_jac(par, prof, x_grid, eps=1e-4):
    """FD Jacobian ``R[:, j] = d aLy_corrected(x)/d theta_j`` on ``x_grid``,
    through the FULL reconstruction (mtanh base + interior residual corrector).

    Perturbs ``par._theta_offset[prof]`` and rebuilds via ``get_aLy`` -- exactly
    the path the real-model scan takes -- so R inherits the corrector's knot
    re-pinning: because ``aLy_corrected(knot_i)`` is the fixed input DV regardless
    of theta, R's rows at the knots are ~0.  The parameterizer's ``_theta_offset``
    is restored on exit (never pollutes the live scan state)."""
    x_grid = np.asarray(x_grid, dtype=float)
    pv = getattr(par, "params", {}).get(prof)
    if pv is None:
        return None
    prev = getattr(par, "_theta_offset", None)

    def rebuild(offset):
        par._theta_offset = {prof: np.asarray(offset, dtype=float)}
        return np.asarray(par.get_aLy({prof: pv}, x_grid)[prof], dtype=float)

    try:
        a0 = rebuild(np.zeros(4))
        R = np.zeros((x_grid.size, 4))
        for j in range(4):
            off = np.zeros(4); off[j] = eps
            R[:, j] = (rebuild(off) - a0) / eps
    finally:
        par._theta_offset = prev
    return np.nan_to_num(R)


def _projected_directions(par, prof, cov, theta, x_grid, rel_cap=0.10,
                          rel_tol=1e-3, max_dirs=None):
    """Corrector-projected fit-error directions for one channel.

    Returns ``(dcols, S, frac)``: ``dcols[:, k]`` is the 4-vector theta offset
    whose corrector-projected profile response is the k-th dominant 1-sigma mode
    of ``C_P = R Sigma_theta Rᵀ`` (so a unit scan amplitude = 1 sigma of that
    mode, matching the old eigen_columns convention); ``S[k]`` is that mode's
    profile-space std; ``frac`` is the fraction of tr(C_P) the retained modes
    carry.  Falls back to the theta-space ``eigen_columns`` if R is unavailable."""
    Sigma = cap_covariance(cov, theta, rel_cap=rel_cap)
    wc, Vc = np.linalg.eigh(np.asarray(Sigma, dtype=float))
    wc = np.clip(wc, 0.0, None)
    Lc = Vc * np.sqrt(wc)                       # Lc Lcᵀ = Sigma  (4x4)

    R = _corrector_projected_jac(par, prof, x_grid)
    if R is None:
        cols = eigen_columns(Sigma, max_dirs=max_dirs)
        return cols, np.sqrt((cols ** 2).sum(axis=0)), 1.0

    B = R @ Lc                                  # (n_x, 4): profile response of unit-normal theta draws
    U, S, Vt = np.linalg.svd(B, full_matrices=False)
    total = float((S ** 2).sum())
    if total <= 0.0 or S.size == 0 or S.max() <= 0.0:
        return np.zeros((4, 0)), np.zeros(0), 0.0
    keep = S > rel_tol * S.max()
    dcols = (Lc @ Vt.T)[:, keep]                # theta offsets producing 1-sigma profile modes
    S_keep = S[keep]
    order = np.argsort(-S_keep)
    dcols, S_keep = dcols[:, order], S_keep[order]
    if max_dirs is not None:
        dcols, S_keep = dcols[:, :max_dirs], S_keep[:max_dirs]
    frac = float((S_keep ** 2).sum() / total)
    return dcols, S_keep, frac


def eigen_columns(cov, rel_tol=1e-6, max_dirs=None):
    """Columns L s.t. cov ~= L Lᵀ, one per retained eigen-direction (scaled by
    sqrt eigenvalue = the 1-sigma perturbation).  Drops near-zero directions."""
    w, V = np.linalg.eigh(np.asarray(cov, dtype=float))
    w = np.clip(w, 0.0, None)
    if w.max() <= 0:
        return np.zeros((cov.shape[0], 0))
    keep = w > rel_tol * w.max()
    cols = V[:, keep] * np.sqrt(w[keep])
    # dominant first
    order = np.argsort(-(cols ** 2).sum(axis=0))
    cols = cols[:, order]
    if max_dirs is not None:
        cols = cols[:, :max_dirs]
    return cols


def augment_with_fit_error(x0, L, layout, powerstate, predicted_channels,
                           rel_cap=0.10, max_dirs_per_channel=None,
                           sigma_prior=None):
    """
    Append corrector-projected fit-error direction columns to (x0, L, layout).

    Reads ``powerstate.parameterizer._fit_cov[prof]`` and ``._current_theta[prof]``
    (populated by the baseline calculate()).  For each channel it propagates the
    theta covariance through the corrector-projected profile Jacobian
    (:func:`_projected_directions`) and keeps the dominant profile-space modes, so
    the columns are ordered by true observable impact and pinch to zero at the
    knots.  Returns (x0_aug, L_aug, layout, n_new).  Each new column is a unit
    amplitude (nominal 0) whose scatter sets
    ``_theta_offset[prof] = amplitude * direction``.
    """
    par = getattr(powerstate, "parameterizer", None)
    fitcov = getattr(par, "_fit_cov", None) if par is not None else None
    theta_all = getattr(par, "_current_theta", {}) if par is not None else {}
    if not fitcov:
        print("[UQ] no mtanh fit covariance found; fit-error propagation skipped",
              typeMsg="w")
        return x0, L, layout, 0

    dtype = x0.dtype
    sigma_prior = sigma_prior or {}
    x_grid = _edge_grid(powerstate, par)
    directions = []   # (prof, direction 4-vec)
    for prof in predicted_channels:
        theta = theta_all.get(prof)
        if theta is None:
            continue
        # Fold the LCFS y/aLy BC uncertainties in as foot soft-priors when
        # available (regularizes the degenerate directions); else data-only cov.
        pri = sigma_prior.get(prof, {})
        if "y" in pri and "aLy" in pri:
            cov = regularized_fit_cov(par, prof, pri["y"], pri["aLy"])
        else:
            cov = fitcov.get(prof)
        if cov is None:
            continue
        # Propagate Sigma_theta through the corrector-projected profile Jacobian
        # so directions pinch to zero at the knots and rank by true impact.
        cols, _S, frac = _projected_directions(
            par, prof, cov, theta, x_grid,
            rel_cap=rel_cap, max_dirs=max_dirs_per_channel)
        if cols.shape[1]:
            print(f"[UQ] fit-error {prof}: {cols.shape[1]} corrector-projected "
                  f"mode(s), capturing {frac:.1%} of profile fit-variance",
                  typeMsg="i")
        for j in range(cols.shape[1]):
            directions.append((prof, cols[:, j]))

    n_new = len(directions)
    if n_new == 0:
        return x0, L, layout, 0

    base = x0.numel()
    x0_aug = torch.cat([x0, torch.zeros(n_new, dtype=dtype, device=x0.device)])
    k_in = L.shape[1]
    L_aug = torch.zeros((x0_aug.numel(), k_in + n_new), dtype=dtype, device=x0.device)
    L_aug[:L.shape[0], :k_in] = L
    for i, (prof, direction) in enumerate(directions):
        slot = base + i
        L_aug[slot, k_in + i] = 1.0
        layout[f"fit_{prof}_{i}"] = {
            "slots": [slot],
            "sigma": torch.ones(1, dtype=dtype),
            "corr_factor": None,
            "scatter": ScatterSpec(
                kind="theta_pert", key=f"fit_{prof}_{i}", slots=[slot],
                space="absolute", meta={"prof": prof,
                                        "direction": [float(v) for v in direction]},
            ),
        }
    print(f"[UQ] fit-error: added {n_new} theta-perturbation scan columns "
          f"across {len(set(p for p,_ in directions))} channels (cap {rel_cap:.0%})",
          typeMsg="i")
    return x0_aug, L_aug, layout, n_new
