"""
edge_tools.uq.gp_propagate
---------------------------
Cheap-tier input-uncertainty propagation through the TRAINED flux surrogate.

Two-tier scheme (see edge_uq_package_architecture / the design discussion):

  anchor tier  (run.run_edge_uq, mode="real_scan"): ONCE per real outer eval,
     propagate the uncertain inputs (LCFS ne/te/ti, sources, D/V, mtanh fit)
     through the REAL model to get the truth residual band AND the input-induced
     covariance of the GP FEATURES (aLne/aLte/aLti/nuei/tite/betae) at the
     confirmed point -> feature factor ``L_feat`` (n_feat, k).  Cost: n_inputs
     extra real evals, baseline reused.

  cheap tier   (this module): hold ``L_feat`` over the trust region and push it
     through the trained GP posterior mean ANALYTICALLY, so the input band is
     available at surrogate cost (NO transport) as the metric for the 1000s of
     inner-solve / acquisition calls:

        Sigma_flux(x) = J_gp(x) Sigma_feat J_gp(x)^T + Sigma_gp_posterior(x)
                      = L_flux(x) L_flux(x)^T
        L_flux(x) = [ push_columns(gp_mean_of_features, feat(x), L_feat)
                      | diag(sigma_gp(x)) ]

The features->flux map is deterministic (~0 nugget), so this band rides on the
RESIDUAL / OBJECTIVE (Mahalanobis W + chi2 convergence), never on the GP training
targets -- which keeps the surrogate informative regardless of the band size.

Typical wiring (per outer iteration, both CALM and BO):
    # anchor tier (truth band + feature factor), once per real eval:
    res = run_edge_uq(powerstate, uq_inputs, X_dvs=x_anchor, inject_into=None,
                      observable_keys=FEATURE_KEYS_DEFAULT + (...,))
    L_feat, feat_order = feature_factor_from_L_obs(res["L_obs"], FEATURE_KEYS_DEFAULT)
    # cheap tier (metric held constant over the inner solve):
    flux0, L_flux = gp_flux_covariance(gp_mean_of_features, feat0, L_feat, sigma_gp)
    Sigma_flux = cov_from_factor(L_flux)
"""

import torch

from mitim_tools.edge_tools.uq import propagate
from mitim_tools.misc_tools.LOGtools import printMsg as print


# GP feature set for the edge flux surrogates (local, per rhoCP).
FEATURE_KEYS_DEFAULT = ("aLne", "aLte", "aLti", "nuei", "tite", "betae")


def feature_factor_from_L_obs(L_obs, feature_keys=FEATURE_KEYS_DEFAULT):
    """
    Stack the per-feature covariance factors produced by ``run_edge_uq`` (its
    ``L_obs`` dict, keyed by observable) into a single feature factor
    ``L_feat`` (n_feat_points, k).

    Rows are ordered as ``feature_keys`` x rhoCP (each key contributes an
    (n_rhoCP,)-length block), matching the flat feature vector the GP consumes
    when the caller flattens ``transformationInputs`` in the same order.  All
    blocks share the SAME k input columns, so cross-feature (and, downstream,
    cross-channel) correlation is preserved through the push.

    Returns
    -------
    L_feat : (sum_k n_pts_key, k) torch.Tensor, or None if no feature key present.
    order  : list[(key, local_index)] naming each row (for alignment / debugging).
    """
    blocks, order = [], []
    for key in feature_keys:
        Lk = L_obs.get(key, None)
        if Lk is None:
            continue
        blocks.append(Lk)
        order.extend((key, i) for i in range(Lk.shape[0]))
    if not blocks:
        print("[UQ] feature_factor_from_L_obs: no feature keys in L_obs "
              f"(wanted {list(feature_keys)}); no input band available", typeMsg="w")
        return None, []
    # All factors must share the column count k (they were propagated from the
    # same input columns); guard so a mismatched observable can't be concatenated.
    k = blocks[0].shape[1]
    if any(b.shape[1] != k for b in blocks):
        raise ValueError("feature factors have inconsistent column counts; they "
                         "must come from the same run_edge_uq propagation")
    L_feat = torch.cat(blocks, dim=0)
    return L_feat, order


def gp_flux_covariance(gp_mean_of_features, feat0, L_feat, sigma_gp=None,
                       batched=False):
    """
    Push the input-induced feature covariance factor through the trained GP
    posterior mean (delta method), optionally folding the GP's own posterior
    (epistemic) std in as independent columns.

    Parameters
    ----------
    gp_mean_of_features : callable(features_1d) -> flux_1d
        Differentiable GP posterior-mean as a function of the FLAT feature vector
        (composed of torch ops so ``torch.func.jvp`` can differentiate it).
        Supplied by the caller (CALM / BO), which owns the fitted gpmodel and the
        feature ordering; ``feat0`` and ``L_feat`` rows must match that ordering.
    feat0 : (n_feat,) torch.Tensor
        Flat feature vector at the evaluation point.
    L_feat : (n_feat, k) torch.Tensor
        Input-induced feature covariance factor (anchor tier).
    sigma_gp : (m,) torch.Tensor or None
        Per-output GP posterior std (epistemic); folded via add_epistemic_columns.
    batched : bool
        Passed to push_columns (False loops the JVPs; use False for gpmodels that
        are not vmap-safe).

    Returns
    -------
    flux0 : (m,) GP posterior mean at feat0.
    L_flux : (m, k [+ m]) output covariance factor; Sigma_flux = L_flux @ L_flux.T.
    """
    if L_feat is None:
        flux0 = gp_mean_of_features(feat0)
        m = flux0.numel()
        L_flux = (propagate.add_epistemic_columns(
                      torch.zeros((m, 0), dtype=flux0.dtype, device=flux0.device),
                      sigma_gp)
                  if sigma_gp is not None
                  else torch.zeros((m, 0), dtype=flux0.dtype, device=flux0.device))
        return flux0, L_flux

    flux0, L_flux = propagate.push_columns(gp_mean_of_features, feat0, L_feat,
                                           batched=batched)
    if sigma_gp is not None:
        L_flux = propagate.add_epistemic_columns(L_flux, sigma_gp)
    return flux0, L_flux


def flux_covariance(L_flux):
    """Dense Sigma_flux = L_flux @ L_flux.T (convenience re-export)."""
    return propagate.cov_from_factor(L_flux)


def flux_std(L_flux):
    """Marginal per-output std = sqrt(row sum of squares) (convenience re-export)."""
    return propagate.std_from_factor(L_flux)
