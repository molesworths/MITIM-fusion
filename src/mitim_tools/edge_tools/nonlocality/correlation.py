"""correlation.py -- eddy correlation length and radial kernels for the nonlocal operators.

One correlation-length closure feeds BOTH nonlocal mechanisms (ExB smearing and
turbulence spreading), so they cannot drift apart:

    lambda_c(r) = C * rho_s(r)          [m]

with C ("lambda_c_mult") a fixed, literature-anchored multiplier. BES-measured
ion-scale radial correlation lengths are ~5-12 rho_s (McKee / Shafer et al.,
DIII-D); the default C=8 lands mid-band (~1 cm at typical DIII-D edge rho_s).
C is deliberately NOT derived from per-evaluation spectra: the kx_rms->lambda_c
conversion carries its own O(1) convention ambiguity, so a fixed C plus the
x{0.5, 1, 2} sensitivity option is an honest restatement of the actual
uncertainty, and it keeps gamma_E,eff analytic (profile-only -> usable both as
the physical input and as a deterministic GP feature).

Kernel conventions (two, deliberately different -- see the module ledger):
  * ExB smearing (input-field averaging): Gaussian weights, CLIP-RENORMALIZED
    over the domain. The field beyond the boundary is unknown; renormalizing
    over known support is the right estimator for an average.
  * Spreading (transported-quantity redistribution): Gaussian kernel integrated
    over donor cells, NOT renormalized at the domain edges -- kernel mass beyond
    the boundary is absorbed (lost to SOL/core). "Let the kernel handle the
    losses"; the dropped mass is exported as a diagnostic, never re-injected.

Kernel shape: Gaussian default (consistent with the Gaussian-envelope fits of
the BES correlation functions that anchor C). The MinT draft's Lorentzian
(Eq. A40) is available via kernel="lorentzian" for consistency studies; shape
differences are O(1) and absorbable into C.
"""

import torch

__all__ = ["lambda_c", "smear_weights", "rms_smear", "spread_matrix"]


def lambda_c(rho_s, mult=8.0):
    """Eddy radial correlation length lambda_c = C * rho_s [m]."""
    return mult * rho_s


def _kernel(dr2, sigma2, kernel="gauss"):
    if kernel == "gauss":
        return torch.exp(-dr2 / (2.0 * sigma2))
    if kernel == "lorentzian":
        # MinT draft Eq. (A40): 1 / ((r-r')^2 + Delta^2), Delta ~ sigma
        return sigma2 / (dr2 + sigma2)
    raise ValueError(f"[nonlocality] unknown kernel {kernel!r}")


def smear_weights(r, sigma, kernel="gauss"):
    """(n, n) input-averaging weights W_ij for a field sampled on the shared 1-D
    grid ``r`` [m], with receiver-centered width ``sigma`` (n,) [m].

    Row-normalized over the domain (clip-renormalize convention). ``r`` is
    assumed uniformly spaced (the oversampled rotation grid is), so quadrature
    weights cancel in the normalization. Detached: kernel geometry is treated
    as a constant w.r.t. the autograd graph (the smeared FIELD stays on it).
    """
    r = r.detach()
    sigma = sigma.detach().clamp(min=1e-6)
    dr2 = (r.unsqueeze(-1) - r.unsqueeze(-2)) ** 2
    w = _kernel(dr2, (sigma ** 2).unsqueeze(-1), kernel=kernel)
    return w / w.sum(dim=-1, keepdim=True)


def rms_smear(field, r, sigma, kernel="gauss"):
    """RMS kernel average: sqrt( sum_j W_ij field_j^2 ), batched over dim 0.

    RMS (not signed mean) is load-bearing physics, not a numerical choice:
    shear decorrelation is sign-blind (opposite-sign flanks both tear an eddy),
    and a SIGNED kernel average of gamma_E across a symmetric Er well cancels
    at the well bottom -- exactly the pathology the nonlocal model exists to
    fix. The rms window also captures the flow-curvature ("bowing") distortion
    that survives at the shear zero crossing. Quadratic weighting is consistent
    with the shearing rate entering the Hahm-Burrell two-point decorrelation
    quadratically.

    field : (batch, n)   e.g. gamma_exb on the fine rotation grid [any units]
    r     : (n,)         radius [m]
    sigma : (n,)         kernel width [m]
    """
    w = smear_weights(r, sigma, kernel=kernel)              # (n, n)
    return torch.sqrt(torch.einsum("ij,bj->bi", w, field ** 2).clamp(min=0.0))


def spread_matrix(r_anchors, r_bounds, sigma, kernel="gauss"):
    """Tier-A spreading operator M (n, n): Q_spread_i = sum_j M_ij Q_j.

    M_ij = integral over donor cell j of the receiver-centered kernel
    K(r_i - r'; sigma_i), cells partitioned by anchor midpoints and the domain
    bounds. Exact Gaussian cell integrals (erf), so:
      * sigma -> 0 gives M -> identity (zero-operator limit, validation gate);
      * rows are NOT renormalized: kernel mass beyond [r_bounds] is absorbed
        (SOL beyond the LCFS, core inside the inner bound). Row sums < 1 near
        the edges are the losses, exported as diagnostics by the caller.

    r_anchors : (n,) anchor radii [m], increasing (rhoCP surfaces)
    r_bounds  : (2,) domain bounds [m]
    sigma     : (n,) receiver kernel width [m]
    """
    if kernel != "gauss":
        raise NotImplementedError("[nonlocality] spread_matrix: Gaussian only "
                                  "(cell integrals are analytic)")
    r = r_anchors.detach()
    sigma = sigma.detach().clamp(min=1e-6)
    n = r.shape[0]
    # Donor-cell boundaries: domain bound | midpoints | domain bound
    b = torch.empty(n + 1, dtype=r.dtype, device=r.device)
    b[0], b[-1] = r_bounds[0], r_bounds[1]
    b[1:-1] = 0.5 * (r[1:] + r[:-1])

    def _cdf(x, s):  # Gaussian CDF
        return 0.5 * (1.0 + torch.erf(x / (s * 1.4142135623730951)))

    # M_ij = Phi((b_{j+1}-r_i)/sigma_i) - Phi((b_j - r_i)/sigma_i)
    upper = _cdf(b[1:].unsqueeze(0) - r.unsqueeze(-1), sigma.unsqueeze(-1))
    lower = _cdf(b[:-1].unsqueeze(0) - r.unsqueeze(-1), sigma.unsqueeze(-1))
    return upper - lower
