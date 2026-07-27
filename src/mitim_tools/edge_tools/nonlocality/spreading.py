"""spreading.py -- tier-A turbulence spreading operator (frozen coupling matrix).

Tier A of the NONLOCAL-OFFLINE design: the physical turbulent flux profiles are
convolved radially with a Gaussian kernel of width ~lambda_c, absorbing at both
domain edges (losses to SOL beyond the LCFS and to the core inside the inner
bound are dropped, exported as diagnostics -- "let the kernel handle the
losses"). In-loop, with n_cp anchors, this is honestly an (n_cp x n_cp)
coupling matrix; sub-anchor redistribution structure is declared unobservable.

The matrix is FROZEN on first construction (lambda from the then-current
rho_s(r)) and stored on the powerstate as ``_nonlocal_spread_M``:
  * the GP path (portals_edge.scalarized_objective) applies the SAME matrix to
    surrogate-predicted fluxes, so real and surrogate residuals see an
    identical linear operator (spreading cannot be a GP feature -- it depends
    on the flux field the GP predicts);
  * rho_s drift over a run (~10-20%) is subsumed by the lambda x{0.5,1,2}
    sensitivity that accompanies any production result.

Same-kernel-for-all-channels: spread turbulence carries its donor's
cross-phases, so channel ratios remain those of the donor region (tier-A
donor convention; no local QL response is fabricated anywhere).

Physics tier (Fisher/KPP steady state with alpha(r) = gamma_eff/I_local
calibration, single constant D0) is deliberately NOT in-loop: its
redistribution factor depends on the intensity profile, which would break
real/GP-path consistency mid-iteration. It belongs in the offline pipeline
and final-point analysis.
"""

import numpy as np
import torch

from mitim_tools.misc_tools.LOGtools import printMsg as print
from . import correlation

__all__ = ["get_spread_matrix", "build_spread_matrix"]


def build_spread_matrix(ps, nl_options):
    """Construct the (n_cp, n_cp) tier-A operator from the current powerstate."""
    p = ps.plasma
    a = float(p["a"].reshape(-1)[0])

    # Anchor radii and domain bounds in meters (roa * a); batch 0 geometry.
    roa1d = p["roa"][0] if p["roa"].dim() > 1 else p["roa"]
    rho1d = p["rho"][0] if p["rho"].dim() > 1 else p["rho"]
    roa_cp = torch.from_numpy(
        np.interp(ps.rhoCP.detach().cpu().numpy(),
                  rho1d.detach().cpu().numpy(),
                  roa1d.detach().cpu().numpy())
    ).to(ps.dfT)
    r_anchors = a * roa_cp
    r_bounds = torch.tensor([a * float(roa1d.min()), a * float(roa1d.max())],
                            dtype=r_anchors.dtype)

    mult = nl_options.get("spread_lambda_mult") or nl_options["lambda_c_mult"]
    rho_s_cp = ps._interp_tensor_from_rho_to_rhoCP(p["rho_s"])[0]
    lam = correlation.lambda_c(rho_s_cp, float(mult))
    sigma = 0.5 * lam

    M = correlation.spread_matrix(r_anchors, r_bounds, sigma)
    print("[nonlocality] spread matrix frozen: lambda_c[mm]="
          + "/".join(f"{1e3*float(l):.1f}" for l in lam)
          + "; row sums=" + "/".join(f"{float(s):.3f}" for s in M.sum(dim=-1)),
          typeMsg="i")
    return M


def get_spread_matrix(ps, nl_options):
    """Frozen-per-run accessor: build once, reuse everywhere (real + GP paths)."""
    M = getattr(ps, "_nonlocal_spread_M", None)
    if M is None:
        M = build_spread_matrix(ps, nl_options)
        ps._nonlocal_spread_M = M
    return M
