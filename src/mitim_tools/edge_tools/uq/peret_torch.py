"""
edge_tools.uq.peret_torch
--------------------------
Torch-differentiable PeretSSF LCFS gradient-scale-length model (step 2 of the UQ
plan).

``boundary._ssf_decay_lengths`` solves a scalar fixed point for lambda_p with
``brentq`` -- robust but a graph break.  Here the *value* is still obtained by a
bracketed root solve (reusing brentq when available), and gradients are restored
by a single Newton step at the converged root:

    lp = lp*  -  r(lp*) / r'(lp*)

Because ``r(lp*) = 0`` numerically the value is unchanged, but autograd now sees
the implicit-function-theorem sensitivity

    d lp*/d theta = -(dr/dlp)^{-1} (dr/dtheta) |_{lp*}

so aL{ne,te,ti} become differentiable w.r.t. the LCFS (ne, Te, Ti) inputs -- and,
critically, *correlated* with those inputs (shared parents), which is exactly
what the covariance propagation needs.

The mirror of ``boundary._ssf_decay_lengths`` / ``aLy_PeretSSF.solve``; keep the
physics in sync with that file.
"""

import math
from typing import Dict

import numpy as np
import torch

try:
    from scipy.optimize import brentq
except Exception:  # pragma: no cover
    brentq = None


def _residual_np(lp, gamma, rho_s, Lpar, g, Lambda, f_Delta, alpha_s):
    """NumPy residual for the bracketed value solve (matches boundary.py)."""
    lp = max(float(lp), 1e-12 * rho_s)
    sqrt_gamma = math.sqrt(gamma)
    lT = sqrt_gamma / (sqrt_gamma - 1.0) * lp
    beta = f_Delta * (1.0 / Lambda + lp / lT)
    alpha_ExB = -0.43 * beta * Lambda * (rho_s / lp) ** 1.5 / g ** 0.5
    lhs = lp / rho_s
    rhs = (3.9 * g ** (3.0 / 11.0) * (2.0 * rho_s / Lpar) ** (-6.0 / 11.0)
           * gamma ** (-4.0 / 11.0) / (1.0 + (alpha_s + alpha_ExB) ** 2) ** (9.0 / 11.0))
    return lhs - rhs


def _residual_torch(lp, gamma, rho_s, Lpar, g, Lambda, f_Delta, alpha_s):
    """Differentiable residual; ``lp`` and the physics args may carry grad."""
    sqrt_gamma = torch.sqrt(gamma)
    lT = sqrt_gamma / (sqrt_gamma - 1.0) * lp
    beta = f_Delta * (1.0 / Lambda + lp / lT)
    alpha_ExB = -0.43 * beta * Lambda * (rho_s / lp) ** 1.5 / g ** 0.5
    lhs = lp / rho_s
    rhs = (3.9 * g ** (3.0 / 11.0) * (2.0 * rho_s / Lpar) ** (-6.0 / 11.0)
           * gamma ** (-4.0 / 11.0) / (1.0 + (alpha_s + alpha_ExB) ** 2) ** (9.0 / 11.0))
    return lhs - rhs


def _solve_lambda_p_value(gamma, rho_s, Lpar, g, Lambda, f_Delta, alpha_s):
    """Bracketed root of the residual on [rho_s, 1000 rho_s] -> float."""
    args = (float(gamma), float(rho_s), float(Lpar), float(g),
            float(Lambda), float(f_Delta), float(alpha_s))
    grid = float(rho_s) * np.logspace(0.0, 3.0, 64)
    fvals = np.array([_residual_np(x, *args) for x in grid])
    sc = np.where(np.sign(fvals[:-1]) != np.sign(fvals[1:]))[0]
    if sc.size == 0:
        raise RuntimeError("PeretSSF(torch): no bracketed root for lambda_p")
    i = int(sc[0])
    if brentq is not None:
        return float(brentq(lambda x: _residual_np(x, *args),
                            grid[i], grid[i + 1], xtol=1e-6 * float(rho_s), rtol=1e-8))
    # bisection fallback
    lo, hi = grid[i], grid[i + 1]
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _residual_np(lo, *args) * _residual_np(mid, *args) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# Physical floor on g = G0*rho_s/R0. In the small-g regime (small rho_s, or the
# degenerate G0=1 equilibrium fallback) the ExB term alpha_ExB ~ 1/sqrt(g) diverges and
# the SSF residual lp/rho_s - rhs(lp) loses its lambda_p root (rhs < lhs everywhere).
# Clamping g to G_MIN keeps the model in the regime where a physical root exists;
# _g_floor_events logs the raw g each time the floor bites so callers can flag it.
G_MIN = 1.0e-3
_g_floor_events = []


def ssf_decay_lengths_torch(
    *,
    te: torch.Tensor,          # LCFS electron temperature [keV] (differentiable)
    ti: torch.Tensor,          # LCFS ion temperature [keV] (differentiable)
    rho_s_ref: float,          # nominal sound gyroradius rho_s [m] at te_ref
    te_ref: float,             # te at which rho_s_ref was evaluated [keV]
    R0: float, Lpar: float, a: float,
    G0: float, alpha_s: float, Lambda: float, f_Delta: float,
    me_over_mi: float,
) -> Dict[str, torch.Tensor]:
    """
    Differentiable (lambda_p, lambda_n, lambda_T, lambda_q, gamma) and aLy.

    ``rho_s`` scales as sqrt(Te) (its only Te dependence at fixed B), so the LCFS
    Te sensitivity enters both gamma and rho_s.  Density-independent, per Peret
    section 4.2; aLne = a/lambda_n follows the T-driven lengths.

    Returns a dict of 0-d tensors: aLne, aLte, aLti, aLni, lambda_q (and the
    intermediate lengths for inspection), all differentiable w.r.t. te, ti.
    """
    te = te.reshape(())
    ti = ti.reshape(())

    # rho_s(Te) = rho_s_ref * sqrt(Te / Te_ref)
    rho_s = rho_s_ref * torch.sqrt(te / te_ref)

    gamma_0 = 2.5 * ti / te - 0.5 * math.log(2.0 * math.pi * me_over_mi) \
        - 0.5 * torch.log(1.0 + ti / te)
    gamma = 2.0 * gamma_0 / 3.0
    g = G0 * rho_s / R0
    if float(g.detach()) < G_MIN:
        _g_floor_events.append(float(g.detach()))
        g = torch.clamp(g, min=G_MIN)

    # ---- value: bracketed solve on detached scalars ----
    lp_star = _solve_lambda_p_value(
        gamma.detach(), rho_s.detach(), Lpar, g.detach(), Lambda, f_Delta, alpha_s
    )
    lp_star_t = torch.as_tensor(lp_star, dtype=te.dtype, device=te.device)

    # ---- gradient reattach: one Newton step at the root ----
    lp_leaf = lp_star_t.detach().clone().requires_grad_(True)
    r_at = _residual_torch(lp_leaf, gamma.detach(), rho_s.detach(), Lpar,
                           g.detach(), Lambda, f_Delta, alpha_s)
    (drdlp,) = torch.autograd.grad(r_at, lp_leaf, create_graph=False)
    # r evaluated with physics args ATTACHED (carries d/dtheta), lp detached:
    r_theta = _residual_torch(lp_star_t.detach(), gamma, rho_s, Lpar,
                              g, Lambda, f_Delta, alpha_s)
    lambda_p = lp_star_t.detach() - r_theta / drdlp.detach()

    sqrt_gamma = torch.sqrt(gamma)
    lambda_n = sqrt_gamma * lambda_p
    lambda_T = sqrt_gamma / (sqrt_gamma - 1.0) * lambda_p
    lambda_q = (2.0 / 7.0) * lambda_T

    aLne = a / lambda_n.clamp_min(1e-30)
    aLte = a / lambda_T.clamp_min(1e-30)
    aLti = (te / ti.clamp_min(1e-30)) * aLte
    aLni = aLne

    return {
        "aLne": aLne, "aLte": aLte, "aLti": aLti, "aLni": aLni,
        "lambda_p": lambda_p, "lambda_n": lambda_n,
        "lambda_T": lambda_T, "lambda_q": lambda_q, "gamma": gamma,
    }


def state_scalars_from_lcfs(state) -> Dict[str, float]:
    """
    Pull the geometry-fixed scalars PeretSSF needs out of a boundary._LCFSState
    (or any object exposing the same attributes), so the caller can hold Te/Ti as
    the only differentiable inputs.
    """
    return dict(
        rho_s_ref=float(state.rho_s), te_ref=float(state.te),
        R0=float(state.R0), Lpar=float(state.Lpar), a=float(state.a),
        me_over_mi=float(state.me_over_mi),
    )
