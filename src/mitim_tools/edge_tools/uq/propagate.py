"""
edge_tools.uq.propagate
------------------------
Matrix-free covariance propagation engine.

Uncertainty is represented by a Cholesky factor ``L`` of shape ``(n, k)`` with

    Sigma = L @ L.T

where ``k`` is the number of uncertain degrees of freedom (small: a handful of
LCFS / source / vtor dofs, optionally widened by surrogate epistemic columns).
Each column ``L[:, j]`` is a tangent direction; pushing it through a
differentiable map ``f`` with a Jacobian-vector product gives the corresponding
output column ``J f @ L[:, j]``.  Stacking the columns reconstructs the output
factor ``L_out`` (and hence ``Sigma_out``) without ever materializing a dense
Jacobian.

The same columns can instead be run through the *full* (nonlinear) map as
sigma-points for an unscented estimate at nodes where linearization is poor --
see ``sigma_points`` / ``unscented_moments``.
"""

import torch
from torch.func import jvp, vmap

from mitim_tools.misc_tools.LOGtools import printMsg as print


# --------------------------------------------------------------------------- #
# Linear (delta-method) propagation
# --------------------------------------------------------------------------- #

def push_columns(fwd, x0, L, batched=True):
    """
    Push the columns of a Cholesky factor through a differentiable map.

    Parameters
    ----------
    fwd : callable
        ``fwd(x) -> y`` where ``x`` is a 1-D tensor of shape ``(n,)`` and ``y``
        is a 1-D tensor of shape ``(m,)``.  Must be composed of torch ops so
        ``torch.func.jvp`` can differentiate it (this is the differentiable
        slice of ``powerstate.calculate()``: modify -> SplineMtanhAnalytic ->
        PeretSSF -> analytic rotation/targets, plus surrogate transport).
    x0 : torch.Tensor, shape (n,)
        Nominal input (the mean).
    L : torch.Tensor, shape (n, k)
        Input Cholesky factor (columns are tangent directions).
    batched : bool
        If True, evaluate all ``k`` JVPs in a single vmapped pass (one graph
        traversal).  Set False to loop (useful when ``fwd`` is not vmap-safe,
        e.g. contains an ``autograd.Function`` surrogate that does not batch).

    Returns
    -------
    y0 : torch.Tensor, shape (m,)
        ``fwd(x0)`` (the propagated mean).
    L_out : torch.Tensor, shape (m, k)
        Output Cholesky factor, ``Sigma_out ~= L_out @ L_out.T``.
    """
    if L.ndim != 2:
        raise ValueError(f"L must be 2-D (n, k); got shape {tuple(L.shape)}")

    def single(col):
        y, jvp_col = jvp(fwd, (x0,), (col,))
        return y, jvp_col

    if batched:
        # in_dims=1: map over columns of L; out_dims: y0 shared (None), cols stacked
        y0, L_out = vmap(single, in_dims=1, out_dims=(None, 1))(L)
    else:
        cols = []
        y0 = None
        for j in range(L.shape[1]):
            y0, jvp_col = single(L[:, j])
            cols.append(jvp_col)
        L_out = torch.stack(cols, dim=1)

    return y0, L_out


def cov_from_factor(L_out):
    """Sigma = L_out @ L_out.T  (dense covariance from a Cholesky factor)."""
    return L_out @ L_out.transpose(-1, -2)


def std_from_factor(L_out):
    """Marginal std per output = sqrt(row-wise sum of squares of L_out)."""
    return torch.sqrt((L_out ** 2).sum(dim=-1).clamp_min(0.0))


def difference_factor(L_a, L_b):
    """
    Factor of the covariance of ``a - b`` when ``a`` and ``b`` are propagated
    from the *same* input columns (shared parents).

        (a - b) = (L_a - L_b) xi ,   xi ~ N(0, I)
        Cov(a - b) = (L_a - L_b)(L_a - L_b).T

    This is the whole point of carrying factors rather than variances: the
    target<->transport correlation (both are functions of the same profiles) is
    preserved, including cancellation, instead of being lost to a quadrature sum.
    """
    if L_a.shape != L_b.shape:
        raise ValueError(
            f"factors must share columns to difference correctly: "
            f"{tuple(L_a.shape)} vs {tuple(L_b.shape)}"
        )
    return L_a - L_b


def add_epistemic_columns(L_out, sigma_gp_diag):
    """
    Fold a surrogate posterior std (epistemic, assumed independent of the input
    columns) into the factor as extra diagonal columns.

    Parameters
    ----------
    L_out : torch.Tensor, shape (m, k)
    sigma_gp_diag : torch.Tensor, shape (m,)
        Per-output posterior std from a GP/NN surrogate (Sigma_GP is diagonal in
        the surrogate's output basis; generalize to a block if it is not).

    Returns
    -------
    L_wide : torch.Tensor, shape (m, k + m)
    """
    extra = torch.diag_embed(sigma_gp_diag)  # (m, m)
    return torch.cat([L_out, extra], dim=-1)


# --------------------------------------------------------------------------- #
# Unscented (sigma-point) propagation  -- escalation for nonlinear nodes
# --------------------------------------------------------------------------- #

def sigma_points(x0, L, alpha=1.0):
    """
    Symmetric sigma points x0 +/- alpha * L[:, j] (2k points, no gradient needed).

    Returns a stacked tensor of shape (2k, n).  Run these through the *full*
    (nonlinear) module and recombine with ``unscented_moments``.
    """
    plus = x0.unsqueeze(0) + alpha * L.transpose(0, 1)
    minus = x0.unsqueeze(0) - alpha * L.transpose(0, 1)
    return torch.cat([plus, minus], dim=0)


def unscented_moments(Y, alpha=1.0):
    """
    Mean and Cholesky-like factor from 2k sigma-point outputs ``Y`` (2k, m).

    Equal-weight symmetric scheme: mean is the sample mean, and the factor
    columns are ``(y_plus - y_minus) / (2 alpha)`` so that
    ``Sigma ~= sum_j col_j col_j.T`` matches the linear result to 2nd order.
    """
    k2 = Y.shape[0]
    if k2 % 2 != 0:
        raise ValueError("expected an even number of sigma points (2k)")
    k = k2 // 2
    y_plus, y_minus = Y[:k], Y[k:]
    mean = Y.mean(dim=0)
    L_out = ((y_plus - y_minus) / (2.0 * alpha)).transpose(0, 1)  # (m, k)
    return mean, L_out


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #

def variance_decomposition(g, L_out, input_names=None):
    """
    Delta-method objective variance and its per-column decomposition.

    sigma_J^2 = g.T Sigma_out g = || L_out.T g ||^2 = sum_j (g . L_out[:, j])^2

    Returns (sigma_J2, contributions) where contributions[j] is the variance
    attributable to input column j -- the "which input dominates" ranking, free.
    """
    proj = L_out.transpose(-1, -2) @ g  # (k,)
    contributions = proj ** 2
    sigma_J2 = contributions.sum()
    if input_names is not None and len(input_names) == contributions.shape[-1]:
        order = torch.argsort(contributions, descending=True)
        print("[UQ] objective variance decomposition:", typeMsg="i")
        for idx in order.tolist():
            print(f"      {input_names[idx]:>24s} : {contributions[idx].item():.3e}",
                  typeMsg="i")
    return sigma_J2, contributions
