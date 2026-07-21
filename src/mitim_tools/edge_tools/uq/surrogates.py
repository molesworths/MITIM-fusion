"""
edge_tools.uq.surrogates
-------------------------
Differentiable stand-ins for the external codes, and adapters that read epistemic
(surrogate posterior) uncertainty into the ``Sigma_GP`` dicts ``inject_stds``
consumes.

Three cases:

  * TGLF / NEO -- already surrogate-wrapped in the solve; their posterior std is
    written to ``*_tr_turb_stds`` / ``*_tr_neoc_stds`` by TRANSPORTtools.  No new
    model is needed: ``read_transport_gp_stds`` just harvests those into the
    per-channel dicts.

  * Aurora -- graph-breaking (numpy).  ``BlackBoxSurrogate`` wraps a trained
    differentiable surrogate of the *reduced* functionals {qrad, Zeff/dilution}
    as an ``autograd.Function`` with a manual Jacobian.  The impurity balance is
    exactly linear in the source rate for fixed D/V + background, so that Jacobian
    block is analytic and free; D/V and background use the surrogate Jacobian.

  * rotation vgen -- graph-breaking external NEO solve for w0.  Same
    ``BlackBoxSurrogate`` pattern on w0(vtor, profiles); only used when
    rotation_options["model"] == "vgen".

A real surrogate is trained offline from the code evaluations already produced in
the BO loop; here we provide the plumbing (autograd splice + std adapter), not a
fitted model.
"""

from typing import Callable, Dict, Optional

import torch

from mitim_tools.misc_tools.LOGtools import printMsg as print


# --------------------------------------------------------------------------- #
# TGLF / NEO: harvest existing surrogate posterior stds
# --------------------------------------------------------------------------- #

def read_transport_gp_stds(powerstate) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Collect per-channel surrogate posterior stds already present in the
    powerstate into the dicts ``inject_stds`` expects.

    Returns {"turb": {ch: sigma_over_rhoCP}, "neoc": {ch: sigma_over_rhoCP}} where
    each value is a tensor indexed by rhoCP (interpolated from the fine grid).
    Channels without a std field are omitted.
    """
    plasma = powerstate.plasma
    profile_map = powerstate.profile_map
    out = {"turb": {}, "neoc": {}}

    for ch in powerstate.predicted_channels:
        base = profile_map[ch][0]          # flux base, e.g. "QeMWm2"
        for suffix, bucket in (("_tr_turb_stds", "turb"), ("_tr_neoc_stds", "neoc")):
            key = base + suffix
            if key not in plasma:
                continue
            fine = plasma[key]
            if not isinstance(fine, torch.Tensor):
                continue
            # sample fine-grid std at each rhoCP
            sig = _interp_to_rhocp(powerstate, fine)
            if sig is not None:
                out[bucket][ch] = sig
    n = sum(len(v) for v in out.values())
    print(f"[UQ] harvested {n} channel surrogate-std vectors (TGLF/NEO Sigma_GP)",
          typeMsg="i")
    return out


def _interp_to_rhocp(powerstate, fine_tensor) -> Optional[torch.Tensor]:
    """Sample a (batch, n_fine) std tensor at the rhoCP control points -> (n_cp,)."""
    try:
        import numpy as np
        rho_fine = powerstate.plasma["rho"][0].detach().cpu().numpy()
        vals = []
        row = fine_tensor[0] if fine_tensor.dim() >= 2 else fine_tensor
        for cp in range(len(powerstate.rhoCP)):
            idx = int(np.argmin(np.abs(rho_fine - float(powerstate.rhoCP[cp]))))
            vals.append(row[idx])
        return torch.stack(vals)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Black-box surrogate splice (Aurora, vgen)
# --------------------------------------------------------------------------- #

class BlackBoxSurrogate(torch.nn.Module):
    """
    Splice a differentiable surrogate of a graph-breaking code into the autograd
    graph, with an optional analytic Jacobian block for inputs the surrogate
    should not have to learn (e.g. exact source-rate linearity in Aurora).

    Parameters
    ----------
    mean_fn : callable(x) -> y
        Differentiable surrogate mean (a torch module / GP posterior mean).  Its
        autograd Jacobian is used directly for the ``learned`` inputs.
    std_fn : callable(x) -> sigma   (optional)
        Surrogate posterior std (epistemic Sigma_GP), folded in downstream.
    linear_block : dict (optional)
        {"index": i, "coeff": tensor} declaring y is exactly linear in x[i] with
        the given (analytic) coefficient -- overrides the surrogate for that input
        so the exact d y/d x[i] = coeff is used.
    """

    def __init__(self, mean_fn: Callable, std_fn: Optional[Callable] = None,
                 linear_block: Optional[dict] = None):
        super().__init__()
        self.mean_fn = mean_fn
        self.std_fn = std_fn
        self.linear_block = linear_block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mean_fn(x)
        if self.linear_block is not None:
            # Re-express the linear input exactly: replace the surrogate's
            # contribution along that input with the analytic linear term.  The
            # surrogate mean is trusted for the value; the gradient along index i
            # is pinned to the exact coefficient via a straight-through term.
            i = self.linear_block["index"]
            coeff = self.linear_block["coeff"]
            xi = x[..., i]
            y = y + coeff * (xi - xi.detach())   # value unchanged, grad = coeff
        return y

    def epistemic_std(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        return None if self.std_fn is None else self.std_fn(x)


def finite_difference_jacobian(fn: Callable, x0: torch.Tensor, rel_step=1e-3):
    """
    Offline helper: central-difference Jacobian of a numpy-backed code, for
    *building/validating* a surrogate (never called in-loop).  Validate that the
    Jacobian plateaus as the step shrinks -- if it does not, solver noise is
    contaminating it (train a smoothing surrogate instead).
    """
    x0 = x0.reshape(-1)
    n = x0.numel()
    y0 = fn(x0)
    m = y0.numel()
    J = torch.zeros((m, n), dtype=x0.dtype)
    for j in range(n):
        h = rel_step * max(abs(float(x0[j])), 1e-12)
        xp = x0.clone(); xp[j] += h
        xm = x0.clone(); xm[j] -= h
        J[:, j] = (fn(xp) - fn(xm)) / (2 * h)
    return y0, J
