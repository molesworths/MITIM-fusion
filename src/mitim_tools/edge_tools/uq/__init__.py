"""
edge_tools.uq
-------------
Linearized-covariance uncertainty propagation for the PORTALS-edge flux-matching
solver.

Design (see the UQ design sketch):
  * Uncertainty is carried as a *Cholesky factor* ``L`` (a set of perturbation
    columns), NOT per-branch scalar variances, so correlations induced by shared
    parents (profiles feeding both targets and transport) are preserved exactly.
  * The columns are pushed matrix-free through the differentiable slice of
    ``powerstate.calculate()`` with ``torch.func.jvp`` / ``vmap``.  Every module
    that is already torch-differentiable (SplineMtanhAnalytic, PeretSSF, analytic
    rotation, analytic targets) contributes its Jacobian for free.
  * External codes (TGLF, NEO, Aurora, rotation-vgen) never run in-loop: each is
    replaced by a *differentiable surrogate* that supplies its own autograd
    Jacobian and a posterior (epistemic) std ``Sigma_GP`` folded in as extra
    independent columns.

This package supersedes the former ``edge_tools.edge_uq`` (offline-MC,
independent-quadrature combine), which has been removed: covariance factors are
propagated through the real model so cross-point / cross-channel /
target<->transport correlations are preserved instead of quadrature-summed.
"""

from mitim_tools.edge_tools.uq.inputs import UQInputs, InputEntry, ScatterSpec
from mitim_tools.edge_tools.uq.propagate import (
    push_columns,
    cov_from_factor,
    std_from_factor,
    add_epistemic_columns,
    difference_factor,
)
from mitim_tools.edge_tools.uq.driver import UQState, residual_factor
from mitim_tools.edge_tools.uq.peret_torch import ssf_decay_lengths_torch
from mitim_tools.edge_tools.uq.surrogates import (
    read_transport_gp_stds,
    BlackBoxSurrogate,
    finite_difference_jacobian,
)
from mitim_tools.edge_tools.uq.run import run_edge_uq
from mitim_tools.edge_tools.uq.gp_propagate import (
    FEATURE_KEYS_DEFAULT,
    feature_factor_from_L_obs,
    gp_flux_covariance,
    flux_covariance,
    flux_std,
)
from mitim_tools.edge_tools.uq.plotting import (
    plot_breakdown_bars,
    plot_breakdown_heatmap,
)
from mitim_tools.edge_tools.uq.objective import (
    split_stacked,
    residual_from_split,
    inject_stds,
    residual_covariance,
    objective_sigma_delta,
    objective_sigma_sampling,
    robust_objective,
    chi2_convergence,
    column_source_labels,
    variance_breakdown,
    report_source_breakdown,
)

__all__ = [
    "UQInputs",
    "InputEntry",
    "ScatterSpec",
    "push_columns",
    "cov_from_factor",
    "std_from_factor",
    "add_epistemic_columns",
    "difference_factor",
    "UQState",
    "residual_factor",
    "split_stacked",
    "residual_from_split",
    "inject_stds",
    "residual_covariance",
    "objective_sigma_delta",
    "objective_sigma_sampling",
    "robust_objective",
    "chi2_convergence",
    "column_source_labels",
    "variance_breakdown",
    "report_source_breakdown",
    "plot_breakdown_bars",
    "plot_breakdown_heatmap",
    "ssf_decay_lengths_torch",
    "read_transport_gp_stds",
    "BlackBoxSurrogate",
    "finite_difference_jacobian",
    "run_edge_uq",
    "FEATURE_KEYS_DEFAULT",
    "feature_factor_from_L_obs",
    "gp_flux_covariance",
    "flux_covariance",
    "flux_std",
]
