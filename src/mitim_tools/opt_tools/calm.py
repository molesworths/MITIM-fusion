"""
calm.py  --  CALM: Confidence-Adaptive Levenberg-Marquardt
==========================================================

A surrogate-assisted nonlinear least-squares solver for PORTALS-edge flux
matching. It pairs per-raw-output GP surrogates (reused from MITIM) with a
*weighted* Levenberg-Marquardt step whose weight matrix W is derived from the
GP posterior variances, and gates real (expensive) evaluations on the
confidence of the proposed step. See CALM_algorithm.md for the spec.

Relationship to the existing solvers
------------------------------------
This is a clean reimplementation that supersedes ``gntrs.py`` (GP-surrogate
Gauss-Newton trust-region). It deliberately reuses the same MITIM data + GP +
residual plumbing that solver relied on -- it is the *solver core* that is new,
not the substrate:

  reused from MITIM (unchanged)
  -----------------------------
  * BOgraphics.optimization_data       -> optimization_data.csv (real evals)
  * SURROGATEtools.surrogate_model     -> one GP per raw output + surrogate_data.csv
  * scalarized_objective               -> residual composition
  * runModelEvaluator_edge             -> the real transport evaluation
  * the simple-relaxation seeding + the MITIM_BO plotting artifacts

  new here (this file)
  --------------------
  * the monotonicity reparametrization transform layer (torch, autodiff)
  * GP-posterior-variance propagation through the affine residual map -> W
  * the surrogate-trust-region loop: each outer iteration (one real eval) FULLY
    solves the cheap GP-surrogate weighted least-squares inside an adaptive
    trust-region box with ``scipy.optimize.least_squares`` (trust-region
    reflective), then evaluates the real model ONCE at the surrogate optimum. The
    measured flux misfit is the arbiter (accept iff it improves) and the trust
    region is resized by the real LM gain ratio ``rho`` -- so model error is
    measured against truth, not estimated from the low-N GP posterior. The fine
    inner FD step resolves the stiff critical-gradient valley, and the zeroth-
    order correction keeps the surrogate interpolating the anchor so ``rho`` is
    calibrated. This replaces a hand-rolled damped-normal-equations LM with
    explicit damping/line-search/gating that stalled ~4x short of the BO optimum.

Honest scope notes (see the review in the conversation / CALM_algorithm.md):
  * The GPs are per *raw output* (Qe_tr_turb_i, Qe_tar_i, ...), not per
    residual. The residual vector is an affine composite of shared raw outputs,
    so W = diag(1/sigma_F^2) is a *diagonal approximation* of Sigma_r (it drops
    the cross-residual covariance from shared raw outputs, e.g. the turbulent-
    exchange term entering Qe and Qi with opposite sign). The empirical
    cross-residual diagnostic ``C`` (CALM init step 2) is computed once after
    seeding so this assumption is measured, not assumed.
  * The Jacobian is computed by central FD over ALL DVs (cheap: 2*n_dv GP/
    parameterizer passes, no transport runs).
"""

import re
import sys
import copy
import shutil
import numpy as np
import pandas as pd
import torch
from scipy.optimize import least_squares

from mitim_tools.opt_tools import SURROGATEtools, BOTORCHtools
from mitim_tools.opt_tools.utils import BOgraphics
from mitim_tools.opt_tools.optimizers import multivariate_tools
from mitim_modules.portals.PORTALSedge import runModelEvaluator_edge
from mitim_tools.misc_tools import IOtools, LOGtools
from mitim_tools.misc_tools.LOGtools import printMsg as print


# =====================================================================
# Transform layer (monotonicity reparametrization)
# =====================================================================
# All torch so the transform Jacobian is exact autograd. The optimizer works in
# x_opt; the physics / GP code works in x_phys.
#
#   monotonicity:  for each (i, j) with x[i] <= x[j], substitute
#                  x[j] = x[i] + softplus(delta_j),  delta_j unbounded below.

def _inv_softplus(y):
    y = torch.clamp(y, min=1e-12)
    return y + torch.log(-torch.expm1(-y))


def forward_transform(x_phys, mono_pairs):
    """x_phys (torch, (d,)) -> x_opt (torch, (d,))."""
    x_opt = x_phys.clone()
    for (i, j) in mono_pairs:
        x_opt[j] = _inv_softplus(x_phys[j] - x_phys[i])
    return x_opt


def inverse_transform(x_opt, mono_pairs):
    """x_opt (torch, (d,)) -> x_phys (torch, (d,)), autograd-friendly."""
    x_phys = x_opt.clone()
    for (i, j) in mono_pairs:
        x_phys = x_phys.clone()
        x_phys[j] = x_phys[i] + torch.nn.functional.softplus(x_opt[j])
    return x_phys


# =====================================================================
# Monotonicity pairs  (consecutive interior knots within each channel)
# =====================================================================

def build_monotonicity_pairs(dv_names):
    """
    Consecutive-index (i_lo, i_hi) pairs within each channel (ordered by radial
    knot position) for the soft-monotonicity constraint x[i_lo] <= x[i_hi].
    """
    patterns = [
        re.compile(r"^(?P<channel>.+)_aLy(?P<position>\d+)$"),
        re.compile(r"^(?P<channel>.+)_d(?P<position>\d+)$"),
        re.compile(r"^aL(?P<channel>.+)_(?P<position>\d+)$"),
    ]
    groups = {}
    for idx, name in enumerate(dv_names):
        sname = str(name)
        for pat in patterns:
            m = pat.match(sname)
            if m:
                groups.setdefault(m.group("channel"), []).append(
                    (int(m.group("position")), idx)
                )
                break
    pairs = []
    for entries in groups.values():
        entries.sort(key=lambda e: e[0])
        for k in range(len(entries) - 1):
            pairs.append((entries[k][1], entries[k + 1][1]))
    return pairs


# =====================================================================
# Global-surrogate pooling  (sibling-rhoCP samples into a turbulent-channel GP)
# =====================================================================

def _global_pool(out, i, train_X, train_Y, train_Yvar, outputs, surrogate_parameters, dfT):
    if (outputs is None) or ("transformationInputs" not in surrogate_parameters):
        return None
    typ = "_".join(out.split("_")[:-1])
    tin = surrogate_parameters["transformationInputs"]
    tout = surrogate_parameters.get("transformationOutputs", None)
    tvars = surrogate_parameters["surrogate_transformation_variables_lasttime"]

    x_keep = torch.from_numpy(train_X).to(dfT)
    X_list, Y_list, Yvar_list = [], [], []
    for j, outj in enumerate(outputs):
        if (j == i) or (outj is None) or ("_".join(outj.split("_")[:-1]) != typ):
            continue
        with torch.no_grad():
            xFit, _ = tin(x_keep, outj, surrogate_parameters, tvars)
            factor = (tout(x_keep, surrogate_parameters, outj).cpu().numpy()
                      if tout is not None else np.ones((train_X.shape[0], 1)))
        X_list.append(xFit.cpu().numpy())
        Y_list.append(train_Y[:, j:j + 1] / factor)
        Yvar_list.append(train_Yvar[:, j:j + 1] / (factor ** 2))

    if not X_list:
        return None
    return (np.concatenate(X_list, axis=0),
            np.concatenate(Y_list, axis=0),
            np.concatenate(Yvar_list, axis=0))


# =====================================================================
# Minimal MITIM_BO "step" stand-in for the plotting shim
# =====================================================================

class _CALMStep:
    """
    Duck-typed replacement for STEPtools.OPTstep, exposing only what the
    PORTALS analyzer / plotter and the MITIM_BO save/read cycle touch.
    """
    def __init__(self, GP, train_X, train_Y, train_Ystd, bounds):
        self.GP = GP
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_Ystd = train_Ystd
        self.train_Yvar = train_Ystd ** 2
        self.x = train_X
        self.bounds = bounds

    def defineFunctions(self, scalarized_objective):
        self.scalarized_objective = scalarized_objective


# =====================================================================
# Driver
# =====================================================================

class CALM:
    """
    Confidence-Adaptive Levenberg-Marquardt for PORTALS-edge flux matching.

    Parameters
    ----------
    portals_fun : portals_edge
        Already ``prep()``-ed: carries ``powerstate``, ``surrogate_parameters``,
        ``optimization_options['problem_options']`` (dvs/ofs/dvs_min/max/base)
        and ``name_transformed_ofs``.
    mono_pairs : list[(i, j)] of DV indices requiring x[i] <= x[j] (default:
        consecutive interior knots per channel).
    options : dict, optional  -- overrides for the CALM knobs below.
    """

    _defaults = dict(
        # --- GP-posterior residual weighting ------------------------------
        stds=2,                 # GP CI half-width in units of sigma (for sigma_y)
        # Cap the spread of the inverse-variance weights W=1/Sigma_F so a few
        # over-confident GP rows cannot dominate the weighted direction at low N
        # (where the GP posterior variances are unreliable). The smallest weights
        # are floored to W.max()/w_max_cond, so max(W)/min(W) <= w_max_cond. Set to
        # None to disable. Default 1e2 keeps genuine credibility differences but
        # kills the runaway spread seen at N~5.
        w_max_cond=1e2,
        rho_lo=0.25,            # TR schedule: rho < rho_lo -> shrink TR
        rho_hi=0.75,            # TR schedule: rho > rho_hi -> grow TR
        # --- overshoot backtrack (rescue a good direction from a single bad end) --------
        # A grossly-negative rho means the FULL inner-solve step rode the collinear driver
        # ridge past a stiff-flux knee (one radius's flux blows up), so the MEASURED misfit
        # jumped even though a PARTIAL move along the same direction improves (verified by
        # the tglfdb joint-flatten check: half the step still lowers the ion residual). The
        # plain reject path only shrinks the TR and re-solves the SAME collinear subproblem
        # from the anchor, so ONE eval-1 overshoot collapses the TR to its floor in ~4 evals
        # and freezes the loop (the H_LowColl/Imode rho=-15..-2e5 stall). When
        # backtrack_on_overshoot is on, an unaccepted rho<0 step instead spends ONE real eval
        # at a fractional step anchor + backtrack_frac*(x_trial-anchor); if that beats the
        # full step it becomes this iteration's move (and the TR adapts on the move actually
        # taken). Bounded by backtrack_budget so a genuine feasibility floor -- where every
        # step overshoots -- does not double the eval budget before the plateau stop fires.
        backtrack_on_overshoot=True,
        backtrack_frac=0.3,     # fractional step length along the rejected direction
        backtrack_budget=3,     # max backtrack evals per run (targets the early freeze)
        # --- adaptive trust region (box around the last real-confirmed point) --
        # Half-width (fraction of dv_scale) that shrinks on a poor/rejected step
        # and grows on a good one, coupling the inner-solve box to MEASURED model
        # quality (the real gain ratio rho). As it shrinks near the solution the
        # surrogate only has to be locally linear over a tiny box, which is what
        # lets a smooth GP still drive the fine endgame steps to 1e-4.
        max_total_rel_step=0.4,  # CEILING on the TR half-width (frac of dv_scale)
        # Initial TR half-width (frac of dv_scale). Kept SMALL: when the seed is
        # already near a stiff critical-gradient minimum, a large first box lets the
        # smooth GP's in-box optimum ride to the box corner (a wild physical point),
        # producing a gross overshoot (rho<<0) on the very first real eval. Starting
        # local keeps the first step inside the GP's trustworthy neighbourhood.
        tr_init=0.01,
        # Floor on the TR half-width. Must be small RELATIVE TO THE OPT-SPACE
        # dv_scale: the softplus-gap parametrization inflates dv_scale (~40/dim
        # here), so tr_min=1e-3 still permits ~4%-of-range steps -- an order of
        # magnitude too coarse for the ~0.2-0.6% steps the 1e-4 valley needs, and
        # the run stalled at exactly that floor. 1e-6 lets the TR reach the
        # descending-side step regime before the stall check fires.
        tr_min=1e-6,
        tr_shrink=0.4,           # multiplicative shrink on reject / low rho
        # FLOOR on the per-step shrink factor. A grossly negative gain ratio means
        # "the surrogate is globally unreliable in this box", NOT "the step was 100x
        # too big" -- the old unbounded 1/(1-rho) hard-shrink turned a single
        # rho=-66 into a 67x shrink, collapsing the TR to its floor in ~4 evals and
        # stalling on an otherwise-good anchor. Bounding the shrink keeps the TR
        # adaptive without euthanizing the run on a transient bad cluster.
        tr_shrink_min=0.2,
        tr_grow=2.0,             # multiplicative grow on a good (rho>rho_hi) step
        # Suppress TR GROWTH until this many real evals exist -- a single lucky
        # accept (rho>rho_hi) on a still-loosely-trained GP must not enlarge the box
        # and license the next, bigger overshoot (the calm eval-2 rho=-65 jump). The
        # box may still SHRINK freely. None => 2x the seed count (so growth is held
        # for roughly one seed-worth of loop evals, then released as the GP matures).
        tr_grow_min_pts=None,
        # Cap the TR box half-width at this multiple of the per-dim training-data
        # std (in opt space). The GP is only trustworthy inside its data hull, so a
        # loosely-trained GP (few, clustered points) cannot propose a step far
        # outside it -- regardless of the inflated softplus-gap dv_scale that makes
        # tr_init/tr_rel a weak handle on the absolute step. As evals concentrate
        # near the solution this envelope shrinks, tightening the endgame for free.
        # None disables (pure tr_rel*dv_scale box).
        tr_data_cap=1.5,
        # On hitting the TR floor with no improvement, reset the TR back to tr_init
        # (forcing a full GP refit) instead of stalling immediately. With the
        # covariance whitening (B) repairing the step direction, the reset gives the
        # solver a genuine second descent attempt rather than giving up at the floor.
        tr_resets=1,
        # --- convergence (on real evaluations only) -----------------------
        res_tol=1e-5,           # converge when PORTALS flux residual 1/N*||cal-of|| < this
        xtol=1e-4,              # ||x_trial - anchor|| step-size convergence
        max_real_evals=25,      # SOFT iteration budget (see windowed extension below)
        # --- input-uncertainty band (edge-UQ two-tier propagation) --------------
        # ANCHOR TIER: when portals_fun._edge_uq_inputs is set, runModelEvaluator_edge
        # runs run_edge_uq at every real eval and leaves the TRUTH residual band
        # (residual-space Sigma_r, target<->transport correlation preserved, plus a
        # chi2 match test) on portals_fun._edge_uq_last. It is NOT injected into the
        # GP training stds (inject_training_stds=False), so the features->flux map
        # stays ~0-noise and the GP stays informative; instead the residual/objective
        # INHERIT the band here:
        #   input_uq_band: quadrature-add the per-(channel,rhoCP) input std (scaled by
        #     the channel scalar_multiplier into CALM residual units) to the GP-
        #     posterior residual variance Sigma_F, so the weighted-LM weight
        #     W=1/Sigma_F (and, with cov_whitening, the step direction) accounts for
        #     the input uncertainty -- WITHOUT it being GP training noise. Held over
        #     the trust region at its anchor value (the GP posterior supplies the
        #     x-varying part; the gp_propagate delta-method is the later refinement).
        #   input_uq_chi2_converge: also declare convergence when run_edge_uq's chi2
        #     test says the real residual is matched to WITHIN the input band (you
        #     cannot flux-match below the input-uncertainty floor). Complements
        #     res_tol (whichever fires first).
        input_uq_band=True,
        input_uq_chi2_converge=True,
        # --- windowed convergence-rate stopping ---------------------------------
        # max_real_evals is a SOFT target, not a hard wall: if the budget is reached
        # while the best objective is still dropping fast, halting throws away the
        # most productive evals (calm2 stopped at the budget right as the post-reset
        # descent got going). The windowed rate is the per-eval geometric relative
        # reduction of the best flux residual over the last conv_window real evals:
        #     rate = 1 - (f_best_now / f_best_{window_ago})**(1/conv_window).
        # When the budget is hit, the run EXTENDS (up to max_real_evals_hard) as long
        # as rate >= conv_rate_tol; otherwise it stops. With early_stop_on_plateau,
        # the run also stops BEFORE the budget once rate falls below conv_rate_tol
        # (a flat window), saving evals on a converged-but-not-at-res_tol plateau.
        conv_window=3,
        conv_rate_tol=0.05,         # 5%/eval best-residual improvement = "still productive"
        max_real_evals_hard=None,   # ceiling for windowed extension (None => 2*max_real_evals)
        # Stop early on a flat window -- but ONLY after the TR resets are exhausted.
        # A pre-reset plateau can precede a breakthrough (calm2: 7 flat rejects, then
        # the reset broke through to ~50%/eval), so an ungated plateau-stop would
        # kill runs the reset would have rescued. Gating on resets-exhausted lets the
        # rescue run first and only cuts losses on a plateau that survives it (the
        # H_LowNu post-reset rho=-1.00 stall). Set tr_resets=0 to gate on the very
        # first plateau, or early_stop_on_plateau=False to only ever stop at budget.
        early_stop_on_plateau=True,
        # --- surrogate-stall guard (residual-gated stationary break) -------------
        # When the inner LM step collapses below xtol, the surrogate is stationary
        # in the TR. That is genuine convergence ONLY if the REAL flux residual is
        # also matched; a stationary surrogate at HIGH residual is a stall (the local
        # GP sees no descent direction -- e.g. a collinear/degenerate neighbourhood),
        # not a solution. With stall_guard on, such a stall does NOT stop the run:
        # first shrink the TR (free retry, tightens the GP linear model), and if the
        # TR is already at floor, spend one real eval at a jittered anchor to break
        # the flat GP neighbourhood (the in-loop analogue of the seed-phase
        # perturbation), then reset the TR. Bounded by tr_resets. Set False to revert
        # to the old behaviour (stationary step => immediate stop).
        stall_guard=True,
        stall_res_tol=None,         # residual below which a stationary step IS convergence (None => res_tol)
        stall_reexplore_frac=0.05,  # interior-DV jitter (frac of dv_scale) for the floor re-explore eval; 0 disables
        # --- misc ----------------------------------------------------------
        train_Ystd_rel=None,
        # --- hybrid seed (seed="hybrid"): multi-start relaxation from LHS anchors ---
        # The pure-relaxation seed follows ONE physical trajectory, so every surrogate
        # is trained near a single basin -> local capture. The hybrid seed instead
        # draws several space-filling anchors in the softplus-gap OPT space (so each
        # is monotonic-feasible by construction, not a rejected raw-aL* corner) and
        # runs a SHORT relaxation march of seed_iters_per_anchor steps from each, so
        # every anchor lands on/near the feasible flux-match manifold. The first
        # anchor is always the physical base x0 (keeps the relaxation basin anchored);
        # the rest are LHS-in-monotonic-space. Total real seed evals == the requested
        # initialization_options['initial_training'] (n_anchors = ceil(total/iters)).
        # iters_per_anchor -> total recovers pure relaxation (single base anchor);
        # iters_per_anchor=1 recovers pure LHS-in-monotonic-space.
        seed_iters_per_anchor=5,   # relaxation march length per LHS anchor
        seed_lhs_seed=0,           # RNG seed for the anchor LHS design (reproducibility)
        # Interior-DV LHS band half-width (frac of the opt-space box each side of the
        # anchor). <1 keeps anchors physical -- the full box (1.0) can integrate an
        # aggressive gradient combo inward to a ~few-eV temperature that crashes NEO.
        seed_interior_lhs_frac=0.5,
        # --- seed-phase interior-knot perturbation -------------------------------
        # Off by default (0.0). When >0, the interior knot components of
        # each relaxation seed AFTER the first are jittered by Gaussian noise of
        # std = seed_interior_perturb_frac * (dvs_max - dvs_min), clipped to bounds.
        # This deliberately breaks the near-collinear aLte/aLti/aLne exploration the
        # plain relaxation produces, so the seeded GPs see independent per-driver
        # variation (needed to resolve the r4 ITG/ETG decoupling). Perturbed seeds
        # are NOT warm-started from the relaxation transport folders (fresh eval).
        seed_interior_perturb_frac=0.0,
        seed_interior_perturb_seed=0,   # RNG seed for reproducibility
        cross_residual_diag=True,  # compute the empirical C diagnostic after seeding
        # --- cross-residual covariance whitening (uses the C diagnostic) ---------
        # The flux residual rows share raw outputs (the turbulent-exchange Qie term
        # enters Qe and Qi with OPPOSITE sign), so the diagonal W = diag(1/sigma_F^2)
        # drops real cross-residual covariance. When the empirical correlation from
        # the C diagnostic exceeds cov_whitening_corr_thresh, replace the diagonal
        # flux weighting with a full whitening L (L^T L = Sigma_flux^{-1}) built from
        # the per-row GP-posterior variances (well-conditioned diagonal scale) and
        # the empirical correlation matrix, shrunk toward I by cov_whitening_shrink
        # for invertibility/conditioning at low N. This re-orients the LM step along
        # the coupled channels instead of fighting them. Disable by setting
        # cov_whitening=False.
        cov_whitening=True,
        cov_whitening_corr_thresh=0.3,  # engage only when max|off-diag corr| exceeds this
        cov_whitening_shrink=0.1,       # shrink corr toward I (Ledoit-Wolf-style)
        # --- parameter-space (DV) collinearity regularization --------------------
        # cov_whitening (above) whitens the OUTPUT/residual coupling, but the DUAL
        # pathology lives in DV space: the interior gradient drivers (aLte/aLti/aLne)
        # collapse onto a near-1D ridge (driver corr 0.89-0.97, Jacobian cond# ~1e3),
        # so the flux Jacobian is rank-deficient in the parameters. An unregularized
        # inner LS step is then non-unique along that ridge and slides to a box corner
        # -- the aLte<->aLti trade and the aLti overshoot. Two complementary fixes:
        #   (1) inner_x_scale='jac': scipy TRF rescales the columns by the Jacobian
        #       norms every iteration (Marquardt scaling), the textbook numerical fix
        #       for a collinear/ill-scaled Jacobian (default was unit x_scale).
        #   (2) an explicit Levenberg-Marquardt ridge toward the anchor, added as
        #       weighted pseudo-residual rows sqrt(lambda)*(x - anchor)/dv_scale over
        #       the INTERIOR DVs. This is a MAP prior (minimum-norm move), not just
        #       solver damping: it makes the inner objective strictly convex so the
        #       step is UNIQUE even where the flux data is uninformative (the near-null
        #       ridge), selecting the smallest DV move that matches the flux -- exactly
        #       the collinear direction the data cannot resolve. lambda is expressed
        #       RELATIVE to the median flux weight (so it scales with the data term
        #       regardless of GP confidence) and adapted by the MEASURED gain ratio rho
        #       in lockstep with the TR: grow on a poor/rejected step, shrink on a good
        #       one (classic LM), so the asymptotic ridge bias -> lm_ridge_min as the
        #       run converges. Set lm_ridge=False to disable.
        lm_ridge=True,
        lm_ridge_rel=1e-2,      # initial ridge weight relative to median flux weight
        lm_ridge_min=1e-4,      # floor on the relative ridge (never fully un-damped)
        lm_ridge_max=1e2,       # ceiling on the relative ridge
        lm_ridge_grow=3.0,      # grow lambda on a poor/rejected step (rho<rho_lo)
        lm_ridge_shrink=0.33,   # shrink lambda on a good step (rho>rho_hi)
        inner_x_scale="jac",    # scipy least_squares column scaling (Marquardt); or 1.0
        # --- inner GP-surrogate NLLS solve (scipy least_squares, TRF) ------
        # Each outer iteration fully solves the CHEAP GP residual inside the
        # trust-region box with scipy's trust-region-reflective LM, then spends ONE
        # real eval at the surrogate optimum. The FD step here is on the smooth GP
        # (no transport runs), so it can be small enough to resolve the stiff
        # critical-gradient valley -- the resolution the old 5%-of-range real-model
        # FD Jacobian could not reach. Tolerances are tight because the subproblem
        # is cheap and we want it converged, not truncated.
        inner_diff_step=1e-4,   # relative FD step for the surrogate Jacobian
        inner_xtol=1e-6,
        inner_ftol=1e-6,
        inner_gtol=1e-6,
        inner_max_nfev=200,     # cap on (cheap) surrogate evals per inner solve
    )

    def __init__(self, portals_fun, mono_pairs=None, options=None):
        self.fun = portals_fun
        self.ps = portals_fun.powerstate
        self.dfT = self.ps.dfT

        po = portals_fun.optimization_options["problem_options"]
        self.dvs = list(po["dvs"])
        self.ofs = list(po["ofs"])
        self.dvs_min = np.asarray(po["dvs_min"], dtype=float)
        self.dvs_max = np.asarray(po["dvs_max"], dtype=float)
        self.dvs_base = np.asarray(po["dvs_base"], dtype=float)

        self.surrogate_parameters = portals_fun.surrogate_parameters
        self.surrogate_options = portals_fun.optimization_options["surrogate_options"]
        self.mono_pairs = build_monotonicity_pairs(self.dvs) if mono_pairs is None else mono_pairs

        self.global_surrogates = bool(self.surrogate_options.get("global_surrogates", False))

        self.folderOutputs = IOtools.expandPath(portals_fun.folder) / "Outputs"
        self.folderOutputs.mkdir(parents=True, exist_ok=True)
        self.folderExec = IOtools.expandPath(portals_fun.folder) / "Execution"
        self.folderExec.mkdir(parents=True, exist_ok=True)

        o = dict(self._defaults)
        o.update(options or {})
        self.options = o

        self.optimization_data = BOgraphics.optimization_data(
            self.dvs, self.ofs,
            file=self.folderOutputs / "optimization_data.csv",
        )

        # accumulated real training set (raw outputs)
        self.train_X = np.empty((0, len(self.dvs)))
        self.train_Y = np.empty((0, len(self.ofs)))
        self.train_Ystd = np.empty((0, len(self.ofs)))

        # GP state
        self.gp_combined = None
        self.gp_individual = None
        self._transition_pos = None
        self._eval_counter = 0

        # Zeroth-order-consistent surrogate correction (the GP outputs are shifted
        # so the model interpolates the current real anchor, making the LM gain
        # ratio rho meaningful).
        self._corr_delta_Y = None
        self._anchor_x_opt = None   # TR center (opt space) = last real-confirmed point
        self._anchor_Y = None
        self._last_eval_Y = None

        # RNG for the surrogate-stall re-explore jitter (reproducible)
        self._stall_rng = np.random.default_rng(
            int(self.options.get("seed_interior_perturb_seed", 0)) + 1)

        # bounds + dv_scale set in run()
        self._lo = self._hi = self._dv_scale = None

        # adaptive trust-region half-width (fraction of dv_scale); shrunk on
        # reject, grown on a good step. Clamped to [tr_min, max_total_rel_step].
        self._tr_rel = float(np.clip(o["tr_init"], o["tr_min"], o["max_total_rel_step"]))

        # adaptive Levenberg-Marquardt ridge weight (relative to the median flux
        # weight); grown on a poor step and shrunk on a good one in lockstep with the
        # TR, to regularize the collinear DV ridge (cov_whitening does the output dual).
        self._lm_ridge_rel = float(np.clip(
            o["lm_ridge_rel"], o["lm_ridge_min"], o["lm_ridge_max"]))

        # interior DV indices: the ridge and Marquardt scaling target the collinear
        # gradient drivers.
        self._interior_idx = np.arange(len(self.dvs), dtype=int)

        # number of flux residual rows (set on first _compose).
        self.n_flux = None

        # empirical flux-row correlation matrix + its max off-diagonal magnitude
        # (set by the cross-residual diagnostic after seeding). Consumed by the
        # covariance-whitening weighting when the coupling is high (option B).
        self._residual_corr = None
        self._residual_corr_maxoff = 0.0

        # Anchor-tier edge-UQ input band (set by _update_input_band after each real
        # eval when portals_fun._edge_uq_inputs is configured): per-flux-row input
        # std in CALM residual units (length n_flux), and the chi2 match summary.
        self._input_band_std = None
        self._input_chi2 = None
        # channel -> scalar_multipliers index (calculate_residuals convention).
        self._scalar_mult_idx = {"te": 0, "ti": 1, "ne": 2, "nZ": 3, "w0": 4}

    # ==================================================================
    # Real model evaluation
    # ==================================================================

    def _evaluate_real(self, x_phys_np):
        n = self._eval_counter
        self._eval_counter += 1

        folder = self.folderExec / f"Evaluation.{n}"
        folder.mkdir(parents=True, exist_ok=True)

        dictDVs = {name: {"value": float(x_phys_np[i])} for i, name in enumerate(self.dvs)}
        dictOFs = {of: {"value": np.nan, "error": np.nan} for of in self.ofs}

        self.fun.powerstate._solver_scratch_folder = folder / "transport_simulation_folder"
        powerstate_result, dictOFs = runModelEvaluator_edge(
            self.fun, folder, dictDVs, f"calm_ev{n}",
            numPORTALS=n, dictOFs=dictOFs,
            remove_folder_upon_completion=not self.fun.portals_parameters["solution"]["keep_full_model_folder"],
        )

        self._save_powerstate_for_plotting(n, powerstate_result)

        Y = np.array([float(dictOFs[of]["value"]) for of in self.ofs])
        Ystd = np.array([float(dictOFs[of]["error"]) for of in self.ofs])
        return Y, Ystd

    def _save_powerstate_for_plotting(self, n, powerstate_result):
        try:
            ps_dir = (IOtools.expandPath(self.fun.folder)
                      / "Initialization" / "initialization_simple_relax"
                      / f"portals_sr_ev_{n}")
            ps_dir.mkdir(parents=True, exist_ok=True)
            powerstate_result.save(ps_dir / "powerstate.pkl")

            prof_dir = self.folderOutputs / "portals_profiles"
            prof_dir.mkdir(parents=True, exist_ok=True)
            powerstate_result.from_powerstate(
                write_input_gacode=prof_dir / f"input.gacode.{n}",
                postprocess_input_gacode=self.fun.portals_parameters["transport"]["applyCorrections"],
            )
        except Exception as e:
            print(f"\t- CALM: could not save powerstate for plotting (eval {n}): {e}",
                  typeMsg="w")

    def _store_real(self, x_phys_np, Y, Ystd):
        self.train_X = np.append(self.train_X, x_phys_np[None, :], axis=0)
        self.train_Y = np.append(self.train_Y, Y[None, :], axis=0)
        self.train_Ystd = np.append(self.train_Ystd, Ystd[None, :], axis=0)

        _, _, R = self._compose(torch.from_numpy(Y).to(self.dfT).unsqueeze(0),
                                torch.from_numpy(x_phys_np).to(self.dfT))
        R = R.detach().cpu().numpy()
        # Reported objective is the PORTALS flux residual -1/n_flux * ||cal - of||,
        # identical to MITIM_BO's definition.
        obj = -self._portals_res(R)
        self.optimization_data.update_points(
            self.train_X, Y=self.train_Y, Ystd=self.train_Ystd,
            objective=np.append(np.full(self.train_X.shape[0] - 1, np.nan), obj),
        )

    def _log_surrogate_predictions(self, eval_idx, pred_mean, pred_std, Y_real):
        f = self.folderOutputs / "surrogate_predictions.csv"
        write_header = not f.exists()
        rows = []
        for j, out in enumerate(self.ofs):
            real = float(Y_real[j])
            pm, ps = float(pred_mean[j]), float(max(pred_std[j], 1e-30))
            pct = 100.0 * (pm - real) / real if abs(real) > 1e-30 else np.nan
            rows.append({
                "eval": int(eval_idx), "output": out,
                "pred_mean": pm, "pred_std": ps, "real": real,
                "pct_error": pct, "z_score": (pm - real) / ps,
            })
        pd.DataFrame(rows).to_csv(f, mode="a", header=write_header, index=False)

    # ==================================================================
    # Residual composition (raw OF vector -> normalized residual vector)
    # ==================================================================

    def _compose(self, Y_row, X_phys):
        """
        Y_row : (1, n_ofs) raw outputs (GP mean OR real eval).
        X_phys: (n_dvs,) physical DVs (kept for signature symmetry).
        Returns (of, cal, R). R is the flux misfit ``cal - of`` -- the SAME residual
        the BO objective uses, taken straight from scalarized_objective (the per-
        channel scalar_multipliers, and the target-normalization for the ne->Ge
        "particle" mode, are already applied there). In the default convective (Ce)
        mode this is the ABSOLUTE MW/m^2 misfit -- we deliberately do NOT divide by
        ``cal`` here, since the absolute total-power residual is the intended (and
        MITIM_BO-comparable) objective. In the ne->Ge mode scalarized_objective has
        already returned target-normalized (dimensionless) of/cal, so ``cal - of``
        inherits that normalization automatically. The PORTALS objective is then
        -1/N * ||cal - of||.

        R is left UNnormalized here (no 1/N); the reported flux objective applies
        the PORTALS 1/n_flux factor in _store_real.
        """
        of, cal, _ = self.fun.scalarized_objective(Y_row)
        source = (cal - of).squeeze(0)                     # (n_flux,) flux misfit
        self.n_flux = int(source.shape[-1])
        R = source
        return of, cal, R

    def _flux_sq(self, R_np):
        """Flux squared residual norm ||cal - of||^2."""
        nf = getattr(self, "n_flux", None) or R_np.shape[-1]
        rf = R_np[:nf]
        return float(rf @ rf)

    def _portals_res(self, R_np):
        """PORTALS flux objective magnitude: (1/n_flux) * ||cal - of||. This is
        the quantity directly comparable to the MITIM_BO residual target."""
        nf = getattr(self, "n_flux", None) or R_np.shape[-1]
        return float(np.linalg.norm(R_np[:nf]) / nf)

    # residual vector from the *surrogate* at x_opt (with the zeroth-order correction)
    def _residual_vector(self, x_opt):
        x_phys = inverse_transform(x_opt, self.mono_pairs)
        y_mean, _, _, _ = self.gp_combined.predict(x_phys.unsqueeze(0))
        if self._corr_delta_Y is not None:
            y_mean = y_mean + self._corr_delta_Y
        _, _, R = self._compose(y_mean, x_phys)
        return R

    def _residuals_np(self, x_opt_np):
        x_opt = torch.from_numpy(x_opt_np).to(self.dfT)
        with torch.no_grad():
            return self._residual_vector(x_opt).detach().cpu().numpy()

    # ------------------------------------------------------------------
    # GP-posterior variance of the residual vector  ->  W = diag(1/sigma_F^2)
    # ------------------------------------------------------------------

    def _residual_variance_np(self, x_opt_np):
        """
        Diagonal posterior variance of each residual element, Sigma_F_diag.

        Flux rows: propagate the per-raw-output GP variance Sigma_y through the
        affine residual map  source = (cal - of)/cal  via its Jacobian
        G = d(source)/d(y_raw):  Var(source_i) = sum_j G[i,j]^2 Sigma_y[j], then
        scaled by 1/N for the R = source/sqrt(N) normalization. This keeps the
        diagonal-only approximation (drops cross-residual covariance from shared
        raw outputs -- see module docstring and the empirical C diagnostic).
        """
        stds = self.options["stds"]
        x_phys = inverse_transform(
            torch.from_numpy(x_opt_np).to(self.dfT), self.mono_pairs)

        with torch.no_grad():
            y_mean, y_upper, y_lower, _ = self.gp_combined.predict(x_phys.unsqueeze(0))
            sigma_y = (y_upper - y_lower).squeeze(0).abs() / (2.0 * stds)
            Sigma_y_diag = (sigma_y ** 2).clamp(min=1e-12)

        # G = d(flux source)/d(y_raw) with source = cal - of (ABSOLUTE, matching
        # PORTALS); affine in y_raw so the Jacobian at the mean is exact.
        def flux_source(y_raw):
            of, cal, _ = self.fun.scalarized_objective(y_raw.unsqueeze(0))
            return (cal - of).squeeze(0)

        y_raw_in = y_mean.squeeze(0).detach().clone()
        _, G = multivariate_tools.mitim_jacobian(flux_source, y_raw_in, vectorize=True)
        G = G.detach()

        R = self._residual_vector(torch.from_numpy(x_opt_np).to(self.dfT)).detach()
        n_total = int(R.shape[-1])
        n_flux = int(G.shape[0])
        self.n_flux = n_flux    # (also set in _compose)

        var_flux = ((G ** 2) @ Sigma_y_diag).clamp(min=1e-12)
        var_flux_np = var_flux.cpu().numpy()

        Sigma_F = np.empty(n_total)
        Sigma_F[:n_flux] = var_flux_np
        # Inherit the anchor-tier input-uncertainty band: quadrature-add the
        # per-flux-row input variance (target<->transport residual band, already in
        # CALM residual units) to the GP-posterior residual variance, so W=1/Sigma_F
        # accounts for input uncertainty without it being GP TRAINING noise. Held at
        # its anchor value over the trust region (GP posterior gives the x-varying
        # part). No-op until _update_input_band populates it (post-seed, edge-UQ on).
        if (self.options.get("input_uq_band", False)
                and self._input_band_std is not None
                and self._input_band_std.shape[0] == n_flux):
            Sigma_F[:n_flux] = var_flux_np + self._input_band_std ** 2
        return Sigma_F

    # ------------------------------------------------------------------
    # Finite-difference Jacobian of the surrogate residual over ALL DVs
    # ------------------------------------------------------------------

    def _clip_weight_spread(self, W):
        """
        Floor the smallest inverse-variance weights to W.max()/w_max_cond so a
        handful of (at low N, unreliable) over-confident GP rows cannot dominate
        the weighted-LM direction. Caps max(W)/min(W) <= w_max_cond; the overall
        scale is irrelevant (lambda absorbs it). Disabled when w_max_cond is None.
        """
        cond = self.options.get("w_max_cond", None)
        if cond is None or not np.isfinite(cond) or cond <= 1.0:
            return W
        return np.maximum(W, W.max() / cond)

    def _cov_whitening_active(self):
        """True when the full cross-residual whitening should replace the diagonal
        flux weighting: enabled, an empirical correlation matrix is available, and
        its max off-diagonal magnitude exceeds the threshold (i.e. the diagonal-W
        approximation is provably dropping real coupling)."""
        if not self.options.get("cov_whitening", False):
            return False
        if self._residual_corr is None:
            return False
        return self._residual_corr_maxoff >= self.options["cov_whitening_corr_thresh"]

    def _flux_whitening(self, w_flux):
        """
        Whitening matrix L for the flux residual block: L^T L = Sigma_flux^{-1}, so
        that minimizing ||L @ R_flux||^2 = R_flux^T Sigma_flux^{-1} R_flux is the
        Mahalanobis (correlated-noise) objective rather than the diagonal one.

        w_flux : (nf,) per-row inverse-variance weights (already spread-clipped) ->
                 the well-conditioned diagonal SCALE d = 1/sqrt(w_flux). The
                 off-diagonal COUPLING comes from the empirical correlation matrix
                 (the C diagnostic), shrunk toward I by cov_whitening_shrink so it is
                 positive-definite/invertible at low N (the empirical corr is rank
                 <= N over nf rows). Returns None on any numerical failure so the
                 caller falls back to the diagonal sqrtW.
        """
        try:
            corr = np.asarray(self._residual_corr, dtype=float)
            nf = corr.shape[0]
            if w_flux.shape[0] != nf:
                return None
            a = float(self.options["cov_whitening_shrink"])
            corr_reg = (1.0 - a) * corr + a * np.eye(nf)
            d = 1.0 / np.sqrt(np.clip(w_flux, 1e-300, None))     # diagonal std scale
            Sigma = (d[:, None] * corr_reg * d[None, :])
            Lc = np.linalg.cholesky(Sigma)                       # Sigma = Lc Lc^T
            L = np.linalg.inv(Lc)                                # L^T L = Sigma^{-1}
            return L
        except Exception as e:
            print(f"\t- CALM: covariance whitening failed ({e}); using diagonal W",
                  typeMsg="w")
            return None

    @staticmethod
    def _apply_weighting(R, sqrtW, L, nf):
        """Weighted residual vector for the inner NLLS: full whitening L on the flux
        block (when available), diagonal sqrtW otherwise."""
        if L is None:
            return sqrtW * R
        out = np.empty_like(R)
        out[:nf] = L @ R[:nf]
        out[nf:] = sqrtW[nf:] * R[nf:]
        return out

    def _ridge_weight(self, W, nf):
        """Absolute per-DV weight sqrt(lambda) for the Levenberg-Marquardt anchor
        ridge, so that the pseudo-residual sqrt(lambda)*(x-anchor)/dv_scale is
        commensurate with the weighted flux rows. lambda = lm_ridge_rel * median flux
        weight ties the ridge to the DATA term (not GP confidence or the arbitrary
        dv_scale). Returns 0.0 when disabled or no flux rows are present."""
        if not self.options.get("lm_ridge", False):
            return 0.0
        w_flux = W[:nf]
        if w_flux.size == 0:
            return 0.0
        return float(np.sqrt(self._lm_ridge_rel * float(np.median(w_flux))))

    def _inner_residual(self, x_opt_np, anchor, sqrtW, L, nf, ridge_w):
        """Inner-NLLS residual: the weighted (whitened) data+prior residual, augmented
        with the parameter-space LM ridge rows sqrt(lambda)*(x-anchor)/dv_scale over
        the interior DVs. scipy's FD Jacobian differentiates the ridge rows for free,
        so no analytic Jacobian is needed; the ridge makes the subproblem strictly
        convex along the collinear driver ridge and yields the minimum-norm step."""
        R = self._apply_weighting(self._residuals_np(x_opt_np), sqrtW, L, nf)
        idx = self._interior_idx
        if ridge_w > 0.0 and idx.size:
            pen = ridge_w * (x_opt_np[idx] - anchor[idx]) / self._dv_scale[idx]
            return np.concatenate([R, pen])
        return R

    # ------------------------------------------------------------------
    # Real evaluation wrapper (logs surrogate prediction vs truth)
    # ------------------------------------------------------------------

    def _real_eval_np(self, x_opt_np):
        x_phys = inverse_transform(
            torch.from_numpy(x_opt_np).to(self.dfT), self.mono_pairs
        ).detach().cpu().numpy()

        pred_mean, pred_std = None, None
        if self.gp_combined is not None:
            with torch.no_grad():
                m, u, l, _ = self.gp_combined.predict(
                    torch.from_numpy(x_phys).to(self.dfT).unsqueeze(0))
            pred_mean = m.squeeze(0).cpu().numpy()
            pred_std = ((u - l).squeeze(0).abs() / 4.0).cpu().numpy()

        Y, Ystd = self._evaluate_real(x_phys)
        self._store_real(x_phys, Y, Ystd)
        self._update_input_band()
        self._last_eval_Y = np.asarray(Y, dtype=float)
        if pred_mean is not None:
            self._log_surrogate_predictions(self._eval_counter - 1, pred_mean, pred_std, Y)

        with torch.no_grad():
            _, _, R = self._compose(torch.from_numpy(Y).to(self.dfT).unsqueeze(0),
                                    torch.from_numpy(x_phys).to(self.dfT))
            R = R.detach().cpu().numpy()
        # Acceptance / convergence are on the FLUX misfit only (partitioned).
        f_flux = self._flux_sq(R)
        return R, f_flux

    def _update_input_band(self):
        """
        Capture the anchor-tier edge-UQ residual band from the last real eval.

        runModelEvaluator_edge leaves run_edge_uq's result on
        ``self.fun._edge_uq_last`` (present only when ``self.fun._edge_uq_inputs``
        is configured and the eval is past the training-phase gate).  We read:

          * ``residual_std`` : per-(channel, rhoCP) input-induced std of the
            residual (target - transport), RAW flux units, order =
            predicted_channels x rhoCP (== calculate_residuals' flux-row order).
            Each channel is scaled by its scalar_multiplier to match CALM's
            residual units and stored as ``self._input_band_std`` (length n_flux).
          * ``converged`` / chi2 : the residual-matched-within-band test
            (self-consistent in run_edge_uq's own raw space, so scale/order
            independent), stored for the chi2 convergence gate.

        Any availability / shape mismatch leaves the band as None -- safe: the
        weighting falls back to GP-posterior only and convergence to res_tol.
        """
        self._input_band_std = None
        self._input_chi2 = None
        last = getattr(self.fun, "_edge_uq_last", None)
        if not isinstance(last, dict):
            return

        if last.get("converged", None) is not None:
            self._input_chi2 = {
                "converged": bool(last["converged"]),
                "stat": float(last.get("chi2_stat", np.nan)),
                "dof": int(last.get("dof", 0)),
                "threshold": float(last.get("chi2_threshold", np.nan)),
            }

        rstd = last.get("residual_std", None)
        if rstd is None:
            return
        try:
            rstd = np.asarray(
                rstd.detach().cpu().numpy() if hasattr(rstd, "detach") else rstd,
                dtype=float).reshape(-1)
            chans = list(self.ps.predicted_channels)
            n_cp = len(self.ps.rhoCP)
            if rstd.shape[0] != len(chans) * n_cp:
                return  # order/length mismatch -> skip (safe)
            sm = self.fun.portals_parameters["solution"].get("scalar_multipliers", None)
            scaled = np.empty_like(rstd)
            for ci, ch in enumerate(chans):
                mult = 1.0
                if sm is not None:
                    idx = self._scalar_mult_idx.get(ch, None)
                    if idx is not None and idx < len(sm):
                        mult = float(sm[idx])
                sl = slice(ci * n_cp, (ci + 1) * n_cp)
                scaled[sl] = np.abs(rstd[sl]) * mult
            self._input_band_std = scaled
        except Exception as e:
            print(f"\t- CALM: input-UQ band capture failed ({e}); GP-posterior W only",
                  typeMsg="w")
            self._input_band_std = None

    # ==================================================================
    # Zeroth-order-consistent surrogate correction + TR anchor
    # ==================================================================

    def _recompute_correction(self):
        if self._anchor_x_opt is None or self.gp_combined is None:
            self._corr_delta_Y = None
            return
        x_phys = inverse_transform(
            torch.from_numpy(self._anchor_x_opt).to(self.dfT), self.mono_pairs)
        with torch.no_grad():
            gp_anchor, _, _, _ = self.gp_combined.predict(x_phys.unsqueeze(0))
        anchor_Y = torch.from_numpy(self._anchor_Y).to(self.dfT).unsqueeze(0)
        self._corr_delta_Y = (anchor_Y - gp_anchor).detach()

    def _set_anchor(self, x_opt_np, Y_real):
        self._anchor_x_opt = np.asarray(x_opt_np, dtype=float).copy()
        self._anchor_Y = np.asarray(Y_real, dtype=float).copy()
        self._recompute_correction()

    # ==================================================================
    # GP fitting (per raw output) with feature-transition-aware refit
    # ==================================================================

    def _current_transition_pos(self, n_points):
        keys = self.surrogate_parameters.get("surrogate_transformation_variables_alltimes", None)
        if not keys:
            return 0
        keys = sorted(int(k) for k in keys.keys())
        for k in keys:
            if n_points < k:
                return k
        return keys[-1]

    def _train_Ystd(self):
        rel = self.options["train_Ystd_rel"]
        if rel is None:
            return self.train_Ystd
        return np.abs(self.train_Y) * rel

    def _fit_surrogates(self, force_full=False):
        n = self.train_X.shape[0]
        if n == 0:
            return
        pos = self._current_transition_pos(n)
        full_fit = (self.gp_combined is None) or (pos != self._transition_pos) or force_full

        if full_fit:
            print(f"\t- CALM: full GP fit (feature set #{pos}, {n} pts)", typeMsg="i")
            self._build_gps(optimize=True)
            self._transition_pos = pos
        else:
            print(f"\t- CALM: warm GP update (reuse hyperparams, {n} pts)", typeMsg="i")
            try:
                self._build_gps(optimize=False)
            except Exception as e:
                print(f"\t- CALM warm update failed ({e}); falling back to full fit",
                      typeMsg="w")
                self._build_gps(optimize=True)

    def _build_gps(self, optimize):
        Ystd = self._train_Ystd()
        Yvar = Ystd ** 2
        prev = self.gp_individual if not optimize else None

        fileTraining = self.folderOutputs / "surrogate_data.csv"
        if optimize:
            backup = fileTraining.with_suffix(".csv.bak")
            if fileTraining.exists():
                fileTraining.replace(backup)

        individual = [None] * len(self.ofs)
        for i, out in enumerate(self.ofs):
            sopt = copy.deepcopy(self.surrogate_options)
            if sopt.get("surrogate_selection", None) is not None:
                sopt = sopt["surrogate_selection"](out, sopt)

            out_t = (self.fun.name_transformed_ofs[i]
                     if getattr(self.fun, "name_transformed_ofs", None) else out)

            extra_added_points = None
            if self.global_surrogates and "_".join(out.split("_")[:-1]).endswith("_tr_turb"):
                extra_added_points = _global_pool(
                    out, i, self.train_X, self.train_Y, Yvar,
                    self.ofs, self.surrogate_parameters, self.dfT,
                )

            GP = SURROGATEtools.surrogate_model(
                self.train_X, self.train_Y[:, i:i + 1], Yvar[:, i:i + 1],
                self.surrogate_parameters,
                output=out, output_transformed=out_t,
                dfT=self.dfT, surrogate_options=sopt,
                fileTraining=fileTraining if optimize else None,
                extra_added_points=extra_added_points,
            )
            if optimize:
                GP.fit()
            else:
                GP.gpmodel.load_state_dict(prev[i].gpmodel.state_dict())
                GP.normalization_pass(
                    GP.gpmodel.input_transform["tf1"], GP.gpmodel.input_transform["tf2"],
                    GP.gpmodel.outcome_transform["tf1"], GP.gpmodel.outcome_transform["tf2"],
                )
                GP.gpmodel.eval(); GP.gpmodel.likelihood.eval()
            individual[i] = GP

        if optimize:
            fileTraining.with_suffix(".csv.bak").unlink(missing_ok=True)

        combined = SURROGATEtools.surrogate_model(
            self.train_X, self.train_Y, Yvar, self.surrogate_parameters,
            dfT=self.dfT, surrogate_options=self.surrogate_options,
        )
        combined.gpmodel = BOTORCHtools.ModifiedModelListGP(*(g.gpmodel for g in individual))

        self.gp_individual = individual
        self.gp_combined = combined

    # ==================================================================
    # CALM weighted-LM step (normal equations + box / soft-monotonicity QP)
    # ==================================================================

    # ==================================================================
    # Empirical cross-residual diagnostic C  (CALM init step 2)
    # ==================================================================

    def _cross_residual_diagnostic(self):
        """
        C = (1/N) sum_n eps_n eps_n^T over GP residual-prediction errors at the
        training points; report the max off-diagonal correlation. A large value
        means the diagonal-W assumption is dropping real residual coupling.
        """
        n = self.train_X.shape[0]
        if n < 2 or self.gp_combined is None:
            return
        try:
            E = []
            for k in range(n):
                x_opt = forward_transform(
                    torch.from_numpy(self.train_X[k]).to(self.dfT), self.mono_pairs)
                with torch.no_grad():
                    mu, _, _, _ = self.gp_combined.predict(
                        inverse_transform(x_opt, self.mono_pairs).unsqueeze(0))
                    _, _, R_pred = self._compose(mu, torch.from_numpy(self.train_X[k]).to(self.dfT))
                    _, _, R_real = self._compose(
                        torch.from_numpy(self.train_Y[k]).to(self.dfT).unsqueeze(0),
                        torch.from_numpy(self.train_X[k]).to(self.dfT))
                E.append((R_real - R_pred).cpu().numpy())
            E = np.asarray(E)                       # (N, n_res)
            # This diagnostic tests the diagonal-W assumption on the GP-predicted
            # flux residual, so restrict it to the flux rows.
            nf = getattr(self, "n_flux", None)
            if nf is not None and E.shape[1] > nf:
                E = E[:, :nf]
            C = (E.T @ E) / E.shape[0]
            d = np.sqrt(np.clip(np.diag(C), 1e-30, None))
            corr = C / np.outer(d, d)
            off = corr - np.diag(np.diag(corr))
            mx = float(np.abs(off).max()) if off.size else 0.0
            # retain for covariance whitening (option B): the empirical flux-row
            # correlation structure is what the diagonal W drops.
            self._residual_corr = corr
            self._residual_corr_maxoff = mx
            verdict = "diagonal W OK" if mx < 0.1 else "WARNING: residual coupling present"
            print(f"\t- CALM cross-residual diagnostic: max|corr_off-diag| = {mx:.3f} "
                  f"({verdict})", typeMsg="i")
            pd.DataFrame(C, index=None).to_csv(
                self.folderOutputs / "cross_residual_covariance.csv", index=False)
        except Exception as e:
            print(f"\t- CALM: cross-residual diagnostic failed: {e}", typeMsg="w")

    # ==================================================================
    # Seeding: simple-relaxation trajectory
    # ==================================================================

    def _relax_from(self, x0, n_iters, subfolder, idx_offset):
        """Run an ``n_iters``-step simple-relaxation march from physical start ``x0``.

        Copies each march point's transport folder to ``Evaluation.{idx_offset+i}``
        (so the subsequent real re-evaluation warm-starts) and returns the
        ``(<=n_iters, d)`` Xopt trajectory.
        """
        init_opts = self.fun.optimization_options.get("initialization_options", {})

        ps = copy.deepcopy(self.fun.powerstate)
        ps.modify(torch.from_numpy(np.asarray(x0, dtype=float)).to(self.dfT).unsqueeze(0))

        MainFolder = IOtools.expandPath(self.fun.folder) / "Initialization" / subfolder
        MainFolder.mkdir(parents=True, exist_ok=True)
        naming = "powerstate_sr_ev"

        solver_options = {
            "tol": None, "maxiter": n_iters, "relax": 0.2, "dx_max": 0.2,
            "relax_dyn": False, "dx_max_abs": None, "dx_min_abs": 0.1,
            "print_each": 1, "folder": MainFolder, "namingConvention": naming,
        }
        solver_options.update(copy.deepcopy(init_opts.get("simple_relax_options", {})))
        solver_options["maxiter"] = n_iters
        solver_options["folder"], solver_options["namingConvention"] = MainFolder, naming

        # Optional seed-phase interior-knot perturbation (collinearity breaking):
        # perturb each calculated next step *inside* the relaxation march, so the
        # march evaluations ARE the perturbed seeds -- no extra evaluations.
        perturb_frac = float(self.options.get("seed_interior_perturb_frac", 0.0))
        if perturb_frac > 0.0:
            solver_options["interior_perturb_frac"] = perturb_frac
            solver_options["interior_perturb_seed"] = int(
                self.options.get("seed_interior_perturb_seed", 0))
            print(f"\t- CALM: seeding march perturbs interior knots "
                  f"(frac={perturb_frac:.3f} of DV range)", typeMsg="i")

        ps.flux_match(algorithm="simple_relax", solver_options=solver_options)
        Xopt = ps.FluxMatch_Xopt.detach().cpu().numpy()

        for i in range(Xopt.shape[0]):
            src = MainFolder / f"{naming}_{i}" / "transport_simulation_folder"
            dst = self.folderExec / f"Evaluation.{idx_offset + i}" / "transport_simulation_folder"
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    IOtools.shutil_rmtree(dst)
                shutil.copytree(src, dst)

        return Xopt

    def _seed_relaxation(self, x0):
        n_seed = int(self.fun.optimization_options
                     .get("initialization_options", {}).get("initial_training", 5))
        return self._relax_from(x0, n_seed, "calm_simple_relax", 0)

    def _lhs_monotonic_starts(self, n, x0):
        """``n`` LHS points drawn in the softplus-gap OPT space (monotonic-feasible by
        construction), returned in PHYSICAL space. All explored columns are banded
        around the anchor start ``x0`` (in opt space) rather than spanning the full box:
          - interior DVs by ``seed_interior_lhs_frac`` (default 0.5). The FULL opt-space
            box lets an aggressive gradient combination integrate inward to a collapsed
            (~few-eV) temperature profile that NEO cannot solve -- the empty
            out.neo.transport_flux -> NEOtools UnboundLocalError seen at
            calm_hybrid_anchor2 (TEMP_1~3e-3 keV). Banding keeps anchors physical while
            still space-filling; set the frac to 1.0 to recover the full-box sweep.
        """
        if n <= 0:
            return []
        lo, hi = self._opt_space_bounds()
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        d = lo.shape[0]
        rng = np.random.default_rng(int(self.options.get("seed_lhs_seed", 0)))
        U = np.zeros((n, d))
        for j in range(d):
            U[:, j] = (rng.permutation(n) + rng.random(n)) / n
        int_frac = float(self.options.get("seed_interior_lhs_frac", 0.5))
        x0 = np.asarray(x0, dtype=float)
        # anchor in opt space -> band centre for each column
        x0_opt = forward_transform(torch.from_numpy(x0).to(self.dfT),
                                   self.mono_pairs).detach().cpu().numpy()
        x0_opt = np.clip(x0_opt, lo, hi)

        def _band(c, frac):
            blo = max(lo[c], x0_opt[c] - frac * (x0_opt[c] - lo[c]))
            bhi = min(hi[c], x0_opt[c] + frac * (hi[c] - x0_opt[c]))
            return blo + U[:, c] * (bhi - blo)

        X = lo[None, :] + U * (hi - lo)[None, :]
        for c, name in enumerate(self.dvs):
            X[:, c] = _band(c, int_frac)               # interior DV: banded (physical)
        phys = np.stack([
            inverse_transform(torch.from_numpy(X[k]).to(self.dfT),
                              self.mono_pairs).detach().cpu().numpy()
            for k in range(n)])
        phys = np.clip(phys, self.dvs_min, self.dvs_max)
        return list(phys)

    def _seed_hybrid(self, x0):
        """Multi-start relaxation seed: base anchor + LHS-in-monotonic-space anchors,
        each given a short ``seed_iters_per_anchor`` relaxation march. Total real evals
        equal ``initial_training``.
        """
        n_total = int(self.fun.optimization_options
                      .get("initialization_options", {}).get("initial_training", 5))
        iters = max(1, int(self.options.get("seed_iters_per_anchor", n_total)))
        n_anchors = max(1, int(np.ceil(n_total / iters)))

        x0 = np.asarray(x0, dtype=float)
        starts = [x0] + self._lhs_monotonic_starts(n_anchors - 1, x0)
        print(f"\t- CALM: hybrid seed -- {n_anchors} anchor(s) x up to {iters} relax "
              f"iters (base + {n_anchors - 1} LHS-in-monotonic-space) => {n_total} evals",
              typeMsg="i")

        seeds, idx = [], 0
        for a, xs in enumerate(starts):
            take = min(iters, n_total - len(seeds))
            if take <= 0:
                break
            march = self._relax_from(xs, take, f"calm_hybrid_anchor{a}", idx)
            seeds.extend(list(march))
            idx += march.shape[0]
        return np.asarray(seeds[:n_total], dtype=float)

    # ==================================================================
    # Public entry point
    # ==================================================================

    def _opt_space_bounds(self):
        """
        Correct opt-space box bounds for the softplus-gap monotonicity transform.

        For a free DV the opt variable IS the physical value, so its bounds are the
        physical [dmin, dmax]. For a monotonic *higher* knot j paired with lower
        knot i, the opt variable is x_opt[j] = inv_softplus(gap), gap = x[j]-x[i].
        The naive forward_transform(dvs_min) / forward_transform(dvs_max) computes
        the opt lower bound from the SAME-SIDE gap dmin[j]-dmin[i] -- i.e. it
        encodes a *minimum gap* equal to the difference of the two knots' physical
        floors. Whenever adjacent knot boxes overlap (always, here) that implied
        floor is spuriously large, so a physically valid seed with a smaller gap is
        OUTSIDE the box and gets clipped -- silently relocating the anchor far from
        the seed (the te_aLy3 15.6 -> 18.2 corruption). The feasible gap range is
            gap_lo = max(0, dmin[j] - dmax[i])   (knots may touch -> 0)
            gap_hi = dmax[j] - dmin[i]
        which is what the box must be built from.
        """
        lo = self.dvs_min.copy()
        hi = self.dvs_max.copy()
        for (i, j) in self.mono_pairs:
            gap_lo = max(0.0, float(self.dvs_min[j] - self.dvs_max[i]))
            gap_hi = max(gap_lo, float(self.dvs_max[j] - self.dvs_min[i]))
            t = torch.from_numpy(np.array([gap_lo, gap_hi], dtype=float)).to(self.dfT)
            inv = _inv_softplus(t).detach().cpu().numpy()
            lo[j], hi[j] = float(inv[0]), float(inv[1])
        return lo, hi

    def run(self, x0=None, seed="relaxation"):
        log_file = self.folderOutputs / "optimization_log.txt"
        prev_stdout = sys.stdout
        sys.stdout = LOGtools.Logger(logFile=log_file, writeAlsoTerminal=True)
        try:
            return self._run_impl(x0=x0, seed=seed)
        finally:
            sys.stdout = prev_stdout

    def _run_impl(self, x0=None, seed="relaxation"):
        x0 = self.dvs_base.copy() if x0 is None else np.asarray(x0, dtype=float)

        # --- seed the surrogate training set with real evaluations ---
        if seed == "relaxation":
            seeds = list(self._seed_relaxation(x0))
        elif seed == "hybrid":
            seeds = list(self._seed_hybrid(x0))
        elif seed is None:
            seeds = [x0]
        else:
            seeds = list(seed)

        seed_f = []
        for xs in seeds:
            xs = np.asarray(xs, dtype=float)
            Y, Ystd = self._evaluate_real(xs)
            self._store_real(xs, Y, Ystd)
            with torch.no_grad():
                _, _, R = self._compose(torch.from_numpy(Y).to(self.dfT).unsqueeze(0),
                                        torch.from_numpy(xs).to(self.dfT))
            seed_f.append(self._flux_sq(R.cpu().numpy()))   # flux-only (partitioned)
        self._fit_surrogates()

        if self.options.get("cross_residual_diag", True):
            self._cross_residual_diagnostic()

        # bounds + characteristic DV scale in optimizer space
        lo, hi = self._opt_space_bounds()
        self._lo, self._hi = np.minimum(lo, hi), np.maximum(lo, hi)
        self._dv_scale = np.maximum(self._hi - self._lo, 1e-6)

        # start from the best seed
        i_best = int(np.argmin(seed_f))
        x_start = np.asarray(seeds[i_best], dtype=float)
        f0 = seed_f[i_best]
        x_cur = forward_transform(torch.from_numpy(x_start).to(self.dfT),
                                  self.mono_pairs).detach().cpu().numpy()
        x_cur = np.clip(x_cur, self._lo, self._hi)
        self._set_anchor(x_cur, self.train_Y[i_best])
        print(f"\t- CALM: starting LM loop from seed {i_best} (f={f0:.3e})", typeMsg="i")

        x_best_opt, f_hist, x_hist = self._calm_loop(x_cur, f0)

        x_best = inverse_transform(
            torch.from_numpy(x_best_opt).to(self.dfT), self.mono_pairs
        ).detach().cpu().numpy()

        try:
            self._write_plotting_artifacts()
        except Exception as e:
            print(f"\t- CALM: could not write plotting artifacts: {e}", typeMsg="w")

        return x_best, f_hist, x_hist

    # ==================================================================
    # CALM main loop
    # ==================================================================

    def _shrink_tr(self):
        self._tr_rel = max(self._tr_rel * self.options["tr_shrink"], self.options["tr_min"])

    def _grow_tr(self):
        self._tr_rel = min(self._tr_rel * self.options["tr_grow"], self.options["max_total_rel_step"])

    def _grow_ridge(self):
        o = self.options
        self._lm_ridge_rel = min(self._lm_ridge_rel * o["lm_ridge_grow"], o["lm_ridge_max"])

    def _shrink_ridge(self):
        o = self.options
        self._lm_ridge_rel = max(self._lm_ridge_rel * o["lm_ridge_shrink"], o["lm_ridge_min"])

    def _reexplore_jitter(self, anchor_opt):
        """Gaussian jitter of the interior DVs about the anchor, in opt space,
        scaled by stall_reexplore_frac * dv_scale and clipped to the box. Used by
        the stall guard to seed a fresh real eval that breaks a flat/degenerate GP
        neighbourhood (in-loop analogue of the seed-phase perturbation).
        """
        frac = float(self.options.get("stall_reexplore_frac", 0.0))
        x = np.asarray(anchor_opt, dtype=float).copy()
        idx = self._interior_idx
        if frac > 0.0 and idx.size:
            x[idx] = x[idx] + self._stall_rng.normal(0.0, 1.0, size=idx.size) \
                * (frac * self._dv_scale[idx])
            x = np.clip(x, self._lo, self._hi)
        return x

    @staticmethod
    def _windowed_rate(best_hist, window):
        """Per-eval geometric relative reduction of the best flux residual over the
        last ``window`` real evals: 1 - (f_now/f_then)^(1/window). Returns +inf until
        there is enough history (so the run is never stopped for lack of a window)."""
        if len(best_hist) <= window:
            return float("inf")
        f_then = best_hist[-window - 1]
        f_now = max(best_hist[-1], 0.0)
        if f_then <= 0.0:
            return 0.0
        return 1.0 - (f_now / f_then) ** (1.0 / window)

    def _data_envelope_half(self):
        """Per-dim half-width (opt space) of the training-data hull, scaled by
        tr_data_cap. Used to cap the TR box so a step never lands far outside the
        region the GP has data in. Returns None when disabled / too few points."""
        cap = self.options.get("tr_data_cap", None)
        if cap is None or self.train_X.shape[0] < 2:
            return None
        with torch.no_grad():
            Xopt = np.stack([
                forward_transform(torch.from_numpy(xp).to(self.dfT),
                                  self.mono_pairs).cpu().numpy()
                for xp in self.train_X])
        return float(cap) * np.maximum(Xopt.std(axis=0), 1e-12)

    def _flux_pred_np(self, x_opt_np):
        """Surrogate-predicted FLUX squared-residual ||cal - of||^2 at x_opt (the
        partitioned data term). Uses the zeroth-order-corrected GP, so at the
        current anchor it returns exactly the last real flux misfit -- which makes
        the surrogate-predicted reduction directly comparable to the measured one
        (the LM gain ratio rho)."""
        R = self._residuals_np(x_opt_np)
        nf = getattr(self, "n_flux", None) or R.shape[-1]
        return float(R[:nf] @ R[:nf])

    def _calm_loop(self, x_cur, f_cur):
        """
        Surrogate-trust-region Levenberg-Marquardt.

        Each outer iteration (1 real eval):
          1. Build inverse-variance weights W from the GP posterior at the anchor
             (fixed for the subproblem) so the weighted NLLS is well-defined.
          2. FULLY solve the CHEAP GP-surrogate weighted residual inside the TR box
             with scipy ``least_squares`` (trust-region reflective). The GP supplies
             the descent direction for free, and the fine inner FD step resolves the
             stiff critical-gradient valley the old 5%-of-range real-model FD could
             not. The zeroth-order correction keeps the surrogate interpolating the
             anchor, so its predicted reduction is calibrated.
          3. Take ONE real eval at the surrogate optimum. The MEASURED flux misfit
             is the arbiter: accept iff it improves, and adapt the TR by the real
             gain ratio rho (model error is measured here, not estimated from the
             low-N GP posterior). Refit the GP -- as real evals concentrate near the
             solution the GP sharpens exactly where the fine endgame steps are taken.
        """
        o = self.options
        self._tr_rel = float(np.clip(o["tr_init"], o["tr_min"], o["max_total_rel_step"]))
        self._lm_ridge_rel = float(np.clip(
            o["lm_ridge_rel"], o["lm_ridge_min"], o["lm_ridge_max"]))
        f_hist = [f_cur]
        x_hist = [x_cur.copy()]
        real_evals = 0
        tr_resets_used = 0
        backtracks_used = 0
        # hold TR growth until the GP has matured past a loosely-trained regime
        grow_min_pts = o["tr_grow_min_pts"] or (2 * self.train_X.shape[0])
        # best-flux-residual-so-far history for the windowed convergence rate
        best_hist = [f_cur]
        hard_cap = o["max_real_evals_hard"] or (2 * o["max_real_evals"])

        while True:
            anchor = self._anchor_x_opt

            # --- inverse-variance weights (fixed over this inner subproblem) ---
            Sigma_F = self._residual_variance_np(anchor)   # also sets self.n_flux
            nf = self.n_flux
            W = 1.0 / np.clip(Sigma_F, 1e-12, None)
            W[:nf] = self._clip_weight_spread(W[:nf])
            sqrtW = np.sqrt(W)

            # full cross-residual whitening on the flux block when the empirical
            # coupling is high (option B); else the diagonal sqrtW. Built once per
            # outer iteration (fixed over the inner subproblem, like W).
            L = self._flux_whitening(W[:nf]) if self._cov_whitening_active() else None

            # parameter-space LM anchor-ridge weight (regularizes the collinear DV
            # ridge; adapted with the TR by the measured rho below). Fixed over the
            # inner subproblem, like W and L.
            ridge_w = self._ridge_weight(W, nf)

            # --- trust-region box around the last real-confirmed anchor ---
            half = self._tr_rel * self._dv_scale
            # cap to the training-data envelope: never step far outside the GP's
            # data hull (loosely-trained GP), independent of the inflated dv_scale.
            env = self._data_envelope_half()
            if env is not None:
                half = np.minimum(half, np.maximum(env, o["tr_min"] * self._dv_scale))
            lo_box = np.maximum(self._lo, anchor - half)
            hi_box = np.minimum(self._hi, anchor + half)
            degenerate = hi_box <= lo_box
            hi_box[degenerate] = lo_box[degenerate] + 1e-12
            x0 = np.clip(anchor, lo_box, hi_box)

            # --- inner: fully solve the cheap GP-surrogate weighted NLLS ---
            sol = least_squares(
                lambda x: self._inner_residual(x, anchor, sqrtW, L, nf, ridge_w), x0,
                jac="2-point", diff_step=o["inner_diff_step"],
                bounds=(lo_box, hi_box), method="trf", x_scale=o["inner_x_scale"],
                xtol=o["inner_xtol"], ftol=o["inner_ftol"], gtol=o["inner_gtol"],
                max_nfev=o["inner_max_nfev"],
            )
            x_trial = np.clip(sol.x, self._lo, self._hi)
            step = float(np.linalg.norm(x_trial - anchor))

            # surrogate-predicted FLUX reduction (partitioned objective)
            pred_red = f_cur - self._flux_pred_np(x_trial)

            if step < o["xtol"]:
                if not o.get("stall_guard", True):
                    print(f"CALM converged (surrogate stationary in TR): |step|={step:.2e}",
                          typeMsg="i")
                    break
                # Residual-gated: a stationary surrogate is convergence ONLY if the
                # real flux residual is matched; otherwise it is a stall to escape.
                res_cur = np.sqrt(max(f_cur, 0.0)) / self.n_flux
                stall_res_tol = o["stall_res_tol"] if o["stall_res_tol"] is not None else o["res_tol"]
                if res_cur < stall_res_tol:
                    print(f"CALM converged (surrogate stationary, res={res_cur:.3e} < "
                          f"{stall_res_tol:.0e}, |step|={step:.2e})", typeMsg="i")
                    break
                # (i) shrink the TR and retry the cheap inner solve (no real eval):
                #     a tighter box restores a non-zero step if the GP has any local
                #     gradient at the anchor.
                if self._tr_rel > o["tr_min"] * (1.0 + 1e-9):
                    self._shrink_tr()
                    print(f"\t- CALM surrogate stall (res={res_cur:.3e} >> "
                          f"{stall_res_tol:.0e}); shrink TR -> {self._tr_rel:.2e}, retry",
                          typeMsg="w")
                    x_cur = anchor.copy(); self._recompute_correction()
                    continue
                # (ii) TR at floor: the GP neighbourhood is flat/degenerate. Spend one
                #      real eval at a jittered anchor to inject new information, reset
                #      the TR, refit, and continue. Bounded by the tr_resets budget.
                if (tr_resets_used < o["tr_resets"]) and (o["stall_reexplore_frac"] > 0.0):
                    tr_resets_used += 1
                    x_jit = self._reexplore_jitter(anchor)
                    _, f_jit = self._real_eval_np(x_jit)
                    real_evals += 1
                    f_hist.append(f_jit); x_hist.append(x_jit.copy())
                    Y_jit = np.asarray(self._last_eval_Y, dtype=float).copy()
                    self._fit_surrogates(force_full=True)
                    self._tr_rel = float(np.clip(o["tr_init"], o["tr_min"],
                                                 o["max_total_rel_step"]))
                    self._lm_ridge_rel = float(np.clip(
                        o["lm_ridge_rel"], o["lm_ridge_min"], o["lm_ridge_max"]))
                    if f_jit < f_cur:
                        x_cur = x_jit; f_cur = f_jit; self._set_anchor(x_jit, Y_jit)
                    else:
                        x_cur = anchor.copy(); self._recompute_correction()
                    best_hist.append(f_cur)
                    print(f"\t- CALM surrogate stall at TR floor: re-explore jitter "
                          f"(real eval {real_evals}, res_jit="
                          f"{np.sqrt(max(f_jit,0.0))/self.n_flux:.3e}); TR reset, "
                          f"resets {tr_resets_used}/{o['tr_resets']}", typeMsg="w")
                    self._checkpoint_plotting_artifacts()
                    continue
                # (iii) escalation budget exhausted: genuinely stuck.
                print(f"CALM stalled: surrogate stationary at TR floor, res={res_cur:.3e} "
                      f"(re-explore budget exhausted); stopping", typeMsg="w")
                break

            # --- one real evaluation at the surrogate optimum ---
            _, f_real = self._real_eval_np(x_trial)
            real_evals += 1
            f_hist.append(f_real); x_hist.append(x_trial.copy())
            best_Y = np.asarray(self._last_eval_Y, dtype=float).copy()

            actual_red = f_cur - f_real
            rho = (actual_red / pred_red) if pred_red > 1e-14 \
                else (1.0 if actual_red > 0 else -1.0)
            accepted = f_real < f_cur

            # --- overshoot backtrack: rescue the productive direction ---------------
            # An unaccepted rho<0 step overshot a stiff-flux knee; a fractional move along
            # the SAME direction often still improves (partial flatten of the responsive
            # channel without riding the collinear ridge to the corner). Spend ONE real eval
            # at the fractional point and adopt it iff it beats the full step, so a good
            # direction is not discarded for overshooting its end. Bounded by backtrack_budget
            # (a true floor overshoots every step -- do not double its budget). Recompute rho
            # and |step| on the adopted partial move so the TR adapts on what was taken.
            if (o.get("backtrack_on_overshoot", True) and (not accepted)
                    and rho < 0.0 and step > o["xtol"]
                    and backtracks_used < o["backtrack_budget"]):
                backtracks_used += 1
                f_full = f_real
                x_bt = np.clip(anchor + o["backtrack_frac"] * (x_trial - anchor),
                               self._lo, self._hi)
                _, f_bt = self._real_eval_np(x_bt)
                real_evals += 1
                f_hist.append(f_bt); x_hist.append(x_bt.copy())
                if f_bt < f_full:
                    x_trial = x_bt
                    best_Y = np.asarray(self._last_eval_Y, dtype=float).copy()
                    f_real = f_bt
                    step = float(np.linalg.norm(x_bt - anchor))
                    actual_red = f_cur - f_bt
                    pred_red_bt = f_cur - self._flux_pred_np(x_bt)
                    rho = (actual_red / pred_red_bt) if pred_red_bt > 1e-14 \
                        else (1.0 if actual_red > 0 else -1.0)
                    accepted = f_bt < f_cur
                    print(f"\t- CALM overshoot backtrack (frac={o['backtrack_frac']:.2f}): "
                          f"full f={f_full:.3e} -> partial f={f_bt:.3e} "
                          f"({'adopted, ACCEPT' if accepted else 'adopted'}); "
                          f"budget {backtracks_used}/{o['backtrack_budget']}", typeMsg="i")
                else:
                    print(f"\t- CALM overshoot backtrack (frac={o['backtrack_frac']:.2f}): "
                          f"partial f={f_bt:.3e} not better than full f={f_full:.3e}; "
                          f"budget {backtracks_used}/{o['backtrack_budget']}", typeMsg="i")

            # adapt the TR on the MEASURED gain ratio (real residual, not GP
            # estimate). A gross overshoot (rho << 0, the GP minimum-in-box landing
            # past the stiff critical-gradient knee) shrinks HARDER than a mild
            # miss, so the TR reaches the descending-side regime in one or two evals
            # instead of crawling down by a fixed factor and burning the budget.
            if (rho < o["rho_lo"]) or (not accepted):
                shrink = o["tr_shrink"]
                if rho < 0.0:
                    # extra shrink for an overshoot, but BOUNDED below by
                    # tr_shrink_min: a single grossly negative rho (globally
                    # unreliable surrogate) must not collapse the TR to its floor in
                    # one step and strand the run on an otherwise-good anchor.
                    shrink = max(min(shrink, 1.0 / (1.0 - rho)), o["tr_shrink_min"])
                self._tr_rel = max(self._tr_rel * shrink, o["tr_min"])
                # grow the LM ridge in lockstep: a poor/rejected step means the step
                # rode the collinear ridge too far, so damp harder toward the anchor.
                self._grow_ridge()
            elif rho > o["rho_hi"]:
                # grow only once the GP is sufficiently trained; a lucky early
                # accept must not enlarge the box for the next (bigger) overshoot.
                if self.train_X.shape[0] >= grow_min_pts:
                    self._grow_tr()
                # a good step -> relax the ridge (reduce the minimum-norm bias) so the
                # endgame is driven by the flux data, not the anchor prior.
                self._shrink_ridge()

            res_real = np.sqrt(max(f_real, 0.0)) / self.n_flux   # PORTALS flux residual
            print(f"\t- CALM eval {real_evals}: res_flux={res_real:.3e} "
                  f"(target<{o['res_tol']:.0e}) rho={rho:+.2f} tr={self._tr_rel:.2e} "
                  f"ridge={self._lm_ridge_rel:.2e} "
                  f"|step|={step:.2e} {'ACCEPT' if accepted else 'reject'}", typeMsg="i")

            # refit GPs (force hyperparam re-opt on reject -- the model mispredicted)
            self._fit_surrogates(force_full=not accepted)

            if accepted:
                x_cur = x_trial
                f_cur = f_real
                self._set_anchor(x_trial, best_Y)
            else:
                # keep the new training point (refit above) but re-ground on the
                # anchor; the shrunk TR makes the next candidate sit closer in.
                x_cur = anchor.copy()
                self._recompute_correction()

            best_hist.append(f_cur)

            # end-of-iteration checkpoint: keep the run plottable if killed mid-loop
            self._checkpoint_plotting_artifacts()

            rate = self._windowed_rate(best_hist, o["conv_window"])

            # --- convergence (real-evaluation history only) ---
            res_cur = np.sqrt(max(f_cur, 0.0)) / self.n_flux
            if res_cur < o["res_tol"]:
                print(f"CALM converged (flux residual): res={res_cur:.3e} < {o['res_tol']:.0e}",
                      typeMsg="i")
                break

            # Chi2 gate: the real residual is matched to WITHIN the propagated input
            # uncertainty (you cannot flux-match below the input-uncertainty floor).
            # Complements res_tol; whichever fires first. run_edge_uq's test is
            # computed in its own self-consistent raw space (scale/order independent).
            if (o.get("input_uq_chi2_converge", False)
                    and accepted and self._input_chi2 is not None
                    and self._input_chi2.get("converged", False)):
                print(f"CALM converged (residual within input-uncertainty band): "
                      f"chi2={self._input_chi2['stat']:.3g} dof={self._input_chi2['dof']} "
                      f"thr={self._input_chi2['threshold']:.3g}, res={res_cur:.3e}",
                      typeMsg="i")
                break

            # windowed budget: max_real_evals is a SOFT target -- keep going past it
            # while the best residual is still dropping fast (up to the hard cap),
            # and (optionally) stop early on a flat window before the budget.
            if real_evals >= o["max_real_evals"]:
                if (real_evals < hard_cap) and np.isfinite(rate) and (rate >= o["conv_rate_tol"]):
                    print(f"\t- CALM: budget {o['max_real_evals']} reached but windowed "
                          f"improvement {rate:.1%}/eval >= {o['conv_rate_tol']:.1%}; "
                          f"extending to {hard_cap}", typeMsg="i")
                else:
                    why = (f"rate {rate:.1%}/eval < {o['conv_rate_tol']:.1%}"
                           if np.isfinite(rate) else "no improvement window")
                    print(f"CALM: reached evaluation budget ({real_evals} evals; {why})",
                          typeMsg="i")
                    break
            elif (o["early_stop_on_plateau"] and (tr_resets_used >= o["tr_resets"])
                  and np.isfinite(rate) and (rate < o["conv_rate_tol"])):
                print(f"CALM stopping (windowed plateau, resets exhausted): improvement "
                      f"{rate:.1%}/eval < {o['conv_rate_tol']:.1%} over last "
                      f"{o['conv_window']} evals", typeMsg="w")
                break

            if (not accepted) and self._tr_rel <= o["tr_min"] * (1.0 + 1e-9):
                if tr_resets_used < o["tr_resets"]:
                    tr_resets_used += 1
                    self._tr_rel = float(np.clip(o["tr_init"], o["tr_min"],
                                                 o["max_total_rel_step"]))
                    self._lm_ridge_rel = float(np.clip(
                        o["lm_ridge_rel"], o["lm_ridge_min"], o["lm_ridge_max"]))
                    print(f"CALM: TR hit floor; reset #{tr_resets_used}/{o['tr_resets']} "
                          f"to tr_init={self._tr_rel:.2e} (full GP refit)", typeMsg="w")
                    self._fit_surrogates(force_full=True)
                    self._recompute_correction()
                else:
                    print(f"CALM stalled (TR at floor {self._tr_rel:.2e}, no improvement, "
                          f"resets exhausted)", typeMsg="w")
                    break

        i_best = int(np.argmin(f_hist))
        return x_hist[i_best], f_hist, x_hist

    # ==================================================================
    # Plotting shim: write optimization_object.pkl / _results.out / _extra.pkl
    # ==================================================================

    def _checkpoint_plotting_artifacts(self):
        """End-of-iteration checkpoint: re-write the MITIM plotting artifacts so a
        killed CALM run stays plottable with mitim_plot_portals_edge. Best-effort --
        a checkpoint failure must never abort the optimization loop, and the
        (expensive) read-back verification is skipped here; it runs on the final
        write in _run_impl instead."""
        try:
            self._write_plotting_artifacts(verify=False)
        except Exception as e:
            print(f"\t- CALM: iteration checkpoint of plotting artifacts failed "
                  f"({type(e).__name__}: {e}); continuing", typeMsg="w")

    def _write_plotting_artifacts(self, verify=True):
        from collections import OrderedDict
        import dill as pickle_dill
        from mitim_tools.opt_tools.STRATEGYtools import MITIM_BO
        from mitim_tools.opt_tools.utils.BOgraphics import optimization_results
        from mitim_modules.powertorch import STATEtools

        n = self.train_X.shape[0]
        if n == 0:
            return

        print("\t- CALM: writing MITIM-compatible plotting artifacts...", typeMsg="i")

        bounds = OrderedDict(
            (dv, [float(self.dvs_min[i]), float(self.dvs_max[i])])
            for i, dv in enumerate(self.dvs)
        )

        m = MITIM_BO(self.fun, onlyInitialize=True, askQuestions=False, cold_start=False)
        m.train_X, m.train_Y, m.train_Ystd = self.train_X, self.train_Y, self.train_Ystd
        m.dfT = self.dfT
        m.cold_start = False
        m.outputs = list(self.ofs)
        m.bounds = bounds
        m.bounds_orig = copy.deepcopy(bounds)
        m.scalarized_objective = self.fun.scalarized_objective
        m.steps = [_CALMStep(
            {"individual_models": self.gp_individual, "combined_model": self.gp_combined},
            self.train_X, self.train_Y, self.train_Ystd, bounds,
        )]

        res_f = []
        for k in range(n):
            with torch.no_grad():
                _, _, R = self._compose(
                    torch.from_numpy(self.train_Y[k]).to(self.dfT).unsqueeze(0),
                    torch.from_numpy(self.train_X[k]).to(self.dfT))
            res_f.append(self._flux_sq(R.cpu().numpy()))   # best = best FLUX match
        ibest = int(np.argmin(res_f))
        # Per-eval maximization objective (negative flux residual). Stored as
        # BOmetrics["overall"]["Residual"] because the MITIM read path runs the
        # PORTALS stopping criteria during optimization_results.read()->getBest:
        # stopping_criteria_default reads Residual[0] and stopping_criteria_by_dvs
        # reads currentIteration. Without these, getBest throws and from_folder
        # silently falls back to the bare PORTALS initializer.
        maxobj = -np.sqrt(np.asarray(res_f, dtype=float)) / self.n_flux
        m.currentIteration = n
        m.BOmetrics = {"overall": {
            "xBest": torch.from_numpy(self.train_X[ibest]).to(self.dfT),
            "indBest": ibest,
            "Residual": torch.from_numpy(maxobj).to(self.dfT),
        }}

        res = optimization_results(file=self.folderOutputs / "optimization_results.out")
        m.optimization_results = res
        res.initialize(m)
        res.OriginalLines = res.lines
        res.addPoints([0, n], executed=True, predicted=False, forceWrite=True)

        # optimization_extra.pkl: per-eval powerstates. PORTALSanalyzer.prep_metrics
        # finds the last iteration by walking the integer keys until it hits the
        # first NON-dict entry (so the iteration count comes from where the dicts
        # stop). We must therefore mirror a real run: contiguous {0..n-1} dict
        # entries followed by a nan sentinel. A missing powerstate is written as
        # nan in place (never left as a gap), so the walk terminates correctly and
        # index 0 is guaranteed present.
        dictStore = {}
        missing = []
        for i in range(n):
            pkl = (IOtools.expandPath(self.fun.folder) / "Initialization"
                   / "initialization_simple_relax" / f"portals_sr_ev_{i}" / "powerstate.pkl")
            if pkl.exists():
                dictStore[i] = {"powerstate": STATEtools.read_saved_state(pkl)}
            else:
                dictStore[i] = np.nan
                missing.append(i)
        dictStore[n] = np.nan   # sentinel: prep_metrics -> ilast = n-1
        if missing:
            print(f"\t- CALM: WARNING missing powerstate pkls for evals {missing}; "
                  f"plotting will truncate at the first gap", typeMsg="w")
        m.optimization_extra = self.folderOutputs / "optimization_extra.pkl"
        self.fun.optimization_extra = m.optimization_extra
        with open(m.optimization_extra, "wb") as h:
            pickle_dill.dump(dictStore, h, protocol=4)

        m.save()

        if verify:
            self._verify_plotting_artifacts()

    def _verify_plotting_artifacts(self):
        """
        Round-trip the artifacts exactly as ``PORTALSanalyzer.from_folder`` does
        (read_from_scratch -> steps[-1] -> optimization_object.surrogate_parameters
        ["powerstate"]) so a broken pickle surfaces as an explicit error here
        instead of a silent fall-back to the bare PORTALS initializer at plot time.
        """
        from mitim_tools.opt_tools.STRATEGYtools import read_from_scratch
        from mitim_tools.opt_tools.utils.BOgraphics import optimization_results
        try:
            mm = read_from_scratch(self.folderOutputs / "optimization_object.pkl")
            _ = mm.steps[-1].GP["combined_model"]
            _ = mm.optimization_object.surrogate_parameters["powerstate"]
            _ = IOtools.unpickle_mitim(mm.optimization_object.optimization_extra)
            # optimization_results.out read + getBest (stopping_criteria runs here
            # and sets best_absolute_index, which PORTALSanalyzer.prep_metrics needs).
            rr = optimization_results(file=self.folderOutputs / "optimization_results.out")
            rr.readClass(mm)
            rr.read()
            if rr.best_absolute_index is None:
                raise RuntimeError("optimization_results.read() could not determine best_absolute_index")
            print("\t- CALM: plotting artifacts verified (mitim_plot_portals_edge ready)",
                  typeMsg="i")
        except Exception as e:
            print(f"\t- CALM: WARNING plotting artifacts failed read-back ({type(e).__name__}: {e}); "
                  f"mitim_plot_portals_edge will fall back to the initializer", typeMsg="w")
