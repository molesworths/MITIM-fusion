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
Gauss-Newton trust-region) and ``MITIM_wLM`` (STRATEGYtools). It deliberately
reuses the same MITIM data + GP + residual plumbing that both of those relied
on -- it is the *solver core* that is new, not the substrate:

  reused from MITIM (unchanged)
  -----------------------------
  * BOgraphics.optimization_data       -> optimization_data.csv (real evals)
  * SURROGATEtools.surrogate_model     -> one GP per raw output + surrogate_data.csv
  * scalarized_objective / _lcfs_residual_source -> residual composition
  * runModelEvaluator_edge             -> the real transport evaluation
  * the simple-relaxation seeding + the MITIM_BO plotting artifacts

  new here (this file)
  --------------------
  * the monotonicity reparametrization transform layer (torch, autodiff)
  * GP-posterior-variance propagation through the affine residual map -> W
  * the confidence-adaptive LM loop: weighted normal equations with global
    damping ``lambda`` and a per-parameter-class differential damping vector
    ``d``; a risk-penalized acceptance test ``dL - kappa*sigma_dL``; and a
    step-confidence gate ``s2_delta`` that defers real evaluations while the
    surrogate is trustworthy (inner surrogate iterations) and triggers a real
    eval otherwise.

Honest scope notes (see the review in the conversation / CALM_algorithm.md):
  * The GPs are per *raw output* (Qe_tr_turb_i, Qe_tar_i, ...), not per
    residual. The residual vector is an affine composite of shared raw outputs,
    so W = diag(1/sigma_F^2) is a *diagonal approximation* of Sigma_r (it drops
    the cross-residual covariance from shared raw outputs, e.g. the turbulent-
    exchange term entering Qe and Qi with opposite sign). The empirical
    cross-residual diagnostic ``C`` (CALM init step 2) is computed once after
    seeding so this assumption is measured, not assumed.
  * LCFS aL{ch}_lcfs are first-class soft-priored DVs (lcfs_bc_model.md sec 6).
    Their prior mean F_y is retained through the (y-F_y)/sigma_y residual rows
    (in ``_compose``); the differential damping vector ``d`` additionally
    regularizes their columns. The Jacobian is computed by central FD over ALL
    DVs, which sidesteps the numpy-parameterizer autograd sever for the LCFS
    columns (cheap: 2*n_dv GP/parameterizer passes, no transport runs).
"""

import re
import sys
import copy
import shutil
import numpy as np
import pandas as pd
import torch

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
    LCFS boundary DVs (suffix ``_lcfs``) are excluded.
    """
    patterns = [
        re.compile(r"^(?P<channel>.+)_aLy(?P<position>\d+)$"),
        re.compile(r"^(?P<channel>.+)_d(?P<position>\d+)$"),
        re.compile(r"^aL(?P<channel>.+)_(?P<position>\d+)$"),
    ]
    groups = {}
    for idx, name in enumerate(dv_names):
        sname = str(name)
        if sname.endswith("_lcfs"):
            continue
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
        ``optimization_options['problem_options']`` (dvs/ofs/dvs_min/max/base),
        ``name_transformed_ofs`` and ``_lcfs_dv_names``.
    mono_pairs : list[(i, j)] of DV indices requiring x[i] <= x[j] (default:
        consecutive interior knots per channel).
    options : dict, optional  -- overrides for the CALM knobs below.
    """

    _defaults = dict(
        # --- weighted LM step ---------------------------------------------
        lambda_init=1.0,        # initial global damping
        lambda_min=1e-6,        # Gauss-Newton floor
        lambda_max=1e3,         # damping ceiling (force a real eval at/above this)
        lambda_up=3.0,          # multiplicative increase on reject / low rho
        lambda_down=0.4,        # multiplicative decrease on good rho
        d_interior=1.0,         # differential damping for interior knot DVs
        d_lcfs=3.0,             # differential damping for LCFS boundary DVs (extra regularization)
        stds=2,                 # GP CI half-width in units of sigma (for sigma_y)
        # --- acceptance / gating ------------------------------------------
        kappa=1.0,              # conservatism in risk-penalized gain dL - kappa*sigma_dL
        step_conf_rel=0.5,      # surrogate step iff per-component step std < this fraction of step size
        inner_max_surrogate=8,  # cap on consecutive surrogate-only steps between real evals
        rho_lo=0.25,            # LM schedule: rho < rho_lo -> damp more
        rho_hi=0.75,            # LM schedule: rho > rho_hi -> damp less
        # --- trust region (box around the last real-confirmed point) ------
        max_total_rel_step=0.4,  # half-width of the TR box as a fraction of dv_scale
        # --- constrained QP (soft monotonicity) ---------------------------
        rho_base=0.0,           # 0 => box-only (np.linalg.solve + clip); >0 => scipy QP
        x_ref=1.0,
        # --- convergence (on real evaluations only) -----------------------
        ftol_abs=1e-6,
        gtol_rate=1e-5,
        n_rate=4,
        grad_tol=1e-4,          # ||J^T W mu_F|| stationarity
        xtol=1e-6,              # ||delta|| step-size convergence
        max_real_evals=25,
        # --- misc ----------------------------------------------------------
        train_Ystd_rel=None,
        lcfs_seed_frac=0.3,     # LHS band for LCFS DVs during the relaxation seed only
        cross_residual_diag=True,  # compute the empirical C diagnostic after seeding
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

        lcfs_names = getattr(portals_fun, "_lcfs_dv_names", []) or []
        self._lcfs_col_idx = [self.dvs.index(n) for n in lcfs_names if n in self.dvs]

        self.global_surrogates = bool(self.surrogate_options.get("global_surrogates", False))

        self.folderOutputs = IOtools.expandPath(portals_fun.folder) / "Outputs"
        self.folderOutputs.mkdir(parents=True, exist_ok=True)
        self.folderExec = IOtools.expandPath(portals_fun.folder) / "Execution"
        self.folderExec.mkdir(parents=True, exist_ok=True)

        o = dict(self._defaults)
        o.update(options or {})
        self.options = o

        # Differential damping vector d (opt-space, per-DV). LCFS columns get the
        # heavier d_lcfs (CALM step 7 "differential damping").
        self.d_vec = np.full(len(self.dvs), float(o["d_interior"]))
        for j in self._lcfs_col_idx:
            self.d_vec[j] = float(o["d_lcfs"])

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

        # bounds + dv_scale set in run()
        self._lo = self._hi = self._dv_scale = None

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
        obj = -float(np.linalg.norm(R.detach().cpu().numpy()) / np.sqrt(R.numel()))
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
        X_phys: (n_dvs,) physical DVs (for the LCFS soft-prior rows).
        Returns (of, cal, R) where R is the *normalized* residual vector so that
        ||R||^2 matches the PORTALS scalar residual^2 (flux rows + LCFS rows).
        """
        of, cal, _ = self.fun.scalarized_objective(Y_row)
        source = ((cal - of) / cal).squeeze(0)             # (n_flux,) flux rows

        lcfs = self.fun._lcfs_residual_source(X_phys.unsqueeze(0), source.unsqueeze(0))
        if lcfs is not None:
            source = torch.cat((source, lcfs.squeeze(0)), dim=-1)

        R = source / np.sqrt(source.shape[-1])
        return of, cal, R

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

        LCFS rows are NOT GP predictions: R_LCFS = (y - F_y)/sigma_y is a
        deterministic function of the DV y and the fixed prior (F_y, sigma_y), so
        its GP-posterior variance is genuinely zero (1/sigma^2 would diverge). But
        the residual is *already standardized* -- the prior credibility sigma_y is
        baked into the residual value -- so under the prior y ~ N(F_y, sigma_y^2)
        that defines the soft constraint, (y - F_y)/sigma_y has unit variance by
        construction. Its inverse-variance weight is therefore exactly 1 (in
        source units; 1/N after the R = source/sqrt(N) normalization). This is not
        a tuning choice: it is the consistent extension of W to a pre-standardized
        analytic row, the boundary where CALM's GP-posterior-variance definition
        of W stops applying.
        """
        stds = self.options["stds"]
        x_phys = inverse_transform(
            torch.from_numpy(x_opt_np).to(self.dfT), self.mono_pairs)

        with torch.no_grad():
            y_mean, y_upper, y_lower, _ = self.gp_combined.predict(x_phys.unsqueeze(0))
            sigma_y = (y_upper - y_lower).squeeze(0).abs() / (2.0 * stds)
            Sigma_y_diag = (sigma_y ** 2).clamp(min=1e-12)

        # G = d(flux source)/d(y_raw), evaluated at the posterior mean (exact: the
        # map is affine in y_raw).
        def flux_source(y_raw):
            of, cal, _ = self.fun.scalarized_objective(y_raw.unsqueeze(0))
            return ((cal - of) / cal).squeeze(0)

        y_raw_in = y_mean.squeeze(0).detach().clone()
        _, G = multivariate_tools.mitim_jacobian(flux_source, y_raw_in, vectorize=True)
        G = G.detach()

        # current full residual length (flux + LCFS), for the 1/N scaling
        R = self._residual_vector(torch.from_numpy(x_opt_np).to(self.dfT)).detach()
        n_total = int(R.shape[-1])
        n_flux = int(G.shape[0])

        var_flux = ((G ** 2) @ Sigma_y_diag).clamp(min=1e-12) / n_total
        var_flux_np = var_flux.cpu().numpy()

        Sigma_F = np.empty(n_total)
        Sigma_F[:n_flux] = var_flux_np
        # Pre-standardized analytic rows (LCFS soft prior): unit variance in
        # source units -> 1/n_total under the R = source/sqrt(n_total) scaling.
        Sigma_F[n_flux:] = 1.0 / n_total
        return Sigma_F

    # ------------------------------------------------------------------
    # Finite-difference Jacobian of the surrogate residual over ALL DVs
    # ------------------------------------------------------------------

    def _jacobian_np(self, x_opt_np):
        """
        Central-difference Jacobian d(R)/d(x_opt) over all DVs.

        The edge SplineMtanh parameterizer's ``modify()`` is numpy-based, so an
        autograd Jacobian returns exact zeros for the LCFS columns (and detaches
        the interior knot DVs from the GP features). The forward residual runs
        correctly through that numpy reconstruction, so the true Jacobian is
        recovered by central FD. Cost: 2*n_dv cheap GP/parameterizer passes.
        """
        n_dv = x_opt_np.shape[0]
        f0 = self._residuals_np(x_opt_np)
        J = np.zeros((f0.shape[0], n_dv))
        for j in range(n_dv):
            scale = max(abs(float(self.dvs_max[j] - self.dvs_min[j])), 1.0)
            eps = 1e-3 * scale
            xp = x_opt_np.copy(); xp[j] += eps
            xm = x_opt_np.copy(); xm[j] -= eps
            J[:, j] = (self._residuals_np(xp) - self._residuals_np(xm)) / (2.0 * eps)
        return J

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
        self._last_eval_Y = np.asarray(Y, dtype=float)
        if pred_mean is not None:
            self._log_surrogate_predictions(self._eval_counter - 1, pred_mean, pred_std, Y)

        with torch.no_grad():
            _, _, R = self._compose(torch.from_numpy(Y).to(self.dfT).unsqueeze(0),
                                    torch.from_numpy(x_phys).to(self.dfT))
            R = R.detach().cpu().numpy()
        f_real = float(np.dot(R, R))
        return R, f_real

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

    def _solve_step(self, x_opt_np, R, J, W_diag, lam):
        """
        Build and solve the damped weighted normal equations:
            A = J^T W J + lam * diag(d),   b = J^T W R,   delta = -A^{-1} b
        constrained to the box bounds, the TR box around the current real anchor,
        and (optionally, rho_base>0) the soft per-channel monotonicity rows.

        Returns (delta, A, JtWJ, b) with delta already clipped into the feasible
        box. A and JtWJ are returned for the step-confidence sandwich covariance.
        """
        JtW = J.T * W_diag[None, :]
        JtWJ = JtW @ J
        A = JtWJ + lam * np.diag(self.d_vec)
        b = JtW @ R

        # TR box around the last real-confirmed anchor, intersected with bounds.
        anchor = self._anchor_x_opt if self._anchor_x_opt is not None else x_opt_np
        half = self.options["max_total_rel_step"] * self._dv_scale
        lo_box = np.maximum(self._lo, anchor - half)
        hi_box = np.minimum(self._hi, anchor + half)
        lo_box = np.minimum(lo_box, x_opt_np)   # never exclude the current point
        hi_box = np.maximum(hi_box, x_opt_np)

        if self.options["rho_base"] > 0.0 and self.mono_pairs:
            delta = self._solve_step_qp(x_opt_np, A, b, lo_box, hi_box, lam)
        else:
            try:
                delta = np.linalg.solve(A, -b)
            except np.linalg.LinAlgError:
                delta = np.linalg.solve(A + 1e-8 * np.eye(A.shape[0]), -b)
            delta = np.clip(x_opt_np + delta, lo_box, hi_box) - x_opt_np

        return delta, A, JtWJ, b

    def _solve_step_qp(self, x_opt_np, A, b, lo_box, hi_box, lam):
        """Box + soft-monotonicity QP via SciPy trust-constr.
        min 0.5 z'Pz + q'z, z=[delta;s];  P=blkdiag(2A,0), q=[2b; rho]."""
        from scipy.optimize import minimize, LinearConstraint
        n = x_opt_np.shape[0]
        pairs = self.mono_pairs
        m = len(pairs)
        rho_base = self.options["rho_base"]
        x_ref = self.options["x_ref"]
        lambda_max = self.options["lambda_max"]

        P = np.zeros((n + m, n + m))
        P[:n, :n] = 2.0 * A
        rho = np.array([
            rho_base * (x_opt_np[i_lo] / x_ref) * (lambda_max / max(lam, 1e-300))
            for (i_lo, _i_hi) in pairs
        ]) if m else np.zeros(0)
        q = np.concatenate([2.0 * b, rho])

        rows, lb, ub = [], [], []
        box = np.zeros((n, n + m)); box[:, :n] = np.eye(n)
        rows.append(box); lb.append(lo_box - x_opt_np); ub.append(hi_box - x_opt_np)
        if m:
            Am = np.zeros((m, n + m)); lmono = np.empty(m)
            for k, (i_lo, i_hi) in enumerate(pairs):
                Am[k, i_hi] = 1.0; Am[k, i_lo] = -1.0; Am[k, n + k] = 1.0
                lmono[k] = -(x_opt_np[i_hi] - x_opt_np[i_lo])
            rows.append(Am); lb.append(lmono); ub.append(np.full(m, np.inf))
            Sl = np.zeros((m, n + m)); Sl[:, n:] = np.eye(m)
            rows.append(Sl); lb.append(np.zeros(m)); ub.append(np.full(m, np.inf))
        Acon = np.vstack(rows); lvec = np.concatenate(lb); uvec = np.concatenate(ub)

        try:
            res = minimize(
                lambda z: 0.5 * z @ P @ z + q @ z, np.zeros(n + m),
                method="trust-constr", jac=lambda z: P @ z + q, hess=lambda z: P,
                constraints=[LinearConstraint(Acon, lvec, uvec)],
                options={"maxiter": 200, "gtol": 1e-9, "xtol": 1e-10, "verbose": 0},
            )
            return np.clip(x_opt_np + res.x[:n], lo_box, hi_box) - x_opt_np
        except Exception as e:
            print(f"\t- CALM: QP step failed ({e}); damped GN fallback", typeMsg="w")
            delta = np.linalg.solve(A + 1e-8 * np.eye(n), -b)
            return np.clip(x_opt_np + delta, lo_box, hi_box) - x_opt_np

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
            C = (E.T @ E) / E.shape[0]
            d = np.sqrt(np.clip(np.diag(C), 1e-30, None))
            corr = C / np.outer(d, d)
            off = corr - np.diag(np.diag(corr))
            mx = float(np.abs(off).max()) if off.size else 0.0
            verdict = "diagonal W OK" if mx < 0.1 else "WARNING: residual coupling present"
            print(f"\t- CALM cross-residual diagnostic: max|corr_off-diag| = {mx:.3f} "
                  f"({verdict})", typeMsg="i")
            pd.DataFrame(C, index=None).to_csv(
                self.folderOutputs / "cross_residual_covariance.csv", index=False)
        except Exception as e:
            print(f"\t- CALM: cross-residual diagnostic failed: {e}", typeMsg="w")

    # ==================================================================
    # Seeding: simple-relaxation trajectory with LCFS sampling
    # ==================================================================

    def _seed_relaxation(self, x0):
        init_opts = self.fun.optimization_options.get("initialization_options", {})
        n_seed = int(init_opts.get("initial_training", 5))

        ps = copy.deepcopy(self.fun.powerstate)
        ps.modify(torch.from_numpy(x0).to(self.dfT).unsqueeze(0))

        MainFolder = IOtools.expandPath(self.fun.folder) / "Initialization" / "calm_simple_relax"
        MainFolder.mkdir(parents=True, exist_ok=True)
        naming = "powerstate_sr_ev"

        if getattr(ps, "_lcfs_dv_enabled", False) and getattr(ps, "lcfs_dv_channels", None):
            lcfs_idx = [self.dvs.index(f"aL{ch}_lcfs")
                        for ch in ps.lcfs_dv_channels if f"aL{ch}_lcfs" in self.dvs]
            if lcfs_idx and n_seed > 1:
                d = len(lcfs_idx)
                frac = float(self.options.get("lcfs_seed_frac", 0.1))
                base = self.dvs_base[lcfs_idx]
                lo_full, hi_full = self.dvs_min[lcfs_idx], self.dvs_max[lcfs_idx]
                lo = np.maximum(lo_full, base - frac * (base - lo_full))
                hi = np.minimum(hi_full, base + frac * (hi_full - base))
                rng = np.random.default_rng(0)
                M = n_seed - 1
                lhs = np.zeros((M, d))
                for j in range(d):
                    lhs[:, j] = (rng.permutation(M) + rng.random(M)) / M
                sched = np.vstack([base[None, :], lo[None, :] + lhs * (hi - lo)[None, :]])
                ps._lcfs_init_schedule = torch.from_numpy(sched).to(self.dfT)

        solver_options = {
            "tol": None, "maxiter": n_seed, "relax": 0.2, "dx_max": 0.2,
            "relax_dyn": False, "dx_max_abs": None, "dx_min_abs": 0.1,
            "print_each": 1, "folder": MainFolder, "namingConvention": naming,
        }
        solver_options.update(copy.deepcopy(init_opts.get("simple_relax_options", {})))
        solver_options["maxiter"] = n_seed
        solver_options["folder"], solver_options["namingConvention"] = MainFolder, naming

        ps.flux_match(algorithm="simple_relax", solver_options=solver_options)
        Xopt = ps.FluxMatch_Xopt.detach().cpu().numpy()

        for i in range(Xopt.shape[0]):
            src = MainFolder / f"{naming}_{i}" / "transport_simulation_folder"
            dst = self.folderExec / f"Evaluation.{i}" / "transport_simulation_folder"
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    IOtools.shutil_rmtree(dst)
                shutil.copytree(src, dst)

        return Xopt

    # ==================================================================
    # Public entry point
    # ==================================================================

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
            seed_f.append(float(np.dot(R.cpu().numpy(), R.cpu().numpy())))
        self._fit_surrogates()

        if self.options.get("cross_residual_diag", True):
            self._cross_residual_diagnostic()

        # bounds + characteristic DV scale in optimizer space
        lo = forward_transform(torch.from_numpy(self.dvs_min).to(self.dfT),
                               self.mono_pairs).detach().cpu().numpy()
        hi = forward_transform(torch.from_numpy(self.dvs_max).to(self.dfT),
                               self.mono_pairs).detach().cpu().numpy()
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

    def _calm_loop(self, x_cur, f_cur):
        o = self.options
        lam = o["lambda_init"]
        f_hist = [f_cur]
        x_hist = [x_cur.copy()]
        n_surrogate = 0           # consecutive surrogate-only steps since last real eval
        real_evals = 0

        while real_evals < o["max_real_evals"]:
            x_cur = np.clip(x_cur, self._lo, self._hi)

            # --- Step 1-3: surrogate query, W, weighted normal equations ---
            R = self._residuals_np(x_cur)
            J = self._jacobian_np(x_cur)
            Sigma_F = self._residual_variance_np(x_cur)
            W = 1.0 / np.clip(Sigma_F, 1e-12, None)

            delta, A, JtWJ, b = self._solve_step(x_cur, R, J, W, lam)
            nrm_delta = float(np.linalg.norm(delta))

            # weighted-gradient stationarity (convergence diagnostic)
            grad_norm = float(np.linalg.norm(b))

            if nrm_delta < 1e-14:
                print(f"CALM converged (zero step): |grad|={grad_norm:.2e}", typeMsg="i")
                break

            # --- Step 4: step-confidence sandwich covariance ---
            try:
                Ainv = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                Ainv = np.linalg.inv(A + 1e-8 * np.eye(A.shape[0]))
            Sigma_delta = Ainv @ JtWJ @ Ainv
            s2_delta = float(delta @ Sigma_delta @ delta / max(nrm_delta ** 2, 1e-30))

            # --- Step 5: predicted reduction + risk-penalized gain ---
            Rp = R + J @ delta
            dL = float(R @ (W * R) - Rp @ (W * Rp))
            var_dL = float((2.0 * b) @ Sigma_delta @ (2.0 * b))
            sigma_dL = np.sqrt(max(var_dL, 0.0))
            dL_cons = dL - o["kappa"] * sigma_dL

            forced_real = lam >= o["lambda_max"]

            if dL_cons <= 0 and not forced_real:
                lam = min(lam * o["lambda_up"], o["lambda_max"])
                print(f"\t- CALM: reject (dL_cons={dL_cons:+.2e} <= 0), lam->{lam:.2e}",
                      typeMsg="i")
                continue

            # --- Step 6: real-evaluation gating on step confidence ---
            # per-component step std vs per-component step size
            per_comp_std = np.sqrt(max(s2_delta, 0.0))
            per_comp_step = nrm_delta / np.sqrt(delta.shape[0])
            rel_unc = per_comp_std / max(per_comp_step, 1e-30)

            take_surrogate = (
                not forced_real
                and rel_unc < o["step_conf_rel"]
                and n_surrogate < o["inner_max_surrogate"]
            )

            if take_surrogate:
                x_cur = x_cur + delta
                n_surrogate += 1
                print(f"\t- CALM: surrogate step {n_surrogate} "
                      f"(rel_unc={rel_unc:.2f}, |d|={nrm_delta:.2e})", typeMsg="i")
                continue

            # --- Case B: real evaluation ---
            x_cand = np.clip(x_cur + delta, self._lo, self._hi)
            _, f_real = self._real_eval_np(x_cand)
            real_evals += 1
            f_hist.append(f_real); x_hist.append(x_cand.copy())
            n_surrogate = 0

            # --- Step 7: LM gain ratio + lambda schedule ---
            rho = (f_cur - f_real) / dL if dL > 1e-14 else (1.0 if f_real < f_cur else -1.0)
            accepted = rho > 0.0 and f_real < f_cur

            if rho < o["rho_lo"]:
                lam = min(lam * o["lambda_up"], o["lambda_max"])
            elif rho > o["rho_hi"]:
                lam = max(lam * o["lambda_down"], o["lambda_min"])

            print(f"\t- CALM eval {real_evals}: f={f_real:.3e} rho={rho:+.2f} "
                  f"lam={lam:.2e} rel_unc={rel_unc:.2f} "
                  f"{'ACCEPT' if accepted else 'reject'}", typeMsg="i")

            # refit GPs (force hyperparam re-opt on reject -- the model mispredicted)
            self._fit_surrogates(force_full=not accepted)

            if accepted:
                x_cur = x_cand
                f_cur = f_real
                self._set_anchor(x_cand, self._last_eval_Y)
            else:
                # revert to the anchor; keep the new training point + refit
                x_cur = self._anchor_x_opt.copy()
                self._recompute_correction()

            # --- Step 8: convergence (real-evaluation history only) ---
            if f_cur < o["ftol_abs"]:
                print(f"CALM converged (absolute): f={f_cur:.3e}", typeMsg="i")
                break
            if grad_norm < o["grad_tol"] and nrm_delta < o["xtol"]:
                print(f"CALM converged (stationary): |grad|={grad_norm:.2e}", typeMsg="i")
                break
            if len(f_hist) >= o["n_rate"]:
                imp = (min(f_hist[:-o["n_rate"] + 1]) - min(f_hist[-o["n_rate"]:])) \
                    / (abs(f_hist[0]) + 1e-14)
                if imp < o["gtol_rate"] and lam >= o["lambda_max"]:
                    print(f"CALM converged (rate, damped out): {imp:.3e}", typeMsg="i")
                    break

        i_best = int(np.argmin(f_hist))
        return x_hist[i_best], f_hist, x_hist

    # ==================================================================
    # Plotting shim: write optimization_object.pkl / _results.out / _extra.pkl
    # ==================================================================

    def _write_plotting_artifacts(self):
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
            res_f.append(float(np.dot(R.cpu().numpy(), R.cpu().numpy())))
        ibest = int(np.argmin(res_f))
        m.BOmetrics = {"overall": {
            "xBest": torch.from_numpy(self.train_X[ibest]).to(self.dfT),
            "indBest": ibest,
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
