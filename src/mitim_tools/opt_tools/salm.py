"""
salm.py  --  SALM: Surrogate-Accelerated Levenberg-Marquardt
============================================================

A nonlinear least-squares flux-matcher for PORTALS-edge: a Levenberg-Marquardt
solve of the flux residual, accelerated by a per-raw-output GP surrogate whose
cheap in-box optimum is confirmed by ONE real transport evaluation per outer
iteration, all inside an adaptive trust region. (The Levenberg damping IS the
trust-region constraint; the surrogate is what cuts the number of expensive real
evals -- hence "surrogate-accelerated".)

Seed options (run(seed=...))
----------------------------
  * "relaxation" -- a simple-relaxation march from x0. Good for a FAR start (rapid
    approach toward the basin); it overshoots a near-basin start and scatters the
    seeds to worse points, so prefer "lhs" when x0 is already near the solution.
  * "lhs" -- base x0 (unperturbed) + a banded Latin-hypercube design over a relative
    bounds fraction (seed_lhs_frac). Every seed stays in-basin, so the GP gets a
    clean LOCAL model at x0; the loop then TRAVELS to the solution via boundary-
    limited trust-region growth. Best for a near-basin start.

Algorithm (one real eval per outer iteration)
---------------------------------------------
  1. seed (relaxation or lhs) -> a handful of real evals
  2. monotonicity reparametrization (softplus-gap) so knot ordering is feasible
  3. per-raw-output GP fit + a zeroth-order correction, so the LM gain ratio rho is
     calibrated against the current real anchor
  4. residual composition: the absolute flux misfit (cal - of), the data term
  5. inner scipy ``least_squares`` (TRF) FULLY solving the cheap GP residual inside
     an adaptive trust-region box, with a fixed Levenberg ridge toward the anchor
  6. ONE real eval at the surrogate optimum; accept iff the measured flux misfit
     improves; adapt the TR by the measured gain ratio rho (boundary-limited growth
     enlarges the box when a step is pressed against the TR edge)
  7. one trust-region reset / escape probe on a floor-stall or plateau
  8. convergence on res_tol / xtol / a hard eval budget / a diminishing-returns
     plateau (an escape probe fires before the plateau stop is honored)
  9. the MITIM plotting-artifact shim (needed for mitim_plot_portals_edge)

Scope (kept lean by design)
---------------------------
The GP-posterior variance sets the residual weighting; a single FIXED Levenberg
ridge toward the anchor (not adapted), together with the TRF Marquardt column
scaling, keeps the collinear-driver inner subproblem uniquely solvable at
negligible cost. Deliberately NOT included: cross-residual covariance whitening,
per-eval input-UQ band + chi2 gating (the edge-UQ band is instead run ONCE on the
best evaluation, edge_uq_final_only), global surrogate pooling, and seed-
perturbation jitter (measured ineffective -- it scattered seeds off-manifold and
degraded the local GP).
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
# Transform layer (monotonicity reparametrization; torch autograd-exact)
# =====================================================================

def _inv_softplus(y):
    y = torch.clamp(y, min=1e-12)
    return y + torch.log(-torch.expm1(-y))


# Both transforms index the LAST axis, so they accept a single DV vector (n,) or a whole
# batch (B, n) unchanged -- the batched form is what lets one posterior call cover a full
# finite-difference stencil (see _inner_jac).

def forward_transform(x_phys, mono_pairs):
    x_opt = x_phys.clone()
    for (i, j) in mono_pairs:
        x_opt[..., j] = _inv_softplus(x_phys[..., j] - x_phys[..., i])
    return x_opt


def inverse_transform(x_opt, mono_pairs):
    x_phys = x_opt.clone()
    for (i, j) in mono_pairs:
        x_phys = x_phys.clone()
        x_phys[..., j] = x_phys[..., i] + torch.nn.functional.softplus(x_opt[..., j])
    return x_phys


def build_monotonicity_pairs(dv_names):
    """Consecutive-index (i_lo, i_hi) pairs within each channel for the soft
    x[i_lo] <= x[i_hi] constraint."""
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
                    (int(m.group("position")), idx))
                break
    pairs = []
    for entries in groups.values():
        entries.sort(key=lambda e: e[0])
        for k in range(len(entries) - 1):
            pairs.append((entries[k][1], entries[k + 1][1]))
    return pairs


# =====================================================================
# Minimal MITIM_BO "step" stand-in for the plotting shim
# =====================================================================

class _SALMStep:
    """Duck-typed replacement for STEPtools.OPTstep, exposing only what the
    PORTALS analyzer / plotter and the MITIM_BO save/read cycle touch."""
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

class SALM:
    """Surrogate-Accelerated Levenberg-Marquardt for PORTALS-edge flux matching.
    See the module docstring for the algorithm and seed options."""

    _defaults = dict(
        # GP-posterior residual weighting
        stds=2,                  # GP CI half-width in units of sigma (for sigma_y)
        # Force gradients-only GP features (bypass the nuei/tite/beta_e transition
        # schedule) for the whole run. Robust to global_surrogates count-inflation, which
        # otherwise triggers the full high-dim (collinear -> flat surrogate) set at eval 1
        # regardless of the namelist thresholds. See __init__.
        gradients_only_features=False,
        # Run the (expensive) edge-UQ input-band propagation ONCE on the best evaluation
        # at the end of the run, instead of at every loop eval. The per-eval real_scan
        # re-runs ~1 transport per uncertain input (~16 TGLF scans/eval, ~84% of the TGLF
        # cost in the ch5 logs) and SALM consumes none of it mid-run -- the band
        # is only needed on the SOLUTION. Suppresses fun._edge_uq_inputs during the loop
        # (restored on exit) and leaves the final result on fun._edge_uq_last /
        # _edge_uq_history for downstream analysis. False reverts to per-eval UQ.
        edge_uq_final_only=True,
        # trust region (fraction of opt-space dv_scale)
        # NB tr_init MUST stay above tr_min: run() sets the live radius with
        # np.clip(tr_init, tr_min, max_total_rel_step), so any tr_init <= tr_min is
        # silently swallowed and the TR starts pinned AT its floor. That also makes the
        # "TR floor stall" reset (which restores tr_init) a no-op on the radius -- the
        # run can then only ratchet between tr_min and a couple of doublings and every
        # step comes out boundary-limited (step_frac ~ 1). The old default 5e-3 < tr_min
        # did exactly this; every working driver was passing tr_init=5e-2 by hand.
        tr_init=5e-2,
        tr_min=1e-2,
        max_total_rel_step=0.4,
        tr_shrink=0.4,
        tr_grow=2.0,
        rho_lo=0.25,
        rho_hi=0.75,
        # Boundary-limited TR growth. Classic TR grows only on rho>rho_hi, so a run
        # whose gain ratio settles in the neutral band [rho_lo, rho_hi] with steps
        # pressed against the TR edge FREEZES the box tiny and crawls (the
        # micro-step stall: every step accepted, rho~0.4-0.6, |step|~=box radius, TR
        # frozen tiny, profiles barely move). When an ACCEPTED step is boundary-
        # limited (max per-dim |step|/half >= this) and rho>rho_lo, grow anyway -- the
        # optimizer is capped and still improving. Shrink still wins on rho<rho_lo /
        # reject, so an over-predicting step is never enlarged. 1.0 disables (revert to
        # rho_hi-only growth).
        tr_grow_boundary_frac=0.9,
        tr_resets=1,             # the one loop-side rescue kept (proven useful)
        tr_base_relative=True,   # TR step ~ tr_rel * |base DV| (opt space), box-independent.
        tr_scale_floor=0.1,      # floor on the per-DV base scale (guards near-zero base DVs)
        seed_lhs_base_relative=True,  # LHS seed band ~ frac * |base DV|, box-independent
        # FIXED Levenberg ridge toward the anchor over interior DVs (uniqueness on
        # the collinear driver ridge); not adapted. 0 disables.
        lm_ridge_rel=1e-2,
        inner_x_scale="jac",     # TRF Marquardt column scaling
        # convergence (on real evaluations only)
        res_tol=1e-5,
        xtol=1e-4,
        # DV-based convergence: stop once the design vector stops moving, i.e. an
        # ACCEPTED step changes x by less than this RELATIVE L2 fraction
        # (|x_new - x_old| / |x_old|). 0.01 = the "converged to 1% in the DVs" rule.
        #
        # This is deliberately NOT `xtol`. `xtol` is an ABSOLUTE step threshold that
        # only gates the surrogate-stationary branch below, where convergence is
        # additionally conditional on res_cur < res_tol; when the residual is above
        # res_tol (always, in practice -- res_tol=1e-5 is never reached on these edge
        # cases) that branch treats a small step as a STALL and shrinks the TR. So
        # raising `xtol` makes runs stall EARLIER rather than declaring convergence.
        # xtol_rel gives the DV test its own exit that does not consult the residual.
        # 0.0 disables (previous behaviour).
        xtol_rel=0.0,
        max_real_evals=25,
        # Diminishing-returns early exit: stop when an ACCEPTED step improves the best
        # flux_sq by less than plateau_rel_tol per eval (geometric mean) over the last
        # plateau_window evals -- the accepted micro-step tail that otherwise crawls to the
        # budget. Evaluated only on accepts, so reject-recovery is not cut short.
        # plateau_window=0 disables. Note this is on flux_sq (~2x the rate on res).
        plateau_window=3,
        plateau_rel_tol=0.02,   # <1%/eval improvement in flux_sq over the window => stop
        # After seeding, delete the transient simple-relax march folder
        # (Initialization/salm_simple_relax). Its per-point transport_simulation_folder
        # is byte-copied into Execution/Evaluation.{i} for the warm-started seed re-eval
        # and is never read again, so it is pure duplicate storage. The per-eval
        # powerstate pkls under Initialization/initialization_simple_relax (needed by
        # PORTALSanalyzer) are NOT touched.
        cleanup_seed_march=True,
        # --- LHS seed (run(seed="lhs")): base x0 + banded LHS design -------------------
        # Half-width of the per-DV LHS design as a fraction of the DV range (relative
        # bounds fraction). Small keeps every seed in-basin (clean local GP); the loop
        # travels far via TR growth, so this need NOT be large even when x0 is far from
        # the solution.
        seed_lhs_frac=0.5,
        seed_lhs_seed=0,
        # inner GP-surrogate NLLS solve (scipy least_squares, TRF)
        inner_diff_step=1e-4,
        inner_xtol=1e-6,
        inner_ftol=1e-6,
        inner_gtol=1e-6,
        inner_max_nfev=200,
        # Build the inner Jacobian from one batched GP posterior instead of scipy's
        # point-by-point 2-point differencing (identical stencil, ~13x fewer posterior
        # calls). False reverts to scipy's own finite differences.
        inner_batched_jac=True,
        train_Ystd_rel=None,
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

        self.folderOutputs = IOtools.expandPath(portals_fun.folder) / "Outputs"
        self.folderOutputs.mkdir(parents=True, exist_ok=True)
        self.folderExec = IOtools.expandPath(portals_fun.folder) / "Execution"
        self.folderExec.mkdir(parents=True, exist_ok=True)

        o = dict(self._defaults)
        o.update(options or {})
        self.options = o

        # Force the GP onto the GRADIENTS-ONLY feature set for the whole run, robustly.
        # The namelist portals_transformation_variables schedule is selected on
        # SURROGATEtools.num_training_points = native + POOLED (global_surrogates) points,
        # so with pooling the count blows past the [20,50,...] thresholds at eval 1 and the
        # full nuei/tite/beta_e set (VIF up to ~160 on aLne -> flat surrogate, rho=+/-1) is
        # used regardless of how the thresholds are delayed in the namelist. This override
        # replaces the whole schedule with a SINGLE line = the least-feature (gradients-only)
        # line already in the config, keyed above any reachable count, so it is always
        # selected. Does not mutate the shared portals_fun.surrogate_parameters.
        if o.get("gradients_only_features", False):
            alltimes = self.surrogate_parameters.get(
                "surrogate_transformation_variables_alltimes", None)
            if alltimes:
                grads = copy.deepcopy(alltimes[min(alltimes.keys())])
                sp = dict(self.surrogate_parameters)
                sp["surrogate_transformation_variables_alltimes"] = {10000: grads}
                sp["surrogate_transformation_variables_lasttime"] = copy.deepcopy(grads)
                self.surrogate_parameters = sp
                feats = list(grads.keys()) if isinstance(grads, dict) else grads
                print(f"\t- SALM: gradients_only_features ON -> GP uses only {feats} "
                      f"for all evals (transition schedule bypassed)", typeMsg="i")

        # edge_uq_final_only: suppress the per-eval UQ trigger during the loop by
        # stashing fun._edge_uq_inputs; the final UQ runs once on the best eval in
        # _run_impl, and run() restores the attribute on exit (even on error).
        self._edge_uq_inputs_stashed = None
        if o.get("edge_uq_final_only", True):
            uq_in = getattr(portals_fun, "_edge_uq_inputs", None)
            if uq_in is not None:
                self._edge_uq_inputs_stashed = uq_in
                portals_fun._edge_uq_inputs = None
                print("\t- SALM: edge_uq_final_only ON -> per-eval edge-UQ suppressed; "
                      "run_edge_uq will run ONCE on the best eval at the end", typeMsg="i")
                # persist the run's own UQ spec so the offline/CLI best-index UQ
                # (edge_tools.uq.best_uq) pulls THESE inputs+options from the folder
                # instead of re-declared defaults -- even for a crashed/killed run.
                try:
                    import dill as pickle_dill
                    with open(self.folderOutputs / "edge_uq_spec.pkl", "wb") as h:
                        pickle_dill.dump(
                            {"uq_inputs": uq_in,
                             "uq_options": copy.deepcopy(
                                 getattr(portals_fun, "_edge_uq_options", {}) or {})},
                            h, protocol=4)
                except Exception as e:
                    print(f"\t- SALM: could not persist edge_uq_spec ({e})", typeMsg="w")

        self.optimization_data = BOgraphics.optimization_data(
            self.dvs, self.ofs, file=self.folderOutputs / "optimization_data.csv")

        # accumulated real training set (raw outputs)
        self.train_X = np.empty((0, len(self.dvs)))
        self.train_Y = np.empty((0, len(self.ofs)))
        self.train_Ystd = np.empty((0, len(self.ofs)))

        # GP + anchor state
        self.gp_combined = None
        self.gp_individual = None
        self._transition_pos = None
        self._eval_counter = 0
        self._corr_delta_Y = None   # zeroth-order-consistent surrogate correction
        self._anchor_x_opt = None   # TR center (opt space) = last real-confirmed point
        self._anchor_Y = None
        self._last_eval_Y = None
        self._n_seed = None         # # of seed (initialization) evals; set in _run_impl

        # bounds + dv_scale set in run()
        self._lo = self._hi = self._dv_scale = None
        self._tr_rel = float(np.clip(o["tr_init"], o["tr_min"], o["max_total_rel_step"]))

        # interior DV indices for the anchor ridge
        self._interior_idx = np.arange(len(self.dvs), dtype=int)

        # number of flux residual rows (set on first _compose / _residual_variance)
        self.n_flux = None

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
            self.fun, folder, dictDVs, f"salm_ev{n}",
            numPORTALS=n, dictOFs=dictOFs,
            remove_folder_upon_completion=not self.fun.portals_parameters["solution"]["keep_full_model_folder"],
        )
        self._save_powerstate_for_plotting(n, powerstate_result)

        Y = np.array([float(dictOFs[of]["value"]) for of in self.ofs])
        Ystd = np.array([float(dictOFs[of]["error"]) for of in self.ofs])
        return Y, Ystd

    def _powerstate_pkl_path(self, n):
        """Where eval ``n``'s powerstate.pkl lives. SEED (initialization) evals go to
        Initialization/initialization_simple_relax (the PORTALS convention -- that
        folder holds ONLY the initial training, and feeds the init-only
        PORTALSinitializer fallback). LOOP (optimization) evals are co-located with
        their transport under Execution/Evaluation.{n}. Both are consolidated into
        optimization_extra.pkl for the main PORTALSanalyzer plotter, so relocating
        the loop pkls does not truncate the iteration history."""
        n_seed = self._n_seed
        if n_seed is None or n < n_seed:
            return (IOtools.expandPath(self.fun.folder) / "Initialization"
                    / "initialization_simple_relax" / f"portals_sr_ev_{n}" / "powerstate.pkl")
        return self.folderExec / f"Evaluation.{n}" / "powerstate.pkl"

    def _save_powerstate_for_plotting(self, n, powerstate_result):
        try:
            pkl = self._powerstate_pkl_path(n)
            pkl.parent.mkdir(parents=True, exist_ok=True)
            powerstate_result.save(pkl)

            prof_dir = self.folderOutputs / "portals_profiles"
            prof_dir.mkdir(parents=True, exist_ok=True)
            powerstate_result.from_powerstate(
                write_input_gacode=prof_dir / f"input.gacode.{n}",
                postprocess_input_gacode=self.fun.portals_parameters["transport"]["applyCorrections"],
            )
        except Exception as e:
            print(f"\t- SALM: could not save powerstate for plotting (eval {n}): {e}",
                  typeMsg="w")

    def _store_real(self, x_phys_np, Y, Ystd):
        self.train_X = np.append(self.train_X, x_phys_np[None, :], axis=0)
        self.train_Y = np.append(self.train_Y, Y[None, :], axis=0)
        self.train_Ystd = np.append(self.train_Ystd, Ystd[None, :], axis=0)

        _, _, R = self._compose(torch.from_numpy(Y).to(self.dfT).unsqueeze(0),
                                torch.from_numpy(x_phys_np).to(self.dfT))
        R = R.detach().cpu().numpy()
        # Reported objective is the PORTALS flux residual only (partitioned).
        obj = -self._portals_res(R)
        self.optimization_data.update_points(
            self.train_X, Y=self.train_Y, Ystd=self.train_Ystd,
            objective=np.append(np.full(self.train_X.shape[0] - 1, np.nan), obj))

    # ==================================================================
    # Residual composition (raw OF vector -> normalized residual vector)
    # ==================================================================

    def _compose(self, Y_row, X_phys):
        """Returns (of, cal, R). R is the ABSOLUTE flux misfit cal - of (the
        PORTALS data term), UNnormalized."""
        of, cal, _ = self.fun.scalarized_objective(Y_row)
        source = (cal - of).squeeze(0)
        self.n_flux = int(source.shape[-1])
        return of, cal, source

    def _flux_sq(self, R_np):
        nf = getattr(self, "n_flux", None) or R_np.shape[-1]
        rf = R_np[:nf]
        return float(rf @ rf)

    def _portals_res(self, R_np):
        """PORTALS flux objective magnitude (1/n_flux)*||cal - of||."""
        nf = getattr(self, "n_flux", None) or R_np.shape[-1]
        return float(np.linalg.norm(R_np[:nf]) / nf)

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

    def _residuals_batch_np(self, X_opt_np):
        """Residuals for a BATCH of points (B, ndv) -> (B, n_res), in ONE posterior call.

        A single-point posterior over the per-output ModelList costs ~0.24 s, essentially
        all fixed overhead, so evaluating a 13-point finite-difference stencil one point
        at a time costs ~3 s where the batch costs ~0.25 s. Same arithmetic, ~13x less
        botorch bookkeeping."""
        X_opt = torch.from_numpy(np.atleast_2d(X_opt_np)).to(self.dfT)
        with torch.no_grad():
            X_phys = inverse_transform(X_opt, self.mono_pairs)
            y_mean, _, _, _ = self.gp_combined.predict(X_phys)
            if self._corr_delta_Y is not None:
                y_mean = y_mean + self._corr_delta_Y
            of, cal, _ = self.fun.scalarized_objective(y_mean)
            R = (cal - of)
            self.n_flux = int(R.shape[-1])
            return R.detach().cpu().numpy()

    def _inner_jac(self, x_opt_np, anchor, sqrtW, ridge_w, lo_box, hi_box):
        """Forward-difference Jacobian of _inner_residual, built from ONE batched
        posterior instead of scipy's point-by-point `jac="2-point"`.

        Steps are clipped into the trust-region box and the ACTUAL taken step is used as
        the denominator, so a variable pinned to a box face gets a one-sided difference
        rather than a step that silently leaves the region."""
        o = self.options
        n = x_opt_np.size
        h = o["inner_diff_step"] * np.maximum(np.abs(x_opt_np), 1.0)

        # keep every perturbed point inside the box; flip to a backward step at the edge
        xp = np.repeat(x_opt_np[None, :], n, axis=0)
        idx = np.arange(n)
        step = h.copy()
        forward = (x_opt_np + h) <= hi_box
        step = np.where(forward, h, -h)
        too_low = (x_opt_np + step) < lo_box
        step = np.where(too_low, hi_box - x_opt_np, step)          # degenerate box: span it
        step = np.where(np.abs(step) < 1e-30, 1e-12, step)
        xp[idx, idx] = x_opt_np + step

        R = self._residuals_batch_np(np.vstack([x_opt_np[None, :], xp]))   # (n+1, n_res)
        J = ((R[1:] - R[0][None, :]) / step[:, None]).T                    # (n_res, n)
        J = sqrtW[:, None] * J

        idxs = self._interior_idx
        if ridge_w > 0.0 and idxs.size:
            P = np.zeros((idxs.size, n))
            P[np.arange(idxs.size), idxs] = ridge_w / self._dv_scale[idxs]
            J = np.vstack([J, P])
        return J

    def _flux_pred_np(self, x_opt_np):
        """Surrogate-predicted flux squared-residual (partitioned data term)."""
        R = self._residuals_np(x_opt_np)
        nf = getattr(self, "n_flux", None) or R.shape[-1]
        return float(R[:nf] @ R[:nf])

    # ------------------------------------------------------------------
    # Diagonal GP-posterior residual variance -> W = diag(1/sigma_F^2)
    # ------------------------------------------------------------------

    def _residual_variance_np(self, x_opt_np):
        """Flux rows: propagate per-raw-output GP variance through the affine
        residual Jacobian G = d(cal-of)/d(y_raw)."""
        stds = self.options["stds"]
        x_phys = inverse_transform(
            torch.from_numpy(x_opt_np).to(self.dfT), self.mono_pairs)

        with torch.no_grad():
            y_mean, y_upper, y_lower, _ = self.gp_combined.predict(x_phys.unsqueeze(0))
            sigma_y = (y_upper - y_lower).squeeze(0).abs() / (2.0 * stds)
            Sigma_y_diag = (sigma_y ** 2).clamp(min=1e-12)

        def flux_source(y_raw):
            of, cal, _ = self.fun.scalarized_objective(y_raw.unsqueeze(0))
            return (cal - of).squeeze(0)

        y_raw_in = y_mean.squeeze(0).detach().clone()
        _, G = multivariate_tools.mitim_jacobian(flux_source, y_raw_in, vectorize=True)
        G = G.detach()

        R = self._residual_vector(torch.from_numpy(x_opt_np).to(self.dfT)).detach()
        n_total = int(R.shape[-1])
        n_flux = int(G.shape[0])
        self.n_flux = n_flux

        var_flux_np = ((G ** 2) @ Sigma_y_diag).clamp(min=1e-12).cpu().numpy()
        Sigma_F = np.empty(n_total)
        Sigma_F[:n_flux] = var_flux_np
        return Sigma_F

    def _ridge_weight(self, W, nf):
        """FIXED Levenberg ridge sqrt(lambda), lambda = lm_ridge_rel * median flux
        weight (ties the ridge to the data term). 0 when disabled / no flux rows."""
        rel = self.options.get("lm_ridge_rel", 0.0)
        if rel <= 0.0 or nf <= 0:
            return 0.0
        w_flux = W[:nf]
        if w_flux.size == 0:
            return 0.0
        return float(np.sqrt(rel * float(np.median(w_flux))))

    def _inner_residual(self, x_opt_np, anchor, sqrtW, ridge_w):
        """Weighted data+prior residual augmented with the fixed anchor-ridge rows
        sqrt(lambda)*(x-anchor)/dv_scale over the interior DVs."""
        R = sqrtW * self._residuals_np(x_opt_np)
        idx = self._interior_idx
        if ridge_w > 0.0 and idx.size:
            pen = ridge_w * (x_opt_np[idx] - anchor[idx]) / self._dv_scale[idx]
            return np.concatenate([R, pen])
        return R

    # ------------------------------------------------------------------
    # Real evaluation wrapper
    # ------------------------------------------------------------------

    def _real_eval_np(self, x_opt_np):
        x_phys = inverse_transform(
            torch.from_numpy(x_opt_np).to(self.dfT), self.mono_pairs
        ).detach().cpu().numpy()

        # capture the GP prediction at this point BEFORE the real eval, so the
        # surrogate accuracy (pred vs truth) can be logged per output.
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
        return R, self._flux_sq(R)

    def _log_surrogate_predictions(self, eval_idx, pred_mean, pred_std, Y_real):
        """Per-output GP accuracy at each real eval -> Outputs/surrogate_accuracy.csv:
        predicted mean/std vs the measured value, % error, and z-score
        (pred-real)/pred_std. A well-calibrated GP has |z|~1 and small %error; large
        |z| or %error flags where the surrogate is mispredicting (the GP-performance
        check the original SALM exposed)."""
        try:
            f = self.folderOutputs / "surrogate_accuracy.csv"
            write_header = not f.exists()
            rows = []
            for j, out in enumerate(self.ofs):
                real = float(Y_real[j])
                pm, ps = float(pred_mean[j]), float(max(pred_std[j], 1e-30))
                pct = 100.0 * (pm - real) / real if abs(real) > 1e-30 else np.nan
                rows.append({"eval": int(eval_idx), "output": out,
                             "pred_mean": pm, "pred_std": ps, "real": real,
                             "pct_error": pct, "z_score": (pm - real) / ps})
            pd.DataFrame(rows).to_csv(f, mode="a", header=write_header, index=False)
        except Exception as e:
            print(f"\t- SALM: surrogate accuracy log failed ({e})", typeMsg="w")

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
            print(f"\t- SALM: full GP fit (feature set #{pos}, {n} pts)", typeMsg="i")
            self._build_gps(optimize=True)
            self._transition_pos = pos
        else:
            print(f"\t- SALM: warm GP update (reuse hyperparams, {n} pts)", typeMsg="i")
            try:
                self._build_gps(optimize=False)
            except Exception as e:
                print(f"\t- SALM warm update failed ({e}); full fit", typeMsg="w")
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

            GP = SURROGATEtools.surrogate_model(
                self.train_X, self.train_Y[:, i:i + 1], Yvar[:, i:i + 1],
                self.surrogate_parameters,
                output=out, output_transformed=out_t,
                dfT=self.dfT, surrogate_options=sopt,
                fileTraining=fileTraining if optimize else None,
            )
            if optimize:
                GP.fit()
            else:
                # Copy only the entries whose shape still matches. The point of the warm
                # path is to reuse the HYPERPARAMETERS (lengthscales, outputscale, mean,
                # noise level) -- everything sized by the training-set length is rebuilt
                # right after by normalization_pass anyway.
                #
                # A plain load_state_dict() here NEVER succeeded: the outcome transform
                # keeps (n_points, 1) buffers (means / stdvs / _stdvs_sq), so every time
                # the training set grew by one point it raised a size mismatch and the
                # caller silently fell back to a FULL hyperparameter refit (~50 s/iter).
                #
                # This filter clears that blocker (21 hyperparameter tensors reused, 3
                # size-dependent buffers rebuilt), but the warm path is STILL not reached:
                # normalization_pass -> input_transform_physics(train_X) then fails with
                # "Expected at least 4 params for 'te', got 3", i.e. the physics input
                # transform is re-applied to already-transformed features. That second
                # blocker lives in the shared SURROGATEtools / Transformation_Inputs
                # plumbing (the `parameters_combined` cache) that every PORTALS run uses,
                # so it is deliberately NOT patched here. Until it is fixed the try/except
                # in _fit_surrogates still falls back to a full fit -- behaviour unchanged.
                sd_src = prev[i].gpmodel.state_dict()
                sd_dst = GP.gpmodel.state_dict()
                compatible = {k: v for k, v in sd_src.items()
                              if (k in sd_dst) and (sd_dst[k].shape == v.shape)}
                n_skipped = len(sd_src) - len(compatible)
                GP.gpmodel.load_state_dict(compatible, strict=False)
                if i == 0 and n_skipped:
                    print(f"\t- SALM warm update: reused {len(compatible)} hyperparameter "
                          f"tensors, rebuilt {n_skipped} size-dependent buffer(s)",
                          typeMsg="i")
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
            dfT=self.dfT, surrogate_options=self.surrogate_options)
        combined.gpmodel = BOTORCHtools.ModifiedModelListGP(*(g.gpmodel for g in individual))

        self.gp_individual = individual
        self.gp_combined = combined

    # ==================================================================
    # Seeding: simple-relaxation trajectory
    # ==================================================================

    def _relax_from(self, x0, n_iters, subfolder, idx_offset):
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

        # Constrain the relaxation march to the DV box. simple_relaxation is an
        # unconstrained Picard fixed-point iteration; without bounds it can march a
        # gradient DV outside [dvs_min, dvs_max] (e.g. a pinned channel), producing an
        # INFEASIBLE "best" seed. The loop then clips that anchor back into the box but
        # keeps the infeasible point's residual as f_cur, so every feasible step is
        # rejected (guaranteed stall). _sr_step already clamps to bounds[0]/bounds[1]
        # (shape [2, n_dv], in dvs order) when given -- we just have to pass them.
        bounds = torch.stack([
            torch.from_numpy(self.dvs_min).to(self.dfT),
            torch.from_numpy(self.dvs_max).to(self.dfT),
        ])
        ps.flux_match(algorithm="simple_relax", solver_options=solver_options, bounds=bounds)
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
        return self._relax_from(x0, n_seed, "salm_simple_relax", 0)

    def _seed_lhs(self, x0):
        """LHS seed for a near-basin start: base x0 (unperturbed) + a small banded LHS
        design (half-width = seed_lhs_frac * DV range, a relative bounds fraction), all
        in-basin and fresh-evaluated. Unlike the relaxation march (a fixed-point iteration
        that overshoots a good x0 and scatters seeds to worse far points), this gives the
        GP a clean LOCAL model at x0; the loop then TRAVELS to the solution via boundary-
        limited TR growth."""
        n_seed = int(self.fun.optimization_options
                     .get("initialization_options", {}).get("initial_training", 5))
        x0 = np.asarray(x0, dtype=float)
        seeds = [x0.copy()]
        m = n_seed - 1
        idx = self._interior_idx
        frac = float(self.options.get("seed_lhs_frac", 0.05))
        if m <= 0 or idx.size == 0 or frac <= 0.0:
            return seeds
        rng = np.random.default_rng(int(self.options.get("seed_lhs_seed", 0)))
        # Band scale. DEFAULT base-relative (frac * |base DV|), box-INDEPENDENT -- the
        # legacy box-relative `frac * (dvs_max-dvs_min)` blows up on wide/absolute bounds
        # (e.g. [0,100] -> frac 0.25 = +-25 in aLy -> clips gradients to 0 -> unphysical
        # seeds). Set seed_lhs_base_relative=False for the legacy box-relative band.
        if self.options.get("seed_lhs_base_relative", True):
            span = np.maximum(np.abs(self.dvs_base[idx]), 1e-2)
            band_desc = "of |base DV|"
        else:
            span = (self.dvs_max - self.dvs_min)[idx]
            band_desc = "of DV range"
        # LHS in [-1, 1]^d (space-filling around x0), one row per extra seed
        U = np.zeros((m, idx.size))
        for j in range(idx.size):
            U[:, j] = (rng.permutation(m) + rng.random(m)) / m
        U = 2.0 * U - 1.0
        for k in range(m):
            x = x0.copy()
            x[idx] = np.clip(x[idx] + U[k] * frac * span, self.dvs_min[idx], self.dvs_max[idx])
            seeds.append(x)
        print(f"\t- SALM: LHS seed -- base x0 + {m} banded LHS points "
              f"(frac={frac:.3f} {band_desc}); loop travels via TR growth", typeMsg="i")
        return seeds

    # ==================================================================
    # Public entry point
    # ==================================================================

    def _opt_space_bounds(self):
        """Opt-space box bounds for the softplus-gap transform (feasible gap range
        gap_lo=max(0, dmin[j]-dmax[i]), gap_hi=dmax[j]-dmin[i])."""
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
        # Reproducibility manifest: stamp the code versions (commit + local-modification
        # state) into the log and Outputs/version_manifest.json, so A-B comparisons across
        # runs can be audited (see audit_runs.py). A bare hash is not enough when the working
        # tree is locally modified -- diff_sha1 pins the exact state.
        try:
            from mitim_tools.misc_tools import IOtools as _IO
            import json as _json
            _man = _IO.code_version_manifest()
            print("=" * 70)
            print("[MITIM] code version manifest (reproducibility):")
            for _k, _v in _man.items():
                if "error" in _v:
                    print(f"    {_k}: (git error) {_v['error']}")
                    continue
                _c = (_v.get("commit") or "?")[:10]
                _state = "clean" if _v.get("clean") else \
                    f"+{_v.get('n_uncommitted')} uncommitted [diff {_v.get('diff_sha1')}]"
                print(f"    {_k}: {_c} ({_v.get('branch')}) {_state}")
            print("=" * 70)
            with open(self.folderOutputs / "version_manifest.json", "w") as _fh:
                _json.dump(_man, _fh, indent=2)
        except Exception as _e:
            print(f"[MITIM] version manifest failed (non-fatal): {_e!r}")
        try:
            return self._run_impl(x0=x0, seed=seed)
        finally:
            # restore the suppressed per-eval UQ hook (edge_uq_final_only), even on error
            if self._edge_uq_inputs_stashed is not None:
                self.fun._edge_uq_inputs = self._edge_uq_inputs_stashed
            sys.stdout = prev_stdout

    def _run_impl(self, x0=None, seed="relaxation"):
        x0 = self.dvs_base.copy() if x0 is None else np.asarray(x0, dtype=float)

        if seed == "relaxation":
            seeds = list(self._seed_relaxation(x0))
        elif seed == "lhs":
            seeds = list(self._seed_lhs(x0))
        elif seed is None:
            seeds = [x0]
        else:
            seeds = list(seed)
        self._n_seed = len(seeds)   # seed/loop boundary for pkl placement

        seed_f = []
        for xs in seeds:
            xs = np.asarray(xs, dtype=float)
            Y, Ystd = self._evaluate_real(xs)
            self._store_real(xs, Y, Ystd)
            with torch.no_grad():
                _, _, R = self._compose(torch.from_numpy(Y).to(self.dfT).unsqueeze(0),
                                        torch.from_numpy(xs).to(self.dfT))
            seed_f.append(self._flux_sq(R.cpu().numpy()))
        self._fit_surrogates()

        # drop the transient seed-march folder (its transport folders are already
        # duplicated into Execution/Evaluation.{i}; nothing reads it after seeding).
        if self.options.get("cleanup_seed_march", True):
            march = IOtools.expandPath(self.fun.folder) / "Initialization" / "salm_simple_relax"
            if march.exists():
                try:
                    IOtools.shutil_rmtree(march)
                    print("\t- SALM: removed transient seed-march folder "
                          "(Initialization/salm_simple_relax)", typeMsg="i")
                except Exception as e:
                    print(f"\t- SALM: could not remove seed-march folder ({e})", typeMsg="w")

        lo, hi = self._opt_space_bounds()
        self._lo, self._hi = np.minimum(lo, hi), np.maximum(lo, hi)
        # Trust-region step scale. DEFAULT: BASE-RELATIVE -- tr_rel is a fraction of the
        # base DV magnitude (in opt space), so the TR step is INDEPENDENT of the DV box
        # width. Widening/opening the bounds no longer silently rescales tr_min/tr_init
        # (the legacy box-relative scale made tr_min=1e-2 mean ~1% of base only for the
        # then-current bounds; a wider box turned it into a much coarser step -> surrogate
        # mispredict -> TR-floor stall). Set tr_base_relative=False to recover the legacy
        # box-relative behavior.
        if self.options.get("tr_base_relative", True):
            x0_opt = forward_transform(
                torch.as_tensor(np.asarray(self.dvs_base, dtype=float)).to(self.dfT),
                self.mono_pairs).detach().cpu().numpy()
            self._dv_scale = np.maximum(np.abs(x0_opt),
                                        float(self.options.get("tr_scale_floor", 0.1)))
        else:
            self._dv_scale = np.maximum(self._hi - self._lo, 1e-6)

        i_best = int(np.argmin(seed_f))
        x_start = np.asarray(seeds[i_best], dtype=float)
        f0 = seed_f[i_best]
        x_cur = forward_transform(torch.from_numpy(x_start).to(self.dfT),
                                  self.mono_pairs).detach().cpu().numpy()
        x_cur = np.clip(x_cur, self._lo, self._hi)
        self._set_anchor(x_cur, self.train_Y[i_best])
        print(f"\t- SALM: starting LM loop from seed {i_best} (f={f0:.3e})", typeMsg="i")

        x_best_opt, f_hist, x_hist = self._salm_loop(x_cur, f0)
        x_best = inverse_transform(
            torch.from_numpy(x_best_opt).to(self.dfT), self.mono_pairs
        ).detach().cpu().numpy()

        try:
            self._write_plotting_artifacts()
        except Exception as e:
            print(f"\t- SALM: could not write plotting artifacts: {e}", typeMsg="w")

        # edge_uq_final_only: the single, end-of-run UQ on the best evaluation
        if self._edge_uq_inputs_stashed is not None:
            try:
                self._run_final_edge_uq()
            except Exception as e:
                print(f"\t- SALM: final edge-UQ failed ({type(e).__name__}: {e})",
                      typeMsg="w")
        return x_best, f_hist, x_hist

    def _run_final_edge_uq(self):
        """Run run_edge_uq ONCE on the best real evaluation. Loads the best eval's
        saved powerstate and issues the same call the PORTALSedge per-eval hook makes,
        leaving the result on fun._edge_uq_last / _edge_uq_history (keyed by the best
        eval index) so downstream analysis (analyze_solve) finds it unchanged."""
        from mitim_tools.edge_tools.uq.run import run_edge_uq
        from mitim_modules.powertorch import STATEtools

        # best real eval = min flux-only residual over all stored evals
        res = []
        for k in range(self.train_X.shape[0]):
            with torch.no_grad():
                _, _, R = self._compose(
                    torch.from_numpy(self.train_Y[k]).to(self.dfT).unsqueeze(0),
                    torch.from_numpy(self.train_X[k]).to(self.dfT))
            res.append(self._flux_sq(R.cpu().numpy()))
        ibest = int(np.argmin(res))

        pkl = self._powerstate_pkl_path(ibest)
        if not pkl.exists():
            print(f"\t- SALM: final edge-UQ skipped (no powerstate pkl for best eval "
                  f"{ibest})", typeMsg="w")
            return
        ps = STATEtools.read_saved_state(pkl)

        uq_opts = dict(getattr(self.fun, "_edge_uq_options", {}) or {})
        for key in ("inject_into", "start_after_eval"):
            uq_opts.pop(key, None)
        folder = uq_opts.pop("folder", self.folderOutputs / "edge_uq_best")
        X = torch.from_numpy(self.train_X[ibest]).to(self.dfT).unsqueeze(0)

        print(f"\t- SALM: running final edge-UQ ONCE on best eval {ibest} "
              f"(res={np.sqrt(max(res[ibest], 0.0)) / self.n_flux:.3e})", typeMsg="i")
        last = run_edge_uq(ps, self._edge_uq_inputs_stashed, X_dvs=X, inject_into=ps,
                           folder=folder, **uq_opts)
        self.fun._edge_uq_last = last
        if not hasattr(self.fun, "_edge_uq_history"):
            self.fun._edge_uq_history = []
        self.fun._edge_uq_history.append((ibest, last))
        # persist the band so completed runs keep it (offline analysis / best_uq CLI)
        try:
            import dill as pickle_dill
            with open(self.folderOutputs / "edge_uq_best.pkl", "wb") as h:
                pickle_dill.dump({"ibest": ibest, "result": last}, h, protocol=4)
            print(f"\t- SALM: final edge-UQ saved to Outputs/edge_uq_best.pkl", typeMsg="i")
        except Exception as e:
            print(f"\t- SALM: could not pickle edge_uq_best ({e})", typeMsg="w")
        # and persist it where the PLOTTER reads it: run_edge_uq injected the
        # {key}_uq_std bands into the loaded powerstate in-memory only, but
        # _write_plotting_artifacts already ran -- re-save the pkl and rebuild the
        # artifacts so optimization_extra.pkl carries the band (else no bands appear
        # in mitim_plot_portals_edge).
        try:
            ps.save(pkl)
            self._checkpoint_plotting_artifacts()
            print("\t- SALM: UQ band persisted into plotting artifacts", typeMsg="i")
        except Exception as e:
            print(f"\t- SALM: could not persist UQ band into plotting artifacts ({e})",
                  typeMsg="w")

    # ==================================================================
    # SALM main loop (minimal)
    # ==================================================================

    def _log_progress(self, eval_idx, res_flux, rho, step, accepted, reject_streak,
                      step_frac=np.nan):
        """Append the collapse-vs-open signal to Outputs/salm_progress.csv: per-eval TR
        half-width, gain ratio rho, step, accept flag, consecutive-reject streak, active
        GP feature set, training-set size, and step_frac (max per-dim |step|/half; ~1
        means the step is pressed against the TR edge -> a frozen-tiny TR is throttling
        progress). OPENING run => tr trends UP with accepts; COLLAPSING => tr ratchets to
        the floor with a growing reject streak; THROTTLED => accepts with step_frac~1 and
        flat tr (the boundary-grow fix targets this)."""
        try:
            f = self.folderOutputs / "salm_progress.csv"
            row = {
                "eval": int(eval_idx),
                "n_pts": int(self.train_X.shape[0]),
                "feature_pos": (int(self._transition_pos)
                                if self._transition_pos is not None else -1),
                "res_flux": float(res_flux),
                "rho": float(rho),
                "tr_rel": float(self._tr_rel),
                "step": float(step),
                "step_frac": float(step_frac),
                "accepted": int(bool(accepted)),
                "reject_streak": int(reject_streak),
            }
            pd.DataFrame([row]).to_csv(f, mode="a", header=not f.exists(), index=False)
        except Exception as e:
            print(f"\t- SALM: progress log failed ({e})", typeMsg="w")

    def _shrink_tr(self):
        self._tr_rel = max(self._tr_rel * self.options["tr_shrink"], self.options["tr_min"])

    def _grow_tr(self):
        self._tr_rel = min(self._tr_rel * self.options["tr_grow"], self.options["max_total_rel_step"])

    def _salm_loop(self, x_cur, f_cur):
        """Surrogate-trust-region LM. Each outer iteration = 1 real eval: build
        GP-posterior weights at the anchor, fully solve the cheap GP residual in
        the TR box (scipy TRF), eval the real model once at the optimum, accept iff
        the measured flux misfit improves, adapt the TR by the real gain ratio rho."""
        o = self.options
        self._tr_rel = float(np.clip(o["tr_init"], o["tr_min"], o["max_total_rel_step"]))
        f_hist = [f_cur]
        x_hist = [x_cur.copy()]
        best_hist = [f_cur]   # monotone best flux_sq (for the diminishing-returns exit)
        real_evals = 0
        tr_resets_used = 0
        reject_streak = 0   # consecutive rejects (collapse signal)
        plateau_guard = 0   # eval index before which the plateau-stop may not fire
                            # (set after an escape-probe reset to give it a full window)

        while True:
            anchor = self._anchor_x_opt

            Sigma_F = self._residual_variance_np(anchor)   # also sets self.n_flux
            nf = self.n_flux
            W = 1.0 / np.clip(Sigma_F, 1e-12, None)
            sqrtW = np.sqrt(W)
            ridge_w = self._ridge_weight(W, nf)

            half = self._tr_rel * self._dv_scale
            lo_box = np.maximum(self._lo, anchor - half)
            hi_box = np.minimum(self._hi, anchor + half)
            degenerate = hi_box <= lo_box
            hi_box[degenerate] = lo_box[degenerate] + 1e-12
            x0 = np.clip(anchor, lo_box, hi_box)

            # Supply the Jacobian instead of letting scipy finite-difference it: scipy
            # would call the residual once per DV perturbation, and each call is a full
            # ModelList posterior (~0.24 s of mostly fixed overhead). _inner_jac gets the
            # whole stencil from ONE batched posterior. Set inner_batched_jac=False to
            # fall back to scipy's 2-point differencing.
            if o.get("inner_batched_jac", True):
                jac_arg = lambda x: self._inner_jac(x, anchor, sqrtW, ridge_w, lo_box, hi_box)
            else:
                jac_arg = "2-point"
            sol = least_squares(
                lambda x: self._inner_residual(x, anchor, sqrtW, ridge_w), x0,
                jac=jac_arg, diff_step=o["inner_diff_step"],
                bounds=(lo_box, hi_box), method="trf", x_scale=o["inner_x_scale"],
                xtol=o["inner_xtol"], ftol=o["inner_ftol"], gtol=o["inner_gtol"],
                max_nfev=o["inner_max_nfev"],
            )
            x_trial = np.clip(sol.x, self._lo, self._hi)
            step = float(np.linalg.norm(x_trial - anchor))
            pred_red = f_cur - self._flux_pred_np(x_trial)

            # --- stationary surrogate: convergence iff the real residual is matched ---
            if step < o["xtol"]:
                res_cur = np.sqrt(max(f_cur, 0.0)) / self.n_flux
                # Same DV rule on the stationary-surrogate path: if the proposed step is
                # this small RELATIVE to x, the design vector has settled -- report that
                # as convergence instead of shrinking the TR into a floor stall.
                if float(o.get("xtol_rel", 0.0) or 0.0) > 0.0 and \
                        step / max(float(np.linalg.norm(anchor)), 1e-12) < o["xtol_rel"]:
                    print(f"SALM converged (surrogate stationary, DV step "
                          f"|dx|/|x|={step / max(float(np.linalg.norm(anchor)), 1e-12):.3e} "
                          f"< xtol_rel, res={res_cur:.3e})", typeMsg="i")
                    break
                if res_cur < o["res_tol"]:
                    print(f"SALM converged (surrogate stationary, res={res_cur:.3e}, "
                          f"|step|={step:.2e})", typeMsg="i")
                    break
                if self._tr_rel > o["tr_min"] * (1.0 + 1e-9):
                    self._shrink_tr()
                    print(f"\t- SALM stall (res={res_cur:.3e}); shrink TR -> "
                          f"{self._tr_rel:.2e}, retry", typeMsg="w")
                    x_cur = anchor.copy(); self._recompute_correction()
                    continue
                if tr_resets_used < o["tr_resets"]:
                    tr_resets_used += 1
                    self._tr_rel = float(np.clip(o["tr_init"], o["tr_min"], o["max_total_rel_step"]))
                    self._fit_surrogates(force_full=True)
                    self._recompute_correction()
                    print(f"SALM: TR floor stall; reset #{tr_resets_used}/{o['tr_resets']}",
                          typeMsg="w")
                    continue
                print(f"SALM stalled: surrogate stationary at TR floor, res={res_cur:.3e}",
                      typeMsg="w")
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

            # boundary-limited: the accepted step pressed against the TR edge (in at
            # least one dim), so the optimizer wants more room. Grow even in the neutral
            # rho band -- otherwise the TR freezes tiny and the run crawls.
            step_frac = float(np.max(np.abs(x_trial - anchor) / np.maximum(half, 1e-30)))
            boundary_limited = step_frac >= o["tr_grow_boundary_frac"]
            if (rho < o["rho_lo"]) or (not accepted):
                self._shrink_tr()
            elif (rho > o["rho_hi"]) or (accepted and rho > o["rho_lo"] and boundary_limited):
                self._grow_tr()

            reject_streak = 0 if accepted else reject_streak + 1
            res_real = np.sqrt(max(f_real, 0.0)) / self.n_flux
            print(f"\t- SALM eval {real_evals}: res_flux={res_real:.3e} "
                  f"(target<{o['res_tol']:.0e}) rho={rho:+.2f} tr={self._tr_rel:.2e} "
                  f"|step|={step:.2e} rej_streak={reject_streak} "
                  f"{'ACCEPT' if accepted else 'reject'}", typeMsg="i")
            self._log_progress(real_evals, res_real, rho, step, accepted, reject_streak,
                               step_frac=step_frac)

            self._fit_surrogates(force_full=not accepted)

            if accepted:
                x_cur = x_trial
                f_cur = f_real
                self._set_anchor(x_trial, best_Y)
                # DV convergence (xtol_rel): the accepted step no longer moves the
                # design vector appreciably, so the solution has settled regardless of
                # what the residual floor happens to be. `anchor` is still the PRE-step
                # x here, and step = |x_trial - anchor|, so this is |dx|/|x|.
                if float(o.get("xtol_rel", 0.0) or 0.0) > 0.0:
                    rel_dx = step / max(float(np.linalg.norm(anchor)), 1e-12)
                    if rel_dx < o["xtol_rel"]:
                        print(f"SALM converged (DV step |dx|/|x|={rel_dx:.3e} < "
                              f"xtol_rel={o['xtol_rel']:.3e}, res={res_real:.3e})",
                              typeMsg="i")
                        break
            else:
                x_cur = anchor.copy()
                self._recompute_correction()

            self._checkpoint_plotting_artifacts()

            res_cur = np.sqrt(max(f_cur, 0.0)) / self.n_flux
            if res_cur < o["res_tol"]:
                print(f"SALM converged (flux residual): res={res_cur:.3e} < {o['res_tol']:.0e}",
                      typeMsg="i")
                break

            # diminishing-returns early exit: when ACCEPTED steps yield less than
            # plateau_rel_tol per-eval geometric improvement in the best flux_sq over the
            # last plateau_window evals, the remaining budget buys almost nothing -> stop.
            # Gated on `accepted` so a reject-recovery (TR still adjusting / about to reset)
            # is NOT cut short; the reject-stuck case is caught by the TR-floor stop below.
            best_hist.append(f_cur)
            w = int(o.get("plateau_window", 0) or 0)
            if (w > 0 and accepted and len(best_hist) > w + 1
                    and real_evals >= plateau_guard):
                f_then, f_now = best_hist[-w - 1], max(best_hist[-1], 0.0)
                rate = (1.0 - (f_now / f_then) ** (1.0 / w)) if f_then > 0 else 0.0
                if rate < o["plateau_rel_tol"]:
                    if tr_resets_used < o["tr_resets"]:
                        # ESCAPE PROBE first: a plateau can be a collapsed-small TR, not
                        # the basin floor -- the TR can settle well above its floor with
                        # the reset budget still untouched. Spend a reset -- TR back to
                        # tr_init + full GP refit -- so the solver gets one big-step
                        # attempt out of the micro-step regime; only stop if the plateau
                        # survives a fresh window after the probe.
                        tr_resets_used += 1
                        self._tr_rel = float(np.clip(o["tr_init"], o["tr_min"],
                                                     o["max_total_rel_step"]))
                        self._fit_surrogates(force_full=True)
                        self._recompute_correction()
                        plateau_guard = real_evals + w
                        print(f"SALM plateau ({rate:.2%}/eval < "
                              f"{o['plateau_rel_tol']:.2%}): TR reset "
                              f"#{tr_resets_used}/{o['tr_resets']} as ESCAPE PROBE "
                              f"before stopping (tr -> {self._tr_rel:.2e})", typeMsg="w")
                        x_cur = self._anchor_x_opt.copy()
                        continue
                    print(f"SALM stopping (diminishing returns, resets exhausted): best "
                          f"flux_sq improving {rate:.2%}/eval < "
                          f"{o['plateau_rel_tol']:.2%} over last {w} evals "
                          f"(res={res_cur:.3e})", typeMsg="i")
                    break

            if real_evals >= o["max_real_evals"]:
                print(f"SALM: reached evaluation budget ({real_evals} evals)", typeMsg="i")
                break
            if (not accepted) and self._tr_rel <= o["tr_min"] * (1.0 + 1e-9):
                if tr_resets_used < o["tr_resets"]:
                    tr_resets_used += 1
                    self._tr_rel = float(np.clip(o["tr_init"], o["tr_min"], o["max_total_rel_step"]))
                    print(f"SALM: TR hit floor; reset #{tr_resets_used}/{o['tr_resets']} "
                          f"(full GP refit)", typeMsg="w")
                    self._fit_surrogates(force_full=True)
                    self._recompute_correction()
                else:
                    print(f"SALM stalled (TR at floor {self._tr_rel:.2e}, resets exhausted)",
                          typeMsg="w")
                    break

        i_best = int(np.argmin(f_hist))
        return x_hist[i_best], f_hist, x_hist

    # ==================================================================
    # Plotting shim: write optimization_object.pkl / _results.out / _extra.pkl
    # ==================================================================

    def _checkpoint_plotting_artifacts(self):
        try:
            self._write_plotting_artifacts(verify=False)
        except Exception as e:
            print(f"\t- SALM: iteration checkpoint of plotting artifacts failed "
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
        print("\t- SALM: writing MITIM-compatible plotting artifacts...", typeMsg="i")

        bounds = OrderedDict(
            (dv, [float(self.dvs_min[i]), float(self.dvs_max[i])])
            for i, dv in enumerate(self.dvs))

        m = MITIM_BO(self.fun, onlyInitialize=True, askQuestions=False, cold_start=False)
        m.train_X, m.train_Y, m.train_Ystd = self.train_X, self.train_Y, self.train_Ystd
        m.dfT = self.dfT
        m.cold_start = False
        m.outputs = list(self.ofs)
        m.bounds = bounds
        m.bounds_orig = copy.deepcopy(bounds)
        m.scalarized_objective = self.fun.scalarized_objective
        m.steps = [_SALMStep(
            {"individual_models": self.gp_individual, "combined_model": self.gp_combined},
            self.train_X, self.train_Y, self.train_Ystd, bounds)]

        res_f = []
        for k in range(n):
            with torch.no_grad():
                _, _, R = self._compose(
                    torch.from_numpy(self.train_Y[k]).to(self.dfT).unsqueeze(0),
                    torch.from_numpy(self.train_X[k]).to(self.dfT))
            res_f.append(self._flux_sq(R.cpu().numpy()))
        ibest = int(np.argmin(res_f))
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

        dictStore = {}
        missing = []
        for i in range(n):
            pkl = self._powerstate_pkl_path(i)   # seed -> init folder, loop -> Execution
            if pkl.exists():
                dictStore[i] = {"powerstate": STATEtools.read_saved_state(pkl)}
            else:
                dictStore[i] = np.nan
                missing.append(i)
        dictStore[n] = np.nan
        if missing:
            print(f"\t- SALM: WARNING missing powerstate pkls for evals {missing}; "
                  f"plotting will truncate at the first gap", typeMsg="w")
        m.optimization_extra = self.folderOutputs / "optimization_extra.pkl"
        self.fun.optimization_extra = m.optimization_extra
        with open(m.optimization_extra, "wb") as h:
            pickle_dill.dump(dictStore, h, protocol=4)

        m.save()
        if verify:
            self._verify_plotting_artifacts()

    def _verify_plotting_artifacts(self):
        from mitim_tools.opt_tools.STRATEGYtools import read_from_scratch
        from mitim_tools.opt_tools.utils.BOgraphics import optimization_results
        try:
            mm = read_from_scratch(self.folderOutputs / "optimization_object.pkl")
            _ = mm.steps[-1].GP["combined_model"]
            _ = mm.optimization_object.surrogate_parameters["powerstate"]
            _ = IOtools.unpickle_mitim(mm.optimization_object.optimization_extra)
            rr = optimization_results(file=self.folderOutputs / "optimization_results.out")
            rr.readClass(mm)
            rr.read()
            if rr.best_absolute_index is None:
                raise RuntimeError("optimization_results.read() could not determine best_absolute_index")
            print("\t- SALM: plotting artifacts verified (mitim_plot_portals_edge ready)",
                  typeMsg="i")
        except Exception as e:
            print(f"\t- SALM: WARNING plotting artifacts failed read-back ({type(e).__name__}: {e})",
                  typeMsg="w")
