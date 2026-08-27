"""
edge_tools.uq.driver
---------------------
Forward driver: turn a flat leaf vector ``x0`` (from ``UQInputs.build_x0``) into
the flux-match residual vector, scattering each ``ScatterSpec`` back into a
working powerstate, then push the input covariance factor through it.

Two propagation routes, chosen per input by ``ScatterSpec.kind``:

  * TORCH route (jvp) -- inputs that reach the residual through a fully
    differentiable path: LCFS separatrix values (bc_tensors) and the analytic
    rotation Vtor.  These columns are pushed matrix-free with
    ``propagate.push_columns``.

  * BLACK-BOX route (finite difference / surrogate) -- inputs that pass through a
    graph-breaking external model: the neutral source rate, the impurity source
    rate, D_z, V0 (Aurora / neutral models return numpy).  These columns are
    formed by perturbing the scalar option, resetting the cached model instance,
    re-running ``calculate()``, and differencing.  This is an *offline / occasional*
    cost, not per-iteration -- or replace it with a differentiable surrogate whose
    autograd Jacobian folds into the torch route.

The residual is signed ``r = P - P_tr`` (target minus transport) at the flux-match
control points; carrying it as a factor preserves the target<->transport
correlation (both are functions of the same profiles).
"""

import copy
import shutil
from typing import Dict, List, Optional, Sequence

import torch

from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools.edge_tools.uq import propagate
from mitim_tools.edge_tools.uq.inputs import ScatterSpec


# Inputs whose derivative can be taken with autograd/jvp end-to-end (linear mode).
_TORCH_KINDS = {"lcfs", "vtor"}
# Inputs that traverse a graph-breaking external model / a lazily-rebuilt model
# option (scanned through the real chain).
_BLACKBOX_KINDS = {"source", "lcfs_bc"}
# ScatterSpec.kind -> (option-dict attribute, model-instance attribute to reset)
_OPTION_ROUTES = {
    "cs":      ("_cs_model_options",  "_cs_model_instance"),
    "neutral": ("_neu_model_options", "_neu_model_instance"),
    "bc":      ("_bc_model_options",  "_bc_model_instance"),
}


class UQState:
    """
    Bind a powerstate + flux-match DVs + input layout, and expose the residual as
    a function of the flat leaf vector ``x0``.

    Parameters
    ----------
    powerstate : powerstate_edge
        A state that has been through at least one ``calculate()`` so that
        ``bc_tensors``, options, and the fine grid exist.  The driver operates on
        a deep copy so the caller's solver state is never mutated.
    X_dvs : torch.Tensor
        The flux-match degrees of freedom (concatenated ``aL*``) held fixed while
        input uncertainty is propagated -- typically the matched solution
        ``FluxMatch_Xopt``.
    layout : dict
        The ``layout`` returned by ``UQInputs.build_x0``.
    batch : int
        Batch row the UQ is evaluated on.
    """

    def __init__(self, powerstate, X_dvs, layout: Dict, batch: int = 0,
                 stacked: bool = False, rotation_proxy: bool = True,
                 transport_proxy=None, folder=None):
        self.ps0 = powerstate
        self.X_dvs = X_dvs.detach()
        self.layout = layout
        self.batch = batch
        self.scatters: List[ScatterSpec] = [v["scatter"] for v in layout.values()]
        # When the run uses the vgen rotation backend (non-differentiable NEO w0),
        # force the differentiable analytic backend on the UQ forward so vtor
        # uncertainty propagates on the torch route (analytic-as-proxy for vgen).
        # Captures input-induced rotation uncertainty, not the ~7% vgen-vs-analytic
        # model discrepancy -- add a fixed epistemic term for that if wanted.
        self.rotation_proxy = rotation_proxy
        # Robust-outlier factor for garbage-corner screening (None disables).
        self.outlier_factor = 8.0
        # Absolute companion gate: a column must exceed this multiple of the median
        # residual response before it can be dropped. Guards against MAD collapse
        # when most inputs are inert -- see _screen_garbage_cols.
        self.outlier_min_ratio = 3.0
        # Smallest fraction of a requested theta direction that may survive the
        # parameterizer's conditioning and still be extrapolated back to 1 sigma.
        # Below it the applied perturbation is a different direction, so 1/a_eff
        # would manufacture a huge sensitivity out of an unrelated response.
        self.theta_proj_floor = 0.1
        # CRITICAL for in-loop use: the UQ forward re-runs powerstate.calculate()
        # on copies.  If the transport evaluator is the real code (TGLF/NEO) that
        # re-runs it many times per pass -- unusable.  ``transport_proxy`` is a
        # callable(ps) -> None that swaps in a fast DIFFERENTIABLE transport on
        # the UQ copy (e.g. the flux GP surrogate, or fingerprints_analytic).
        # None is fine only when the configured transport is already analytic
        # (fingerprints) or when propagating targets-only.
        self.transport_proxy = transport_proxy
        # When True the forward returns cat([P (target), P_tr (transport)]) so
        # the two nodes can be split for per-node std injection; the residual
        # factor is then their difference (correlation preserved).  When False it
        # returns the signed residual P - P_tr directly.
        self.stacked = stacked
        # Base folder for the real-model transport re-runs.  Each evaluation gets
        # a UNIQUE subfolder ("uq_<tag>") so the remote scratch name (a
        # deterministic hash of the LOCAL run path, IOtools.path_overlapping) is
        # unique per column/pass.  Without this the re-runs default to
        # calculate(folder="~/scratch/"), giving one CONSTANT remote folder shared
        # by every UQ transport run across columns, evals AND concurrent discharge
        # optimizations -> tarball collisions ("Not all received").  None keeps the
        # legacy default (fine only when transport is a proxy / no remote runs).
        self.folder = folder

    # ------------------------------------------------------------------ #
    # Scatter x0 back into a (fresh) powerstate
    # ------------------------------------------------------------------ #

    def _scatter_torch(self, ps, x0: torch.Tensor):
        """
        Write the differentiable (torch-route) inputs from x0 into ``ps`` while
        preserving the autograd graph.  bc_tensors[ch]["val"] is *rebuilt* from x0
        slices (not written in place) so the graph x0 -> LCFS -> profiles -> flux
        stays intact.
        """
        for sc in self.scatters:
            if sc.kind == "lcfs":
                val = sc.to_value(x0)                     # (1,) physical space
                entry = ps.bc_tensors[sc.key]
                new_val = entry["val"].clone()
                new_val = new_val.index_put(
                    (torch.tensor([sc.batch]), torch.tensor([0])), val.reshape(())
                )
                ps.bc_tensors[sc.key] = {**entry, "val": new_val}
            elif sc.kind == "vtor":
                self._scatter_vtor(ps, sc, x0)

    def _scatter_vtor(self, ps, sc: ScatterSpec, x0: torch.Tensor):
        """Write Vtor into rotation options (fixed_pairs knots or global mult)."""
        opts = dict(getattr(ps, "_rotation_options", {}) or {})
        vals = sc.to_value(x0)
        if sc.meta.get("mode") == "fixed_pairs":
            knots = sc.meta.get("knots") or []
            opts["vtor_source"] = "user"
            opts["vtor_pairs"] = [
                (float(r), vals[j]) for j, (r, _v) in enumerate(knots)
            ]
        else:  # initial_gacode: single global multiplier
            opts["vtor_source"] = "extract_initial"
            opts["vtor_multiplier"] = vals.reshape(())
        ps._rotation_options = opts

    def _scatter_blackbox(self, ps, x0: torch.Tensor):
        """
        Write scalar model options (source rates, D_z, V0, and LCFS values that
        live in the BC-model options) as *floats* and reset the corresponding
        cached model instance so the new value takes effect on the next
        calculate() -- and, for LCFS, survives calculateBoundaryConditions, which
        rebuilds bc_tensors from _bc_model_options each call.
        """
        for sc in self.scatters:
            # Correlated aLy perturbation (PeretSSF-projected covariance direction):
            # add amplitude*delta[ch] to _bc_model_options[aL{ch}] for all channels
            # at once, so a Fixed-nominal aLy still carries the PeretSSF foot cov.
            if sc.kind == "aly_cov":
                amp = float(sc.to_value(x0))
                if amp == 0.0:
                    continue
                opts = dict(getattr(ps, "_bc_model_options", {}) or {})
                nominal = sc.meta.get("nominal", {})
                for ch, d in sc.meta.get("delta", {}).items():
                    k = f"aL{ch}"
                    # Seed from the current option value, else the nominal aLy BC
                    # captured at build time.  Without the nominal fallback a
                    # Fixed-nominal aLy with EMPTY _bc_model_options never has the
                    # aL{ch} key, so the perturbation was silently dropped and the
                    # PeretSSF-projected aLy uncertainty never reached the fit.
                    base = opts.get(k, nominal.get(ch))
                    if base is not None:
                        opts[k] = base + amp * d
                ps._bc_model_options = opts
                ps._bc_model_instance = None
                continue
            if sc.kind not in _BLACKBOX_KINDS:
                continue
            val = float(sc.to_value(x0))
            target = sc.meta.get("option_target", "cs")
            optkey = sc.meta.get("option_key", "source_rate")
            opt_attr, inst_attr = _OPTION_ROUTES.get(target, _OPTION_ROUTES["cs"])
            current = dict(getattr(ps, opt_attr, {}) or {})
            current[optkey] = val
            setattr(ps, opt_attr, current)
            setattr(ps, inst_attr, None)      # force lazy rebuild with new value

    # ------------------------------------------------------------------ #
    # Residual evaluation
    # ------------------------------------------------------------------ #

    def _prep_ps(self, ps):
        """Apply forward-pass overrides to a fresh powerstate copy."""
        if self.rotation_proxy:
            opts = dict(getattr(ps, "_rotation_options", {}) or {})
            if opts.get("mode") == "vgen":
                opts["mode"] = "analytic"   # differentiable backend for propagation
                ps._rotation_options = opts
        if self.transport_proxy is not None:
            self.transport_proxy(ps)        # swap in fast differentiable transport
        return ps

    def _run_kwargs(self, tag=None) -> dict:
        """calculate() folder/name kwargs giving this eval a UNIQUE run path when
        ``self.folder`` is set (so the remote scratch hash is unique per eval);
        empty dict (legacy ``~/scratch/`` default) when it is not.

        The folder is CLEARED if it already exists.  The transport drivers read back
        whatever ``out.tglf.*`` / ``out.neo.*`` they find in the run folder, so a
        second UQ pass into the same ``edge_uq_best`` directory writes fresh inputs
        and then silently reads the PREVIOUS pass's outputs.  That happened on the
        bo_prescribed_nZ runs: the 14:29 pass wrote input.tglf_* at 14:29 and read
        out.tglf.* dated 12:58, producing a band whose profile columns and flux
        columns came from different evaluations (and finishing in 1-5 min instead of
        ~50, which is how it was spotted).  Each ``uq_<tag>`` folder belongs to
        exactly one column of one pass, so clearing it is safe and is the only way to
        guarantee the fluxes match the profiles.
        """
        if self.folder is None:
            return {}
        sub = "base" if tag is None else str(tag)
        run_folder = IOtools.expandPath(self.folder) / f"uq_{sub}"
        if run_folder.exists():
            shutil.rmtree(run_folder, ignore_errors=True)
        run_folder.mkdir(parents=True, exist_ok=True)
        return dict(folder=run_folder, nameRun=f"uq_{sub}",
                    evaluation_number=(tag if isinstance(tag, int) else 0))

    def _residual(self, ps, tag=None) -> torch.Tensor:
        """
        Run calculate() at the fixed DVs.  Returns the signed residual P - P_tr,
        or cat([P, P_tr]) when ``self.stacked`` (so target/transport can be split
        downstream).  The two halves of the stacked output share input columns, so
        differencing their factors preserves the target<->transport correlation.

        ``tag`` selects a unique run subfolder (see ``_run_kwargs``) so concurrent
        transport re-runs never share a remote scratch folder.
        """
        P_tr, P, _S, _res = ps.calculate(self.X_dvs, **self._run_kwargs(tag))
        P, P_tr = P.reshape(-1), P_tr.reshape(-1)
        if self.stacked:
            return torch.cat([P, P_tr])
        return P - P_tr      # signed target - transport

    def n_half(self, x0) -> int:
        """Length of one node (target or transport) in a stacked output."""
        return self._eval_at(x0).numel() // (2 if self.stacked else 1)

    def make_torch_fwd(self, x0_ref: torch.Tensor):
        """
        Build ``fwd(x0) -> residual`` over the torch-route inputs, with black-box
        inputs frozen at their x0_ref values.  Suitable for ``push_columns``.
        """
        ps_base = self._prep_ps(copy.deepcopy(self.ps0))
        self._scatter_blackbox(ps_base, x0_ref)   # freeze scalars once

        def fwd(x0):
            ps = copy.deepcopy(ps_base)
            self._scatter_torch(ps, x0)
            return self._residual(ps, tag="torch")

        return fwd

    # ------------------------------------------------------------------ #
    # Propagation
    # ------------------------------------------------------------------ #

    def propagate_torch(self, x0: torch.Tensor, L: torch.Tensor,
                        torch_cols: Optional[Sequence[int]] = None):
        """
        Push the torch-route columns of ``L`` through the differentiable residual.

        Parameters
        ----------
        x0, L : from UQInputs.build_x0 / cholesky_columns.
        torch_cols : indices of L's columns to push via jvp.  If None, inferred
            from the layout (columns whose owning entry is a _TORCH_KINDS input).

        Returns
        -------
        r0 : residual at the mean.
        L_out : output factor over the pushed columns, (n_res, len(torch_cols)).
        """
        if torch_cols is None:
            torch_cols = self._infer_columns(_TORCH_KINDS)
        fwd = self.make_torch_fwd(x0)
        Lt = L[:, torch_cols]
        r0, L_out = propagate.push_columns(fwd, x0, Lt, batched=False)
        print(f"[UQ] pushed {Lt.shape[1]} torch-route columns -> "
              f"residual factor {tuple(L_out.shape)}", typeMsg="i")
        return r0, L_out

    def finite_difference_columns(self, x0: torch.Tensor, L: torch.Tensor,
                                  bb_cols: Optional[Sequence[int]] = None,
                                  rel_step: float = 1.0, mode: str = "forward"):
        """
        Form black-box residual columns by finite-differencing the external model
        (Aurora / neutral) directly -- no offline surrogate needed.

        For each black-box column ``c`` of ``L`` (a scaled input direction), the
        result is J*L[:, c] (a covariance column):
          * mode="forward"  : (r(x0 + h d) - r0) / h   -- 1 eval/column + 1 shared
            baseline = ``n+1`` external evals total.  Cheapest; Aurora ~1 s each.
          * mode="central"  : (r(x0 + h d) - r(x0 - h d)) / 2h -- ``2n`` evals,
            more accurate for larger sigma / mild nonlinearity.
        ``h = rel_step``.

        Aurora rebuilds each call (``_cs_model_instance`` is reset in
        ``_scatter_blackbox``), so this reads the true code sensitivity.  Keep the
        UQ pass to once per accepted evaluation to bound the cost.
        """
        if bb_cols is None:
            bb_cols = self._infer_columns(_BLACKBOX_KINDS)
        if not bb_cols:
            return torch.zeros((self._n_res(x0), 0), dtype=x0.dtype)

        cols = []
        if mode == "forward":
            r0 = self._eval_at(x0, tag="base")           # shared baseline
            for c in bb_cols:
                r_plus = self._eval_at(x0 + rel_step * L[:, c], tag=int(c))
                cols.append((r_plus - r0) / rel_step)
            n_eval = len(bb_cols) + 1
        elif mode == "central":
            for c in bb_cols:
                r_plus = self._eval_at(x0 + rel_step * L[:, c], tag=f"{int(c)}p")
                r_minus = self._eval_at(x0 - rel_step * L[:, c], tag=f"{int(c)}m")
                cols.append((r_plus - r_minus) / (2.0 * rel_step))
            n_eval = 2 * len(bb_cols)
        else:
            raise ValueError(f"unknown fd mode {mode!r} (use 'forward'|'central')")

        L_bb = torch.stack(cols, dim=1)
        print(f"[UQ] finite-differenced {len(bb_cols)} black-box columns "
              f"({mode}, {n_eval} external evals)", typeMsg="i")
        return L_bb

    def scan_columns_real(self, x0: torch.Tensor, L: torch.Tensor,
                          baseline: Optional[torch.Tensor] = None,
                          cols: Optional[Sequence[int]] = None,
                          rel_step: float = 1.0):
        """
        Real-model input scan: forward-difference EVERY input column through the
        full ``calculate()`` (real TGLF + NEO + Aurora), the direct analogue of
        TGLF's ``use_scan_trick_for_stds`` but perturbing the uncertain *inputs*
        (LCFS ne/te/ti, source rates, D_z, V0) instead of the gradient DVs.

        Because one real evaluation returns the stacked ``[P (target), P_tr
        (transport)]``, each input costs a SINGLE real eval and captures both the
        target and transport sensitivity at once.  With ``baseline`` supplied
        (reuse the evaluation that just ran), the cost is exactly ``n_inputs``
        extra evals per pass.

        ``rel_step`` is in units of the input sigma (columns of ``L`` are already
        sigma-scaled): 1.0 probes the 1-sigma "corner", smaller is more linear.

        Returns (baseline_output, L_out) with L_out the stacked residual factor.
        Use with ``stacked=True`` so target/transport can be split downstream.
        """
        if cols is None:
            cols = list(range(L.shape[1]))
        r0 = baseline if baseline is not None else self._eval_at(x0, tag="base")
        out_cols = []
        for c in cols:
            r_plus = self._eval_at(x0 + rel_step * L[:, c], tag=int(c))
            out_cols.append((r_plus - r0) / rel_step)
        L_out = torch.stack(out_cols, dim=1) if out_cols else \
            torch.zeros((r0.numel(), 0), dtype=x0.dtype)
        print(f"[UQ] real-model input scan: {len(cols)} columns "
              f"({len(cols)} extra real evals, baseline reused={baseline is not None})",
              typeMsg="i")
        return r0, L_out

    def stacked_baseline_from_plasma(self, powerstate=None) -> torch.Tensor:
        """
        Read the just-computed stacked ``[P, P_tr]`` baseline straight from the
        powerstate plasma (populated by the calculate() that already ran), so the
        real-model scan does not repeat it.  Requires ``stacked=True``.
        """
        ps = powerstate if powerstate is not None else self.ps0
        P = ps.plasma["P"].reshape(-1)
        P_tr = ps.plasma["P_tr"].reshape(-1)
        return torch.cat([P, P_tr]) if self.stacked else (P - P_tr)

    def _theta_request(self, x0: torch.Tensor) -> Dict[str, "object"]:
        """{prof: theta offset} REQUESTED by the fit-error columns at ``x0``."""
        import numpy as np
        offsets = {}
        for sc in self.scatters:
            if sc.kind != "theta_pert":
                continue
            amp = float(sc.to_value(x0))
            if amp == 0.0:
                continue
            prof = sc.meta["prof"]
            direction = np.asarray(sc.meta["direction"], dtype=float)
            offsets[prof] = offsets.get(prof, 0.0) + amp * direction
        return offsets

    def _scatter_theta(self, ps, x0: torch.Tensor):
        """Set parameterizer._theta_offset from fit-error theta-perturbation columns
        (accumulate per channel; scan activates one at a time).

        Also clears ``_theta_offset_applied`` so ``_applied_theta_scale`` can never
        read a record left by a PREVIOUS evaluation (the parameterizer is deep-copied
        per eval, but the copy inherits whatever the source object last stored)."""
        par = getattr(ps, "parameterizer", None)
        if par is None:
            return
        offsets = self._theta_request(x0)
        par._theta_offset_applied = {}
        if offsets:
            cur = dict(getattr(par, "_theta_offset", {}) or {})
            cur.update(offsets)
            par._theta_offset = cur

    def _applied_theta_scale(self, ps, requested) -> float:
        """
        Effective amplitude actually reconstructed, relative to what was requested.

        ``SplineMtanhAnalytic._apply_theta_offset`` conditions a UQ theta sample
        before rebuilding: it clips components back inside the theta bounds and
        bisects the amplitude down until the sample is peak- and shape-feasible.  The
        reconstruction therefore sees ``theta + delta_applied``, which is neither the
        requested magnitude nor (once a component clips) the requested DIRECTION.

        Dividing the observable difference by the requested step -- which is what the
        scan used to do -- reports ``d obs / d(requested amplitude)`` as if the full
        step had been taken.  Measured on the bo_prescribed_nZ runs the surviving
        fraction ranged from 0.03 to 1.0 ACROSS COLUMNS OF ONE SCAN, so the fit-error
        stds were inflated by up to ~30x and were not reproducible between runs (the
        LM warm start moves the feasibility boundary).

        Returns the least-squares projection of the applied offset onto the requested
        one, i.e. the amplitude of the component the caller actually asked for.  0.0
        means the sample was conditioned away entirely and the column carries no
        information.
        """
        import numpy as np
        par = getattr(ps, "parameterizer", None)
        applied_all = getattr(par, "_theta_offset_applied", None) if par is not None else None
        if not requested:
            return 1.0
        if not isinstance(applied_all, dict) or not applied_all:
            # Parameterizer does not report (e.g. a non-SplineMtanhAnalytic model):
            # assume the offset was taken verbatim, as before.
            return 1.0
        num = den = 0.0
        for prof, req in requested.items():
            req = np.asarray(req, dtype=float).reshape(-1)
            app = np.asarray(applied_all.get(prof, np.zeros_like(req)),
                             dtype=float).reshape(-1)
            n = min(req.size, app.size)
            num += float(np.dot(app[:n], req[:n]))
            den += float(np.dot(req[:n], req[:n]))
        if den <= 0.0:
            return 1.0
        return num / den

    def _profile_probe(self, x0: torch.Tensor, key: str, return_ps: bool = False):
        """Cheap profile-only response (NO transport): scatter x0, run the BC +
        reconstruction + profile functions, and return plasma[key].  Used to pick
        a per-direction fit-error step without paying for TGLF.

        With ``return_ps`` the evaluated copy is returned alongside the value so the
        caller can read back how much of the requested theta offset survived
        conditioning (see :meth:`_applied_theta_scale`)."""
        try:
            ps = self._prep_ps(copy.deepcopy(self.ps0))
            self._scatter_blackbox(ps, x0)
            self._scatter_torch(ps, x0)
            self._scatter_theta(ps, x0)
            with torch.no_grad():
                ps.calculateBoundaryConditions()
                ps.modify(self.X_dvs)
                ps.calculateProfileFunctions()
            val = ps.plasma[key].reshape(-1).clone() if key in ps.plasma else None
        except Exception:
            val, ps = None, None
        return (val, ps) if return_ps else val

    def _column_probe_keys(self, x0, L, c, keys):
        """Observable keys to judge column ``c`` on.

        A fit-error (theta) column perturbs ONE channel, so it is judged on that
        channel's gradient.  Judging every column on one global key -- the old
        behaviour, with ``aLne`` hard-wired as the default -- makes each te/ti theta
        direction probe a profile it cannot move: the response is identically zero,
        which the linearity test reads as PERFECTLY LINEAR and rewards with the full
        1-sigma step.  That is how the te/ti fit-error columns came to be evaluated at
        a corner nobody had checked.

        Non-theta columns (e.g. the PeretSSF-projected aLy directions) move every
        channel at once, so they are judged on all of ``keys``.
        """
        plasma = getattr(self.ps0, "plasma", {})
        req = self._theta_request(x0 + L[:, c])
        picked = []
        for prof in req:
            for cand in (f"aL{prof}", prof):
                if cand in plasma:
                    picked.append(cand)
                    break
        if picked:
            return picked
        return [k for k in keys if k in plasma]

    def fit_step_map(self, x0, L, cols, key="aLne", small_step=0.1,
                     full_step=1.0, nonlin_tol=3.0, min_step=0.01, keys=None,
                     applied_tol=0.02):
        """
        Per fit-error (theta) column, pick the largest step (<= ``full_step``) that is
        a valid LINEARIZATION of that column's direction, so the flux is measured with
        signal >> noise and with no 1/step amplification.

        Each column is judged on the channel it actually moves (see
        :meth:`_column_probe_keys`) against two conditions:

        * the conditioner must be a NO-OP -- ``_apply_theta_offset`` clips the sample
          back inside the theta bounds and bisects its amplitude until the
          reconstruction is peak/shape feasible.  Whenever it bites, the profile that
          gets evaluated is not the requested direction at the requested amplitude, so
          the difference quotient is not a directional derivative of anything.
          Shrinking the step until the sample is feasible on its own makes the
          conditioner inactive and the column an honest derivative.  Measured on the
          bo_prescribed_nZ H-modes, the surviving fraction at the full step ranged
          from 0.02 to 1.0 across columns of a single scan.
        * linearity -- the full-step response is ~2x the half-step response; a regime
          jump makes it much larger.

        Failing either, the step is halved and retested down to ``min_step``; below
        that the 1/step noise amplification on the real-transport columns costs more
        than the residual conditioning bias.  Columns still conditioned at the floor
        are reported and attributed by projection (:meth:`_applied_theta_scale`).

        Only theta columns are treated this way.  Directions that do not pass through
        the theta conditioner (e.g. the PeretSSF-projected aLy columns) are NOISE
        limited at small step -- the LM re-fit moves the mtanh backbone by more than
        the perturbation does, so the difference quotient diverges as the step shrinks
        -- and are left at the full step.
        """
        keys = list(keys) if keys else [key]
        steps, n_full, constrained = {}, 0, []
        for c in cols:
            d = L[:, c]
            if not self._theta_request(x0 + d):
                steps[int(c)] = float(full_step)      # not a theta column
                n_full += 1
                continue
            ckeys = self._column_probe_keys(x0, L, c, keys)
            _v, ps_b = self._profile_probe(x0, ckeys[0], return_ps=True) if ckeys else (None, None)
            base = self._observables(ps_b, ckeys) if ps_b is not None else {}
            if not base:
                steps[int(c)] = small_step
                continue
            step = float(full_step)
            while True:
                xh, xf = x0 + 0.5 * step * d, x0 + step * d
                # One reconstruction per point, all keys read off it -- probing per
                # key would deep-copy and rebuild the powerstate once per key.
                _v, ps_h = self._profile_probe(xh, ckeys[0], return_ps=True)
                _v, ps_f = self._profile_probe(xf, ckeys[0], return_ps=True)
                ph = self._observables(ps_h, base) if ps_h is not None else {}
                pf = self._observables(ps_f, base) if ps_f is not None else {}
                if any(k not in ph for k in base) or any(k not in pf for k in base):
                    step = small_step
                    break
                a_eff = self._applied_theta_scale(ps_f, self._theta_request(xf))
                rh = max(float((ph[k] - base[k]).abs().max()) for k in base)
                rf = max(float((pf[k] - base[k]).abs().max()) for k in base)
                clean = abs(a_eff - 1.0) <= applied_tol
                linear = rf <= nonlin_tol * (2.0 * rh + 1e-12)
                if (clean and linear) or step <= min_step:
                    if not clean:
                        constrained.append((int(c), a_eff))
                    break
                step *= 0.5
            steps[int(c)] = step
            n_full += int(step >= full_step)
        detail = ", ".join(f"col{int(c)}={steps[int(c)]:.3g}" for c in cols)
        print(f"[UQ] fit-error step selection: {n_full}/{len(cols)} directions take "
              f"the full step; per-column steps: {detail}", typeMsg="i")
        if constrained:
            print(f"[UQ] theta directions still conditioned at the minimum step "
                  f"{min_step:g} -- the theta BOUNDS clip them, so the requested "
                  "direction is unreachable at any amplitude: "
                  + ", ".join(f"col{c} surviving fraction {a:.3g}" for c, a in constrained)
                  + "; attributed by projection onto the requested direction",
                  typeMsg="w")
        return steps

    def _eval_full(self, x0: torch.Tensor, tag=None):
        """Evaluate at x0 and return (residual, evaluated_powerstate).  ``tag``
        selects a unique transport run subfolder (see ``_run_kwargs``)."""
        with torch.no_grad():
            ps = self._prep_ps(copy.deepcopy(self.ps0))
            self._scatter_blackbox(ps, x0)
            self._scatter_torch(ps, x0)
            self._scatter_theta(ps, x0)
            res = self._residual(ps, tag=tag)
            return res, ps

    def _eval_at(self, x0: torch.Tensor, tag=None) -> torch.Tensor:
        """Full residual at x0 (scatters both routes); no graph kept."""
        return self._eval_full(x0, tag=tag)[0]

    def _n_res(self, x0) -> int:
        return self._eval_at(x0, tag="nres").numel()

    @staticmethod
    def _observables(ps, keys):
        """Snapshot flattened plasma observables (te/ti/ne/aLte/...) from a ps."""
        return {k: ps.plasma[k].reshape(-1).clone()
                for k in keys if k in ps.plasma}

    def scan_with_observables(self, x0, L, obs_keys, baseline=None,
                              cols=None, rel_step=1.0,
                              small_step_cols=None, small_step=0.1,
                              col_steps=None):
        """
        Real-model input scan that ALSO propagates a set of profile observables
        (e.g. te, ti, ne, aLte, aLti, aLne, w0) through the same perturbed evals
        -- no extra model calls beyond ``scan_columns_real``.

        Columns in ``small_step_cols`` (the mtanh fit-error theta-directions) are
        evaluated at a SMALL step and linearly extrapolated to 1-sigma, so their
        LINEAR sensitivity is captured rather than a full-corner step that can jump
        a reconstruction regime boundary (e.g. c crossing 1) discontinuously.  The
        fit covariance is a linearization by construction, so this is the correct
        propagation; input columns keep the full ``rel_step`` corner.

        Returns (r0, L_res, obs0, L_obs) where obs0[k] is the baseline observable
        vector and L_obs[k] its covariance factor; the marginal std is
        ``std_from_factor(L_obs[k])``, reshapeable back to plasma[k].shape.
        """
        if cols is None:
            cols = list(range(L.shape[1]))
        small = set(small_step_cols or [])
        col_steps = col_steps or {}
        # baseline: reuse the eval that just ran (residual from plasma, obs from ps0)
        if baseline is not None:
            r0 = baseline
            obs0 = self._observables(self.ps0, obs_keys)
        else:
            r0, ps0 = self._eval_full(x0, tag="base")
            obs0 = self._observables(ps0, obs_keys)

        res_cols = []
        obs_cols = {k: [] for k in obs0}
        conditioned, suppressed = [], []
        for c in cols:
            # per-column step: explicit map wins; else small-step set; else full
            step = col_steps.get(int(c), small_step if c in small else rel_step)
            xc = x0 + step * L[:, c]
            requested = self._theta_request(xc)
            r_plus, ps_plus = self._eval_full(xc, tag=int(c))
            # EFFECTIVE step.  A mtanh fit-error column asks the parameterizer for a
            # theta offset, which conditions it (bounds clip + feasibility bisection)
            # before rebuilding -- so the profile that was just evaluated corresponds
            # to a FRACTION of the requested perturbation.  Divide by what the model
            # applied, not by what was asked for, or the column is scaled by 1/alpha.
            a_eff = self._applied_theta_scale(ps_plus, requested)
            eff = step * a_eff
            if requested and abs(a_eff - 1.0) > 1e-6:
                conditioned.append((int(c), a_eff))
            if abs(a_eff) < self.theta_proj_floor:
                # Conditioned (almost) away: the requested direction is unreachable,
                # and rescaling by 1/a_eff would attribute a response measured along a
                # DIFFERENT direction to a large amplitude of this one -- the failure
                # that inflated the mtanh-fit contribution.  A zero column UNDER-counts
                # this source, which is honest and is reported below.
                res_cols.append(torch.zeros_like(r_plus))
                for k in obs0:
                    obs_cols[k].append(torch.zeros_like(obs0[k]))
                suppressed.append((int(c), a_eff))
                continue
            res_cols.append((r_plus - r0) / eff)
            obsp = self._observables(ps_plus, obs_keys)
            for k in obs0:
                obs_cols[k].append((obsp[k] - obs0[k]) / eff)

        if conditioned:
            print("[UQ] theta columns conditioned by the parameterizer (surviving "
                  "fraction of the requested 1-sigma direction): "
                  + ", ".join(f"col{c}={a:.3g}" for c, a in conditioned)
                  + " -- finite differences divided by the APPLIED perturbation",
                  typeMsg="i")
        if suppressed:
            print("[UQ] ZEROED "
                  + ", ".join(f"col{c} (surviving fraction {a:.3g} < "
                              f"{self.theta_proj_floor:g})" for c, a in suppressed)
                  + ": the parameterizer conditions this direction away, so its "
                    "uncertainty is UNDER-counted rather than extrapolated from a "
                    "perturbation the reconstruction never took", typeMsg="w")

        # Corner screening: a scan corner that drives the reconstruction/TGLF into
        # a pathological regime yields a garbage flux (e.g. Ge_tr jumping 70x),
        # which would inject an absurd std.  Zero any column whose residual
        # response is an extreme robust-outlier (>> median), consistently across
        # the residual and observable factors, and warn.  This under-counts that
        # input's uncertainty (safer than injecting garbage) -- the dropped inputs
        # are reported so they can be investigated.
        dropped = self._screen_garbage_cols(res_cols, obs_cols, cols,
                                            factor=self.outlier_factor)
        self._report_observable_excursions(obs0, obs_cols, cols)

        L_res = torch.stack(res_cols, dim=1) if res_cols else \
            torch.zeros((r0.numel(), 0), dtype=x0.dtype)
        L_obs = {k: torch.stack(v, dim=1) for k, v in obs_cols.items() if v}
        msg = (f"[UQ] real-model scan with {len(obs0)} profile observables "
               f"({len(cols)} real evals, baseline reused={baseline is not None})")
        if dropped:
            msg += f"; DROPPED {len(dropped)} garbage corner(s): {dropped}"
        print(msg, typeMsg="w" if dropped else "i")
        return r0, L_res, obs0, L_obs

    def _report_observable_excursions(self, obs0, obs_cols, cols,
                                      rel_warn=1.0, keys=("te", "ti", "ne")):
        """Warn about columns whose per-1-sigma response to a POSITIVE-DEFINITE
        profile exceeds ``rel_warn`` times the profile itself.

        The garbage screen above looks only at the residual vector, so a column can
        wreck the reconstruction while producing an unremarkable flux residual and
        pass unnoticed -- and it is the profile columns, not the residual, that become
        the plotted bands.  This does not drop anything (with the adaptive step and
        the applied-perturbation denominator it should not fire); it makes the case
        visible instead of silently emitting a std larger than the mean.
        """
        flagged = []
        for k in keys:
            if k not in obs0 or k not in obs_cols:
                continue
            denom = obs0[k].abs().clamp_min(1e-30)
            for i, col in enumerate(obs_cols[k]):
                rel = float((col.abs() / denom).max())
                if rel > rel_warn:
                    cid = int(cols[i]) if i < len(cols) else i
                    flagged.append((cid, k, rel))
        if flagged:
            print("[UQ] columns whose 1-sigma response EXCEEDS the profile itself "
                  "(the band will be wider than the mean there, so the linearization "
                  "is not trustworthy): "
                  + ", ".join(f"col{c} {k} x{r:.2f}" for c, k, r in flagged),
                  typeMsg="w")

    def _screen_garbage_cols(self, res_cols, obs_cols, cols, factor=8.0):
        """Zero columns whose residual response is an extreme robust outlier
        (median-absolute-deviation based).  Returns the dropped column ids.

        The MAD test alone is NOT sufficient. When most inputs are inert (the
        neutral/impurity source columns barely move the turbulent flux within one
        evaluation) the surviving norms cluster, MAD collapses toward zero, and
        `med + factor*mad` degenerates to ~med -- so ANY input with genuine
        sensitivity reads as an extreme outlier no matter how large `factor` is.
        Observed on the NT case: all 12 norms within 1.09x of each other, yet the
        three LCFS boundary conditions -- the most physically influential inputs in
        the set -- were dropped, zeroing every separatrix band.

        `outlier_min_ratio` is the ABSOLUTE guard against that: a column must also
        exceed this multiple of the median before it can be dropped. It is a plain
        ratio, so it does not care how tight the cluster is. The historical value
        was a hard-coded 3.0; anything at or below that is far too aggressive for a
        well-behaved scan (a real pathology is the ~70x Ge_tr excursion this screen
        was written for, not a 3x sensitivity).
        """
        if len(res_cols) < 3 or factor is None or factor <= 0:
            return []
        norms = torch.tensor([float(torch.linalg.vector_norm(c)) for c in res_cols])
        pos = norms[norms > 0]
        if pos.numel() < 3:
            return []
        med = float(pos.median())
        mad = float((pos - med).abs().median()) or float(pos.std()) or med
        thresh = med + factor * mad
        min_ratio = float(getattr(self, "outlier_min_ratio", 3.0) or 3.0)
        dropped = []
        for i, n in enumerate(norms):
            if float(n) > thresh and float(n) > min_ratio * med:
                res_cols[i] = torch.zeros_like(res_cols[i])
                for k in obs_cols:
                    obs_cols[k][i] = torch.zeros_like(obs_cols[k][i])
                dropped.append(int(cols[i]) if i < len(cols) else i)
        return dropped

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _infer_columns(self, kinds) -> List[int]:
        cols = []
        for entry in self.layout.values():
            if entry["scatter"].kind in kinds:
                cols.extend(entry["slots"])
        return cols


def residual_factor(uqstate: UQState, x0: torch.Tensor, L: torch.Tensor,
                    do_blackbox: bool = True, fd_mode: str = "forward"):
    """
    Full residual covariance factor: torch-route (jvp) columns plus, optionally,
    black-box (finite-difference) columns, concatenated so
    ``Sigma_r = L_r @ L_r.T`` with correlations preserved.

    Returns (r0, L_r).
    """
    r0, L_torch = uqstate.propagate_torch(x0, L)
    if do_blackbox:
        L_bb = uqstate.finite_difference_columns(x0, L, mode=fd_mode)
        L_r = torch.cat([L_torch, L_bb], dim=1) if L_bb.shape[1] else L_torch
    else:
        L_r = L_torch
    return r0, L_r
