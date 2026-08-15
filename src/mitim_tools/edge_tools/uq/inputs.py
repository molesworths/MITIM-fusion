"""
edge_tools.uq.inputs
--------------------
Specification of the uncertain inputs and construction of the input Cholesky
factor ``L`` (n_in, k) consumed by ``propagate.push_columns``.

Uncertain inputs for PORTALS-edge (each an assumed 1-sigma):
  * LCFS boundary values ne, Te, Ti          -- user-given relative sigma.
    (LCFS aLy are NOT independent inputs: they are a deterministic function of
    these via the PeretSSF model, so their uncertainty is induced downstream on
    the differentiable graph, correlated with the LCFS values themselves.)
  * Source terms (particle / power / impurity source rate).
  * Toroidal velocity, handled on the *user's actual representation* (see
    ``add_vtor``) -- either "fixed" (r, vtor) knot pairs or an "initial" vtor
    profile from input.gacode.  We do NOT synthesize an abstract quadratic form.

Positive-definite channels (ne, Te, Ti, densities) are parameterized in LOG
space so a relative sigma is the natural column magnitude and perturbed values
stay strictly positive.

Each ``InputEntry`` knows how to (a) locate its slot(s) in the flat input vector
``x0`` that ``fwd`` differentiates, and (b) emit its column(s) of ``L``.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

import torch

from mitim_tools.misc_tools.LOGtools import printMsg as print


@dataclass
class ScatterSpec:
    """
    Where a block of the flat leaf vector ``x0`` must be written back before the
    forward pass, and in what space it is stored.

    ``kind`` / ``key`` identify the target the forward driver scatters into:
      * ("lcfs", ch)   -> powerstate.bc_tensors[ch]["val"]  (separatrix value)
      * ("source", nm) -> a source-rate option consumed by neutrals/charge_states
      * ("vtor", tag)  -> the imposed Vtor representation (_resolve_vtor input)
    ``space`` is "log" (x0 holds log(value)) or "linear".
    ``batch`` is the batch row the UQ is evaluated on (UQ is per-batch).
    """
    kind: str
    key: str
    slots: Sequence[int]
    space: str
    batch: int = 0
    meta: dict = field(default_factory=dict)

    def to_value(self, x0: torch.Tensor) -> torch.Tensor:
        """Map the stored slots back to physical value space."""
        block = x0[torch.as_tensor(self.slots, dtype=torch.long, device=x0.device)]
        return torch.exp(block) if self.space == "log" else block


@dataclass
class InputEntry:
    """One uncertain input (may span several correlated slots, e.g. vtor knots)."""

    name: str
    # indices into the flat input vector x0 that this entry perturbs
    slots: Sequence[int]
    # per-slot 1-sigma, expressed in the *space* of x0 (log or linear)
    sigma: torch.Tensor
    # "log": x0 holds log(value); a relative sigma is used directly.
    # "linear": x0 holds the value; sigma is absolute (= |nominal| * rel).
    space: str = "linear"
    # optional cross-slot correlation factor C (len(slots), r): columns = C.
    # If None, slots are independent -> one column per slot with magnitude sigma.
    corr_factor: Optional[torch.Tensor] = None
    group: str = "edge_param"

    def columns(self, n_in: int, dtype, device) -> torch.Tensor:
        """Return this entry's contribution to L, shape (n_in, n_cols)."""
        slots = torch.as_tensor(self.slots, dtype=torch.long, device=device)
        sig = torch.as_tensor(self.sigma, dtype=dtype, device=device).reshape(-1)

        if self.corr_factor is None:
            # independent slots: diagonal block, one column per slot
            n_cols = slots.numel()
            L = torch.zeros((n_in, n_cols), dtype=dtype, device=device)
            L[slots, torch.arange(n_cols, device=device)] = sig
        else:
            C = torch.as_tensor(self.corr_factor, dtype=dtype, device=device)
            # scale correlation directions by the per-slot sigma
            block = sig.unsqueeze(-1) * C          # (len(slots), r)
            n_cols = block.shape[-1]
            L = torch.zeros((n_in, n_cols), dtype=dtype, device=device)
            L[slots, :] = block
        return L


class UQInputs:
    """
    Collect uncertain-input specifications and assemble the input factor L.

    Typical use::

        uq = UQInputs()
        uq.add_lcfs("ne", rel_sigma=0.10)
        uq.add_lcfs("te", rel_sigma=0.10)
        uq.add_lcfs("ti", rel_sigma=0.15)
        uq.add_source("impurity_source_rate", rel_sigma=0.30)
        uq.add_vtor(mode="fixed_pairs", knots=[(0.0, ...), ...], rel_sigma=0.5)
        x0, layout = uq.build_x0(powerstate)   # flat leaf vector + slot layout
        L = uq.cholesky_columns(x0, layout)    # (n_in, k)
    """

    def __init__(self, log_channels=("ne", "te", "ti")):
        self.log_channels = set(log_channels)
        self.entries: List[InputEntry] = []
        # slot bookkeeping filled by build_x0()
        self._layout: dict = {}

    # ------------------------------------------------------------------ #
    # Input registration
    # ------------------------------------------------------------------ #

    def add_lcfs(self, channel: str, rel_sigma: float, via_bc_model: bool = True):
        """
        LCFS boundary value ne/te/ti with a relative sigma (log space).

        via_bc_model=True (default, correct for the real full-chain scan): perturb
        the BC-model source ``_bc_model_options[channel]`` and reset the BC-model
        instance, so the value survives ``calculateBoundaryConditions`` (which
        rebuilds ``bc_tensors`` every calculate()).  Use for "Fixed" y-models whose
        LCFS values live in bc_model_options.

        via_bc_model=False: legacy direct ``bc_tensors`` write -- only valid if the
        forward bypasses calculateBoundaryConditions (e.g. a purely differentiable
        slice), NOT for real_scan.
        """
        space = "log" if channel in self.log_channels else "linear"
        if via_bc_model:
            self.add_bc_option(channel, rel_sigma, space=space)
        else:
            self._pending(channel, rel_sigma, space, group="bc")

    def add_bc_option(self, key: str, rel_sigma: float, space: str = "log"):
        """
        Any scalar in ``_bc_model_options`` as an uncertain input: an LCFS value
        (ne/te/ti) or a boundary gradient scale length (aLne/aLte/aLti), etc.
        Perturbed through the BC model (option + instance reset) so it survives
        ``calculateBoundaryConditions``.  All these quantities are positive here,
        so ``space="log"`` (relative sigma) is the natural default.
        """
        self._pending(key, rel_sigma, space, group="lcfs_bc",
                      meta={"option_target": "bc", "option_key": key})

    def add_source(self, name: str, rel_sigma: float,
                   option_target: str = "cs", option_key: Optional[str] = None,
                   space: str = "log"):
        """
        A scalar option consumed by a graph-breaking external model (neutral /
        charge-state).  ``option_target`` is "cs" (charge_state_model_options ->
        powerstate._cs_model_options) or "neutral" (-> _neu_model_options);
        ``option_key`` is the exact option string (defaults to ``name``).
        """
        self._pending(name, rel_sigma, space, group="source",
                      meta={"option_target": option_target,
                            "option_key": option_key or name})

    # -- convenience registrations for the standard edge_options scalars -------
    # (keys match neutral_model_options / charge_state_model_options in the
    #  workflow inputs, e.g. H_HighColl.py)

    def add_neutral_source(self, rel_sigma: float):
        """Main-ion neutral source rate: _neu_model_options["source_rate"]."""
        self.add_source("neutral_source_rate", rel_sigma,
                        option_target="neutral", option_key="source_rate")

    def add_impurity_source(self, rel_sigma: float):
        """Impurity injection rate: _cs_model_options["source_rate"]."""
        self.add_source("impurity_source_rate", rel_sigma,
                        option_target="cs", option_key="source_rate")

    def add_impurity_D(self, rel_sigma: float, option_key: str = "D_z_m2_s"):
        """Impurity diffusion coefficient D_z (positive -> log space)."""
        self.add_source("impurity_D", rel_sigma,
                        option_target="cs", option_key=option_key, space="log")

    def add_impurity_V(self, rel_sigma: float, option_key: str = "V_z_m_s"):
        """Impurity convection V_z (sign-changing -> LINEAR space).

        Key must match AuroraChargeStates, which reads options["V_z_m_s"]. The
        previous default here was "V0_m_s", which nothing reads: the perturbation
        was applied to a key the charge-state model ignores, so this UQ dimension
        contributed exactly zero variance.
        """
        self.add_source("impurity_V", rel_sigma,
                        option_target="cs", option_key=option_key, space="linear")

    def add_vtor(self, mode: str, rel_sigma: float, knots=None):
        """
        Toroidal velocity uncertainty, attached to the user's representation.

        mode="fixed_pairs":
            uncertainty lives on each supplied (r, vtor) knot value; propagate
            through the vtor interpolation.  ``knots`` gives the number/locations
            so one (correlated-through-interpolation) column per knot is emitted.
        mode="initial_gacode":
            uncertainty on the supplied vtor(rho) profile -- either a single
            global-multiplier column (rel_sigma on the whole profile) or a
            banded per-point set.  Represented here as a single global column by
            default; refine with corr_factor if a radial band is wanted.

        NB: the two modes want genuinely different Sigma structures; this is a
        modeling decision, not a default to hard-code silently.
        """
        if mode not in ("fixed_pairs", "initial_gacode"):
            raise ValueError(f"unknown vtor mode {mode!r}")
        self._pending(f"vtor::{mode}", rel_sigma, "linear", group="vtor",
                      meta={"mode": mode, "knots": knots})

    def _pending(self, name, rel_sigma, space, group, meta=None):
        self.entries.append(
            _PendingEntry(name=name, rel_sigma=float(rel_sigma),
                          space=space, group=group, meta=meta or {})
        )

    # ------------------------------------------------------------------ #
    # Flat leaf vector + factor assembly
    # ------------------------------------------------------------------ #

    def build_x0(self, powerstate, batch: int = 0):
        """
        Build the flat leaf input vector ``x0`` (the mean the graph is
        differentiated about) and a layout mapping each pending entry to its
        slot indices, sigma (in x0-space), and a ScatterSpec.

        Extraction of nominal values, by entry group:
          * "bc"     -> powerstate.bc_tensors[ch]["val"][batch, 0]
                        (scalar fallback: powerstate.bc_dict[ch][0])
          * "source" -> resolved via ``self._source_nominal`` hook (defaults to
                        reading a ``source_rate`` option; override for your wiring)
          * "vtor"   -> "fixed_pairs": the vtor value of each supplied knot
                        "initial_gacode": a single global multiplier (nominal 1.0)

        Positive channels (log_channels + source rates) are stored as log(value)
        so a relative sigma is the column magnitude and values stay positive.

        Returns
        -------
        x0 : torch.Tensor, shape (n_in,)
        layout : dict  name -> {"slots", "sigma", "corr_factor", "scatter"}
        """
        dfT = getattr(powerstate, "dfT", torch.zeros(1, dtype=torch.double))
        dtype, device = dfT.dtype, dfT.device

        values: List[float] = []      # x0 entries, in x0-space
        layout: dict = {}
        cursor = 0

        for pend in self.entries:
            nominal, corr, scatter_meta = self._extract_nominal(
                powerstate, pend, batch
            )
            nominal = torch.as_tensor(nominal, dtype=dtype, device=device).reshape(-1)
            n = nominal.numel()
            slots = list(range(cursor, cursor + n))
            cursor += n

            # store in x0-space (log for positive channels)
            if pend.space == "log":
                x0_block = torch.log(nominal.clamp_min(1e-30))
                sigma = torch.full((n,), pend.rel_sigma, dtype=dtype, device=device)
            else:
                x0_block = nominal
                sigma = (nominal.abs() * pend.rel_sigma).clamp_min(1e-30)

            values.extend(x0_block.tolist())
            layout[pend.name] = {
                "slots": slots,
                "sigma": sigma,
                "corr_factor": corr,
                "scatter": ScatterSpec(
                    kind=(pend.group if pend.group in ("source", "vtor", "lcfs_bc")
                          else "lcfs"),
                    key=pend.name.split("::")[0],
                    slots=slots,
                    space=pend.space,
                    batch=batch,
                    meta=scatter_meta,
                ),
            }

        x0 = torch.as_tensor(values, dtype=dtype, device=device)
        self._layout = layout
        print(f"[UQ] built x0 ({x0.numel()} slots) from {len(self.entries)} "
              f"input entries (batch {batch})", typeMsg="i")
        return x0, layout

    # ------------------------------------------------------------------ #
    # Nominal-value extraction (per group)
    # ------------------------------------------------------------------ #

    def _extract_nominal(self, powerstate, pend, batch):
        """Return (nominal_value(s), corr_factor_or_None, scatter_meta)."""
        if pend.group == "lcfs_bc":
            optkey = pend.meta.get("option_key", pend.name)
            opts = getattr(powerstate, "_bc_model_options", {}) or {}
            if optkey in opts:
                nominal = float(opts[optkey])
            else:  # fall back to the current bc_tensors / bc_dict value
                bc_tensors = getattr(powerstate, "bc_tensors", None)
                if isinstance(bc_tensors, dict) and optkey in bc_tensors:
                    nominal = float(bc_tensors[optkey]["val"][batch, 0])
                else:
                    nominal = float(powerstate.bc_dict[optkey][0])
            return nominal, None, dict(pend.meta)

        if pend.group == "bc":
            ch = pend.name
            bc_tensors = getattr(powerstate, "bc_tensors", None)
            if isinstance(bc_tensors, dict) and ch in bc_tensors:
                val = float(bc_tensors[ch]["val"][batch, 0])
            else:  # scalar fallback
                val = float(powerstate.bc_dict[ch][0])
            return val, None, {}

        if pend.group == "source":
            target = pend.meta.get("option_target", "cs")
            optkey = pend.meta.get("option_key", pend.name)
            opts = (powerstate._neu_model_options if target == "neutral"
                    else powerstate._cs_model_options) or {}
            if optkey in opts:
                nominal = float(opts[optkey])
            else:  # fall back to a user-supplied hook
                nominal = self._source_nominal(powerstate, pend.name)
            return nominal, None, dict(pend.meta)

        if pend.group == "vtor":
            mode = pend.meta.get("mode")
            if mode == "fixed_pairs":
                knots = pend.meta.get("knots") or []
                # knots = [(rho, vtor_m_s), ...]; uncertainty on each vtor value.
                vtor_vals = [float(v) for (_r, v) in knots]
                # correlation across knots is induced downstream by the vtor
                # interpolation, so keep the knots independent here.
                return vtor_vals, None, {"knots": knots}
            elif mode == "initial_gacode":
                # single global multiplier on the extracted vtor profile.
                return 1.0, None, {"mode": mode}
            raise ValueError(f"unknown vtor mode {mode!r}")

        raise ValueError(f"cannot extract nominal for group {pend.group!r}")

    def _source_nominal(self, powerstate, name):
        """
        Resolve a source-rate nominal.  Default reads a ``source_rate`` option
        from the neutrals / charge-state options; override this method (or set
        ``self.source_nominal_fn``) to match your exact option wiring.
        """
        if getattr(self, "source_nominal_fn", None) is not None:
            return float(self.source_nominal_fn(powerstate, name))
        raise NotImplementedError(
            f"source nominal for {name!r}: set UQInputs.source_nominal_fn or "
            f"override _source_nominal to read the source-rate option "
            f"(neutrals.source_rate / impurity source_rate) from powerstate."
        )

    def cholesky_columns(self, x0: torch.Tensor, layout: dict) -> torch.Tensor:
        """
        Assemble L (n_in, k) by concatenating each entry's columns.

        Independence across *entries* is assumed (block-diagonal L); intra-entry
        correlation (e.g. across vtor knots) is carried via ``corr_factor``.  A
        cross-entry correlation, if ever needed, is added by supplying a joint
        corr_factor rather than changing this machinery.
        """
        n_in = x0.numel()
        dtype, device = x0.dtype, x0.device
        blocks = []
        for pend in self.entries:
            slots = layout[pend.name]["slots"]
            sigma = layout[pend.name]["sigma"]        # already in x0-space
            entry = InputEntry(
                name=pend.name, slots=slots, sigma=sigma,
                space=pend.space, group=pend.group,
                corr_factor=layout[pend.name].get("corr_factor"),
            )
            blocks.append(entry.columns(n_in, dtype, device))
        if not blocks:
            raise ValueError("no uncertain inputs registered")
        L = torch.cat(blocks, dim=1)
        print(f"[UQ] input factor L: {tuple(L.shape)} "
              f"({len(self.entries)} entries)", typeMsg="i")
        return L


@dataclass
class _PendingEntry:
    name: str
    rel_sigma: float
    space: str
    group: str
    meta: dict = field(default_factory=dict)
