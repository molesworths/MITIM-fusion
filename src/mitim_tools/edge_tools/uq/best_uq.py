"""
best_uq.py -- run the best-index edge-UQ band on a COMPLETED SALM run folder.

The in-run path (salm ``edge_uq_final_only``, default ON) does this
automatically at the end of an optimization. This module is the offline / CLI
equivalent for runs that finished without it, or where the band should be
re-run with different options.

Usage (module)::

    from mitim_tools.edge_tools.uq.best_uq import run_best_uq
    result = run_best_uq(run_folder)                       # ch5-standard spec
    result = run_best_uq(run_folder, uq_inputs=..., uq_options=...)

Usage (CLI)::

    python -m mitim_tools.edge_tools.uq.best_uq /path/to/run_folder \
        [--rel-step 1.0] [--fit-error-max-dirs 1] [--no-fit-error]

Notes
-----
* Needs only the run FOLDER (the directory containing Outputs/ and Execution/):
  the best eval is read from Outputs/optimization_data.csv (max
  maximization_objective), its saved powerstate from
  Execution/Evaluation.{i}/powerstate.pkl (new layout) or
  Initialization/initialization_simple_relax/portals_sr_ev_{i}/powerstate.pkl
  (old layout), and X_dvs from the CSV DV columns (declared DV order).
  ``run_edge_uq`` itself needs no portals_fun -- the pickled powerstate carries
  the transport configuration.
* Default spec = the ch5-standard block (LCFS ne/te/ti 10%, neutral source 20%,
  impurity source 20%, impurity D/V 50%) with SplineMtanh fit-error propagation
  restricted to the LEADING eigendirection per channel (fit_error_max_dirs=1).
* The result is saved to Outputs/edge_uq_best.pkl and returned.
"""

import csv
import argparse
from pathlib import Path

import numpy as np
import torch


def default_ch5_uq_inputs():
    from mitim_tools.edge_tools.uq import UQInputs
    uq = UQInputs()
    uq.add_lcfs("ne", 0.10); uq.add_lcfs("te", 0.10); uq.add_lcfs("ti", 0.10)
    uq.add_neutral_source(0.2); uq.add_impurity_source(0.2)
    uq.add_impurity_D(0.5); uq.add_impurity_V(0.5)
    return uq


def default_ch5_uq_options():
    return {
        "mode": "real_scan", "rel_step": 1.0,
        "rotation_proxy": True, "use_surrogate_gp": False, "lcfs_aly_mode": "peret",
        "propagate_fit_error": True, "fit_error_max_dirs": 1,
    }


def load_spec(run_folder):
    """The run's OWN persisted UQ spec (Outputs/edge_uq_spec.pkl, written by
    salm at init when edge_uq_final_only stashes the per-eval hook).
    Returns (uq_inputs, uq_options) or (None, None) when absent (older runs)."""
    p = Path(run_folder) / "Outputs" / "edge_uq_spec.pkl"
    if not p.exists():
        return None, None
    import dill
    with open(p, "rb") as h:
        spec = dill.load(h)
    return spec.get("uq_inputs"), spec.get("uq_options")


def find_best_eval(run_folder):
    """Best eval index + its DV row from Outputs/optimization_data.csv.
    DV columns are those between 'Iteration' and the first output column."""
    f = Path(run_folder) / "Outputs" / "optimization_data.csv"
    with open(f) as h:
        reader = csv.DictReader(h)
        cols = list(reader.fieldnames)
        rows = list(reader)
    obj = [k for k in cols if "objective" in k.lower()][-1]
    vals = []
    for i, r in enumerate(rows):
        try:
            vals.append((float(r[obj]), i))
        except (TypeError, ValueError):
            pass
    if not vals:
        raise RuntimeError(f"no numeric '{obj}' values in {f}")
    _, ibest = max(vals)
    out0 = next(k for k in cols if ("_tr_turb_" in k or "_tar_" in k))
    dv_cols = cols[cols.index("Iteration") + 1: cols.index(out0)]
    x = np.array([float(rows[ibest][k]) for k in dv_cols], dtype=float)
    return ibest, x, dv_cols


def load_best_powerstate(run_folder, ibest):
    from mitim_modules.powertorch import STATEtools
    run_folder = Path(run_folder)
    candidates = [
        run_folder / "Execution" / f"Evaluation.{ibest}" / "powerstate.pkl",
        run_folder / "Initialization" / "initialization_simple_relax"
        / f"portals_sr_ev_{ibest}" / "powerstate.pkl",
    ]
    for p in candidates:
        if p.exists():
            return STATEtools.read_saved_state(p), p
    raise FileNotFoundError(
        f"no powerstate.pkl for best eval {ibest}; looked in {[str(c) for c in candidates]}")


def run_best_uq(run_folder, uq_inputs=None, uq_options=None, save=True):
    """Run run_edge_uq once at the best evaluation of a completed run folder.

    Spec resolution (per item): explicit argument > the run's own stored spec
    (Outputs/edge_uq_spec.pkl) > the ch5-standard defaults."""
    from mitim_tools.edge_tools.uq.run import run_edge_uq
    run_folder = Path(run_folder)

    stored_inputs, stored_opts = load_spec(run_folder)
    if uq_inputs is None:
        uq_inputs = stored_inputs
        src = "stored spec" if uq_inputs is not None else "ch5 defaults"
        if uq_inputs is None:
            uq_inputs = default_ch5_uq_inputs()
        print(f"[best_uq] uq_inputs from {src}")
    opts = dict(default_ch5_uq_options())
    if stored_opts:
        opts.update(stored_opts)
    opts.update(uq_options or {})
    for key in ("inject_into", "start_after_eval"):
        opts.pop(key, None)
    uq_folder = opts.pop("folder", run_folder / "Outputs" / "edge_uq_best")

    ibest, x, dv_cols = find_best_eval(run_folder)
    ps, pkl = load_best_powerstate(run_folder, ibest)
    X = torch.as_tensor(x, dtype=ps.dfT.dtype, device=ps.dfT.device).unsqueeze(0)

    print(f"[best_uq] best eval {ibest} ({pkl.name}); {len(dv_cols)} DVs; "
          f"running edge-UQ once (fit_error_max_dirs="
          f"{opts.get('fit_error_max_dirs')})...")
    result = run_edge_uq(ps, uq_inputs, X_dvs=X, inject_into=ps,
                         folder=uq_folder, **opts)

    # Persist the band where the plotter actually reads it: mitim_plot_portals_edge's
    # UQ-donor lookup pulls "{key}_uq_std" from the powerstates stored in
    # optimization_extra.pkl -- an in-memory injection alone shows NO bands.
    _persist_band(run_folder, ibest, ps, pkl)

    if save:
        import dill
        out = run_folder / "Outputs" / "edge_uq_best.pkl"
        with open(out, "wb") as h:
            dill.dump({"ibest": ibest, "result": result}, h, protocol=4)
        print(f"[best_uq] saved {out}")
    return result


def _persist_band(run_folder, ibest, ps, pkl):
    """Write the band-injected powerstate back to its pkl AND into the best-eval
    entry of Outputs/optimization_extra.pkl (what PORTALSanalyzer/the plotter reads)."""
    import dill
    try:
        ps.save(pkl)
        print(f"[best_uq] band-injected powerstate re-saved to {pkl}")
    except Exception as e:
        print(f"[best_uq] WARNING: could not re-save powerstate ({e})")
    extra = Path(run_folder) / "Outputs" / "optimization_extra.pkl"
    if not extra.exists():
        print("[best_uq] NOTE: no optimization_extra.pkl (plotting artifacts absent); "
              "bands will appear once artifacts are (re)built from the pkls")
        return
    try:
        with open(extra, "rb") as h:
            store = dill.load(h)
        if isinstance(store, dict) and isinstance(store.get(ibest), dict):
            store[ibest]["powerstate"] = ps
            with open(extra, "wb") as h:
                dill.dump(store, h, protocol=4)
            print(f"[best_uq] optimization_extra.pkl updated -- eval {ibest} now "
                  f"carries the UQ band (replot to see it)")
        else:
            print(f"[best_uq] WARNING: optimization_extra.pkl has no dict entry for "
                  f"eval {ibest}; plotter may not find the band")
    except Exception as e:
        print(f"[best_uq] WARNING: could not update optimization_extra.pkl ({e})")


def reinject_band(run_folder):
    """Re-inject an ALREADY-COMPUTED band (Outputs/edge_uq_best.pkl) into the plotting
    artifacts -- no transport re-runs. For runs where run_best_uq/the in-run final UQ
    computed the band before persistence was added."""
    import dill
    run_folder = Path(run_folder)
    f = run_folder / "Outputs" / "edge_uq_best.pkl"
    with open(f, "rb") as h:
        saved = dill.load(h)
    ibest, result = int(saved["ibest"]), saved["result"]
    pstd = result.get("profile_std") or {}
    if not pstd:
        raise RuntimeError(f"{f} has no profile_std -- re-run run_best_uq instead")
    ps, pkl = load_best_powerstate(run_folder, ibest)
    suffix = "_uq_std"
    for k, sig in pstd.items():
        ps.plasma[k + suffix] = sig
    # minimal summary so the plotter's suffix lookup + residual panels work
    ps._edge_uq_summary = {
        "residual_mean": result.get("residual_mean"),
        "residual_std": result.get("residual_std"),
        "mu_J": result.get("mu_J"), "sigma_J": result.get("sigma_J"),
        "robust_objective": result.get("robust_objective"),
        "converged": result.get("converged"), "chi2_stat": result.get("chi2_stat"),
        "dof": result.get("dof"), "chi2_threshold": result.get("chi2_threshold"),
        "near_optimum": result.get("near_optimum"),
        "profile_std_keys": list(pstd.keys()),
        "profile_std_suffix": suffix,
    }
    print(f"[best_uq] re-injected {len(pstd)} profile-std keys into eval {ibest}")
    _persist_band(run_folder, ibest, ps, pkl)
    return ps


def _main():
    ap = argparse.ArgumentParser(
        description="Run the best-index edge-UQ band on a completed CALM run folder.")
    ap.add_argument("folder", help="run folder (contains Outputs/ and Execution/)")
    ap.add_argument("--rel-step", type=float, default=None)
    ap.add_argument("--fit-error-max-dirs", type=int, default=None)
    ap.add_argument("--no-fit-error", action="store_true",
                    help="disable SplineMtanh fit-error propagation")
    ap.add_argument("--reinject", action="store_true",
                    help="re-inject an already-computed Outputs/edge_uq_best.pkl into "
                         "the plotting artifacts (no transport re-runs)")
    args = ap.parse_args()
    if args.reinject:
        reinject_band(args.folder)
        return
    opts = {}
    if args.rel_step is not None:
        opts["rel_step"] = args.rel_step
    if args.fit_error_max_dirs is not None:
        opts["fit_error_max_dirs"] = args.fit_error_max_dirs
    if args.no_fit_error:
        opts["propagate_fit_error"] = False
    run_best_uq(args.folder, uq_options=opts)


if __name__ == "__main__":
    _main()
