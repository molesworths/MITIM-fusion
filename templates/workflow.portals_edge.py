"""
PORTALS-E(dge) workflow template.
=================================

Copy this file, edit the CONFIG block, run it. Everything below the CONFIG block is the standard
edge launch sequence; the physics choices all live in `edge_options`.

    python workflow.portals_edge.py

Two equivalent ways to configure a run:
  (a) this script — set `portals_fun.portals_parameters[...]` before `prep()`, as done here;
  (b) a YAML namelist — `portals_edge(folder, portals_namelist=".../namelist.portals_edge.yaml")`,
      which documents every option at its default.

See README_PORTALSedge.md (src/mitim_modules/portals/) for what each option does.
"""

import os
from pathlib import Path

import numpy as np
import torch

from mitim_modules.portals.PORTALSedge import portals_edge
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.opt_tools import STRATEGYtools

# ==============================================================================================
# CONFIG
# ==============================================================================================

INPUTGACODE = Path("/path/to/input.gacode")
FOLDERWORK  = Path("/path/to/results/my_edge_run")

COLD_START = True

ROA        = [0.88, 0.91, 0.94, 0.97]   # radial control points (also the parameterizer knots)
CHANNELS   = ["te", "ti", "ne"]         # add "nZ" to predict the impurity density too

# Impurity: total injection rate [1/s]. Zeff-1 is linear in it, so `calibrate_impurity_source`
# below hits a measured LCFS Zeff in one measurement. Ignored (inert) when
# impurity_density_source = "prescribed".
IMPURITY_SOURCE_RATE = 1.0e20

# Main-ion neutral influx [1/s]. In steady state this equals the integrated ionization source,
# i.e. the LCFS particle efflux — take it from the reconstruction rather than tuning it to
# n_e,ped, so that n_e,ped stays a predicted output.
NEUTRAL_SOURCE_RATE = 8.0e20


# ==============================================================================================
# SETUP
# ==============================================================================================

def setup(cold_start=COLD_START, impurity_source_rate=IMPURITY_SOURCE_RATE):
    """Build the configured portals_edge object and the initial plasma state."""

    if cold_start and FOLDERWORK.exists():
        os.system(f"rm -r {FOLDERWORK.resolve()}")

    portals_fun = portals_edge(
        FOLDERWORK,
        tensor_options={"dtype": torch.double, "device": torch.device("cpu")},
    )

    # ---------------------------------------------------------------------------------------
    # Standard PORTALS options
    # ---------------------------------------------------------------------------------------
    opt = portals_fun.optimization_options
    opt["initialization_options"]["initial_training"] = 5
    opt["initialization_options"]["simple_relax_options"] = {}   # {} = seeder defaults
    opt["surrogate_options"]["test_combination_stop_on_failure"] = False
    opt["convergence_options"]["maximum_iterations"] = 20
    opt["convergence_options"]["stop_if_no_new_points"] = True

    sc = opt["convergence_options"]["stopping_criteria_parameters"]
    sc["maximum_value"] = 1e-5          # ABSOLUTE residual floor (see maximum_value_is_rel)
    sc["maximum_value_is_rel"] = False
    sc["ricci_value"] = 1e-4

    # ---------------------------------------------------------------------------------------
    # Solution: radii, channels, parameterization, bounds
    # ---------------------------------------------------------------------------------------
    sol = portals_fun.portals_parameters["solution"]
    sol["predicted_roa"] = ROA
    sol["predicted_channels"] = CHANNELS
    sol["turbulent_exchange_as_surrogate"] = True

    sol["parameterizer"] = "SplineMtanh"
    sol["parameterizer_options"] = {"knots": ROA, "defined_on": "aLy"}

    # ABSOLUTE aLy bounds. A relative box around the initial guess censors the pedestal solution
    # set; keep the upper bound beyond anything physical so the bounds never bind.
    er = sol["exploration_ranges"]
    er["limits_are_relative"] = False
    er["ymin"] = {ch: [0.0] * len(ROA) for ch in CHANNELS}
    er["ymax"] = {ch: [10.0, 20.0, 40.0, 80.0] for ch in CHANNELS}

    # ---------------------------------------------------------------------------------------
    # Transport codes
    # ---------------------------------------------------------------------------------------
    tglf = portals_fun.portals_parameters["transport"]["options"]["tglf"]
    tglf["run"]["code_settings"] = "edge"       # SAT3 + ALPHA_ZF=-1 + EM, pedestal resolution
    tglf["cores_per_tglf_instance"] = 2
    tglf["use_scan_trick_for_stds"] = None      # flat percent_error instead
    # tglf["run"]["extraOptions"] = {}          # overrides ON TOP of the preset

    neo = portals_fun.portals_parameters["transport"]["options"]["neo"]
    neo["run"]["code_settings"] = "edge"        # ROTATION_MODEL=1; DPHI0DR injected per rho

    # ---------------------------------------------------------------------------------------
    # Edge physics
    # ---------------------------------------------------------------------------------------
    edge_options = {}

    edge_options["domain_roa"] = [0.87, 1.0]

    # (1) LCFS BCs frozen at the input-profile separatrix values: keep the Fixed models and pass
    #     NO y/aLy overrides, so boundary.py reads and freezes what the initial profile carries.
    edge_options["bc_model"] = {"y": "Fixed", "aLy": "Fixed"}
    edge_options["bc_model_options"] = {}

    # (2) Main-ion neutrals.
    edge_options["neutral_model"] = "Analytic"
    edge_options["neutral_model_options"] = {
        "source_rate": float(NEUTRAL_SOURCE_RATE),
        "include_cx": True,
    }

    # (3) Impurities. Two amplitude conventions — pick ONE:
    #     "charge_state_model" : Aurora's solve sets the level; calibrate source_rate to the
    #                            measured LCFS Zeff (see calibrate_impurity_source below).
    #     "prescribed"         : the initial profile's f_Z carried with ne sets the level;
    #                            Aurora supplies only the charge-state fractions, so the Zeff
    #                            calibration is INERT.
    edge_options["impurity_density_source"] = "charge_state_model"
    edge_options["charge_state_model"] = "Aurora"
    edge_options["charge_state_model_options"] = {
        "imp": "C",
        "D_z_m2_s": 1.0,
        "V_z_m_s": -1.0,           # or set aLnZ_profile instead (a/L_nZ = V/D in this model)
        "source_rate": float(impurity_source_rate),
        "cxr_flag": True,
    }

    # (4) Rotation. "extract_initial" freezes the reconstructed Vtor and lets the diamagnetic Er
    #     evolve with the profiles; the vtor=0 closure understates gamma_ExB when the discharge
    #     actually rotates.
    edge_options["rotation_options"] = {"vtor_source": "extract_initial"}

    # (5) ELMs / nonlocal corrections — off by default.
    edge_options["elm_model"] = "Null"
    edge_options["elm_model_options"] = {}
    # edge_options["nonlocal_model"] = "Analytic"
    # edge_options["nonlocal_model_options"] = {"ExB": True, "Spreading": True, "lambda_c_mult": 8.0}

    # (6) Targets.
    edge_options["use_edge_targets"] = True
    edge_options["target_multipliers"] = {"Qe": 1.0, "Qi": 1.0, "Ge": 1.0}

    portals_fun.portals_parameters["edge_options"] = edge_options

    # ---------------------------------------------------------------------------------------
    # Initial plasma state
    # ---------------------------------------------------------------------------------------
    plasma_state = PROFILEStools.gacode_state(INPUTGACODE)
    plasma_state.correct(options={
        "recalculate_ptot": True,
        "remove_fast": True,
        "quasineutrality": True,
        # enforce_same_aLn=True pins f_Z constant, so a/L_nZ = a/L_ne exactly. Use with
        # impurity_density_source="prescribed"; leave it off when Aurora owns the density.
        "enforce_same_aLn": False,
    })

    return portals_fun, plasma_state


# ==============================================================================================
# OPTIONAL: calibrate the Aurora source rate to a measured LCFS Zeff
# ==============================================================================================

def _lcfs_zeff(portals_fun, source_rate):
    """Self-consistent LCFS Zeff at a given source_rate, on the already-prepped powerstate.

    Runs the same chain a flux-match evaluation does: Aurora's solve() produces nz_all, then
    _enforce_quasineutrality() closes ni and recomputes Zeff = sum ni Zi^2 / ne. solve() alone
    does NOT touch Zeff — quasineutrality owns it.
    """
    from mitim_tools.edge_tools.charge_states import AuroraChargeStates

    o = dict(portals_fun.portals_parameters["edge_options"]["charge_state_model_options"])
    o["source_rate"] = float(source_rate)
    AuroraChargeStates(o).solve(portals_fun.powerstate, batch_idx=0)
    portals_fun.powerstate._enforce_quasineutrality()

    z = portals_fun.powerstate.plasma["Zeff"]
    a = np.asarray(z.detach().cpu().numpy() if hasattr(z, "detach") else z, dtype=float)
    return float(a.ravel()[-1] if a.ndim == 1 else a[0, -1])


def calibrate_impurity_source(zeff_target, source_ref=1.0e20, cold_start=COLD_START):
    """Return a fresh, calibrated (portals_fun, plasma_state, source_cal).

    Zeff-1 is linear in source_rate (and zero at source=0), so one measurement fixes it:
        source_cal = source_ref * (zeff_target - 1) / (zeff_ref - 1)
    The charge-state + quasineutrality chain mutates the powerstate, so the measurement runs on a
    throwaway build and the run uses a clean rebuild at source_cal (re-prepping the same object
    does NOT propagate the new rate).
    """
    pf, ps = setup(cold_start=cold_start, impurity_source_rate=source_ref)
    pf.prep(ps, cold_start=cold_start, askQuestions=False)

    z_ref = _lcfs_zeff(pf, source_ref)
    source_cal = source_ref * (zeff_target - 1.0) / max(z_ref - 1.0, 1e-6)
    z_chk = _lcfs_zeff(pf, source_cal)
    print(f"[edge] Zeff calib: Zeff({source_ref:.2e})={z_ref:.4f} -> source={source_cal:.3e}; "
          f"verify Zeff={z_chk:.4f} (target {zeff_target:.4f})")

    pf, ps = setup(cold_start=cold_start, impurity_source_rate=source_cal)
    pf.prep(ps, cold_start=cold_start, askQuestions=False)
    return pf, ps, source_cal


# ==============================================================================================
# RUN
# ==============================================================================================

if __name__ == "__main__":

    portals_fun, plasma_state = setup()
    portals_fun.prep(plasma_state, cold_start=COLD_START, askQuestions=False)

    # ...or, to anchor the impurity level on a measured separatrix Zeff:
    # portals_fun, plasma_state, source_cal = calibrate_impurity_source(zeff_target=1.23)

    solver = STRATEGYtools.MITIM_BO(portals_fun, cold_start=COLD_START, askQuestions=False)
    solver.run()