"""
Standalone test for EvaluationArchive featurization + append + readback.

Uses in-memory GACODE input decks (no transport runs) and a fake power_transport
object carrying GB flux/std arrays, to verify:
  - the fixed (gating) vs derived (co-moving) vs dv_image split (Q1)
  - fluxes/sigmas are passed through verbatim from *_turb / *_neoc arrays (Q2)
  - per-(code, rho) records round-trip through the JSONL store

Run: python tests/EVALARCHIVE_test.py
"""

import json
import tempfile
import types
from pathlib import Path

import numpy as np

from mitim_tools.gacode_tools.TGLFtools import TGLFinput
from mitim_tools.simulation_tools.SIMtools import GACODEinput
from mitim_modules.powertorch.utils import EVALarchive
from mitim_modules.powertorch.utils.EVALarchive import EvaluationArchive, featurize_deck


def _tglf_deck():
    d = {
        # identity
        "NS": 2, "SAT_RULE": 2, "UNITS": "CGYRO", "GEOMETRY_FLAG": 1, "USE_BPER": "T",
        # fixed background (geometry/magnetics)
        "RMIN_LOC": 0.5, "RMAJ_LOC": 3.0, "KAPPA_LOC": 1.6, "S_KAPPA_LOC": 0.2,
        "DELTA_LOC": 0.1, "Q_LOC": 2.0, "Q_PRIME_LOC": 5.0, "DRMAJDX_LOC": -0.1,
        # derived (co-moving with DVs)
        "BETAE": 0.004, "XNUE": 0.5, "ZEFF": 1.8, "P_PRIME_LOC": -0.02, "VEXB_SHEAR": 0.1,
        # species 1 = electrons, species 2 = main ion (D)
        "ZS_1": -1.0, "MASS_1": 0.00027, "RLNS_1": 1.0, "RLTS_1": 2.0, "TAUS_1": 1.0, "AS_1": 1.0,
        "ZS_2":  1.0, "MASS_2": 1.0,     "RLNS_2": 1.0, "RLTS_2": 3.0, "TAUS_2": 0.9, "AS_2": 1.0,
    }
    return TGLFinput.initialize_in_memory(d)


def _neo_deck():
    # NEOinput uses the base GACODEinput.process (species live in plasma)
    d = {
        "N_SPECIES": 2, "EQUILIBRIUM_MODEL": 2, "COLLISION_MODEL": 4, "ROTATION_MODEL": 2,
        "RMIN_OVER_A": 0.5, "RMAJ_OVER_A": 3.0, "KAPPA": 1.6, "DELTA": 0.1, "Q": 2.0, "SHEAR": 1.0,
        "NU_1": 0.3, "OMEGA_ROT": 0.05, "DPHI0DR": -0.2,
        "Z_1": -1.0, "MASS_1": 0.00027, "DLNNDR_1": 1.0, "DLNTDR_1": 2.0, "DENS_1": 1.0, "TEMP_1": 1.0,
        "Z_2":  1.0, "MASS_2": 1.0,     "DLNNDR_2": 1.0, "DLNTDR_2": 3.0, "DENS_2": 1.0, "TEMP_2": 0.9,
    }
    obj = GACODEinput()
    obj.code, obj.n_species = "NEO", "N_SPECIES"
    obj.process(d)
    return obj


def _fake_transport(rho):
    n = len(rho)
    t = types.SimpleNamespace()
    t.name = "unit"
    t.evaluation_number = 0
    t.folder = Path("/tmp/unit")
    # powerstate.plasma with rho/roa/Qgb including the rho=0 pad
    plasma = {
        "rho": np.concatenate([[0.0], rho])[None, :],
        "roa": np.concatenate([[0.0], rho * 0.9])[None, :],
        "Qgb": np.concatenate([[0.0], np.ones(n)])[None, :],
    }
    t.powerstate = types.SimpleNamespace(plasma=plasma)
    # GB fluxes + stds, verbatim source of truth
    for gb in ["QeGB", "QiGB", "GeGB", "GZGB", "MtGB", "QieGB"]:
        setattr(t, f"{gb}_turb", np.arange(n, dtype=float) + hash(gb) % 7)
        setattr(t, f"{gb}_turb_stds", np.full(n, 0.1))
    for gb in ["QeGB", "QiGB", "GeGB", "GZGB", "MtGB"]:
        setattr(t, f"{gb}_neoc", np.arange(n, dtype=float) * 0.1)
        setattr(t, f"{gb}_neoc_stds", np.full(n, 0.01))
    return t


def main():
    # --- featurization: Q1 split ---
    feats = featurize_deck("TGLF", _tglf_deck())
    assert "KAPPA_LOC" in feats["fixed"], "geometry must be fixed/gating"
    assert "Q_PRIME_LOC" in feats["fixed"], "magnetic shear must be fixed/gating"
    assert "BETAE" in feats["derived"] and "XNUE" in feats["derived"], "betae/nu co-move -> derived"
    assert "P_PRIME_LOC" in feats["derived"], "pressure gradient co-moves -> derived"
    assert "RLTS__r0" in feats["dv_image"] and "RLTS__r1" in feats["dv_image"], "gradients -> dv_image"
    assert "TAUS__r1" in feats["derived"], "Ti/Te co-moves -> derived"
    # electron (Z<0) canonicalized to role 0
    assert feats["species_signature"][0][0] == -1.0
    assert "KAPPA_LOC" not in feats["derived"] and "BETAE" not in feats["fixed"]
    print("OK  TGLF featurization: fixed/derived/dv_image split correct")

    nf = featurize_deck("NEO", _neo_deck())
    assert "KAPPA" in nf["fixed"] and "SHEAR" in nf["fixed"]
    assert "NU_1" in nf["derived"] and "DPHI0DR" in nf["derived"]
    assert "DLNTDR__r0" in nf["dv_image"] and "DLNTDR__r1" in nf["dv_image"]
    print("OK  NEO featurization: fixed/derived/dv_image split correct")

    # different equilibrium -> different identity_key partition
    feats2 = featurize_deck("TGLF", _tglf_deck())
    assert feats["identity_key"] == feats2["identity_key"], "same deck -> same key"

    # --- records + Q2 flux pass-through + JSONL round-trip ---
    rho = np.array([0.4, 0.6, 0.8])
    transport = _fake_transport(rho)
    monkey = EvaluationArchive  # build records via the real method, patch deck access
    with tempfile.TemporaryDirectory() as tmp:
        arch = EvaluationArchive(tmp)
        # supply per-rho decks via the .inputs_files contract
        tglf_obj = types.SimpleNamespace(inputs_files={r: _tglf_deck() for r in rho})
        neo_obj = types.SimpleNamespace(inputs_files={r: _neo_deck() for r in rho})
        records = arch.records_from_transport(transport, tglf_obj, neo_obj)

        assert len(records) == 2 * len(rho), f"expected {2*len(rho)} records, got {len(records)}"
        tglf_recs = [r for r in records if r["code"] == "TGLF"]
        # Q2: flux equals the GB array verbatim at that rho index
        i = 1  # rho=0.6
        rec = sorted(tglf_recs, key=lambda r: r["rho"])[i]
        assert rec["fluxes"]["Qe"] == float(transport.QeGB_turb[i]), "TGLF flux not verbatim"
        assert rec["fluxes_sigma"]["Qe"] == float(transport.QeGB_turb_stds[i]), "TGLF sigma not verbatim"
        neo_rec = sorted([r for r in records if r["code"] == "NEO"], key=lambda r: r["rho"])[i]
        assert neo_rec["fluxes"]["Qe"] == float(transport.QeGB_neoc[i]), "NEO flux not verbatim"
        assert "Qie" not in neo_rec["fluxes"], "NEO has no turbulent exchange channel"
        print("OK  records: Q2 flux/sigma pass-through verbatim (TGLF turb, NEO neoc)")

        arch.append(records)
        lines = (Path(tmp) / "code=TGLF" / "records.jsonl").read_text().splitlines()
        assert len(lines) == len(rho)
        back = json.loads(lines[0])
        assert back["code"] == "TGLF" and "raw_deck" in back and "identity_key" in back
        print("OK  JSONL store: append + readback round-trips")

    print("\nALL EVALARCHIVE TESTS PASSED")


if __name__ == "__main__":
    main()
