import re
import shutil
from pathlib import Path
import numpy as np
from mitim_tools import __mitimroot__
from mitim_tools.gacode_tools.CGYROtools import CGYROinput
from mitim_tools.gacode_tools.utils import GACODEdefaults
from mitim_tools.simulation_tools import SIMtools
from mitim_tools.misc_tools import CONFIGread, IOtools
from mitim_tools.misc_tools.LOGtools import printMsg as print

# ---------------------------------------------------------------------------------------------
# DMD post-processing of per-ky linear CGYRO field histories (EMA/early-exit companion)
#
# The patched gacode build (cgyro_freq.F90) adds compile-time-enabled linear early exits
# (marginal, frequency-EMA, ky-aware gate) and writes double-precision field histories
# (HIPREC default flipped to 1). The EMA exit is a STOP signal only: the reported eigenvalue
# is CGYRO's instantaneous estimate at exit. The gate below cross-checks that estimate with a
# seeded DMD extraction from the retained field history and flags kys where they disagree
# (under-converged). Backtested 2026-07-22 on 11 radius-cases (see
# gacode_tools/scripts/qlgyro_dmd/README_DMD_BACKTEST.md): at gate-accepted points the
# eigenvalue is correct 90-98% and QL
# weights are converged to <10-20%; raw DMD-dominant (max-gamma) selection is NOT reliable
# and is not used here.
# ---------------------------------------------------------------------------------------------

# Files a KY_* directory must keep for the DMD gate (pygacode cgyrodata + field history);
# heavy moment histories and restarts are trimmed on harvest.
_CGYRO_RAW_TRIM_PATTERNS = ["bin.cgyro.restart*", "bin.cgyro.kxky_n", "bin.cgyro.kxky_e", "bin.cgyro.kxky_v"]


def cgyro_field_matrix(sim):
    """Stack the per-ky CGYRO field time-histories into a complex [space, time] snapshot
    matrix. Returns (X, split), where split maps each field name to its row range in X
    (for ES/EM decomposition), or (None, None) if no usable history is present."""
    imax = np.asarray(sim.t).size
    blocks, split, i0 = [], {}, 0
    for name in ("kxky_phi", "kxky_apar", "kxky_bpar"):
        arr = getattr(sim, name, None)
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.ndim == 5:  # (2, nr, ntheta, nn, ntime) real/imag split
            y = arr[0, :, :, 0, :imax] + 1j * arr[1, :, :, 0, :imax]
        elif arr.ndim == 4:  # (nr, ntheta, nn, ntime) complex
            y = arr[:, :, 0, :imax]
        else:
            continue
        y = y.reshape(-1, y.shape[-1])
        if y.shape[-1] < 8 or np.max(np.abs(y)) < 1e-30:
            continue
        blocks.append(y)
        split[name] = (i0, i0 + y.shape[0])
        i0 += y.shape[0]
    if not blocks:
        return None, None
    return np.vstack(blocks), split


def hankel_embed(X, delay):
    """Time-delay embed [space, time] -> [space*delay, time-delay+1]. Augments the spatial
    basis so DMD can resolve modes from a rank-1-dominated (growing) linear signal; the
    eigenvalues are unchanged, the conditioning improves. Returns (Xh, delay_used)."""
    ns, nt = X.shape
    if nt <= delay + 2:
        return X, 1
    return np.vstack([X[:, i:nt - delay + 1 + i] for i in range(delay)]), delay


def _dmd_eigenvalues(kydir, delay=6, rank=10):
    """DMD eigenvalues (gamma, omega) from a per-ky CGYRO field history.
    Returns (list of (gamma, omega), dt) or (None, reason)."""
    try:
        from pydmd import DMD
        from pygacode.cgyro.data import cgyrodata
    except ImportError as e:
        return None, f"missing dependency ({e})"

    try:
        sim = cgyrodata(str(kydir) + "/", silent=True)
        sim.getbigfield()
    except Exception as e:
        return None, f"could not read CGYRO data ({e})"

    t = np.asarray(sim.t)
    if t.size < 20:
        return None, "record too short"

    X, _ = cgyro_field_matrix(sim)
    if X is None:
        return None, "no usable field history (HIPREC_FLAG=0 build?)"

    n = X.shape[1]
    i0 = n // 3  # drop transient
    Xw = X[:, i0:]
    if Xw.shape[1] < 12:
        return None, "post-transient window too short"

    Xw, _ = hankel_embed(Xw, delay)

    dt = t[1] - t[0]
    d = DMD(svd_rank=min(rank, Xw.shape[1] - 2, Xw.shape[0]))
    try:
        d.fit(Xw)
    except Exception as e:
        return None, f"DMD fit failed ({e})"
    if len(d.eigs) == 0:
        return None, "DMD returned no eigenvalues"

    omega = 1j * np.log(d.eigs) / dt  # gamma = imag, omega_r = real
    return [(float(g), float(w)) for g, w in zip(omega.imag, omega.real)], None


def dmd_agreement_gate(
    raw_folder,
    gamma_rel_tol=0.1,
    gamma_abs_tol=0.05,
    gamma_marginal=0.05,
):
    """Seeded-DMD agreement gate over the per-ky CGYRO runs of one radius.

    For each KY_* directory: take CGYRO's running eigenvalue estimate (last row of
    out.cgyro.freq, columns (omega, gamma)), select the DMD eigenvalue nearest to it
    (seeded selection - NOT max-gamma, which is unreliable), and accept if they agree
    within max(gamma_rel_tol*|gamma|, gamma_abs_tol). Damped/marginal kys
    (gamma < gamma_marginal) are skipped: DMD cannot assess them and the marginal exit
    already handles them (flux ~ 0).

    Returns dict with per-ky records and 'rejected_fraction' over assessable unstable kys,
    or None if the raw folder is absent.
    """
    raw_folder = Path(raw_folder)
    if not raw_folder.exists():
        return None

    def _ky_of(p):
        m = re.search(r"KY_([0-9.]+)_PX0", str(p))
        return float(m.group(1)) if m else np.nan

    records = []
    for kydir in sorted(raw_folder.glob("**/KY_*_PX0_*"), key=_ky_of):
        ky = _ky_of(kydir)
        rec = {"ky": ky, "status": "no_data", "gamma_freq": None, "omega_freq": None,
               "gamma_dmd": None, "omega_dmd": None, "reason": None}
        freq_file = kydir / "out.cgyro.freq"
        if not freq_file.exists():
            rec["reason"] = "missing out.cgyro.freq"
            records.append(rec)
            continue
        try:
            fr = np.loadtxt(freq_file)
            fr = fr[None, :] if fr.ndim == 1 else fr
            omega_f, gamma_f = float(fr[-1, 0]), float(fr[-1, 1])
        except Exception as e:
            rec["reason"] = f"could not parse out.cgyro.freq ({e})"
            records.append(rec)
            continue
        rec["gamma_freq"], rec["omega_freq"] = gamma_f, omega_f

        if gamma_f < gamma_marginal:
            rec["status"] = "stable"
            records.append(rec)
            continue

        eigs, reason = _dmd_eigenvalues(kydir)
        if eigs is None:
            rec["reason"] = reason
            records.append(rec)
            continue

        g_dmd, w_dmd = min(eigs, key=lambda e: abs((e[0] - gamma_f) + 1j * (e[1] - omega_f)))
        rec["gamma_dmd"], rec["omega_dmd"] = g_dmd, w_dmd
        agree = abs(g_dmd - gamma_f) <= max(gamma_rel_tol * abs(gamma_f), gamma_abs_tol)
        rec["status"] = "accepted" if agree else "rejected"
        records.append(rec)

    assessed = [r for r in records if r["status"] in ("accepted", "rejected")]
    rejected = [r for r in assessed if r["status"] == "rejected"]
    return {
        "records": records,
        "n_assessed": len(assessed),
        "n_rejected": len(rejected),
        "rejected_kys": [r["ky"] for r in rejected],
        "rejected_fraction": len(rejected) / len(assessed) if assessed else 0.0,
    }


class QLGYRO(SIMtools.mitim_simulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        def code_call(folder, p, n=1, nomp=1, additional_command="", **kwargs):
            return f"qlgyro -e {folder} -n {n} -nomp {nomp} {additional_command}"

        def code_slurm_settings(name, minutes, total_cores_required, cores_per_code_call, type_of_submission, array_list=None, **kwargs_slurm):
            slurm_settings = {
                "name": name,
                "minutes": minutes,
            }

            machine_settings = CONFIGread.machineSettings(code="qlgyro")

            if type_of_submission == "slurm_standard":
                slurm_settings["ntasks"] = total_cores_required // cores_per_code_call
                if machine_settings["gpus_per_node"] > 0:
                    slurm_settings["gpuspertask"] = cores_per_code_call
                else:
                    slurm_settings["cpuspertask"] = cores_per_code_call
            elif type_of_submission == "slurm_array":
                slurm_settings["ntasks"] = 1
                if machine_settings["gpus_per_node"] > 0:
                    slurm_settings["gpuspertask"] = cores_per_code_call
                else:
                    slurm_settings["cpuspertask"] = cores_per_code_call
                slurm_settings["job_array"] = ",".join(array_list)

            return slurm_settings

        self.run_specifications = {
            "code": "qlgyro",
            "input_file": "input.cgyro",
            "code_call": code_call,
            "code_slurm_settings": code_slurm_settings,
            "control_function": GACODEdefaults.addCGYROcontrol,
            "controls_file": "input.cgyro.controls",
            "state_converter": "to_cgyro",
            "input_class": CGYROinput,
            "complete_variation": None,
            "default_cores": 16,
            "output_class": QLGYROoutput,
        }

        print("\n-----------------------------------------------------------------------------------------")
        print("\t\t\t QLGYRO class module")
        print("-----------------------------------------------------------------------------------------\n")

        self.ResultsFiles_minimal = [
            "out.qlgyro.gbflux",
            "out.qlgyro.status",
            "out.qlgyro.units",
        ]

        self.ResultsFiles = self.ResultsFiles_minimal + [
            "out.qlgyro.run",
            "out.qlgyro.version",
            "out.qlgyro.ky_spectrum",
            "out.qlgyro.eigenvalue_spectrum",
            "out.qlgyro.QL_weight_spectrum",
            "out.qlgyro.field_spectrum",
            "out.qlgyro.flux_spectrum",
            "out.qlgyro.sat_geo_spectrum",
            "out.qlgyro.kxrms_spectrum",
            "out.qlgyro.taskmapping",
        ]

        self.qlgyro_input_files = {}

        # Harvest per-ky raw CGYRO dirs (field histories) on retrieval, enabling the DMD
        # agreement gate at read time. ~O(100 MB)/radius after trimming; caller cleans up.
        self.keep_cgyro_raw = False

    def _organize_results(self, code_executor, tmpFolder, filesToRetrieve):

        if self.keep_cgyro_raw:
            print("\t- Harvesting per-ky raw CGYRO folders for DMD gate")
            for subfolder_sim in code_executor:
                for rho in code_executor[subfolder_sim].keys():
                    dest = code_executor[subfolder_sim][rho]["folder"] / f"cgyro_raw_{rho:.4f}"
                    kydirs = sorted((tmpFolder / subfolder_sim / f"rho_{rho:.4f}").glob("KY_*_PX0_*"))
                    if not kydirs:
                        print(f"\t!! no per-ky CGYRO folders found for rho={rho:.4f} (DMD gate will be skipped)", typeMsg="w")
                        continue
                    if dest.exists():
                        IOtools.shutil_rmtree(dest)
                    dest.mkdir(parents=True)
                    for kydir in kydirs:
                        shutil.move(str(kydir), str(dest / kydir.name))
                        for pattern in _CGYRO_RAW_TRIM_PATTERNS:
                            for f in (dest / kydir.name).glob(pattern):
                                f.unlink()

        super()._organize_results(code_executor, tmpFolder, filesToRetrieve)

    def prep(self, mitim_state, FolderGACODE, cold_start=False, forceIfcold_start=False):
        cdf = super().prep(
            mitim_state,
            FolderGACODE,
            cold_start=cold_start,
            forceIfcold_start=forceIfcold_start,
        )

        qlgyro_inputs_folder = self.FolderGACODE / "qlgyro_inputs"
        for rho in self.rhos:
            qlgyro_controls = GACODEdefaults.addQLGYROcontrol("default")
            qlgyro_controls["GAMMA_E"] = self.inputs_files[rho].plasma.get("GAMMA_E", qlgyro_controls["GAMMA_E"])

            qlgyro_input = QLGYROinput.initialize_in_memory(qlgyro_controls)

            qlgyro_file = qlgyro_inputs_folder / f"rho_{rho:.4f}" / "input.qlgyro"
            qlgyro_file.parent.mkdir(parents=True, exist_ok=True)
            qlgyro_input.file = qlgyro_file
            qlgyro_input.write_state()

            self.qlgyro_input_files[rho] = qlgyro_file

        return cdf

    def _run_prepare(self, subfolder_simulation, additional_files_to_send=None, **kwargs):
        merged_files = {} if additional_files_to_send is None else {rho: list(files) for rho, files in additional_files_to_send.items()}

        for rho in self.rhos:
            merged_files.setdefault(rho, [])
            if rho in self.qlgyro_input_files:
                merged_files[rho].append(self.qlgyro_input_files[rho])

        return super()._run_prepare(
            subfolder_simulation,
            additional_files_to_send=merged_files,
            **kwargs,
        )


class QLGYROoutput(SIMtools.GACODEoutput):
    def __init__(self, folder, suffix=None, dmd_gate=False, dmd_gate_options=None, **kwargs):
        super().__init__()

        self.folder = Path(folder)
        self.suffix = suffix or ""

        self.inputFile = None
        self.input_qlgyro = None

        input_cgyro_file = self.folder / f"input.cgyro{self.suffix}"
        if input_cgyro_file.exists():
            self.inputFile = input_cgyro_file.read_text()

        rho_label = self.suffix[1:] if self.suffix.startswith("_") else self.suffix
        if rho_label:
            qlgyro_input_file = self.folder / "qlgyro_inputs" / f"rho_{rho_label}" / "input.qlgyro"
            if qlgyro_input_file.exists():
                self.input_qlgyro = qlgyro_input_file.read_text()

        parsed_input = SIMtools.buildDictFromInput(self.inputFile) if self.inputFile else {}
        n_species = int(parsed_input.get("N_SPECIES", 0))

        gbflux_file = self.folder / f"out.qlgyro.gbflux{self.suffix}"
        if not gbflux_file.exists():
            raise FileNotFoundError(f"Could not find {gbflux_file}")

        gbflux = np.fromstring(gbflux_file.read_text(), sep=" ")
        if n_species == 0:
            if gbflux.size % 4 != 0:
                raise ValueError(f"Unexpected QLGYRO gbflux length {gbflux.size} in {gbflux_file}")
            n_species = gbflux.size // 4

        expected_length = 4 * n_species
        if gbflux.size != expected_length:
            raise ValueError(f"Expected {expected_length} entries in {gbflux_file}, found {gbflux.size}")

        gamma = gbflux[0:n_species]
        heat = gbflux[n_species:2 * n_species]
        momentum = gbflux[2 * n_species:3 * n_species]
        exchange = gbflux[3 * n_species:4 * n_species]

        self.Gamma_e = float(gamma[0])
        self.Gamma_i = np.array(gamma[1:])
        self.Qe = float(heat[0])
        self.Qi_species = np.array(heat[1:])
        self.Pi_e = float(momentum[0])
        self.Pi_i = np.array(momentum[1:])
        self.Se = float(exchange[0])
        self.Si = np.array(exchange[1:])

        self.Ge_mean = self.Gamma_e
        self.Qe_mean = self.Qe
        self.Qi_mean = float(np.sum(self.Qi_species))
        self.Mt_mean = float(np.sum(self.Pi_i))
        self.Qie_mean = self.Se

        self.Ge_std = 0.0
        self.Qe_std = 0.0
        self.Qi_std = 0.0
        self.Mt_std = 0.0
        self.Qie_std = 0.0

        status_file = self.folder / f"out.qlgyro.status{self.suffix}"
        self.status = status_file.read_text() if status_file.exists() else ""
        if self.status and "unconverged" in self.status.lower():
            print(f"\t- QLGYRO status reports unconverged points in {IOtools.clipstr(status_file)}", typeMsg="w")

        # Seeded-DMD agreement gate on the harvested per-ky field histories (EMA-exit companion)
        self.dmd_gate = None
        if dmd_gate:
            raw_folder = self.folder / f"cgyro_raw{self.suffix}"
            self.dmd_gate = dmd_agreement_gate(raw_folder, **(dmd_gate_options or {}))
            if self.dmd_gate is None:
                print(f"\t- DMD gate skipped: no raw CGYRO folders at {IOtools.clipstr(raw_folder)}", typeMsg="i")
            elif self.dmd_gate["n_assessed"] == 0:
                reasons = {r["reason"] for r in self.dmd_gate["records"] if r["reason"]}
                print(f"\t- DMD gate could not assess any ky ({'; '.join(reasons) if reasons else 'no unstable kys'})", typeMsg="i")
            else:
                fr = self.dmd_gate["rejected_fraction"]
                msg = (f"\t- DMD gate{self.suffix}: {self.dmd_gate['n_rejected']}/{self.dmd_gate['n_assessed']} "
                       f"unstable kys rejected (under-converged eigenvalue)")
                if self.dmd_gate["n_rejected"] > 0:
                    print(msg + f" -> kys {['%.2f' % k for k in self.dmd_gate['rejected_kys']]}", typeMsg="w")
                else:
                    print(msg)


class QLGYROinput(SIMtools.GACODEinput):
    def __init__(self, file=None):
        super().__init__(
            file=file,
            controls_file=__mitimroot__ / "templates" / "input.qlgyro.controls",
            code="QLGYRO",
        )