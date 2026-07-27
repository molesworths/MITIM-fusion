"""exb.py -- nonlocal ExB shear: rms-smeared gamma_E and the external per-ky quench.

Primary ExB treatment (quench-primary, cross-code consistent):
  * TGLF/QLGYRO run SHEAR-FREE (VEXB_SHEAR=0, ALPHA_QUENCH=0; QLGYRO internal
    rotation quench off) -- injected by ``nonlocality.inject_run_settings``.
  * The suppression is applied externally, per ky, on the code's own flux
    spectrum:

        gamma_net(ky) = max( gamma(ky) - alpha_E * gamma_E_eff, 0 )
        F(ky)         = ( gamma_net(ky) / gamma(ky) )^p          (p ~ 2)
        flux_quenched = sum_ky F(ky) * flux(ky)

    Waltz-Kerbel-Milovich quench rule per ky; alpha_E ~ O(1) anchored to the
    nonlinear-GYRO alpha_E scans (Kinsey-Waltz-Candy). ky structure comes from
    the case's own gamma(ky) spectrum (low ky quenched first because gamma(ky)
    is smallest there), not from a core-fit exponent. Known, declared
    limitation (uniform across codes): no eigenmode/QL-weight modification.

  * gamma_E_eff is the RMS kernel average of the diamagnetic gamma_E over
    lambda_c (see correlation.py), in a/c_s units to match gamma(ky).

Verified file identities this module relies on (H_LowColl SAT3, 2026-07-22):
  * sum of out.tglf.sum_flux_spectrum rows == out.tglf.gbflux per species/type
    (dky quadrature weights are folded into the rows), so per-ky factors applied
    to the rows re-sum consistently;
  * out.tglf.field_spectrum phi^2 is NOT the SAT3 QL intensity multiplier
    (ky-dependent x4-15 mismatch) -- never reconstruct flux from QL x phi^2.
"""

import re
import numpy as np

__all__ = ["read_tglf_spectra", "quench_factors", "fit_exb_envelope", "CHANNEL_TYPES"]

_NUM = re.compile(r"^\s*[-+0-9.]")

# sum_flux_spectrum column layout per species block
_TYPE_COLS = {"particle": 0, "energy": 1, "stress_tor": 2, "stress_par": 3, "exchange": 4}


def _read_ky(fn):
    return np.loadtxt(fn, skiprows=2)


def _read_eigenvalues(fn):
    """(nky, 2*nmodes) -> gamma of the most-unstable mode per ky."""
    ev = np.atleast_2d(np.loadtxt(fn, skiprows=2))
    gammas = ev[:, 0::2]
    return gammas.max(axis=1)


def _read_sum_flux(fn):
    """{species_index(1-based): (nky, 5)} from out.tglf.sum_flux_spectrum."""
    lines = open(fn).readlines()
    blocks, out = [], {}
    for i, l in enumerate(lines):
        if "species" in l.lower():
            blocks.append(i)
    blocks.append(len(lines))
    for bi in range(len(blocks) - 1):
        rows = []
        for l in lines[blocks[bi] + 1: blocks[bi + 1]]:
            if _NUM.match(l):
                vals = l.split()
                try:
                    rows.append([float(v) for v in vals])
                except ValueError:
                    continue
        out[bi + 1] = np.array(rows)
    return out


def read_tglf_spectra(folder, rho):
    """ky grid, gamma(ky) (most-unstable mode), and per-species flux rows for the
    surface labelled ``rho`` in a TGLF run folder. Fails loudly if the spectrum
    files are missing (keep_files must retain them)."""
    suffix = f"{rho:.4f}"
    paths = {name: folder / f"out.tglf.{name}_{suffix}"
             for name in ("ky_spectrum", "eigenvalue_spectrum", "sum_flux_spectrum")}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise RuntimeError(
            "[nonlocality] TGLF spectrum files missing (transport keep_files must "
            f"retain spectra for the nonlocal quench): {missing}")
    ky = _read_ky(paths["ky_spectrum"])
    gamma = _read_eigenvalues(paths["eigenvalue_spectrum"])
    flux = _read_sum_flux(paths["sum_flux_spectrum"])
    n = min(len(ky), len(gamma))
    return ky[:n], gamma[:n], {sp: arr[:n] for sp, arr in flux.items() if arr.size}


def fit_exb_envelope(roa, gamma_exb_abs, form="gauss", window=(0.85, 1.0)):
    """Single smooth envelope fit to |gamma_E|(roa) across the pedestal.

    Reduced-model alternative to the local / rms-smeared shear. The local
    |gamma_E| has a sharp mid-pedestal NOTCH (-> 0 where dEr/dr=0, the Er-well
    bottom); the rms smear only partially fills it (lambda_c < well width), so a
    knot landing on the notch still sees a flux spike and the flux-match residual
    is ill-conditioned. Replacing |gamma_E| with ONE pedestal-wide bump makes the
    quench factor monotone-in-radius (no notch) and hence solver-friendly, at the
    cost of local fidelity: it over-suppresses the notch (~4x vs the physical rms
    estimate) and slightly under-suppresses the shear-layer flanks. This is the
    approximation reduced pedestal models make implicitly with a single ExB
    factor. See offline study (H_LowColl/H_HighColl iter-0).

    Returns |gamma_E|_env evaluated at every input ``roa`` (extrapolated outside
    ``window``). Falls back to the input array on fit failure.
    """
    from scipy.optimize import curve_fit
    roa = np.asarray(roa, float)
    g = np.abs(np.asarray(gamma_exb_abs, float))
    m = (roa >= window[0]) & (roa <= window[1])
    if m.sum() < 5:
        return g
    x, y = roa[m], g[m]
    if form == "gauss":
        f = lambda r, A, r0, w, c: A * np.exp(-0.5 * ((r - r0) / w) ** 2) + c
    elif form in ("lorentz", "lorentzian"):
        f = lambda r, A, r0, w, c: A * w ** 2 / ((r - r0) ** 2 + w ** 2) + c
    else:
        raise ValueError(f"[nonlocality] unknown exb_envelope form {form!r}")
    p0 = [max(y.max(), 1e-30), x[np.argmax(y)], 0.03, 0.0]
    try:
        popt, _ = curve_fit(f, x, y, p0=p0, maxfev=40000)
        return np.clip(f(roa, *popt), 0.0, None)
    except Exception:
        return g


def quench_factors(folder, rho, gamma_e_eff, alpha_e=1.0, p=2.0,
                   impurity_species=3):
    """Per-channel quench factors (quenched/raw flux ratio) for one surface.

    gamma_e_eff : |gamma_E,eff| in a/c_s units (matches eigenvalue_spectrum).
    Returns (factors, diags):
      factors : {"Qe","Qi","Ge","GZ","Mt","Qie"} -> float in [0, 1]
      diags   : {"F_ky","ky","gamma","fully_quenched_kys"}

    Channel aggregation (species 1 = electrons, >=2 ions; verified ordering):
      Qe = sp1 energy; Qi = sum over ion species energy; Ge = sp1 particle;
      GZ = impurity_species particle; Mt = all-species toroidal stress;
      Qie = sp1 exchange. A channel with |raw total| ~ 0 gets factor 1.
    """
    ky, gamma, flux = read_tglf_spectra(folder, rho)
    g = np.clip(gamma, 1e-10, None)
    F = np.clip((g - alpha_e * abs(gamma_e_eff)) / g, 0.0, None) ** p

    def _ratio(rows_list, col):
        raw = sum(r[:, col].sum() for r in rows_list)
        if abs(raw) < 1e-12:
            return 1.0
        quenched = sum((F * r[:, col]).sum() for r in rows_list)
        return float(quenched / raw)

    sp_e = [flux[1]] if 1 in flux else []
    sp_ions = [flux[s] for s in flux if s >= 2]
    sp_imp = [flux[impurity_species]] if impurity_species in flux else []

    factors = {
        "Qe":  _ratio(sp_e, _TYPE_COLS["energy"]),
        "Qi":  _ratio(sp_ions, _TYPE_COLS["energy"]),
        "Ge":  _ratio(sp_e, _TYPE_COLS["particle"]),
        "GZ":  _ratio(sp_imp, _TYPE_COLS["particle"]),
        "Mt":  _ratio(list(flux.values()), _TYPE_COLS["stress_tor"]),
        "Qie": _ratio(sp_e, _TYPE_COLS["exchange"]),
    }
    diags = {"F_ky": F, "ky": ky, "gamma": gamma,
             "fully_quenched_kys": ky[F <= 0.0]}
    return factors, diags
