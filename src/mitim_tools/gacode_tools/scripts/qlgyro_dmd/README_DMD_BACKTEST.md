# DMD / early-stopping backtest for linear CGYRO in QLGYRO (2026-07-22)

Adjudicates the `dmd_extract.py` note ("DMD-dominant unreliable at high ky")
and answers: can DMD cut linear-CGYRO wall-clock in the QLGYRO flux-match
harness (`transport_qlgyro.py`)?

Data: 11 radius-cases `qlgyro_*_noaln/` with preserved `cgyro_raw/` per-ky
field + flux time histories (40–277 snapshots/ky) and final native
`out.qlgyro.{eigenvalue,QL_weight,ky}_spectrum_*`. That tree is ~1.5 GB and is
NOT in the repository — point the scripts at it with `--root <dir>` or the
`QLGYRO_DMD_DATA_ROOT` environment variable. The saved result arrays
(`dmd_backtest_records.npy`, `dmd_backtest_ql_rows.npy`) ARE kept here, so the
tables below can be reproduced without the raw tree.

## Verdicts

**1. The high-ky note is misdiagnosed.** (`dmd_vs_native.py`, 706 ky-points)
The real failure predictor is a DAMPED/MARGINAL native mode (γ<0.05): 87%
failure (99% at ky≥4). ky is secondary: 20% fail at ky<1 (not safe), 56% at
ky≥4 — but strongly-growing high-ky ETG often matches native to 3–4 decimals.
Failure mode: spurious large-γ, large-|ω| (5–200) eigenvalues out-rank the
physical mode under a max-γ selection rule; the native eigenvalue is in DMD's
top-3 only 34% of failed rows.

**2. Eigenvalue truncation backtest.** (`dmd_backtest.py`) Success =
|γ−γ_nat| ≤ max(10%, 0.05), unstable modes, vs fraction of record used:

| window | cgyro-freq (baseline) | dmd-maxγ | dmd-maxamp | dmd-seeded |
|---|---|---|---|---|
| 30% | 48% | 50% | 47% | **65%** |
| 50% | 75% | 63% | 63% | 76% |
| 70% | 88% | 66% | 74% | 90% |
| 100% | 100%* | **68%** | 89% | 97% |

*by construction. max-γ selection is broken independent of window length
(68% on FULL records). Raw DMD beats simply reading `out.cgyro.freq` early
ONLY at ~30% window. dmd-seeded = DMD eigenvalue nearest the running-freq
estimate (polish-a-seed).

**3. The usable scheme: agreement-gated adaptive stop.** Stop a ky when
dmd-seeded and cgyro-freq agree within max(10%, 0.05):

| checkpoint | kys accepted | γ correct | QL weights <10% err | <20% |
|---|---|---|---|---|
| 30% of record | 48% | 90% | 73% | 91% |
| 50% | 73% | 92% | 94% | 98% |
| 70% | 88% | 98% | 99% | 100% |

→ ~2× further record cut beyond the existing `cgyro_freq.F90` early exits,
with quantified error rates.

**4. QL weights are NOT the limiter.** (`dmd_backtest_ql.py`) They are an
eigenfunction-shape property and converge FASTER than the frequency estimate
(median 1.4% error at half-window). The eigenvalue gate is the binding
condition.

**5. Damped modes: unsalvageable by any estimator** (all 10–33% at
truncation; QL weights 22–74% median error). Keep them on the existing
marginal-exit; never consult DMD when running γ ≲ 0.05.

## Convention gotchas (cost real debugging time)

- CGYRO linear `bin.cgyro.ky_flux` is ALREADY amplitude-normalized (flux trace
  flat while `kxky_phi` grows ~1e10). Do NOT divide by |φ|².
- `out.qlgyro.QL_weight_spectrum`: C-order `(nky, nmodes, ns, field, type)`,
  type fastest, species ELECTRONS-FIRST (TGLF convention) ≠ CGYRO input order.
  With reorder: computed/native = constant (2.38), 100% sign agreement.
- `out.cgyro.freq` columns are `(omega, gamma)` in that order.
- DMD needs `HIPREC_FLAG=1` runs and preserved scratch
  (`removeScratchFolders=False` monkeypatch).

## Files

- `common.py` — data-root resolution (`--root` / `$QLGYRO_DMD_DATA_ROOT`),
  native-spectrum reader, `ky` parsing
- `dmd_vs_native.py` — DMD-dominant vs native cross-check (reads the saved
  `dmd_<case>_roa<X>.txt` tables + native spectra; no raw data needed)
- `dmd_backtest.py` — eigenvalue truncation backtest (needs `cgyro_raw/`,
  pydmd, pygacode) → `dmd_backtest_records.npy`
- `dmd_backtest_ql.py` — QL-weight convergence + gate combination (needs
  `dmd_backtest_records.npy`) → `dmd_backtest_ql_rows.npy`

Field preprocessing (`cgyro_field_matrix`, `hankel_embed`) is imported from
`mitim_tools.gacode_tools.QLGYROtools`, so the backtest exercises the same code
path as the production gate rather than a copy of it.

Console entry points (after `pip install -e .`):

```
mitim_qlgyro_dmd_vs_native     --root <dir>
mitim_qlgyro_dmd_backtest      --root <dir>
mitim_qlgyro_dmd_backtest_ql   --root <dir>
```

## MITIM wiring (added 2026-07-22)

`QLGYROtools.py` has `dmd_agreement_gate()` (seeded selection, damped kys
skipped) plus raw-folder harvesting (`QLGYRO.keep_cgyro_raw`, trimmed KY_*
dirs land as `cgyro_raw_{rho}`), and `QLGYROoutput(dmd_gate=True)` runs the
gate at read time. `transport_qlgyro.py` enables it via
`simulation_options["qlgyro"]["dmd"] = {"enabled": True, "std_inflation": 1.0,
"keep_raw": False, "gate_options": {...}}` (default enabled, graceful no-op
without raw data) and inflates per-radius flux stds by
`1 + std_inflation * rejected_fraction`. Verified end-to-end against
`qlgyro_H_HighColl_roa0.95_noaln`: 31/31 kys assessed, 0 rejected at default
tolerances (full records), 27/31 rejected at 1e-4 tolerance (rejection path).

## Not yet done

- The IN-RUN adaptive stop (kill a ky at 30-50% of its record when the gate
  accepts) is NOT wired - that needs a runtime monitor or a further
  `cgyro_freq.F90` exit; current wiring is post-hoc QA -> surrogate std
  inflation only. The wall-clock win still comes from the compile-time
  EMA/marginal/ky-aware exits in the patched build.
