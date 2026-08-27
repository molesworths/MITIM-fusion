# PORTALS-E(dge)

Pedestal/edge flux-matching on top of PORTALS. `portals_edge` inherits everything from `portals`
and swaps the model evaluation to a `powerstate_edge`, which inserts boundary-condition, rotation,
neutral, impurity, nonlocal and ELM physics between profile reconstruction and target evaluation.

**Unchanged from core PORTALS:** `prep()`, the BO loop (acquisition, GP surrogates, transforms),
the relaxation initialization, and the analysis/plot layer.
**Changed:** the profile parameterization at the edge, the LCFS boundary treatment, the target
evaluator, and the physics chain listed below.

---

## Quick start

```python
from mitim_modules.portals.PORTALSedge import portals_edge
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.opt_tools import STRATEGYtools

portals_fun = portals_edge(folder)                       # or portals_namelist=<yaml>
portals_fun.portals_parameters["solution"]["predicted_roa"] = [0.88, 0.91, 0.94, 0.97]
portals_fun.portals_parameters["edge_options"] = {...}   # see below

plasma_state = PROFILEStools.gacode_state(input_gacode)
portals_fun.prep(plasma_state, cold_start=True, askQuestions=False)
STRATEGYtools.MITIM_BO(portals_fun, cold_start=True, askQuestions=False).run()
```

Two starting points, both fully annotated:

| File | Use |
|---|---|
| [`templates/namelist.portals_edge.yaml`](../../../templates/namelist.portals_edge.yaml) | Every option at its default, with comments. Pass as `portals_namelist=`. |
| [`templates/workflow.portals_edge.py`](../../../templates/workflow.portals_edge.py) | Copy-and-edit launch script, including the Zeff calibration helper. |

The whole YAML becomes `portals_fun.portals_parameters`, so anything in it can equivalently be set
from the launch script *before* `prep()`. There is also a CLI: `run_portals_edge <folder> --edge-*`
(a subset of the options), and `read_portals_edge <folder>` for analysis/plots.

---

## Evaluation chain

`powerstate_edge.calculate()`, with `>>>` marking what edge adds over core PORTALS:

```
>>> 1.  calculateBoundaryConditions()   LCFS n, T and their scale lengths from the SOL model
    2.  modify(X)                       reconstruct profiles from the DVs; pin the LCFS BCs
    3.  calculateProfileFunctions()     GB units, nuei, rho_s, c_s, R, Bp, BT
>>> 3b. calculateRotation()             w0, Er, E_rad, vexb_shear, mach — autodiff-safe
>>> 3c. calculateNeutrals()             n0, S_ion_main, nu_ioniz_main, tau_n0
>>> 3d. calculateImpurities()           Aurora charge states, radiation, Zeff
    4.  calculateTargets()              analytical_model_edge
    5.  calculateTransport()            TGLF / NEO (or CGYRO / QLGYRO)
>>> 5b. nonlocality.calculateNonlocal() per-ky ExB quench, then turbulence spreading
>>> 6.  calculateElm()                  peeling-ballooning penalty on turbulent fluxes
    7.  calculateMetrics()              residuals
```

---

## `solution` block — what differs at the edge

| Key | Edge choice | Why |
|---|---|---|
| `predicted_roa` | `[0.88, 0.91, 0.94, 0.97]` | Specify in r/a, not rho. Also the parameterizer knots. |
| `predicted_channels` | `["te","ti","ne"]` | Add `"nZ"` to predict the impurity density (see below). |
| `parameterizer` | `"SplineMtanh"` | mtanh backbone + a linear corrector that pins interior `aLy` knots. Aliases: `spline_mtanh`, `mtanh_spline`, `MtanhSpline`, `SplineMtanhAnalytic`. |
| `parameterizer_options` | `{knots: ..., defined_on: "aLy"}` | `SplineMtanh` accepts `defined_on: "aLy"` only. Knots must reach the pedestal gradient peak. |
| `exploration_ranges.limits_are_relative` | `false` | Absolute `aLy` bounds. A relative box around the initial guess censors the pedestal solution set. |

Other parameterizers: `"spline"` (use `parameterizer_options.spline_type: "akima"` for Akima —
the bare name `"akima"` is **not** accepted by `powerstate_edge`), `"mtanh"` (true global mtanh,
DVs `[log_A, log_u1, delta, m]`), `"ClampedHermite"`.

Transport presets: `code_settings: "edge"` for both TGLF (SAT3 + `ALPHA_ZF=-1` + electromagnetic +
pedestal resolution) and NEO (`ROTATION_MODEL=1`; per-rho `DPHI0DR` injected from
`plasma["E_rad"]`).

---

## `edge_options`

A single top-level block, read by `portals_edge` → `powerstate_edge`. All keys optional; defaults
shown. Full annotated listing in `templates/namelist.portals_edge.yaml`.

### Domain and boundary conditions

| Key | Default | Notes |
|---|---|---|
| `domain_roa` | `null` | Trim the plasma grid, e.g. `[0.87, 1.0]`. Only the lower bound is used. Takes priority over `domain_rho`. |
| `domain_rho` | `null` | Same in sqrt-toroidal-flux. |
| `lcfs_bc` | `{}` | Initial separatrix anchors (`te`/`ti` keV, `ne` 1e19 m⁻³, `aLte`/`aLti`/`aLne`). Empty = read and freeze the input-profile values. |
| `bc_model` | `"FixedInitial"` | `{y: ..., aLy: ...}` or a legacy alias. |
| `bc_model_options` | `{}` | Flat keys go to all sub-models; nest under `y` / `aLy` / `combined` to target one. |

`y` models: `Fixed` (freeze), `TFTP` (two-fluid two-point — back-calculates upstream `n,T` from
divertor targets; options `ne_target` 3.0, `te_target` 0.005, `ti_target` null, `fq_e`/`fq_i` 0.5,
`fmom` 0.5, `lambda_q_scalar` 1.0).

`aLy` models: `Fixed`, `PeretSSF` (Peret 2025 SSF turbulent flux-decay lengths; `G0` 1.25,
`shear_ref` 5.0 — sign is +LSN/−USN, `f_Delta` 1.0, `Lambda`, `gfile_path`), `EichManz` (SepOS
scaling, no options), `Ensemble` (mean of the two; per-model options nest under `PeretSSF` /
`EichManz`).

Combined-solver options: `max_iter` 100, `tol` 1e-3, `damping` 0.5, `Zeff`, `Lpar`.

Aliases: `"Fixed"`/`"FixedInitial"` → `{Fixed, Fixed}`; `"TwoFluid_PeretSSF"` → `{TFTP, PeretSSF}`;
`"TwoFluid_EichManz"`/`"Tftp_SepOS"` → `{TFTP, EichManz}`; `"Synthesis"` → the coupled
`TwoFluidSynthesis` solver.

### Impurity density — the three conventions

This is the option most likely to change your answer, so it is worth being explicit about who owns
the impurity **density level**.

| `impurity_density_source` | Who sets the level | Zeff calibration |
|---|---|---|
| `"charge_state_model"` (default) | Aurora's steady-state solve, amplitude set by `source_rate` | **Active** — calibrate `source_rate` to the measured LCFS Zeff |
| `"prescribed"` (aliases `experimental`, `initial`) | The initial profile's `f_Z` carried with `ne`. Aurora still supplies the charge-state *fractions* for radiation, `qpar_imp`, `qpar_Z` and Zeff | **Inert** — the solve no longer sets the level |
| `"nZ"` in `predicted_channels` | The optimizer (one DV per knot, plus the `GZ`/`CZ` residual). Overrides both rows above; `impurity_density_source` then only sets the starting profile | Inert |

Related keys:

| Key | Default | Notes |
|---|---|---|
| `impurity_representation` | `"charge_density"` | How the charge-state ladder collapses onto the single GACODE impurity species. `charge_density`: `n_rep = Σ_z z n_z / Z_imp`, exact in the charge density (and therefore in quasineutrality, dilution and the impurity drive). `fully_stripped`: `n_z(Z_imp)` alone — legacy, for reproducing archived runs only. |
| `impurity_zeff_override` | `true` | Write the exact all-stage Zeff per radius into the transport-code inputs. The single-species representation makes the code-derived ZEFF high by `Z_imp/⟨Z⟩` (~9% of Zeff−1 at the pedestal top, >60% near the LCFS). |
| `charge_state_model` | `"Null"` | `"Null"` or `"Aurora"`. |

`charge_state_model_options` (Aurora): `imp` `"C"`, `main_element` `"D"`, `D_z_m2_s` 0.1,
`V_z_m_s` −0.5 (linear ramp `V(r) = V_z_m_s · r/r_lcfs`), `aLnZ_profile` null (scalar or `(x, y)`
on `x = r/r_lcfs` — **replaces** `V_z_m_s` as the peaking input; `a/L_nZ = V(r)/D` holds exactly in
this model), `source_rate` 1e21, `cxr_flag` false (uses `plasma["n0"]`, so it needs a neutral
model), `max_dilution_fraction` 1.0, `max_source_rate_iters` 5, `update_ni_charge_balance` true,
`main_ion_species_index` 0, `verbose` false.

> Unrecognized keys are **warned about, not raised**. `V0_m_s` is not `V_z_m_s`.

#### `D_z_m2_s` is not a free knob

Rescaling (`"prescribed"` or predicted `nZ`, see the table above) pins the impurity **total** but
leaves Aurora owning the charge-state **fractions**, and those fractions still come from the solve
— i.e. from the competition between transport at `D_z_m2_s` and the atomic ionization/recombination
rates. The impurity electron source
`qpar_imp = Σ_z scd_z n_z − Σ_z acd_z n_{z+1}` (which feeds `qpar_wall`, hence the `Ge` target) is a
near-cancelling difference of two large sums over that ladder, so it is **very sensitive to `D_Z`**
even at fixed impurity density. Do not treat `D_z_m2_s` as a shape-only or cosmetic parameter: it
moves the electron particle target.

The second constraint is that `D_Z` and the pinch are not independent. Under `aLnZ_profile` the
model enforces `V(r) = −D · (a/L_nZ)(r) / a` pointwise, so the peaking factor you ask for *fixes*
the pinch once `D` is chosen. For the edge peaking factors in these cases:

| `D_z_m2_s` | implied \|V_Z\| | verdict |
|---|---|---|
| ~0.1 m²/s | ~1–10 m/s | physically reasonable |
| ~1 m²/s | ~50–100 m/s | **not realistic** — no measured edge impurity pinch is this large |

So a "large-D" run is not a conservative choice: it buys the same density shape only by demanding
an unphysical convective velocity, and it changes `qpar_imp` on the way.

**Recommendation.** Unless the D/V peaking factor (the same quantity as `a/L_nZ`) can be estimated
by proxy — from a measured impurity profile, a companion impurity-transport analysis, or a
neoclassical/turbulent D/V estimate at these radii — do **actual impurity transport modelling**
rather than picking `D_z_m2_s` / `V_z_m_s` by hand. When a proxy *is* available, supply it as
`aLnZ_profile` (not `V_z_m_s`), keep `D_z_m2_s` near 0.1, and report both numbers with the run.

Zeff−1 is linear in `source_rate` and zero at zero, so one measurement calibrates it:
`source_cal = source_ref · (Zeff_target − 1)/(Zeff_ref − 1)`. Note that `solve()` alone does not
set `plasma["Zeff"]` — quasineutrality does — and that the charge-state + QN chain mutates the
powerstate, so measure on a throwaway build and re-build cleanly at the calibrated rate
(`calibrate_impurity_source` in the workflow template does exactly this).

### Neutrals

`neutral_model`: `"Null"` (default) or `"Analytic"`. Writes `n0`, `S_ion_main`, `nu_ioniz_main`,
`tau_n0`, `Ge_core_reinject`.

`neutral_model_options`: `source_rate` 1e21 (D⁰ crossing the LCFS inward, sets the `n0`
amplitude), `mu_amu` 2.014, `include_cx` true, `two_population` true (coupled cold Franck-Condon +
CX-generated hot; `false` reverts to the single-population Knudsen-selected solver, which
under-estimates opacity), `T_cold_eV` 3.0, `Kn_thresh` 0.3, `Kn_eval_fraction` 0.3.

### Rotation / E_r

`rotation_options` (forwarded to `edge_tools.rotation`) produces `w0`, `Er`, `E_rad` (NEO
`DPHI0DR`), `vexb`, `gamma_exb`, `vexb_shear` (TGLF `VEXB_SHEAR`), `mach`, `gamma_p`, `w0_n`,
`aLw0_n` — all differentiable on the analytic backend.

| Key | Default | Notes |
|---|---|---|
| `mode` | `"analytic"` | `"vgen"` runs an external NEO DKE solve: higher fidelity, **not** differentiable. |
| `K_neo` | `"sauter"` | Neoclassical poloidal-flow coefficient; `0.0` = pure diamagnetic Er. |
| `oversample` | `2` | Refine before differentiating — `gamma_E ~ d²φ/dr²` is under-resolved on the coarse grid. |
| `vtor_source` | `"zero"` | `"zero"` = vtor=0 closure. `"user"` reads `vtor_pairs=[(rho, v_m_s), ...]`. `"extract_initial"` holds the reconstructed Vtor fixed and lets the diamagnetic Er evolve. |
| `vtor_extract_smooth` | `5` | Odd window (grid points) smoothing the backed-out Vtor at the foot. 5–7 useful; >~9 erodes pedestal-top shear; ≤1 disables. |
| `vgen_every` / `vgen_drho` / `vgen_options` | `1` / `0.01` / `{}` | `mode: "vgen"` only. |

The model is stateless by design: no under-relaxed `w0` is carried across iterations, so the
DV→profile map stays reproducible. Relaxation belongs in the outer solver.

### ELMs

`elm_model`: `"Null"` (default), `"AnalyticPB"` (inline s–α peeling-ballooning criterion), or
`"EPED"` (needs `elm_model_options.eped_folder`). The result multiplies **turbulent** fluxes only.

`elm_model_options` (AnalyticPB): `stiffness` 10.0, `stiffness_power` 1.0, `s_hat_min` 0.1,
`s_peel_frac` 1.5, `roa_min` 0.8, `geometry_correction` true.

### Nonlocal corrections

`nonlocal_model`: `"Null"` (default) or `"Analytic"`. When active, the transport code is forced
shear-free (TGLF `VEXB_SHEAR=0`, `ALPHA_QUENCH=0`; QLGYRO `ROTATION_FLAG=0`) so this module owns
**all** ExB physics. Quenched fluxes overwrite `*_tr_turb` (GP-facing, learnable via the
`gamma_exb_nl` feature); spread totals land in `*_tr_nl` (what `calculateMetrics` reads). The
spread matrix is frozen at the first evaluation so the real and GP paths use the same operator.

`nonlocal_model_options`: `ExB` true, `Spreading` true, `lambda_c_mult` 8.0 (C in `λ_c = C ρ_s`;
literature band 5–12), `kernel` `"gauss"`, `alpha_e` 1.0, `p` 2.0, `spread_lambda_mult` null,
`exb_envelope` null (rms smear; `"gauss"`/`"lorentzian"`/`"curvature"` give a pedestal-wide bump),
`exb_envelope_window` `(0.85, 1.0)`, `gamma_exb_nl_feature` false.
Validation gate: `lambda_c_mult → 0` with `ExB` off must reproduce the raw per-radius code output.

### Targets and objective

| Key | Default | Notes |
|---|---|---|
| `use_edge_targets` | `true` | Swap in `targets_analytic_edge.analytical_model_edge`. |
| `target_multipliers` | `{}` | Scale a fixed target source at x0 and shift the whole profile. Keys `Qe`, `Qi`, `Ge`, `GZ`, `Mt`. |
| `targets_scaled` | `false` | Normalize each residual by its target, `(cal − of)/cal`. |
| `ne_flux_channel` | `"convective"` | `"convective"`: match the convective energy flux `Ce = (3/2) Te Ge`, so every channel is MW/m² and the absolute residual is a consistent total-power objective. `"particle"`: match the raw particle flux — de-conflates ne from Te, but mixes units, so target-normalized residuals are turned on automatically. |

### Surrogate structure

| Key | Default | Notes |
|---|---|---|
| `global_surrogates` | `false` | One GP per turbulent channel spanning the domain (samples from all radii pooled in gyro-Bohm space, radial label appended) instead of one GP per radius. |
| `global_surrogates_options.extra_features` | `["shear"]` | Any per-radius `powerstate.plasma` key, e.g. `"vexb_shear"`. Rotation-derived keys force the rotation model onto the transform path. |
| `nonstationary_exb` | `false` | L→H flux surrogate: `ln(GB flux)` outcome transform + a learnable-knee physics mean (ITG turn-on vs ExB suppression) + ARD Matérn residual. Note the key name is `nonstationary_exb` (module: `edge_tools.exb_nonstationary`). |
| `defined_on` | `"y"` | Fallback control-point semantics when `solution.parameterizer_options` omits it. |

---

## Uncertainty quantification

`mitim_tools.edge_tools.uq` propagates input uncertainty as a Cholesky factor through the
differentiable slice of `calculate()`, preserving correlations induced by shared parents.

```python
from mitim_tools.edge_tools.uq import UQInputs, run_edge_uq

inp = UQInputs()
inp.add_lcfs("te", 0.15); inp.add_neutral_source(0.30); inp.add_impurity_source(0.50)
out = run_edge_uq(powerstate, inp, X_dvs=powerstate.FluxMatch_Xopt,
                  mode="real_scan", folder=run_folder, inject_into=powerstate)
```

- `mode="real_scan"` (default) scans each uncertain input through the real TGLF/NEO/Aurora chain —
  `n_inputs` extra evaluations. `mode="linear"` is the cheap jvp path and needs a differentiable
  `transport_proxy`.
- `inject_training_stds=False` (default): input uncertainty is consumed at the **objective** level
  (`sigma_J`, `robust_objective`, `Sigma_r`), i.e. propagated *through* the trained map, not added
  as GP training noise.
- Pass a `folder` for `real_scan`: each column re-runs in its own `uq_<col>` subfolder, avoiding
  remote-scratch collisions.

---

## Where the code lives

| Path | Contents |
|---|---|
| `mitim_modules/portals/PORTALSedge.py` | `portals_edge`, edge `initializeProblem`, surrogate wiring |
| `mitim_modules/powertorch/STATEedge.py` | `powerstate_edge` — option defaults (`_EDGE_KEYS_DEFAULTS`) and the evaluation chain |
| `mitim_modules/powertorch/physics_models/targets_analytic_edge.py` | Edge target evaluator |
| `mitim_modules/powertorch/physics_models/parameterizers.py` | `SplineMtanhAnalytic`, `Spline`, `Mtanh`, `ClampedHermite` |
| `mitim_tools/edge_tools/boundary.py` | LCFS `y` / `aLy` models and the combined solver |
| `mitim_tools/edge_tools/charge_states.py` | Aurora steady-state charge states |
| `mitim_tools/edge_tools/neutrals.py` | Analytic D⁰ solver |
| `mitim_tools/edge_tools/rotation.py` | Torchified Er / w0 / shear |
| `mitim_tools/edge_tools/elm.py` | Peeling-ballooning models |
| `mitim_tools/edge_tools/nonlocality/` | ExB quench + turbulence spreading |
| `mitim_tools/edge_tools/exb_nonstationary.py` | Nonstationary L→H flux surrogate |
| `mitim_tools/edge_tools/uq/` | Linearized-covariance UQ |

---

## Gotchas

- `define_ranges_from_profiles` is **not supported** in edge mode and raises.
- Unknown keys in `charge_state_model_options` are warned about, not raised — a typo reads as
  "this input has no effect" and the run silently uses the default.
- Re-running `prep()` on an existing object does not propagate a changed `source_rate`; rebuild.
- `bc_model: {y: Fixed, aLy: Fixed}` with **no** `bc_model_options` is what freezes the LCFS at the
  input-profile values. Supplying overrides replaces them.
- `cxr_flag: true` needs `plasma["n0"]`, i.e. a non-`Null` `neutral_model`.
- With `nonlocal_model: "Analytic"`, any user `extraOptions` that set `VEXB_SHEAR`/`ALPHA_QUENCH`
  are overridden (with a warning).