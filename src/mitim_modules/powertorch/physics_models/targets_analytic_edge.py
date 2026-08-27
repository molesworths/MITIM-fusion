"""
targets_analytic_edge.py
------------------------
Edge-specific target model that extends ``analytical_model`` with the following
physics additions relevant for the pedestal / SOL:

  - Fine/coarse-grid interpolation of edge-specific plasma keys
  - Aurora impurity radiation replacing the TGYRO Chebyshev contribution
    for the tracked impurity (avoids double-counting)
  - Aurora H/D main-ion neutral radiation
  - Particle flux targets with ionization sources (D⁰ + impurity net ionization)
  - Ionization power loss in the electron energy channel

Usage
-----
Pass as ``target_options['evaluator'] = analytical_model_edge`` when constructing a
``powerstate_edge`` object.

The three edge-specific methods read the following keys populated by
``powerstate_edge.calculateChargeStates()`` (which runs before ``calculateTargets()``
in the overridden ``calculate()`` sequence):

  plasma['nz_all']         (batch, rho, nZ+1)   [1e19 m⁻³]
      Impurity charge-state density profiles; index 0 = impurity neutral (Z=0).
  plasma['qrad_aurora']    (batch, rho)          [W cm⁻³]
      Total impurity radiation (line + continuum) from Aurora.
  plasma['nu_scd_imp']     (batch, rho, nZ+1)   [s⁻¹]
      Effective impurity ionisation frequency per charge state (= n_e × S_cd).
  plasma['nu_acd_imp']     (batch, rho, nZ+1)   [s⁻¹]
      Effective impurity recombination frequency per charge state (= n_e × α_cd).

Edge keys consumed
------------------
  plasma['nz_all']         (batch, rho, nZ+1)   [1e19 m⁻³]
  plasma['qrad_aurora']    (batch, rho)          [W cm⁻³]
  plasma['nu_scd_imp']     (batch, rho, nZ+1)   [s⁻¹]
  plasma['nu_acd_imp']     (batch, rho, nZ+1)   [s⁻¹]
  plasma['n0']             (batch, rho)          [1e19 m⁻³]
  plasma['S_ion_main']     (batch, rho)          [1e19 m⁻³ s⁻¹]
  plasma['tau_n0']         (batch, rho)          [dimensionless]
    plasma['qpar_main']     (batch, rho)          [1e20 m⁻³ s⁻¹]
    plasma['qpar_imp']      (batch, rho)          [1e20 m⁻³ s⁻¹]
    plasma['qpar_wall']     (batch, rho)          [1e20 m⁻³ s⁻¹]
    plasma['qpar_Z']        (batch, rho)          [1e20 m⁻³ s⁻¹]
"""

from mitim_tools.misc_tools import PLASMAtools
import numpy as np
import torch

from mitim_modules.powertorch.physics_models.targets_analytic import analytical_model
from mitim_tools.misc_tools.LOGtools import printMsg as print


# Edge-specific plasma keys that require grid interpolation
_EDGE_KEYS_2D = ("n0", "tau_n0", "S_ion_main", "nu_ioniz_main", "qrad_aurora", "qiziz_loss", "Zeff_exact")
_EDGE_KEYS_3D = ("nz_all", "nu_scd_imp", "nu_acd_imp")


def _ensure_1e19_units(x: "torch.Tensor", name: str, threshold: float = 1e16) -> "torch.Tensor":
    """
    Ensure internal MITIM normalization (1e19-based units) for density-like arrays.

    Some upstream paths can provide SI values (m^-3 or m^-3 s^-1) while this
    model assumes 1e19-based storage. If values are clearly SI-scale, convert
    by 1e-19 to prevent O(1e19) flux blow-ups.
    """
    if not isinstance(x, torch.Tensor) or x.numel() == 0:
        return x

    vmax = x.detach().abs().max().item()
    if vmax > threshold:
        print(
            f"[analytical_model_edge] {name} appears to be in SI units "
            f"(max={vmax:.3e}); converting to 1e19-based units.",
            typeMsg="w",
        )
        return x * 1e-19

    return x


def _edge_postprocessing(powerstate, integrated_targets=None, force_zero_particle_flux=False, relative_error_assumed=1.0):
    """
    Shared postprocessing for edge target models.

    Plugs in fixed targets with integrated source contributions, computes convective fluxes,
    assigns standard errors, and produces GB-normalised outputs.
    """
    p = powerstate.plasma
    P = integrated_targets

    # **************************************************************************************************
    # Combine edge targets with integrated source contributions
    # **************************************************************************************************

    # P may contain 2 blocks (QeMWm2, QiMWm2 only — legacy / no qpar evolution)
    # or 4 blocks (QeMWm2, QiMWm2, Ge, GZ — edge flux_integrate always cats all four).
    # Use the batch size to determine the block stride.
    batch = p["te"].shape[0]
    n_blocks = P.shape[0] // batch  # 2 (legacy) or 4 (edge with ge/gz blocks)

    p["QeMWm2"]  = p["Qe_edgetargets"] + P[         : batch,     :]  # MW/m^2
    p["QiMWm2"]  = p["Qi_edgetargets"] + P[batch    : 2 * batch, :]  # MW/m^2

    if n_blocks >= 4:
        p["Ge1E20m2"] = p["Ge_edgetargets"] + P[2 * batch : 3 * batch, :]  # 1E20/s/m^2
        p["GZ1E20m2"] = p["GZ_edgetargets"] + P[3 * batch :,            :]  # 1E20/s/m^2
    else:
        p["Ge1E20m2"] = p["Ge_edgetargets"]   # 1E20/s/m^2
        p["GZ1E20m2"] = p["GZ_edgetargets"]   # 1E20/s/m^2

    # Core-bound neutral escape (hot neutrals crossing the inner boundary) re-emerges
    # in steady state as an outward electron flux; add it as a rarefied enclosed flow
    # (Φ_inner/volp), analogous to the frozen wall term in Ge_edgetargets.
    if "Ge_core_reinject" in p:
        p["Ge1E20m2"] = p["Ge1E20m2"] + p["Ge_core_reinject"]  # 1E20/s/m^2

    p["MtJm2"] = p["Mt_edgetargets"]  # J/m^2  (no integrated source contribution)

    if force_zero_particle_flux:
        p["Ge1E20m2"] = p["Ge1E20m2"] * 0

    # Convective fluxes
    p["Ce"] = PLASMAtools.convective_flux(p["te"], p["Ge1E20m2"])  # MW/m^2
    p["CZ"] = PLASMAtools.convective_flux(p["te"], p["GZ1E20m2"])  # MW/m^2

    # **************************************************************************************************
    # Error
    # **************************************************************************************************

    variables_to_error = ["QeMWm2", "QiMWm2", "Ce", "CZ", "MtJm2", "Ge1E20m2", "GZ1E20m2"]

    for i in variables_to_error:
        p[i + "_stds"] = abs(p[i]) * relative_error_assumed / 100

	# **************************************************************************************************
	# GB Normalized (Note: This is useful for mitim surrogate variables of targets)
	# **************************************************************************************************

    p["QeGB"] = p["QeMWm2"]   / p["Qgb"]
    p["QiGB"] = p["QiMWm2"]   / p["Qgb"]
    p["GeGB"] = p["Ge1E20m2"] / p["Ggb"]
    p["GZGB"] = p["GZ1E20m2"] / p["Ggb"]
    p["CeGB"] = p["Ce"]       / p["Qgb"]
    p["CZGB"] = p["CZ"]       / p["Qgb"]
    p["MtGB"] = p["MtJm2"]    / p["Pgb"]


class analytical_model_edge(analytical_model):
    """
    Edge-specific subclass of ``analytical_model``.

    Execution order in ``evaluate()``
    ----------------------------------
    1.  ``_evaluate_radiation()`` override (this class):
          - Zeros the TGYRO Chebyshev contribution for the Aurora-tracked impurity,
            calls base ``_evaluate_radiation()``, restores coefficients, adds
            ``qrad_aurora``, and adds Aurora H/D neutral radiation.
    2.  ``_evaluate_particle_fluxes()`` — D⁰ ionisation source, impurity net electron
        source (exact by charge conservation; NOT restricted to the trace regime), and
        the tracked-impurity source ``qpar_Z``.
    3.  ``_evaluate_ionization_loss()`` — ionisation energy cost from qie.
    """

    def __init__(self, powerstate, **kwargs):
        super().__init__(powerstate, **kwargs)

    def flux_integrate(self):
        """
		**************************************************************************************************
		Calculate integral of all targets, and then sum aux.
		Reason why I do it this convoluted way is to make it faster in mitim, not to run the volume integral all the time.
		Run once for all the batch and also for electrons and ions
		(in MW/m^2)
		**************************************************************************************************
		"""

        qe = torch.zeros_like(self.powerstate.plasma["te"])
        qi = torch.zeros_like(self.powerstate.plasma["te"])
        ge = torch.zeros_like(self.powerstate.plasma["te"])
        gz = torch.zeros_like(self.powerstate.plasma["te"])
        
        if "qie" in self.powerstate.target_options['options']['targets_evolve']:
            qe += -self.powerstate.plasma["qie"]
            qi +=  self.powerstate.plasma["qie"]
            if "qiz" in self.powerstate.plasma:
                # Ionization energy is an electron-channel sink, not an e-i exchange term.
                qe -= self.powerstate.plasma["qiz"]
            if "qcx" in self.powerstate.plasma:
                # Charge-exchange with cold neutrals is an ion-channel energy sink.
                qi -= self.powerstate.plasma["qcx"]

        if "qfus" in self.powerstate.target_options['options']['targets_evolve']:
            qe +=  self.powerstate.plasma["qfuse"]
            qi +=  self.powerstate.plasma["qfusi"]

        if "qrad" in self.powerstate.target_options['options']['targets_evolve']:
            qe -=  self.powerstate.plasma["qrad"]

        if "qpar" in self.powerstate.target_options['options']['targets_evolve']:
            ge += self.powerstate.plasma["qpar_wall"]
            gz += self.powerstate.plasma["qpar_Z"]

        q = torch.cat((qe, qi, ge, gz)).to(qe)
        self.P = self.powerstate.from_density_to_flux(q, force_dim=q.shape[0])

    # ------------------------------------------------------------------
    # evaluate() — extend base with edge terms
    # ------------------------------------------------------------------

    def evaluate(self):
        """
        Extend base ``analytical_model.evaluate()`` with edge-specific physics.

        The call order is:
             1. ``qie`` / ``qfus`` / ``qrad`` are evaluated exactly as in the
                 base model (with edge radiation overrides).
             2. ``_evaluate_particle_fluxes()`` populates local source densities
                 ``qpar_main``, ``qpar_imp``, ``qpar_wall`` and ``qpar_Z``
                 [1e20 m^-3 s^-1].
             3. ``_evaluate_ionization_loss()`` subtracts ionisation energy cost
                 from ``qie`` when that channel is evolved.
        """

        if "qie" in self.powerstate.target_options["options"]["targets_evolve"]:
            self._evaluate_energy_exchange()
            self._evaluate_ionization_loss()
            self._evaluate_cx_loss()

        if "qfus" in self.powerstate.target_options["options"]["targets_evolve"]:
            self._evaluate_alpha_heating()

        if "qrad" in self.powerstate.target_options["options"]["targets_evolve"]:
            self._evaluate_radiation()

        if "qpar" in self.powerstate.target_options["options"]["targets_evolve"]:
            self._evaluate_particle_fluxes()


    # ------------------------------------------------------------------
    # Override: _evaluate_radiation()
    # ------------------------------------------------------------------

    def _evaluate_radiation(self):
        """
        Compute edge radiation only (Aurora impurity + Aurora H/D neutrals).

        This intentionally skips ``super()._evaluate_radiation()`` if edge modules are active
        and builds all radiation channels directly on the active grid (the current
        ``plasma['te']`` shape) to avoid fine/coarse length mismatches.
        """
        p = self.powerstate.plasma
        p["qrad_bremms"] = p["te"] * 0.0
        p["qrad_line"] = p["te"] * 0.0
        p["qrad_sync"] = p["te"] * 0.0
        p["qrad"] = p["te"] * 0.0

        has_aurora_rad = (
            "qrad_aurora" in p
            and p["qrad_aurora"].abs().max().item() > 1e-30
        )
        has_neutrals = (
            "n0" in p and p["n0"].abs().max().item() > 1e-30
        )

        if has_aurora_rad:
            qrad_aurora = p["qrad_aurora"]
            if qrad_aurora.shape != p["qrad"].shape:
                if (
                    qrad_aurora.dim() == 2
                    and p["qrad"].dim() == 2
                    and qrad_aurora.shape[0] == p["qrad"].shape[0]
                ):
                    src_x = np.linspace(0.0, 1.0, qrad_aurora.shape[-1])
                    dst_x = np.linspace(0.0, 1.0, p["qrad"].shape[-1])
                    qrad_np = qrad_aurora.detach().cpu().numpy()
                    qrad_aurora = torch.from_numpy(
                        np.stack([np.interp(dst_x, src_x, qrad_np[i, :]) for i in range(qrad_np.shape[0])], axis=0)
                    ).to(p["qrad"])
                else:
                    print(
                        f"[analytical_model_edge] qrad_aurora shape {qrad_aurora.shape} "
                        f"incompatible with qrad shape {p['qrad'].shape}; skipping.",
                        typeMsg="w",
                    )
                    qrad_aurora = None

            if qrad_aurora is not None:
                qrad_aurora = qrad_aurora.to(p["qrad"])
                p["qrad"] = p["qrad"] + qrad_aurora

        if has_neutrals:
            # Add Aurora H/D main-ion neutral line radiation
            self._add_aurora_H_radiation()

        # Add with legacy method if needed (bremss + line + sync)
        if not has_aurora_rad:
            super()._evaluate_radiation()

    # ------------------------------------------------------------------
    # Aurora H radiation
    # ------------------------------------------------------------------

    def _add_aurora_H_radiation(self):
        """
        Compute and add line radiation from main-ion (D/H) neutrals using
        Aurora's ``compute_rad``.

        Constructs ``nz_H[t=0, z, r]`` from:
          - ``plasma['n0']``  → D⁰ ground-state population (z=0)
          - ``plasma['ni'][:,:,0]`` → D⁺ main-ion population (z=1)

        Calls ``aurora.radiation.compute_rad("H", ...)`` and adds the
        resulting total radiated power to ``plasma['qrad']``.
        """
        try:
            import aurora as _aurora_pkg
        except ImportError:
            return

        p = self.powerstate.plasma
        if "n0" not in p or p["n0"].abs().max().item() < 1e-30:
            return

        for b in range(p["te"].shape[0]):
            ne_cm3 = p["ne"][b, :].cpu().numpy() * 1e13    # cm⁻³
            Te_eV  = p["te"][b, :].cpu().numpy() * 1e3     # eV
            n0_cm3 = p["n0"][b, :].cpu().numpy() * 1e13    # cm⁻³

            if "ni" in p and p["ni"].dim() == 3:
                ni_D_cm3 = p["ni"][b, :, 0].cpu().numpy() * 1e13
            else:
                ni_D_cm3 = ne_cm3.copy()

            # nz_H shape: (nt=1, nZ+1=2, n_rho)  — [D⁰, D⁺]
            nz_H   = np.array([n0_cm3, ni_D_cm3])[np.newaxis, :, :]   # (1, 2, n_rho)
            ne_arr = ne_cm3[np.newaxis, :]                             # (1, n_rho)
            Te_arr = Te_eV[np.newaxis, :]                              # (1, n_rho)

            try:
                rad_res  = _aurora_pkg.radiation.compute_rad(
                    "H", nz_H, ne_arr, Te_arr, prad_flag=True)
                qrad_H_t = torch.from_numpy(
                    rad_res["tot"][0, :].copy()).to(p["qrad"])
            except Exception as exc:
                if self.powerstate.target_options["options"].get("verbose", False):
                    print(
                        f"[analytical_model_edge] Aurora H radiation compute_rad failed for batch {b}: {exc}",
                        typeMsg="w",
                    )
                return   # fail gracefully; H ADAS data may not be available

            if qrad_H_t.shape[-1] != p["qrad"].shape[-1]:
                src_x = np.linspace(0.0, 1.0, qrad_H_t.shape[-1])
                dst_x = np.linspace(0.0, 1.0, p["qrad"].shape[-1])
                qrad_H_t = torch.from_numpy(
                    np.interp(dst_x, src_x, qrad_H_t.detach().cpu().numpy())
                ).to(p["qrad"])

            p["qrad"][b] = p["qrad"][b] + qrad_H_t

    # ------------------------------------------------------------------
    # particle flux sources from neutral ionization
    # ------------------------------------------------------------------

    def _evaluate_particle_fluxes(self):
        """
        Populate local particle source densities ``qpar_main``, ``qpar_imp``,
        ``qpar_wall`` and ``qpar_Z``.

        Two contributions:
        1.  **D⁰ ionisation** — from ``plasma['S_ion_main']`` [1e19 m⁻³ s⁻¹]
            (ADAS-based, pre-computed by ``calculateNeutrals()``).
        2.  **Impurity net ionisation (electrons)** —
            ``Σ(scd·nz) - Σ(acd·nz)`` over charge states.  This is the freed-
            electron source ``Σ_z z·Ṅ_z`` and is exact by charge conservation
            regardless of impurity concentration, so it is applied
            unconditionally.  The peak charge-weighted dilution
            ``max(Σ z·nz / ne)`` is only checked against
            ``impurity_dilution_warn`` (default 0.5) to warn when the
            quasineutrality-based ``ni`` reconstruction is being stressed.
        3.  **Tracked-impurity source (``qpar_Z``, the GZ target)** — the source of
            whichever impurity quantity was handed to the transport codes, selected by
            ``impurity_source_convention``: ``"charge"`` (default) for the
            charge-weighted representative density, ``"nuclei"`` for impurity nuclei.
            NOT the fully-stripped-stage source, which is what this used to be and is
            neither. Measured on the L case, the charge convention puts ``GZ_target`` at
            8-44% of the impurity transport flux, so it is NOT negligible.

        Unit accounting
        ---------------
        ``qpar_*`` are stored as [1e20 m⁻³ s⁻¹] by multiplying source rates
        [1e19 m⁻³ s⁻¹] by 0.1. They are converted to fluxes in ``flux_integrate``.
        """
        p = self.powerstate.plasma

        p["qpar_main"] = p["te"] * 0.0
        p["qpar_imp"] = p["te"] * 0.0
        p["qpar_wall"] = p["te"] * 0.0
        p["qpar_Z"] = p["te"] * 0.0

        # ── 1. D⁰ ionisation source ─────────────────────────────────────────
        if "n0" in p and p["n0"].abs().max().item() > 1e-30:
            n0_1e19 = _ensure_1e19_units(p["n0"], "n0")
            ne_1e19 = _ensure_1e19_units(p["ne"], "ne")

            if "S_ion_main" in p:
                S_ion = _ensure_1e19_units(p["S_ion_main"], "S_ion_main")
            else:
                Te_eV   = p["te"] * 1e3
                # Fallback to Voronov (1997) H ionization fit, consistent with neutrals.py.
                U = 13.6 / Te_eV.clamp(0.1)
                sigma_v = 2.91e-14 * (U ** 0.39) * torch.exp(-U) / (0.232 + U)  # m^3/s
                # n0 and ne are stored as multiples of 1e19 m^-3, so the physical product
                # carries 1e38: rate = (n0*1e19)(ne*1e19)*sigma_v [m^-3 s^-1]. Returning that
                # in the same 1e19-based units divides by 1e19, leaving a NET FACTOR OF +1e19.
                # This previously read 1e-19, i.e. 1e38 too small -- which would silently zero
                # the main-ion particle source, and with it most of the Ge target, whenever
                # S_ion_main was unavailable. Only reachable via this fallback, which warns.
                S_ion = n0_1e19 * ne_1e19 * sigma_v * 1e19
                print(
                    "[analytical_model_edge] Using fallback S_ion estimate because S_ion_main is missing.",
                    typeMsg="w",
                )

            p["qpar_main"] = p["qpar_main"] + S_ion * 0.1

        # ── 2. Impurity net electron source (charge-conserving, always valid) ─
        if (
            "nz_all"     in p
            and "nu_scd_imp" in p
            and "nu_acd_imp" in p
        ):
            nz  = p["nz_all"]       # (batch, rho, nZ+1)  [1e19 m⁻³]
            scd = p["nu_scd_imp"]   # (batch, rho, nZ+1)  [s⁻¹]
            acd = p["nu_acd_imp"]   # (batch, rho, nZ+1)  [s⁻¹]
            ne  = p["ne"]           # (batch, rho)         [1e19 m⁻³]

            nz = _ensure_1e19_units(nz, "nz_all")
            ne = _ensure_1e19_units(ne, "ne")

            nZ_plus1 = nz.shape[-1]
            Z_vec = torch.arange(nZ_plus1, dtype=nz.dtype, device=nz.device)
            charge_dens = (nz * Z_vec).sum(dim=-1)
            dilution = charge_dens / ne.clamp(min=1e-30)

            # nu_scd_imp / nu_acd_imp already include ne multiplication (s^-1).
            # S_imp_net = Σ_z z·Ṅ_z is the freed-electron source; it is exact by
            # charge conservation and does NOT require a trace-impurity assumption.
            # The z-weighting telescopes, leaving Σ_w (S_w n_w − α_w n_w), so both sums are
            # unweighted -- but they must be taken over the STAGE THAT REACTS, not the same
            # index. Aurora stores both rate arrays with the pad at the LAST index
            # (core.py: Sne_rates[:, :-1] = Sne.T, likewise Rne_rates) and indexes them by the
            # LOWER stage of each transition: Sne_rates[z] ionizes z -> z+1, Rne_rates[z]
            # recombines z+1 -> z (hence Rne_rates[:, 0] = 0 to block recombination to
            # neutral). So ionization pairs acd/scd index z with n_z, while recombination out
            # of stage z pairs index z-1 with n_z. Pairing acd[z] with n_z instead multiplied
            # the fully stripped population by the zero pad, dropping the single largest
            # recombination term and leaving a "net" that was ~99.8% gross ionization.
            S_imp_iz  = (scd[:, :, :-1] * nz[:, :, :-1]).sum(dim=-1)
            S_imp_rec = (acd[:, :, :-1] * nz[:, :, 1:  ]).sum(dim=-1)
            S_imp_net = S_imp_iz - S_imp_rec

            p["qpar_imp"] = p["qpar_imp"] + S_imp_net * 0.1

            # ── Source for the TRACKED impurity species (the GZ channel) ────────
            #
            # The tracked species is whatever calculateChargeStates handed the transport
            # codes, so its source has to be the source of THAT quantity. The previous
            # expression, scd[-2]nz[-2] - acd[-1]nz[-1], is the net source into the fully
            # stripped stage alone -- neither an impurity particle source nor the source of
            # the charge-weighted density. It is one arbitrary term of the charge sum, and
            # with <Z> falling 5.8 -> 4.2 across this domain it is dominated by carbon
            # redistributing between stages as it crosses the Te gradient.
            #
            # Two physically meaningful conventions, selected by
            # target_options["options"]["impurity_source_convention"]:
            #
            #  "charge"  (default) -- matches the D1 representation
            #        n_rep = sum_z z n_z / Z_imp, so Q = Z_imp n_rep and the single-species
            #        approximation gives  div(Gamma_rep) = S_Q / Z_imp  with S_Q the net
            #        electron-liberation rate. NOTE this is NOT small: an impurity flowing
            #        inward through a rising Te keeps ionizing, so the charge-weighted
            #        density has a genuine volumetric source. Do not assume GZ_target ~ 0.
            #
            #  "nuclei" -- the tracked species is impurity NUCLEI, sum_z n_z, a conserved
            #        quantity whose only source is ionization out of the neutral stage.
            #        Small in the interior (most impurity ionizes in the SOL). Consistent
            #        only if the species density handed to the codes is sum_z n_z, which it
            #        is NOT under the default representation -- so this is for testing the
            #        sensitivity of GZ to the convention, not for production.
            if nZ_plus1 >= 2:
                convention = str(self.powerstate.target_options["options"].get(
                    "impurity_source_convention", "auto")).lower()
                if convention == "auto":
                    # Pair the target with the representation: the GZ target must be the source
                    # of whichever density the codes were actually handed. A conserved partition
                    # density has only the nuclei source; the charge-weighted n_rep has S_Q/Z_imp.
                    rep = str(getattr(self.powerstate, "_impurity_representation",
                                      "charge_density")).lower()
                    convention = ("nuclei" if rep in ("partition", "conserved", "nuclei")
                                  else "charge")
                Z_imp = self.powerstate._impurity_Z() if hasattr(
                    self.powerstate, "_impurity_Z") else None

                if str(convention).lower() == "nuclei":
                    S_Z = scd[:, :, 0] * nz[:, :, 0] - acd[:, :, 1] * nz[:, :, 1]
                elif Z_imp is None:
                    print(
                        "[analytical_model_edge] impurity_source_convention='charge' needs the "
                        "impurity charge, which is unavailable; falling back to the nuclei "
                        "source for qpar_Z.",
                        typeMsg="w",
                    )
                    S_Z = scd[:, :, 0] * nz[:, :, 0] - acd[:, :, 1] * nz[:, :, 1]
                else:
                    Zt = Z_imp.to(S_imp_net) if torch.is_tensor(Z_imp) else torch.as_tensor(
                        Z_imp, dtype=S_imp_net.dtype, device=S_imp_net.device)
                    S_Z = S_imp_net / Zt.clamp(min=1e-30)

                p["qpar_Z"] = p["qpar_Z"] + S_Z * 0.1

            # Diagnostic only: the electron source above is unconditional; this
            # threshold flags when the quasineutrality-based ni reconstruction
            # (done in calculateChargeStates) is being pushed hard.
            dilution_warn = self.powerstate.target_options["options"].get(
                "impurity_dilution_warn", 0.5
            )
            peak_dilution = dilution.max().item()
            if peak_dilution > dilution_warn:
                print(
                    f"[analytical_model_edge] peak impurity charge dilution "
                    f"max(Σz·nz/ne) = {peak_dilution:.3f} exceeds "
                    f"{dilution_warn:.2f}; electron source retained but "
                    f"quasineutral ni reconstruction may be stressed.",
                    typeMsg="w",
                )

        # Electron-wall source is main-ion plus impurity electron source.
        p["qpar_wall"] = p["qpar_main"] + p["qpar_imp"]

    # ------------------------------------------------------------------
    # ionization power loss
    # ------------------------------------------------------------------

    def _aurora_H_radiation_active(self):
        """
        Return True when the Aurora H/D neutral *line* radiation channel is
        available and will be added to ``qrad`` by ``_add_aurora_H_radiation``.

        This gates the ionisation energy cost (see ``_evaluate_ionization_loss``)
        so that the excitation/line-radiation part of the cost is not counted
        twice — once analytically in ``qiz`` and once explicitly in ``qrad``.
        """
        p = self.powerstate.plasma
        if "n0" not in p or p["n0"].abs().max().item() < 1e-30:
            return False
        try:
            import aurora as _aurora_pkg  # noqa: F401
        except ImportError:
            return False
        return True

    def _evaluate_ionization_loss(self):
        """
        ionisation power subtracted from qe.

        Energy cost per D⁰ ionisation event:

          - When Aurora H/D neutral line radiation is active, the excitation
            radiation preceding ionisation is already accounted for explicitly
            in ``qrad`` (via ``_add_aurora_H_radiation``).  Counting it again
            here would double-subtract it from the electron channel, so only
            the bare 13.6 eV ionisation potential is charged to ``qiz``.

          - Otherwise, a lumped effective cost of ~40 eV (13.6 eV potential
            + ~26 eV of prior excitation radiation) is used, since the
            radiative losses are not represented anywhere else.

        Uses ``plasma['S_ion_main']`` when available; falls back to the
        analytic rate when only ``plasma['n0']`` is present.

        If ``plasma['n0']`` is absent, this is a no-op.
        """
        p = self.powerstate.plasma
        p["qiz"] = torch.zeros_like(p["te"])
        if "n0" not in p or p["n0"].abs().max().item() < 1e-30:
            return

        if "S_ion_main" in p:
            S_ion = _ensure_1e19_units(p["S_ion_main"], "S_ion_main")
        else:
            raise NotImplementedError("Ionisation loss evaluation requires S_ion_main")

        # Avoid double-counting excitation/line radiation already carried by qrad.
        E_ion_eff_eV = 13.6 if self._aurora_H_radiation_active() else 40.0
        E_ion_eff_J = E_ion_eff_eV * 1.60218e-19
        # Numerically equivalent to MW/m^3; kept in the same units as qie/qrad arrays.
        Q_ion_MWm3 = S_ion * 1e19 * E_ion_eff_J * 1e-6

        p["qiz"] = Q_ion_MWm3

    # ------------------------------------------------------------------
    # charge-exchange ion energy loss
    # ------------------------------------------------------------------

    def _evaluate_cx_loss(self):
        """
        Ion-channel energy sink from charge exchange with cold neutrals.

        Each CX event (D⁺_hot + D⁰_cold → D⁰_hot + D⁺_cold) replaces a thermal
        ion at the local ion temperature ``Ti`` with one at the neutral
        temperature ``T0``, draining ``(3/2) k (Ti − T0)`` of ion energy per
        event.  The volumetric event rate is ``R_cx = n0 × ν_cx`` where
        ``ν_cx = n_i ⟨σv⟩_cx`` is provided by the neutrals solver
        (``plasma['nu_cx_main']`` [s⁻¹]).

        Neutral temperature model
        -------------------------
        Neutrals entering the closed-flux region have undergone many CX events
        while crossing the SOL and are approximated as thermalised to the ion
        temperature *at the LCFS*, ``T0 ≈ Ti(r=LCFS)``, held constant inward.
        Because the pedestal/core ``Ti`` rises above the edge value, the sink
        ``(Ti − T0)`` is positive across the pedestal and vanishes at the LCFS,
        consistent with the near-thermal boundary neutrals.

        This is an *ion* energy sink only; the ionisation potential and line
        radiation are charged to the electron channel elsewhere (``qiz``,
        ``qrad``).  A no-op when neutrals or the CX rate are absent.
        """
        p = self.powerstate.plasma
        p["qcx"] = torch.zeros_like(p["te"])
        if "n0" not in p or p["n0"].abs().max().item() < 1e-30:
            return
        if "nu_cx_main" not in p or p["nu_cx_main"].abs().max().item() < 1e-30:
            return

        n0_1e19 = _ensure_1e19_units(p["n0"], "n0")     # (batch, rho) [1e19 m⁻³]
        nu_cx   = p["nu_cx_main"]                        # (batch, rho) [s⁻¹]

        # Ion temperature and boundary (LCFS) neutral temperature [keV]
        ti = p["ti"] if p["ti"].dim() == 2 else p["ti"][..., 0]
        T0 = ti[:, -1:].expand_as(ti)                   # neutral temp ≈ Ti(LCFS)
        dT_keV = (ti - T0).clamp(min=0.0)               # only cooling (Ti ≥ T0)

        # R_cx = n0 × ν_cx  [m⁻³ s⁻¹]; energy per event (3/2)(Ti−T0) [J].
        R_cx_m3s = n0_1e19 * 1e19 * nu_cx
        E_cx_J   = 1.5 * dT_keV * 1e3 * 1.60218e-19
        # W/m³ × 1e-6 → MW/m³ (numerically == W/cm³, matching qie/qiz/qrad units).
        p["qcx"] = R_cx_m3s * E_cx_J * 1e-6

    def postprocessing(self, force_zero_particle_flux=False, relative_error_assumed=1.0):
        _edge_postprocessing(
            self.powerstate,
            integrated_targets=self.P,
            force_zero_particle_flux=force_zero_particle_flux,
            relative_error_assumed=relative_error_assumed,
        )


class analytical_model_legacy_edge_compat(analytical_model):
    """
    Compatibility adapter for running legacy analytical targets with powerstate_edge.

    This preserves legacy analytical evaluate/flux_integrate physics and only
    patches postprocessing shape assumptions that can break after edge-domain
    trimming.
    """

    def postprocessing(self, force_zero_particle_flux=False, relative_error_assumed=1.0):
        _edge_postprocessing(
            self.powerstate,
            integrated_targets=self.P,
            force_zero_particle_flux=force_zero_particle_flux,
            relative_error_assumed=relative_error_assumed,
        )