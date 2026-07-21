"""
edge_tools.uq.objective
------------------------
From a propagated residual factor ``L_r`` to:

  1. per-node std injection into the powerstate ``*_tr_turb_stds`` /
     ``*_tr_neoc_stds`` / ``*_tar_stds`` fields (so the rest of PORTALS consumes
     the propagated uncertainty unchanged),
  2. the residual covariance ``Sigma_r`` and per-residual error bars,
  3. the objective mean and sigma -- delta method away from the optimum, honest
     sampling of the residual Gaussian near it (the objective is a normalized L2
     norm, non-smooth at r=0), and a risk-aware ``mu + k sigma``,
  4. a chi-square convergence test of the residual vector against zero, relative
     to the propagated input uncertainty.

The PORTALS-edge objective is
    source = cal - of                     # target - transport, per (channel, cp)
    res    = -(1/N) * ||source||_2        # maximization -> negative L2 norm
(see PORTALStools.calculate_residuals).  We work with J = ||source||_2 / N and
report sigma_J; the sign is a convention handled by the caller.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools.edge_tools.uq.propagate import cov_from_factor, std_from_factor


# --------------------------------------------------------------------------- #
# (1) Split a stacked factor into target / transport, and form the residual
# --------------------------------------------------------------------------- #

# The actual OF flux plasma keys per predicted channel (turb, neoc, target),
# matching map_powerstate_to_portals' mapper.  NOTE the particle/impurity channels
# use Ge/GZ (particle-flux representation), NOT the powertorch profile_map Ce/CZ
# (convective) keys -- injecting into Ce/CZ would never reach the GP training std.
OF_FLUX_KEYS = {
    "te": ("QeMWm2_tr_turb",     "QeMWm2_tr_neoc",     "QeMWm2"),
    "ti": ("QiMWm2_tr_turb",     "QiMWm2_tr_neoc",     "QiMWm2"),
    # particle/impurity fluxes live under the canonical Ge1E20m2 / GZ1E20m2 keys
    # (the powertorch convective Ce/CZ are DERIVED = (3/2)Te*Ge1E20m2), so the GP
    # trains on Ge1E20m2_* -- inject there, not Ce/CZ or a bare "Ge".
    "ne": ("Ge1E20m2_tr_turb",   "Ge1E20m2_tr_neoc",   "Ge1E20m2"),
    "nZ": ("GZ1E20m2_tr_turb",   "GZ1E20m2_tr_neoc",   "GZ1E20m2"),
    "w0": ("MtJm2_tr_turb",      "MtJm2_tr_neoc",      "MtJm2"),
}


def flux_observable_keys(predicted_channels):
    """Flat list of OF flux plasma keys to scan for the predicted channels."""
    keys = []
    for ch in predicted_channels:
        keys.extend(OF_FLUX_KEYS.get(ch, ()))
    return keys


def inject_flux_stds(powerstate, L_obs, predicted_channels, scale_factor=1.0):
    """
    Quadrature-add the propagated input uncertainty into the OF flux std fields,
    using the NATIVE std of each flux (from its scan observable) written to the
    exact key the GP reads (``{flux_key}_stds``).

    Correct for every channel by construction -- turb and neoc come straight from
    scanning ``*_tr_turb`` / ``*_tr_neoc`` (no magnitude-split guess), and the
    particle/impurity channels land in Ge/GZ, not Ce/CZ.
    """
    plasma = powerstate.plasma
    n = 0
    for ch in predicted_channels:
        for key in OF_FLUX_KEYS.get(ch, ()):
            if key not in L_obs or key not in plasma:
                continue
            std_key = key + "_stds"
            if std_key not in plasma:
                continue
            sig = (std_from_factor(L_obs[key]) * scale_factor).reshape(plasma[key].shape)
            plasma[std_key] = torch.sqrt(plasma[std_key] ** 2 + sig ** 2)
            n += 1
    print(f"[UQ] injected input-propagation into {n} OF flux std fields "
          f"(native units, correct keys incl. Ge/GZ)", typeMsg="i")
    return n


# Group raw column/source names into interpretable categories for reporting.
def _source_group(name: str) -> str:
    if name.startswith("fit_"):
        return "mtanh-fit"
    if name.startswith("peret_aly"):
        return "aLy(PeretSSF)"
    if name in ("ne", "te", "ti", "ni"):
        return f"LCFS {name}"
    if name in ("aLne", "aLte", "aLti"):
        return f"LCFS {name}"
    if "source" in name:
        return "source rate"
    if name.startswith("impurity"):
        return "impurity D/V"
    if name.startswith("vtor"):
        return "vtor"
    return name


def column_source_labels(layout, L):
    """Map each column of ``L`` to the layout entry (uncertainty source) it came
    from, via the first nonzero row (slot).  Returns a list of length L.shape[1]."""
    slot_to_name = {}
    for name, e in layout.items():
        for s in e.get("slots", []):
            slot_to_name[int(s)] = name
    labels = []
    for j in range(L.shape[1]):
        nz = torch.nonzero(L[:, j].abs() > 0).flatten()
        labels.append(slot_to_name.get(int(nz[0]), f"col{j}") if nz.numel()
                      else f"col{j}")
    return labels


def variance_breakdown(L_key, labels, group=True):
    """
    Per-source contribution to an observable's uncertainty from its factor.

    ``L_key`` is (n_points, n_cols); variance is additive across columns, so the
    contribution of each source is the sum of its columns squared.  Returns
    {source: std_contribution (n_points,)} plus {"__total__": total std}.  With
    ``group=True`` sources are aggregated into interpretable categories.
    """
    n_pts = L_key.shape[0]
    var = {}
    for j, lab in enumerate(labels[:L_key.shape[1]]):
        key = _source_group(lab) if group else lab
        var[key] = var.get(key, torch.zeros(n_pts, dtype=L_key.dtype)) + L_key[:, j] ** 2
    out = {k: torch.sqrt(v.clamp_min(0.0)) for k, v in var.items()}
    total_var = sum(var.values()) if var else torch.zeros(n_pts, dtype=L_key.dtype)
    out["__total__"] = torch.sqrt(total_var.clamp_min(0.0))
    return out


def report_source_breakdown(source_breakdown, key, top=10):
    """Print a ranked per-source uncertainty breakdown for observable ``key``
    (RMS over radius).  Returns the ranked [(source, rms_std, %variance)] list."""
    bd = (source_breakdown or {}).get(key)
    if bd is None:
        print(f"[UQ] no breakdown for {key!r}", typeMsg="w")
        return []
    def rms(v):
        return float(torch.sqrt((v ** 2).mean()))
    tot = rms(bd["__total__"])
    items = sorted(((s, rms(v)) for s, v in bd.items() if s != "__total__"),
                   key=lambda x: -x[1])
    tvar = tot ** 2 if tot > 0 else 1.0
    ranked = [(s, c, 100.0 * c ** 2 / tvar) for s, c in items]
    print(f"[UQ] uncertainty sources for {key} (RMS over radius; total={tot:.3g}):",
          typeMsg="i")
    for s, c, pv in ranked[:top]:
        print(f"      {s:18s} std={c:.3g}  ({pv:4.0f}% of variance)", typeMsg="i")
    return ranked


def split_stacked(y0: torch.Tensor, L_out: torch.Tensor):
    """
    Split a stacked ``cat([P, P_tr])`` mean/factor into per-node pieces.

    Returns (tar0, tr0, L_tar, L_tr) with tar0/tr0 shape (m,) and the factors
    (m, k), where m is the number of (channel, cp) control points.
    """
    m = y0.numel() // 2
    return y0[:m], y0[m:], L_out[:m], L_out[m:]


def residual_from_split(tar0, tr0, L_tar, L_tr):
    """
    Residual mean and factor from the split node factors.

        source = tar - tr ,  L_source = L_tar - L_tr   (shared columns -> keeps
        the target<->transport correlation, including cancellation).
    """
    return tar0 - tr0, L_tar - L_tr


# --------------------------------------------------------------------------- #
# (1b) std injection into the powerstate
# --------------------------------------------------------------------------- #

def inject_stds(
    powerstate,
    L_tar: torch.Tensor,
    L_tr: torch.Tensor,
    sigma_turb_gp: Optional[Dict[str, torch.Tensor]] = None,
    sigma_neoc_gp: Optional[Dict[str, torch.Tensor]] = None,
    order: Optional[List[Tuple[str, int]]] = None,
    scale_factor: float = 1.0,
):
    """
    Write propagated + surrogate stds into the powerstate std fields.

    Parameters
    ----------
    L_tar, L_tr : (m, k) factors for target and transport over the residual
        control points, m = len(predicted_channels) * len(rhoCP).
    sigma_turb_gp, sigma_neoc_gp : optional per-channel epistemic (surrogate
        posterior) stds keyed by channel ("te","ti","ne",...), each a tensor over
        rhoCP; folded in quadrature into the turb / neoc std fields.
    order : the (channel, cp) order of the rows of L_tar / L_tr.  If None it is
        rebuilt as predicted_channels x rhoCP (the calculate() concatenation).

    Splitting the input-induced transport std into turbulent vs neoclassical:
    apportioned by the nominal |turb| / |neoc| magnitudes at that point (the
    input perturbation moves both); the surrogate epistemic terms are added to
    their own channel.  Target std goes to ``*_tar_stds``.
    """
    plasma = powerstate.plasma
    profile_map = powerstate.profile_map
    if order is None:
        order = _default_order(powerstate)

    sig_tar = std_from_factor(L_tar) * scale_factor      # (m,)
    sig_tr = std_from_factor(L_tr) * scale_factor        # (m,)

    n_inflated = 0
    for row, (ch, cp) in enumerate(order):
        if ch not in powerstate.predicted_channels:
            continue
        # The std keys use the target/flux base (e.g. "QeMWm2"): target std is
        # "{base}_stds", transport stds are "{base}_tr_turb_stds" /
        # "{base}_tr_neoc_stds" (see TRANSPORTtools / targets_analytic_edge).
        base = profile_map[ch][0]
        fine_idx = _fine_index(powerstate, cp)

        # --- target ---
        _add_quad(plasma, f"{base}_stds", fine_idx, sig_tar[row])

        # --- transport: split input-induced into turb / neoc by magnitude ---
        w_turb, w_neoc = _turb_neoc_split(plasma, base, fine_idx)
        _add_quad(plasma, f"{base}_tr_turb_stds", fine_idx, sig_tr[row] * w_turb)
        _add_quad(plasma, f"{base}_tr_neoc_stds", fine_idx, sig_tr[row] * w_neoc)

        # --- surrogate epistemic (per channel) ---
        if sigma_turb_gp and ch in sigma_turb_gp:
            _add_quad(plasma, f"{base}_tr_turb_stds", fine_idx,
                      sigma_turb_gp[ch][cp] * scale_factor)
        if sigma_neoc_gp and ch in sigma_neoc_gp:
            _add_quad(plasma, f"{base}_tr_neoc_stds", fine_idx,
                      sigma_neoc_gp[ch][cp] * scale_factor)
        n_inflated += 1

    print(f"[UQ] injected propagated stds into {n_inflated} (channel, cp) "
          f"target/transport fields", typeMsg="i")


def _turb_neoc_split(plasma, base, fine_idx):
    """Apportion input-induced transport std by |turb|/|neoc| magnitude."""
    t = _abs_at(plasma, f"{base}_tr_turb", fine_idx)
    n = _abs_at(plasma, f"{base}_tr_neoc", fine_idx)
    tot = t + n
    if tot <= 0:
        return 1.0, 0.0     # default all to turbulent
    return float(t / tot), float(n / tot)


def _abs_at(plasma, key, fine_idx):
    if key not in plasma:
        return 0.0
    v = plasma[key]
    try:
        return float(v[..., fine_idx].abs().reshape(-1)[0])
    except Exception:
        return 0.0


def _add_quad(plasma, key, fine_idx, sigma, optional=False):
    """In-place quadrature add of ``sigma`` into plasma[key][..., fine_idx]."""
    if key not in plasma:
        if not optional:
            pass  # key genuinely absent for this run (e.g. neoc off); skip silently
        return
    s = torch.as_tensor(sigma, dtype=plasma[key].dtype, device=plasma[key].device)
    stds = plasma[key]
    if stds.dim() >= 2 and stds.shape[-1] > fine_idx:
        stds[..., fine_idx] = torch.sqrt(stds[..., fine_idx] ** 2 + s ** 2)


def _fine_index(powerstate, cp):
    rho_cp_val = float(powerstate.rhoCP[cp])
    rho_fine = powerstate.plasma["rho"][0].detach().cpu().numpy()
    return int(np.argmin(np.abs(rho_fine - rho_cp_val)))


def _default_order(powerstate):
    order = []
    for ch in powerstate.predicted_channels:
        for cp in range(len(powerstate.rhoCP)):
            order.append((ch, cp))
    return order


# --------------------------------------------------------------------------- #
# (2) residual covariance
# --------------------------------------------------------------------------- #

def residual_covariance(L_r: torch.Tensor):
    """Return (Sigma_r, per-residual std) from the residual factor."""
    Sigma_r = cov_from_factor(L_r)
    sigma = std_from_factor(L_r)
    return Sigma_r, sigma


# --------------------------------------------------------------------------- #
# (3) objective sigma
# --------------------------------------------------------------------------- #

def objective_sigma_delta(source: torch.Tensor, L_r: torch.Tensor):
    """
    Delta-method objective mean and sigma for J = ||source||_2 / N.

        g = dJ/dsource = source / (N ||source||)
        sigma_J^2 = g.T Sigma_r g = || L_r.T g ||^2

    Valid away from the optimum; degrades as ||source|| -> 0 (the norm cusp),
    where ``objective_sigma_sampling`` should be used instead.
    """
    N = source.numel()
    norm = torch.linalg.vector_norm(source)
    mu_J = norm / N
    if float(norm) < 1e-30:
        return mu_J, torch.zeros((), dtype=source.dtype)
    g = source / (N * norm)                       # (m,)
    proj = L_r.transpose(-1, -2) @ g              # (k,)
    sigma_J = torch.sqrt((proj ** 2).sum().clamp_min(0.0))
    return mu_J, sigma_J


def objective_sigma_sampling(source_mean: torch.Tensor, L_r: torch.Tensor,
                             n_samples: int = 20000, quantiles=(0.9, 0.95, 0.99),
                             seed: int = 0):
    """
    Honest objective distribution near the optimum: sample
    ``source ~ N(source_mean, Sigma_r)`` via the factor (no model calls) and
    evaluate J = ||source||_2 / N on each sample.

    Returns dict with mean, std, and requested upper quantiles of J -- the correct
    (skewed, generalized-chi-square) behaviour the Gaussian delta method misses.
    """
    g = torch.Generator(device=L_r.device).manual_seed(seed)
    k = L_r.shape[1]
    N = source_mean.numel()
    xi = torch.randn((n_samples, k), generator=g, dtype=L_r.dtype, device=L_r.device)
    samples = source_mean.unsqueeze(0) + xi @ L_r.transpose(0, 1)   # (n, m)
    J = torch.linalg.vector_norm(samples, dim=1) / N               # (n,)
    out = {"mean": float(J.mean()), "std": float(J.std())}
    qs = torch.quantile(J, torch.tensor(quantiles, dtype=J.dtype))
    out["quantiles"] = {q: float(v) for q, v in zip(quantiles, qs)}
    return out


def robust_objective(mu_J, sigma_J, k: float = 2.0):
    """Risk-aware scalar ``mu_J + k sigma_J`` (upper-tail, minimization)."""
    return mu_J + k * sigma_J


# --------------------------------------------------------------------------- #
# (4) chi-square convergence
# --------------------------------------------------------------------------- #

def chi2_convergence(source_mean: torch.Tensor, Sigma_r: torch.Tensor,
                     alpha: float = 0.05, rcond: float = 1e-8):
    """
    Mahalanobis test of the residual vector against zero, relative to the
    propagated uncertainty:

        stat = source.T Sigma_r^+ source   <=   chi2(dof, 1 - alpha)

    Sigma_r is rank-limited (rank <= number of uncertain input columns) when the
    residuals outnumber inputs, so the pseudoinverse is used and dof is its rank.
    Directions orthogonal to the input-uncertainty subspace get no propagated
    sigma -- their misfit is model/optimizer error, not input noise, and is
    excluded from the flag (as it should be).

    Returns (converged: bool, stat: float, dof: int, threshold: float).
    """
    Spinv = torch.linalg.pinv(Sigma_r, rcond=rcond, hermitian=True)
    stat = float(source_mean @ Spinv @ source_mean)
    dof = int(torch.linalg.matrix_rank(Sigma_r, rtol=rcond).item())
    dof = max(dof, 1)
    threshold = _chi2_ppf(1.0 - alpha, dof)
    converged = stat <= threshold
    print(f"[UQ] chi2 convergence: stat={stat:.3g}  dof={dof}  "
          f"thr={threshold:.3g}  -> {'CONVERGED' if converged else 'not yet'}",
          typeMsg="i")
    return converged, stat, dof, threshold


def _chi2_ppf(p: float, dof: int) -> float:
    """chi-square inverse-CDF; SciPy if available, else Wilson-Hilferty."""
    try:
        from scipy.stats import chi2
        return float(chi2.ppf(p, dof))
    except Exception:
        # Wilson-Hilferty normal approximation
        from math import sqrt
        z = _norm_ppf(p)
        t = 1.0 - 2.0 / (9.0 * dof) + z * sqrt(2.0 / (9.0 * dof))
        return dof * t ** 3


def _norm_ppf(p: float) -> float:
    # Acklam's rational approximation to the standard-normal quantile
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
