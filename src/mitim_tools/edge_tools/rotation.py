"""rotation.py -- self-consistent E×B rotation inputs for powerstate edge loops.

Torchified port of the standalone rotation model: everything on the analytic
path is expressed with batched ``torch`` tensors and only autodiff-safe ops
(``torch.gradient``, elementwise algebra), so the rotation quantities are
differentiable functions of the GP-parameterized profiles (ne, te, ti, ...).

Derives the rotation quantities that TGLF/CGYRO and NEO consume, from a
neoclassical / diamagnetic radial electric field under the *vtor = 0* closure
(no imposed toroidal mass rotation).

GACODE conventions (all built from a single w0 profile)::

    w0      = Er / (R * Bp)                 [rad/s]  E×B toroidal rotation
    GAMMA_E = (r/|q|) * (dw0/dr) * (a/cs)   [-]      TGLF VEXB_SHEAR = CGYRO GAMMA_E
    GAMMA_P =  R      * (dw0/dr) * (a/cs)   [-]      TGLF VPAR_SHEAR = CGYRO GAMMA_P
    MACH    =  R      *  w0      /  cs      [-]      TGLF VPAR       = CGYRO MACH

Radial force balance (main ion ``i``)::

    Er = dp_i/dr / (Z_i e n_i)  +  Vtor * Bp  -  Vpol * Bt
    Vpol = K_neo * Bt * dT_i/dr / (Z_i e B^2)          (B^2 = Bt^2 + Bp^2)

``Vtor`` is the imposed toroidal (mass) rotation velocity [m/s]; the default
``vtor = 0`` closure recovers the pure diamagnetic + neoclassical-poloidal Er.
A finite ``Vtor`` simply shifts w0 by ``Vtor / R`` (it enters Er as Vtor*Bp, and
w0 = Er/(R Bp)), so it propagates consistently to MACH / GAMMA_P / GAMMA_E.

``K_neo`` is the neoclassical poloidal-flow coefficient. ``0.0`` recovers the
pure diamagnetic Er that the previous ``calculateProfileFunctions`` used;
``"sauter"`` adds the Sauter-Angioni-Lin-Liu (1999) banana/plateau estimate
(collisionality + trapped-fraction + Zeff aware, ~7% vs NEO with no calibration).

Design notes for use inside the flux-match / GP loop
----------------------------------------------------
* Stateless by design. The original standalone model carried an under-relaxed
  ``w0`` across outer iterations; that makes the DV→profile map path-dependent
  and non-reproducible (a class of bug already seen in this project), and it is
  incompatible with a clean autograd graph. Relaxation, if wanted, belongs in
  the outer solver, not here.
* Derivatives are taken w.r.t. r = a * roa using ``torch.gradient`` on the
  shared ``roa`` grid, then divided by the per-batch minor radius ``a`` (chain
  rule). This assumes the ``roa`` grid is shared across the batch, matching the
  convention already used in ``STATEtools.calculateProfileFunctions``.
* Coarse grid handling: the shear terms (GAMMA_E ~ d²φ/dr²) are under-resolved
  on a ~15-point grid. Rather than smoothing, set ``oversample > 1`` to refine
  the profiles onto an ``oversample``x-finer grid with a differentiable natural
  cubic spline, evaluate the rotation there, and interpolate the results back.
  This resolves the curvature without adding a smoothing bias and stays on the
  autograd graph. (Mathematically it is a curvature model, not new information;
  natural cubic can overshoot at a sharp pedestal foot -- reduce ``oversample``
  or pre-smooth if that appears.)

Backends
--------
* "analytic" (default): differentiable diamagnetic + neoclassical-poloidal Er.
* "vgen": real NEO DKE solve via ``profiles_gen -vgen`` (vtor=0), providing a
  higher-fidelity w0. This is an *external* code: it is not differentiable, so
  the returned w0 (and its shear variants) are detached from the autograd graph.
  That is acceptable here because the turbulent/neoclassical transport that
  consumes VEXB_SHEAR/MACH is itself evaluated by external codes (TGLF/CGYRO/NEO)
  and surrogated by a GP -- the DV->flux gradient never flows analytically through
  the rotation inputs, so severing it costs nothing. It would only matter if an
  *analytic* target/residual differentiated w.r.t. w0 (none does today).
"""

from __future__ import annotations

import copy
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
import torch

QE = 1.602176634e-19       # C
EPS0 = 8.8541878128e-12    # F/m
MD = 2.0 * 1.6726219e-27   # deuteron mass [kg]


# --------------------------------------------------------------------------- #
# differentiable helpers
# --------------------------------------------------------------------------- #
def _roa_1d(roa):
    """Return the shared 1-D roa grid (torch.gradient needs 1-D coordinates)."""
    return roa[0] if roa.dim() > 1 else roa


def ddr(y, roa, a):
    """d(y)/dr on r = a * roa, autodiff-safe and batched.

    ``y``   : (batch, rho)
    ``roa`` : (batch, rho) or (rho,) -- assumed shared across the batch
    ``a``   : (batch, 1) minor radius [m]
    """
    dydroa = torch.gradient(y, spacing=(_roa_1d(roa),), dim=-1)[0]
    return dydroa / a


def smooth(y, window=1):
    """Optional differentiable moving-average (odd ``window``, reflect-padded).
    ``window <= 1`` is a no-op. Kept separate so the physics defaults to the raw
    (already-smooth) parameterized profile."""
    if window is None or window <= 1:
        return y
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = torch.ones((1, 1, window), dtype=y.dtype, device=y.device) / window
    yp = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode="reflect")
    return torch.nn.functional.conv1d(yp, kernel).squeeze(1)


# --------------------------------------------------------------------------- #
# differentiable natural-cubic-spline interpolation (for grid refinement)
# --------------------------------------------------------------------------- #
def cubic_interp(x, y, xq):
    """Natural cubic spline interpolation, batched and autodiff-safe.

    ``x``  : (n,)          strictly increasing shared nodes
    ``y``  : (batch, n)    node values
    ``xq`` : (m,)          query points
    returns (batch, m). The spline coefficients are a linear (hence
    differentiable) function of ``y``; the tridiagonal system is small (n~15)
    so a dense solve is cheap.
    """
    n = x.shape[0]
    h = x[1:] - x[:-1]                                   # (n-1,)
    # Assemble the natural-spline moment system A M = rhs (M = 2nd derivatives).
    A = torch.zeros((n, n), dtype=y.dtype, device=y.device)
    A[0, 0] = 1.0
    A[-1, -1] = 1.0
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2.0 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
    dy = (y[:, 1:] - y[:, :-1]) / h                      # (batch, n-1)
    rhs = torch.zeros_like(y)
    rhs[:, 1:-1] = 6.0 * (dy[:, 1:] - dy[:, :-1])
    M = torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)  # (batch, n)

    # Locate each query point in [x_j, x_{j+1}].
    j = torch.clamp(torch.bucketize(xq, x) - 1, 0, n - 2)  # (m,)
    xj, xj1 = x[j], x[j + 1]
    hj = xj1 - xj
    Mj = M[:, j]; Mj1 = M[:, j + 1]
    yj = y[:, j]; yj1 = y[:, j + 1]
    A_ = (xj1 - xq) / hj
    B_ = (xq - xj) / hj
    return (
        A_ * yj + B_ * yj1
        + ((A_ ** 3 - A_) * Mj + (B_ ** 3 - B_) * Mj1) * (hj ** 2) / 6.0
    )


def trapped_fraction(eps):
    """Circular-flux-surface trapped-particle fraction
    f_t = 1 - (1-eps)^2 / [ sqrt(1-eps^2) (1 + 1.46 sqrt(eps)) ]."""
    eps = eps.clamp(1e-6, 0.999)
    return 1.0 - (1.0 - eps) ** 2 / (
        torch.sqrt(1.0 - eps ** 2) * (1.0 + 1.46 * torch.sqrt(eps))
    )


def ion_collisionality_star(ni_m3, ti_keV, rmin, rmaj, q,
                            Zi=1.0, Zeff=1.0, ln_lambda=17.0, mi=MD):
    """Normalized ion collisionality nu*_i = nu_ii q R / (eps^{3/2} v_thi).
    SI inputs (ni [m^-3], ti [keV]); dimensionless output."""
    T_J = ti_keV * 1e3 * QE
    vthi = torch.sqrt(2.0 * T_J / mi)
    eps = (rmin / rmaj).clamp(1e-6)
    nu_ii = (ni_m3 * (Zi * QE) ** 4 * Zeff * ln_lambda) / (
        12.0 * torch.pi ** 1.5 * EPS0 ** 2 * torch.sqrt(torch.as_tensor(mi)) * T_J ** 1.5
    )
    return nu_ii * q.abs() * rmaj / (eps ** 1.5 * vthi)


def k_neo_sauter(nu_star, ftrap):
    """Neoclassical ion poloidal-flow coefficient, Sauter, Angioni & Lin-Liu,
    Phys. Plasmas 6, 2834 (1999) (as in NEO's compute_Sauter). Returns
    K = -alpha in Vpol = K Bt Ti'/(Z e B^2)."""
    a0 = -1.17 * (1.0 - ftrap) / (1.0 - 0.22 * ftrap - 0.19 * ftrap ** 2)
    aS = (
        (a0 + 0.25 * (1.0 - ftrap ** 2) * torch.sqrt(nu_star)) / (1.0 + 0.5 * torch.sqrt(nu_star))
        + 0.315 * nu_star ** 2 * ftrap ** 6
    ) / (1.0 + 0.15 * nu_star ** 2 * ftrap ** 6)
    return -aS


def resolve_K_neo(K_neo, ni_m3, ti_keV, rmin, rmaj, q, Z, Zeff, ln_lambda=17.0):
    """Return the poloidal-flow coefficient K as a (batch, rho) tensor.
    ``K_neo`` may be a float / tensor (used directly) or the string ``"sauter"``."""
    if isinstance(K_neo, str):
        name = K_neo.lower()
        if name == "sauter":
            nus = ion_collisionality_star(
                ni_m3, ti_keV, rmin, rmaj, q, Zi=Z, Zeff=Zeff, ln_lambda=ln_lambda
            )
            return k_neo_sauter(nus, trapped_fraction(rmin / rmaj))
        raise ValueError(f"unknown K_neo string {K_neo!r} (use 'sauter' or a float)")
    return torch.as_tensor(K_neo, dtype=ti_keV.dtype, device=ti_keV.device)


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def calculate_rotation(
    *,
    ni_m3,          # (batch, rho) main-ion density [m^-3]
    ti_keV,         # (batch, rho) ion temperature [keV]
    te_keV,         # (batch, rho) electron temperature [keV]
    rmin,           # (batch, rho) r [m]
    R,              # (batch, rho) major radius [m]
    Bt,             # (batch, rho) toroidal field [T]
    Bp,             # (batch, rho) poloidal field [T]
    q,              # (batch, rho) safety factor
    c_s,            # (batch, rho) sound speed [m/s]
    a,              # (batch, 1) minor radius [m]
    roa,            # (batch, rho) or (rho,) normalized radius (shared grid)
    Z_main=None,    # (batch, 1) main-ion charge (default 1)
    Zeff=None,      # (batch, rho) effective charge (default 1)
    K_neo="sauter", # "sauter" = neoclassical Vpol; float (e.g. 0.0) for a fixed K
    ln_lambda=17.0,
    smooth_window=1,
    mach_cut2=0.0,  # M^2 below which MACH / GAMMA_P are zeroed (0 = never)
    oversample=2,   # >1: refine onto an oversample x finer grid before differentiating
    vtor=None,      # (batch, rho) imposed toroidal velocity [m/s]; None = vtor=0 closure
    w0_override=None,  # (batch, rho) externally supplied w0 [rad/s] (e.g. NEO vgen);
                       # bypasses the analytic Er and just builds the shear variants
):
    """Compute all E×B rotation quantities from the current profiles.

    Returns a dict with (batch, rho) tensors, all differentiable on the analytic
    path (``w0_override`` severs the graph -- see the module Backends note):
        w0         [rad/s]  = Er / (R Bp)
        Er         [V/m]
        E_rad      [-]      NEO DPHI0DR convention = -Er a / (te[keV] 1e3)
        vexb       [m/s]    = E_rad / Bt
        gamma_exb  [1/s]    = (r/|q|) dw0/dr
        vexb_shear [-]      GAMMA_E = gamma_exb * (a / c_s) = (r/|q|)(dw0/dr)(a/cs)
        gamma_p    [-]      GAMMA_P = R dw0/dr (a/c_s)
        mach       [-]      MACH    = R w0 / c_s
        w0_n       [-]      w0 / c_s
        aLw0_n     [-]      -(dw0/dr) a / c_s
    """
    roa1d = _roa_1d(roa)

    if oversample and oversample > 1:
        # Refine profiles onto a finer grid, evaluate there, interpolate back.
        # The shear terms (~d^2/dr^2) are the ones that benefit from resolution.
        n = roa1d.shape[0]
        roa_f = torch.linspace(float(roa1d[0]), float(roa1d[-1]),
                               oversample * (n - 1) + 1,
                               dtype=roa1d.dtype, device=roa1d.device)
        def up(y):
            return cubic_interp(roa1d, y, roa_f)
        out_f = calculate_rotation(
            ni_m3=up(ni_m3), ti_keV=up(ti_keV), te_keV=up(te_keV),
            rmin=up(rmin), R=up(R), Bt=up(Bt), Bp=up(Bp), q=up(q), c_s=up(c_s),
            a=a, roa=roa_f,
            Z_main=Z_main, Zeff=(None if Zeff is None else up(Zeff)),
            K_neo=K_neo, ln_lambda=ln_lambda, smooth_window=smooth_window,
            mach_cut2=mach_cut2, oversample=1,
            vtor=(None if vtor is None else up(vtor)),
            w0_override=(None if w0_override is None else up(w0_override)),
        )
        # Sample the (smooth) rotation outputs back onto the caller's grid.
        return {k: cubic_interp(roa_f, v, roa1d) if v.shape[-1] == roa_f.shape[0] else v
                for k, v in out_f.items()}

    dtype, device = ti_keV.dtype, ti_keV.device
    if Z_main is None:
        Z_main = torch.ones((ti_keV.shape[0], 1), dtype=dtype, device=device)
    if Zeff is None:
        Zeff = torch.ones_like(ti_keV)

    ni_m3 = ni_m3.clamp(min=1e-30)
    Ti_J = ti_keV * 1e3 * QE

    if w0_override is not None:
        # Externally supplied w0 (e.g. NEO vgen); reconstruct Er consistently.
        w0 = w0_override
        Er = w0 * R * Bp
    else:
        # --- radial electric field (force balance) ------------------------- #
        #   Er = (1/Z e n) dp/dr  +  Vtor Bp  -  Vpol Bt
        B2 = Bt ** 2 + Bp ** 2
        pi = smooth(ni_m3 * Ti_J, smooth_window)          # main-ion pressure [Pa]
        dpidr = ddr(pi, roa, a)
        Er_dia = dpidr / (Z_main * QE * ni_m3)

        K = resolve_K_neo(K_neo, ni_m3, ti_keV, rmin, R, q, Z_main, Zeff, ln_lambda)
        dTidr = ddr(smooth(Ti_J, smooth_window), roa, a)
        Vpol = K * Bt * dTidr / (Z_main * QE * B2)
        Er = Er_dia - Vpol * Bt                            # [V/m]
        if vtor is not None:                               # imposed toroidal rotation
            Er = Er + vtor * Bp

        RBp = R * Bp
        w0 = torch.where(RBp.abs() > 0,
                         Er / torch.where(RBp == 0, torch.ones_like(RBp), RBp),
                         torch.zeros_like(Er))             # [rad/s]

    # --- NEO / TGLF normalizations ---------------------------------------- #
    E_rad = -Er * a / (te_keV * 1e3)                       # NEO DPHI0DR
    vexb = E_rad / Bt                                      # [m/s]
    tau_norm = a / c_s                                     # [s]

    dw0dr = ddr(w0, roa, a)

    # GAMMA_E = (r/|q|) (dw0/dr) (a/c_s)  [TGLF VEXB_SHEAR = CGYRO GAMMA_E].
    # Built directly from w0 = Er/(R Bp), NOT from the normalized NEO DPHI0DR
    # field E_rad = -Er a/(Te 1e3): the previous r d/dr(vexb/r) form fed the
    # dimensionless E_rad in as a velocity and came out ~|q| Te/(a Bt) too small
    # (~180x here), disagreeing with this function's own documented GAMMA_E.
    gamma_exb = (rmin / q.abs().clamp(min=1e-6)) * dw0dr   # [1/s]
    vexb_shear = gamma_exb * tau_norm                      # [-]

    gamma_p = R * dw0dr * tau_norm                         # [-]
    mach = R * w0 / c_s                                    # [-]

    w0_n = w0 / c_s
    aLw0_n = -dw0dr * a / c_s

    if mach_cut2 > 0.0:
        small = mach ** 2 < mach_cut2
        gamma_p = torch.where(small, torch.zeros_like(gamma_p), gamma_p)
        mach = torch.where(small, torch.zeros_like(mach), mach)

    return {
        "w0": w0,
        "Er": Er,
        "E_rad": E_rad,
        "vexb": vexb,
        "gamma_exb": gamma_exb,
        "vexb_shear": vexb_shear,
        "gamma_p": gamma_p,
        "mach": mach,
        "w0_n": w0_n,
        "aLw0_n": aLw0_n,
        "tau_norm": tau_norm,
    }


# --------------------------------------------------------------------------- #
# vgen backend (NEO DKE solve; non-differentiable, higher fidelity)
# --------------------------------------------------------------------------- #
_FLOAT = re.compile(r"[-+]?\d*\.\d+E[-+]?\d+")


def _zero_w0_vtor(path, tags=("# w0", "# vtor")):
    """Zero the numeric rows of the ``# w0`` and ``# vtor`` blocks of an
    input.gacode, enforcing the vtor = 0 closure while preserving column widths."""
    lines = open(path).read().splitlines()
    out, i = [], 0
    while i < len(lines):
        l = lines[i]
        if any(l.rstrip() == t or l.startswith(t + " ") for t in tags):
            out.append(l); i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith("#"):
                out.append(_FLOAT.sub(
                    lambda m: "0.0000000E+00".rjust(len(m.group(0))), lines[i]))
                i += 1
        else:
            out.append(l); i += 1
    open(path, "w").write("\n".join(out) + "\n")


def w0_from_vgen(
    gacode_path,
    rho_target,           # (batch, rho) numpy/tensor: sqrt-tor-flux grid to interp onto
    *,
    profiles_gen="profiles_gen",
    vel_mode=2,           # 1 = weak, 2 = strong rotation
    n_ion=3,              # thermal-ion count for input.neo template
    ix_main=1,            # ion index to match (1 = main ion)
    python_shim_dir=None, # dir with a `python` -> python3 link, prepended to PATH
    zero_w0=True,         # zero the # w0 block (w0 is the vgen output)
    zero_vtor=True,       # zero the # vtor block (True = vtor=0 closure). Set False
                          # to keep an imposed toroidal rotation already written into
                          # the file's # vtor block (er_method=2 matches vtor_measured).
    dtype=None,
    device=None,
):
    """Run ``profiles_gen -vgen -er 2`` (NEO weak-rotation Er) on ``gacode_path``
    and return w0 [rad/s] interpolated onto ``rho_target``.

    With ``zero_vtor=True`` (default) this enforces the vtor = 0 closure: Er is
    the diamagnetic + neoclassical-poloidal field only. With ``zero_vtor=False``
    the file's ``# vtor`` block is left intact, so er_method=2 matches that
    imposed toroidal rotation (Er gains the Vtor*Bp term) -- the caller is
    responsible for having written the desired Vtor into the ``# vtor`` block.

    NOT differentiable (external DKE solve). Feed the result to
    ``calculate_rotation(w0_override=...)`` to build the shear variants. Expensive:
    the caller should cache it on an outer cadence rather than every evaluation.
    """
    rho_np = np.atleast_2d(_np(rho_target))
    work = tempfile.mkdtemp(prefix="vgen_")
    try:
        gac = os.path.join(work, "input.gacode")
        shutil.copy2(gacode_path, gac)
        tags = tuple(t for t, z in (("# w0", zero_w0), ("# vtor", zero_vtor)) if z)
        if tags:
            _zero_w0_vtor(gac, tags=tags)
        env = dict(os.environ)
        if python_shim_dir:
            env["PATH"] = python_shim_dir + os.pathsep + env["PATH"]
        proc = subprocess.run(
            [profiles_gen, "-vgen", "-i", "input.gacode",
             "-er", "2", "-vel", str(vel_mode),
             "-in", str(n_ion), "-ix", str(ix_main)],
            cwd=work, env=env, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        vel_path = os.path.join(work, "vgen", "out.vgen.vel")
        if not os.path.exists(vel_path):
            # vgen returns exit code 0 even on physics errors (e.g. wrong -in ion
            # count → "Negative ion density"), so surface its log explicitly.
            raise RuntimeError(
                "profiles_gen -vgen produced no out.vgen.vel "
                f"(n_ion={n_ion}, ix_main={ix_main}). vgen output:\n{proc.stdout}")
        vel = np.loadtxt(vel_path)
        rho_v, w0_v = vel[:, 0], vel[:, 2]                  # col3 = w0 [rad/s]
        w0 = np.stack([np.interp(rho_np[b], rho_v, w0_v) for b in range(rho_np.shape[0])])
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return torch.as_tensor(w0, dtype=dtype, device=device)


def _np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def vtor_from_pairs(roa_grid, pairs, dtype=None, device=None):
    """Interpolate a toroidal-velocity profile from ``pairs = [(rho, vtor_m_s), ...]``
    onto ``roa_grid`` (matched index-wise to the other plasma arrays). A single
    pair means a spatially constant V_tor. Imposed input, so numpy interpolation
    (no autograd path) is fine; returns a (batch, rho) tensor."""
    grid = np.atleast_2d(_np(roa_grid))
    pr = np.array([x[0] for x in pairs], dtype=np.float64)
    pv = np.array([x[1] for x in pairs], dtype=np.float64)
    o = np.argsort(pr)                                     # len-1 -> constant
    v = np.vstack([np.interp(grid[b], pr[o], pv[o]) for b in range(grid.shape[0])])
    return torch.as_tensor(v, dtype=dtype, device=device)


# --------------------------------------------------------------------------- #
# powerstate_edge orchestration (moved out of STATEedge.powerstate_edge)
# --------------------------------------------------------------------------- #

def w0_from_vgen_powerstate(ps, vtor=None):
    """NEO ``vgen`` w0 [rad/s] for every batch element of a ``powerstate_edge``.

    A fresh input.gacode is written per batch element, cropped to the plasma domain
    (``ps.plasma["rho"]``) and coarsened to ``rotation_options['vgen_drho']`` (~0.01)
    so NEO only solves the pedestal surfaces and stays fast. When an imposed toroidal
    rotation ``vtor`` (batch, rho) [m/s] is supplied it is written into the ``# vtor``
    block so that ``vgen -er 2`` chains it through to w0/Er; otherwise the vtor = 0
    closure is used. Returns a detached (batch, rho) tensor on the plasma grid. The
    external DKE solve is not differentiable; caching (``vgen_every``) is the caller's
    job (see ``powerstate_edge._rotation_w0_vgen``).
    """
    opts = getattr(ps, "_rotation_options", {}) or {}
    vgen_options = dict(opts.get("vgen_options", {}))
    drho = float(opts.get("vgen_drho", 0.01))

    rho = ps.plasma["rho"]
    batch = rho.shape[0]
    rho_np = ps._as_numpy_cpu(rho)
    vtor_np = None if vtor is None else ps._as_numpy_cpu(vtor)

    rows = []
    with tempfile.TemporaryDirectory(prefix="rot_vgen_") as work:
        for b in range(batch):
            gac = f"{work}/input.gacode.{b}"

            # Build the gacode state for this batch element (rederive_profiles=False
            # so the powerstate state is not mutated).
            prof = copy.deepcopy(
                ps.from_powerstate(position_in_powerstate_batch=b, rederive_profiles=False)
            )

            # Crop to the plasma domain and coarsen to drho so NEO only solves the
            # pedestal surfaces (base resolution -> short runtime).
            rho_lo = float(rho_np[b].min())
            rho_new = np.clip(np.arange(rho_lo, 1.0 + 0.5 * drho, drho), rho_lo, 1.0)
            prof.changeResolution(rho_new=rho_new)

            nexp = prof.profiles["rho(-)"].shape[0]
            nion = prof.profiles["ni(10^19/m^3)"].shape[1]

            # w0 is the vgen output; zero it. Impose vtor via the # vtor block.
            prof.profiles["w0(rad/s)"] = np.zeros(nexp)
            if vtor_np is not None:
                vtor_b = np.interp(rho_new, rho_np[b], vtor_np[b])
                # Rigid toroidal rotation: same Vtor for every ion column.
                prof.profiles["vtor(m/s)"] = np.repeat(vtor_b[:, None], nion, axis=1)

            prof.write_state(file=gac)

            # Match NEO's thermal-ion count to the profiles unless overridden.
            vgen_kw = dict(vgen_options)
            vgen_kw.setdefault("n_ion", nion)

            rows.append(
                w0_from_vgen(
                    gac,
                    rho[b : b + 1],
                    zero_w0=True,
                    # The file already has vtor imposed; don't let w0_from_vgen
                    # re-zero the # vtor block we just wrote.
                    zero_vtor=(vtor_np is None),
                    dtype=ps.dfT.dtype,
                    device=ps.dfT.device,
                    **vgen_kw,
                )
            )
    return torch.cat(rows, dim=0).to(ps.dfT)


def resolve_vtor_powerstate(ps, inputs=None):
    """Imposed toroidal (fluid) velocity Vtor [m/s] as a (batch, rho) tensor, or
    ``None`` for the vtor = 0 closure. The same Vtor is used by both backends:
    analytic (Er += Vtor*Bp) and vgen (written into the # vtor block).

    Controlled by ``ps._rotation_options["vtor_source"]``:
        "zero" (default) : no imposed toroidal rotation.
        "user"           : interpolate ``vtor_pairs=[(rho, vtor_m_s), ...]``.
        "extract_initial": capture Vtor once from the input profiles and hold it
                           fixed (cached on ``ps._vtor_extracted``). A ``vtor(m/s)``
                           main-ion block is used directly; otherwise the residual
                           fluid Vtor is backed out of the initial force balance
                           from ``w0(rad/s)``:  Vtor = (Er_init - Er_closure) / Bp,
                           Er_init = w0_input * R * Bp, Er_closure = vtor=0 closure.
    """
    opts = getattr(ps, "_rotation_options", {}) or {}
    src = opts.get("vtor_source", "zero")
    if not src or src == "zero":
        return None

    rho = ps.plasma["rho"]                       # (batch, rho) sqrt-tor-flux
    batch = rho.shape[0]

    if src == "user":
        pairs = opts.get("vtor_pairs", None)
        if not pairs:
            raise ValueError("[rotation] vtor_source='user' needs "
                             "rotation_options['vtor_pairs']=[(rho, vtor_m_s), ...]")
        vtor = vtor_from_pairs(rho, pairs, dtype=ps.dfT.dtype, device=ps.dfT.device)
        return vtor if vtor.shape[0] == batch else vtor[:1].expand(batch, -1)

    if src == "extract_initial":
        cached = getattr(ps, "_vtor_extracted", None)
        if cached is not None and cached.shape == rho.shape:
            return cached

        prof = ps.profiles.profiles
        rho_src = np.asarray(prof["rho(-)"]).reshape(-1)
        rho_np = ps._as_numpy_cpu(rho)

        # (a) A measured main-ion vtor block is already the fluid velocity: use it.
        #     Guard on the main-ion column specifically -- CER reconstructions can
        #     store the measured rotation in the impurity column and leave the
        #     main-ion column zero, in which case fall through to the w0 extraction.
        vtor_main = (np.asarray(prof["vtor(m/s)"])[:, 0]
                     if "vtor(m/s)" in prof else None)
        if vtor_main is not None and np.abs(vtor_main).max() > 0:
            vtor_np = np.vstack([np.interp(rho_np[b], rho_src, vtor_main) for b in range(batch)])
            ps._vtor_extracted = torch.from_numpy(vtor_np).to(ps.dfT)
            return ps._vtor_extracted

        # (b) Back the residual fluid Vtor out of w0 via the force balance.
        if "w0(rad/s)" not in prof:
            raise ValueError("[rotation] vtor_source='extract_initial' needs "
                             "a 'vtor(m/s)' or 'w0(rad/s)' block in the input profiles")
        w0_src = np.asarray(prof["w0(rad/s)"]).reshape(-1)
        w0_input = torch.from_numpy(
            np.vstack([np.interp(rho_np[b], rho_src, w0_src) for b in range(batch)])
        ).to(ps.dfT)

        R, Bp = ps.plasma["R"], ps.plasma["B_p"]
        Er_init = w0_input * R * Bp
        if inputs is None:
            inputs = ps._rotation_inputs()
        # vtor=0 closure Er (diamagnetic + neoclassical poloidal), detached.
        Er_closure = calculate_rotation(**inputs, vtor=None)["Er"].detach()
        ps._vtor_extracted = (Er_init - Er_closure) / Bp
        return ps._vtor_extracted

    raise ValueError(f"[rotation] unknown vtor_source {src!r}")
