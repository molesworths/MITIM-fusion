"""Truncation backtest of DMD eigenvalue extraction on preserved cgyro_raw field
histories (`<root>/qlgyro_*_noaln/cgyro_raw`).

Question: if the CGYRO run had been stopped at fraction f of its (already
early-exited) record, how well does DMD recover the FINAL native eigenvalue —
and does it beat CGYRO's own running frequency estimate at the same time?

Estimators per (ky, window):
  R0 dmd-maxgam : DMD eig with max gamma (max-gamma dominant-mode rule)
  R1 dmd-maxamp : DMD eig with max energy |amp|*||mode||
  R2 dmd-gated  : R1 restricted to |omega|<=25
  R3 dmd-seeded : DMD eig nearest (complex dist) to CGYRO running freq at cutoff
  B  cgyro-freq : CGYRO's own running (gamma,omega) at cutoff  [baseline]

Success: |g - g_nat| <= max(0.1*|g_nat|, 0.05).

Field preprocessing is imported from QLGYROtools so that the backtest exercises the
same code path as the production `dmd_agreement_gate`.

    mitim_qlgyro_dmd_backtest --root <dir of qlgyro_* runs>
"""
import argparse
import warnings

import numpy as np

from mitim_tools.gacode_tools.QLGYROtools import cgyro_field_matrix, hankel_embed
from mitim_tools.gacode_tools.scripts.qlgyro_dmd.common import (
    HERE,
    add_root_argument,
    ky_of,
    read_native,
    resolve_root,
)

FRACS = [0.3, 0.5, 0.7, 1.0]
DELAY = 6
RANK = 10
RULES = ["maxgam", "maxamp", "gated", "seeded"]


def dmd_eigs(X, dt):
    """Return list of (gamma, omega, energy) for all retained DMD eigenvalues."""
    from pydmd import DMD

    Xh, _ = hankel_embed(X, DELAY)
    rank = min(RANK, Xh.shape[1] - 2, Xh.shape[0])
    d = DMD(svd_rank=rank)
    d.fit(Xh)
    if len(d.eigs) == 0:
        return []
    om = 1j * np.log(d.eigs) / dt
    gam, omr = om.imag, om.real
    energy = np.abs(d.amplitudes) * np.linalg.norm(d.modes, axis=0)
    return list(zip(gam, omr, energy))


def select(eigs, rule, seed=None):
    if not eigs:
        return None
    if rule == "maxgam":
        return max(eigs, key=lambda e: e[0])[:2]
    if rule == "maxamp":
        return max(eigs, key=lambda e: e[2])[:2]
    if rule == "gated":
        cand = [e for e in eigs if abs(e[1]) <= 25.0]
        return max(cand, key=lambda e: e[2])[:2] if cand else max(eigs, key=lambda e: e[2])[:2]
    if rule == "seeded":
        g0, w0 = seed
        return min(eigs, key=lambda e: abs((e[0] - g0) + 1j * (e[1] - w0)))[:2]
    raise ValueError(rule)


def collect(root):
    """Run the truncation backtest over every *_noaln case under root."""
    from pygacode.cgyro.data import cgyrodata

    records = []  # (case, ky, g_nat, frac, estimator, g_est, w_est)
    for qdir in sorted(root.glob("qlgyro_*_noaln")):
        raw = qdir / "cgyro_raw" / "lin"
        if not raw.exists():
            continue
        nat = read_native(qdir / "lin")
        if nat is None:
            continue
        kys_nat, ev_nat = nat
        case = qdir.name.replace("qlgyro_", "")
        for kdir in sorted(raw.glob("rho_*/KY_*_PX0_*"), key=ky_of):
            ky = ky_of(kdir)
            j = int(np.argmin(np.abs(kys_nat - ky)))
            if abs(kys_nat[j] - ky) > 0.02 * max(ky, 1):
                continue
            g_nat, w_nat = ev_nat[j]
            try:
                sim = cgyrodata(str(kdir) + "/", silent=True)
                sim.getbigfield()
            except Exception:
                continue
            t = sim.t
            if t.size < 20:
                continue
            X, _ = cgyro_field_matrix(sim)
            if X is None:
                continue
            try:
                fr = np.loadtxt(kdir / "out.cgyro.freq")  # cols: omega, gamma
                if fr.ndim == 1:
                    fr = fr[None, :]
            except Exception:
                fr = None
            N = X.shape[1]
            dt = t[1] - t[0]
            for f in FRACS:
                n = max(int(round(f * N)), 14)
                if n > N:
                    continue
                i0 = n // 3  # drop first third as transient
                Xw = X[:, i0:n]
                if Xw.shape[1] < 12:
                    continue
                try:
                    eigs = dmd_eigs(Xw, dt)
                except Exception:
                    eigs = []
                # baseline: CGYRO running freq at cutoff (row n-1, clipped)
                if fr is not None and fr.shape[0] >= 1:
                    r = fr[min(n, fr.shape[0]) - 1]
                    gB, wB = r[1], r[0]
                    records.append((case, ky, g_nat, f, "cgyro-freq", gB, wB))
                    seed = (gB, wB)
                else:
                    seed = None
                for rule in RULES:
                    if rule == "seeded" and seed is None:
                        continue
                    s = select(eigs, rule, seed=seed)
                    if s is None:
                        continue
                    records.append((case, ky, g_nat, f, "dmd-" + rule, s[0], s[1]))
    return records


def ok(g_est, g_nat):
    return abs(g_est - g_nat) <= max(0.1 * abs(g_nat), 0.05)


def report(records):
    print(f"\n{len(records)} records over cases:", sorted({r[0] for r in records}))
    print("\nSuccess rate (|dg| <= max(10%,0.05)) vs window fraction")
    for subset, sel in [("UNSTABLE native g>=0.05", lambda g: g >= 0.05),
                        ("DAMPED/MARGINAL g<0.05", lambda g: g < 0.05)]:
        print(f"\n--- {subset} ---")
        ests = ["cgyro-freq", "dmd-maxgam", "dmd-maxamp", "dmd-gated", "dmd-seeded"]
        print(f"  {'frac':>5} | " + " | ".join(f"{e:>10}" for e in ests))
        for f in FRACS:
            row = []
            for e in ests:
                rr = [r for r in records if r[3] == f and r[4] == e and sel(r[2])]
                row.append(f"{100*np.mean([ok(r[5], r[2]) for r in rr]):6.0f}% n={len(rr):3d}" if rr else "     -")
            print(f"  {f:5.1f} | " + " | ".join(f"{c:>10}" for c in row))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_root_argument(parser)
    parser.add_argument("--out", type=str, default=str(HERE / "dmd_backtest_records.npy"),
                        help="Where to save the record array (consumed by dmd_backtest_ql).")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    records = collect(resolve_root(args))
    np.save(args.out, np.array(records, dtype=object), allow_pickle=True)
    report(records)


if __name__ == "__main__":
    main()
