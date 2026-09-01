"""Part 2 of the truncation backtest: QL-weight convergence under early stopping.

QLGYRO consumes, per ky: the dominant eigenvalue AND the QL weights. Part 1
(dmd_backtest.py) validated eigenvalue recovery; here we ask whether the QL
weights have also converged at the truncated stopping points the eigenvalue
agreement-gate accepts.

KEY CONVENTION FACTS (verified 2026-07-22, H_HighColl roa0.95 ky=0.35):
  * CGYRO's linear `bin.cgyro.ky_flux` is ALREADY amplitude-normalized: the
    flux trace is flat (converged ~4th digit) while kxky_phi grows ~1e10.
    Dividing by |phi|^2 gives garbage. ky_flux(final) IS the QL weight
    (up to QLGYRO's overall normalization).
  * `out.qlgyro.QL_weight_spectrum` layout: C-order (nky, nmodes, ns, field,
    type), type fastest, species order electrons-FIRST (TGLF convention),
    NOT the CGYRO input-species order of ky_flux. With that reorder,
    computed/native = constant (2.38 here), 100% sign agreement.
  * `out.cgyro.freq` columns are (omega, gamma) in that order.

w(t) = tail-mean (last NTAIL snapshots before cutoff) of
       ky_flux[s, {particle,energy,tor.stress}, field-summed, 0, t].
Error at window fraction f = ||w(n_f) - w(N)||_2 / ||w(N)||_2.

Requires dmd_backtest_records.npy from dmd_backtest.py (for the gate flags).

RESULT (11 *_noaln cases, ~330 kys): weights converge FASTER than the
frequency estimate. Gate-accepted points within 10% of final: 73/94/99% at
30/50/70% window (91/98/100% within 20%). The eigenvalue gate is the binding
condition; QL weights are not the limiter.

    mitim_qlgyro_dmd_backtest_ql --root <dir of qlgyro_* runs>
"""
import argparse
import warnings

import numpy as np

from mitim_tools.gacode_tools.scripts.qlgyro_dmd.common import (
    HERE,
    add_root_argument,
    ky_of,
    resolve_root,
)

FRACS = [0.3, 0.5, 0.7]
NTAIL = 5


def load_estimates(records_file):
    """{(case, ky, frac, estimator): (gamma_est, gamma_native)} from dmd_backtest.py."""
    R = np.load(records_file, allow_pickle=True)
    est = {}
    for case, ky, gn, f, e, g, w in R:
        est[(case, round(float(ky), 2), float(f), e)] = (float(g), float(gn))
    return est


def gate_ok(est, case, ky, f):
    a = est.get((case, round(ky, 2), f, "dmd-seeded"))
    b = est.get((case, round(ky, 2), f, "cgyro-freq"))
    if a is None or b is None:
        return None
    return abs(a[0] - b[0]) <= max(0.1 * abs(b[0]), 0.05)


def qlw(flux, n):
    i0 = max(n - NTAIL, 0)
    return flux[:, :3, :, 0, i0:n].sum(axis=2).mean(axis=-1)  # (ns, 3) field-summed


def collect(root, est):
    from pygacode.cgyro.data import cgyrodata

    rows = []  # case, ky, g_nat, frac, gate_accepted, ql_err
    for qdir in sorted(root.glob("qlgyro_*_noaln")):
        raw = qdir / "cgyro_raw" / "lin"
        if not raw.exists():
            continue
        case = qdir.name.replace("qlgyro_", "")
        for kdir in sorted(raw.glob("rho_*/KY_*_PX0_*"), key=ky_of):
            ky = ky_of(kdir)
            gg = est.get((case, round(ky, 2), 1.0, "cgyro-freq"))
            g_nat = gg[1] if gg else np.nan
            try:
                sim = cgyrodata(str(kdir) + "/", silent=True)
                sim.getflux()
            except Exception:
                continue
            flux = np.asarray(sim.ky_flux)
            N = flux.shape[-1]
            if N < 20:
                continue
            wref = qlw(flux, N)
            nref = np.linalg.norm(wref)
            if not np.isfinite(nref) or nref == 0:
                continue
            for f in FRACS:
                n = max(int(round(f * N)), 14)
                if n > N - 2:
                    continue
                werr = np.linalg.norm(qlw(flux, n) - wref) / nref
                rows.append((case, ky, g_nat, f, gate_ok(est, case, ky, f), werr))
    return rows


def report(rows):
    print(f"{len(rows)} (ky,frac) points  [w = tail-mean ky_flux, field-summed, (ns x pcl/en/stress)]")
    print("\nQL-weight L2 rel. error vs window fraction (UNSTABLE native g>=0.05):")
    print(f"  {'frac':>5} | {'all: med / p90 / <10% / <20%':>40} | {'gate-ACCEPTED: med / p90 / <10% / <20%':>44}")

    def st(rr):
        if not rr:
            return "-"
        e = np.array([x[5] for x in rr])
        return (f"{np.median(e):6.3f} / {np.percentile(e, 90):6.3f} / "
                f"{100*np.mean(e < 0.10):3.0f}% / {100*np.mean(e < 0.20):3.0f}%  n={len(e):3d}")

    for f in FRACS:
        sub = [r for r in rows if r[3] == f and np.isfinite(r[2]) and r[2] >= 0.05]
        acc = [r for r in sub if r[4]]
        print(f"  {f:5.1f} | {st(sub):>40} | {st(acc):>44}")
    print("\nDAMPED/MARGINAL g<0.05 (context only):")
    for f in FRACS:
        sub = [r for r in rows if r[3] == f and np.isfinite(r[2]) and r[2] < 0.05]
        if sub:
            e = np.array([x[5] for x in sub])
            print(f"  frac {f}: med {np.median(e):.3f}, <20% {100*np.mean(e < 0.2):.0f}%  n={len(e)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_root_argument(parser)
    parser.add_argument("--records", type=str, default=str(HERE / "dmd_backtest_records.npy"),
                        help="Record array written by dmd_backtest.py.")
    parser.add_argument("--out", type=str, default=str(HERE / "dmd_backtest_ql_rows.npy"))
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    root = resolve_root(args)
    rows = collect(root, load_estimates(args.records))
    np.save(args.out, np.array(rows, dtype=object), allow_pickle=True)
    report(rows)


if __name__ == "__main__":
    main()
