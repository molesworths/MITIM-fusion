"""Cross-check DMD-dominant gamma/omega vs CGYRO-native (out.qlgyro.eigenvalue_spectrum)
per ky, for every radius where both a dmd_<case>_roa<X>.txt table and a
qlgyro_<case>_roa<X>/lin native spectrum exist. Goal: is DMD-dominant failure specific to
DAMPED high-ky modes (H_LowColl 0.85) or does it also fail where high-ky ETG is genuinely
UNSTABLE?

Reads only the saved dmd_<case>.txt tables and the native spectra, so it needs no
preserved cgyro_raw histories.

    mitim_qlgyro_dmd_vs_native --root <dir of qlgyro_* runs + dmd_*.txt tables>
"""
import argparse
import re
from pathlib import Path

import numpy as np

from mitim_tools.gacode_tools.scripts.qlgyro_dmd.common import (
    add_root_argument,
    read_native,
    resolve_root,
)


def read_dmd(txt):
    """Return {ky: (gamma1, omega1)} for mode-1 rows of a saved dmd_<case>.txt table."""
    out = {}
    for ln in Path(txt).read_text().splitlines():
        m = re.match(r"\s+([\d.]+)\s+\|\s+1\s+(-?[\d.]+)\s+(-?[\d.]+)", ln)
        if m:
            out[round(float(m.group(1)), 2)] = (float(m.group(2)), float(m.group(3)))
    return out


def report(root):
    for qdir in sorted(root.glob("qlgyro_*")):
        case = qdir.name.replace("qlgyro_", "")
        dmdtxt = root / f"dmd_{case}.txt"
        if not dmdtxt.exists():
            continue
        nat = read_native(qdir / "lin")
        if nat is None:
            continue
        ky, ev = nat
        dmd = read_dmd(dmdtxt)
        if not dmd:
            continue
        print(f"\n=== {case} ===")
        print(f"  {'ky':>6} | {'nat_g':>8} {'nat_w':>8} | {'dmd_g':>8} {'dmd_w':>8} | {'dg/|g|':>7}")
        for i, k in enumerate(ky):
            kk = round(k, 2)
            if kk not in dmd:
                continue
            ng, nw = ev[i]
            dg, dw = dmd[kk]
            rel = (dg - ng) / max(abs(ng), 1e-3)
            flag = "  <-- BAD" if abs(rel) > 0.5 and (abs(dg - ng) > 0.1) else ""
            print(f"  {k:6.2f} | {ng:8.3f} {nw:8.3f} | {dg:8.3f} {dw:8.3f} | {rel:7.2f}{flag}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_root_argument(parser)
    args = parser.parse_args()
    report(resolve_root(args))


if __name__ == "__main__":
    main()
