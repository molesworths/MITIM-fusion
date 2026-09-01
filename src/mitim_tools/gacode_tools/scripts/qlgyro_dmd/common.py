"""Shared plumbing for the QLGYRO/DMD early-exit backtest scripts.

The backtest reads a tree of completed QLGYRO linear runs that is far too large to
version (~1.5 GB): `<root>/qlgyro_<case>/` with `lin/out.qlgyro.*_spectrum_*` and, for
the truncation studies, preserved per-ky CGYRO histories under `cgyro_raw/lin/rho_*/KY_*`.
That tree lives outside the repository; point the scripts at it with `--root` or the
`QLGYRO_DMD_DATA_ROOT` environment variable. Everything else (the DMD preprocessing, the
saved result arrays) is self-contained here.
"""
import os
import re
from pathlib import Path

import numpy as np

ENV_VAR = "QLGYRO_DMD_DATA_ROOT"

# Directory holding this script and its saved result arrays
HERE = Path(__file__).resolve().parent


def add_root_argument(parser):
    """Register the --root option on an argparse parser."""
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help=(
            "Directory of completed QLGYRO linear runs (qlgyro_<case>/...). "
            f"Defaults to ${ENV_VAR}."
        ),
    )
    return parser


def resolve_root(args):
    """Return the validated data root from --root or $QLGYRO_DMD_DATA_ROOT."""
    root = getattr(args, "root", None) or os.environ.get(ENV_VAR)
    if not root:
        raise SystemExit(
            "No data root given. Pass --root <dir> or set "
            f"{ENV_VAR} to the directory holding the qlgyro_<case>/ run folders."
        )
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Data root does not exist: {root}")
    if not any(root.glob("qlgyro_*")):
        raise SystemExit(f"No qlgyro_* run folders found under: {root}")
    return root


def ky_of(path):
    """ky from a `KY_<value>_PX0_*` CGYRO run directory name."""
    m = re.search(r"KY_([0-9.]+)_PX0", str(path))
    return float(m.group(1)) if m else np.nan


def read_native(lindir):
    """Native QLGYRO (ky, [gamma, omega]) spectra from a `lin/` directory.
    Returns (kys, eigenvalues) or None if the spectrum files are absent."""
    lindir = Path(lindir)
    kyf = sorted(lindir.glob("out.qlgyro.ky_spectrum_*"))
    evf = sorted(lindir.glob("out.qlgyro.eigenvalue_spectrum_*"))
    if not kyf or not evf:
        return None

    kys = []
    for ln in kyf[0].read_text().splitlines():
        t = ln.split()
        if len(t) == 1:
            try:
                kys.append(float(t[0]))
            except ValueError:
                pass
    # drop a leading count token (e.g. "31") if it matches the remaining length
    if kys and kys[0] == round(kys[0]) and int(kys[0]) == len(kys) - 1:
        kys = kys[1:]

    ev = []
    for ln in evf[0].read_text().splitlines():
        t = ln.split()
        if len(t) == 2:
            try:
                ev.append((float(t[0]), float(t[1])))
            except ValueError:
                pass

    n = min(len(kys), len(ev))
    return np.array(kys[:n]), np.array(ev[:n])
