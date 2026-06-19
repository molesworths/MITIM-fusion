"""
EvaluationArchive
=================

A persistent, append-only store of *real* transport-code evaluations (TGLF, NEO
now; QLGYRO / CGYRO later) keyed on the realized input deck (input.tglf /
input.neo), so that future surrogate-assisted solves can retrieve physically
relevant nearest neighbours and warm-start their GPs.

Design (see surrogate_assisted_solver discussion):

  * The archive keys on the *transport-code input deck*, NOT on (x_lcfs, x_int).
    The deck is the domain of the invariant map ``deck -> fluxes``; (x_lcfs,
    x_int) is just the upstream parameterization and never appears here.

  * Each deck variable is classified into one of four groups:

      identity   - categorical / functional-identity keys (SAT_RULE, geometry
                   model, EM flags, species (Z, MASS) signature). Hard partition
                   keys: a TGLF point is never a neighbour of a NEO point, a
                   SAT_RULE=2 point never a neighbour of SAT_RULE=3.

      fixed      - continuous *fixed background*: geometry shape + magnetics
                   (kappa, delta, rmin, rmaj, q, magnetic shear, ...). These are
                   constant under the (x_lcfs, x_int) parameterization at a given
                   equilibrium, so they (and only they) gate ADMISSION: a record
                   at a different equilibrium is a different transformation.

      derived    - continuous quantities that CO-MOVE with the DVs/profiles
                   (betae, collisionality, Zeff, p_prime, ExB shear, per-species
                   Ti/Te and densities, ...). These are the same "secondary
                   features" the critical-gradient GP mean consumes. They enter
                   the NN *metric* but do NOT gate admission -- a record at the
                   same equilibrium with a different betae is still the same
                   transformation, just a different point on the manifold.

      dv_image   - continuous quantities the solver directly moves: the driving
                   gradients (RLTS/RLNS, i.e. a/Lx). Metric only.

  * Fluxes and their uncertainties are pulled VERBATIM from the run's
    ``*_turb`` / ``*_turb_stds`` (TGLF) and ``*_neoc`` / ``*_neoc_stds`` (NEO)
    arrays already computed by the transport modules -- nothing is recomputed.

This module currently implements featurization + the append store + record
construction from a ``power_transport`` evaluation. ``retrieve`` and ``admit``
are typed stubs: retrieval (deck-space NN) establishes *relevance*; admission
(tight background tolerance now, background-augmented GP later) establishes the
right to inject a record as same-transformation GP training data.
"""

import json
import uuid
import hashlib
import datetime
from pathlib import Path

import numpy as np

from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.LOGtools import printMsg as print

SCHEMA_VERSION = 1

# Plasma variables excluded from the matching metric (still kept verbatim in
# raw_deck for provenance / exact reconstruction). The MXH Fourier shape
# coefficients are redundant for retrieval: the low-order Miller moments
# (KAPPA/DELTA/ZETA/Q/SHEAR/...) already capture the magnetic geometry well
# enough, and the dozens of SHAPE_* columns would otherwise dominate the
# (gating) fixed-subspace distance with near-constant, high-dimensional noise.
IGNORE_FEATURE_PREFIXES = ("SHAPE_",)

# ----------------------------------------------------------------------------------------------------------------------------
# Per-code variable classification.  Names not listed fall through to a safe
# default (continuous -> "derived" = metric-but-not-gating; non-numeric ->
# ignored), so an unrecognized variable can never silently become an admission
# gate or be dropped from provenance (raw_deck always keeps the full deck).
#
# NEO names follow the GACODE standard and should be validated against a real
# input.neo from your runs; anything missing simply lands in "derived".
# ----------------------------------------------------------------------------------------------------------------------------

#   - the whole ``controls`` block (model flags + numerics) is the functional
#     identity: only its CATEGORICAL members (bool/int/str -- SAT_RULE, UNITS,
#     GEOMETRY_FLAG, COLLISION_MODEL, ...) enter the identity hash, so continuous
#     solver knobs that may drift per-eval (WIDTH found by bisection, KY, factors)
#     don't fragment the partition.
#   - the ``plasma`` block is geometry-dominated, so its default is ``fixed``
#     (admission-gating); only the explicitly listed profile-coupled quantities
#     go to ``derived``.  Validated against real input.tglf / input.neo decks
#     (MXH SHAPE_* coefficients land in ``fixed`` via the default).
#   - ``species`` gradients -> dv_image, the rest -> derived, (Z, MASS) -> signature.
CODE_SPECS = {
    "TGLF": {
        "species_identity_vars": ["ZS", "MASS"],
        "species_dv_vars": ["RLNS", "RLTS"],
        "species_derived_vars": ["TAUS", "AS", "VPAR", "VPAR_SHEAR", "VNS_SHEAR", "VTS_SHEAR"],
        # profile/DV-coupled plasma scalars (everything else in plasma -> fixed)
        "derived_plasma": [
            "VEXB", "VEXB_SHEAR", "XNUE", "ZEFF", "DEBYE", "BETAE",
            "P_PRIME_LOC", "BETA_LOC",
        ],
    },
    "NEO": {
        "species_identity_vars": ["Z", "MASS"],
        "species_dv_vars": ["DLNNDR", "DLNTDR"],
        "species_derived_vars": ["TEMP", "DENS", "ANISO"],
        "derived_plasma": [
            "OMEGA_ROT", "OMEGA_ROT_DERIV", "DPHI0DR", "BETA_STAR",
            "NU_1", "NU_2", "NU_3", "NU_4", "NU_5",
        ],
    },
}

# Per-code flux variable maps: GB flux array attribute on power_transport ->
# canonical channel name.  TGLF reads the *_turb arrays, NEO the *_neoc arrays.
_FLUX_VARS = {
    "TGLF": {
        "suffix": "turb",
        "channels": ["Qe", "Qi", "Ge", "GZ", "Mt", "Qie"],
        "gb_attr": {  # canonical -> "<X>GB_turb"
            "Qe": "QeGB", "Qi": "QiGB", "Ge": "GeGB",
            "GZ": "GZGB", "Mt": "MtGB", "Qie": "QieGB",
        },
    },
    "NEO": {
        "suffix": "neoc",
        "channels": ["Qe", "Qi", "Ge", "GZ", "Mt"],  # NEO has no turbulent exchange
        "gb_attr": {
            "Qe": "QeGB", "Qi": "QiGB", "Ge": "GeGB",
            "GZ": "GZGB", "Mt": "MtGB",
        },
    },
}


# ----------------------------------------------------------------------------------------------------------------------------
# Small coercion helpers
# ----------------------------------------------------------------------------------------------------------------------------

def _as_float(x):
    """Best-effort scalar float; returns None for non-numeric / sequence values."""
    try:
        if isinstance(x, (list, tuple, np.ndarray)):
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _to_np(arr):
    """Coerce a torch tensor / numpy array / list to a 1D numpy float array."""
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr, dtype=float).reshape(-1)


def _split_index(key):
    """'RLTS_2' -> ('RLTS', 2); 'KAPPA_LOC' -> ('KAPPA_LOC', None)."""
    if "_" in key:
        base, tail = key.rsplit("_", 1)
        if tail.isdigit():
            return base, int(tail)
    return key, None


# ----------------------------------------------------------------------------------------------------------------------------
# Featurization
# ----------------------------------------------------------------------------------------------------------------------------

def _is_categorical(v):
    """True for bool/int/str (functional-identity), False for floats/sequences.

    bool is a subclass of int, so it is captured; floats (continuous solver knobs)
    are excluded so they don't fragment the identity partition.
    """
    if isinstance(v, bool) or isinstance(v, str):
        return True
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return True
    return False


def _split_species(inputclass, spec):
    """Return ({idx: {base: val}}, set_of_consumed_plasma_keys).

    Handles TGLFinput (``.species`` dict) and base GACODEinput/NEOinput (species
    live in ``.plasma`` as ``DLNTDR_1`` etc.).
    """
    bases = set(spec["species_identity_vars"]) | set(spec["species_dv_vars"]) | set(spec["species_derived_vars"])
    species = getattr(inputclass, "species", None)
    if isinstance(species, dict) and species:
        out = {i: {k: v for k, v in sp.items()} for i, sp in species.items() if isinstance(sp, dict)}
        return out, set()  # species already separated from plasma by TGLFinput
    # parse from the plasma block
    out, consumed = {}, set()
    plasma = getattr(inputclass, "plasma", {}) or {}
    for key, val in plasma.items():
        base, idx = _split_index(key)
        if idx is not None and base in bases:
            out.setdefault(idx, {})[base] = val
            consumed.add(key)
    return out, consumed


def featurize_deck(code, inputclass):
    """Classify a deck into identity / fixed / derived / dv_image groups.

    controls block -> identity (categorical members hashed into identity_key);
    plasma block    -> fixed by default, derived for listed profile-coupled vars;
    species         -> dv_image (gradients) / derived (rest), role-canonicalized
                       (``RLTS__r0`` = electrons, ``__r1`` = main ion, then
                       impurities by Z) so the same physical role aligns across
                       decks with different species counts/orders; (Z, MASS) form
                       the species signature.
    """
    if code not in CODE_SPECS:
        raise KeyError(f"[EvaluationArchive] no feature spec for code '{code}'")
    spec = CODE_SPECS[code]

    controls = dict(getattr(inputclass, "controls", {}) or {})
    plasma = dict(getattr(inputclass, "plasma", {}) or {})

    id_vars = spec["species_identity_vars"]
    dv_vars = set(spec["species_dv_vars"])
    der_species = set(spec["species_derived_vars"])
    derived_plasma = set(spec["derived_plasma"])

    # --- species: gather, role-order, canonicalize ---
    species_by_idx, consumed = _split_species(inputclass, spec)

    def _role_sort_key(item):
        _, d = item
        z = _as_float(d.get(id_vars[0])) or 0.0
        m = _as_float(d.get(id_vars[1])) if len(id_vars) > 1 else 0.0
        return (0 if z < 0 else 1, z, m or 0.0)

    ordered = sorted(species_by_idx.items(), key=_role_sort_key)
    species_signature = []
    dv_image, derived = {}, {}
    for role, (idx, d) in enumerate(ordered):
        z = _as_float(d.get(id_vars[0]))
        m = _as_float(d.get(id_vars[1])) if len(id_vars) > 1 else None
        species_signature.append([None if z is None else round(z, 3),
                                  None if m is None else round(m, 3)])
        for base, val in d.items():
            fval = _as_float(val)
            if fval is None or base in id_vars:
                continue
            cname = f"{base}__r{role}"
            if base in dv_vars:
                dv_image[cname] = fval
            elif base in der_species:
                derived[cname] = fval
            # unknown species var -> ignore (kept in raw_deck)

    # --- plasma scalars: fixed by default, derived for profile-coupled ---
    fixed = {}
    for key, val in plasma.items():
        if key in consumed:
            continue
        if key.startswith(IGNORE_FEATURE_PREFIXES):
            continue  # excluded from metric, retained in raw_deck below
        fval = _as_float(val)
        if fval is None:
            continue
        if key in derived_plasma:
            derived[key] = fval
        else:
            fixed[key] = fval

    # --- identity = controls; hash only categorical members ---
    identity = {k: _json_safe(v) for k, v in controls.items()}
    categorical = {k: v for k, v in controls.items() if _is_categorical(v)}
    identity_key = _identity_key(code, categorical, species_signature)

    # raw_deck = full provenance (controls + plasma + species flattened)
    raw_deck = {k: _json_safe(v) for k, v in controls.items()}
    raw_deck.update({k: _json_safe(v) for k, v in plasma.items()})
    for idx, d in species_by_idx.items():
        for base, val in d.items():
            raw_deck[f"{base}_{idx}"] = _json_safe(val)

    return {
        "identity": identity,
        "fixed": fixed,
        "derived": derived,
        "dv_image": dv_image,
        "identity_key": identity_key,
        "species_signature": species_signature,
        "raw_deck": raw_deck,
    }


def _identity_key(code, categorical, species_signature):
    payload = json.dumps(
        {"code": code,
         "controls": {k: _json_safe(v) for k, v in sorted(categorical.items())},
         "species": species_signature},
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _json_safe(v):
    f = _as_float(v)
    if f is not None:
        return f
    if isinstance(v, (list, tuple, np.ndarray)):
        return [(_as_float(x) if _as_float(x) is not None else str(x))
                for x in np.asarray(v).reshape(-1).tolist()]
    return str(v)


# ----------------------------------------------------------------------------------------------------------------------------
# The archive
# ----------------------------------------------------------------------------------------------------------------------------

class EvaluationArchive:
    """Append-only, deck-keyed store of real transport evaluations.

    Backend: one JSON-lines file per code under ``<path>/code=<CODE>/records.jsonl``.
    JSONL is chosen for cheap, lock-light appends (parquet is not row-appendable
    and pyarrow is not guaranteed present); a later compaction step can roll
    these into a partitioned parquet dataset for DuckDB querying without changing
    the record schema.
    """

    def __init__(self, path):
        self.path = IOtools.expandPath(path)
        self.path.mkdir(parents=True, exist_ok=True)

    # -- write ------------------------------------------------------------

    def _code_file(self, code):
        d = self.path / f"code={code}"
        d.mkdir(parents=True, exist_ok=True)
        return d / "records.jsonl"

    def append(self, records):
        """Append a list of record dicts, grouped into per-code JSONL files."""
        by_code = {}
        for r in records:
            by_code.setdefault(r["code"], []).append(r)
        for code, recs in by_code.items():
            with open(self._code_file(code), "a") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")

    # -- record construction from a transport evaluation ------------------

    def records_from_transport(self, transport, turbulence, neoclassical):
        """Build per-(code, rho) records from a completed power_transport eval.

        Pulls fluxes verbatim from the GB arrays already on ``transport``
        (``QeGB_turb``/``QeGB_turb_stds`` for TGLF, ``QeGB_neoc``/``..._stds``
        for NEO).  Returns [] for any code whose deck objects are absent (e.g.
        analytic models), never raising.
        """
        parent_eval_id = uuid.uuid4().hex
        prov = self._provenance(transport)
        rho = _to_np(transport.powerstate.plasma["rho"][0, 1:])
        roa = _to_np(transport.powerstate.plasma["roa"][0, 1:]) if "roa" in transport.powerstate.plasma else np.full_like(rho, np.nan)
        qgb = _to_np(transport.powerstate.plasma["Qgb"][0, 1:]) if "Qgb" in transport.powerstate.plasma else np.full_like(rho, np.nan)

        records = []
        for code, code_obj in (("TGLF", turbulence), ("NEO", neoclassical)):
            if code_obj is None or code not in CODE_SPECS:
                continue
            inputs_files = getattr(code_obj, "inputs_files", None)
            if not isinstance(inputs_files, dict) or len(inputs_files) == 0:
                continue
            try:
                records += self._records_for_code(
                    code, transport, inputs_files, parent_eval_id, prov, rho, roa, qgb
                )
            except Exception as e:
                print(f"\t* [EvaluationArchive] {code} record build skipped (non-fatal): {e}", typeMsg="w")
        return records

    def _records_for_code(self, code, transport, inputs_files, parent_eval_id, prov, rho, roa, qgb):
        fspec = _FLUX_VARS[code]
        suffix = fspec["suffix"]

        # flux + sigma arrays (canonical channel -> np array over rho), verbatim
        flux_mean, flux_std = {}, {}
        for ch, gb in fspec["gb_attr"].items():
            mean_attr, std_attr = f"{gb}_{suffix}", f"{gb}_{suffix}_stds"
            if hasattr(transport, mean_attr):
                flux_mean[ch] = _to_np(getattr(transport, mean_attr))
                flux_std[ch] = _to_np(getattr(transport, std_attr)) if hasattr(transport, std_attr) else None

        rho_keys = list(inputs_files.keys())
        out = []
        for i in range(len(rho)):
            inputclass = _match_rho(inputs_files, rho_keys, rho[i])
            if inputclass is None:
                continue
            feats = featurize_deck(code, inputclass)
            fluxes = {ch: (float(flux_mean[ch][i]) if i < len(flux_mean[ch]) else None)
                      for ch in flux_mean}
            sig = {ch: (float(flux_std[ch][i]) if (flux_std[ch] is not None and i < len(flux_std[ch])) else None)
                   for ch in flux_mean}
            out.append({
                "record_id": uuid.uuid4().hex,
                "parent_eval_id": parent_eval_id,
                "schema_version": SCHEMA_VERSION,
                "code": code,
                "rho": float(rho[i]),
                "roa": float(roa[i]) if i < len(roa) else None,
                "Qgb": float(qgb[i]) if i < len(qgb) else None,
                "provenance": prov,
                "identity": {k: _json_safe(v) for k, v in feats["identity"].items()},
                "identity_key": feats["identity_key"],
                "species_signature": feats["species_signature"],
                "fixed": feats["fixed"],
                "derived": feats["derived"],
                "dv_image": feats["dv_image"],
                "fluxes": fluxes,
                "fluxes_sigma": sig,
                "raw_deck": feats["raw_deck"],
            })
        return out

    def _provenance(self, transport):
        prov = {
            "name": getattr(transport, "name", None),
            "evaluation_number": getattr(transport, "evaluation_number", None),
            "folder": str(getattr(transport, "folder", "")),
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        try:
            from mitim_tools import __mitimroot__
            prov["code_sha"] = IOtools.get_git_info(__mitimroot__)[0] if hasattr(IOtools, "get_git_info") else None
        except Exception:
            prov["code_sha"] = None
        return prov

    # -- retrieve / admit (typed stubs) -----------------------------------

    def retrieve(self, code, query_features, k=8,
                 w_fixed=4.0, w_derived=1.0, w_dv=1.0, tau_derived=None):
        """Deck-space nearest-neighbour retrieval (RELEVANCE).

        Hard-filter by ``identity_key``, then weighted-Euclidean distance over
        scaled (fixed, derived, dv_image) features, returning the k nearest with
        their fluxes and a *background-subspace* distance ``d_fixed`` (used by
        ``admit``).

        On a tolerance for the DERIVED quantities (the question): NO hard gate by
        default. A hard tolerance is reserved for the ``fixed`` background, which
        defines transformation identity (admission). The ``derived`` features
        (betae, collisionality, Ti/Te, rotation, ...) CO-MOVE with the DVs and
        are exactly what the GP is meant to interpolate over; a neighbour that is
        far in ``derived`` but close in ``fixed`` is still the SAME transformation
        and is legitimately informative (it teaches the GP the derived-dependence,
        e.g. the critical-gradient mean's secondary coefficients). So ``derived``
        enters only as a *soft metric weight* (``w_derived``) and is then further
        down-weighted by the GP kernel once admitted -- not rejected at retrieval.
        ``tau_derived`` (default None = off) is offered only as an optional
        generous cap to drop pathological neighbours; it should stay >> the kernel
        lengthscale in derived-space if used at all. Not yet implemented.
        """
        raise NotImplementedError(
            "retrieve(): identity-blocked deck NN; fixed-weighted metric, "
            "derived as soft weight (no hard gate; optional tau_derived), "
            "dv_image free. To be implemented once population is underway."
        )

    def admit(self, hits, step, mode="tight", tau_fixed=None, kappa=0.0):
        """Turn retrieved hits into GP added-points (RIGHT TO USE).

        tight (now): admit only hits whose background distance d_fixed < tau_fixed
        (same transformation), inflating Yvar by (kappa*d_fixed)^2 so residual
        background mismatch becomes uncertainty, not bias; inject via the
        ExactGPcustom train_X_added / train_Y_added path.

        augmented (later): GP input augmented with the 'fixed' background dims so
        any hit admits directly -- gated to the separate nonstationary-GP work.
        """
        raise NotImplementedError(
            "admit(): tight (background-tolerance) admission feeds the "
            "train_X_added path; augmented admission awaits the background-"
            "augmented GP. To be implemented after retrieve()."
        )


def _match_rho(inputs_files, rho_keys, rho_val, atol=1e-6):
    """Find the inputclass for a rho value (exact, else nearest key)."""
    if rho_val in inputs_files:
        return inputs_files[rho_val]
    best, best_d = None, np.inf
    for k in rho_keys:
        kf = _as_float(k)
        if kf is None:
            continue
        d = abs(kf - rho_val)
        if d < best_d:
            best, best_d = inputs_files[k], d
    return best
