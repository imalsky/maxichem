"""Load the VULCAN truth paths and test panels that bench/kintest.py scores."""

import json

import numpy as np

from maxichem import paths
from maxichem._checks import FLOORS

FLOOR = np.log10(FLOORS)
TINY = 1e-300  # mole fraction: keeps log10 finite, far below every floor
HOLD = 1500.0  # s: one row of a held path


def floored(x):
    """Return the floored log10 of the 12 gases' mole fractions; a negative or non-finite value is NaN (a failure)."""
    return np.maximum(np.log10(np.maximum(np.where(np.isfinite(x) & (x >= 0), x, np.nan), TINY)), FLOOR)


def check_jobs():
    """Refuse a truth set whose paths differ from its job list."""
    jobs = json.loads((paths.qual() / "jobs.json").read_text())
    rel = [j["out"].split(f"/{paths.qual().name}/", 1)[-1] for j in jobs]
    want = set(rel)
    have = {str(f.relative_to(paths.qual())) for f in paths.qual().rglob("*.npz")}
    if want != have or len(want) != len(rel):
        raise SystemExit(
            f"{paths.qual()}: {len(rel) - len(want)} duplicate jobs, {len(want - have)} of {len(want)} listed "
            f"truth paths missing {sorted(want - have)[:3]}, {len(have - want)} not listed "
            f"{sorted(have - want)[:3]}"
        )
    short = []
    for j, r in zip(jobs, rel, strict=True):
        z = np.load(paths.qual() / r)
        hold = "saved_h" in z.files
        s0 = int(z["h0"] if hold else z["k0"]) if str(z["kind"]) == "twin" else 0
        rows = len(z["saved_h"]) if hold else len(z["x_call"])
        if rows != min(int(j["n"]), len(np.load(j["src"])["T"]) - s0):
            short.append(r)
    if short:
        raise SystemExit(f"{paths.qual()}: {len(short)} truth paths shorter than their jobs, e.g. {short[:3]}")


def load_paths(families, species, gas_columns, hold):
    """Load every path of the given families."""
    folders = [paths.qual() / family for family in families]
    return [load_path(f, species, gas_columns, hold) for folder in folders for f in sorted(folder.glob("*.npz"))]


def load_path(f, species, gas_columns, hold):
    """Load one qual truth path.

    Forcing rows (calls, or 1500-s holds if hold), start, mixes keyed to rows, stored states
    (row r = the state after row r) as floored log10 of the 12 gases. A kick twin starts at its kick row from x_kicked;
    its mix at that row is already in x_kicked.
    """
    z = np.load(f)
    if [str(q) for q in z["species"]] != species or not bool(z["accepted"]):
        raise SystemExit(f"{f}: its species differ from the other paths', or the run was not accepted")
    twin = str(z["kind"]) == "twin"
    first_row = int(z["h0"] if hold else z["k0"]) if twin else 0
    if hold:
        rows, stored = z["saved_h"], z["x_saved"]
    elif "x_call" in z.files:
        stored = z["x_call"]
        rows = np.arange(first_row, first_row + len(stored))
        if first_row + len(stored) != len(z["T"]):
            raise SystemExit(f"{f}: {first_row + len(stored)} stored rows, {len(z['T'])} forcing rows")
    else:
        rows, stored = z["saved_call"], z["x_saved"]
    return {
        "T": z["T"],
        "P": z["Pbar"],
        "dt": None if hold else z["dt"],
        "first_row": first_row,
        "x0": z["x_kicked"] if twin else z["x0"],
        "mix": [
            (int(a), float(fr), y / y.sum())
            for a, fr, y in zip(z["mix_at"], z["mix_f"], z["mix_x"], strict=True)
            if int(a) > first_row or not twin
        ],
        "rows": rows.astype(np.int64),
        "truth": floored(stored[:, gas_columns]).astype(np.float32),
        "seg": (z["seg_start"], [str(q) for q in z["seg_names"]]) if "seg_start" in z.files else None,
        "meta": {
            "id": int(z["id"]),
            "panel": str(z["panel"]),
            "kind": str(z["kind"]),
            "stratum": str(z["stratum"]),
            "start": str(z["start"]),
            "comp": str(z["comp"]),
        },
    }


def load_p(species):
    """Decision panel P: GCM-like paths (T, P every call) with their recorded mixes; no truth at a fixed dt."""
    design = paths.designs() / "decision/P"
    r, s, forcing = (np.load(design / f"{f}.npz") for f in ("rows", "starts", "forcing"))
    s53 = [str(q) for q in s["species53"]]

    def onto_network(a):
        return np.stack([a[..., s53.index(q)] if q in s53 else np.zeros(a.shape[:-1]) for q in species], -1)

    T, p_bar, mix_path, mix_row, mix_fraction, mix_gas = (
        forcing[k] for k in ("T", "Pbar", "mix_row", "mix_k", "mix_f", "mix_x")
    )
    x53, mix_gas = onto_network(s["x53"]), onto_network(mix_gas)  # by species name (the others 0)
    out = []
    for p in range(len(r["N"])):
        mixes = np.flatnonzero(mix_path == p)
        out.append(
            {
                "T": T[p],
                "P": p_bar[p],
                "dt": None,
                "first_row": 0,
                "x0": x53[r["unit"][p]],
                "mix": [(int(mix_row[i]), float(mix_fraction[i]), mix_gas[i] / mix_gas[i].sum()) for i in mixes],
                "rows": None,
                "truth": None,
                "seg": None,
                "meta": {
                    "id": int(r["job"][p]),
                    "panel": "P",
                    "kind": "main",
                    "stratum": "-",
                    "start": f"age{int(r['age_idx'][p])}",
                    "comp": "-",
                },
            }
        )
    return out
