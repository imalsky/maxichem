"""Each default network's VULCAN truth sets, and loading their boxes, paths and test panels for bench/kintest.py."""

import hashlib
import json
import re
from dataclasses import dataclass

import numpy as np

from maxichem import paths
from maxichem._checks import H2S_PER_H2_SOLAR, floored_log10

HOLD = 1500.0  # s: one row of a held path
BOX_KEYS = ("box", "X", "T", "P", "J", "lit", "kind")


@dataclass(frozen=True)
class TruthSets:
    """A network's VULCAN truth: its box sets, and its qual paths by family (an empty family is not in the set).

    no_result: per box set, the boxes whose VULCAN run ended before a stop (the generator's step cap; the manifest
    lists each as an error "stop ... s not hit"): their count and the sha256 of their sorted ids, first 16 hex digits.
    A set with any other error, or with other such boxes, is refused.
    """

    boxes: tuple[str, ...]
    no_result: dict[str, tuple[int, str]]
    qual: str
    per_call: tuple[str, ...]  # the QC family: T, P and dt every call
    holds: tuple[str, ...]  # the QH family: 1500-s holds
    long_holds: tuple[str, ...]  # QL
    long_calls: tuple[str, ...]  # QLc


TRUTH = {
    "NCHO_photo_network.txt": TruthSets(
        ("boxes_ncho",), {}, "qual_ncho", ("QC", "QCs", "twins/QC"), ("QH", "QHs", "twins/QH"), ("QL",), ("QLc",)
    ),
    "SNCHO_photo_network_2025.txt": TruthSets(
        ("boxes_sncho25", "boxes_sncho25p"),
        {"boxes_sncho25": (117, "21b9ae4b86f02ac9"), "boxes_sncho25p": (54, "58d5356b1e18f59f")},
        "qual_sncho25",
        ("QC", "QCs", "twins/QC"),
        (),
        ("QL",),
        (),
    ),
}
# Misses against VULCAN that are understood, by box set and box: the worst error (dex) and why it is kept. kintest fails
# on any other miss, and on a pinned one that is gone or whose worst error moved by more than its PINNED_CHANGE.
KNOWN_MISSES = {
    ("boxes_sncho25", 1939): (
        4.341,
        "S2; VULCAN's error: a Radau reference is within 0.084 dex of maxichem (notes.md, 2026-11-04 15:47)",
    ),
    ("boxes_sncho25p", 96): (
        0.155,
        "S2 at 386 K, 9e-5 bar, lit; maxichem's error at atol 1e-25, the default the owner keeps (0.003 dex at 1e-28)",
    ),
}


def _no_result(folder, manifest, declared):
    """Return the ids of the boxes the manifest's errors list as runs that ended before a stop.

    Refuses any other error, and a set whose such boxes differ from what TRUTH declares (count and ids' digest).
    """
    ids, other = [], []
    for error in manifest.get("errors", []):
        stopped = re.fullmatch(r"box (\d+): RuntimeError: stop \S+ s not hit", error)
        if stopped:
            ids.append(int(stopped.group(1)))
        else:
            other.append(error)
    ids = sorted(ids)
    digest = hashlib.sha256(np.asarray(ids, np.int64).tobytes()).hexdigest()[:16]
    if other or (len(ids), digest) != (declared or (0, hashlib.sha256(b"").hexdigest()[:16])):
        raise SystemExit(
            f"{folder}: errors that are not a run ended before a stop {other[:3]}, or {len(ids)} such boxes (ids "
            f"sha256 {digest}) where the network's TRUTH declares {declared}"
        )
    return np.asarray(ids, int)


def complete_box_set(folder, species, declared):
    """Read a box set's manifest; refuse a set that is not complete: (manifest, shards, first shard, columns).

    columns: the network's species in the shards' species; the shards' other species must be 0 in every state.
    """
    manifest = json.loads((folder / "manifest.json").read_text())
    shards = [folder / f for f in manifest.get("shards", [])]
    no_result = _no_result(folder, manifest, declared)
    bad = {k: len(manifest.get(k, [])) for k in ("bad_states", "drift_over_1e-6")}
    ids = np.concatenate([np.load(f)["box"] for f in shards]) if shards else np.zeros(0, int)
    cover = np.zeros(manifest["boxes"], int)
    np.add.at(cover, np.concatenate([ids, np.asarray(manifest.get("quarantine", []), int), no_result]), 1)
    if manifest.get("pending", 1) or any(bad.values()) or len(ids) != manifest["accepted"] or (cover != 1).any():
        raise SystemExit(
            f"{folder}: not complete (manifest: {manifest.get('pending')} boxes not run, {bad}, "
            f"{manifest['accepted']} accepted; shards hold {len(ids)} boxes, design boxes not held exactly once "
            f"(accepted, quarantined or without a VULCAN result) {int((cover != 1).sum())})"
        )
    print(
        f"{folder.name}: {manifest['accepted']:,} of {manifest['boxes']:,} accepted, "
        f"{len(manifest['quarantine'])} quarantined (left out), {len(no_result)} without a VULCAN result (its run "
        "ended before a stop, at the generator's step cap; not scored)"
    )
    first = np.load(shards[0])
    truth_species = [str(q) for q in first["species"]]
    for f in shards:
        z = np.load(f)
        if (
            [str(q) for q in z["species"]] != truth_species
            or not np.array_equal(z["stops"], first["stops"])
            or not np.array_equal(z["lines"], first["lines"])
        ):
            raise SystemExit(f"{f}: species, stops or photolysis lines differ from the other shards'")
    columns = network_columns(truth_species, species, folder)
    if len(columns) < len(truth_species):  # the truth-only species must be 0: a shard at a time
        for f in shards:
            network_columns(truth_species, species, f, np.load(f)["X"])
    return manifest, shards, first, columns


def load_boxes(shards, cap):
    """Return the first cap boxes (None: all) of the shards, by field."""
    box_set = {k: [] for k in BOX_KEYS}
    loaded = 0
    for f in shards:
        if cap is not None and loaded >= cap:
            break
        z = np.load(f)
        for k in BOX_KEYS:
            box_set[k].append(z[k])
        loaded += len(z["box"])
    return {k: np.concatenate(v)[:cap] for k, v in box_set.items()}


def network_columns(truth_species, species, where, *states):
    """Return the columns of the network's species in a truth file's species.

    A truth species the network lacks (a condensate the reader skips) must be 0 in every state given.
    """
    missing = [q for q in species if q not in truth_species]
    extra = [i for i, q in enumerate(truth_species) if q not in species]
    if missing or any(np.abs(s[..., extra]).max(initial=0.0) != 0.0 for s in states if extra):
        raise SystemExit(f"{where}: network species {missing} not in the truth, or truth-only species not 0")
    return [truth_species.index(q) for q in species]


def check_jobs(qual):
    """Refuse a truth set whose paths differ from its job list."""
    jobs = json.loads((qual / "jobs.json").read_text())
    rel = [j["out"].split(f"/{qual.name}/", 1)[-1] for j in jobs]
    want = set(rel)
    have = {str(f.relative_to(qual)) for f in qual.rglob("*.npz")}
    if want != have or len(want) != len(rel):
        raise SystemExit(
            f"{qual}: {len(rel) - len(want)} duplicate jobs, {len(want - have)} of {len(want)} listed "
            f"truth paths missing {sorted(want - have)[:3]}, {len(have - want)} not listed "
            f"{sorted(have - want)[:3]}"
        )
    short = []
    for j, r in zip(jobs, rel, strict=True):
        z = np.load(qual / r)
        hold = "saved_h" in z.files
        s0 = int(z["h0"] if hold else z["k0"]) if str(z["kind"]) == "twin" else 0
        rows = len(z["saved_h"]) if hold else len(z["x_call"])
        if rows != min(int(j["n"]), len(np.load(j["src"])["T"]) - s0):
            short.append(r)
    if short:
        raise SystemExit(f"{qual}: {len(short)} truth paths shorter than their jobs, e.g. {short[:3]}")


def load_paths(qual, families, species, gas_columns, log_floors, hold):
    """Load every path of the given families of the truth set qual."""
    folders = [qual / family for family in families]
    return [
        load_path(f, species, gas_columns, log_floors, hold) for folder in folders for f in sorted(folder.glob("*.npz"))
    ]


def load_path(f, species, gas_columns, log_floors, hold):
    """Load one qual truth path, on the network's species.

    Forcing rows (calls, or 1500-s holds if hold), start, mixes keyed to rows, stored states
    (row r = the state after row r) as floored log10 of the scored gases. A kick twin starts at its kick row from
    x_kicked; its mix at that row is already in x_kicked.
    """
    z = np.load(f)
    if not bool(z["accepted"]):
        raise SystemExit(f"{f}: the run was not accepted")
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
    x0 = z["x_kicked"] if twin else z["x0"]
    columns = network_columns([str(q) for q in z["species"]], species, f, stored, x0, z["mix_x"])
    stored = stored[:, columns]
    return {
        "T": z["T"],
        "P": z["Pbar"],
        "dt": None if hold else z["dt"],
        "first_row": first_row,
        "x0": x0[columns],
        "mix": [
            (int(a), float(fr), y[columns] / y[columns].sum())
            for a, fr, y in zip(z["mix_at"], z["mix_f"], z["mix_x"], strict=True)
            if int(a) > first_row or not twin
        ],
        "rows": rows.astype(np.int64),
        "truth": floored_log10(stored[:, gas_columns], log_floors, np).astype(np.float32),
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
    """Decision panel P: GCM-like paths (T, P every call) with their recorded mixes; no truth at a fixed dt.

    Its gas has 53 species, mapped by name; on a network with H2S, H2S is H2S_PER_H2_SOLAR times H2.
    """
    design = paths.designs() / "decision/P"
    r, s, forcing = (np.load(design / f"{f}.npz") for f in ("rows", "starts", "forcing"))
    s53 = [str(q) for q in s["species53"]]

    def onto_network(a):
        x = np.stack([a[..., s53.index(q)] if q in s53 else np.zeros(a.shape[:-1]) for q in species], -1)
        if "H2S" in species and "H2S" not in s53:
            x[..., species.index("H2S")] = H2S_PER_H2_SOLAR * x[..., species.index("H2")]
        return x

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
