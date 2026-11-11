"""Data locations of the checks and benches, under one root.

The solver reads only the network files it is given. Each function reads its environment variable when it is called,
not at import. README.md lists the data. A check whose data are missing fails through ``need``.
"""

import os
from pathlib import Path

__all__ = [
    "ad_boxes",
    "boxes",
    "data",
    "designs",
    "jref_sulfur",
    "libref",
    "libref_sulfur",
    "need",
    "qual",
    "truth",
    "vulcan_thermo",
    "yard",
]


def data() -> Path:
    """Return the data root: CHEM_DATA, or data/ beside this package (empty counts as unset)."""
    return Path(os.environ.get("CHEM_DATA") or Path(__file__).resolve().parents[1] / "data")


def truth(name: str) -> Path:
    """Return a VULCAN truth set by name: boxes (Sobol design, shards) or paths (QC, QCs, QL, ...).

    truth_v3 holds the truth made with the collision limit that counts M (_constants.RATE_RULE); truth_v2 holds the
    truth of the rule before, kept only as a record.
    """
    return data() / "truth_v3" / name


def qual() -> Path:
    """Return the VULCAN truth paths on NCHO_photo (QC, QCs, QL, ...)."""
    return truth("qual_ncho")


def boxes() -> Path:
    """Return the VULCAN truth boxes on NCHO_photo (Sobol design, shards)."""
    return truth("boxes_ncho")


def designs() -> Path:
    """Return the test designs: decision/P holds panel P's rows, starts and forcing."""
    return data() / "benches/designs"


def yard() -> Path:
    """Return the yardstick design (design/: recorded starts, rows, saved states)."""
    return data() / "benches/yardstick"


def libref() -> Path:
    """Return the torch reference call's outputs on the checks' inputs, NCHO_photo."""
    return data() / "benches/kincuda/libref10.npz"


def libref_sulfur() -> Path:
    """Return the torch reference call's outputs on the checks' inputs, SNCHO_photo_network_2025."""
    return data() / "benches/kincuda/libref10_SNCHO25.npz"


def jref_sulfur() -> Path:
    """Return J of a 1-D WASP-39b model's layers on SNCHO_photo_network_2025 (lines, J)."""
    return data() / "benches/kincuda/jref_SNCHO25.npz"


def ad_boxes() -> Path:
    """Return the input states of the derivative checks."""
    return data() / "truth_v2/old_network/boxes_ncho_r5e-3/boxes_000.npz"


def vulcan_thermo() -> Path:
    """Return VULCAN_THERMO, the folder with the VULCAN networks and NASA9/; raise RuntimeError if it is unset."""
    value = os.environ.get("VULCAN_THERMO")
    if not value:
        raise RuntimeError(
            "VULCAN_THERMO is not set: point it to VULCAN-JAX's src/vulcan3/thermo (the VULCAN "
            "networks and NASA9/; README.md, Requirements)"
        )
    return Path(value)


def need(*files: str | Path) -> None:
    """Raise FileNotFoundError naming every missing file or folder of these (reference data not in the repository)."""
    missing = [str(f) for f in files if not Path(f).exists()]
    if missing:
        raise FileNotFoundError(
            "missing reference data: " + ", ".join(missing) + "\nThe checks and benches read data that are "
            'not part of this repository (README.md, "Tests and the data they need"); set CHEM_DATA '
            "to the folder that holds them."
        )
