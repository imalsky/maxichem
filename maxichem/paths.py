"""Data locations of the checks and benches, under one root.

Public names: ``DATA`` and the paths below it, ``VULCAN_THERMO``, ``need``. The solver reads only the network files it
is given. CHEM_DATA sets the root (default: data/ in a checkout). VULCAN_THERMO is the folder with the VULCAN networks
and NASA9/; reading it while the variable is unset raises. README.md lists the data. Each name reads its environment
variable when it is used, not at import.
"""

import os
from pathlib import Path

_UNDER_DATA = {
    "QUAL": "truth_v2/qual_ncho",  # VULCAN truth paths on NCHO_photo (QC, QCs, QL, ...)
    "BOXES": "truth_v2/boxes_ncho",  # VULCAN truth boxes on NCHO_photo (Sobol design, shards)
    "DESIGNS": "benches/designs",  # test designs: decision/P (panel P's rows, starts and forcing)
    "YARD": "benches/yardstick",  # yardstick design (design/: recorded starts, rows, saved states)
    "LIBREF": "benches/kincuda/libref8.npz",  # the torch reference call's outputs on the checks' inputs, NCHO_photo
    "LIBREF_S": "benches/kincuda/libref9_SNCHO25.npz",  # the same for the sulfur network SNCHO_photo_network_2025
    "JREF_S": "benches/kincuda/jref_SNCHO25.npz",  # J of a 1-D WASP-39b model's layers on that network (lines, J)
    "AD_BOXES": "truth_v2/old_network/boxes_ncho_r5e-3/boxes_000.npz",  # input states of the derivative checks
}
__all__ = [
    "AD_BOXES",
    "BOXES",
    "DATA",
    "DESIGNS",
    "JREF_S",
    "LIBREF",
    "LIBREF_S",
    "QUAL",
    "VULCAN_THERMO",
    "YARD",
    "need",
]
# Served by __getattr__.
AD_BOXES: Path
BOXES: Path
DATA: Path
DESIGNS: Path
JREF_S: Path
LIBREF: Path
LIBREF_S: Path
QUAL: Path
VULCAN_THERMO: Path
YARD: Path


def _data_root() -> Path:
    """Return the data root: CHEM_DATA, or data/ beside this package (empty counts as unset)."""
    return Path(os.environ.get("CHEM_DATA") or Path(__file__).resolve().parents[1] / "data")


def __getattr__(name: str) -> Path:
    """Serve the data locations when they are read."""
    if name == "DATA":
        return _data_root()
    if name in _UNDER_DATA:
        return _data_root() / _UNDER_DATA[name]
    if name == "VULCAN_THERMO":
        value = os.environ.get("VULCAN_THERMO")
        if not value:
            raise RuntimeError(
                "VULCAN_THERMO is not set: point it to VULCAN-JAX's src/vulcan3/thermo (the VULCAN "
                "networks and NASA9/; README.md, Requirements)"
            )
        return Path(value)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List the public names."""
    return sorted(__all__)


def need(*files: str | Path) -> None:
    """Raise FileNotFoundError naming every missing file or folder of these (reference data not in the repository)."""
    missing = [str(f) for f in files if not Path(f).exists()]
    if missing:
        raise FileNotFoundError(
            "missing reference data: " + ", ".join(missing) + "\nThe checks and benches read data that are "
            'not part of this repository (README.md, "Tests and the data they need"); set CHEM_DATA '
            "to the folder that holds them."
        )
