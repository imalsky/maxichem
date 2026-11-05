"""Data locations. The solver reads only the network files it is given; the checks and benches read reference data that
are not part of the repository (README.md, "Tests and the data they need"), all under one root:

DATA: the environment variable CHEM_DATA (empty = unset; default: data/ beside this package, i.e. in a source checkout).
VULCAN_THERMO: the environment variable VULCAN_THERMO, the folder with the VULCAN networks and NASA9/: VULCAN-JAX's
src/vulcan3/thermo (the checks use its NCHO_photo_network.txt and SNCHO_photo_network_2025.txt; no default: reading it
while unset raises).
"""

import os
from pathlib import Path

DATA = Path(os.environ.get("CHEM_DATA") or Path(__file__).resolve().parents[1] / "data")
QUAL = DATA / "truth_v2/qual_ncho"  # VULCAN truth paths on NCHO_photo (QC, QCs, QL, ...)
BOXES = DATA / "truth_v2/boxes_ncho"  # VULCAN truth boxes on NCHO_photo (sobol design, shards)
DESIGNS = DATA / "benches/designs"  # test designs: decision/P (panel P's rows, starts and forcing)
YARD = DATA / "benches/yardstick"  # yardstick design (design/: recorded starts, rows, saved states)
LIBREF = DATA / "benches/kincuda/libref7.npz"  # the torch reference call's outputs on the checks' inputs
# (tests/libref.py), NCHO_photo
LIBREF_S = DATA / "benches/kincuda/libref8_SNCHO25.npz"  # the same for the sulfur network SNCHO_photo_network_2025
JREF_S = DATA / "benches/kincuda/jref_SNCHO25.npz"  # J of a 1-D WASP-39b model's layers on that network (lines, J)
AD_BOXES = DATA / "truth_v2/old_network/boxes_ncho_r5e-3/boxes_000.npz"  # input states of tests/kincuda_ad_check.py


def __getattr__(name):
    if name == "VULCAN_THERMO":
        v = os.environ.get("VULCAN_THERMO")
        if not v:
            raise RuntimeError(
                "VULCAN_THERMO is not set: point it to VULCAN-JAX's src/vulcan3/thermo (the VULCAN "
                "networks and NASA9/; README.md, Requirements)"
            )
        return Path(v)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def need(*files):
    """Exit with a message naming every missing file or folder of these (reference data not in the repository)."""
    miss = [str(f) for f in files if not Path(f).exists()]
    if miss:
        raise SystemExit(
            "missing reference data: " + ", ".join(miss) + "\nThe checks and benches read data that are "
            'not part of this repository (README.md, "Tests and the data they need"); set CHEM_DATA '
            "to the folder that holds them."
        )
