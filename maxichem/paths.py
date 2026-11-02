"""Data locations. The solver reads only the network files it is given; the checks and benches read reference data that
are not part of the repository (README.md, "Data for the checks and benches"), all under one root:

DATA: the environment variable CHEM_DATA (empty = unset; default: data/ beside this package, i.e. in a source checkout).
VULCAN_THERMO: the environment variable VULCAN_THERMO, the folder with VULCAN's NCHO_thermo_network.txt and NASA9/
(no default: reading it while unset raises).
"""
import os
from pathlib import Path

DATA = Path(os.environ.get("CHEM_DATA") or Path(__file__).resolve().parents[1] / "data")
QUAL = DATA / "truth/qual"                               # VULCAN truth set (folders QC, QCs, QL, ...; manifest.json)
DESIGNS = DATA / "benches/designs"                       # test designs: decision/P (panel P's rows, starts and forcing)
YARD = DATA / "benches/yardstick"                        # yardstick design (design/: recorded starts, rows, saved states)
AD_REF = DATA / "benches/kincuda_ad/refdata2.npz"        # torch-autograd derivatives of the torch reference call
LIBREF = DATA / "benches/kincuda/libref2.npz"            # the torch reference call's outputs and rate coefficients on
                                                         # the checks' inputs (tests/libref.py)
TRACER_REF = DATA / "benches/kintracer/ref3.npz"         # the reduced-tracer layer's reference outputs on the inputs of
                                                         # tests/kintracer_check.py
KINJAX = DATA / "caches/kinjax"                          # the JAX shim built from kinjax.cc at first use


def __getattr__(name):
    if name == "VULCAN_THERMO":
        v = os.environ.get("VULCAN_THERMO")
        if not v:
            raise RuntimeError("VULCAN_THERMO is not set: point it to the folder with VULCAN's NCHO_thermo_network.txt "
                               "and NASA9/ (README.md, Requirements)")
        return Path(v)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def need(*files):
    """Exit with a message naming every missing file or folder of these (reference data not in the repository)."""
    miss = [str(f) for f in files if not Path(f).exists()]
    if miss:
        raise SystemExit("missing reference data: " + ", ".join(miss) + "\nThe checks and benches read data that are "
                         "not part of this repository (README.md, \"Data for the checks and benches\"); set CHEM_DATA "
                         "to the folder that holds them.")
