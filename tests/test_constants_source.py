"""Each setting is written once: the C++ caller's copies equal maxichem/_constants.py."""

import ast
import re
from pathlib import Path

from maxichem import _constants

ROOT = Path(__file__).parents[1]
# The C++ caller's copies of Python settings, by name: it cannot import them, so this test holds them equal.
CPP_COPIES = {
    "kExportFormat": "EXPORT_FORMAT",
    "kGuardBlockThreads": "GUARD_BLOCK_THREADS",
    "kWarpLanes": "WARP_LANES",
    "kPhotolysisArgument": "PHOTOLYSIS_ARGUMENT",
    "kDefaultRtol": "DEFAULT_RTOL",
    "kDefaultMaxSteps": "DEFAULT_MAX_STEPS",
}


def test_cpp_copies_equal_the_python_settings():
    text = "".join((ROOT / "cpp" / name).read_text() for name in ("maxichem.hpp", "maxichem.cpp"))
    for cpp, python in CPP_COPIES.items():
        values = re.findall(rf"constexpr [\w:* ]+?\b{cpp}\s*=\s*([^;]+);", text)
        assert len(values) == 1, f"{cpp}: {len(values)} definitions in cpp/"
        assert ast.literal_eval(values[0].strip()) == getattr(_constants, python), f"{cpp} differs from {python}"
