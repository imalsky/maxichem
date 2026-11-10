"""Each setting is written once, in maxichem/_constants.py or beside its formula in the kernels.

The C++ caller's copies equal _constants.py, the kernels write no bare setting, and docs/method.md names only settings
that exist.
"""

import ast
import re
from pathlib import Path

from maxichem import _constants, _kernel_header

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
COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
FLOAT = re.compile(r"(?<![\w.])(\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)[fFlL]?(?![\w.])")
# Floats a kernel may write bare: 0 and 1, a half (the tie of a min or max, a mean) and 2 (a +-2 stoichiometric
# coefficient, a square's derivative, Rodas3's weight of K1 and 1 / GAMMA, all documented in docs/method.md, and the
# guard's clamping allowance of 2 atol per accepted step). Any other value is a setting, a constant or a formula's
# coefficient, and gets a name.
BARE_FLOATS = {0.0, 0.5, 1.0, 2.0}
SETTING_NAME = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")  # T_MAX, CAP, FACACC; not T3 or ADSolver


def test_cpp_copies_equal_the_python_settings():
    text = "".join((ROOT / "cpp" / name).read_text() for name in ("maxichem.hpp", "maxichem.cpp"))
    for cpp, python in CPP_COPIES.items():
        values = re.findall(rf"constexpr [\w:* ]+?\b{cpp}\s*=\s*([^;]+);", text)
        assert len(values) == 1, f"{cpp}: {len(values)} definitions in cpp/"
        assert ast.literal_eval(values[0].strip()) == getattr(_constants, python), f"{cpp} differs from {python}"


def test_kernels_write_no_bare_floats_but_arithmetic():
    bare = []
    for name, text in _kernel_header.kernel_sources("").items():
        for number, line in enumerate(COMMENT.sub(lambda m: "\n" * m.group().count("\n"), text).splitlines(), 1):
            if re.match(r"\s*(?:constexpr|static_assert)\b", line):  # a named constant's definition, or a check of one
                continue
            bare += [f"{name}:{number}: {value}" for value in FLOAT.findall(line) if float(value) not in BARE_FLOATS]
    assert not bare, f"name these in _constants.py or beside their formula: {bare}"


def test_kernels_take_the_method_settings_from_the_header():
    text = "".join(_kernel_header.kernel_sources("").values())
    for name in _kernel_header._METHOD_SETTINGS:
        assert re.findall(rf"constexpr double {name} = (\w+);", text) == [f"{name}_VALUE"], name


def test_settings_named_in_the_method_exist():
    kernels = COMMENT.sub("", "".join(_kernel_header.kernel_sources("").values()))
    spans = re.findall(r"`([^`]+)`", (ROOT / "docs" / "method.md").read_text())
    names = {name for span in spans for name in SETTING_NAME.findall(span)}
    missing = sorted(n for n in names if not hasattr(_constants, n) and not re.search(rf"\b{n}\b", kernels))
    assert names
    assert not missing, f"docs/method.md names settings that do not exist: {missing}"
