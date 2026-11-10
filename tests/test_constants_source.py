"""Each setting is written once, in maxichem/_constants.py or as a classified constant of the kernels.

What is enforced, by text scans of the kernel files with comments removed (no C++ parser):
- the names the kernels declare `constexpr` are exactly HEADER_SETTINGS, FORMULA_COEFFICIENTS and KERNEL_CODES, so a new
  kernel constant fails until it is classified here; each header setting is initialised from its header macro;
- outside `constexpr` declarations and `static_assert`s a kernel writes no float literal but 0, 1/2, 1 and 2;
- the C++ caller's copies equal _constants.py; every UPPER_CASE name docs/method.md quotes is a setting or a classified
  kernel constant.
Not enforced: integer literals, and what a 0, 1/2, 1 or 2 means where it is written.
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
# The kernels' constexprs, by kind: settings from _constants.py through the header (name: macro); published formula
# coefficients, which stay beside their formula (docs/method.md); and the kernels' own codes and offsets.
HEADER_SETTINGS = {name: f"{name}_VALUE" for name in _kernel_header._METHOD_SETTINGS} | {
    "NASA_SPLIT_T": "NASA_SPLIT_T_K"
}
FORMULA_COEFFICIENTS = {"GAMMA", "RODAS_C21", "RODAS_C43", "TROE_C0", "TROE_C1", "TROE_N0", "TROE_N1", "TROE_D"}
KERNEL_CODES = {
    "REFUSED_COLD", "REFUSED_BAD_INPUT", "REFUSED_HOT",  # a cell's screening status
    "GUARD_NONFINITE", "GUARD_UNFINISHED", "GUARD_DRIFT",  # the guard's failure bits
    "OZ", "ON1", "CP",  # the adjoint's work-array offsets and its checkpoint count
}  # fmt: skip
COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
DECLARATION = re.compile(r"\b(?:constexpr|static_assert)\b[^;]*;", re.DOTALL)  # to its end, over lines
FLOAT = re.compile(r"(?<![\w.])(\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)[fFlL]?(?![\w.])")
# Floats a kernel may write bare: 0 and 1, a half (the tie of a min or max, a mean) and 2 (a +-2 stoichiometric
# coefficient, a square's derivative, Rodas3's weight of K1 and 1 / GAMMA, all documented in docs/method.md, and the
# guard's clamping allowance of 2 atol per accepted step).
BARE_FLOATS = {0.0, 0.5, 1.0, 2.0}
SETTING_NAME = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")  # T_MAX, CAP, FACACC; not T3 or ADSolver


def _blank(match: re.Match[str]) -> str:
    return "\n" * match.group().count("\n")


def _kernel_files() -> dict[str, str]:
    """Return each kernel file's text, comments blanked so that line numbers stay right."""
    return {name: COMMENT.sub(_blank, text) for name, text in _kernel_header.kernel_sources("").items()}


def _kernel_text() -> str:
    return "".join(_kernel_files().values())


def _declarations(text: str) -> list[tuple[str, str, str]]:
    """Return (type, name, initialiser) of every constexpr declarator, a declaration wrapped over lines included."""
    out = []
    for match in re.finditer(r"\bconstexpr\s+([\w:]+)\s+([^;]*);", text, re.DOTALL):
        depth, start = 0, 0
        body = match.group(2)
        for i, c in enumerate(body + ","):
            depth += (c in "({[") - (c in ")}]")
            if c == "," and depth == 0:
                name, _, value = body[start:i].partition("=")
                out.append((match.group(1), name.split("[")[0].strip(), " ".join(value.split())))
                start = i + 1
    return out


def test_cpp_copies_equal_the_python_settings():
    text = "".join((ROOT / "cpp" / name).read_text() for name in ("maxichem.hpp", "maxichem.cpp"))
    for cpp, python in CPP_COPIES.items():
        values = re.findall(rf"constexpr [\w:* ]+?\b{cpp}\s*=\s*([^;]+);", text)
        assert len(values) == 1, f"{cpp}: {len(values)} definitions in cpp/"
        assert ast.literal_eval(values[0].strip()) == getattr(_constants, python), f"{cpp} differs from {python}"


def test_kernel_constants_are_classified_and_settings_come_from_the_header():
    declared = _declarations(_kernel_text())
    names = {name for _, name, _ in declared}
    classified = set(HEADER_SETTINGS) | FORMULA_COEFFICIENTS | KERNEL_CODES
    assert names == classified, (
        f"classify in this test: {sorted(names - classified)}; gone: {sorted(classified - names)}"
    )
    for name, macro in HEADER_SETTINGS.items():
        assert [(t, v) for t, n, v in declared if n == name] == [("double", macro)], name


def test_kernels_write_no_bare_floats_but_arithmetic():
    bare = []
    for name, text in _kernel_files().items():
        for number, line in enumerate(DECLARATION.sub(_blank, text).splitlines(), 1):
            bare += [f"{name}:{number}: {value}" for value in FLOAT.findall(line) if float(value) not in BARE_FLOATS]
    assert not bare, f"name these in _constants.py or beside their formula: {bare}"


def test_settings_named_in_the_method_exist():
    kernel_names = {name for _, name, _ in _declarations(_kernel_text())}
    spans = re.findall(r"`([^`]+)`", (ROOT / "docs" / "method.md").read_text())
    names = {name for span in spans for name in SETTING_NAME.findall(span)}
    missing = sorted(n for n in names if not hasattr(_constants, n) and n not in kernel_names)
    assert names
    assert not missing, f"docs/method.md names settings that do not exist: {missing}"
