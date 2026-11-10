"""The C++ caller against the Python call, bit for bit, counts included.

Needs a CUDA toolkit: set CUDAToolkit_ROOT (CMake reads it) if CMake does not find one.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from maxichem import _constants, paths

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

ROOT = Path(__file__).parents[1]
CELLS = 4096
FORCED_TRIES = 2  # a small budget: most cells are unfinished, and the recovery recomputes them


@pytest.fixture(scope="module")
def export_program(tmp_path_factory, gpu):  # noqa: ARG001  # without a GPU it skips here, before building
    """Build the C++ caller and return its test program."""
    build = tmp_path_factory.mktemp("cpp_build")
    configured = subprocess.run(
        ["cmake", "-S", ROOT / "cpp", "-B", build, f"-DPython3_EXECUTABLE={sys.executable}"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert configured.returncode == 0, configured.stderr
    subprocess.run(["make", "-C", build, "-j8"], check=True, capture_output=True)
    return build / "test_export"


@pytest.mark.parametrize("network", ["NCHO_photo_network.txt", "SNCHO_photo_network_2025.txt"])
@pytest.mark.parametrize(
    ("photo", "tries"),
    [(1, _constants.DEFAULT_MAX_STEPS), (0, _constants.DEFAULT_MAX_STEPS), (1, FORCED_TRIES)],
    ids=["photolysis", "thermal", "recovery"],
)
def test_cpp_call_is_the_python_call(export_program, tmp_path, network, photo, tries):
    paths.need(paths.vulcan_thermo())
    case = [sys.executable, ROOT / "tests" / "cpp_export_case.py", tmp_path, CELLS, photo, tries, network]
    subprocess.run([str(q) for q in case], check=True, capture_output=True)
    result = subprocess.run(
        [export_program, tmp_path / "export", tmp_path / "case"], check=False, capture_output=True, text=True
    )
    print(result.stdout, flush=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "bitwise identical" in result.stdout
    assert ("(default arguments)" in result.stdout) == (tries == _constants.DEFAULT_MAX_STEPS)


@pytest.mark.parametrize(("edit", "refused"), [("sum", True), ("nan", True), ("inside", False)])
def test_cpp_refuses_what_python_refuses(export_program, tmp_path, edit, refused):
    paths.need(paths.vulcan_thermo())
    network = "NCHO_photo_network.txt"
    case = [sys.executable, ROOT / "tests" / "cpp_export_case.py", tmp_path, 1024, 1, 4000, network, edit]
    subprocess.run([str(q) for q in case], check=True, capture_output=True)
    assert (tmp_path / "case" / "refused.txt").exists() == refused
    result = subprocess.run(
        [export_program, tmp_path / "export", tmp_path / "case"], check=False, capture_output=True, text=True
    )
    print(result.stdout, flush=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert ("refused as the Python call" if refused else "bitwise identical") in result.stdout


def test_cpp_refuses_another_export_format(export_program, tmp_path):
    paths.need(paths.vulcan_thermo())
    case = [sys.executable, ROOT / "tests" / "cpp_export_case.py", tmp_path, 64, 1, FORCED_TRIES]
    subprocess.run([str(q) for q in case], check=True, capture_output=True)
    meta = tmp_path / "export" / "meta.txt"
    written = f"export_format {_constants.EXPORT_FORMAT}\n"
    assert written in meta.read_text()
    meta.write_text(meta.read_text().replace(written, f"export_format {_constants.EXPORT_FORMAT + 1}\n"))
    result = subprocess.run(
        [export_program, tmp_path / "export", tmp_path / "case"], check=False, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "export format" in result.stderr
