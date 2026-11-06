"""The C++ caller against the Python call, bit for bit, counts included.

Needs a CUDA toolkit: set CUDAToolkit_ROOT (CMake reads it) if CMake does not find one.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import helpers
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

ROOT = Path(__file__).parents[1]
CELLS = 4096
DEFAULT_TRIES = 4000
FORCED_TRIES = 2  # a small budget: most cells are unfinished, and the recovery recomputes them


@pytest.fixture(scope="module")
def test_export(tmp_path_factory, gpu):
    """Build the C++ caller and return its test program."""
    assert gpu is not None
    if shutil.which("cmake") is None:
        pytest.skip("cmake is needed to build the C++ caller")
    build = tmp_path_factory.mktemp("cpp_build")
    configured = subprocess.run(
        ["cmake", "-S", ROOT / "cpp", "-B", build, f"-DPython3_EXECUTABLE={sys.executable}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if configured.returncode != 0 and "CUDAToolkit" in configured.stderr:
        pytest.skip("no CUDA toolkit: set CUDAToolkit_ROOT")
    assert configured.returncode == 0, configured.stderr
    subprocess.run(["make", "-C", build, "-j8"], check=True, capture_output=True)
    return build / "test_export"


@pytest.mark.parametrize("network", ["NCHO_photo_network.txt", "SNCHO_photo_network_2025.txt"])
@pytest.mark.parametrize(
    ("photo", "tries"),
    [(1, DEFAULT_TRIES), (0, DEFAULT_TRIES), (1, FORCED_TRIES)],
    ids=["photolysis", "thermal", "recovery"],
)
def test_cpp_call_is_the_python_call(test_export, tmp_path, network, photo, tries):
    helpers.thermo_folder()
    case = [sys.executable, ROOT / "tests" / "cpp_export_case.py", tmp_path, CELLS, photo, tries, network]
    subprocess.run([str(q) for q in case], check=True, capture_output=True)
    result = subprocess.run(
        [test_export, tmp_path / "export", tmp_path / "case"], check=False, capture_output=True, text=True
    )
    print(result.stdout, flush=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "bitwise identical" in result.stdout
