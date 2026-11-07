"""The built wheel holds what the package needs at run time."""

import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
ASSEMBLE = """
import maxichem, sys
from maxichem import _kernel_header, _tables, kinvulcan
net = kinvulcan.VulcanNetwork(sys.argv[1], sys.argv[2])
assert "network.h" in _kernel_header.kernel_sources(_kernel_header.call_header(_tables.tables(net, 2), 1, True, 1e-6))
print(maxichem.__file__)
"""


@pytest.mark.slow
def test_wheel_assembles_kernels_without_the_checkout(tmp_path, nasa_dir):
    subprocess.run(["uv", "build", "--wheel", "--out-dir", tmp_path, ROOT], check=True, capture_output=True)
    wheel = next(tmp_path.glob("*.whl"))
    site = tmp_path / "site"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(site)
    result = subprocess.run(
        [sys.executable, "-c", ASSEMBLE, nasa_dir.parent / "synthetic_network.txt", nasa_dir],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(site)},
    )
    assert Path(result.stdout.strip()).is_relative_to(site)
