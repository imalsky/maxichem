"""The solver against VULCAN-0D truth: no misses, and no path's worst error moved from a saved run."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from maxichem import paths

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

WORST_ERROR_CHANGE = 0.01  # dex: against the saved run, per path and per box


@pytest.mark.usefixtures("gpu")
def test_quick_tier_has_no_miss(tmp_path):
    paths.need(paths.vulcan_thermo())
    reference = paths.data() / "benches/kintest/quick_ref6.npz"
    paths.need(reference, paths.boxes(), paths.qual(), paths.designs())
    saved = tmp_path / "quick.npz"
    script = Path(__file__).parents[1] / "bench" / "kintest.py"
    command = [sys.executable, script, "quick", "--ref", reference, "--save", saved]
    result = subprocess.run([str(q) for q in command], check=False, capture_output=True, text=True)
    print(result.stdout, flush=True)
    assert result.returncode == 0, result.stdout + result.stderr
    new, old = np.load(saved), np.load(reference)
    assert json.loads(str(new["run"]))["commit"]
    assert np.abs(new["worst"] - old["worst"]).max() <= WORST_ERROR_CHANGE
    assert np.abs(new["b_worst"] - old["b_worst"]).max() <= WORST_ERROR_CHANGE
