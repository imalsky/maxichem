"""The solver against VULCAN-0D truth on both default networks: no unpinned miss, no path's worst error moved."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from maxichem import paths

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

WORST_ERROR_CHANGE = 0.01  # dex: against the saved run, per path and per box
REFERENCES = {  # bench/kintest.py quick, saved
    "NCHO_photo_network.txt": "benches/kintest/quick_ref6.npz",
    "SNCHO_photo_network_2025.txt": "benches/kintest/quick_sncho25_ref1.npz",
}


@pytest.mark.usefixtures("gpu")
@pytest.mark.parametrize("network", list(REFERENCES))
def test_quick_tier_has_no_miss(tmp_path, network):
    paths.need(paths.vulcan_thermo())
    reference = paths.data() / REFERENCES[network]
    paths.need(reference, paths.designs())
    saved = tmp_path / "quick.npz"
    script = Path(__file__).parents[1] / "bench" / "kintest.py"
    command = [sys.executable, script, "quick", "--network", network, "--ref", reference, "--save", saved]
    result = subprocess.run([str(q) for q in command], check=False, capture_output=True, text=True)
    print(result.stdout, flush=True)
    assert result.returncode == 0, result.stdout + result.stderr
    new, old = np.load(saved), np.load(reference)
    assert json.loads(str(new["run"]))["commit"]
    assert np.abs(new["worst"] - old["worst"]).max() <= WORST_ERROR_CHANGE
    assert np.abs(new["b_worst"] - old["b_worst"]).max() <= WORST_ERROR_CHANGE
