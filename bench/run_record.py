"""What a reported bench run saves beside its output: the code, the settings and the network data it ran with."""

import argparse
import hashlib
import importlib.metadata
import subprocess
from pathlib import Path

import torch

from maxichem import _constants

_REPOSITORY = Path(__file__).parents[1]


def _git(*arguments: str) -> str:
    """Return the output of a git command in this repository."""
    return subprocess.run(
        ["git", *arguments], cwd=_REPOSITORY, check=True, capture_output=True, text=True
    ).stdout.strip()


def run_record(
    network_file: Path, nasa_dir: Path, species: list[str], drift: float, arguments: argparse.Namespace
) -> dict:
    """Return the commit, the solver's settings, the versions and a digest of the network's files, as JSON values."""
    data = hashlib.sha256(network_file.read_bytes())
    for name in species:
        data.update((nasa_dir / f"{name}.txt").read_bytes())
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "arguments": vars(arguments),
        "rtol": _constants.DEFAULT_RTOL,
        "atol": _constants.ATOL,
        "max_steps": _constants.DEFAULT_MAX_STEPS,
        "rec_steps": _constants.REC_STEPS,
        "drift": drift,
        "torch": torch.__version__,
        "nvrtc": importlib.metadata.version("nvidia-cuda-nvrtc-cu12"),
        "network": network_file.name,
        "network_sha256": data.hexdigest(),
    }
