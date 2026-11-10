"""Stored outputs of the torch reference solver for the checks (paths.libref()).

The call computed by batched fp64 Rodas3 in torch (rate coefficients, exact Jacobian, LU without row exchanges, the call
kernel's step control), written once on exactly the checks' inputs. Every entry carries the sha256 of the inputs it was
computed from (the network's reaction list and NASA-9 files, the inputs, max_steps); a check computes the same digest
and refuses a mismatch. A file made with another rate rule than RATE_RULE is refused when it is opened.
"""

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from maxichem import paths
from maxichem._checks import RATE_RULE  # the input digests cannot see a change of the kernels' rates

_D: dict[Path, Any] = {}  # the open reference files: a cache


def digest(*items):
    """Return the sha256 over arrays or tensors (dtype, shape and bytes), strings and bytes, in order."""
    h = hashlib.sha256()
    for item in items:
        raw = item.encode() if isinstance(item, str) else item
        if isinstance(raw, bytes):
            h.update(b"b%d:" % len(raw) + raw)
            continue
        a = np.ascontiguousarray(raw.detach().cpu().numpy() if hasattr(raw, "detach") else np.asarray(raw))
        h.update(f"a{a.dtype.str}{a.shape}:".encode() + a.tobytes())
    return h.hexdigest()


def netsig(network_file, nasa_dir, species):
    """Return the digest of a network: its reaction list file, the species order and each species' NASA-9 file."""
    return digest(
        Path(network_file).read_bytes(), "|".join(species), *(Path(f"{nasa_dir}/{q}.txt").read_bytes() for q in species)
    )


def load(path=None):
    """Return the stored arrays (an open npz, loaded once per path)."""
    path = Path(path or paths.libref())
    if path not in _D:
        paths.need(path)
        stored = np.load(path)
        rule = str(stored["rate_rule"]) if "rate_rule" in stored.files else "none recorded"
        if rule != RATE_RULE:
            raise ValueError(
                f"{path} was made with the rate rule '{rule}', not '{RATE_RULE}': make a new reference with "
                "the torch generator (reference/make_libref.py) and point maxichem/paths.py at it"
            )
        _D[path] = stored
    return _D[path]


def arrays(key, path=None):
    """Return the arrays stored directly under key: {name: array}."""
    stored = load(path)
    return {
        k[len(key) + 1 :]: stored[k]
        for k in stored.files
        if k.startswith(key + "/") and k.count("/") == key.count("/") + 1
    }


def entry(key, *inputs, path=None):
    """Return arrays(key) after checking the digest of the inputs against key/sha."""
    stored = load(path)
    if f"{key}/sha" not in stored.files:
        raise KeyError(f"{key}: not in {path or paths.libref()}")
    if str(stored[f"{key}/sha"]) != digest(*inputs):
        raise ValueError(f"{key}: the inputs differ from those the stored reference was written for")
    return arrays(key, path)
