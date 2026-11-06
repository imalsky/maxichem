#!/usr/bin/env python3
"""Write the data header for the test network, so the kernel files compile and lint with no solver and no GPU.

Public name: ``main``. Run from the repository root: ``PYTHONPATH=. python tools/write_kernel_header.py``. It writes
``build/kernels/network.h``; ``tools/nvrtc_compile.py`` and ``compile_flags.txt`` look for it there.
"""

from __future__ import annotations

from pathlib import Path

from maxichem._ad_tables import ad_tables
from maxichem._kernel_header import derivative_header
from maxichem._tables import tables
from maxichem.kinvulcan import VulcanNetwork

DATA = Path("tests/data")
OUTPUT = Path("build/kernels/network.h")
WARPS_PER_CELL = 2
WARPS_PER_BLOCK = 2


def main() -> None:
    """Write the header."""
    net = VulcanNetwork(DATA / "synthetic_network.txt", DATA / "NASA9")
    tab = tables(net, WARPS_PER_CELL, photo=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(derivative_header(tab, ad_tables(net, tab), "jvp", WARPS_PER_BLOCK))


if __name__ == "__main__":
    main()
