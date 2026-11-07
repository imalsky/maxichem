#!/usr/bin/env python3
"""Write the data header for the test network, so the kernel files compile and lint with no solver and no GPU.

Run from the repository root: ``PYTHONPATH=. python tools/write_kernel_header.py``. It writes
``build/kernels/network.h``; ``tools/nvrtc_compile.py`` and ``compile_flags.txt`` look for it there.
"""

from __future__ import annotations

from pathlib import Path

from maxichem import VulcanNetwork, _ad_tables, _constants, _kernel_header, _tables

DATA = Path("tests/data")
OUTPUT = Path("build/kernels/network.h")
WARPS_PER_CELL = 2
WARPS_PER_BLOCK = 2


def main() -> None:
    """Write the header."""
    net = VulcanNetwork(DATA / "synthetic_network.txt", DATA / "NASA9")
    tab = _tables.tables(net, WARPS_PER_CELL, photo=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        _kernel_header.derivative_header(
            tab, _ad_tables.ad_tables(net, tab), "jvp", WARPS_PER_BLOCK, _constants.CAP, _constants.DRIFT
        )
    )


if __name__ == "__main__":
    main()
