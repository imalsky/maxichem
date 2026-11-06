#!/usr/bin/env python3
"""Compile CUDA kernels with NVRTC to check that they build. Needs no GPU and no CUDA driver.

Public name: ``main``. Run with ``uv run --no-project --with nvidia-cuda-nvrtc-cu12 python tools/nvrtc_compile.py``.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

import nvidia.cuda_nvrtc

DEFAULT_ARCH = "89"  # sm_89: the RTX 6000 Ada, the card the kernels run on
NVRTC_LIBRARY = "libnvrtc.so.12"  # the library inside the nvidia-cuda-nvrtc-cu12 wheel


def compile_kernel(nvrtc: ctypes.CDLL, path: Path, options: list[str]) -> str:
    """Compile one kernel and return an empty string, or the compiler's log on a failure.

    Parameters
    ----------
    nvrtc : ctypes.CDLL
        The loaded NVRTC library.
    path : pathlib.Path
        The kernel source.
    options : list of str
        Options passed to the compiler.

    Returns
    -------
    str
        Empty when the kernel compiled, else the compiler's log.
    """
    program = ctypes.c_void_p()
    source = path.read_text().encode()
    if nvrtc.nvrtcCreateProgram(ctypes.byref(program), source, str(path).encode(), 0, None, None) != 0:
        return "NVRTC could not create the program"
    encoded = [option.encode() for option in options]
    result = nvrtc.nvrtcCompileProgram(program, len(encoded), (ctypes.c_char_p * len(encoded))(*encoded))
    size = ctypes.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(size))
    log = ctypes.create_string_buffer(size.value)
    nvrtc.nvrtcGetProgramLog(program, log)
    nvrtc.nvrtcDestroyProgram(ctypes.byref(program))
    return "" if result == 0 else log.value.decode() or f"NVRTC returned {result} with no log"


def main() -> int:
    """Compile each kernel named on the command line.

    Returns
    -------
    int
        0 when all compiled, else 1.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernels", nargs="+", type=Path)
    parser.add_argument("--arch", default=DEFAULT_ARCH, help="sm_ architecture, as digits")
    parser.add_argument("-I", dest="includes", action="append", default=[], help="include folder")
    args = parser.parse_args()
    library = Path(next(iter(nvidia.cuda_nvrtc.__path__))) / "lib" / NVRTC_LIBRARY
    nvrtc = ctypes.CDLL(str(library))
    options = [f"-arch=sm_{args.arch}", "-std=c++17", *[f"-I{folder}" for folder in args.includes]]
    failures = 0
    for path in args.kernels:
        log = compile_kernel(nvrtc, path, options)
        print(f"{path}: {'FAILED' if log else 'ok'}")
        if log:
            print(log)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
