"""Write a solver's kernels and tables for a caller without Python (cpp/maxichem.hpp)."""

import json
from pathlib import Path
from typing import Any

from ._constants import (
    ATOL,
    COUNTS,
    EXPORT_FORMAT,
    KERNEL_ARGUMENTS,
    PHOTOLYSIS_ARGUMENT,
    REC_STEPS,
    SOLVER_KERNEL_NAMES,
    T_MIN,
)
from ._cuda import SOLVER_KERNELS, compile_cubin
from ._kernel_header import call_header, kernel_sources


def write_export(solver: Any, folder: str | Path) -> None:
    """Write what a caller without Python needs to run a solver's kernels (cpp/maxichem.hpp loads it).

    The folder gets the kernels' source (source/), their cubin for this device (kernel.cubin), every table the kernels
    read (tables/<name>.bin, raw little-endian), the names and launch sizes (meta.json, and meta.txt for C++). A call
    through the export gives the same bits. A cubin is for one GPU architecture: export on the machine that runs it.

    Parameters
    ----------
    solver : CudaSolver
        The solver.
    folder : str or pathlib.Path
        Where to write; made if missing.
    """
    tab, wsh, arch = solver._build
    folder = Path(folder)
    (folder / "tables").mkdir(parents=True, exist_ok=True)
    sources = kernel_sources(call_header(tab, solver.cpb, wsh, solver.drift))
    (folder / "source").mkdir(exist_ok=True)
    for name, text in sources.items():
        (folder / "source" / name).write_text(text)
    (folder / "kernel.cubin").write_bytes(compile_cubin(SOLVER_KERNELS, sources, arch))
    meta = {
        "export_format": EXPORT_FORMAT,
        "arch": arch,
        "ns": solver.ns,
        "nr": solver.nr,
        "nj": solver.nj,
        "w": solver.w,
        "cpb": solver.cpb,
        "smem": solver.smem,
        "smem_rec": solver.smem_rec,
        "stride": solver.stride,
        "stride_rec": solver.stride_rec,
        "sround": solver._rounds["sround"].value,
        "jround": solver._rounds["jround"].value,
        "rec_steps": REC_STEPS,
        "atol": ATOL,
        "t_min": T_MIN,
        "drift": solver.drift,
        "counts": list(COUNTS),
        "species": solver._names[0],
        "elements": solver._names[1],
        "tables": {},
        "arguments": {
            kernel: [*KERNEL_ARGUMENTS[kernel], *([PHOTOLYSIS_ARGUMENT] if solver.nj else [])]
            for kernel in SOLVER_KERNEL_NAMES
        },
    }
    for name, t in solver._tables.items():
        a = t.cpu().contiguous()
        a.numpy().tofile(folder / "tables" / f"{name}.bin")
        meta["tables"][name] = {"dtype": str(a.dtype).replace("torch.", ""), "shape": list(a.shape)}
    (folder / "meta.json").write_text(json.dumps(meta, indent=1))
    with (folder / "meta.txt").open("w") as f:  # the same, one "key value..." per line, for C++
        for k, v in meta.items():
            if k == "tables":
                for name, d in v.items():
                    f.write(f"table {name} {d['dtype']} {' '.join(str(n) for n in d['shape'])}\n")
            elif k == "arguments":
                for kernel, names in v.items():
                    f.write(f"args {kernel} {' '.join(names)}\n")
            elif k in ("counts", "species", "elements"):
                f.write(k + " " + " ".join(v) + "\n")
            else:
                f.write(f"{k} {v!r}\n" if isinstance(v, float) else f"{k} {v}\n")
