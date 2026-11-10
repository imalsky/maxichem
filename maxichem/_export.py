"""Write a solver's kernels and tables for a caller without Python (cpp/maxichem.hpp)."""

from pathlib import Path
from typing import TYPE_CHECKING

from ._constants import (
    ATOL,
    COUNTS,
    EXPORT_FORMAT,
    KERNEL_ARGUMENTS,
    PHOTOLYSIS_ARGUMENT,
    PHOTOLYSIS_KERNEL_NAMES,
    REC_STEPS,
    SOLVER_KERNEL_NAMES,
    T_MAX,
    T_MIN,
    X_SUM_TOL,
)
from ._cuda import SOLVER_KERNELS, compile_cubin
from ._kernel_header import call_header, kernel_sources

if TYPE_CHECKING:
    from .kincuda import CudaSolver


def write_export(solver: "CudaSolver", folder: str | Path) -> None:
    """Write what a caller without Python needs to run a solver's kernels (cpp/maxichem.hpp loads it); makes the folder.

    The folder gets the kernels' source (source/), their cubin for this device (kernel.cubin), every table the kernels
    read (tables/<name>.bin, raw little-endian), the names and launch sizes (meta.txt). A call through the export gives
    the same bits. A cubin is for one GPU architecture: export on the machine that runs it.
    """
    tab, wsh, arch = solver._build
    folder = Path(folder)
    (folder / "tables").mkdir(parents=True, exist_ok=True)
    sources = kernel_sources(call_header(tab, solver.cpb, wsh, solver.drift))
    (folder / "source").mkdir(exist_ok=True)
    for name, text in sources.items():
        (folder / "source" / name).write_text(text)
    (folder / "kernel.cubin").write_bytes(compile_cubin(SOLVER_KERNELS, sources, arch))
    lines = [
        f"export_format {EXPORT_FORMAT}",
        f"arch {arch}",
        f"ns {solver.ns}",
        f"nj {solver.nj}",
        f"w {solver.w}",
        f"cpb {solver.cpb}",
        f"smem {solver.smem}",
        f"smem_rec {solver.smem_rec}",
        f"stride {solver.stride}",
        f"stride_rec {solver.stride_rec}",
        f"sround {solver._rounds['sround'].value}",
        f"jround {solver._rounds['jround'].value}",
        f"rec_steps {REC_STEPS}",
        f"atol {ATOL!r}",
        f"t_min {T_MIN!r}",
        f"t_max {T_MAX!r}",
        f"x_sum_tol {X_SUM_TOL!r}",
        f"counts {' '.join(COUNTS)}",
        f"species {' '.join(solver._names[0])}",
        f"elements {' '.join(solver._names[1])}",
    ]
    for name, t in solver._tables.items():
        a = t.cpu().contiguous()
        a.numpy().tofile(folder / "tables" / f"{name}.bin")
        dtype = str(a.dtype).replace("torch.", "")
        lines.append(f"table {name} {dtype} {' '.join(str(n) for n in a.shape)}")
    for kernel in SOLVER_KERNEL_NAMES:
        photolysis = solver.nj and kernel in PHOTOLYSIS_KERNEL_NAMES
        names = [*KERNEL_ARGUMENTS[kernel], *([PHOTOLYSIS_ARGUMENT] if photolysis else [])]
        lines.append(f"args {kernel} {' '.join(names)}")
    (folder / "meta.txt").write_text("\n".join(lines) + "\n")
