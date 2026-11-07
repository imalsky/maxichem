"""Write an export of a network's solver (default NCHO_photo) and one call's inputs and results, for cpp/test_export.

    VULCAN_THERMO=<VULCAN-JAX's src/vulcan3/thermo> python tests/cpp_export_case.py <folder> [cells] [photo] [max_steps]
        [network file, e.g. SNCHO_photo_network_2025.txt]

photo 0 exports the thermal network (no J); a small max_steps (e.g. 2) leaves most cells unfinished, so the guard
flags them and the recovery kernel recomputes them, and the counts are compared too.

<folder>/export: CudaSolver.export; <folder>/case: x, T, P, dt, h, J of a GCM-like call (cells over 800-2000 K and
1e-4 to 10 bar, H2S = 1.41e-5 x H2 on a sulfur network, relaxed to equilibrium, then 50 s with J = 1e-4 s^-1 above
0.03 bar) and the Python call's
x, h, accepted steps and done, as raw little-endian files. Then: cpp/build/test_export <folder>/export <folder>/case
"""

import sys
from pathlib import Path
from typing import Any

import torch

from maxichem import CudaSolver, VulcanNetwork, paths
from maxichem._checks import H2S_PER_H2_SOLAR, SOLAR_START


def main() -> None:
    """Write the export and the case."""
    out = Path(sys.argv[1])
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    photo = bool(int(sys.argv[3])) if len(sys.argv) > 3 else True
    max_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 4000
    network = sys.argv[5] if len(sys.argv) > 5 else "NCHO_photo_network.txt"
    th = paths.vulcan_thermo()
    net = VulcanNetwork(f"{th}/{network}", f"{th}/NASA9")
    sp = list(net.species)
    dev = torch.device("cuda")
    f64 = {"dtype": torch.float64, "device": dev}
    T = torch.linspace(800.0, 2000.0, B, **f64)
    P = 1e6 * torch.logspace(-4.0, 1.0, B, **f64)
    x = torch.zeros(B, len(sp), **f64)
    for s, v in SOLAR_START.items():
        x[:, sp.index(s)] = v
    if "H2S" in sp:
        x[:, sp.index("H2S")] = H2S_PER_H2_SOLAR * x[:, sp.index("H2")]
    x /= x.sum(1, keepdim=True)
    solver = CudaSolver(net, dev, photo=photo)
    J: Any = torch.zeros(B, len(net.photo), **f64) if photo else None
    x, h, _, _ = solver(x, T, P, torch.full_like(T, 1e10), torch.full_like(T, 1e-10), J=J)
    if photo:
        J[P < 3e4] = 1e-4
    dt = torch.full_like(T, 50.0)
    xo, ho, do, so = solver(x, T, P, dt, h, J=J, max_steps=max_steps)
    counts = solver.count.clone()
    torch.cuda.synchronize()
    solver.export(out / "export")
    case = out / "case"
    case.mkdir(parents=True, exist_ok=True)
    for name, t in (
        ("x", x),
        ("T", T),
        ("P", P),
        ("dt", dt),
        ("h", h),
        ("J", J),
        ("x_out", xo),
        ("h_out", ho),
        ("steps_out", so),
    ):
        if t is None:
            continue
        t.cpu().contiguous().numpy().tofile(case / f"{name}.bin")
    do.cpu().to(torch.int8).numpy().tofile(case / "done_out.bin")
    counts.cpu().numpy().tofile(case / "counts_out.bin")
    (case / "max_steps.txt").write_text(str(max_steps))
    (case / "B.txt").write_text(str(B))
    print(
        f"export and case written to {out}: {network}, {B} cells, photo {photo}, max_steps {max_steps}, steps per call "
        f"{float(so.mean()):.2f}, counts {solver.fallback}"
    )


if __name__ == "__main__":
    main()
