"""Write an export of the NCHO_photo solver and one call's inputs and results, for cpp/test_export.

    VULCAN_THERMO=<VULCAN-JAX's src/vulcan3/thermo> python tests/cpp_export_case.py <folder> [cells] [photo] [max_steps]

photo 0 exports the thermal network (no J); a small max_steps (e.g. 2) leaves most cells unfinished, so the guard
flags them and the recovery kernel recomputes them, and the counts are compared too.

<folder>/export: CudaSolver.export; <folder>/case: x, T, P, dt, h, J of a GCM-like call (cells over 800-2000 K and
1e-4 to 10 bar, relaxed to equilibrium, then 50 s with J = 1e-4 s^-1 above 0.03 bar) and the Python call's
x, h, accepted steps and done, as raw little-endian files. Then: cpp/build/test_export <folder>/export <folder>/case
"""
import os
import sys

import torch

from maxichem import CudaSolver, VulcanNetwork, paths

out = sys.argv[1]
B = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
photo = bool(int(sys.argv[3])) if len(sys.argv) > 3 else True
max_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 4000
th = paths.VULCAN_THERMO
net = VulcanNetwork(f"{th}/NCHO_photo_network.txt", f"{th}/NASA9")
sp = list(net.species)
dev = torch.device("cuda")
f64 = dict(dtype=torch.float64, device=dev)
T = torch.linspace(800.0, 2000.0, B, **f64)
P = 1e6 * torch.logspace(-4.0, 1.0, B, **f64)
x = torch.zeros(B, len(sp), **f64)
for s, v in {"H2": 0.84, "He": 0.16, "H2O": 9e-4, "CO": 5e-4, "CH4": 1e-5, "N2": 6e-5, "NH3": 1e-6}.items():
    x[:, sp.index(s)] = v
x /= x.sum(1, keepdim=True)
solver = CudaSolver(net, dev, photo=photo)
J = torch.zeros(B, len(net.photo), **f64) if photo else None
x, h, done, steps = solver(x, T, P, torch.full_like(T, 1e10), torch.full_like(T, 1e-10), J=J)
if photo:
    J[P < 3e4] = 1e-4
dt = torch.full_like(T, 50.0)
xo, ho, do, so = solver(x, T, P, dt, h, J=J, max_steps=max_steps)
counts = solver.count.clone()
torch.cuda.synchronize()
solver.export(os.path.join(out, "export"))
case = os.path.join(out, "case")
os.makedirs(case, exist_ok=True)
for name, t in (("x", x), ("T", T), ("P", P), ("dt", dt), ("h", h), ("J", J), ("x_out", xo), ("h_out", ho),
                ("steps_out", so)):
    if t is None:
        continue
    t.cpu().contiguous().numpy().tofile(os.path.join(case, name + ".bin"))
do.cpu().to(torch.int8).numpy().tofile(os.path.join(case, "done_out.bin"))
counts.cpu().numpy().tofile(os.path.join(case, "counts_out.bin"))
with open(os.path.join(case, "max_steps.txt"), "w") as f:
    f.write(str(max_steps))
with open(os.path.join(case, "B.txt"), "w") as f:
    f.write(str(B))
print(f"export and case written to {out}: {B} cells, photo {photo}, max_steps {max_steps}, steps per call "
      f"{float(so.mean()):.2f}, counts {solver.fallback}")
