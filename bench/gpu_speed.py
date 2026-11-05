"""Cost of one CudaSolver call on a GCM-like batch, for comparing GPUs (the call is fp64 throughout).

    VULCAN_THERMO=<VULCAN3>/src/vulcan3/thermo python bench/gpu_speed.py [--cells 1382400]
        [--network NCHO_photo_network.txt | SNCHO_photo_network_2025.txt] [--calls 10]

The batch: cells spread over 800-2000 K and 1e-4 to 10 bar (log-spaced), a solar-like gas (H2S = 1.41e-5 x H2 for a
sulfur network) relaxed by one 1e10 s call. Then --calls calls of 70 s each, every call starting from the previous
call's output (about one solver step per call, as in a GCM), first dark, then with J = 1e-4 s^-1 on every branch above
0.03 bar. A first call of each kind is not timed. Prints one JSON line: the GPU, the batch, and per kind the median
us per cell-call, the mean solver steps per call and the guard counts.
"""
import argparse
import json
import statistics
import time

import torch

from maxichem import CudaSolver, VulcanNetwork, paths

ap = argparse.ArgumentParser()
ap.add_argument("--cells", type=int, default=1382400)
ap.add_argument("--network", default="NCHO_photo_network.txt")
ap.add_argument("--calls", type=int, default=10)
a = ap.parse_args()

th = paths.VULCAN_THERMO
net = VulcanNetwork(f"{th}/{a.network}", f"{th}/NASA9")
sp = list(net.species)
dev = torch.device("cuda")
f64 = dict(dtype=torch.float64, device=dev)
B = a.cells
T = torch.linspace(800.0, 2000.0, B, **f64)
P = 1e6 * torch.logspace(-4.0, 1.0, B, **f64)          # dyn cm^-2
x = torch.zeros(B, len(sp), **f64)
for s, v in {"H2": 0.84, "He": 0.16, "H2O": 9e-4, "CO": 5e-4, "CH4": 1e-5, "N2": 6e-5, "NH3": 1e-6}.items():
    if s in sp:
        x[:, sp.index(s)] = v
if "H2S" in sp:
    x[:, sp.index("H2S")] = 1.41e-5 * x[:, sp.index("H2")]
x /= x.sum(1, keepdim=True)
photo = len(net.photo) > 0
solver = CudaSolver(net, dev, photo=photo)
J0 = torch.zeros(B, len(net.photo), **f64) if photo else None
x, h, done, _ = solver(x, T, P, torch.full_like(T, 1e10), torch.full_like(T, 1e-10), J=J0)
torch.cuda.synchronize()
res = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, network=a.network, species=len(sp),
           cells=B, relaxed_unfinished=int((~done.bool()).sum()))
kinds = [("dark", J0)]
if photo:
    J1 = J0.clone()
    J1[P < 3e4] = 1e-4
    kinds.append(("lit", J1))
for kind, J in kinds:
    xc, hc, dt = x.clone(), h.clone(), torch.full_like(T, 70.0)
    times, steps = [], []
    for c in range(a.calls + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        xc, hc, d, nok = solver(xc, T, P, dt, hc, J=J)
        torch.cuda.synchronize()
        if c:
            times.append(time.perf_counter() - t0)
            steps.append(float(nok.mean()))
    res[kind] = dict(us_per_cell_call=1e6 * statistics.median(times) / B, steps_per_call=statistics.mean(steps),
                     unfinished=int((~d.bool()).sum()), counts=solver.fallback)
print(json.dumps(res))
