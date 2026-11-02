"""Quick start: VULCAN's C-H-N-O network on the GPU, first the all-species call, then the reduced-tracer call (design
G31, 31 advected tracers) on the same cells.

    VULCAN_THERMO=<folder with NCHO_thermo_network.txt and NASA9/> python examples/quickstart.py
"""
import torch

from maxichem import CudaSolver, TracerSolver, VulcanNetwork, paths

th = paths.VULCAN_THERMO
net = VulcanNetwork(f"{th}/NCHO_thermo_network.txt", f"{th}/NASA9")       # 53 species, 290 reversible reactions
sp = list(net.species)
dev = torch.device("cuda")
B = 512
f64 = dict(dtype=torch.float64, device=dev)
full = lambda v: torch.full((B,), v, **f64)                                # noqa: E731

# Cells over a hot-Jupiter-like range: T 800-2000 K, P 1e-4 to 10 bar (P in dyn cm^-2: 1 bar = 1e6).
T = torch.linspace(800.0, 2000.0, B, **f64)
P = 1e6 * torch.logspace(-4.0, 1.0, B, **f64)

# A roughly solar starting gas (mole fractions, every other species zero), relaxed to equilibrium by one long call.
x = torch.zeros(B, len(sp), **f64)
for s, v in {"H2": 0.84, "He": 0.16, "H2O": 9e-4, "CO": 5e-4, "CH4": 1e-5, "N2": 6e-5, "NH3": 1e-6}.items():
    x[:, sp.index(s)] = v
x /= x.sum(1, keepdim=True)
solver = CudaSolver(net, dev)
x, h_eq, done, steps = solver(x, T, P, full(1e10), full(1e-10))
print(f"equilibrium start: {int(done.sum())} of {B} calls finished, {int(steps.max())} steps at most")

# GCM-like calls: dt 50 s, T changing from call to call, each cell's step size h carried to its next call.
x_eq, x_full, h = x.clone(), x, h_eq.clone()
for k in range(5):
    Tk = T * (1.0 + 0.02 * torch.sin(torch.arange(B, **f64) + k))
    x_full, h, done, steps = solver(x_full, Tk, P, full(50.0), h)
counts = solver.check()                                                    # raises if a cell was below 300 K
atoms = torch.as_tensor(net.atoms, **f64)
share = lambda x: (x @ atoms) / (x @ atoms).sum(1, keepdim=True)          # noqa: E731
print(f"all-species call: 5 calls, guard counts {counts}, largest change of an element share "
      f"{float(((share(x_full) - share(x_eq)).abs() / share(x_eq).clamp_min(1e-300)).max()):.1e}")

# The reduced-tracer call: the GCM advects q (27 species and the C, H, N, O atoms of the other 25, as mass mixing
# ratios); each cell keeps mem (the 25 hidden species) and h between calls, also in restart files.
ts = TracerSolver(net, "G31", dev)
q, mem = ts.split(x_eq)                                                   # a new run would start with h = 1e-10
h = h_eq.clone()
for k in range(5):
    Tk = T * (1.0 + 0.02 * torch.sin(torch.arange(B, **f64) + k))
    # (the GCM's transport would mix q between cells here; mem stays with its cell)
    q, mem, h, mu, cp, Q, flag, counts = ts(q, mem, Tk, P, full(50.0), h, heating=True)
print(f"reduced call: {q.shape[1]} tracers, flags {torch.bincount(flag).tolist()}, mean molar mass "
      f"{float(mu.min()):.3f}-{float(mu.max()):.3f} g/mol, cp {float(cp.min()):.3e}-{float(cp.max()):.3e} erg/g/K")
q_full, _ = ts.split(x_full)
K = len(ts.carried)
floor = 1e-30
dex = (torch.log10(q[:, :K].clamp_min(floor)) - torch.log10(q_full[:, :K].clamp_min(floor))).abs()
print(f"carried species, reduced call against the all-species call: largest difference {float(dex.max()):.1e} dex")
print("counts since construction:", ts.check())
