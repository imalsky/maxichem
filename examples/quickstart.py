"""Quick start: VULCAN-JAX's C-H-N-O network with photolysis on the GPU.

    VULCAN_THERMO=<VULCAN-JAX's src/vulcan3/thermo folder> python examples/quickstart.py
"""
import torch

from maxichem import CudaSolver, VulcanNetwork, paths

th = paths.VULCAN_THERMO
net = VulcanNetwork(f"{th}/NCHO_photo_network.txt", f"{th}/NASA9")    # 69 species, 388 reactions, 48 photolysis branches
sp = list(net.species)
dev = torch.device("cuda")
B = 512
f64 = dict(dtype=torch.float64, device=dev)
full = lambda v: torch.full((B,), v, **f64)                            # noqa: E731

# Cells over a hot-Jupiter-like range: T 800-2000 K, P 1e-4 to 10 bar (P in dyn cm^-2: 1 bar = 1e6).
T = torch.linspace(800.0, 2000.0, B, **f64)
P = 1e6 * torch.logspace(-4.0, 1.0, B, **f64)

# A roughly solar starting gas (mole fractions, every other species zero), relaxed to equilibrium by one long dark call.
x = torch.zeros(B, len(sp), **f64)
for s, v in {"H2": 0.84, "He": 0.16, "H2O": 9e-4, "CO": 5e-4, "CH4": 1e-5, "N2": 6e-5, "NH3": 1e-6}.items():
    x[:, sp.index(s)] = v
x /= x.sum(1, keepdim=True)
solver = CudaSolver(net, dev, photo=True)
J = torch.zeros(B, len(net.photo), **f64)                             # photolysis rates (s^-1), one column per branch
x, h, done, steps = solver(x, T, P, full(1e10), full(1e-10), J=J)
print(f"equilibrium start: {int(done.sum())} of {B} calls finished, {int(steps.max())} steps at most")

# Day side: photolysis rates from the GCM's radiative transfer would go here. As a stand-in, every branch gets 1e-4 s^-1
# in the upper half of the cells (P below 0.03 bar). Then GCM-like calls: dt 50 s, T changing from call to call, each
# cell's step size h carried to its next call.
J[P < 3e4] = 1e-4
x0 = x.clone()
for k in range(5):
    Tk = T * (1.0 + 0.02 * torch.sin(torch.arange(B, **f64) + k))
    x, h, done, steps = solver(x, Tk, P, full(50.0), h, J=J)
counts = solver.check()                                                # raises if a cell was below 300 K
atoms = torch.as_tensor(net.atoms, **f64)
share = lambda x: (x @ atoms) / (x @ atoms).sum(1, keepdim=True)      # noqa: E731
print(f"5 calls: guard counts {counts}, largest change of an element share "
      f"{float(((share(x) - share(x0)).abs() / share(x0).clamp_min(1e-300)).max()):.1e}")
i = sp.index("CH4")
print(f"CH4 at 1e-4 bar: {float(x0[0, i]):.3e} (dark start) -> {float(x[0, i]):.3e} after 250 s of light")
