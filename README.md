# maxichem

maxichem is a GPU solver for gas-phase chemistry in exoplanet atmospheres, built to be called from a general
circulation model (GCM). Each call advances the composition of many independent cells by one time step dt at each
cell's temperature T and pressure P. The chemistry is VULCAN's C-H-N-O network (53 species, 290 reversible
reactions), solved as a physical 0-D kinetics problem in CUDA kernels generated for the network and batched over cells.
Every call can also be differentiated (tangent and adjoint), in PyTorch or in JAX.

- **Physics.** Mass-action kinetics with forward rates from VULCAN's reaction list and reverse rates from NASA-9
  thermodynamics, at constant T and P within a call: dx/dt = (S v - x sum(S v)) / n_tot, with n_tot = P / kT.
  Element conservation, positivity and the thermodynamic equilibrium hold by construction.
- **Integrator.** Rodas3 (order 3, L-stable) with the exact Jacobian, a sparse LU factorization without row exchanges in
  a minimum-degree order, adaptive steps (rtol 1e-3, atol 1e-16 in mole fraction; a secant step shrink after repeated
  rejections, which reuse the rejected try's Jacobian) and each cell's step size carried to its next call.
- **Robustness.** After each call a guard kernel flags cells with a non-finite state, an unfinished call or an element
  drift, and a recovery kernel redoes them on the GPU (dense LU with threshold pivoting, ten times the step budget). A
  call never waits for the host; the counts are read on demand.
- **Derivatives.** Tangent (JVP) and adjoint (VJP) kernels of one call with respect to the state, T, P, dt, the step
  size, per-cell rate multipliers and the Arrhenius table.
- **JAX.** The call and its derivatives as JAX primitives (jit, vmap, grad, jvp, scan), bitwise identical to the CUDA
  solvers.
- **Fewer tracers.** A reduced-tracer call for a GCM that advects fewer tracers than the network has species: a design
  names the carried species and the advected element budgets, each cell keeps the other species as a memory, and the
  full mechanism runs inside every call. Design **G31** (31 tracers) is the recommended one for C/O up to 1.

Any VULCAN-format network up to 254 species can be read (the sulfur network has been run as well); NCHO is the default
and the one the numbers below refer to.

## Requirements

- Linux with an NVIDIA GPU and its driver. The kernels run in double precision and their speed follows the GPU's fp64
  throughput. Only sm_89 (RTX 6000 Ada) has been tested.
- No CUDA toolkit is needed: the kernels are compiled at run time with NVRTC from the `nvidia-cuda-nvrtc-cu12` wheel.
- Python >= 3.10, NumPy and PyTorch with CUDA. Tested with Python 3.12, torch 2.8.0+cu126, NumPy 2.5.3.
- For the JAX layer: `jax[cuda12]` 0.12.0.1 (the layer uses JAX internals, so the version is pinned) and g++, which
  builds a small XLA FFI shim at first use.
- VULCAN's network and thermodynamic files, which are not bundled. The tested files are the `thermo/` folder of
  [VULCAN-0D](https://github.com/shami-EEG/VULCAN-0D) at commit `b3ccf4f` (`NCHO_thermo_network.txt` and `NASA9/`).
  The NCHO network in VULCAN's main repository is a different, larger network and has not been tested with this code.

## Install

    git clone https://github.com/imalsky/maxichem.git
    cd maxichem
    pip install -e .            # or: pip install -e ".[jax]"

Environment variables:

| Variable | Meaning |
| --- | --- |
| `VULCAN_THERMO` | folder with `NCHO_thermo_network.txt` and `NASA9/` (used by the example, checks and benches) |
| `CHEM_DATA` | data root of the checks and benches; also where the JAX shim is built (`caches/kinjax`). Default: `data/` in the checkout; set it to a writable folder for a non-editable install |
| `XLA_PYTHON_CLIENT_PREALLOCATE=false` | recommended for JAX on a shared GPU |

## Quick start

    VULCAN_THERMO=/path/to/VULCAN-0D/thermo python examples/quickstart.py

The example relaxes 512 cells (800-2000 K, 1e-4 to 10 bar) to equilibrium, runs five GCM-like calls with the
all-species solver, then the same calls with the reduced-tracer call (G31) and compares the two. In short:

```python
import torch
from maxichem import CudaSolver, TracerSolver, VulcanNetwork

th = "/path/to/VULCAN-0D/thermo"
net = VulcanNetwork(f"{th}/NCHO_thermo_network.txt", f"{th}/NASA9")   # 53 species, He last
solver = CudaSolver(net, torch.device("cuda"))
x, h, done, steps = solver(x, T, P, dt, h)        # float64 CUDA tensors, one row per cell
counts = solver.check()                           # guard counts; raises if a cell was below 300 K
```

## Using it in a GCM

- **All species.** One call per cell and GCM step, after the GCM's transport: `CudaSolver(net, device)(x, T, P, dt, h)`
  (`KinJax(net)` in JAX) takes mole fractions `x` [cells, species] in the network's species order (`net.species`), `T`
  in K, `P` in dyn cm^-2 (1 bar = 1e6), `dt` in s and the step size `h` in s that the cell's previous call returned
  (any positive value at the first call, e.g. 1e-10), all float64 CUDA tensors. It returns the new `x`, the step size to
  keep, a per-cell `done` flag and the accepted steps. T and P are fixed within a call; cells never couple.
- GCMs advect mass mixing ratios: convert to mole fractions before the call and back after it (the chemistry conserves
  elements and mass). The caller keeps `h` per cell between calls and in restart files.
- **Fewer tracers.** `TracerSolver(net, design, device)(q, mem, T, P, dt, h, *, photo=None, heating=False)` returns
  `(q, mem, h, mu, cp, Q, flag, counts)`. `q` [cells, K + M] are the advected tracers in mass mixing ratios (the
  design's carried species, then its rows: an element row is that element's mass per mass of gas in the other species);
  `mem` [cells, H] is the cell's memory of the species it does not advect (mass mixing ratios), kept per cell and in
  restart files, not advected. He is the mass remainder. `mu` is the mean molar mass (g/mol), `cp` the specific heat at
  constant pressure (erg g^-1 K^-1), `Q` the chemical heating rate (erg g^-1 s^-1, with `heating=True`), `flag` a
  per-cell status (bits 0-1 the chemistry guard, bits 2-3 the reconstruction, bit 4 a defect). A run starts from
  `q, mem = ts.split(x)` and `h = 1e-10`. Designs are YAML files in `maxichem/designs/` (G31, N30sN2H, G30b, N30, N26)
  or any file of the same form; a design is passed by name or path.
- Results do not depend on batch size or cell order (on one GPU type), so domain decomposition does not change them.
  Cells below 300 K are not stepped and are counted; `check()` raises for them.
- Not done here: radiation, advection, vertical mixing, condensation. The interface is Python only (no C or Fortran
  entry point yet).
- Photolysis is not included yet. It is meant to enter as per-cell rates from the GCM: one rate in s^-1 per photolysis
  line of the network file, fixed within the call like T and P; the module will not compute actinic fluxes. The `photo`
  argument is reserved and raises unless `None`.

## Accuracy and speed

Accuracy is judged against VULCAN-0D runs, misses first. A miss is a scored state where any of 12 gases (H2O, CO, CO2,
CH4, NH3, HCN, C2H2, H, OH, O, H2, N2) differs from VULCAN by more than 0.1 dex above a per-gas floor (1e-10 for the
first seven, 1e-8 for H, OH, O, 1e-6 for H2 and N2). The target domain is 1e-5 to 100 bar, 300 to 3000 K, metallicity
0.1 to 300 times solar, C/O 0.1 to 1, with time steps of 1 to 500 s.

- All species: no misses on the development's VULCAN-0D test sets, the two largest of which hold 13.5 million and
  6.26 million scored states (worst error 0.033 dex on the latter; measured with the earlier step rule), except one
  start of the 395,114-row yardstick: a 300 K gas far from equilibrium, CH4 0.67 dex off in its first call (17 rows).
- G31: no path with a miss on the main test domain (worst error 0.094 dex); in 1000-day runs (with the earlier step
  rule) it stays within 0.002 dex of the all-species call. Known weak spots: carbon-rich gas (C/O 1 to 2) that keeps
  mixing (up to 1.26 dex), and a few cells at 300 times solar metallicity at 0.33 dex in the first calls after an
  equilibrium start. The per-cell memory is required (without it the errors reach about 6 dex).
- Speed, on an RTX 6000 Ada (fp64 at 1/64 of its fp32 rate) with no other job. In a coupled GCM run (snapy, a C48 x 100
  hot Jupiter, 1.38 million cells, G31 every 70-s step) the whole reduced-tracer call takes 0.26 us per cell-call with
  its gather and scatter (projection 0.04, chemistry 0.21): the gas changes little between calls and a cell takes about
  one step per call.
  The all-species kernel takes about 0.24 us per cell-call at large batches when the gas is not mixed between calls
  (1.08 steps per call). On mixing test beds, where every cell's gas is mixed far from its last state before each call,
  the all-species call takes 2.2-2.3 us per cell-call (69,120 to 2.46 million cells, about 14 tries per cell and call).

## Tests and the data they need

| Check | What it verifies |
| --- | --- |
| `tests/kincuda_check.py <VULCAN_THERMO>` | the CUDA call against stored outputs of a torch reference solver (cold, scrambled and 2,000-call chained calls; guard and recovery; batch size and order) |
| `tests/kintracer_check.py` | the reduced-tracer call against its stored reference; spill path, outputs, batch size and order |
| `tests/kincuda_ad_check.py` | the derivative kernels against stored torch-autograd derivatives, central differences and each other |
| `tests/kinjax_check.py` | the JAX layer, bitwise against the CUDA solvers |
| `bench/kintest.py quick --ref <saved run>` | the solver against VULCAN-0D truth, misses first, paired with a saved run |
| `bench/kintracer_gate.py --design G31` | a tracer design paired with the all-species call against VULCAN-0D, misses first |

### Data for the checks and benches

The checks and benches read stored references and VULCAN-0D truth sets that are not part of this repository (about
6 GB). They are found under `CHEM_DATA`:

    benches/yardstick/design/          recorded starts, rows and saved states (units, rows, saved, truth_saved .npz)
    benches/designs/decision/P/        test panel P (rows, starts, forcing .npz)
    benches/kincuda/libref3.npz        reference outputs and rate coefficients of the torch reference solver
    benches/kincuda_ad/refdata3.npz    reference derivatives (torch autograd)
    benches/kintracer/ref4.npz         reference outputs of the reduced-tracer call
    benches/kintest/quick_ref3.npz     a saved quick run of bench/kintest.py
    truth/qual/                        VULCAN-0D truth paths (QC, QCs, QL, QLc, QH, QHs, twins)

Each check stops with a message naming the missing files when they are absent. The stored references are tied to the
exact network files: each entry carries a digest of its inputs, and a check refuses inputs that differ.

## Layout

    maxichem/kinvulcan.py    reader of VULCAN-format reaction lists and NASA-9 files (VulcanNetwork)
    maxichem/kincuda.py      the production call as generated CUDA kernels (CudaSolver)
    maxichem/kincuda_ad.py   tangent and adjoint kernels of one call (ADSolver)
    maxichem/kinjax.py       the call and its derivatives in JAX (KinJax), with the FFI shim kinjax.cc
    maxichem/kintracer.py    the reduced-tracer call (TracerSolver) and designs/
    maxichem/paths.py        data locations (CHEM_DATA, VULCAN_THERMO)
    tests/, bench/           the checks and the benches
    examples/quickstart.py   the example above
    notes.md                 the project's research record
    CLAUDE.md                engineering rules for this repository

## Citation

maxichem has no paper of its own yet; `CITATION.cff` describes the software. The project's published paper describes
an earlier neural-network emulator, not this solver:

> I. Malsky, X. Zhang, T. Kataria, M. Graham, Z. Huang, B. Bonev, S.-M. Tsai, E. K. H. Lee, "Accelerating Chemical
> Kinetics for Exoplanet Atmospheres using Neural Networks", arXiv:2609.00428 (accepted for publication in ApJ).

Its code is at tag `paper-v1` of [Autoregressive-Chemulator](https://github.com/imalsky/Autoregressive-Chemulator).
The chemical network is VULCAN's; please also cite VULCAN (see its repository for the papers).

## License

MIT (`LICENSE`). VULCAN, VULCAN-0D and their data files are separate projects under their own licenses; this
repository only reads their network and thermodynamic files.
