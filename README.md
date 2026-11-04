# maxichem

maxichem is a GPU chemistry solver for exoplanet atmospheres. It is built to replace the chemistry step of a general
circulation model (GCM). Each call advances the gas of many independent cells by one time step. Each cell has its own
temperature, pressure and photolysis rates.

The chemistry is a VULCAN reaction network. The default is the C-H-N-O network of
[VULCAN-JAX](https://github.com/imalsky/VULCAN3) (`NCHO_photo_network.txt`: 69 species, 390 reversible reactions, 48
photolysis branches). maxichem solves it as a 0-D kinetics problem in CUDA kernels. The kernels are generated for the
network and run over many cells at once.

- **Physics.** Mass-action kinetics at constant T and P within a call: dx/dt = (S v - x sum(S v)) / n_tot, with
  n_tot = P / kT. Forward rates come from the network file. Reverse rates come from NASA-9 thermodynamics. Elements are
  conserved.
- **Collision limit.** No thermal rate may exceed 1e-9 cm^3 s^-1 as a two-body rate (three-body rates times n_tot).
  When the faster direction of a reaction is above it, both directions are scaled down by the same factor, so the
  equilibrium is kept. It acts where a fit is used far outside its range, for example the reverse of CH + M -> C + H + M
  below about 600 K. VULCAN-JAX and the truth data use the same limit.
- **Photolysis.** Each photolysis branch is a first-order reaction with a rate J (s^-1) given per cell by the caller,
  for example from the GCM's radiative transfer. maxichem does not compute actinic fluxes. At night, J = 0.
- **Integrator.** Rodas3 (order 3, L-stable) with the exact Jacobian and a sparse LU factorization. Adaptive steps
  (rtol 1e-3, atol 1e-25 in mole fraction). Each cell keeps its step size from one call to the next.
- **Robustness.** After each call, a guard kernel flags cells with a non-finite state, an unfinished call or a change
  in their element shares. A recovery kernel redoes those cells on the GPU with pivoting and ten times the step budget.
  The host never waits during a call. The counts are read on demand.

## Networks

Any VULCAN-format network up to 254 species can be read, with these rate laws: one or two Arrhenius terms, falloff,
the Troe factor, efficiencies and explicit reverse laws; at most 3 species on a side. All 11 networks of VULCAN-JAX
load and run. In a short test of loading and running (4,096 cells from NCHO states, one long dark call, then 20 calls
with J) no cell failed and the element shares moved by at most 4e-11 over the 20 calls; this is not an accuracy test:

| Network | Species | Thermal reactions | Photolysis branches | Notes |
| --- | --- | --- | --- | --- |
| NCHO_photo (default), NCHO_thermo, NCHO_photo_lowT_Jupiter | 69 | 390 | 48 (0 for NCHO_thermo) | |
| NCHO_full_photo | 74 | 416 | 57 | |
| CHO_photo | 41 | 207 | 36 | |
| CHO_photo_lowT | 35 | 149 | 32 | thermal only: its photolysis branches make O_1 and CH2_1, which its thermal part lacks |
| SNCHO_photo | 88 | 513 | 60 | tested against its own stored reference |
| SNCHO_photo_2025, SNCHO_DMS_photo_Tsai2024 | 91-95 | 532-536 | 61-64 | |
| SNCHO_photo_C3 | 92 | 524 | 60 | one duplicate reaction |
| TiSNCHO_photo | 96 | 541 | 65 | three duplicate reactions |

The reader refuses a reaction whose atoms do not balance. A duplicate reaction (the same species on both sides) is
kept, and the rates add, as in VULCAN; the reader lists the duplicates in `net.duplicates`.

## Requirements

- Linux with an NVIDIA GPU. The kernels run in double precision, so their speed follows the GPU's fp64 throughput.
  Only the RTX 6000 Ada (sm_89) has been tested.
- No CUDA toolkit. The kernels are compiled at run time with NVRTC from the `nvidia-cuda-nvrtc-cu12` wheel.
- Python 3.10 or newer, NumPy and PyTorch with CUDA. Tested with Python 3.12, torch 2.8.0+cu126 and NumPy 2.5.3.
- The network and NASA-9 files of VULCAN-JAX (its `src/vulcan3/thermo` folder). They are not bundled.

## Install

    git clone https://github.com/imalsky/maxichem.git
    cd maxichem
    pip install -e .

| Variable | Meaning |
| --- | --- |
| `VULCAN_THERMO` | VULCAN-JAX's `src/vulcan3/thermo` folder (used by the example, the checks and the benches) |
| `CHEM_DATA` | data root of the checks and benches. Default: `data/` in the checkout |

## Quick start

    VULCAN_THERMO=/path/to/VULCAN-JAX/src/vulcan3/thermo python examples/quickstart.py

The example relaxes 512 cells to equilibrium, then runs five GCM-like calls with photolysis in the upper cells.
In short:

```python
import torch
from maxichem import CudaSolver, VulcanNetwork

th = "/path/to/VULCAN-JAX/src/vulcan3/thermo"
net = VulcanNetwork(f"{th}/NCHO_photo_network.txt", f"{th}/NASA9")    # 69 species, He last
solver = CudaSolver(net, torch.device("cuda"), photo=True)
x, h, done, steps = solver(x, T, P, dt, h, J=J)    # float64 CUDA tensors, one row per cell
counts = solver.check()                             # guard counts; raises if a cell was below 300 K
```

## Using it in a GCM

- Call it once per GCM step, after transport. Inputs (float64 CUDA tensors, one row per cell):
  - `x` [cells, species]: mole fractions in the order of `net.species`;
  - `T` [cells] in K; `P` [cells] in dyn cm^-2 (1 bar = 1e6); `dt` [cells] in s;
  - `h` [cells]: the step size the cell's last call returned (any positive value at the first call, for example 1e-10).
    It is only a first guess: the call clips it to [1e-12 s, dt], and a non-finite h acts as dt;
  - `J` [cells, branches]: photolysis rates in s^-1, one column per branch in the order of `net.photo` (each entry
    names its absorber and branch index as in the network file). Use zeros at night.
- Outputs: the new `x`, the step size to keep, a per-cell `done` flag and the number of accepted steps.
- T, P and J stay fixed within a call. Cells never couple. The results do not depend on the batch size or the order of
  the cells, so domain decomposition does not change them.
- GCMs advect mass mixing ratios. Convert to mole fractions before the call and back after it. The chemistry conserves
  elements and mass. Keep `h` per cell between calls and in restart files.
- A cell below 300 K, or with a non-finite x, T, P or dt, or a negative or non-finite J, is not stepped. It is counted,
  and `check()` raises for cold cells.
- Not done here: radiation, transport, condensation. The interface is Python only.

## Accuracy

Accuracy is judged against VULCAN-0D, misses first. A miss is a stored state where one of 12 gases (H2O, CO, CO2, CH4,
NH3, HCN, C2H2, H, OH, O, H2, N2) differs from VULCAN by more than 0.1 dex above a per-gas floor (1e-10 for the first
seven, 1e-8 for H, OH and O, 1e-6 for H2 and N2); on the sulfur network also H2S, SO2, COS and S2 (floor 1e-10). The
truth data come from VULCAN-0D with corrections to its step
control and the network data (see notes.md), run at rtol 5e-4. The domain is 300 to 3000 K, 1e-5 to 100 bar,
metallicity 0.1 to 300 times solar, C/O 0.1 to 2, with and without photolysis.

Results on the truth data of 2026-11-03 (rtol 1e-3, atol 1e-25, the defaults):

| Test | States scored | Misses | Worst error |
| --- | --- | --- | --- |
| NCHO_photo: 131,072 boxes, 0 to 1e8 s, half lit | 6,160,384 | 0 | 0.066 dex |
| NCHO_photo: 904 GCM-like paths (recorded forcing and mixing) | 1,016,411 | 0 | 0.004 dex |
| NCHO_photo: 64 paths with photolysis on a one-day cycle, 1,000 calls each | 64,000 | 0 | 0.002 dex |
| SNCHO_photo (88 species): 993 dark boxes | 46,671 | 0 | 0.029 dex |
| SNCHO_photo: 962 lit boxes | 45,214 | 7, in 1 box | 4.34 dex |

The one sulfur miss is an error in VULCAN-0D. In that box (511 K, 2.6e-3 bar, lit), VULCAN-0D ends with no S2. Radau
(rtol 1e-10, atol 1e-30, on the same chemistry as VULCAN-0D) reaches S2 = 2e-6 at 1e8 s, and maxichem is within 0.084 dex
of Radau.

Sulfur species in lit gas below 800 K can grow by many orders of magnitude from tiny seeds, so both codes need a small
absolute tolerance. The sulfur truth uses a VULCAN-0D cutoff of 1e-28 in mole fraction. VULCAN-0D's default is 1e-2
molecules cm^-3, a mole fraction near 1e-16 at 1e-5 bar, and with it the truth is off by more than 0.1 dex in 18 of 991
boxes. At maxichem's earlier default of atol 1e-16, 22 of the 1,955 boxes miss, 17 of them by more than 1 dex (worst 6.7
dex).

Limits of the sulfur result:

- It covers 2,048 boxes. 93 of them have no VULCAN-0D result (92 stopped at the 2e6-step cap, 1 forced a step at the
  minimum size), so the table does not score them. We solved 25 of the 93 with Radau: maxichem has 0 misses on them
  (1,175 states, worst 0.002 dex). The other 68 have no reference.
- VULCAN-0D's cutoff can still lose a trace species, as in the box above. Radau on 44 other boxes finds VULCAN-0D
  within 0.0005 dex and maxichem within 0.005 dex.

The NCHO truth keeps VULCAN-0D's default tolerance: 255 of 256 sampled boxes, rerun with the new cutoff, change by at most
0.003 dex (the other one stopped at the step cap).

The guard recomputed none of the NCHO cell-calls in these tests and 3 of the 91,885 sulfur ones (all recovered).

## Speed

On an RTX 6000 Ada with no other job, at a GCM-sized batch (1,382,400 cells: the states of 4,096 truth boxes at 1e4 s,
repeated), with 70-s calls on gas that changes little between calls (about one step per call). "Lit" replays the
boxes' J, so half of the cells are lit:

| Network | Species | atol | us per cell-call, dark / lit |
| --- | --- | --- | --- |
| NCHO_photo | 69 | 1e-25 (default) | 0.354 / 0.349 |
| NCHO_photo | 69 | 1e-16 | 0.346 / 0.349 |
| SNCHO_photo (sulfur gas from the truth boxes) | 88 | 1e-25 (default) | 0.658 / 0.635 |
| SNCHO_photo (sulfur gas from the truth boxes) | 88 | 1e-16 | 0.636 / 0.636 |

Gas that changes between calls needs more steps, and atol 1e-25 needs about twice as many as atol 1e-16. This test bed has
the same batch size, with 2,048 truth-box states at 1e4 s. Before each 70-s call, it mixes in 1 % of the gas of the nearest
box in T and P (median 27 K and 0.06 dex away; metallicity and C/O can differ):

| Network | atol | us per cell-call | Steps per call |
| --- | --- | --- | --- |
| NCHO_photo | 1e-25 (default) | 7.49 | 27.2 |
| NCHO_photo | 1e-16 | 3.89 | 13.7 |
| SNCHO_photo | 1e-25 (default) | 17.0 | 30.1 |
| SNCHO_photo | 1e-16 | 9.23 | 16.0 |

We expect real transport to be gentler than this bed, but that is not yet measured on these networks. A coupled GCM run on
the older 53-species network, at atol 1e-16, took about one step per call. The cost in a GCM on the current networks is not
known until a coupled run. atol is an argument of the call.

## Derivatives and JAX

The package also holds tangent and adjoint kernels of a call (`ADSolver`). They give the derivatives of the call's
x and h with respect to x, T, P, dt, h, the photolysis rates J, per-cell rate multipliers and the Arrhenius
parameters, with the step sizes frozen or with the step control differentiated. They repeat the call's arithmetic, so
their result is bitwise the call's. They take networks up to 96 species with photolysis, the Troe factor and the
collision limit (every VULCAN-JAX network). They are checked on CHO_photo, NCHO_photo and SNCHO_photo against finite
differences of the call and by dot products between the two kernels (a development check, not yet in tests/): every
case passes on CHO_photo and NCHO_photo; on SNCHO_photo two cases (the J tangent on box starts at dt 500 s) are just
above their bar. In a dark cell, the J tangent at exactly J = 0 holds only for a tiny, state-dependent range of J, too
small for finite differences to confirm at the default atol. For a J derivative there, call at a small positive rate
(the check uses 1e-8 s^-1). Not yet done: an independent reference for these networks, and the JAX layer (`KinJax`), which
refuses to run until it is ported.

The GCM advects every species. An earlier reduced-tracer call (31 advected tracers for the old 53-species network) was
removed; notes.md keeps its method and results.

## Tests and the data they need

| Check | What it verifies |
| --- | --- |
| `tests/kincuda_check.py <VULCAN_THERMO>` | the CUDA call against stored outputs of an independent torch reference solver: cold, scrambled and 2,000-call chained calls; photolysis calls, a lit chain and lit recovery; bad inputs; guard and recovery; batch size and order |
| `bench/kintest.py quick --ref <saved run>` | the solver against VULCAN-0D truth boxes and truth paths, misses first, paired with a saved run |

The checks and benches read stored references and truth data that are not part of this repository. They are found
under `CHEM_DATA`:

    benches/kincuda/libref7.npz        outputs of the torch reference solver on the check's inputs (NCHO_photo)
    benches/kincuda/libref7_SNCHO.npz  the same for the sulfur network
    benches/yardstick/design/          recorded starts used as check inputs
    benches/designs/decision/P/        test panel P (paths, starts, forcing)
    truth_v2/boxes_ncho/               VULCAN-0D truth boxes (Sobol design, 131,072 boxes, 0 to 1e8 s)
    truth_v2/qual_ncho/                VULCAN-0D truth paths (GCM-like forcing with mixing)
    benches/kintest/                   saved quick runs of bench/kintest.py

Each check stops with a message that names the missing files. Each stored reference of `tests/kincuda_check.py`
carries a digest of its inputs, and the check refuses inputs that differ. `bench/kintest.py` refuses a truth set that
is not complete (boxes not run, failed runs, bad states, or truth paths missing from its job list), and a paired run
refuses a saved run made on other truth data (a digest of the scored truth).

## Layout

    maxichem/kinvulcan.py    reader of VULCAN-format networks and NASA-9 files (VulcanNetwork)
    maxichem/kincuda.py      the call as generated CUDA kernels (CudaSolver)
    maxichem/kincuda_ad.py   tangent and adjoint kernels (ADSolver; networks up to 96 species)
    maxichem/kinjax.py       the JAX layer (KinJax), with the FFI shim kinjax.cc
    maxichem/paths.py        data locations (CHEM_DATA, VULCAN_THERMO)
    tests/, bench/           the checks and the benches
    examples/quickstart.py   the example above
    notes.md                 the research record

## Citation

maxichem has no paper yet; `CITATION.cff` describes the software. The project's published paper describes an earlier
neural-network emulator, not this solver:

> I. Malsky, X. Zhang, T. Kataria, M. Graham, Z. Huang, B. Bonev, S.-M. Tsai, E. K. H. Lee, "Accelerating Chemical
> Kinetics for Exoplanet Atmospheres using Neural Networks", arXiv:2609.00428 (accepted for publication in ApJ).

Its code is at tag `paper-v1` of [Autoregressive-Chemulator](https://github.com/imalsky/Autoregressive-Chemulator).
The networks are VULCAN's; please also cite VULCAN (see its repository for the papers).

## License

MIT (`LICENSE`). VULCAN, VULCAN-0D, VULCAN-JAX and their data files are separate projects under their own licenses;
this repository only reads their network and thermodynamic files.
