# maxichem

maxichem is a GPU chemistry solver for exoplanet atmospheres. It is built to replace the chemistry step of a general
circulation model (GCM). Each call advances the gas of many independent cells by one time step. Each cell has its own
temperature, pressure and photolysis rates.

The chemistry is a VULCAN reaction network. The default is the C-H-N-O network of
[VULCAN-JAX](https://github.com/imalsky/VULCAN3) (`NCHO_photo_network.txt`: 69 species, 390 reversible reactions, 48
photolysis branches). With sulfur, the default is its `SNCHO_photo_network_2025.txt` (91 species, 532 reversible
reactions, 61 photolysis branches). maxichem solves the network as a 0-D kinetics problem in CUDA kernels. The kernels
are compiled for the network and run over many cells at once.

- **Physics.** Mass-action kinetics at constant T and P within a call: dx/dt = (S v - x sum(S v)) / n_tot, with
  n_tot = P / kT. Forward rates come from the network file. Reverse rates come from NASA-9 thermodynamics. Elements are
  conserved.
- **Collision limit.** No thermal rate may exceed 1e-9 cm^3 s^-1 as a bimolecular-equivalent rate, with M counted as a
  collision partner (a dissociation A + M counts with its rate coefficient k). When the faster direction of a reaction
  is above it, both directions are scaled down by the same factor, so the equilibrium is kept. It acts where a fit is
  used far outside its range, for example the reverse of CH + M -> C + H + M below about 600 K. VULCAN-JAX uses the
  same rule, and so does the truth data (truth_v3).
- **Photolysis.** Each photolysis branch is a first-order reaction with a rate J (s^-1) given per cell by the caller,
  for example from the GCM's radiative transfer. maxichem does not compute actinic fluxes. At night, J = 0.
- **Integrator.** Rodas3 (order 3, L-stable) with the exact Jacobian and a sparse LU factorization. Adaptive steps
  (rtol 1e-3, atol 1e-25 in mole fraction). Each cell keeps its step size from one call to the next.
- **Guard.** After each call, a guard kernel flags cells with a non-finite state, an unfinished call or a change
  in their element shares. A recovery kernel redoes those cells on the GPU with pivoting and ten times the step budget.
  The host never waits during a call. The counts are read on demand.

## Networks

Any VULCAN-format network up to 254 species can be read, with these rate laws: one or two Arrhenius terms, falloff,
the Troe factor and efficiencies; at most 3 species on a side. Ten of the 11 networks of VULCAN-JAX
load and run. In a short test of loading and running (4,096 cells from NCHO states, one long dark call, then 20 calls
with J) no cell failed and the element shares moved by at most 4e-11 over the 20 calls; this is not an accuracy test:

| Network | Species | Thermal reactions | Photolysis branches | Notes |
| --- | --- | --- | --- | --- |
| NCHO_photo (default), NCHO_thermo, NCHO_photo_lowT_Jupiter | 69 | 390 | 48 (0 for NCHO_thermo) | |
| NCHO_full_photo | 74 | 416 | 57 | |
| CHO_photo | 41 | 207 | 36 | |
| CHO_photo_lowT | 35 | 149 | 32 | thermal only: its photolysis branches make O_1 and CH2_1, which its thermal part lacks |
| SNCHO_photo_2025 (the sulfur network) | 91 | 532 | 61 | tested against its stored reference and VULCAN-0D (Accuracy) |
| SNCHO_photo | 88 | 513 | 60 | the earlier sulfur network; its tests are superseded |
| SNCHO_DMS_photo_Tsai2024 | 95 | 536 | 64 | |
| SNCHO_photo_C3 | 92 | 524 | 60 | one duplicate reaction |
| TiSNCHO_photo | 96 | 541 | 65 | refused: the NASA-9 fit of TiH is unsound below 1000 K; also three duplicate reactions |

On SNCHO_photo_2025 every case of the parity check against the stored reference passes.

The reader refuses a reaction whose atoms do not balance, a reaction no heading gives a law, an unknown heading and a
NASA-9 file whose two ranges do not meet at 1000 K (docs/api.md). A duplicate reaction (the same species on both sides) is
kept, and the rates add, as in VULCAN; the reader lists the duplicates in `net.duplicates`.

## Requirements

- Linux with an NVIDIA GPU. The kernels run in double precision, so their speed follows the GPU's fp64 throughput.
  Only the RTX 6000 Ada (sm_89) has been tested.
- No CUDA toolkit. The kernels are compiled at run time with NVRTC from the `nvidia-cuda-nvrtc-cu12` wheel.
- Python 3.11 or newer, NumPy and PyTorch with CUDA. Tested with Python 3.11, torch 2.10.0+cu128 and NumPy 2.4.6 (the
  environment of the snapy GCM), and with Python 3.12, torch 2.8.0+cu126 and NumPy 2.5.3.
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
net = VulcanNetwork(f"{th}/NCHO_photo_network.txt", f"{th}/NASA9")  # 69 species, He last
solver = CudaSolver(net, torch.device("cuda"), photo=True)
x, h, done, steps = solver(x, T, P, dt, h, J=J)  # float64 CUDA tensors, one row per cell
counts = solver.check()  # guard counts; raises if a cell was outside 300-3000 K
```

## Using it in a GCM

- Call it once per GCM step, after transport. Inputs (float64 CUDA tensors, one row per cell):
  - `x` [cells, species]: mole fractions in the order of `net.species`;
  - `T` [cells] in K; `P` [cells] in dyn cm^-2 (1 bar = 1e6); `dt` [cells] in s;
  - `h` [cells]: the step size the cell's last call returned (any positive value at the first call, for example 1e-10).
    It is only a first guess: the call clips it to [1e-12 s, dt], and a non-finite h acts as dt and is counted;
  - `J` [cells, branches]: photolysis rates in s^-1, one column per branch in the order of `net.photo` (each entry
    names its absorber and branch index as in the network file). Required with `photo=True`; use zeros at night.
- Outputs: the new `x`, the step size to keep, a per-cell `done` flag (the call finished and its result passed the
  guard) and the number of accepted steps, as a named tuple (`CallResult`: `x`, `h`, `done`, `steps`), which
  unpacks as before.
- Tolerances are arguments of every call, in Python keyword only (`rtol=1e-3, atol=1e-25`; also `max_steps=` and
  `J=`) and in the C++ caller (`rtol`, `atol`; a negative `atol` takes the export's default). Interface change: a
  Python call that passes `rtol`, `atol`, `max_steps` or `J` by position now raises TypeError; name them, as also the
  constructors' `drift`, `photo` and `cap`. The default atol, `kincuda.ATOL` = 1e-25 in mole fraction, follows
  trace species down to that level. A smaller atol (for example 1e-30) follows seeds of trace species further, at the
  cost of more steps: in one cool, thin, lit sulfur box (386 K, 9e-5 bar) S2 is off by 0.15 dex at 1e-25 and by 0.002
  dex at 1e-30 against a tight Radau reference. A GCM coupling should pass atol through as a setting.
- T, P and J stay fixed within a call. Cells never couple. The results do not depend on the batch size or the order of
  the cells, so domain decomposition does not change them.
- GCMs advect mass mixing ratios. Convert to mole fractions before the call and back after it. The chemistry conserves
  elements and mass: a call that passes the guard puts every element's share back to its input value (except shares
  near atol, and elements tied together), so rounding does not build up over a long run. Keep `h` per cell between
  calls and in restart files.
- The call refuses a cell below 300 K or above 3000 K (the NASA-9 fits reach 6000 K, but accuracy against VULCAN is
  checked only up to 3000 K), or with a non-finite x, T, P or dt, a negative mole fraction (clamp at 0 before
  the call), mole fractions that do not sum to 1 within `kincuda.X_SUM_TOL` = 1e-10 (normalize in float64 before the
  call), no mole fraction above 0, P <= 0 or so large that P / (k_B T) overflows, dt < 0, or a negative or non-finite J: the kernels do not step it, and the
  whole call raises ValueError (C++: `std::invalid_argument`) when any cell is refused, naming the count and the first
  cell of each kind. The sum is judged once, in the order the call kernel adds it. Reading the counts back for this is
  the call's one host sync. Inputs of the wrong shape, dtype or device raise before any launch.
- Not done here: radiation, transport, condensation.
- From C++ (no Python in the time loop): `solver.export(folder)` writes the kernels (source and cubin for this GPU),
  their tables and the launch sizes; `cpp/maxichem.hpp` (`maxichem::Solver`, libtorch and the CUDA driver API) loads
  the export and launches the same kernels with the same arguments, so a call gives the same bits as in Python.
  Export on the machine that runs it (a cubin is for one GPU architecture). Build: `cmake -S cpp -B cpp/build
  -DPython3_EXECUTABLE=<python with torch> -DCUDAToolkit_ROOT=<CUDA toolkit>`. Check: `tests/test_export_cpp.py` writes
  an export and one call, and compares the C++ call with it bit for bit. Interface change: the C++ solver's sizes,
  names and counts are read-only accessors now (`chem.ns()`, `chem.nj()`, `chem.species()`, `chem.count()`, ...; the
  tensors come back as copies), and an export without photolysis takes no J or an empty [B, 0] one.

## Accuracy

Accuracy is judged against VULCAN-0D, misses first. A miss is a stored state where one of 12 gases (H2O, CO, CO2, CH4,
NH3, HCN, C2H2, H, OH, O, H2, N2) differs from VULCAN by more than 0.1 dex above a per-gas floor (1e-10 for the first
seven, 1e-8 for H, OH and O, 1e-6 for H2 and N2); on the sulfur network also H2S, SO2, COS and S2 (floor 1e-10). The
truth data come from VULCAN-0D with corrections to its step
control and the network data (see notes.md), run at rtol 5e-4. The domain is 300 to 3000 K, 1e-5 to 100 bar,
metallicity 0.1 to 300 times solar, C/O 0.1 to 2, with and without photolysis.

Results at the default tolerances (rtol 1e-3, atol 1e-25), against the truth made with the current collision limit
(truth_v3); the day-night rows (marked *) were measured against the earlier truth, which differs from it by at most
4e-4 dex on those paths:

| Test | States scored | Misses | Worst error |
| --- | --- | --- | --- |
| NCHO_photo: 131,072 boxes, 0 to 1e8 s, half lit | 6,160,384 | 0 | 0.066 dex |
| NCHO_photo: 1,096 GCM-like paths (recorded forcing and mixing) | 1,326,198 | 0 | 0.004 dex |
| NCHO_photo: 64 paths with photolysis on a one-day cycle, 1,000 calls each (*) | 64,000 | 0 | 0.002 dex |
| SNCHO_photo_2025 (91 species): 986 dark boxes | 46,342 | 0 | 0.030 dex |
| SNCHO_photo_2025: 945 lit boxes | 44,415 | 7, in 1 box | 4.34 dex |
| SNCHO_photo_2025: 350 targeted lit boxes (C/O >= 0.8 at 500-1500 K, below 800 K, metallicity >= 100) | 16,450 | 1 | 0.155 dex |
| SNCHO_photo_2025: 208 GCM-like paths (recorded forcing and mixing) | 245,328 | 0 | 0.007 dex |
| SNCHO_photo_2025: 272 GCM-like paths, 64 of them with a one-day light cycle (*) | 312,000 | 0 | 0.007 dex |

One NCHO box (2973 K, 43 bar) has a call that the guard recomputes and the recovery does not save: its last step of
5.6e7 s moves the element shares by 5.5e-6, above the allowance; its error is 0.0003 dex (notes.md).

The sulfur boxes use the photolysis rates of a 1-D WASP-39b model on the same network; the paths replay the NCHO paths'
forcing with sulfur added as H2S (S/H 1.41e-5 times the metallicity). Besides the 16 scored gases, C3H5, allene,
propyne, H2CS, CS and CS2 (floor 1e-10) stay within 0.013 dex everywhere.

The two sulfur misses were checked against Radau (rtol 1e-10, atol 1e-30, on the same chemistry as VULCAN-0D):

- The large one is an error in VULCAN-0D. In that box (511 K, 2.6e-3 bar, lit), VULCAN-0D ends with no S2; Radau
  reaches S2 = 2e-6 at 1e8 s, and maxichem is within 0.084 dex of Radau. The same box fails the same way on the earlier
  sulfur network.
- The small one is maxichem's. In a cool, thin, lit box (386 K, 9e-5 bar) S2 is off by 0.155 dex at 1.9e7 s, while
  VULCAN-0D is within 0.001 dex of Radau. A tighter rtol does not change it; atol 1e-30 brings it to 0.002 dex. The
  default stays 1e-25; atol is a setting of every call (Using it in a GCM).

Sulfur species in lit gas below 800 K can grow by many orders of magnitude from tiny seeds, so both codes need a small
absolute tolerance. The sulfur truth uses a VULCAN-0D cutoff of 1e-28 in mole fraction. On the earlier sulfur network,
maxichem with atol 1e-16 missed in 22 of 1,955 boxes, 17 of them by more than 1 dex; this is why the default is 1e-25.

Limits of the sulfur result:

- 118 of the 2,048 boxes and 54 of the 404 targeted boxes have no VULCAN-0D result (stopped at the 2e6-step cap, or
  one forced step at the minimum size), so the table does not score them. Radau solved 54 of them, all from unreacted
  gas: maxichem is within 0.016 dex there. 118 have no reference: 10 unreacted starts whose Radau runs did not finish
  in the time given, and 108 starts that need a VULCAN-0D donor run. The boxes left out are more often lit and cool, so
  the score under-represents the hardest gas.
- Radau on 60 more boxes that VULCAN-0D finished finds maxichem within 0.0021 dex and VULCAN-0D within 0.0011 dex.

The NCHO truth keeps VULCAN-0D's default tolerance: 255 of 256 sampled boxes, rerun with the 1e-28 cutoff, change by at
most 0.003 dex (the other one stopped at the step cap).

The guard recomputed 2 of the 6,160,384 cell-calls of the NCHO boxes, both recovered, and none in the other tests.

## Speed

On an RTX 6000 Ada with no other job, at a GCM-sized batch (1,382,400 cells: the states of 4,096 truth boxes at 1e4 s,
repeated; on the sulfur network the 1,930 accepted boxes of its first 2,048, 1,381,880 cells), with 70-s calls on gas
that changes little between calls (about one step per call). "Lit" replays the boxes' J, so half of the cells are lit:

| Network | Species | atol | us per cell-call, dark / lit |
| --- | --- | --- | --- |
| NCHO_photo | 69 | 1e-25 (default) | 0.354 / 0.349 |
| NCHO_photo | 69 | 1e-16 | 0.346 / 0.349 |
| SNCHO_photo_2025 (sulfur gas from its truth boxes) | 91 | 1e-25 (default) | 0.684 / 0.651 |
| SNCHO_photo_2025 (sulfur gas from its truth boxes) | 91 | 1e-16 | 0.661 / 0.658 |

Gas that changes between calls needs more steps, and atol 1e-25 needs about twice as many as atol 1e-16. This test bed has
the same batch size, with 2,048 truth-box states at 1e4 s (1,930 on the sulfur network). Before each 70-s call, it mixes in 1 % of the gas of the nearest
box in T and P (median 27 K and 0.06 dex away; metallicity and C/O can differ):

| Network | atol | us per cell-call | Steps per call |
| --- | --- | --- | --- |
| NCHO_photo | 1e-25 (default) | 7.49 | 27.2 |
| NCHO_photo | 1e-16 | 3.89 | 13.7 |
| SNCHO_photo_2025 | 1e-25 (default) | 16.9 | 29.0 |
| SNCHO_photo_2025 | 1e-16 | 9.05 | 15.1 |

The timings include the mixing step. atol is an argument of the call.

In coupled snapy runs every species is a passive tracer and the chemistry is called after every dynamics step, over
all cells at once. These runs were dark unless marked, at the default atol, on one RTX 6000 Ada, from a solar-like gas
(with sulfur as H2S, S/H 1.41e-5, on the sulfur network) relaxed by one long call. Medians over the steps:

| Run | Network | Cells | Steps per call | us per cell-call |
| --- | --- | --- | --- | --- |
| Cartesian box, 3 model days, 70 s steps | NCHO_photo | 921,600 | 1.43 | 0.47 |
| Cubed sphere C48x100, 1 model day, 70 s steps | NCHO_photo | 1,382,400 | 1.40 | 0.45 |
| Cubed sphere C64x100, 1 model day, 50 s steps | NCHO_photo | 2,457,600 | 1.32 | 0.42 |
| Cartesian box, the first 100 steps of 70 s | SNCHO_photo_2025 | 921,600 | 1.96 | 1.24 |
| The same, lit (J of a 1-D WASP-39b model's layer nearest in P, times cos z) | SNCHO_photo_2025 | 921,600 | 1.96 | 1.25 |

No cell failed. Real transport sits close to the quiet-gas end of the two benches above.

## Derivatives

The package also holds tangent and adjoint kernels of a call (`ADSolver`). They give the derivatives of the call's
x and h with respect to x, T, P, dt, h, the photolysis rates J, per-cell rate multipliers and the Arrhenius
parameters, with the step sizes frozen or with the step control differentiated. They repeat the call's arithmetic, so
their primal result is bitwise the call's. They take networks up to 96 species with photolysis, the Troe factor and the
collision limit (every VULCAN-JAX network the reader accepts).

`tests/test_ad.py` checks them against finite differences of the call and by dot products between the two
kernels. Every case passes on CHO_photo, NCHO_photo and SNCHO_photo_2025 (on the sulfur network the
branches the check's boxes lack take the J of a 1-D WASP-39b model). In a dark cell, the J tangent at exactly J = 0 is not verified at the default
atol: finite differences cannot resolve it. We infer that it holds only for a tiny, state-dependent range of J. For a
J derivative there, call at a small positive rate (the check uses 1e-8 s^-1, where it passes). There is no independent
reference for these derivatives on the current networks.

## Tests and the data they need

`pytest -m "not slow and not gpu"` runs the CPU tests, and `pytest -m "slow and not gpu"` the slow ones. `pytest -m gpu -s` runs the
GPU tests and prints each case's numbers.
[docs/testing.md](docs/testing.md) lists what each test verifies.

The checks and benches read stored references and truth data that are not part of this repository. They are found
under `CHEM_DATA`:

    benches/kincuda/libref10.npz       outputs of the torch reference solver on the check's inputs (NCHO_photo)
    benches/kincuda/libref10_SNCHO25.npz  the same for the sulfur network SNCHO_photo_network_2025
    benches/kincuda/jref_SNCHO25.npz   J of a 1-D WASP-39b model on that network (the derivative check's sulfur J)
    benches/yardstick/design/          recorded starts used as check inputs
    benches/designs/decision/P/        test panel P (paths, starts, forcing)
    truth_v3/boxes_ncho/               VULCAN-0D truth boxes (Sobol design, 131,072 boxes, 0 to 1e8 s)
    truth_v3/qual_ncho/                VULCAN-0D truth paths (GCM-like forcing with mixing)
    truth_v3/boxes_sncho25/, boxes_sncho25p/, qual_sncho25/   the same for SNCHO_photo_network_2025
    truth_v2/old_network/boxes_ncho_r5e-3/boxes_000.npz   input states of the derivative check
    benches/kintest/                   saved quick runs of bench/kintest.py (quick_ncho_v3_ref1.npz NCHO, quick_sncho25_v3_ref1.npz sulfur)

Each check stops with a message that names the missing files. Each stored reference of the parity tests
carries a digest of its inputs, and the check refuses inputs that differ. The truth sets and the references record
the collision rule they were made with (`RATE_RULE`, docs/method.md), and the checks refuse any other; truth_v2/ holds
the truth of the earlier rule, kept only as a record. `bench/kintest.py` refuses a truth set that
is not complete (boxes not run, failed runs, bad states, or truth paths missing from its job list), and a paired run
refuses a saved run made on other truth data (a digest of the scored truth).

## Layout

    maxichem/kinvulcan.py    reader of VULCAN-format networks and NASA-9 files (VulcanNetwork)
    maxichem/kincuda.py      the call (CudaSolver)
    maxichem/kernels/        the CUDA kernels; _kernel_header.py writes each network's numbers for them
    maxichem/kincuda_ad.py   tangent and adjoint kernels (ADSolver; networks up to 96 species)
    maxichem/paths.py        data locations (CHEM_DATA, VULCAN_THERMO)
    cpp/                     the C++ caller of an exported solver (maxichem.hpp) and its test
    tests/, bench/           the checks and the benches
    examples/quickstart.py   the example above
    docs/                    the method, the architecture, the API and a glossary
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
