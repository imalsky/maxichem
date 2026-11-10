# Testing

```bash
pytest -m "not slow and not gpu"      # the CPU tests: what CI runs
pytest -m "slow and not gpu"          # the slow CPU tests: VULCAN's own files (needs VULCAN_THERMO), the derivative kernels' compile, the wheel
pytest -m gpu -s                      # the GPU tests, with each case's numbers
```

CI runs the CPU tests and compiles every kernel with NVRTC, with no GPU. The GPU tests stay local, and skip only where there is no GPU.
A check whose `VULCAN_THERMO`, reference data (README.md lists them) or CUDA toolkit is missing fails, and its message names what is missing.

| File | What it tests | Needs |
| --- | --- | --- |
| `test_reader.py` | The reader on invented networks: every rate law, atoms, refusals of reactions, repeated species, headings and NASA-9 files; on VULCAN's own files, only TiSNCHO refused | CPU; the VULCAN files: `VULCAN_THERMO`, marked slow |
| `test_tables_source.py` | The term streams hold the stoichiometry, on a small network too; the header selects the rate-law forms; the argument table matches every kernel's signature; a hand-built reverse law, `drift`, `cap` and a constructor that fails | CPU |
| `test_kernels_compile.py` | Every form of the kernels compiles, including forms no real network reaches | CPU |
| `test_paths.py` | Data locations follow the environment; missing data give a clear error | CPU |
| `test_constants_source.py` | Each setting is written once: the C++ caller's copies equal `_constants.py`; the kernels write no float but 0, 1/2, 1 and 2 outside a named constant; every setting `docs/method.md` names exists | CPU |
| `test_coupling_radiation.py` | The column heating: none from a height-constant flux on a sphere, where the area-weighted form gives some; cooling where the net upward flux grows; each column's budget to round-off; the budget check refuses a NaN flux, naming its column, the area-weighted heating, an overflowing residual, bad thicknesses and other precisions than float64; face temperatures exact for a linear T(z); the radiation's firing steps over 200,000-500,000 steps of 70, 70.1, 0.1 and 35.7 s (periods 300 and 300.1 s) are the first step at or after each multiple, worked out in exact decimal arithmetic | CPU |
| `test_coupling_composition.py` | A hand-worked H2, He, H2O mixture; q to x and back, and the tracer mass and element moles kept through a reaction; hand-worked absolute element totals with unequal cell masses; the start gas's element ratios equal `START_PER_H` and the solar file's rows (typed in the test as dex) with and without N and S; a missing carrier, bad molar masses, mass shapes, negative x or mass, non-finite atoms refused; the dumps' signed mole fractions | CPU |
| `test_coupling_chemistry.py` | The unfinished-cell stop names the counted cause (cold, hot, bad input, still failing) and only its fix; every count of the solver is either such a cause or listed as leading to one | CPU |
| `test_coupling_flow_file.py` | A saved flow or start column loads into its own deck; another gas (composition, cv_R or molar mass alone), grid or shape, a start column whose levels are not [nz + 1], finite and positive, and an old file with no record, are refused | CPU |
| `test_coupling_photolysis.py` | The stand-in J: nearest layer in log pressure, zero at night, day-side cells above and below the table counted; a NaN or negative table, an empty table and a mask that is not boolean refused | CPU |
| `test_packaging.py` | The built wheel holds the kernels and assembles them without the checkout | CPU, slow |
| `test_rate_laws.py` | The forward rate coefficient of every law the reader takes, at three (T, P), against the law written out in the test: Arrhenius, low-pressure only with and without an efficiency, Lindemann and Troe falloff, the special law's two k0 terms and its Troe width, and the collision limit on directions of two and three molecules; once with the kernels' Troe form and once with their plain falloff; the falloff law where Pr is 0, infinite or 0 / 0, in the call and its tangent in T | GPU |
| `test_known_answers.py` | First-order decay at constant pressure; long calls against the NASA-9 equilibrium, also from ranges that differ below and above the split; element conservation; a small network in every kernel; photolysis branches each at its own J; a first try too long, rejected and retried; H + H + M at constant pressure | GPU |
| `test_order.py` | Rodas3's order of accuracy, on a linear problem and on one where the constant-pressure term of the Jacobian is not zero | GPU |
| `test_kernel_copies.py` | The call, recovery, tangent and adjoint kernels return the same x, h, done, steps, status and flag on edge inputs: a non-finite or too long starting step, dt = 0, refused cells beside good ones, a first try rejected twice running | GPU |
| `test_parity.py` | The call against a stored reference of an independent torch solver | GPU, data |
| `test_guard.py` | The guard and recovery; the guard alone on hand-made results whose element share moved by 0.5 to 2 times the drift allowance (flagged only above it); refused cells (the call raises, and the kernels' counts and the other cells' results, read with the raise switched off); independence from the batch; what a closed solver refuses and releases; the call's and `launch`'s named results and the flag they set, which is None after a call that raised | GPU, data |
| `test_dense_lu.py` | The recovery's dense LU and solve, run by a harness kernel (`tests/dense_lu_check.cu`), against threshold partial pivoting written out in numpy: the row order, L U = P A and the solve, on matrices that exchange rows at a third of the pivots or more and keep rows where plain partial pivoting would not, and a tie in the pivot search | GPU |
| `test_call_interface.py` | Every optional argument of the call, `launch`, `jvp`, `vjp` and both constructors is keyword only, checked one parameter at a time, and a setting passed by position raises TypeError; their results are named tuples whose fields keep the old order | CPU |
| `test_ad.py` | Derivatives against finite differences and against each other | GPU, data, slow |
| `test_export_cpp.py` | The C++ caller, bit for bit, through its default arguments too; it refuses the inputs the Python call refuses, with the same message and counts; it refuses another export format, and at construction an export whose kernel argument lists differ from its own (unknown, missing, repeated or misplaced name, J on the guard); a thermal export takes an empty [B, 0] J with the same bits and refuses a [B, 1] one; changing what the accessors return leaves the solver as it was | GPU, CUDA toolkit, slow |
| `test_vulcan_truth.py` | `bench/kintest.py quick` against VULCAN-0D, on both default networks | GPU, data, slow |

`tests/data` holds invented networks with constant-Cp NASA-9 data, so a test can state its answer in closed form. Their `N_i`
reactions have zero rates: they take a network past the 32 rows of the trailing block, so the sparse leading part runs too;
`synthetic_small` has none.

## Known answers

The known-answer tests (`test_rate_laws.py`, `test_known_answers.py`, `test_order.py`) state each expected value from a
formula written out in the test: the rate laws, the Troe factor (Venot et al. 2012, Appendix C; the special law's width as in
docs/method.md), the NASA-9 polynomials (McBride et al. 2002) and closed-form solutions of small reactors. k_B, the
standard pressure and `K_COLL` are typed in `helpers.py` and the tests, not imported from the solver, so that a wrong
constant there fails them. Tests that need other data than `tests/data` write their network and NASA-9 files into a
temporary folder (`helpers.write_network`) and read them with the public reader. The rate laws are seen through
isomerizations beside unchanged spectators, whose decay to equilibrium is exactly exponential at (k_f + k_r) times the
spectators' densities. A direction of one molecule and M stays far below the collision limit, so the limit's rule for it
is not pinned. The falloff law is also checked where Pr is 0, infinite or 0 / 0, against Troe's limit
$F_c^{d^2/(1+d^2)}$ and its T derivative. The equilibrium of H + H + M = H2 + M uses NASA-9 data read from the files in
the test and the standard pressure typed there, so a wrong P0 or a wrong $(k_B T / P_0)^{\Delta\nu}$ fails it. A first
try is rejected for its error alone: its exact Rodas3 trial is positive, and rtol puts either the RMS or only the
largest error above 1.

Not covered by an independent test: efficiencies in a mixed gas (only the bulk gas's is used) and inside a falloff's Pr;
the collision limit on a pair of unequal molecularity; the special law's limits of Pr and its pressure tangent; a NASA-9
split moved by only 50 K (the tests' points are 100 K from it); a photolysis branch whose first product repeats; ties
in the dense LU beyond column 0; element drift in an element other than He (the guard's test moves only He's share).

## The derivative cases

`test_ad.py` runs on NCHO_photo, SNCHO_photo_network_2025 and CHO_photo. Each state is a truth box mapped by species name. The log
multipliers are lm = 0.1 N(0, 1) per reaction unless named lm = 0.

- **Primal.** The tangent and adjoint kernels' results are bitwise `CudaSolver.launch`'s: settled gas at dt 100 s, box starts at dt
  500 s, cold boxes (T < 650 K, where the collision limit acts) at dt 100 s.
- **Non-finite derivatives.** Any cell whose tangents or finite differences are not finite fails its case, unless the guard flagged it 2;
  the errors below cannot judge such a cell, since its scale S is not finite. These cases run at the default max_steps, where any cell with
  flag 2 fails its case too.
- **Tangents against central differences.** Per input class x, T, P, J, lm, rc, one random direction each; the x direction
  keeps sum(x) = 1, since the call refuses a state off it, and the T direction is shortened where the largest step would leave
  [T_MIN, T_MAX]. The part of the x tangent along x itself is checked apart: with the step sizes held it is 0 but for rounding. Per cell and step e in
  1e-4..1e-7: err_e = the largest difference over species (x > 1e-20) divided by S, the larger side's largest value. A plateau is where
  the steps agree: the same accepted-step count both sides, and two consecutive e within 1e-2 of S. The best err over the plateau counts.
- **J in dark cells**, along dJ >= 0: 12 cases. Near 0, at the default atol: the tangent at J = 1e-8 dJ against central differences.
  At exactly 0, atol 1e-16: one-sided differences with Richardson extrapolation, with no coverage bar, since a tiny J often changes the
  accepted-step count, and such a cell has no plateau.
- **Dot products** $\langle c, J v\rangle$ against $\langle J^T c, v\rangle$ per cell, both contracts, per class, with $c_i = N(0, 1) / x_i$ on species above
  1e-20. Trace species near 0 carry rounding of the clamp at 0, so they take 0.
- **Recovered cells** (max_steps 3): primal and flags; dot products; the rc cotangent of a batch of them alone; finite derivatives on flags
  0 and 1, NaN on flag 2.
- **The rc batch dot product**, several directions in one call, determinism, a non-finite starting step, a negative x, a NaN log
  multiplier and a missing J.
- **A failed cell in a batch**, by a NaN log multiplier and by running out of tries: every other cell's results and the `rc` cotangent are
  bitwise those of the batch without it, its own cotangents are NaN, and `still_failed` is 1.

The limits, and why each is what it is, are at the top of `test_ad.py`.

## `bench/kintest.py`

`--network` picks one of the two default networks; `TRUTH` in `bench/kintest_truth.py` names each one's truth sets.

| Network | Boxes | Paths |
| --- | --- | --- |
| `NCHO_photo_network.txt` | boxes_ncho | qual_ncho: QC, QCs, QC twins, QH, QHs, QH twins, QL, QLc |
| `SNCHO_photo_network_2025.txt` | boxes_sncho25, boxes_sncho25p | qual_sncho25: QC, QCs, QC twins, QL |

The sulfur sets score 16 gases: the 12 and H2S, SO2, COS, S2 at 1e-10 (`maxichem/_checks.py`). Their truth holds two condensates the
reader skips, which must be 0 in every state. A box set's manifest lists as errors the boxes whose VULCAN run ended before a stop, at the
truth generator's step cap: `TRUTH` declares their count and a digest of their ids (117 and 54), kintest prints them and scores the rest,
and it refuses a set with any other error or other such boxes. Misses that are understood are pinned in `KNOWN_MISSES` point by point:
set, box, gas, each missed stored state with its floored error, and the reason. Pinned are S2 in box 1939 of boxes_sncho25 (VULCAN's
error) and S2 in box 96 of boxes_sncho25p (maxichem's at atol 1e-25). The run fails on any missed point that is not pinned, in any gas
of any box, the pinned boxes included, on a pinned point that is gone, and on one that moved by more than 0.01 dex. On the sulfur network panel P's gas
gets H2S = 1.41e-5 H2. The day-night path sets (daynight_ncho, daynight_sncho25) are not run: their J changes every call, and the
paths' calls here are dark.

Every panel is a block of cells with its own forcing (T, P, dt per call), mixes (x <- (1 - f) x + f y before a call), kick-twin starts
and VULCAN truth. One loop of ticks: a block's call j runs on tick j * stride, and all scheduled cells go to the solver in one call per
tick. Counters stay on the GPU and are read back every 1,000 ticks; the guard's counts come from differences of `solver.check()`. Per path they hold the worst floored error against VULCAN over the
12 scored gases with its call and gas, stored states with a miss (> 0.1 dex) or a large miss (> 1 dex), unfinished calls, non-finite and
negative states, the largest change of an element ratio (to H) on paths without mixes, accepted steps and the guard's counts.

- **quick.** The first 32,768 boxes (chained calls between their stored stops, 0 to 1e8 s). Then the paths: QC, QCs and QC kick twins (T,
  P and dt every call, every state scored); QH, QHs and QH kick twins, one call per 1500-s hold; QL at dt 500 s (T and P held per hold,
  truth at hold ends) and QLc at dt 500 s; panel P's 256 paths cycled at dt 25, 50 and 500 s for 10,000 calls each (no truth: stability
  counters only; starts mapped by species name). The paths are as long as the truth set holds: QC family 1,000 of 10,000 calls; QH family
  128 of the 1,664 paths, 1,000 holds; QL 3,500 holds; QLc 3,000 calls.
- **full.** Every box. Then a clock of 691,200 ticks: QL at dt 25 s (every tick) and 500 s (every 20th), QLc (every 20th), the QH family at dt 50 s
  (every 2nd), the QC family and the QH family at 1500-s holds as in quick: every VULCAN path, as long as the truth set holds it. Panel P at dt 25 s for 200
  days and dt 50 s for 400 days (both every tick, 69 cycles of its forcing), and at dt 500 s for 1000 days (every 2nd tick, 17 cycles). 90 min is the limit on any
  test or check run: at tick 12,000 it prints an estimate from the last 1,000 ticks' rate and stops above 1.5 h, and it stops whenever the run reaches 1.5 h. A run takes 54 min.

  A chain is sequential, one launch per tick, and a tick costs about what its slowest cell costs: 3.4 ms of kernels at 256 cells, 1.9 ms at 64. So the blocks
  share launches, and the clock's length sets the time. A block's stride is only its place on that schedule.

`--ref` pairs a run with a saved run of the same tier and the same truth (digests of the scored truth and forcing): per path the change of
the worst error and the largest state change, counts above 0.01 and 0.03 dex, steps per call, unfinished calls and latency.
