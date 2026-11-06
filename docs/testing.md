# Testing

```bash
pytest -m "not slow and not gpu"      # the CPU tests: what CI runs
pytest -m gpu -s                      # the GPU tests, with each case's numbers
```

CI runs the CPU tests and compiles every kernel with NVRTC, with no GPU. The GPU tests stay local. A GPU test skips, and says why,
when `VULCAN_THERMO` is unset or its reference data are missing (README.md lists them).

| File | What it tests | Needs |
| --- | --- | --- |
| `test_reader.py` | The reader on invented networks: every rate law, atoms, refusals | CPU |
| `test_tables_source.py` | The term streams hold the stoichiometry; the header selects the rate-law forms; the argument table matches the kernels' signatures | CPU |
| `test_kernels_compile.py` | Every form of the kernels compiles, including forms no real network reaches | CPU |
| `test_paths.py` | Data locations follow the environment; missing data give a clear error | CPU |
| `test_packaging.py` | The built wheel holds the kernels and assembles them without the checkout | CPU, slow |
| `test_known_answers.py` | First-order decay at constant pressure; long calls against the NASA-9 equilibrium; element conservation | GPU |
| `test_order.py` | Rodas3's order of accuracy | GPU |
| `test_parity.py` | The call against a stored reference of an independent torch solver | GPU, data |
| `test_guard.py` | The guard and recovery, bad inputs, independence from the batch | GPU, data |
| `test_ad.py` | Derivatives against finite differences and against each other | GPU, data, slow |
| `test_export_cpp.py` | The C++ caller, bit for bit | GPU, CUDA toolkit, slow |
| `test_vulcan_truth.py` | `bench/kintest.py quick` against VULCAN-0D | GPU, data, slow |

`tests/data` holds invented networks with constant-Cp NASA-9 data, so a test can state its answer in closed form. Their `N_i`
reactions have zero rates: they take a network past the 32 rows of the trailing block, so the sparse leading part runs too;
`synthetic_small` has none.

## The derivative cases

`test_ad.py` runs on NCHO_photo, SNCHO_photo_network_2025 and CHO_photo. Each state is a truth box mapped by species name. The log
multipliers are lm = 0.1 N(0, 1) per reaction unless named lm = 0.

- **Primal.** The tangent and adjoint kernels' results are bitwise `CudaSolver.launch`'s: settled gas at dt 100 s, box starts at dt
  500 s, cold boxes (T < 650 K, where the collision limit acts) at dt 100 s.
- **Tangents against central differences.** Per input class x, T, P, J, lm, rc, one random direction each. Per cell and step e in
  1e-4..1e-7: err_e = the largest difference over species (x > 1e-20) divided by S, the larger side's largest value. A plateau is where
  the steps agree: the same accepted-step count both sides, and two consecutive e within 1e-2 of S. The best err over the plateau counts.
- **J in dark cells**, along dJ >= 0: 12 cases. Near 0, at the default atol: the tangent at J = 1e-8 dJ against central differences.
  At exactly 0, atol 1e-16: one-sided differences with Richardson extrapolation, with no coverage bar, since a tiny J often changes the
  accepted-step count, and such a cell has no plateau.
- **Dot products** $\langle c, J v\rangle$ against $\langle J^T c, v\rangle$ per cell, both contracts, per class, with $c_i = N(0, 1) / x_i$ on species above
  1e-20. Trace species near 0 carry rounding of the clamp at 0, so they take 0.
- **Recovered cells** (max_steps 3): primal and flags; dot products; the rc cotangent of a batch of them alone; finite derivatives on flags
  0 and 1, NaN on flag 2.
- **The rc batch dot product**, several directions in one call, determinism, a non-finite starting step and a negative x.

The limits, and why each is what it is, are at the top of `test_ad.py`.

## `bench/kintest.py`

Every panel is a block of cells with its own forcing (T, P, dt per call), mixes (x <- (1 - f) x + f y before a call), kick-twin starts
and VULCAN truth. One loop of ticks: a block's call j runs on tick j * stride, and all scheduled cells go to the solver in one call per
tick. Counters stay on the GPU and are read back every 1,000 ticks. Per path they hold the worst floored error against VULCAN over the
12 scored gases with its call and gas, stored states with a miss (> 0.1 dex) or a large miss (> 1 dex), unfinished calls, non-finite and
negative states, the largest change of an element ratio (to H) on paths without mixes, accepted steps and the guard's counts.

- **quick.** The first 32,768 boxes (chained calls between their stored stops, 0 to 1e8 s). Then the paths: QC, QCs and QC kick twins (T,
  P and dt every call, every state scored); QH, QHs and QH kick twins, one call per 1500-s hold; QL at dt 500 s (T and P held per hold,
  truth at hold ends) and QLc at dt 500 s; panel P's 256 paths cycled at dt 25, 50 and 500 s for 10,000 calls each (no truth: stability
  counters only; starts mapped by species name). The paths are as long as the truth set holds: QC family 1,000 of 10,000 calls; QH family
  128 of the 1,664 paths, 1,000 holds; QL 3,500 holds; QLc 3,000 calls.
- **full.** Every box. Then a 25-s clock of 3,456,000 ticks (1000 days): QL at dt 25 s (every tick) and 500 s (every 20th), QLc (every 20th),
  the QH family at dt 50 s (every 2nd), the QC family and the QH family at 1500-s holds as in quick, panel P at dt 25, 50 and 500 s (every
  1st, 2nd, 20th) for 1000 days. After 1,000 ticks it prints an estimate, and stops above 5 h unless `--force`.

`--ref` pairs a run with a saved run of the same tier and the same truth (digests of the scored truth and forcing): per path the change of
the worst error and the largest state change, counts above 0.01 and 0.03 dex, steps per call, unfinished calls and latency.
