# maxichem

The solver code: VULCAN's C-H-N-O mechanism as a physical 0-D solver in CUDA kernels batched on the GPU (one call per
cell and GCM step), with derivatives of every call for learning where it helps; the reduced-tracer call; the fast test
runner and the checks. `README.md` is the public front page; `notes.md` is the research record (every experiment, its
numbers and conclusion, negative results and the decisions; one file, a later entry supersedes an earlier one). The
earlier neural-network code lines are in github.com/imalsky/Autoregressive-Chemulator (tag `paper-v1`: the published
paper's code). The pure-torch reference solver that wrote the checks' stored references, and the averaged scorer, are
not in this repository.

## Layout and environment
- `maxichem/kinvulcan.py` (the reader: VULCAN-format reaction lists and NASA-9 files read as data, NCHO and the thermal
  gas-phase part of the other VULCAN networks (Troe and efficiency lines, the special and irreversible laws; photolysis
  and condensation sections skipped and counted); `formula`, `KB`, `P0`), `maxichem/kincuda.py` (the production call as
  CUDA kernels generated per network, NVRTC: the call kernel, 2 or 4 warps per cell; a guard kernel that lists failed
  cells; a recover kernel that redoes them on the device with a dense, threshold-pivoted Rodas3 and ten times the step
  budget, 40,000 tries at the default), `maxichem/kincuda_ad.py` (tangent and adjoint kernels of one call, contracts A
  and F), `maxichem/kinjax.py` + `maxichem/kinjax.cc` (the call and its derivatives as JAX primitives through an XLA FFI
  shim, built with g++ into `$CHEM_DATA/caches/kinjax` at first use), `maxichem/kintracer.py` + `maxichem/designs/`
  (the reduced-tracer call), `maxichem/paths.py` (every data location); `bench/kintest.py` (the fast test runner:
  quick and full tiers, misses first, paired with a saved run); `bench/kintracer_gate.py`; `tests/kincuda_check.py`,
  `tests/kincuda_ad_check.py`, `tests/kinjax_check.py`, `tests/kintracer_check.py`, `tests/libref.py` (the checks'
  stored reference); `examples/quickstart.py`.
- `maxichem/__init__.py` loads the classes lazily: importing the package imports neither torch nor JAX. Intra-package
  imports are relative; tests, benches and examples import `maxichem` (`pip install -e .` or `PYTHONPATH=<repo>`).
- Production call: `kincuda.CudaSolver(net, device, drift=None)(x, T, P, dt, h)` -> `(x, h, done, accepted steps)`.
  Numerics (the kernel's module docstring): rates from T and P every call; Rodas3 (gamma 1/2) with the exact Jacobian,
  W = I / (gamma h) - J factored by a sparse LU without row exchanges plus Sherman-Morrison for the constant-P term;
  rtol 1e-3, atol 1e-18; err = the RMS over species of the embedded error scaled by atol + rtol max(|N|, |N'|), except
  on the tries before a call's first accepted step, where err = max(RMS, max-norm) (after an abrupt input change the
  carried step must not hide one fast species in the mean); accept when err <= 1 and no species < -atol, clamp at 0,
  next step 0.9 err^(-1/3) with the same err within [0.2, 5] (<= 0.5 after a rejection), max_steps 4000; each cell's
  step size carried to its next call. drift: the element-drift allowance (default 1e-6, NCHO's; a network whose
  rounding drift r on the parity stress cases exceeds 1e-7 takes 10 r, measured with a drift survey kept outside this
  repository (port it to kinvulcan before the next new network); Venot 2019: 1.61e-5). No host sync: cells the guard
  flags (non-finite state, unfinished call, element drift) are recomputed on the device; `CudaSolver.fallback` reads
  the last call's counts (`kincuda.COUNTS`), `CudaSolver.check()` the totals and raises if any cell was below 300 K
  (such cells are counted and not stepped). `launch()` also returns the per-cell flag (0 passed, 1 recovered, 2 still
  failed).
- Kernel layout: any VULCAN-format network the reader takes, up to 254 species, <= 3 species per side, |S| <= 2 (parity
  tested on 29-106 species); W = 2 warps per cell, 4 when the shared memory holds at most 3 cells per block (the sulfur
  networks); the last min(32, NS) rows factored in registers; cells per block from registers and shared memory. The
  stream cuts follow the lane count, so W 2 and W 4 differ at rounding level: W is fixed per network, never per batch
  size. Batches of 2 up to `CudaSolver.small` cells (the SM count) run a one-cell-per-block build of the same kernel
  (the cells spread over the SMs; bitwise the packed build), so results do not depend on batch size or order. One
  `CudaSolver` runs one call at a time on torch's current stream; only sm_89 is tested.
- Derivatives: `kincuda_ad.ADSolver(net, device).jvp(...)` (1-16 directions per call) and `.vjp(...)` for x, T, P,
  dt, h, per-cell log rate multipliers lm and the Arrhenius table rc. Contract A (adapt=True, default: the step
  control differentiated on the realised branch, the derivative of the computed map, with dt and h derivatives; on a
  first try through the norm err selects, a max shared evenly among tied species) or F (adapt=False: accepted steps
  frozen; no dt / h derivative; closer to the physical sensitivity on multi-step calls). The primal is bitwise the call
  kernel's. Cells the guard flags are redone on the recovery's path and differentiated there (jvp_rec / vjp_rec); flag
  2 (still failed) gives NaN derivatives (and a NaN Arrhenius cotangent of the batch: mask flag-2 cells in training).
  No host sync: the counts stay on the device, `status` and `flag` read them on demand (as CudaSolver's), `check()`
  raises for cells below 300 K. Networks: <= 64 species, <= 32 leading rows, < 512 reactions, the Arrhenius, falloff
  and special laws only (NCHO); others raise NotImplementedError.
- JAX: `kinjax.KinJax(net, drift=None)(x, T, P, dt, h, lm=None, rc=None, adapt=True)` -> (x, h, done, steps, flag,
  status); jvp / vjp / grad / linearize / jacfwd / jacrev / vmap / jit / scan through the same kernels, bitwise
  CudaSolver / ADSolver; the call includes the device-side recovery with or without lm; jax.jvp is one tangent call (its
  primal outputs bitwise the call's); linearize / vjp / grad bind the call plus the tangent / adjoint primitive
  (partial-evaluation rules on JAX internals: re-run tests/kinjax_check.py after a JAX upgrade; the `[jax]` extra pins
  the tested version). `kinjax.check(status)` raises for cold cells. Set XLA_PYTHON_CLIENT_PREALLOCATE=false on a shared
  GPU; jax_enable_x64 must be on.
- Reduced-tracer call: `maxichem/kintracer.py`. A design is a JSON file in `maxichem/designs/` (G31, the recommended one
  for C/O <= 1; N30sN2H, G30b, N30, N26) naming the carried species and the advected rows (element budgets, molar sums);
  He is the mass remainder. Two per-cell CUDA kernels (NVRTC, one warp per cell) run around `CudaSolver`: the
  maximum-entropy projection of the cell's memory onto its mixed budgets (8 Jacobi-scaled Newton iterations, up to 60
  more for unconverged cells, then for cells still unconverged up to 400 damped Newton steps on the convex dual
  objective with an Armijo search, from the phase-2 multipliers and from zero, the lower residual kept), then the exact
  element spill, then the new tracers, mean molar mass, cp and an opt-in heating rate. Batch- and order-independent, no
  host sync: `TracerSolver(net, design, device)(q, mem, T, P, dt, h, *, photo=None, heating=False)` -> `(q, mem, h, mu,
  cp, Q, flag, counts)`; `step` / `prepare` work in specific moles; photo is reserved (raises unless None).
  `bench/kintracer_gate.py` runs a design paired with the full mechanism against VULCAN: the kill gate of 18 paths,
  then `--panels`, misses first, every call scored against the full mechanism.
- Environment: `VULCAN_THERMO` (the folder with VULCAN's `NCHO_thermo_network.txt` and `NASA9/`; no default) and
  `CHEM_DATA` (the data root; default `data/` in the checkout), both read by `maxichem/paths.py`; one thread per math
  library is enough. Tested with Python 3.12, torch 2.8.0+cu126, NumPy 2.5.3, JAX 0.12.0.1.
- All data live outside git, under the data root (truth/, benches/, caches/, ...), found through `maxichem/paths.py`.
  Scripts write to named folders under it, never into tracked files. `data/`, `*.npz`, `*.npy`, `*.h5`, `*.pt` and logs
  are gitignored. A `sealed/` folder under the data root holds held-out test sets: nothing in this repository reads it,
  and every recursive command over the data root prunes it; never `git clean -x/-X/-fdx`, `git stash --all/-a`,
  `git add -f` under `data/` or `git status --ignored`.

## Engineering rules
- Smallest correct change; standard library first; no options nobody asked for; no silent fallbacks (a fallback is
  counted and reported).
- Numerics: keep the production settings above; a change of method, tolerance, pivoting, step control or the
  constant-P form (dx/dt = (S v - x sum(S v)) / n_tot) is a numerics change, checked against VULCAN, not a cleanup.
- The checks' references are stored (`paths.LIBREF`: outputs and rate coefficients of the torch reference solver on
  exactly the checks' inputs, with the current step control; `paths.AD_REF`: its torch-autograd derivatives;
  `paths.TRACER_REF`: the reduced-tracer layer's); `tests/libref.py` refuses inputs whose digest differs (network files,
  inputs, max_steps). A change of a check's inputs or of the numerics needs a new reference; the reference generators
  are kept outside this repository. Each check stops with a message naming missing reference files (`paths.need`).
- Checks:
  - After any change under `maxichem/`: `python tests/kincuda_check.py $VULCAN_THERMO` (parity against the stored
    reference (1e-6 dex; the 2,000-call chain allows one cell-call in 100,000 up to 1e-5) on yardstick cold calls,
    scrambled starts and a 2,000-call chain, the guard and recovery cases, the 300 K count); and
    `python bench/kintest.py quick --ref $CHEM_DATA/benches/kintest/quick_ref2.npz` (~2 min idle): 0 misses, no path's
    worst error changed by more than 0.01 dex.
  - After a change to `kincuda.py` or `kincuda_ad.py`: also `tests/kincuda_ad_check.py` (~2-6 min; reference
    derivatives in `paths.AD_REF`; the starting-step class is judged over 16 finite-difference step scales per cell,
    resolved errors only, plus an exact dot-product check of the step-size derivative); after a change to the JAX layer
    or the kernels: `tests/kinjax_check.py` (~4-7 min).
  - After a change to `kintracer.py`, a design file, `kincuda.py` or the reader: `python tests/kintracer_check.py`
    (~2 min; reference `paths.TRACER_REF`; parity aware of step-control branches, spill path, GCM outputs, batch size
    and order bitwise, 300 K). A new tracer design: `python bench/kintracer_gate.py --design <file>` (kill gate 2-5 min
    on a shared GPU; panels 10-35 min).
  - After a change to the generator or the reader: parity of one non-NCHO network (SNCHO, 4 warps per cell) against its
    stored reference (`benches/kincuda/libref_SNCHO.npz`; the driver is kept outside this repository). For a new
    network, its rounding drift sets drift.
  - A comment-only edit: compare syntax trees (`ast.dump`, docstrings stripped) instead of rerunning; a comment inside
    the CUDA templates changes the generated source: compare the compiled cubins.
- Public-ready: comments and docstrings say what the code does now; no dates, phase names, reviewer, agent or tool
  names, and no private paths. History goes in notes.md.
- Markdown files: README.md, CLAUDE.md and notes.md only.

## GPL boundary
- Nothing in this repository imports VULCAN, VULCAN-0D, the training-data generator (a 0-D wrapper of VULCAN) or
  mini-chem, directly or through a local import; those are GPL-licensed and kept outside it. This repository reads
  only their output files and VULCAN's reaction and NASA-9 data files, and does not redistribute them. Grep the import
  graph before adding any import.

## Git
- Commits as the author, Isaac Malsky: check `git config user.email` first and stop if it is not the author's. Plain
  English subject; no internal tags, tool names, Co-Authored-By or session lines. Stage explicit paths only, never
  `git add -A`. When two jobs may commit at once, wrap add + commit in one `flock` on a shared lock file.
- Push only when the author asks, after checking the author of every outgoing commit.
