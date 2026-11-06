# maxichem

Generic rules (code, docs, records, git, reviews, jobs) are in the global CLAUDE.md and ~/.claude/rules/. This file holds
only what is unique to this project.

## Goal
A fast chemistry solver for the pyharp/snapy GCM. It needs to cost on the order of a microsecond per cell, amortized
over a batch, stay stable when it runs autoregressively for long stretches, match VULCAN to within 0.1 dex at all times,
and work for every VULCAN network, not only NCHO. A feature that works only for NCHO is not finished yet. Wide,
realistic coverage beats polishing a narrow case; flag any rule or decision that over-focuses the project.

The code solves VULCAN networks as a physical 0-D solver in CUDA kernels, batched on the GPU (one call per cell per GCM
step). It also gives derivatives of every call (CUDA tangent and adjoint kernels). The reduced-tracer call (commit
399a347) and the JAX layer were removed; notes.md keeps their records. `README.md` is the public front page. The earlier
neural-network code is in github.com/imalsky/Autoregressive-Chemulator (tag `paper-v1` is the published paper's code).

## Every response
- Start every message to Isaac, status updates included, with "Isaac": a canary that this file is still in context.

## Start of every session
- Before any other work, run `git fetch origin` and `git status -sb`. If the copy is up to date, carry on. If it is
  behind and the working tree is clean, run `git pull --ff-only` and say so. If it has diverged or has uncommitted
  changes, stop and tell Isaac before touching anything.

## Records
- `notes.md` is the experiment log. Each session gets one dated entry with what was worked on, each experiment (what was
  run, its settings, the numbers, the conclusion), negative results, and decisions. A later entry supersedes an
  earlier one. Record numbers and conclusions, not raw output, and point to the experiment folder for the data.
- `todo.md` holds only the tasks Isaac explicitly put off and wants to come back to: one short entry each, with the
  date. Do not add tasks on your own. When a task is done, delete its line and record the result in `notes.md`.
- `experiments/` (gitignored, never pushed) holds experiment data: one folder per experiment,
  `experiments/YYYY-MM-DD_<name>/`, with its scripts, outputs and logs. Code in `maxichem/`, `tests/` and `bench/`
  never reads from it. A script that becomes part of the code moves into the repo; everything else stays in its
  experiment folder. Ask before deleting any experiment's data.

## Environment
This checkout sits in a project folder with its neighbours: `../.venv` (the Python environment: snapy, pyharp, kintera
and maxichem as an editable install), `../vulcan-truth` (the VULCAN-0D truth-data generator; GPL, outside this
repository), `../reference` (the torch reference generator) and `../snapy` (the GCM).
`VULCAN_THERMO` is VULCAN-JAX's `src/vulcan3/thermo` folder (the networks and `NASA9/`; no default). The two default
networks are its newest NCHO and newest sulfur network: `NCHO_photo_network.txt` (69 species, photolysis) and, whenever
sulfur is asked for, `SNCHO_photo_network_2025.txt` (91 species; Isaac, 2026-11-04). The older
`SNCHO_photo_network.txt` is not a default. `CHEM_DATA` is the data root (default `data/` in the checkout,
gitignored); it holds the truth sets, benches, the checks' references and the caches. `maxichem/paths.py` reads both.

## Running jobs
- GPU1 only (`CUDA_VISIBLE_DEVICES=1`). Use GPU0 only when Isaac asks, and leave it as soon as another user starts a
  job there.
- Runs of up to about 1 h need no approval. Estimate anything longer (sequential calls x measured ms per call) and ask.
- Tests are quick by default: minutes, enough to tell whether a change goes the right way. Long-run checks default to
  about 100 simulated days, and 1,000 days only at a final sign-off. A big test runs only at a decision point and takes
  at most 4-5 h; if it would take longer, shrink or batch it.
- Put a long job's log in the experiment folder.

## Decisions
- Within a plan Isaac has approved, make the research calls yourself without stopping to ask: take the recommended
  option at each choice and record it in notes.md. The goal, the domain, the GCM interface, and adopting a research
  result into production code are Isaac's calls.

## Judging results
- A miss is a scored point whose floored error against VULCAN is above 0.1 dex; a large miss is above 1 dex. The 12
  scored gases and their floors are in `bench/kintest.py`. A non-finite prediction is a failed step, never a zero error.
- Lead every report with the misses: the share of rows and of start states with any miss and with any large miss, and
  the worst error. Never call a model close to VULCAN on an averaged score alone; read the worst cases first.

## Figures
- Only when Isaac asks, following the scientific-plotting skill; the usual figure is VULCAN against model trajectories.
  Put every file in one flat folder and give one `scp -r` command that copies it to `~/Desktop` on Isaac's laptop.
  Plotting scripts stay in the experiment folder, never in `maxichem/`.

## GCM-side work (snapy)
Speed work on snapy happens in `../snapy` on branch `isaac/maxichem-coupling`, pushed to Isaac's fork. On this work
Isaac said (2026-11-04): don't ask; investigate, post findings and results in the Lumine thread, and keep working.
Still ask before an upstream PR (or marking one ready). The acceptance rules are in notes.md, "Decisions in force"
(2026-11-03 night and 2026-11-04): default path bit-for-bit, opt-in fused path at round-off, deterministic, the 0.01
dex bound for fp32 or cheaper schemes, C64x100 with 69 tracers for speed numbers; coupled tests on a Cartesian box
with a single passive-tracer convergence check, element totals and guard counts; the main path is fused CUDA called
from snapy's C++ loop (a Python driver only for tests); J from kintera. Development build: `../.venv-snapdev` (snapy built
from source with `../.cuda`); the shared `../.venv` keeps the released snapy wheel.

## Science and engineering rules
- Speed target: under about 1 us per cell-call in the GCM. Simple code beats small speed gains: a GPU-specific
  optimization that saves only a few hundredths or tenths of a microsecond but adds a lot of code is removed, not kept.
  Tuning for one GPU belongs in the GCM integration, not here. This rule covers added code. The global 1 % speed rule
  covers the choice between two forms of the same code.
- Every network Isaac uses must work, up to the largest (the sulfur network SNCHO_photo_network_2025, 91 species), even
  at some cost in speed.
  Reduce the advected tracers only if a clean, simple method exists; if the cost grows about linearly with the number
  of species and stays reasonable, carry all species rather than add code and special cases.
- Numerics: a change of method, tolerance, pivoting, step control or the constant-P form
  (dx/dt = (S v - x sum(S v)) / n_tot) is a numerics change. It has to be checked against VULCAN; it is never a cleanup.
  The method is in `docs/method.md`; the production settings are named constants.
- The checks' references are stored files: `paths.LIBREF` holds the torch reference solver's outputs on exactly the
  checks' inputs. `tests/libref.py` refuses inputs whose digest differs (network files, inputs, max_steps). A change of
  a check's inputs or of the numerics needs a new reference, made with `../reference/make_libref.py`. If a reference
  file is missing, the check stops with a message naming it (`paths.need`).
- GPL boundary: nothing here imports VULCAN-0D, vulcan-truth or mini-chem, directly or through a local import. maxichem
  reads only their output files and VULCAN's reaction and NASA-9 data files. Check the import graph before adding an
  import from those trees.
- Before proposing a method or speed idea, check whether it was tried, in `notes.md` ("Do not retry") and in
  `~/VULCAN_Project/VULCAN-JAX/notes.md` ("Efficiency program", "Ideas not adopted", "Do not retry"), and say so.

## Checks
Run the checks that cover a change once the change is finished, before committing it or saying it works: not after
every edit, and never on a GPU where a timing is running. `docs/testing.md` lists every test.
- Before a commit: `uvx pre-commit run --files <changed files>`.
- After any code change: `pytest -m "not slow and not gpu"` (CPU, under a minute).
- After a change under `maxichem/`: `pytest tests/test_parity.py tests/test_guard.py -m gpu -s` (with `VULCAN_THERMO` set).
  The bar is 1e-6 dex against the stored references (one cell-call in 100,000 may reach 1e-5). Also `pytest
  tests/test_vulcan_truth.py -s`: 0 misses against VULCAN, and no path's worst error may change by more than 0.01 dex.
- After a change to `maxichem/kincuda_ad.py` or the arithmetic it repeats: `pytest tests/test_ad.py -s`. Compare the
  failing cases with the last result in notes.md.
- After a change to the export, the kernels or `cpp/`: `pytest tests/test_export_cpp.py -s`: the C++ call must equal the
  Python call bit for bit, counts included. For a new network, its rounding drift sets `drift`.
- A comment-only edit needs no checks: compare syntax trees (docstrings stripped), or the compiled kernels for a comment
  inside `maxichem/kernels/`.

## Data safety
- Never run `git clean -x`, `-X` or `-fdx`, `git stash --all` / `-a`, or `git add -f` under `data/`. The ignored
  `data/` and `experiments/` folders hold the truth sets, the checks' references (made outside this repository) and
  the experiment data.
- Before deleting a cache or data folder, grep `maxichem/paths.py` and every module for its path: a code default can
  depend on it even when no config names it.
