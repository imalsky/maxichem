# maxichem

## Goal
A fast chemistry solver for the pyharp/snapy GCM. It needs to cost on the order of a microsecond per cell, amortized
over a batch, stay stable when it runs autoregressively for long stretches, match VULCAN to within 0.1 dex at all times,
and work for every VULCAN network, not only NCHO. A feature that works only for NCHO is not finished yet.

The code solves VULCAN networks as a physical 0-D solver in CUDA kernels, batched on the GPU (one call per cell per GCM
step). It also gives derivatives of every call, for learning where learning helps, and has a reduced-tracer call.
`README.md` is the public front page. The earlier neural-network code is in github.com/imalsky/Autoregressive-Chemulator
(tag `paper-v1` is the published paper's code).

## Every response
- Start every response with "Isaac". This is a canary: if the name is missing, this file has dropped out of context.

## Start of every session
- Before any other work, run `git fetch origin` and `git status -sb`. If the copy is up to date, carry on. If it is
  behind and the working tree is clean, run `git pull --ff-only` and say so. If it has diverged or has uncommitted
  changes, stop and tell Isaac before touching anything.

## Records
Keep the records organized and short. Do not add files, folders or text that nobody will read.
- `notes.md` is the experiment log. Each session gets one dated entry with what was worked on, each experiment (what was
  run, its settings, the numbers, the conclusion), negative results, and decisions. A later entry supersedes an
  earlier one. Record numbers and conclusions, not raw output, and point to the experiment folder for the data.
- `todo.md` holds only the tasks Isaac explicitly put off and wants to come back to: one line each, with the date. Do
  not add tasks on your own. When a task is done, delete its line and record the result in `notes.md`.
- `experiments/` (gitignored, never pushed) holds experiment data: one folder per experiment,
  `experiments/YYYY-MM-DD_<name>/`, with its scripts, outputs and logs. Code in `maxichem/`, `tests/` and `bench/`
  never reads from it. A script that becomes part of the code moves into the repo; everything else stays in its
  experiment folder. Ask before deleting any experiment's data.
- The only Markdown files are README.md, CLAUDE.md, notes.md and todo.md. Never leave scratch files at the top level or
  in tracked folders.

## Environment
`VULCAN_THERMO` is the folder with VULCAN's `NCHO_thermo_network.txt` and `NASA9/` (no default). `CHEM_DATA` is the
data root (default `data/` in the checkout, gitignored); it holds the truth sets, benches, the checks' references and
the caches. `maxichem/paths.py` reads both.

## Engineering rules
- Make the smallest correct change and use the standard library first. Add no options nobody asked for, and no silent
  fallbacks: any fallback is counted and reported.
- Numerics: keep the production settings in `maxichem/kincuda.py`'s module docstring. A change of method, tolerance,
  pivoting, step control or the constant-P form (dx/dt = (S v - x sum(S v)) / n_tot) is a numerics change. It has to be
  checked against VULCAN; it is never a cleanup.
- The checks' references are stored files: `paths.LIBREF` holds the torch reference solver's outputs and rate
  coefficients on exactly the checks' inputs, `paths.AD_REF` its torch-autograd derivatives, and `paths.TRACER_REF` the
  reduced-tracer layer's outputs. `tests/libref.py` refuses inputs whose digest differs (network files, inputs,
  max_steps). A change of a check's inputs or of the numerics needs a new reference. The reference generators are kept
  outside this repository. If a reference file is missing, the check stops with a message naming it (`paths.need`).
- Checks:
  - After any change under `maxichem/`, run `python tests/kincuda_check.py $VULCAN_THERMO`. It tests parity against the
    stored reference to 1e-6 dex (the 2,000-call chain allows one cell-call in 100,000 up to 1e-5) on yardstick cold
    calls, scrambled starts and a 2,000-call chain, plus the guard and recovery cases and the 300 K count. Also run
    `python bench/kintest.py quick --ref $CHEM_DATA/benches/kintest/quick_ref2.npz` (~2 min on an idle GPU). It must
    show 0 misses, and no path's worst error may change by more than 0.01 dex.
  - After a change to `kincuda.py` or `kincuda_ad.py`, also run `tests/kincuda_ad_check.py` (~2-6 min). After a change
    to the JAX layer or the kernels, run `tests/kinjax_check.py` (~4-7 min).
  - After a change to `kintracer.py`, a design file, `kincuda.py` or the reader, run `python tests/kintracer_check.py`
    (~2 min). For a new tracer design, run `python bench/kintracer_gate.py --design <file>` (the kill gate takes 2-5 min
    on a shared GPU; the panels take 10-35 min).
  - After a change to the generator or the reader, check the parity of one non-NCHO network (SNCHO, 4 warps per cell)
    against its stored reference (`benches/kincuda/libref_SNCHO.npz`; the driver is kept outside this repository). For
    a new network, its rounding drift sets `drift`.
  - For a comment-only edit, compare syntax trees (`ast.dump`, with docstrings stripped) instead of rerunning the checks.
    A comment inside the CUDA templates changes the generated source, so compare the compiled cubins instead.
- Keep the code public-ready. Comments and docstrings describe what the code does now: no dates, no phase names, no
  reviewer, agent or tool names, and no private paths. History goes in notes.md.

## Git
- Commit as the author, Isaac Malsky: check `git config user.email` first, and stop if it is not the author's. Write
  the subject in plain English, with no internal tags, tool names, Co-Authored-By or session lines. Stage explicit
  paths only, never `git add -A`. When two jobs may commit at once, wrap the add and the commit in one `flock` on a
  shared lock file.
- Push only when the author asks, after checking the author of every outgoing commit.
