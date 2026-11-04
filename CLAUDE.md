# maxichem

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
Keep the records organized and short. Do not add files, folders or text that nobody will read.
- `notes.md` is the experiment log. Each session gets one dated entry with what was worked on, each experiment (what was
  run, its settings, the numbers, the conclusion), negative results, and decisions. A later entry supersedes an
  earlier one. Record numbers and conclusions, not raw output, and point to the experiment folder for the data. It is
  tracked and the repository is public: technical facts only, with no tool, model or agent names and no private paths.
  Take every logged time from a fresh `date` call.
- `todo.md` holds only the tasks Isaac explicitly put off and wants to come back to: one short entry each, with the
  date. Do not add tasks on your own. When a task is done, delete its line and record the result in `notes.md`.
- `experiments/` (gitignored, never pushed) holds experiment data: one folder per experiment,
  `experiments/YYYY-MM-DD_<name>/`, with its scripts, outputs and logs. Code in `maxichem/`, `tests/` and `bench/`
  never reads from it. A script that becomes part of the code moves into the repo; everything else stays in its
  experiment folder. Ask before deleting any experiment's data.
- The only Markdown files are README.md, CLAUDE.md, notes.md and todo.md. Never leave scratch files at the top level or
  in tracked folders.

## Environment
This checkout sits in a project folder with its neighbours: `../.venv` (the Python environment: snapy, pyharp, kintera
and maxichem as an editable install), `../vulcan-truth` (the VULCAN-0D truth-data generator; GPL, outside this
repository), `../reference` (the torch reference generator and the sulfur parity driver) and `../snapy` (the GCM).
`VULCAN_THERMO` is VULCAN-JAX's `src/vulcan3/thermo` folder (the networks and `NASA9/`; no default). The default network
is its `NCHO_photo_network.txt` (69 species, photolysis). `CHEM_DATA` is the data root (default `data/` in the checkout,
gitignored); it holds the truth sets, benches, the checks' references and the caches. `maxichem/paths.py` reads both.

## Running jobs
- GPU1 only (`CUDA_VISIBLE_DEVICES=1`). Use GPU0 only when Isaac asks, and leave it as soon as another user starts a
  job there.
- Time only with nothing else on the GPU, Isaac's other jobs and the checks included.
- Runs of up to about 1 h need no approval. Estimate anything longer (sequential calls x measured ms per call) and ask.
- Tests are quick by default: minutes, enough to tell whether a change goes the right way. Long-run checks default to
  about 100 simulated days, and 1,000 days only at a final sign-off. A big test runs only at a decision point and takes
  at most 4-5 h; if it would take longer, shrink or batch it.
- Launch long jobs as user units, not with nohup or setsid, which die with the tmux session. Put the log in the
  experiment folder:

      systemd-run --user --collect --unit=<name> --working-directory=<dir> -E TMPDIR=$TMPDIR \
        -p StandardOutput=append:<log> -p StandardError=append:<log> /bin/bash <driver.sh>

  Check a unit with `systemctl --user is-active <name>`, not `pgrep -f`. Never edit a running script, and never say a
  step has started until a command has launched it.

## Decisions and reviews
- Within a plan Isaac has approved, make the research calls yourself without stopping to ask: take the recommended
  option at each choice and record it in notes.md. The goal, the domain, the GCM interface, and adopting a research
  result into production code are Isaac's calls.
- Before a numerics change, or a test design that a decision rests on, get one Codex review of the plan (codex skill,
  gpt-6.1-sol at xhigh unless Isaac names another, read-only, run in the background), and one of the results. On a
  usage-limit error, do not retry: use an independent Claude reviewer instead and tell Isaac which review is missing.
- Plan any multi-part job first and get the plan reviewed by Codex (gpt-6.1-sol, high) before starting. Also get a
  Codex review (sol, high) of complicated new code and of every removal of code or features.

## Judging results
- A miss is a scored point whose floored error against VULCAN is above 0.1 dex; a large miss is above 1 dex. The 12
  scored gases and their floors are in `bench/kintest.py`. A non-finite prediction is a failed step, never a zero error.
- Lead every report with the misses: the share of rows and of start states with any miss and with any large miss, and
  the worst error. Never call a model close to VULCAN on an averaged score alone; read the worst cases first.

## Figures
- Only when Isaac asks, following the scientific-plotting skill; the usual figure is VULCAN against model trajectories.
  Put every file in one flat folder and give one `scp -r` command that copies it to `~/Desktop` on Isaac's laptop.
  Plotting scripts stay in the experiment folder, never in `maxichem/`.

## Engineering rules
- Add no options nobody asked for, and no silent fallbacks: any fallback is counted and reported.
- Speed target: under about 1 us per cell-call in the GCM. Simple code beats small speed gains: a GPU-specific
  optimization that saves only a few hundredths or tenths of a microsecond but adds a lot of code is removed, not kept.
  Tuning for one GPU belongs in the GCM integration, not here.
- Every network Isaac uses must work, up to the largest (the sulfur network, 88 species), even at some cost in speed.
  Reduce the advected tracers only if a clean, simple method exists; if the cost grows about linearly with the number
  of species and stays reasonable, carry all species rather than add code and special cases.
- Numerics: keep the production settings in `maxichem/kincuda.py`'s module docstring. A change of method, tolerance,
  pivoting, step control or the constant-P form (dx/dt = (S v - x sum(S v)) / n_tot) is a numerics change. It has to be
  checked against VULCAN; it is never a cleanup.
- The checks' references are stored files: `paths.LIBREF` holds the torch reference solver's outputs on exactly the
  checks' inputs. `tests/libref.py` refuses inputs whose digest differs (network files, inputs, max_steps). A change of
  a check's inputs or of the numerics needs a new reference, made with `../reference/make_libref.py`. If a reference
  file is missing, the check stops with a message naming it (`paths.need`).
- Keep the code public-ready. Comments and docstrings describe what the code does now: no dates, no phase names, no
  reviewer, agent or tool names, and no private paths. History goes in notes.md.
- GPL boundary: nothing here imports VULCAN-0D, vulcan-truth or mini-chem, directly or through a local import. maxichem
  reads only their output files and VULCAN's reaction and NASA-9 data files. Check the import graph before adding an
  import from those trees.
- Before proposing a method or speed idea, check whether it was tried, in `notes.md` ("Do not retry") and in
  `~/VULCAN_Project/VULCAN-JAX/notes.md` ("Efficiency program", "Ideas not adopted", "Do not retry"), and say so.

## Checks
Run the checks that cover a change once the change is finished, before committing it or saying it works: not after
every edit, and never on a GPU where a timing is running. "Every check" means all of the following.
- After a change under `maxichem/`: `python tests/kincuda_check.py $VULCAN_THERMO` (NCHO_photo; ~1 min). It tests
  parity against the stored reference (`paths.LIBREF`) to 1e-6 dex (one cell-call in 100,000 may reach 1e-5) on
  recorded cold starts, scrambled starts, a 2,000-call chain and the photolysis cases (lit calls,
  a lit chain, lit recovery, bad J), plus the guard and recovery cases, the 300 K count and batch size and order. Also
  `python bench/kintest.py quick --ref $CHEM_DATA/benches/kintest/<latest saved quick run>`: it must show 0 misses
  against VULCAN, and no path's worst error may change by more than 0.01 dex.
- After a change to the generated kernels or the reader, also the parity of the sulfur network (SNCHO_photo, 88
  species, 4 warps per cell) against `benches/kincuda/libref7_SNCHO.npz`: `python ../reference/check_nets2.py <this
  checkout>`.
  For a new network, its rounding drift sets `drift`.
- After a change to `maxichem/kincuda_ad.py` (or to the call arithmetic it repeats): `python tests/kincuda_ad_check.py
  <network>` on NCHO_photo, SNCHO_photo and CHO_photo (primal bitwise CudaSolver's, tangents against finite
  differences, JVP / VJP dot products, recovered cells, determinism; exit status 1 on a failure). Compare the failing
  cases with the last result in notes.md.
- For a comment-only edit, compare syntax trees (`ast.dump`, with docstrings stripped) instead of rerunning the checks.
  A comment inside the CUDA templates changes the generated source, so compare the compiled cubins instead.

## Git
- Commit as the author, Isaac Malsky. Write the subject in plain English, with no internal tags, tool names,
  Co-Authored-By or session lines. Stage explicit paths only, never `git add -A`. When two jobs may commit at once, wrap
  the add and the commit in one `flock` on a shared lock file.
- Never run `git clean -x`, `-X` or `-fdx`, `git stash --all` / `-a`, or `git add -f` under `data/`. The ignored
  `data/` and `experiments/` folders hold the truth sets, the checks' references (made outside this repository) and
  the experiment data.
- Before deleting a cache or data folder, grep `maxichem/paths.py` and every module for its path: a code default can
  depend on it even when no config names it.
