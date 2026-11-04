# Notes

The experiment log of maxichem: results, negative results and the author's decisions. A later entry supersedes an
earlier one.

Pared down on 2026-11-03 from the full log (4,632 lines, 2026-06 to 2026-11-03). The full text is in git history (commit
e1847cb and earlier). Removed then: the neural-network emulator's record, data lineage and conventions, and text that
later results superseded. Kept: every number that still matters, every negative result, the decisions in force.

Conventions. Paths are relative to the repository. `experiments/YYYY-MM-DD_<name>/` holds one experiment's data
(gitignored, not pushed). A campaign cited by name only (2026-10-30_kinetic-solver, 2026-10-31_fresh-start,
2026-10-31_data, 2026-10-31_broaden, 2026-10-31_mltracers, 2026-10-31_refresh) was run in the retired emulator project
and is archived with it. "Retired" marks an experiment folder that was not kept. "53-species network" = VULCAN-0D's
NCHO_thermo network (53 species, 290 reversible reactions), the default until 2026-11-03; results on it are labelled.
Speeds are us per cell-call (one cell, one call, amortized over the batch) on an idle RTX 6000 Ada unless stated. This
card runs fp64 at 1/64 of its fp32 speed. "(inferred)" marks a claim not read from data.

## Current state (2026-11-03)

### Code
- maxichem solves a VULCAN-format network as a batched 0-D solver: one call per cell per GCM step. The whole call
  (rates, every step, the step control) runs in one CUDA kernel, generated from the network and compiled at run time
  (NVRTC). Persistent warps take cells from a queue; no host loop and no host sync.
- Default network: VULCAN-JAX's NCHO_photo (69 species, 390 thermal reactions, 48 photolysis branches). All 11
  VULCAN-JAX networks load and run (4,096 cells, 21 calls: 0 unfinished, 0 failed, element shares within 3.8e-11). The
  generator caps a network at 254 species. The sulfur network SNCHO_photo (88 species) runs with 4 warps per cell.
- Production numerics: the module docstring of maxichem/kincuda.py holds every setting. In short: Rodas3; a sparse LU
  without pivoting in a minimum-degree order; rtol 1e-3; atol 1e-25 (kincuda.ATOL); the step size carried between calls;
  the first try judged by its worst species; the V3F step rule; the collision limit on rates; a guard and a device-side
  recovery (40,000 tries). Why each: the topics below.
- Inputs per cell: mole fractions, T, P, dt, the carried step size and, with photo=True, J per photolysis branch (s^-1,
  fixed within a call). A negative or non-finite J makes the cell a bad input: not stepped, counted. Gas below 300 K is
  a hard error. atol is an argument of the call.
- Removed on 2026-11-03: the reduced-tracer call (code at commit 399a347; the GCM now advects every species), the
  one-cell-per-block launch path for small batches, the JAX layer (the derivatives stay in the CUDA kernels). The
  pure-torch solver is only the checks' reference:
  the checks read its outputs, stored once on their exact inputs.

### Accuracy against VULCAN (atol 1e-25; experiments/2026-11-03_atol)
- NCHO_photo on all 131,072 boxes of data/truth_v2/boxes_ncho (half lit, 0 to 1e8 s): 6,160,384 states, 0 misses, worst
  0.0662 dex. 4.7 steps per call (3.0 at atol 1e-16). Guard 0.
- NCHO_photo on the 904 qual_ncho paths (kintest quick): 0 misses, worst 0.004 dex. Day-night set, 64 paths: 0 misses,
  worst 0.0023 dex.
- SNCHO_photo on boxes_sncho2 (16 gases; the 1,955 boxes with a VULCAN result): 91,885 states, 7 with a miss, all large
  (0.0076 %), all in box 1939. Worst 4.342 dex; dark worst 0.029 dex. 5.32 steps per call. Guard: 3 recomputed, all
  recovered. Box 1939 is a VULCAN error (see "VULCAN reference errors"). The score leaves out the 93 boxes without a
  VULCAN result (6.1 % of lit boxes, 3.0 % of dark), so it under-represents lit gas.
- Hard cold lit cells (below 325 K; 2,280 NCHO, 142 SNCHO; dark to 1e8 s, then lit): none unfinished, non-finite or
  still failed. First lit call, steps median / max: NCHO 106 / 201, SNCHO 40 / 169.
- On the 53-species network (2026-10-30 to 11-02): 0 misses on the emulator's decision panels, the qual set, QT2 and the
  mini-chem yardstick (after its VULCAN reference was corrected). QT's only 2 misses, in its every-call block, were
  removed by the 2026-11-01 step-control fix. Details under "Qualification on the 53-species network".

### Speed (experiments/2026-11-03_atol)
- Settled gas, about one step per call, 1,382,400 cells, dark / lit: NCHO_photo 0.354 / 0.349 us, SNCHO_photo 0.658 /
  0.635 us.
- Changing gas costs more, and atol 1e-25 costs more than 1e-16 there. kintest quick against the atol-1e-16 run:
  1.43-1.47x. A moving-gas bed (each call after 1 % of a nearby box's gas is mixed in): NCHO 7.493 us and 27.15 steps
  per call (3.890 and 13.73 at 1e-16); SNCHO 16.995 us (9.231 at 1e-16).
- The cost in a coupled GCM run on the current network and atol is not measured. The coupled run on the 53-species
  network stayed near one step per call (see "GCM coupling").

### Derivatives
- maxichem/kincuda_ad.py: CUDA tangent (JVP) and adjoint (VJP) kernels of the whole call. They cover every VULCAN-JAX
  network up to 96 species (64 leading rows) and 2,047 reactions; explicit reverse laws are refused. Default contract A:
  the step control differentiated on the realised branch.
- Check: tests/kincuda_ad_check.py (until 2026-11-03 experiments/2026-11-03_ad/dev_check2.py). Latest results:
  NCHO_photo 101 of 101, CHO_photo 101 of 101, SNCHO_photo 99 of 101. The two failures are known: the J tangent on box
  starts at dt 500 s, p90 2.4e-4 and 1.4e-4 against the 1e-4 bar.
- The old torch-autograd reference (refdata3, paths.AD_REF) is retired: it replays the old step control. No independent
  autograd reference exists for the current networks. There is no JAX front end any more.

### Open items
- Sulfur: 68 boxes of boxes_sncho2 have no reference (60 need a VULCAN donor run; 8 Radau solves timed out). VULCAN's
  clamp bound (nega_cut -1) is untested.
- The cost of a coupled run on the current network and atol. An independent review recommended atol 1e-25 for sulfur
  networks and 1e-16 for NCHO until such a run; the default is 1e-25.
- The tangent at exactly J = 0 is not verified at the default atol.
- A 1,000-day sign-off of the current code. The last 1,000-day runs were on the 53-species network (2026-11-01).
- A GH200 timing was never run. A timing package with the 2026-10-31 kernel exists in experiments/2026-10-31_gpu-core.
- The drift survey that sets a new network's element-drift allowance no longer runs (it imports a removed module; found
  2026-10-31).
- From the 2026-10-31 survey, not rechecked since: no upper-T guard; third-body efficiencies fixed to H2 (high
  metallicity would need composition weights); the reverse-rate exponent is unclipped.
- A C or Fortran caller needs a C launcher (the kernels are built through NVRTC from Python tables).
- Old SNCHO network, not rerun on the current one: a scrambled-start stress (2026-11-02, atol 1e-16) had local errors up
  to 1.31 dex (5 cell-calls > 0.1 dex, the clamp mechanism of "Photolysis"). A few cold, thin night-side cells (~330 K,
  3e-4 bar; 2026-11-01) changed by up to 4.8 dex for a 1e-10 relative change of the start; chemistry or numerics is open
  (a VULCAN run of such a cell would settle it). The guard fired often on new planets' starts then; SNCHO may need its
  own drift allowance.
- The QN truth set (quench / reheat, out-of-equilibrium mixing partners, tight reruns) was designed on 2026-10-31 and
  paused (experiments/2026-10-31_qn).
- Deferred tasks: todo.md.

## Scoring and truth data

### Miss rule
- A miss is a scored point whose floored error against VULCAN is above 0.1 dex; a large miss is above 1 dex. Error =
  |max(pred, S) - max(truth, S)| in log10, so an invented gas counts by how far it rises above its floor S. 1e-30 is
  only a storage floor.
- A non-finite or negative prediction on a scored point is a failure, never a zero error. Negative or non-finite truth
  is a failure too.
- Scored gases and floors S (bench/kintest.py): 1e-10 for H2O, CO, CO2, CH4, NH3, HCN, C2H2; 1e-8 for H, OH, O; 1e-6 for
  H2, N2. Sulfur scores add H2S, SO2, COS, S2 at 1e-10 (16 gases).
- The rule of 2026-10-11 also required a number density >= 1e-2 cm^-3 (VULCAN's default atol). bench/kintest.py applies
  the floors only.
- Why these floors: absorbers matter for spectra and heating from about 1e-7. JWST detects C2H2, C2H4, CH4 from about
  1e-7 to 1e-6 (Gasman et al. 2022). Hot-Jupiter retrievals reach about 1e-6, the best clear case about 1e-8 (Greene et
  al. 2016). In a coupled hot-Jupiter GCM, species below about 1e-7 add little opacity, and heating changes come mainly
  from CH4 (Drummond et al. 2020). S keeps a margin of about 1000x for gas carried between layers.
- A flat 1e-12 floor (2026-10-08 to 10-11) was dropped: its largest misses (9-12 dex) were traces crossing it.

### Truth generator
- The truth data come from the VULCAN-0D truth generator (vulcan-truth; outside this repository; GPL). It runs VULCAN-0D
  b3ccf4f on VULCAN-JAX's networks and NASA-9 data, with VULCAN-JAX's step-control corrections: the error estimate
  counts a species the step creates; a value below nega_cut rejects the step; a retry restores ymix and the element
  losses; the dt_min fallback keeps the state. It also has the CH3OH broadening, the collision limit and k_B =
  1.380649e-16. Its fast path is bitwise equal to the corrected stock path (20 boxes).
- Settings (experiments/2026-11-03_truth-gen): rtol 5e-4; first step 1e-18 s; dt_min 1e-24 s; step cap 2e5, with reruns
  at 2e6 or 2e7 where needed.
- Absolute tolerance: VULCAN's atol is a number-density cutoff; species below it leave the error control. The default
  1e-2 cm^-3 is a mole fraction near 1e-16 at 1e-5 bar. The sulfur truth uses atol = XATOL n_tot per run, XATOL 1e-28,
  donor runs included, count_max 2e6. The NCHO boxes keep VULCAN's default: 256 of them (192 lit below 800 K and 1e-2
  bar, 64 random) rerun at XATOL 1e-28 differ by at most 0.0030 dex (median 0.0001; 1 stopped at the cap).
- Publication is atomic: the manifest lists the shards and replaces the old one in one rename. A paths worker exits 1 on
  an error and resumes only an output whose job and build digests match.
- Rates agree across codes: VULCAN-JAX against the torch reference within 1e-6 relative (96 T, P points); the generator
  against the torch reference within 8.6e-14 (NCHO_photo) and 8.5e-14 (SNCHO_photo).
- The generator's SNCHO network has a condensation line S8 -> S8_l_s that maxichem skips. It never acts in the boxes
  (S8_l_s stays 0).

### Truth sets in use (data/truth_v2/)
- boxes_ncho: a scrambled Sobol design of 131,072 boxes. T 300-3000 K and P 1e-5-100 bar log-uniform; metallicity
  0.1-300x; C/O 0.1-2. Half lit, with the J of a 1-D model layer times 10^U(-3, 1). Starts: unreacted gas; a reheated,
  mixed or scrambled 1e8-s state; or 10 % of H2 split into H. 48 stored states from 0 to 1e8 s. All accepted (worst
  element drift 9.9e-8).
- qual_ncho: the old qual panels' recorded forcing and mixes, replayed on the new network. 904 paths: QC 512 x 1,000
  calls, QCs 64, QC twins 128, QH 64 x up to 1,000 holds, QHs 32, QH twins 32, QL 64 x up to 3,500 holds, QLc 8 x up to
  3,000 calls. 0 errors.
- daynight_ncho: 64 QC paths x 1,000 calls with J = J_ref s max(0, sin(2 pi t / 1 day + phase)).
- boxes_sncho2 (the sulfur truth): the first 2,048 rows of boxes_sncho's design (a balanced Sobol subset, same box
  numbers), at XATOL 1e-28. 1,955 accepted, 92 stopped at the 2e6-step cap, 1 quarantined (box 810), worst drift 1.8e-8.
  50,922 CPU s; median 7,188 steps. Boxes without a result: dark 3.0 %, lit 6.1 %, 1200-2000 K 7.5 %, lit below 800 K
  6.6 %.
- Superseded: boxes_sncho (8,192 boxes at VULCAN's default atol; on 991 unreacted starts it differs from boxes_sncho2
  by more than 0.1 dex in 18 boxes and more than 1 dex in 16, worst 6.702 dex); boxes_sncho_atol (partial; lit boxes
  below 800 K at VULCAN atol 1e-10 cm^-3). The 53-species sets moved to old_network/.
- kintest quick runs the first 32,768 boxes, the qual_ncho paths and panel P (256 paths; no truth, stability counters
  only). It refuses a set whose manifest lists boxes not run, failed runs, bad states or element drift, or whose shards
  do not hold every design box exactly once. b_set and q_set are digests of the scored truth and forcing; a paired run
  refuses a reference with other digests. It exits 1 on a miss, an unfinished call, a non-finite or negative state, or a
  still-failed cell.

### Check references (data/benches/)
- kincuda/libref7.npz: tests/kincuda_check.py on NCHO_photo at atol 1e-25 (28 of 28). kincuda/libref7_SNCHO.npz: the
  sulfur parity (15 of 15). kintest/quick_ref6.npz: the kintest quick reference. Earlier references belong to superseded
  numerics.

### The 53-species truth sets
- qual (2026-10-31_data; 2,466 files, 65.3 CPU-h): QC 512 per-call GCM paths x 10,000 calls (128 near-solar) + 128 kick
  twins; QCs 64 at C/O 1-2; QL 64 paths of 1,000 days in 57,600 holds of 1500 s; QLc 8 per-call twins of QL (172,800
  calls of 500 s); QH 1,024 cold-hold / reheat histories + 512 kick twins; QHs 128 at C/O 1-2. Its 26 tolerance reruns
  (rtol 5e-4 against 5e-3) all pass, worst 0.018 dex. QL (T, P per hold) and QLc (per call) differ by up to 1.9 dex on
  hot mixing paths (from the campaign log, not re-checked), so a model runs with the forcing its truth used.
- QT (2026-10-31_refresh; 2,320 GCM-like per-call paths, 14.04 M states): Z 0.1-300, C/O 0.1-1 (main) and 1-2 (side),
  He/H 0.078-0.100; mixing by mass with a per-mix element ledger. Blocks: every-call mixing (QTe), QC's mixing law
  (QTq), cold holds with reheats (QTc), far-from-equilibrium donors (QTo), C/O 1-2 (QTs), unmixed (QTu). Tight reruns 26
  / 26 pass (worst 0.014 dex).
- QT2 (2026-10-31_refresh): QT's recipe with new seeds and a disjoint donor pool; 1,128 paths, 6.44 M states; tight
  reruns 26 / 26 (worst 0.0072 dex). QT and QT2 have both been used as judges: a new tracer design needs fresh judge
  data.
- The mini-chem yardstick (experiments/2026-10-19_minichem-yardstick): 395,114 segment rows from VULCAN restart states,
  10 composition cells, up to 2,000 calls of dt 1-500 s. Its reference for one start was corrected (see "VULCAN
  reference errors").

## Solver findings

### Formulation
- The reactor has constant T and P. VULCAN-0D resets the total density to P / kT after every internal step. The solver
  integrates the constant-P form dx/dt = (S v - x sum(S v)) / n_tot. Against mini-chem at rtol 1e-8 with 401 log-spaced
  calls to 1e8 s, the six Fig. 4 cases agree to <= 0.0008 dex; with 41 calls, <= 0.009 dex (the per-call
  renormalization) (2026-10-30_kinetic-solver).
- The reader takes VULCAN's reaction list and NASA-9 files as data: two-body, three-body falloff, low-pressure-only and
  the CH3OH falloff, Troe and efficiency lines, irreversible sections, photolysis lines. It refuses unbalanced reactions
  and lists duplicates (their rates add, as in VULCAN). Not supported: remove_list (VULCAN switches reaction directions
  off by id).
- What a full VULCAN-0D box needs (2026-11-02 gap analysis, experiments/2026-11-02_photo): thermal kinetics and
  photolysis are done. Transport does not exist in a 0-D box; it is the GCM's job. Condensation in VULCAN-0D is a test
  stub (no shipped network has a condensation section), so clouds belong to a GCM cloud scheme (not built). VULCAN-0D
  has no ion chemistry.
- A formula-parser bug (an element appearing twice in a formula counted once, e.g. CH3OH read with one H) was found and
  fixed on 2026-10-30.
- The solver does not snap to equilibrium (experiments/2026-10-31_gpu-core, 53-species network). About half of all
  scored VULCAN states are > 1 dex from local equilibrium and a quarter > 3 dex, mostly quenched gas below 1500 K. On
  the 3.46 M far qual states: 0 misses, worst 0.020 dex. Of 9.26 M far states, the solver put none within 0.5 dex of
  equilibrium where VULCAN stayed > 1 dex away.
- Domain stress (53-species network, 2026-10-30): 300-3000 K, 1e-5-100 bar, scrambles <= 1 dex: 31,600 solves, 0
  failures, element drift <= 2.3e-10 dex. Outside (below 300 K, 2-3 dex scrambles, random gas): 0 non-finite or negative
  states, but up to ~6 % of calls slower than 20,000 steps and drift up to ~1e-4 dex.
- Long histories (53-species network, 2026-10-31): panel P's 256 paths cycled with their mixes, dt 500 s, reached 520
  days (89,900 calls). 0 non-finite and 0 negative states; element drift on unmixed paths <= 8.8e-8 dex; 1.08-1.11 steps
  per call. Unfinished calls were counted on every 200th call only (0).

### Qualification on the 53-species network (2026-10-30 to 11-02)
- The emulator's decision panels (Rodas3): S 157,232 rows, Sraw 39,306, side 15,704, L 2,048 rows of 172,800 calls, P
  256 paths x 10,000 calls: 0 misses, worst 0.02 dex, unfinished 0. On the same panels the best emulator (ov_DT2) missed
  on 0.944 % of S rows, 31.4 % of L and 35.5 % of P.
- qual: QC + QCs + QC twins, 6,896,743 scored calls with 138,480 recorded mixes: 0 misses; worst 0.018 dex (QC), 0.064
  (QCs, CO2 at 1348 K), 0.015 (twins); 1.23 steps per call. QH + twins + QHs, 238,894 states: 0 misses, worst 0.017 dex.
  QL and QLc to 1,000 days: 0 misses, worst 0.019 / 0.017 dex.
- The 1,000-day tier at a 25-s clock was stopped at tick 1.8 M of 3.456 M (520 days): 0 paths with a miss, worst 0.064
  dex; unfinished, non-finite and guard counts 0.
- QT: 13,503,214 states of 2,304 mixing paths: 2 with a miss (one step-control event, fixed: see "Integrator and step
  control"). p99 0.0010-0.0032 dex in every block. After the fix the every-call block has 0 misses, worst 0.015 dex.
- QT2: 0 misses; worst 0.052 dex before the step fix (CO2 right after a composition switch, the same weak spot), 0.033
  with it.
- The mini-chem yardstick, corrected: 395,114 rows, 0 misses, worst 0.035 dex. Mini-chem on the same rows: 15.21 % rows
  with a miss, 5.99 % large (calls 1-200); 17.62 % / 6.87 % over all calls up to 2,000.
- The mini-chem paper's six Fig. 4 boxes (experiments/2026-11-02_figs): worst floored error over six gases for t >= 1e-3
  s: maxichem 0.0003-0.0149 dex, mini-chem 0.21-5.3 dex.
- Old SNCHO network (88 species, 514 thermal reactions; experiments/2026-10-31_sncho): 8 boxes and 19 GCM-like
  histories: 0 misses on 16 gases, worst 0.020 dex. At the GCM call pattern (dt 25 s x 40,000 calls, dt 500 s x 20,000)
  worst 0.0176 / 0.0130 dex. Sulfur shifts HCN, CH4 and H by up to 2-3 dex in transients; the solver matches VULCAN's
  shift within 0.006 dex.

### Integrator and step control
- Rodas3 (4 stages, order 3, L-stable, stiffly accurate) replaced a two-stage ROS2 on 2026-10-31. ROS2's error estimate
  rejects steps after any T change above ~2e-4 per call, so GCM-like paths cost 13.7 steps per call; Rodas3 needs
  1.1-1.2. GCM-like forcing, 65,536 cells, worst error against a tight reference over 40 calls (torch library): ROS2
  6.17 us (2.32 steps, 0.012 dex); Rodas3 3.35 us (1.13 steps, 0.025 dex); Rodas3 with an unpivoted LU 2.71 us (same
  answers).
- The whole-call kernel (2026-10-31) gives the torch library's answers (identical step counts; <= 7e-8 dex on cold
  calls, 1.5e-7 dex over a 2,000-call chain) at 9.5-13.3x its speed (768-262,144 cells).
- The first-step rule (2026-11-01; 2026-10-31_refresh). The carried step is often longer than the call, so a call after
  an abrupt input change is one Rodas3 step. Its error test was an RMS over all species, so one fast-relaxing gas could
  be far off while the mean passed. This gave the first real miss of the full mechanism: QT, CO2 0.163 dex at call 5001
  of the twin paths 4243000312 / 313 (1569 K, 1.3 bar, dt 4.8 s; carried step 10.3 s). VULCAN reruns 10x and 100x
  tighter agreed with the stored truth. How often: QC 2 of 5.76 M calls > 0.03 dex; stress mixes 52 single-step errors
  in 244 k calls, worst 0.80 dex.
- Fix adopted (d1): until a call has accepted a step, err = max(RMS, max-norm). It removes every case. Cost: 1.11-1.45x
  the steps on most panels (QC 1.18x, panel P 1.14x, QT every-call 1.42x), up to 1.92x on 500-s hold panels; ~1.2x time.
- Recovery budget. Cold cells (306-321 K) that receive gas 3-4 dex different can be trapped at tiny steps until
  max_steps (4,000) runs out; it depends chaotically on the starting step. The device recovery got 10 x max_steps
  (40,000) tries; the trapped stress cells finish at 30,000-40,000. Such a cell holds its call for ~8 s (only seen in
  stress mixes). Reverse-mode derivatives take at most 4,000 tries per call.
- Device recovery (2026-10-31): after the guard (non-finite state, unfinished call, element-share drift above the
  allowance), a kernel redoes only the flagged cells, on a dense, threshold-pivoted (0.1) Rodas3. Unflagged cells stay
  bitwise unchanged; no host sync. Plain partial pivoting's solve error reached 12x the step-control scale (against
  1.3e-10).
- Element-drift allowance per network: max(1e-6, 10 r), r the network's worst rounding drift on the parity stress cases
  (3e-11 to 4.5e-9 for six networks; 1.61e-6 for Venot 2019).
- Why mixed gas costs many steps (53-species ring bed, 2026-11-01; a read-only scan, no experiment folder). After a mix
  a few fast trace species are out of balance. The first try fails and the step shrinks (dt 500 s: 14.1 rejections
  before the first accepted step, median first accepted step 2e-4 s), then climbs back through ~19 small steps. Tries /
  accepted steps per call: 38.5 / 24.3 at dt 500 s, 21.0 / 12.6 at 100 s, 9.9 / 6.1 at 25 s. Tries grow with the input
  jump (1.2 below 0.01 dex, 16 at 0.03-0.1, 41 at 0.3-1, 60-85 above 1 dex) and with T (8.6 at 300-650 K, 39 at
  2000-3000 K).
- The V3F step rule (adopted 2026-11-02; experiments/2026-11-01_mixed-speed, 2026-11-01_v3-adopt, 2026-11-02_v3f,
  2026-11-02_adopt). A rejection that follows a rejection at the same state fits err ~ h^p to the two tries and aims at
  err 0.5 (secant shrink, floor 1e-4). Otherwise the factor stays in [0.2, 20]. A rejected try reuses f(N), Jn and w
  (bitwise). V3 set atol 1e-16. V3 had floored the plain factor at 1e-4: the derivative dot test's maximum rose from
  1e-10 to 2-5e-2 on two 300 K cells, so V3F floors it at 0.2 (dot-test maximum 7.6e-8; same speed, 2.287 against 2.282
  us on the bed).
- V3 / V3F results (atol 1e-16): every-call mixing bed, all species, 3.601 -> 2.250 us (69,120 cells), tries 23.1 ->
  14.4. QTe every-call block against VULCAN: 0 misses, worst 0.0180 dex (0.0153 before). Jump stress worst 0.0133
  (0.0106). On live gas (1.03-1.08 tries per call) the rule barely matters (0.582 -> 0.577 us). Species between 1e-16
  and 1e-12 moved by up to 0.69 dex in the live A/B.
- The Rodas3 stage inputs are written as explicit fma (2026-11-03). The compiler had contracted F + 4 ih K1 differently
  in the call and the derivative kernels on NCHO_photo (1 ulp in 9 of 69 rows), which broke their bitwise primal parity.

### Tolerances
- rtol 1e-3 is the working point. rtol 5e-3 gave up to 0.114 dex against a tight reference (2026-10-30). On the mixing
  bed, rtol 2e-3 and 5e-3 give local errors 0.012 and 0.025 dex (> 0.01) (2026-11-01).
- atol 1e-25 since 2026-11-03 (experiments/2026-11-03_atol). At atol 1e-16 both maxichem and VULCAN lost trace sulfur
  species in lit cool gas. In maxichem (box 5932, 519 K, 5.2 bar), COS and S2 (about 1e-18 and 1e-24 at 1.1e4 s) became
  exactly 0 from 3e5 s on and stayed 0. In Radau, COS grows to 3.4e-5 and S2 to 8.7e-10 by 1e8 s. rtol 1e-6 with atol
  1e-16 does not fix it; a smaller atol does. The step that zeroes them was not observed (presumably an accepted value
  above -atol clamped to 0; inferred).
- maxichem against Radau on 37 hard sulfur boxes, same starts: atol 1e-16: 27 off by > 0.1 dex, worst 7.264, 2.2 steps
  per call; 1e-20: 8, 6.702, 3.0; 1e-25: 0, 0.023, 4.1; 1e-30: 0, 0.011, 5.0. Guard counts all 0.
- Cost of atol 1e-25: the NCHO boxes take 4.7 steps per call instead of 3.0 and stay at 0 misses. Settled-gas speed is
  unchanged (NCHO_photo 0.346 / 0.349 -> 0.354 / 0.349 us). Changing gas costs 1.4-1.9x (see "Current state").

### Rates and the collision limit
- Reverse rates come from equilibrium constants, and at low T some exceed any collision rate
  (experiments/2026-11-02_hardcell). CN + M -> C + N + M (A 4.2e-10, 71,000 K) gives a reverse C + N + M -> CN + M of
  3.0e-7 cm^6 s^-1 at 300 K. That is 3e21 times an assumed three-body limit of 1e-28 cm^6 s^-1. The rate is 130x smaller
  at 325 K and 1e4x smaller at 350 K. The fit is from 4,060-6,060 K shock-tube data in argon (Tsang 1992). VULCAN
  computes the same rate.
- Over-fast directions (two-reactant coefficient at 1 bar > 1e-8 cm^3 s^-1, all at 300 K): VULCAN-0D NCHO 7 (CN + M
  7e21x, NH2 + NH2 5e6x, CH + M 6e4x, N + CO 3e4x, CN + C2H2 2e4x, NH3 + CH 3e3x, C2H + M 3e3x). VULCAN 3 NCHO 4 (CH +
  M, NH3 + CH, C2H + M, NCO + HCN 15x). VULCAN 3 SNCHO also S + CN 6e4x. VULCAN 3 had removed CN + M, CN + C2H2 -> C2H +
  HCN and CN + C2H4 -> C2H3 + HCN and reversed N + CO and NH2 + NH2 in 8.2020; VULCAN-0D's files predate that.
- Network edits in VULCAN-JAX (2026-11-03, its ledger C29-C31): NH3 + CH removed; HNCO + M; NCO + HCN and S + CN
  rewritten in their exothermic directions (Tsang 1992; UMIST 2022); N(2D) thermo; CS, CS2, COS, HCS masses; the C3
  network's C2H + C -> C3H + H written C2H + C -> C3 + H (Hickson et al. 2016). CH + M and C2H + M were kept (see "Do
  not retry").
- The collision limit (2026-11-03; in VULCAN-JAX, the truth generator, the torch reference and maxichem). For each
  thermal pair, q = the larger direction's two-body-equivalent rate (k with M folded in, times n_tot^(molecules - 2)).
  If q > K_COLL = 1e-9 cm^3 s^-1, both directions are scaled by 1e-9 / q, so K_eq and the equilibrium are kept. It is a
  regularization ceiling, not a measured rate. In NCHO_photo over 300-3000 K, 1e-5-100 bar it acts on CH + M, C2H + M,
  NH + M, NCO + NH2 and CN + C2H6. The CH + M reverse is capped below 563 K at 1 bar (655 K at 100 bar), the C2H + M
  reverse below 895 K (1828 K) (experiments/2026-11-03_overnight).
- The CH3OH falloff uses the broadening of Visscher & Moses 2011 (the "vm11" width), as VULCAN-JAX and the generator do.
- Networks disagree by more than 0.1 dex (HCN up to 4 dex between two networks; 1.5-3x in HCN / NH3 inside one GCM), so
  broadened models need more than one network's truth (2026-10-31 survey, 2026-10-31_broaden).

### Photolysis
- J per cell and branch is an input, fixed within a call. VULCAN's photolysis lines carry no rate: J = sum over
  wavelength of branching ratio x actinic flux x cross section, which needs the column above the cell. A 0-D box can
  only take J as an input; the GCM provides it. Branches are irreversible (kf = J, kr = 0). photo=False generates the
  same CUDA source as before photolysis was added.
- Realistic J from VULCAN 1-D runs (HD 189733b, HD 209458b, WASP-39b) spans 0 to 39 s^-1 over 1e-5-100 bar. Long-wave
  absorbers (HCO, NO2, H2CO, CH3CHO) stay above 1e-12 s^-1 down to 93 bar, so photolysis is not confined to the top of a
  GCM domain (inferred) (experiments/2026-10-31_photo).
- Photolysis matters (2026-11-02, research build on VULCAN-0D's 55-species NCHO_photo; experiments/2026-11-02_photo,
  2026-11-02_photo_vulcan). It moved 13.5 % of dark-to-lit boxes by > 0.1 dex (up to 3.2). Against VULCAN-0D with
  imposed J (36 boxes) maxichem had 0 misses, worst 0.0004 dex; a dark chain was off the lit VULCAN runs by > 0.1 dex in
  17 % of lit boxes (up to 1.65). VULCAN rtol 5e-3 against 5e-4: within 0.0044 dex. Radau against VULCAN: within 0.0004
  dex.
- Hard lit cells at 300-320 K (2026-11-02, that research build, atol 1e-16). 32 cells needed 2,000-30,000 accepted steps
  for their first lit 100-s call; 2 still failed after the recovery. Cause: CH4 photolysis makes atomic C, NH3
  photolysis makes atomic N, and the over-fast C + N + M -> CN + M titrates them. One accepted step took C (4.5e-26) and
  N (3.9e-27) to -8e-24 / -1.3e-22. Both are far smaller in magnitude than atol (1e-16), so neither the error norm nor
  the -atol test saw them; the clamp set both to 0. At C = N = 0 the titration's Jacobian entries vanish (1e-9 s^-1
  against -3.9e10 with the exact values). Later tries are blind to it in their linear solves but meet it in the stage
  evaluations, so only tries of 1e-5-1e-4 s pass until near 1 s. Without the CN + M line the cells take 45-52 steps.
  VULCAN (2x step growth, 1e-18 s first step) takes 695-2,558 steps on 8 such cells, within 2e-4 dex of Radau.
- On the current network with the collision limit, cold lit cells finish. In their first lit call, 2,280 NCHO cells
  below 325 K take median / max 39 / 129 steps at atol 1e-16 (the atol 1e-25 numbers are under "Current state").
- No published H-rich GCM runs photochemistry inside its chemistry step (2026-10-31 survey, 2026-10-31_broaden).

### VULCAN reference errors
- The yardstick's 17 missed rows (2026-11-02; experiments/2026-11-02_yard17, 2026-11-02_yard17_vulcan). One start: 1x
  solar, C/O 1, 300 K, 35.5 bar, atomic-H gas from 2,617 K at 5e-5 bar; CH4 0.666 dex off at call 1 at every dt.
  maxichem at production and at 1e4x tighter tolerances, and an independent Radau / BDF integration at rtol 1e-10, agree
  to 0.0000 dex (CH4 frozen at 8.82e-9 after ~0.01 s). VULCAN's stored run gives CH4 4.11e-8. Tighter rtol with VULCAN's
  default first step (1e-10 s) stays wrong. A first step of 1e-18 s lands within 0.008 dex of Radau.
- Cause: VULCAN's first step is far too long for the first femtoseconds of the H recombination at 35 bar. Its error
  estimate skips species whose new value is <= 0, and its clip sets small negatives to 0 (CH4 is 0 at its first output,
  1e-14 s). The wrong CH4 then freezes at 300 K.
- Fix: the 918 saved states of those rows hold the corrected run's values
  (data/benches/yardstick/design/truth_saved.npz; the original in experiments/2026-11-02_yard17/backup/). A scan of
  3,818 stored dense VULCAN runs found the same signature in 4 (runs 865, 1401, 1809, 2333); maxichem's worst on the
  other design parents' t0 = 0 rows is 0.002 / 0.001 dex.
- Box 1939 of boxes_sncho2 (511 K, 2.6e-3 bar, lit): VULCAN's S2 is exactly 0 at the late stops, while Radau reaches
  2.2e-6 at 1e8 s. VULCAN against Radau 4.342 dex; maxichem against Radau 0.0835 dex at atol 1e-25, 0.0396 at 1e-26,
  0.0010 at 1e-27. A lost seed (inferred); cause not proven.
- VULCAN's number-density atol drops trace species from its error control. On the 29 lit sulfur boxes that missed at
  VULCAN's default atol, VULCAN-0D was off from Radau by 0.56-6.86 dex in 19. rtol 5e-5 changes nothing. atol 1e-10
  cm^-3 brought all 27 checked boxes within 0.004 dex of Radau. On the 36 boxes of boxes_sncho_atol with a Radau solve,
  VULCAN at that atol was within 0.007 dex on 35 but 7.26 dex off on box 4991 (1e-5 bar: 1e-10 cm^-3 is still a mole
  fraction near 7e-25 there). Hence the sulfur truth's cutoff in mole fraction, XATOL 1e-28. Against Radau, XATOL
  1e-28 gives worst 0.0047 dex (35 of 37 known boxes compared), 1e-26 worst 0.0052 (36 of 37). Box 12 (472 K, 0.33 bar,
  lit) separates them: VULCAN is unchanged at 1e-26 and 1e-27 (153 steps) and moves 0.396 dex at 1e-28 (3,937 steps),
  where it matches Radau within 0.0000 dex.
- Radau spot checks of boxes_sncho2 (2026-11-03): 44 random unreacted starts: VULCAN worst 0.0005 dex, maxichem 0.0049
  dex (at atol 1e-16: 2 boxes off by > 0.1 dex, worst 1.773). 25 of the 93 boxes without a VULCAN result: maxichem 0
  misses of 1,175 states, worst 0.002 dex.
- The truth generator at rtol 5e-3 kept a 4.1-dex CO error against Radau in one pilot box of 4,096; at 5e-4 that box
  agreed with Radau within 0.016 dex and with maxichem within 0.020.
- On the emulator's old training data (2026-10-19): VULCAN-0D's element monitor omits N (atom_list H, O, C, He) and its
  loss_eps 1e3 rejects no step; cold high-pressure runs drifted in N by up to 4.98 dex. The current truth sets are
  accepted with their own element-drift check (box 6602 of boxes_sncho was left out at 1.2e-6 > 1e-6).

### Speed
- Reference: VULCAN-0D's fast path takes ~41 ms per cell-call on one CPU core (0.009 s integration, ~0.031 s rate
  set-up).
- History on the 53-species network, GCM-like forcing without mixing: torch library 2.2-3.4 us (2026-10-31); whole-call
  kernel 0.29-0.39 us; after a speed pass 0.310 / 0.250 / 0.244 us at 32,768 / 131,072 / 262,144 cells and 2.21 ms per
  call at 768 cells (experiments/2026-10-31_gpu-core). The speed pass (two warps per cell, a 32-row register-factored
  trailing block, fma S.v, 32-bit Jacobian words, batched pivot reciprocals, memory reuse) gave 1.19-1.32x.
- Batch scaling (53-species, 2026-10-31): 0.161 ms per call at 1 cell, 1.92 ms at 768, 9.9 ms at 32,768. Per cell-call
  flat at 0.234-0.236 us from 524,288 to 4,194,304 cells. The host returns from a call in 0.061-0.068 ms at every size
  (3 kernel launches, no host sync). Small batches are set by a few heavy cells: at 768 cells the heaviest 10 % take ~88
  % of the call. The call kernel uses 180 registers, no spills.
- Where the time goes (32,768 cells, clock64 counters): rates 10-11 %, Jacobian 12.4 %, u + diagonal 3.7 %, sparse LU 9
  %, trailing LU 17 %, stage solves + right-hand side ~43 %, step control 3 %, guard + recovery 0.9-1.7 %.
- The bound is not settled. Two independent reviews disagree: 11-15 % of the card's fp64 peak, or 64-73 % fp64-pipe busy
  with half of each warp's lanes idle. The hardware counters were not available. Software headroom on this card
  ~1.2-1.5x (estimate). On a GH200 the bound would be the per-cell dependency chain (0.02-0.25 us estimated,
  unmeasured).
- Network size: on the old networks SNCHO cost 1.7-3.1x NCHO (0.87 / 0.31 us at 32,768 cells, 0.76 / 0.24 at 262,144).
  The per-cell layout ends near 130-140 species on this card (inferred); 500-1000 species need a new design.
- Every-call mixing raises the steps and the cost about tenfold: all-species call on 2,458,080 ring-bed cells 3.55 us,
  14.2 accepted steps per cell (53-species, 2026-11-01, before V3F; the V3F numbers are under "Integrator and step
  control"). With jump coupling and every-call mixing, < 1 us is out of reach on this card (inferred): the passing
  settings still need ~14 tries per call at ~0.16 us per try, and the steps follow real fast relaxations after each mix.
  That is why the source coupling was studied (see "GCM coupling").
- Settled lit gas costs the same as dark gas (current state).
- Untried levers (estimates): 2-4 cells per warp for lane use (one cell per thread failed on registers); chemistry lane
  packing and trimming redundant fp64 work (8-12 %); tabulated rates (~10 %); occupancy (~10 %). Under the speed rule, a
  small gain that adds a lot of code is not kept.

### Derivatives
- Design (2026-10-31; experiments/2026-10-31_gpu-core). Derivatives of each call along its accepted steps. Tangent stage
  solves on the primal LU; adjoint by transposed solves with Sherman-Morrison; recomputation inside the call instead of
  storage (~12.9 KB per cell per step otherwise); fixed-order parameter reductions (deterministic). Inputs: x, T, P, dt,
  h, per-cell log rate multipliers lm, the Arrhenius table, J. lm scales forward and reverse together, so it keeps K and
  cannot stand in for photolysis.
- Contracts. A: the step control differentiated on the realised branch (the derivative of the computed map, with d/d
  dt). F: accepted steps frozen (no dt or h derivative). Over 1000-call windows on all 256 panel-P paths, A stays within
  1e-2 of finite differences on every path; F misses > 1e-2 on up to 2.3 % of paths. Against a converged tight solve, F
  is 2-8x closer at the median on multi-step calls. On one-step GCM calls A = F. Default A; F kept as an option. A third
  contract (A0c: the entering step size held) was tested in the torch reference only; its dt derivative tracks A's
  within p90 2.1e-4; not in the kernels.
- Recovered cells: the derivative kernels redo the cells the guard flags as the recover kernel does and differentiate
  that path. A cell that still fails gets NaN derivatives and is counted. The derivative calls have no host sync (guard
  counts stay on the device).
- Cost (53-species network, idle GPU, state advanced every call): JVP / VJP 2.3-3.5x the forward call; ~0.80 / 0.78 us
  per cell-call at 1 M cells. Not remeasured on the current networks.
- The starting-step derivative of ill-conditioned 300 K cold cells is not computable in fp64: an independent torch
  autograd's own forward and reverse modes disagree there by up to 2x. The old check's cold-start starting-step case
  failed on 47 % of 256 direction draws (within the finite differences' own disagreement). It was replaced by a rule
  over 16 step scales per cell plus an exact dot-product check of the step-size derivative.
- Extension to every VULCAN-JAX network (2026-11-03; experiments/2026-11-03_ad; an independent review of the plan and
  code found no formula or index error): photolysis (kf = J exp(lm); tangents and cotangents in J itself, so they hold
  at J = 0); the Troe factor (standard and vm11 widths); the collision limit (weights from the pair before the limit:
  wA = 1 above, 1/2 at the limit; wG = wA times the reverse direction's share of the larger, 1/2 at a tie; the Arrhenius
  cotangent takes the pre-limit part); the 14-column rate table (its 9 Arrhenius columns); the adjoint's photolysis
  products; tsolve with two leading rows per lane; wider gather descriptors (species 7, rate 12, destination 9 bits);
  input dtype and device checks.
- The check: states from truth boxes mapped by species name; primal bitwise CudaSolver's; adaptive tangents against
  central differences per input class incl. lm != 0 (p90 <= 1e-4 with plateau coverage >= 90 %); JVP / VJP dot products
  (p99 <= 1e-6, lm 1e-4; max <= 1e-3); recovered cells; determinism. Near J = 0: the tangent at J = 1e-8 dJ against
  central differences at the default atol (>= 90 % of judged cells with a plateau pooled, >= 80 % per case with 20
  judged cells; three distinct step sizes); the exact-0 case at atol 1e-16.
- J near 0 at atol 1e-25 (2026-11-03). The finite-difference check at J = 0 failed on all three networks (p99 about 1).
  Inferred cause: a species made only by J joins the error control once it passes about atol / rtol, so the tangent at 0
  holds only for a tiny, state-dependent range of J. In the two of the six worst cells that can be resolved, the slope
  reaches the tangent only below e = 1e-18 and 1e-20. At J = 1e-8 dJ the tangent matches central differences (median
  7.0e-8, p90 2.5e-6, p99 4.0e-4); at J = 1e-10 dJ the bar fails. The frozen tangent at 0 is no substitute (median
  relative difference 0.15 from the tangent at 1e-8). Near 0, pooled: NCHO 3,256 of 3,387 cells pass (p99 3.9e-5, max
  1.8e-3); CHO 3,355 of 3,456 (p99 3.8e-5, max 5.2e-4). Exact 0 at atol 1e-16 passes on all three networks, but only 721
  of 3,404 (NCHO), 843 of 3,463 (CHO) and 928 of 3,368 (SNCHO) judged cells show a plateau. No solver change. Not
  covered: sulfur failure states, scale-aware checks of trace species, an independent reference.

### GCM coupling
- Interface (author 2026-10-31): the GCM handles radiation, advection and transport; maxichem stays per cell; J comes
  from the GCM.
- Tracer units: the UM advects mass mixing ratios and converts them to number densities before each chemistry call
  (Drummond et al. 2020). Exo-FMS + mini-chem advects mixing ratios (units not stated; Lee et al. 2023). Published GCMs
  call kinetic chemistry every 1,500 s (Exo-FMS + mini-chem), 3,600 s (Lee, Tan & Tsai 2023) or 3,750 s (UM), with
  dynamical steps of 25-50 s. The author (2026-10-20): chemistry would be called about as often as the dynamics.
- snapy (the GPU rewrite of the ExoCubed dynamical core; snapy 2.12.0 + kintera 2.6.6; 2026-11-01; 2026-10-31_refresh).
  Dry run at C64 x 100 layers = 2,457,600 cells, dt 69.3 s: median step 335 ms with 0 passive tracers, 1,810-1,839 ms
  with 31, 2,810-2,817 ms with 53. Each tracer costs ~47 ms per step (0.14 of the dry step) and ~0.49 GB. Radiation
  (pyharp two-stream, one grey shortwave and one grey longwave band, every 3600 s): 0.05 s per update on 2.46 M cells.
- Coupled run (C48 x 100 = 1.38 M cells, a hot-Jupiter-like test at 833-1961 K; flow + 31 passive tracers + radiation +
  the reduced-tracer call G31; 53-species network, atol 1e-16): chemistry every 70-s step 0.578 us (1.03 accepted steps
  per call; p99 <= 3, max 36); every 7th step 0.852 us (2.06 steps). Guard counts 0; element totals changed by <=
  8.7e-12 relative. Live chemistry was ~6x cheaper than on the every-call-mixing bed. After the 2026-11-02 projection
  work the GCM-facing call took 0.257 us, chemistry 0.21 us. Projected per model day on one card at 2.46 M cells
  (estimate): flow 7 min; + 31 tracers + radiation 38 min; + chemistry every step 67 min (1000 days ~47 days).
- The transported tracers arrive with tiny negative values near the model top (-1e-41 to -6e-23 of the cell's moles,
  450-3,700 Pa; coupled day of 2026-11-02).
- On live gas, tracer transport costs more than chemistry: one hot-Jupiter step costs dynamics 0.35 s without tracers
  and 1.08 s with 31, against ~0.5 s of chemistry (2026-11-02). In the UM, 13 tracers add 3 % and 30 add ~20 % to the
  run time (Christie et al. 2026). Levers on the GCM side (not maxichem): a fused tracer kernel; fp32 tracer
  reconstruction (~2-3x on the tracer part); transport once per step with stage-averaged mass fluxes; PLM reconstruction
  (53 tracers 2.70 -> 2.20 s per step; more diffusive); fewer tracers. Couple chemistry once per step after the RK
  stages.
- kintera's GCM path is one backward-Euler step with a dense LU per cell: ~155 us per cell on a 40-species CHO network,
  ~1 MB of GPU memory per cell, no error control. maxichem's call is >= 30x cheaper per cell.
- Jump against source coupling (experiments/2026-11-01_mixed-speed, 2026-11-01_source-coupling,
  2026-11-02_coupling-test). Every VULCAN truth set applies mixes as jumps before a call. A mix applied as a steady
  source over the call cuts the bed's tries to 1.1-1.5 per call (all species 3.630 -> 0.231 us), but it is a different
  coupling. Against substepped references (not converged at the worst cells) neither form wins everywhere: on the bed
  the jump is off by up to 2.97 dex (2.55 % of cell-calls > 0.1), the source forms by up to 3.7-3.8 dex (1.48-1.49 %).
  With a positive continuous transport and a Strang-split tight reference, the source form has smaller worst local
  errors (2.82 -> 2.50 dex) but not in every statistic; Strang with 1-4 substeps is worse than both. On live gas the
  per-call median jump is 1.9e-4 dex (p99 0.034; bed median 0.073), and the source form saves no time (0.585 against
  0.577 us). Not adopted. A source coupling would need the pre-transport state per cell, an outflow fraction, positive
  transported tracers, a source-aware recovery (prototype: recovered cells match a 40,000-try call to 6e-11 dex, against
  3.4 dex without it) and VULCAN truth with the same source term (a VULCAN-0D pilot: about 35 core-min plus about half a
  day of driver work; not run).

## Do not retry

One item each: what, the result, when. Reduced-tracer results are in "Reduced tracers".

### Solver and kernel
- Fixed-step Rosenbrock (8-48 geometric steps): 6-10 dex errors where linearly implicit steps overshoot (2026-10-30).
- ROS2: 13.7 steps per call on GCM-like paths (Rodas3 1.1-1.2). Ros3: more steps under forcing. Rodas4: rejected in the
  same study, no number recorded (2026-10-31).
- W-method (ROS34PW2) with Jacobian / LU reuse: 2.4-2.6 tries per call against Rodas3's 1.17-1.27, 0.42-0.62x the speed;
  with unlimited reuse 3 QC paths miss (0.64 dex in H) (2026-10-31).
- Rodas3 with an LU factor reused across calls: unlimited reuse misses (7 paths, 0.80 dex); a 10-call limit moves 6
  paths by > 0.03 dex; the conservative rule (refactor at least every 5 calls, or when T moves > 0.1 % or P > 0.5 %)
  passes but gives 0.99-1.01x at 768 cells and 1.18-1.34x at 32,768-262,144, carrying 11.9 KB per cell. The author
  (2026-11-02): "probably don't do this".
- Tried and rejected in VULCAN-JAX (its notes: "Efficiency program", "Ideas not adopted", "Do not retry"):
  layer-parallel solvers (cyclic reduction, BCR), warm starts, float32 in the step, Jacobian / LU reuse across steps,
  tensor cores, register-resident inverses.
- fp32 LU with refinement: 0.07-0.16 dex errors (2026-10-31). fp32 exp: no gain.
- Thread-per-cell generated code: register spill, 2.3x slower than the library. Atomic Jacobian scatter: 15 % slower. A
  generated straight-line step: 131 against 33 us (shared GPU), 575 s compile (2026-10-30 / 31).
- Level-scheduled LU -7 %; a per-round warp vote -8 %; rates in shared memory -7.5 %. Every minimum-degree tie-break
  gives the same structure (2026-10-31).
- Layout-independent numerics for 4 warps per cell at small batches: 2-7 % slower at 2,048-262,144 cells on an idle GPU
  (the earlier 1.4x came from a shared GPU), and it failed the starting-step derivative case (2026-10-31).
- A one-cell-per-block build for small batches: 1.24-1.48x at 4-142 cells only; adopted 2026-10-31, removed 2026-11-03.
- Plain partial pivoting in the recovery: solve error 12x the step-control scale; threshold pivoting (0.1) is kept.
- Learned step sizes; skipping quiet calls: rejected 2026-10-31, no number recorded.
- Chemistry every 2nd / 4th / 10th GCM step: 1.5x / 2.3x faster but trajectories > 3 dex off; every 10th, 3.9 % of
  panel-P paths miss.
- Not advecting the 5 fastest species: 0.36 dex after 50 % mixing (2026-10-31).
- A long call as an equilibrium solver: one 1e13-1e20 s call leaves cold gas up to 9 dex from equilibrium (frozen
  pathways).

### Step control and tolerances
- First-step acceptance by RMS alone: misses after abrupt input changes (QT, CO2 0.163 dex; 2026-11-01).
- First-step alternatives to d1: a stricter RMS limit (0.5 / 0.3) left 15 stress cases at 0.5; a half-call cap on the
  carried step (a2) is clean but costs 1.79x the steps on QC (d1 1.19x); a small first step only after an input jump
  (1.01-1.07x) needs the previous output, which the all-species call does not have (2026-11-01).
- atol 1e-14 (QTe worst 0.070, stress 0.0996). atol 1e-14 with the first-step max-norm kept at 1e-18 passes (QTe 0.023),
  but V3 has more margin at the same speed. atol 1e-12 (bed local 0.016). rtol 2e-3 / 5e-3 (0.012 / 0.025) (2026-11-01).
- atol 1e-16 or 1e-20 for trace sulfur: 27 / 8 of 37 hard boxes off by > 0.1 dex (2026-11-03).
- Shampine's filtered error estimate: 1.34 dex on every try or the first try only (it hides the first-step error); 0.031
  dex after the first accepted step (2026-11-01).
- No or small gain: a growth cap of 20 alone (22.9 tries against 23.1); the carried-step fix alone (23.1); a first-try
  timescale cap (20.9 tries but 16.6 accepted steps against 14.4); remembering last call's first step (3.08 against 3.65
  us; V3 with reuse 2.29) (2026-11-01).
- V3's plain shrink floor 1e-4: the derivative dot-test maximum rose to 2-5e-2 on two 300 K cells; V3F's 0.2 costs
  nothing (2026-11-02).
- A step-acceptance fix for lit cold cells (also reject a try in which a species above 0 falls below -0.5 times its
  value; 2026-11-02): it worked (hard cells 55 / 66 steps) but was not adopted: the derivative kernels keep the old
  test, 0.5 is not justified for SNCHO (0.1 or 0 better), threshold 0 costs +7 % on the NCHO bed. Also tried then: no
  growth after a rejection, clamping to a fraction of the old value, a larger atol (worse: 2 / 14 of 32 cells unfinished
  at 1e-14 / 1e-12). The corrected network and the collision limit made it unnecessary.

### Networks and truth generation
- Removing CH + M and C2H + M from VULCAN-JAX's networks: maxichem moved by > 0.1 dex in 34 of 45,027 truth boxes at
  T >= 1000 K (worst 1.92 dex, C2H2, 1325 K, 0.048 bar, dark). The collision limit handles their low-T reverses
  (2026-11-03).
- Patching VULCAN-0D's networks with VULCAN 3's five 2020 corrections against the old truth: different chemistry
  (yardstick 1.48 % of rows with a miss, 0.26 % large, worst 4.6 dex HCN; qual 165 of 2,440 paths). Superseded by the
  switch to VULCAN-JAX's network with all truth regenerated (2026-11-02 -> 11-03).
- VULCAN rtol 5e-3 for truth (a 4.1-dex CO error in a pilot box); dt_min 1e-20 s (a pilot box stalled); rtol 5e-5 for
  trace sulfur (changes nothing; the cutoff is atol) (2026-11-03).
- A whole-set VULCAN run with workers that write only at the end: stopped after 62 min with nothing written; the
  generator now uses a queue of 512 chunks of 4 boxes (2026-11-03).

### Mechanism size and machine learning
- Mini-chem's 13-species frame with learned rates (2026-10-30): the learned-rate line plateaued at 1.54 % large misses
  on held-out intervals. Best constant rate factors took runs with a miss only from 37.0 to 23.8 % (large: 17.6 to
  11.1 %). In cool gas VULCAN parks C and N in hidden species (HCN 10^-7.4 -> H2CN 10^-7.4 -> HCN), which a 13-species
  frame cannot hold.
- Reduced mechanisms with stock rates need ~48-50 species for <= 0.1 dex (31 species: 29 % / 17 % misses on 20-call
  chains). Learned positive factors on the 31-species skeleton cut misses only to 27.4 % / 12 %. A Gibbs-proximal map
  cannot represent 0.5-2.7 % of calls (2026-10-31).
- Per-cell reduced networks for SNCHO (flux cuts) with a learned picker, a one-step implicit check and a full refresh
  every 10th call: misses in unreacted boxes (13-24 states per picker, worst 0.618 dex) where the static 68-species cut
  has 0; both fail on new planets. On realistic calls (~1 step) the 68 / 42-species networks are only 1.34x / 2.5x
  faster and the check costs 5-9 full calls: 0.15x the full network. A per-species tendency check flags most good calls
  (2026-11-01).
- Machine learning inside the call: no gain on live gas. A cell takes about one Rosenbrock try per call, and verifying a
  learned guess costs about a try (a 2026 study's gate returned 96.8-99.7 % of cells to the solver). Also not
  recommended: surrogates, tabulation (ISAT), cached rate coefficients (T changes every call) (2026-11-02).
- Source coupling instead of jumps: see "GCM coupling" (new truth needed; no gain on live gas).
- On the GCM side: CUDA graphs, torch.compile and stream overlap are not worth it (2026-11-01).
- Where ML could still fit (2026-10-31 survey, 2026-10-31_broaden; 2026-11-01 reviews): learning one input with the
  physical solver downstream (photolysis rates, missing rate or thermo data); per-cell reduced reaction sets with a
  full-mechanism check, only for hundreds of species; learned rates. Ranked last: solver replacement and speed-only ML.
  Ensemble spread correlates weakly with surrogate error (r = 0.31); the schemes that bound errors use a physical
  acceptance test.

## Reduced tracers (retired 2026-11-03)

- Aim (author 2026-10-31): the GCM advects at most ~30 tracers, fewer ideal. Retired 2026-11-03: tracers are reduced
  only if a clean, simple method exists, and the GCM advects every species. Code at commit 399a347 (kintracer.py, the
  designs as JSON, its check and gate).
- Method. The GCM carries some species plus C, H, N, O element budgets (optionally molar-sum rows). Each cell keeps a
  memory of its hidden species. After each GCM mix a maximum-entropy projection rebuilds the hidden species from the
  memory onto the carried budgets, exactly in the elements; then the full mechanism runs. Projection: Newton (phase 1,
  stopped per cell once converged, plus one polishing iteration); more iterations for unconverged cells (phase 2); up to
  400 damped Newton steps on the convex dual objective with an Armijo search, from the phase-2 multipliers and from zero
  (phase 3); then an exact element spill into free atoms, flagged. This is rate-controlled constrained equilibrium /
  species reconstruction from combustion (Keck 1990; Ren et al. 2006), with the memory as the prior. Outputs: tracers in
  mass mixing ratio, memory, step size, mean molar mass, cp, an optional heating rate, flags. Batch- and
  order-independent; a restart from saved tracers, memory and step sizes is bitwise.
- Projection speed (2026-11-02): a one-thread-per-cell kernel (integer row weights 0-12, at most 6 rows; the warp kernel
  redoes cells that need phase 2 or near the exponential's range) brought the projection from 0.36 to 0.0365 us on live
  gas. The G31 call took 0.2514 us replayed, 0.257 us in the GCM.
- Designs (53-species network; judged on VULCAN qual, QT, QT2, every call paired with the all-species control):
  - G31 (31 tracers: N30 + CH3OH + C2 - N2H): 0 paths with a miss in QT2's main domain (worst 0.093 dex from the
    control); 1,000-day QL / QLc clean (worst 0.0019 / 0.0015 from the control); clean on every known development
    regression. The recommended design for C/O <= 1.
  - N30sN2H (30): equal to G31 on QT2, but a 1.6-dex HCN miss near C/O 1 (C/O 0.94, Z 86) that only carrying C2 fixes: C
    and C2 have the same element vector.
  - N30, G30b, D30 (30): fail narrowly (0.10-0.71 dex on QT2's far-donor paths). N26: 66 paths with a miss (13 > 1 dex).
    Every design below 26 fails, hand-made or learned (N22-B1v 238 paths, worst 4.163 dex).
  - Weak spots of G31: carbon-rich gas that keeps mixing (C/O 1-2 ring 1.26 dex; N30sN2H 3.19); 4 cells 0.33 dex in
    calls 3-7 after an equilibrium start at Z 300.
- Findings:
  - The per-cell memory is required: memory-free forms fail by ~6 dex.
  - N30's QT failures were the projection's numerics, not lost information. Cold far-from-equilibrium donors hold their
    hidden carbon as CH3OH. The old Newton, judged by a residual dominated by the tiny N budget, stalled and spilled the
    carbon into free C, which made C2H2 and H. A collision test (other real donor gas with identical tracers) moved the
    scored gases by <= 0.0034 dex over 240 pairs. The phase-3 step fixed most of it (60 development paths: 30 -> 4 with
    a miss); CH3OH and C2 had to be carried.
  - The ring test bed alone cannot judge a design. Pair-sum designs (N22g2, N22g0) passed it and the collision test but
    failed VULCAN QC / QCs by up to 4.8 dex (cold NH3 / CH4 mixed into hot carbon-rich gas; inferred). Judge every
    candidate on VULCAN from the start.
  - Counting every species above 1e-10, the 10-call collision test loses information even at N26 (18 / 54 cells > 0.1
    dex); the 12-gas version is a lower bound.
  - A simple abundance rule fails (2026-11-03_tracer-rule, retired; QT2's 1,056 mixing paths): 30 tracers 16 paths with
    a miss (0.81 dex), 36 tracers 2 (0.17), 36 + HCO + HNCO 0 (0.062). HCO and HNCO hold small element shares but matter
    after mixing.
  - Sulfur: the floor is about 50-62 tracers, twice the limit; the extra non-sulfur species of SNCHO cause it.
  - Photolysis: N26 hides 5 of the 18 photolysis absorbers of the NCHO photo network (C2H6, CH3CHO, H2CO, NO, NO2); N30
    hides 2.
- Machine learning, none adopted (2026-10-31 / 11-01; 2026-10-31_mltracers, 2026-10-31_refresh): learned tracer bases (6
  runs never cut validation misses by the required 30 %); learned rows trained on the failure regime (tie or lose to
  plain designs at the same N); an in-call learned prior (a bounded correction barely moved the failing paths, though an
  oracle showed the information mostly carried); a learned decoder through K calls (no help); a learned warm start for
  the projection (1.8-1.9x slower than early exit + previous multipliers). The useful results were not learned: CH3OH,
  C2, the dual-objective step, the per-cell early exit. The obstacle was the projection on combined rows: a few
  unconverged rebuilds per thousand calls decided most verdicts.
- Cost: the reduced call was 0.4-1.5 us per cell-call slower than the all-species call; it saved only advection (see
  "GCM coupling").
- Literature (2026-10-31_refresh; references below): exoplanet GCMs with kinetic networks advect one tracer per species
  (12-13 with mini-chem, 30 in the UM); Earth models transport fewer species than they solve (GEOS-Chem 250 of 358) and
  use transported families (Jacob et al. 1989: 6 tracers; Douglass et al. 1989: the NOy family). A full mechanism per
  cell behind <= 30 tracers had no precedent found. Greedy hiding of species matches Hiremath et al. 2010. Learned
  solvers in Earth models accumulate error where gas ages or mixes (Keller & Evans 2019; Kelp et al. 2022).
- Untried: a leave-one-out search below 26 (~20 variants); family or reservoir tracers chosen by downstream sensitivity;
  a validated ~35-tracer design (an independent review's suggested target).

## Neural-network emulator (retired)

From 2026-06 to 2026-10-30 a neural network predicted the next state per GCM step (a closed 25-entry state, a relaxation
head and an element projection), trained on VULCAN-0D boxes. The published 12-species model is arXiv 2609.00428 (ApJ).
The best later model (the flow map ov_DT2) missed on 1.25 % of main-domain rows but on about a third of GCM-like paths
(panel P), and rare multi-dex misses never went away. The physical solver replaced it on 2026-10-30. Code:
github.com/imalsky/Autoregressive-Chemulator (tag paper-v1 is the paper's code).

## Decisions in force

- 2026-10-31 ~02:5x: the full mechanism is the default ("Do whatever is more accurate"); ML is not required for its own
  sake.
- 2026-10-31 ~02:5x: the MIT code may read VULCAN's reaction and NASA-9 data files.
- 2026-10-31 ~02:5x: gas below 300 K is a hard error.
- 2026-10-31 ~05:2x: tests are quick and directional by default; a big test takes at most 4-5 h.
- 2026-10-31 ~05:4x: one notes.md file.
- 2026-10-31 ~06:2x-06:5x: a GPU solver for a GCM, differentiable, flexible and as fast as possible. Gradients for ML
  training, retrievals, sensitivities and a differentiable GCM. RTX and GH200; few tuned options; bitwise reproducible
  on the same GPU; mixed precision allowed on RTX if validated.
- 2026-10-31 ~14:0x: derivatives default to contract A; F kept as an option.
- 2026-10-31 ~16:3x-17:0x: the long-term aim is photochemistry, maybe transport, much bigger networks, and the widest
  range of T, P and composition while staying accurate. The GCM handles radiation, advection and transport; maxichem
  stays per cell; photolysis rates come from the GCM.
- 2026-10-31 ~22:1x: a 0-D model for use in the GCM; the CUDA differentiable work stays; later coupling to the GCM's
  transport and radiation must stay possible (not attempted now). Documentation: only README.md, the project
  instructions file, todo.md and notes.md.
- 2026-11-02 ~16:1x: "the best version that works in a realistic scenario", "SUPER fast in a gcm and always accurate".
  Reusing the factored matrix across calls: "probably don't do this". The V3F step rule adopted.
- 2026-11-02 ~18:0x: the yardstick's reference for its one bad start replaced by the corrected VULCAN run.
- 2026-11-03 ~03:1x: maxichem as the GCM's complete chemistry step, under about 1 us per cell-call, for every network up
  to the sulfur ones, even at some cost in speed. GPU code that saves only hundredths or tenths of a microsecond is
  removed. Tracers are reduced only if a clean, simple method exists. Plans and complicated code are reviewed first.
  Simplified technical English everywhere.
- 2026-11-03 ~05:08: extend the derivative layer (done for every VULCAN-JAX network); drop the reduced-tracer code but
  keep its notes; "fix the ch and ch2 stuff"; push to GitHub when needed.
- 2026-11-03: VULCAN-JAX's NCHO_photo is the default network, with photolysis and the collision limit.
- 2026-11-03 afternoon: "Fix the atol and sulfur": default atol 1e-25 (a numerics change) and new sulfur truth
  (boxes_sncho2). "fix the 0 derivate thing easily": the J-near-0 check changed, no kernel change.
- 2026-11-03: the neural-network emulator and the reduced tracers are retired; the JAX layer is dropped (the derivatives
  stay in the CUDA kernels); experiments/2026-11-03_ad/dev_check2.py becomes tests/kincuda_ad_check.py; refdata3
  (paths.AD_REF) is retired.
- 2026-11-03 ~22:5x-23:1x, GCM-side speed work (snapy branch `isaac/maxichem-coupling` on the author's fork). Goal: fast
  chemistry and fast tracer transport, "full CUDA".
  - Order: tracer transport first, then a C++/CUDA entry point so snapy's time loop calls maxichem with no Python in
    the loop (a Python driver only for the first benchmark), then chemistry-solver speed. Stop at diminishing returns
    (when a step of work gives less than ~10 % on the whole coupled step), then report and ask.
  - Correctness: snapy's default path stays bit-for-bit as upstream; fused kernels may differ at round-off, behind an
    opt-in switch. Results must be deterministic run to run (no floating-point atomics or order-dependent reductions).
  - Lower precision and cheaper schemes (fp32 arithmetic inside transport; WENO3 or PLM for tracers instead of WENO5)
    are allowed only where they are much faster and meet a bound fixed in advance: the 12 scored gases within 0.01 dex
    of the fp64 / WENO5 run after 1,000 coupled steps, tracer mass conserved to fp64 round-off, no new negative
    tracers.
  - Tune for the RTX cards first; the GH200 later.
  - Speed numbers are quoted on the hot-Jupiter-like deck at C64x100 (2.46 M cells) with all 69 NCHO_photo species as
    tracers and chemistry every step; sulfur (88) once memory allows.
  - Tiny negative tracers reaching the chemistry: find out first whether they are a transport bug; if not, clamp at 0
    before the call and count them and the mass added.
  - The 48 KB reconstruction-launch fix is held on the fork: no upstream post or PR until the author says.
- 2026-11-04 ~00:1x, the snapy group's calls on the coupling plan, adopted by the author ("incorporate that"):
  - Passive tracers first; hydro species with feedback on the EOS only once the passive path is stable and conserves
    element totals over several model days.
  - A Python driver on unchanged snapy first; a C++ launcher only if the measured time split shows that the Python
    overhead matters. Superseded the same night (below).
  - Photolysis: J from kintera, computed from pyharp's mean intensity, so radiation and chemistry see one radiation
    field; VULCAN's branches mapped with branch ratios; own cross sections only for a branch kintera lacks, listed per
    branch. kintera's J is per reaction (summed over branches) and can be injected as `kf_override`.
  - Develop on one GPU; scale-up runs on GH200s once the coupled test is ready.
  - Coupled tests run on a Cartesian box (independent of the stretched-x1 work). Each includes a convergence check of
    one passive tracer, so transport error is not mixed with chemistry error, plus element totals and guard counts.
    snapy reconstructs a tracer's mixing ratio as (cell-mean tracer) / (cell-mean density), which on a stratified
    column limits transport to about 2nd order (measured 2.0 against 4.7-4.9 with exact averages); an upstream fix is
    planned after the stretched-x1 work.
  - Memory per tracer is to be settled by measurement: persistent arrays about 0.175 GB per tracer at C64x100 (7
    fp64 arrays, a code count) against about 0.47 GB measured at the peak; the gap is transient work in the tracer
    step.
- 2026-11-04 ~00:3x: the author wants the fused-CUDA path as the main path because speed is the priority: tracer
  transport and chemistry as fused CUDA kernels called from snapy's C++ time loop, no Python in the loop; a Python
  driver only for tests and the first coupled box runs (posted to the snapy group). Measured meanwhile on the box
  (921,600 cells, 69 tracers, chemistry every step, 3 steps): the GPU is busy 99.7 % of the wall time, so the Python
  loop's own overhead is about 0.3 %; the gain from C++ comes from fusing work, not from leaving Python.

## Log

New dated entries go below, newest last.

### 2026-11-03 (evening): cleanup, project layout, literature survey
- Request (author): clean up before the GCM integration. Retire the emulator work. Gather maxichem and what it needs in
  one project folder. Remove legacy code (the JAX layer, reduced-tracer leftovers). Concise documents. A survey of
  similar projects for large speedups. No GCM code edited.
- Layout. The checkout sits in a project folder beside the truth generator (vulcan-truth: GPL, now under local version
  control), the torch reference generator and the sulfur parity driver (reference/), the snapy clone, and one Python
  environment shared with snapy, pyharp and kintera (Python 3.11, torch 2.10, NumPy 2.4.6, NVRTC 12.8). The seven
  emulator-project experiments that the maxichem work used moved into experiments/. Retired: three reduced-tracer
  experiments (2026-11-02_projection-exit, 2026-11-02_fast-proj, 2026-11-03_tracer-rule) and data that no kept code,
  check or experiment reads (truth/qt2, truth/sncho, truth_v2/old_network/{qual_ncho, daynight_ncho, pilot_ncho}).
  Absolute paths in experiment scripts and links were rewritten to the new places.
- Code. Removed: the JAX layer (kinjax.py, kinjax.cc, its test, the jax extra), the unused pyyaml dependency,
  CudaSolver.src and tests/libref.py's rates(). The derivative check moved unchanged into tests/kincuda_ad_check.py
  (input paths.AD_BOXES); paths.AD_REF is gone.
- Checks before and after the move (experiments/2026-11-03_relocation; GPU1; before: Python 3.12, torch 2.8, NVRTC
  12.6; after: the shared environment). kincuda_check 28 of 28 PASS. kintest quick: 0 misses, every scored number equal
  to the saved run (only timings differ). NCHO boxes, day-night, hard cold lit cells, sulfur parity (15 of 15) and the
  sulfur score: unchanged. Derivative check: NCHO_photo 101/101, CHO_photo 101/101, SNCHO_photo 99/101, the same cases
  both times (the two J-tangent cases on box starts at dt 500 s); dot-product medians differ at the 1e-16 level
  (summation order). The new toolchain gives the same results.
- Literature survey, checked against "Do not retry" (sources under References, "GPU chemistry elsewhere").
  - Published GPU chemistry codes report speedups over CPU cores, not a cost per cell. Where a cost per cell-call can
    be derived (WRF-Chem, EMAC, CAM4-Chem; older GPUs), it is 10-100x maxichem's.
  - No exoplanet GCM runs a ~70-species photochemical network online. The UM runs a 30-species network (no
    photolysis) every 125 dynamics steps; Exo-FMS runs mini-chem (12 species) with a 1500-s chemistry step.
  - Already tried here and rejected: fp32 LU with refinement, one thread per cell, a W-method, chemistry every Nth
    call, skipping quiet calls, reduced mechanisms. The source coupling is 16.7x faster on the mixing bed but saves no
    time on live gas and needs new truth.
  - Not tried: the LU in float-float arithmetic (pairs of fp32; on this card about 12 digits at about 10x the speed of
    fp64, per NVIDIA's documentation; fp32's exponent range needs scaling at atol 1e-25); a fresh first-step estimate
    after a large input change (as in LSODE's starting-step routine); a predictive step-size controller (Gustafsson
    1994); an equilibrium shortcut for cells whose chemical time is far below dt (quiet cells already cost ~1 step).
  - The largest lever is fp64 hardware (GH200; todo.md).

## References (by topic)

Identifiers as recorded. Verify each before paper use.
- Our work and codes: the one-step emulator paper (Malsky, Zhang et al.), arXiv 2609.00428, accepted in ApJ; VULCAN-0D
  github.com/shami-EEG/VULCAN-0D (b3ccf4f); VULCAN github.com/shami-EEG/VULCAN; mini-chem github.com/ELeeAstro/mini_chem
  (ae021c8); github.com/imalsky/Autoregressive-Chemulator (paper code = tag paper-v1).
- Mini-chem: Tsai et al. 2022, arXiv 2204.04201 (net-reaction tables; sec. 2.2.1 composition recipe; mini-chem vs VULCAN
  "rarely exceeds a factor of two", sec. 3.2); Lee et al. 2023, arXiv 2302.09525 (Exo-FMS + mini-chem, 1500 s chemistry
  step, 1x and 10x, Table 2; tables valid for one metallicity, sec. 2).
- Solar abundances: Lodders 2009 (the training generator; no id); Lodders 2019, arXiv 1912.00844 (Table 8 present-day,
  mini-chem's tables); Asplund, Amarsi and Grevesse 2021, arXiv 2105.01661 (mini-chem's equilibrium tables); Asplund
  2009 (no id).
- GCM chemistry steps and envelope: Drummond et al. 2020, arXiv 2001.11444 (UM: chemistry every 125 steps = 3750 s,
  radiation every 150 s, elements conserved to 99.9 % over 1000 d, heating changes mainly from CH4); GCM T-P grids Roth
  2024 (516 models) and Kennedy 2024 (55 models) (no ids).
- Detection limits behind the per-gas floors: Gasman et al. 2022, arXiv 2202.01151; Greene et al. 2016, arXiv
  1511.05528; Drummond et al. 2020 (above).
- Metallicity range: August et al. 2023 (WASP-77Ab, JWST, [M/H] -0.91), BOWIE-ALIGN (HAT-P-30b 0.2-0.8x), TOI-5205b
  (< 0.03x upper limit) (no ids); not re-read: K2-18b at 180x, arXiv 2506.23891; observed giant-planet metallicity
  0.1-100x and C/O mostly 0-1, arXiv 2510.26785.

### Transported species, families, slow manifolds, tracer counts (verified 2026-10-31; checks in 2026-10-31_refresh)
- Jacob D.J., Sillman S., Logan J.A., Wofsy S.C. 1989, JGR 94, 8497, doi 10.1029/JD094iD06p08497 (6 transported
  tracers).
- Douglass A.R., Jackman C.H., Stolarski R.S. 1989, JGR 94, 9862, doi 10.1029/JD094iD07p09862 (the NOy family).
- Keller C.A., Evans M.J. 2019, GMD 12, 1209, doi 10.5194/gmd-12-1209-2019 (51 transported, 99 not; RF solver drift).
- Kelp M.M., Jacob D.J., Lin H., Sulprizio M.P. 2022, JAMES 14, e2021MS002926, doi 10.1029/2021MS002926.
- Shen L., Jacob D.J., Santillana M., Bates K. et al. 2022, GMD 15, 1677, doi 10.5194/gmd-15-1677-2022.
- Maas U., Pope S.B. 1992, Combust. Flame 88, 239, doi 10.1016/0010-2180(92)90034-M (ILDM).
- Lam S.H., Goussis D.A. 1994, Int. J. Chem. Kinet. 26, 461, doi 10.1002/kin.550260408 (CSP).
- Keck J.C. 1990, Prog. Energy Combust. Sci. 16, 125, doi 10.1016/0360-1285(90)90046-6 (RCCE).
- Hiremath V., Ren Z., Pope S.B. 2010, Combust. Theory Model. 14, 619, doi 10.1080/13647830.2010.499964.
- Ren Z., Pope S.B., Vladimirsky A., Guckenheimer J.M. 2006, J. Chem. Phys. 124, 114111, doi 10.1063/1.2177243.
- Li G., Rabitz H. 1989, Chem. Eng. Sci. 44, 1413, doi 10.1016/0009-2509(89)85014-6 (exact lumping).
- Zamyatina M. et al. 2023, MNRAS, arXiv 2211.09071; Zamyatina M. et al. 2024, arXiv 2402.14535; Steinrueck M.E. et al.
  2025, arXiv 2509.21588; Christie D.A. et al. 2026, arXiv 2604.15850 (UM tracer cost); Baeyens R. et al. 2026, arXiv
  2607.08486 (Exo-FMS + mini-chem, 13 tracers).

### GPU chemistry elsewhere (survey 2026-11-03; numbers as the papers report them)
- Michalakes, Linford, Vachharajani, Sandu, WRF-Chem RADM2 on GPU (www2.mmm.ucar.edu/wrf/WG2/GPU/Chem.htm): KPP
  Rosenbrock, one thread per cell, 59-61 species; fp32 2.1x faster than fp64 on a C1060.
- Alvanos & Christoudias 2017, GMD 10, 3679 (EMAC/MEDINA): KPP to CUDA, one thread per cell, 155 species; kernel 20.4x
  one CPU core, whole model 1.75x. Sophocleous & Christoudias 2022, Atmosphere 13, 1418, doi 10.3390/atmos13091418 (KPP
  Rosenbrock in fp32 within 1 %); Sophocleous et al. 2026, EGUsphere preprint egusphere-2026-2389 (fp32 GPU kernels ~30 %
  faster than fp64).
- Sun et al. 2018, JAMES, doi 10.1029/2018MS001276 (CAM4-Chem, ROS-2 on GPU, 103 species, interleaved cell layout).
- Guzman Ruiz et al. 2024, CPC, arXiv 2405.17363 (CAMP: BDF with an iterative linear solver, one block per cell).
- Balos et al. 2025, IJHPCA, arXiv 2405.01713 (Pele with SUNDIALS: batched direct LU > 10x GMRES).
- Curtis, Niemeyer, Sung 2017, Combust. Flame 179, 312, arXiv 1607.03884 (one thread per ODE: divergence and memory
  traffic limit it); Curtis, Niemeyer, Sung 2018, Combust. Flame 198, 186, arXiv 1809.01029 (pyJac sparse Jacobians).
- Carreon et al. 2025, arXiv 2510.23993 (compacting finished cells; up to 6x on chemistry with an explicit solver).
- Venot et al. 2019, A&A 624, A58, arXiv 1902.04939 (30-species reduced network, ~30x faster, no photolysis).
- Lin et al. 2023, JAMES (KPP per-cell mechanism reduction: 32 % less solver time on CPU, < 1 % error).
- Knoth & Wolke 1999, WIT Trans. Ecol. Env. 36 (operator splitting causes initial transients that force small steps).
- Rang & Angermann 2005, BIT 45, 761 (ROS34PW2, the W-method tried here).
- Gustafsson 1994, ACM TOMS 20, 496, doi 10.1145/198429.198437 (predictive step-size control; not read).
- NVIDIA CCCL documentation, libcudacxx extended fp API (float-float and emulated fp64 throughput on the RTX 6000 Ada).
