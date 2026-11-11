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
  generator caps a network at 254 species. The sulfur network is SNCHO_photo_network_2025 (91 species, 4 warps per
  cell; since 2026-11-04, replacing SNCHO_photo, 88 species).
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
  0.0662 dex. 4.7 steps per call (3.0 at atol 1e-16). Guard 0 at this date; 2 recomputes since 3eb0586 (log, 2026-11-06 17:22).
- NCHO_photo on the 904 qual_ncho paths (kintest quick): 0 misses, worst 0.004 dex. Day-night set, 64 paths: 0 misses,
  worst 0.0023 dex.
- SNCHO_photo_network_2025 (2026-11-04 log): 1,930 boxes, 1 with a miss (box 1939, a VULCAN error); 350 targeted
  boxes, 1 miss (box 96, 0.155 dex S2, maxichem's: fixed by atol 1e-30, default kept); 272 paths, 0 misses.
- Superseded (the old network): SNCHO_photo on boxes_sncho2 (16 gases; the 1,955 boxes with a VULCAN result): 91,885 states, 7 with a miss, all large
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
- A 1,000-day sign-off of the current code at dt 25 and 50 s: `kintest full` now takes them to 200 and 400 days, and only dt 500 s to 1,000 days (log, 2026-11-06 19:25). The last 1,000-day runs at those dt were on the 53-species network (2026-11-01).
- A GH200 timing was never run. A timing package with the 2026-10-31 kernel exists in experiments/2026-10-31_gpu-core.
- Superseded 2026-11-04: the drift survey that sets a new network's element-drift allowance runs again
  (experiments/2026-11-04_net2025/drift_survey.py; it had imported a removed module).
- From the 2026-10-31 survey, not rechecked since: no upper-T guard; third-body efficiencies fixed to H2 (high
  metallicity would need composition weights); the reverse-rate exponent is unclipped.
- Superseded 2026-11-04: a C++ caller loads an export of the solver (cpp/; bitwise the Python call).
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
- kincuda/libref7.npz: tests/kincuda_check.py on NCHO_photo at atol 1e-25 (28 of 28). kincuda/libref8_SNCHO25.npz:
  the same check on SNCHO_photo_network_2025 (--network; 27 of 28, see the 2026-11-04 log); kincuda/jref_SNCHO25.npz:
  its J table. kincuda/libref7_SNCHO.npz: the old sulfur network's parity (15 of 15), superseded. kintest/quick_ref6.npz: the kintest quick reference. Earlier references belong to superseded
  numerics.
- Superseded 2026-11-10: the references of the collision rule that counts M are kincuda/libref10.npz and
  libref10_SNCHO25.npz (they record RATE_RULE) and kintest/quick_ncho_v3_ref1.npz and quick_sncho25_v3_ref1.npz (runs on
  truth_v3). The ones above, libref8 and libref9_SNCHO25 among them, belong to the old rule.

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
- Superseded 2026-11-10 (log entries of that date): all four codes now count M as a collision partner, so a dissociation
  A + M is capped: VULCAN-JAX since 1c6afe1, maxichem, the truth generator (vulcan-truth 0ea6d9c) and the torch reference
  (libref10, libref10_SNCHO25). The truth under data/truth_v3 is made with it; data/truth_v2 is the old-rule record, in
  which a one-molecule side never counted.
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

- 2026-11-04 ~01:1x, after the snapy group's conditions on the fused path (the Python-driver box run stays the
  reference until the fused path matches it; chemistry and transport as separate PRs, the fused transport's design
  approved by snapy's author before code, three sign-offs incl. CUDA; the fused transport built on the corrected
  tracer reconstruction): the 48 KB reconstruction-launch fix goes upstream as its own PR (so unchanged snapy can run
  69-88 tracers for the reference run); the negative-tracer defect was reported, with the repro bundle for its fix.
  The author: "whatever you recommend or is faster" on the shape of the fused path, so: chemistry stays a separate
  fused call from snapy's C++ loop (fusing it into the transport kernels saves little and upstream will not take it);
  the bit-for-bit fused transport stays for our runs, and the upstream version adds the corrected reconstruction as
  a switch once its fix plan is known.
- 2026-11-04 ~12:00, the author, after the overnight runs:
  - No fp32 in the tracer WENO5. The fp32-weights variant (another 9 % per step at C64x100) is not adopted.
  - No GH200 benchmark now.
  - The 48 KB launch-split PR goes to review: marked ready at its unchanged head.
  - The fused-transport design is posted to snapy's author for approval before any upstream PR.
  - The C++/Python pair-comparison dumps (14 GB) are deleted; their logs are kept.
- 2026-11-04 ~15:47, the author: sulfur means VULCAN-JAX's SNCHO_photo_network_2025.txt (91 species). The two default
  networks are the newest NCHO (NCHO_photo_network.txt) and the newest sulfur network; the older SNCHO_photo_network.txt
  is not a default. Test the 2025 network fully, use it in the snapy coupling, push and post.
- 2026-11-04 ~22:10, the author: keep atol 1e-25 as the default, and expose atol as a setting in maxichem (a call
  argument in Python and C++, as it was) and in the snapy coupling (the drivers' --atol; the future snapy chemistry
  step reads it from the deck).
- 2026-11-04, the author, on the GCM-side work: do not ask; investigate, post findings and results in the snapy group's
  thread, and keep working. Still ask before an upstream PR or marking one ready.
- 2026-11-09 ~16:56, snapy's author: maxichem's species go to passive scalars, not hydro species. Vapors and clouds add
  to the total mass, so they are hydro variables in the Riemann solver; chemical tracers do not, so they are stored
  apart and moved on the mass flux the hydro provides. The first of the open coupling decisions is settled.

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

### 2026-11-03 night to 2026-11-04 01:48: coupled runs and snapy tracer transport
Snapy built from source (main 2d9fa419, CUDA 12.8, sm_89) on branch `isaac/maxichem-coupling`; all runs on one RTX
6000 Ada, NCHO_photo (69 species) as passive tracers, dark (J = 0), atol 1e-25, chemistry after every 70 s step,
negative tracers set to 0 before the call and counted. Data: experiments/2026-11-04_gcm-chem and
experiments/2026-11-04_tracer-speed.
- Cubed sphere, hot-Jupiter-like, C48x100 (1,382,400 cells), one model day from a 1.2-day spun-up flow (snapy's
  default transport): 1.40 solver steps per call (median over steps; 1.24-1.45), at most 93 in one cell; 0.449 us per
  cell-call; no failed or unfinished cell, every guard count 0; per step 2.18 s of flow and tracers against 0.70 s of
  chemistry; per-cell element change through the chemistry at most 1.7e-12; totals over transport and chemistry
  conserved to 1e-15; peak 27.6 GB.
- Cartesian box (96 x 96 x 100 = 921,600 cells, periodic sides, 2000 km cells, a day side along one axis), 1 model
  day of spin-up, then 3 model days coupled (fused transport): 1.43 steps per call (1.28-1.65), at most 37; 0.469 us
  per cell-call; no failed cell; per step 0.57 s of flow and tracers against 0.49 s of chemistry; the Python loop's
  own overhead 0.2 % (GPU busy time against wall time over 5 steps). Every tracer total and element total grows by
  about 1.1e-10 per step (4e-7 over the 3 days), the same for all species: the box's dry air mass grows by the same
  amount with no tracers at all (1.0e-10 per step over 20 steps), so it is the flow's own budget in this deck, not
  transport or chemistry. Cause not investigated.
- Negative tracers after transport (cubed sphere: 16,210 entries over the day, up to 54 per step, worst mass
  fraction -3.4e-22). Cause: snapy's implicit x1 tracer transfer, added after the scalar's positivity limiter and
  limited only by the donor's dry air. With it skipped, none (41 CHO_photo tracers, unmodified main plus a switch, 40
  steps each: 20 of 40 steps with negatives against 0). Reported to the snapy group, who are fixing it upstream (a
  tracer-side donor limit). The clamp and count stay as a guard.
- Tracer-count cap: the CUDA reconstruction launch held every tracer's whole line in 48 KB of shared memory, so at
  most 57 fp64 tracers on 106-cell lines; 69 failed. Fixed by splitting the launch (bit for bit the same results);
  offered upstream as its own PR.
- Fused tracer transport (opt-in, SNAP_SCALAR_FUSED=1; bit for bit the default path, tested on five decks): at
  C64x100, 44.2 -> 19.7 ms per tracer per step; 69 tracers 3.52 -> 1.85 s per step; peak memory 0.45 -> 0.38 GB per
  tracer. WENO5 reconstruction is now about half of the fused step.
- One passive tracer, flow held fixed (WENO5): vertical transport through a stratified column converges at order 2.1
  (25 -> 50 cells), horizontal at 4.0, as the snapy group found for the (cell-mean tracer) / (cell-mean density)
  reconstruction.

### 2026-11-04 02:00 to 03:22: box rerun, the coupled loop from C++, convergence check
Same software as the entry above (snapy fork branch at e2b0089, maxichem 1f2cf97). Data:
experiments/2026-11-04_gcm-chem (runs/box, runs/conv, runs/conv2) and experiments/2026-11-04_cpp/box.
- Box drift (supersedes "cause not investigated" above): the driver added the heating after Mesh.forward's ghost
  exchange, so the periodic ghosts were one heating step stale and the two copies of each periodic face saw different
  states. A ghost exchange after the heating fixes it. Rerun of the 3 model days: every element total changes by at
  most 2.4e-13 (relative) over the run (4.0e-7 before), at most 5.4e-16 per step through transport and 1.4e-15
  through chemistry. Chemistry as before: 1.43 steps per call (median), at most 37; 0.465 us per cell-call; no failed
  or unfinished cell. Negatives after transport: 5,256 entries in 28 of 3,705 steps, up to 192 per step, worst -5.9e-43,
  clamped.
- The coupled loop from C++ (box.cpp: snap::Mesh, harp::ToonMcKay89 and cpp/maxichem.hpp, no Python; the same snapy
  libraries for both): after 20 and after 120 coupled steps (fused transport, three radiation updates), the flow, all 69
  tracers, the step sizes and the solver counts are bit for bit the Python driver's.
- One passive tracer, flow held fixed (conv_tracer.py). The first full run was a flawed test: the blob (width 0.1 L)
  started 3 widths above the bottom wall, so part of the "exact" profile came from below the domain, where the
  reflecting wall lets nothing in; the vertical error then grew with resolution (largest error above 1e-3 of the peak
  0.2 -> 4.7 dex from 25 to 200 cells). With width 0.05 L (6 widths from either wall), vertical transport converges at
  order 2.9, 2.7, 2.2, 2.0 (25 -> 400 cells; L1 0.25 -> 2.8e-4; largest error above 1e-3 of the peak 1.2 -> 0.0017
  dex). Horizontal: order 4.0 and 4.3 (25 -> 100 cells), then a floor at L1 2.5e-5 with 8 cells in x1; with 16 or 32
  x1 cells the floor is gone (order 4.9 and 4.8, 100 -> 200 cells), so it is a vertical-resolution term, not the
  horizontal transport. Turning the flux covariance off changes nothing in this Cartesian deck.

### 2026-11-04 03:13 to 05:01: tracer WENO5 arithmetic, the noise floor of the coupled box, branch cleanup
Night queue on one RTX 6000 Ada; data in experiments/2026-11-04_night (summary.txt has every job).
- The fused path's WENO5 was half of its tracer time, mostly fp64 divisions (this card runs fp64 at 1/64 rate).
  Variants, C64x100 with 69 tracers, median per step:
  - default path 3491 ms;
  - fused, bit-exact WENO5 1537 ms;
  - fewer divisions 1391 ms;
  - the same with guards 1327 ms;
  - fp32 smoothness indicators and weights 1205 ms.
  Tracer WENO5 kernels, GPU ms per step: 593, 418, 406, 273. Peak memory: 30.1 GB fused, 37.0 GB default.
  - Fewer divisions: the weights c_k / d_k are multiplied through by d0 d1 d2, one reciprocal serves both faces, and
    phi = src * (1 / mean). That takes 12 divisions by variables down to 2.
  - The guards: the quotient is formed before the scale is applied (the unguarded form underflowed to 0 at 1e-305,
    gave NaN on subnormals and Inf on a 1e303 spike); 1 / mean is skipped below 1e-300 (fp64) or 1e-30 (fp32); the
    default formula is used when scale is false.
- Transport only (cubed sphere C24x40, 200 steps; constants from 1 to 1e-305, zero, fronts):
  - guarded form vs bit-exact: within 3.4e-14 floored dex;
  - fp32 weights: 1.1e-10 dex on smooth tracers, up to 2.2e-7 at sharp fronts;
  - in both, constants and zero are kept and the mass drift is identical.
- Coupled box (96x96x100, 69 tracers, chemistry every step), 1,000 steps against the bit-exact build:
  - the guarded form and the fp32 form both end 0.0027 dex away (floor 1e-30; 0.0007 dex at floor 1e-12), with the
    flow bit-identical and the budgets and negatives identical;
  - the bit-exact build with its start tracers multiplied by (1 + 1e-15) ends exactly as far away (0.00268,
    0.00207, 0.00070 dex at floors 1e-30, 1e-20, 1e-12).
  So the coupled box has its own floor of about 0.003 dex after 1,000 steps: the solver's step control turns any
  round-off change into differences near its tolerance. Both variants are indistinguishable from round-off there,
  and the 0.01-dex bound for cheaper schemes sits only about 4x above that floor.
- Adopted in the fused path (snapy fork 36e8b84): the guarded fewer-divisions form, round-off class. The fp32-weights
  form is held: 9 % more per step on this card, and less on fp64 hardware.
- Tests:
  - a new kernel test compares the fused WENO5 with the default reconstruction on 17 stress lines to 1e-12 of the
    largest input within 4 cells; it fails on the unguarded form and on the fp32 form;
  - the fused-path test compares the flow bit for bit, PLM decks bit for bit and WENO5 decks to 1e-12 of the local
    largest value, and the positivity-limited cells must agree.
- Branch cleanup after an independent whole-branch review (snapy fork 5767512):
  - On a column split in x1 with the covariance on, the fused path stopped the run; that case now keeps the default
    lateral flux, with a CUDA test.
  - Every fused launcher checks its inputs.
  - An unused kernel branch is removed.
  - CPU CTest: every snapy test passes, except test_straka (ncrcat not installed).
- Negative tracers: the snapy group's donor-limit fix (their branch at e3100503), at C48x100 with 41 tracers over 40
  steps: 20 of 40 steps with negatives on main (as before), none with the fix, and the transfer-skipped arms
  identical. The fused path's transfer kernel reproduces today's transfer, so it needs the same limit when the fix
  merges.
- The C++ coupled loop's own glue around the solver call is about 25 ms per step on the box (the default path,
  120 steps): too small for a fused gather/scatter kernel to pay off.

### 2026-11-04 05:01 to 09:00: C64 model day, sulfur on the box, 3 days on the sphere, the dry-source carry
Data in experiments/2026-11-04_night (summary.txt).
- C64x100 hot Jupiter (2,457,600 cells), 69 tracers, chemistry after every 50 s step, one model day after a 1,728-step
  spin-up without tracers:
  - 79.5 min for the day, 1.58 s flow and tracers plus 1.18 s chemistry per step;
  - 1.32 steps per call, 0.42 us per cell-call, no failures, element totals within 1.1e-13 over the day;
  - peak 41.3 GB of 48 in the Python driver. A C++ loop with a two-kernel version of the work around the solver call
    has the same peak as with torch operations (12.6 GB on the box), so the transport sets the peak and the Python
    driver's diagnostics add the rest (inferred). That version saves about 1 % of the step, so it is not adopted;
  - negatives after transport in every step (up to 117, worst -3.7e-16), far larger than at C48.
- SNCHO_photo (88 tracers) on the box, the first 100 steps: 1.21 us per cell-call at 1.96 steps per call (2.6x the
  NCHO cost); one cell caught by the drift guard and recovered; no negatives; 17.2 GB on 921,600 cells.
- Cubed sphere C48x100, 69 tracers, 3 model days, with the ghost refresh:
  - 37.7 min per model day, with 1.14 s flow and tracers per step (the default path took 2.18 s);
  - chemistry at 0.444 us per cell-call, 1.40 / 1.37 / 1.41 steps per call by day, no failures;
  - element totals within 2.4e-13 over the 3 days;
  - negatives after transport in 2,668 of 3,705 steps (worst -3.0e-18).
- Fused dry-source carry (snapy fork 11e9475). With scalars and any forcing (gravity suffices), MeshBlock carries
  the forcings' dry-density increment into the scalars every stage. That was about 10 passes over the tracers, even
  when the increment is zero; it is now one bitwise kernel. Measured back to back at C64x100 with 69 tracers:
  1379 -> 1238 ms per step. The default path takes 3491 ms, so the fused path is now 2.8x faster.
- Run-to-run spread of these timings: the same build measured 1327 ms and 1379 ms five hours apart (about 4 %).

### 2026-11-04 09:00 to 11:44: the box over 8 model days; ghosts after the chemistry
- Box, 69 tracers, 8 model days (9,874 steps), fused path with the dry carry:
  - 18.6 min per model day; per step 0.36 s flow and tracers and 0.55 s chemistry;
  - 0.53 us per cell-call, with 1.46, 1.40, 1.50, 1.53, 1.90, 1.99, 1.72, 1.73 steps per call by day as the flow
    develops (temperatures 1139-1863 K after day 1, 1070-1984 K after day 8);
  - no failed cell, every guard count 0, per-cell element change through the chemistry at most 9.6e-12.
- Transport conservation held at round-off for 4 days, then broke: element totals by up to 1.2e-13 per step from
  day 4.6 (1.4e-10 over the run), fast radicals (N_2D, O_1, then NO2, N) by up to 6.9e-6 of their totals per step.
  - Cause: the driver writes the chemistry's output into the interior after Mesh.forward's last ghost exchange, so
    the next step's first stage reconstructs from stale periodic ghosts. It is the same mechanism as the heating.
  - With a ghost exchange after the chemistry, over 400 steps: the per-species transport budget falls from 6.5e-13
    (median; max 1.7e-12) to 3.6e-16 (max 4.4e-16), and the negatives from 4,168 to 3,840.
  - Fixed in the box, sphere and C++ drivers, which still agree bit for bit.
  - The 3-day box run above was not affected at its precision (round-off through day 4). The 8-day numbers other
    than the budgets (cost, steps per call, failures) stand.
  - For a coupled snapy step: anything that writes scalar_s after the stages needs a ghost exchange before the next
    step, or must also update the ghost cells. snapy's own kintera step in run_hydro.cpp updates whole arrays.
- The exchange after the chemistry costs about 1 ms per step on the box (rough: that run shared the GPU).
- CTest at the final branch commit (d824671): on the CPU, every snapy test passes except test_straka (ncrcat not
  installed); with the GPU visible, 154 pass, 1 is skipped (needs two GPUs) and test_straka fails the same way.

### 2026-11-04 14:57 to 15:39: structure of the C48 hot Jupiter after one coupled day
- C48x100, 69 NCHO_photo tracers, fused path with the dry carry, chemistry every 70 s step, dark, 1,231 steps (one
  model day) from the earlier spin-up (1.2 model days of flow without chemistry); final state dumped for figures
  (experiments/2026-11-04_structure; data in 2026-11-04_gcm-chem/runs/structure).
- Per step (medians): 1.08 s flow and tracers, 0.74 s chemistry (solver 0.62 s, 0.45 us per cell-call), 1.40 solver
  steps per call. No failed cell, every guard count 0. Per-step element budgets of transport and chemistry at
  round-off (about 1e-16); per-cell element change through the chemistry at most 3.2e-13. 16,268 negative tracer
  entries clamped over the run. Peak memory 24.9 GB.
- The atmosphere is young (2.2 model days in all): above about 1 bar it has a day-night contrast (substellar peak
  1,822 K at 0.19 bar), an equatorial jet of about 1.0 km/s near 0.1 bar and cold mid-latitude night-side gyres where
  CH4 at 0.1 bar reaches 1.0e-5 (day-side median 2e-9; CO varies by 2 %); below 10 bar it is still within 25 K of
  the 1,500 K start. The chemistry was relaxed to each cell's T and P at the start, so the maps mostly follow
  temperature, not quenching.
- bench/gpu_speed.py at e4ba969 on this card (1,382,400 cells; experiments/2026-11-04_gpu-speed), the reference
  for the fp64-GPU comparison asked in the group thread: NCHO_photo 0.35 us per cell-call dark (1.00 steps per call),
  0.64 us lit (2.58 steps); SNCHO_photo 0.64 us dark (1.00), 1.18 us lit (2.64). No unfinished cell, guard counts 0.
  The lit load (J = 1e-4 s^-1 on every branch above 0.03 bar) keeps the gas changing; it is not a realistic J.

### 2026-11-04 15:47 to 22:40: SNCHO_photo_network_2025 becomes the sulfur network
Data and scripts: experiments/2026-11-04_net2025 (plan.md, reviewed before the work; the logs named below).
- The network: 91 species, 532 thermal reactions, 61 photolysis branches (old: 88 / 513 / 60). New species C3H5,
  CH2CCH2, CH3CCH, H2CS; C3H4 gone; new branch H2CS -> H2 + CS; CS2 + H -> SH + CS replaced by SH + CS -> CS2 + H with
  its own rate; a new condensation line H2O -> H2O_l_s, skipped like S8 -> S8_l_s. 4 warps per cell, 184 registers,
  2 cells per block; the recovery matrix no longer fits shared memory (global-memory recovery).
- Truth generator: the 2025 network added, condensation pinned off (it was off by default); the C twin bitwise equal to
  stock VULCAN-0D (3,000 trials). Photolysis table: a VULCAN-JAX 1-D WASP-39b run on the 2025 network (the paper
  setup with the network swapped; 4.5 min, converged; SO2 up to 2.6e-5 at 1e-5-1e-3 bar); its common branches are
  within 0.0023 dex of the old run's.
- Parity, tests/kincuda_check.py --network (new option; reference libref8_SNCHO25.npz; also scores H2S, SO2, COS,
  S2): 27 of 28. The 2,000-call chain exceeds the count rule: 2,596 cell-calls above 1e-6 dex (cap 1e-5 holds; worst
  2.1e-6), all in 2 of 1,024 cells at 2,500-2,900 K and in H2S, COS, S2 only; step counts identical everywhere; the
  error grows overall (3e-7 at call 100, 1e-6 near call 660, 2.1e-6 at 2,000; not monotonic) together with the
  kernel's sulfur element share (up to 2.5e-6 relative by call 2,000; the other elements up to 1.2e-8). Supported
  hypothesis: rounding drift of the scarcest element over a long chain (the reference's own element budget was not
  measured, and equal step counts do not prove equal step sequences). The rule was not changed; a review agreed to
  keep it strict and report 27 of 28. NCHO with the edited check: 28 of 28.
- Drift survey (the prescribed stress cases): r = 1.35e-8, so the allowance stays 1e-6 (old sulfur network 1.2e-7,
  NCHO 5.6e-9). One 1e10-s relaxation call flags a few cells on every network; the guard recomputes them, and on the
  2025 network one lit cell of 65,536 stays failed (it keeps the main path's state and is counted).
- Derivatives: 101 of 101 on the 2025 network (the old one had 99 of 101) and on NCHO. C++ export bitwise (photo,
  thermal, forced recovery; the forced case, max_steps 2, leaves all 2,029 recomputed cells failed, so it compares
  the failure path only). 142 hard cold lit cells: none unfinished. Recovery under 65,536 cells (max_steps 100):
  36,851 / 53,266 cells recovered dark / lit, 4 / 91 still failed; the recovered cells that the main path finished
  within 4,000 steps are within 2.8e-4 / 7.5e-3 dex of its answer; x, h and steps bitwise in a permuted batch.
- VULCAN truth (data/truth_v2): boxes_sncho25, the 2,048 rows of boxes_sncho2's design with the 2025 J table (1,930
  accepted; 117 at the step cap, 1 quarantined), and boxes_sncho25p, 404 targeted lit boxes (350 accepted). Paths:
  qual_sncho25 (208) and daynight_sncho25 (64), replayed from the NCHO paths with H2S = 1.41e-5 Z per H atom.
- Scores (score25.py, score_paths25.py): 1,930 boxes, 1 with a miss (box 1939, 4.34 dex S2; dark worst 0.030);
  panel 1 miss (box 96, 0.155 dex S2); paths 312,000 states, 0 misses, worst 0.0072. New species within 0.013 dex.
  Guard: nothing recomputed.
- Radau (radau25.py; stopped at a time cap: 116 of 130 requested boxes): box 1939 is VULCAN's error again (maxichem
  0.084 from Radau). Box 96 (386 K, 9e-5 bar, lit, mixed start) is maxichem's (box96_tol.log): 0.154 / 0.161 / 0.161
  dex at rtol 1e-3 / 1e-4 / 1e-5 with atol 1e-25; 0.003 at atol 1e-28, 0.002 at 1e-30 (219 steps against 180), 1e-4 at
  rtol 1e-6 and atol 1e-30; VULCAN 0.0009. The author keeps atol 1e-25. 54 boxes without a VULCAN result have a Radau
  reference: maxichem within 0.016 dex; 60 more: maxichem within 0.0021, VULCAN within 0.0011. Without any reference:
  118 boxes (10 unreacted starts whose Radau run did not finish, 108 starts that need a VULCAN donor run).
- Coupling: the drivers now start a sulfur network with H2S (the earlier "SNCHO_photo box" number had no sulfur in the
  gas: superseded). First runs had H2S = 1.41e-5 x H2, i.e. S/H 7e-6, half solar (runs/box2025*_halfS): box, 91
  tracers, 100 steps: 1.25 us per cell-call at 1.98 steps, no failure, transport budgets 4e-16, chemistry 4.5e-15,
  17.7 GB; lit (J of the WASP-39b layer nearest in P times cos z) the same within 1 %. At the box's pressures (above
  7e-3 bar) only S4, HCO, NO2-type branches have large J, so this does not test SO2 photochemistry. Python and C++
  drivers bitwise equal (dark, 23 and 8 steps, default atol). Drivers take --atol. Rerun with solar sulfur (S/H
  1.41e-5): 1.24 / 1.25 us per cell-call dark / lit at 1.96 steps, no failure, transport budgets 4e-16, chemistry
  5.3e-15, per-cell element change 1.3e-11. Python and C++ drivers also bitwise equal at atol 1e-30 (8 steps).
- Speed (speed25.py): settled 0.684 / 0.651 us dark / lit; mixing bed 16.9 us at 29.0 steps; atol 1e-16 0.661 /
  0.658 and 9.05 us at 15.1. About the same as the old sulfur network's numbers (measured on an 89-species layout
  that included a condensate, so not a controlled comparison). Transport at C64x100 (bench_scalar_transport.py,
  fused, idle GPU): 0 / 69 / 91 tracers 0.31 / 1.08 / 1.33 s per step (11 ms per tracer), 4.5 / 30 / 38 GB; default
  path at 91: 4.35 s, 43 GB. The flow-only time was 0.47-0.49 s on 2026-11-04 morning; not explained.
- Mistakes found on the way: the first Radau script re-read the whole truth file per box (about 100 GB per worker;
  killed by the system's out-of-memory daemon); the first path run had no sulfur in its starts (VULCAN refused every
  path); the drivers' first sulfur start was half solar. A results review (2026-11-04 night) also asked for: accuracy
  gates in the experiment scorers (they gate integrity only), a C++ recovery case that finishes, a lit Python/C++
  coupling comparison (the C++ driver has no J yet), and the 91-tracer C48 run; the non-default atol comparison is
  done (above).

### 2026-11-04 22:40 to 23:45: code audit; style pass
- Four independent read-only audits (two of maxichem, two of the snapy branch) for leftover code, needless complexity,
  missed speed and errors. Agreed findings (verified by the auditors in the code; not yet acted on):
  - Errors: T, P, dt and h are not checked for shape or device (Python and C++; a short h is read and written out of
    bounds); a row of x summing to 0 or P <= 0 passes the input screen and burns the whole step and recovery budget
    (counted non-finite, not bad input); a cell that finishes in the recovery but fails its final guard returns
    done = True; in the derivatives h = NaN splits the dt / h sensitivity in half; a three-species subtraction lacks
    the fma guard its two-species twin has (latent parity risk); ADSolver.total is int32; CUDA modules are not
    unloaded when a Python solver is deleted.
  - Leftovers and complexity: the rate, Jacobian, error-test and right-hand-side arithmetic is repeated in the call,
    recovery and derivative kernels (about 300 lines whose bitwise parity rests on the copies staying equal); the
    non-aliased shared layout is unused by every network; export's meta.json is never read; ADSolver's cap is never
    passed; explicit reverse laws serve only VULCAN-0D's venot19.
  - Speed: the chemistry is within tens of percent of the practical limit for this algorithm on the RTX card; no 2x
    remains; small items: a bank-conflict-free dense tail (1-5 %), a compact recovery table that fits shared memory
    for 91 species (recovery only). snapy's fused transport: WENO5 with literal sparse stencils (7-11 % of transport),
    earlier release of the face arrays (about -1/3 of the memory per tracer), skipping ghost rows, theta scaling folded
    into the divergence, a wider x1 launch, two fewer divisions; together perhaps 5-30 %, about 1.4x above an
    estimated practical limit. snapy findings also: the fused path's fallbacks are silent; its fp32 code is unused.
  - Records: the open items "the drift survey cannot run" and "a C launcher is needed" (above) are out of date.
- Style (the author's policy, CLAUDE.md "Style"): Ruff format and Ruff check at 120 columns on maxichem/, tests/ and
  bench/; clang-format (Google, 120) on cpp/; three stale comments corrected (warps, cells per block, NaN pivots).
  Proof that nothing else changed (experiments/2026-11-04_style): equal syntax trees except the seven lambdas made
  functions, the CUDA templates' whitespace and comments, and one import order; every compiled kernel (call and
  derivative, NCHO and 2025) bitwise equal.

### 2026-11-05 00:00 to 01:15: snapy transport speed and memory; the audit's error fixes
Approved by the author ("I agree to everything you recommend"). Plan and logs: experiments/2026-11-05_fixes. The plan
had one independent review.
- snapy, branch isaac/maxichem-coupling (commit 7f6ce04):
  - WENO5 of the fused path written out as fma chains over the uniform-grid table's nonzero entries, in index order:
    bitwise the previous fused results (state hashes of hydro_u and scalar_s equal, default and fused path, 69
    tracers, small deck 20 steps and C64x100 5 steps; the unscaled formula and fp32 are not covered by those decks,
    and there a zero face may change its sign). The fused kernels refuse any other table (checked once per table
    state, in its dtype). GPU time of the WENO5 kernels at C64x100, 69 tracers: 404 -> 343 ms per step.
  - The ratio and each face pair freed once used: peak memory 30.1-30.4 -> 26.7 GB (69 tracers), 38.2-38.4 -> 33.7 GB
    (91). The peak stays at the lateral flux, where both face pairs coexist for the cube-sphere exchange, so the gain
    is about 1.8 tracer arrays per tracer (0.37 -> 0.32 GB), not the 5 first estimated.
  - Interleaved timing, idle GPU, fused path, per step: 69 tracers 1132 / 1151 -> 1051 / 1088 ms (-6 %), 91 tracers
    1395 / 1371 -> 1310 / 1312 ms (-5 %).
  - Every part that runs the default path while the switch is on warns once with the reason.
  - Tests: the six fused-path CTests pass; the suite from test 928 on passes except test_straka (ncrcat missing) and
    three restart / output tests that fail only when run four at a time (they pass alone).
- maxichem (this commit):
  - Inputs checked before any launch (Python and C++): shapes, dtype, device, finite positive tolerances, a step
    budget whose recovery tries fit an int32. Python solvers take an explicit device index.
  - Bad input also: no mole fraction above 0, n_tot <= 0, dt < 0 (counted bad_input, not stepped, done 0; it burnt
    the whole step budget before). The test is order-free (a positive maximum, not the sum).
  - A recomputed cell is done only if it also passes the final guard. A non-finite starting step acts as dt (also
    -inf), in the derivatives with no sensitivity to h. The three-species rate difference rounded separately, as the
    two-species one. ADSolver.total int64. Modules unloaded when a solver is deleted; NVRTC programs always destroyed.
    gpu_speed sums the guard counts over every call.
  - Checks: parity NCHO 31 of 31 (3 new cases), 2025 30 of 31 (the known chain case, same numbers); kintest quick
    equal to the saved run; derivatives 102 of 102 on NCHO_photo, SNCHO_photo_network_2025 and CHO_photo; C++ export
    bitwise on both networks (photo, thermal, recovery); and the six saved pre-fix cases reproduced bit for bit by
    the new solver (x, h, steps, done, counts), so nothing changed for valid inputs.

### 2026-11-05 01:15 to 04:25: second audit; end-to-end tests with known answers
Data and scripts: experiments/2026-11-05_e2e (plan.md; findings.md lists every bug; results.jsonl in R/, T/, C/, M/;
audit_round2.md). Every test's answer was known independently of the code under test (analytic, a conservation law,
VULCAN-0D, an independent equilibrium code, an exact symmetry), with the threshold set before the run.
- Second audit (errors, memory, speed; read and measured, not acted on except the driver fixes below; details in
  audit_round2.md). maxichem: a negative mole fraction is not screened (a species below -atol that nothing destroys
  can burn the whole step budget); the Python solvers do not make their device current; a failed constructor prints
  an error at deletion; dt = 0 resets h. Solver memory is inputs and outputs only (scratch 10-30 MB). snapy memory:
  the C64 / 91-tracer peak (33.7 GB) sits at the dim-2 lateral flux with 12.9 tracer arrays per tracer; four changes
  could bring it to about 8 (estimate). snapy speed: tracer work is 712 of 1,039 ms per step at C64 / 69 tracers;
  seven changes are estimated to save about 200 ms; no error found in the fused path.
- Five physics bugs, all in the experiment drivers (gcm_chem.py, gcm_box.py, box.cpp and its Python twin), fixed:
  1. The heating was added after each RK3 stage at the full dt: 11/6 of the intended heating per step. Now
     heat dt wght2(stage). An energy-closure test (snapy's E plus its corrected potential energy, heated minus
     unheated) gives 1.8333334 before and 0.99999999 after, on the sphere and the box.
  2. Chemistry and radiation read T and P from hydro_w, one RK stage old. Now rebuilt from hydro_u after each step;
     equal bit for bit to snapy's own stage-start primitives.
  3. The grey longwave band was 10-1e4 cm^-1, which holds 98.7 % of sigma T^4 at 1500 K and 83.9 % at 2500 K (a
     radiative-equilibrium column came out 6.2 % hotter at depth than Guillot 2010). Now 1-1e5 cm^-1 (1.6-1.9 %, the
     two-stream closure difference).
  4. The Python Mesh.exchange_ghost_zones(vars) applies the conserved type to every array, so at cubed-sphere seams
     tracer rows 2 and 3 (H2O and H in both networks' order) were rotated as momentum: seam ghosts off by up to 295 %,
     a constant mixture drifting up to 1.2e-4, a blob crossing a seam split by 1.3e-3 of its peak after 50 steps.
     Mass stayed conserved and the box was not affected. Now one typed call per kind (hydro_u conserved, scalar_s
     scalar), then scalar_r = scalar_s / rho: a constant mixture stays to 1.9e-14 after 300 steps (C48, default and
     fused paths).
  5. After the driver writes the interior (heating, chemistry), the top and bottom wall ghosts stayed as Mesh.forward
     left them: energy leak 5e-9 of the heating, 2.8e-5 per O(1) composition change in the top cells. Now the walls
     are re-mirrored as snapy's reflecting rule does; right after Mesh.forward the whole refresh changes nothing, bit
     for bit. Also the dump's meridional wind was v_theta (southward) labelled northward (output only).
  Rules for the C++ coupling: add a frozen source as dt wght2(stage) per stage; exchange scalars with the scalar
  type; re-apply the walls after writing between steps.
- Passed: grid volumes and areas (4e-16); mean of max(0, cos z) = 1/4 at second order; the drivers' lon/lat equal
  snapy's cell positions (all 12 seams); Coriolis axis and sense; shortwave beam F mu exp(-tau/mu) to 1.7e-13 and
  the global absorbed power F pi r^2 (1 + 8e-5 at C48); longwave OLR = pi B of an isothermal thick column (2e-16);
  q dt in the energy row raises T by q dt / (rho cv) (1.5e-13); a resting atmosphere stays at 5.5e-7 m/s after one
  rotation (C24); gas and tracer mass drift -5.55e-17 per step (the RK3 weights' rounding); identical tracer rows
  stay identical bit for bit; fused = default bit for bit on every run; the Python and C++ box drivers bitwise
  equal after every fix; flow bitwise unchanged with or without chemistry.
- Chemistry against independent answers: the coupled box at rest (every cell a 0-D box, unrelaxed start, 1 model
  day) vs VULCAN-0D per layer: NCHO_photo dark worst 1.6e-4 dex, SNCHO_photo_network_2025 lit (J of the WASP-39b
  table) 6.8e-4 dex, 0 misses, while the gas moved up to 5 dex; the sphere at rest gives every column of all six
  panels the same chemistry to 3.7e-9. Equilibrium after long calls vs an independent Gibbs minimisation on the same
  NASA-9 data: 6e-4 dex on both networks (cells at 1000-1400 K need 1e13 s or more to settle). Single reactions vs
  Radau 2.2e-3 dex; photolysis alone vs the exact solution 0.009 / 0.019 dex (the solver tolerance); elements over
  2,000 lit calls 1.7e-13; --jref photolysis mapping exact; to_x / to_q keep mass and elements to 7e-16.
- Data: NASA9/HNO3.txt holds three sets (below 200 K, 200-1000, 1000-6000 K) and every reader takes the first two
  (HNO3 equilibrium constant off by 7.6 dex at 999 K). Fixed in VULCAN-JAX (commit 01ce3b8, the below-200 K set
  dropped); only NCHO_full_photo and SNCHO_DMS carry HNO3, so neither default network changes. TiH (TiSNCHO only)
  has a low-T set with negative Cp.
- Observations, not bugs: FastChem's data differ from NASA-9 by 0.1 dex for CH4 and HCN; the drivers' start (one
  1e10 s call) is not equilibrium below about 1400 K; --jref gives cells below the table's 0.945 bar layer that
  layer's J; the longwave cannot damp a 2-dz temperature checkerboard (level T are means of layers).
- C48 hot Jupiter, one coupled day with fixes 1-4 vs the old run (same steps, 69 tracers, fused): T at 0.18 bar,
  day mean 1645 -> 1611 K, night 1306 -> 1355 K, maximum 1854 -> 1760 K; equatorial zonal wind at 0.1 bar 1018 ->
  780 m/s; CH4 at 0.1 bar, night median 2.6e-7 -> 6.1e-7. Per step: dynamics and tracers 0.93 s, chemistry 0.67 s
  (solver 0.42 us per cell-call, 1.26 steps per call); peak 17.5 GB allocated (24.9 before the frees). No failed
  cell, guard counts 0, budgets at round-off. Earlier runs on these drivers (the C48 structure day, the C64 day,
  the 3-day sphere run, the box runs) carry bugs 1-3 (and 4 on the sphere); timing and memory are unaffected.

### 2026-11-05 04:20 to 14:50: 10 model days with and without chemistry; three solver fixes
Data and figures: experiments/2026-11-05_gcm10d (figures/, runs/compare_10d.json), experiments/2026-11-05_threefixes.
- C48x100 hot Jupiter, every driver fix above, one spin-up (500 CFL steps), 12,343 steps of 70 s, dark: A with 69
  NCHO_photo tracers and chemistry every step; B the same tracers carried by the flow only. No failed cell, guard counts
  0, 46,873 negative entries clamped (worst -1.3e-20), budgets 5e-16 (transport) and 1.2e-15 (chemistry) per step, total
  -7e-13 (the RK3 weights' rounding). 5 h 28 min and 3 h 9 min; per step 0.95 s flow and tracers, 0.63 s chemistry (0.39 us
  per cell-call, 1.16 steps per call, at most 110); peak 17.4 GB. Smallest CFL step 131 s.
- Known answers: the flow is bitwise the same in A and B at all six dumps (chemistry has no feedback here); the CFL step
  is equal at all 12,343 steps; C/O, N/O, C/H and He/H stay uniform to 6e-4 dex against a preset 0.01. The C++ and Python
  box drivers stay bitwise equal after the last ghost change.
- Day 10, 0.1 bar, day / night medians with chemistry against without: CH4 4.8e-9 / 2.0e-8 against 4.8e-7 / 7.9e-7, H
  2.3e-4 / 1.3e-6 against 9.4e-5 / 7.8e-5, NH3 8.7e-8 / 1.7e-7 against 1.9e-7 / 2.6e-7; H2O, CO and N2 at most 0.01 dex
  apart at all five levels; at 50 bar nothing beyond 0.04 dex. Against an independent Gibbs minimisation: within 0.04
  dex at 10 bar and below; aloft NH3 and HCN sit above equilibrium (median 0.4-0.8 and 0.6-1.3 dex), night-side CH4 up
  to 5.6 dex below, CO up to 0.61 dex above in the coldest cell (712 K, 0.01 bar). The atmosphere is young: within about
  15 K of the 1,500 K start at 10 bar. Equatorial wind at 0.1 bar 780 m/s on day 1, 1,725 on day 6, 1,447 on day 10.
- A group review of the heating confirmed the stage weights and the longwave band, and measured that heating after the
  vertical implicit solve departs from heating inside it as the acoustic Courant number grows. One column under the
  deck's frozen substellar heating for 6 h (research hook, not in the branch): the two differ by 0.2 % in the energy
  response and 0.14 K rms at Courant 3.4 (this deck's 70 s step), 0.7 % and 0.49 K at 14, 1.4 % and 0.92 K at 27, 3.5 % and
  2.0 K at 53; at 106 the outside form stops with a redo request and the inside form runs.
- The "zigzag" (second difference of T up to 110 K at day 10, 0.01 bar) is not a layer-to-layer pattern: an alternating
  fit over 6 levels holds at most 1e-4 of the windows, at most 6 K. The large values are sharp fronts; 83-99 % of the
  cells above 10 K in the top 0.1 bar have Ri < 1/4, with winds up to Mach 1.9 (Fromang et al. 2016 report shear
  instabilities steepening into weak shocks at a few mbar).
- maxichem (this commit): a negative mole fraction is bad input in the call, recover, guard and derivative kernels;
  the Python solvers make their device current around module loads and launches (not exercised: it needs two visible
  GPUs); a constructor that fails no longer prints an error at deletion, and a partial setup unloads its modules.
  Checks: parity NCHO 32 of 32, 2025 31 of 32 (the known chain case, same numbers); kintest quick unchanged; derivatives
  104 of 104 on NCHO_photo, SNCHO_photo_network_2025 and CHO_photo; C++ export bitwise on both networks; the pre-fix cases
  reproduced bit for bit. An independent review found no logic error.
- VULCAN-JAX: the HNO3 NASA-9 fix is pushed (normal tier 324 passed, 0 failed; CI green).

### 2026-11-05 14:50 to 18:53: the 2025 chain failure fixed; sponge, radiation step, time step, lid
Data: experiments/2026-11-05_chainfix (checks/, speed.jsonl), experiments/2026-11-05_gcm10d/research.
- The one known parity failure (2,000-call chain on SNCHO_photo_network_2025, worst 2.1e-6 dex, H2S, COS and S2 in 2 cells
  at 2,500-2,900 K) came from element shares walking through rounding, up to 2.6e-6 relative over the chain, on different
  paths in the kernel and in the reference. Numerics change: a cell that passes the guard is now corrected,
  x' = x (1 + lambda . a_i), with (A^T X A) lambda = e0 (s1 / s0) - e1, by LDL^T without pivoting. An element whose pivot
  falls below 1e-10 of its diagonal is left out; no species changes by more than GMAX = 1e-3, relative; the row is then
  renormalized. Shares near atol and elements tied together are not fully restored. The step control and the derivative
  kernels' tangents and cotangents ignore it. A first version with one factor per element was rejected on review (noise
  moved through CO into C); this form replaced it.
- New references, made by the generator with the same correction: libref8.npz, libref9_SNCHO25.npz. The drift allowance r
  is now the largest drift at which the guard still flags a cell, since the returned x no longer shows it; kintest labels
  its drift column accordingly.
- Checks on the final code: kintest quick 0 misses and 0 large on all paths and stored states (worst 0.014 dex), no path's
  worst error changed by more than 0.01 dex. Parity 34 of 34 on NCHO_photo and on the 2025 network (a share test on the
  lit chain is new): chain worst 6.9e-10 and 2.9e-8 dex, lit chain 1.6e-9 and 2.8e-9 dex, shares 2.2e-13 and 2.1e-13 over
  the chain, 5.7e-14 and 6.9e-14 over the lit chain. Derivatives 104 of 104 on NCHO_photo, SNCHO_photo_network_2025 and
  CHO_photo. C++ export bitwise on both networks. Two independent reviews; the second's points (the limit, fewer
  operations, the docs, the lit test) are in. The planned outside review is still missing.
- This commit changes results: on the saved cases x moves by at most 6.2e-13 relative (2025 lit), h, step counts, done
  and guard counts stay bitwise. The earlier same-bits statements hold up to the previous commit only.
- Cost, idle GPU, two interleaved pairs of the previous commit and this one, us per cell-call: NCHO dark 0.359 to 0.362
  (+1.0 %), lit 0.659 to 0.662 (+0.5 %); 2025 dark 0.682 to 0.687 (+0.7 %), lit 0.987 to 0.992 (+0.5 %).
- Flow studies, C48 hot Jupiter, flow only, no chemistry unless stated, against the explicit run or the finer run. T rms
  by band 0-0.02 / 0.02-0.1 / 0.1-1 bar. Two runs from the same state differ by 6e-7 K at day 1 and up to 6e-5 K at day
  2, the floor for all of these. Heating placement is in the entry above.
  - Sponge (top-sponge-lyr), day 1 against none: tau 1e4 s over 600 km: |u| max 3,945 to 2,874 m/s, T 50 / 57 / 15 K;
    tau 3,600 s over 1,200 km: |u| max 2,092 m/s, T 71 / 76 / 30 K. A sponge changes the circulation well below itself;
    none is used.
  - Radiation step against 150 s: day 1, hourly 3.4 / 2.6 / 1.0 K and 600 s 0.5 / 0.5 / 0.15 K; day 2, hourly 26 / 27 / 6.4
    K and 600 s 12 / 7.6 / 1.8 K. Hourly radiation is inaccurate, not unstable, at 100 layers.
  - Time step, day 1, 70 s against 35 s (flow and chemistry both): T 2.1 / 1.4 / 0.33 K. Against explicit: 70 s 3.4 / 2.6 /
    0.61 K, 35 s 1.5 / 1.3 / 0.30 K: the error halves with the step. Composition, nine species: median at most 0.0009
    dex, 99th percentile at most 0.08, largest 1.08.
  - Lid, day 2, radiation every 300 s, from rest: 100 layers (top 8.4 mbar) against 120 (top 1.1 mbar), on the 100 shared
    layers: T 103 / 64 / 24 K, 4.2 K below 1 bar; zonal wind 1,152 / 592 / 129 / 13 m/s; equatorial wind 0-0.02 bar -108
    against 807 m/s. The lid changes the flow at all pressures above 1 bar. The high-lid run reaches
    283 K, below the chemistry's 300 K limit (273 K with hourly radiation). 140 layers fails in all 4 variants
    (hourly, 300 s, lid-temperature fix, both) within about 3 model hours, T as low as 27 K before the stop; the
    lid-temperature fix also stops 120 layers at cycle 184. The next step depends on the lid decision (open).

### 2026-11-05 18:53 to 19:44: the sphere drivers' heating formula was wrong; runs stopped
Data: experiments/2026-11-05_gcm10d/research (column_night.py, top_budget.py, re_start.py, flow_exp5.py, runs/re*, runs/pd*).
- The bug, in the experiment drivers on the cubed sphere: heat = -(A_top F_top - A_bot F_bot) / V with the column's
  plane-parallel fluxes. A grows with r, so a flux constant in height gets a false divergence of 2 F / r. A parallel beam in
  vacuum has none; a radial-only form leaves out what the side faces carry. The term is per volume, so per mass it grows as
  1 / rho. It cools the night side and heats the day side, most near the lid. Now heat = -(F_top - F_bot) / dz. Not a snapy,
  pyharp or maxichem bug. Box drivers are not affected (A constant). Energy closure and the check of global absorbed power
  passed because the old form is conservative, so they could not catch it.
- Supersedes: every sphere result above carries the term, including the 10-day runs and their figures and the lid, sponge,
  radiation-step and time-step tables of the previous entry. Its size away from the top is not measured.
- How it was found. 140 layers crashed within about 3 model hours from every start. The cold spot is the top cell over the
  uniform night side, at rest. Budget at 140 layers with the old form (rho about 3e-6 kg/m3): real longwave +0.024 K/s
  (heating; pyharp lies between exact grey solutions with D = 2 and 1.66), area term -0.08 K/s; their sum, -0.056 to
  -0.061 K/s, against -0.052 K/s observed beyond the adiabat. Not the cause: the top level's temperature (T_n against
  2 T_n - T_n-1: T_min 133.0 K against 133.7 K at 7,000 s, and T_n stops 120 layers at cycle 184) or the start (radiative
  equilibrium only moves the stop from cycle 151 to 191).
- One Cartesian column, same gas, longwave only, against a hydrostatic Lagrangian column: T at the lid after 12,600 s is
  1,073 K (implicit, 70 s and 17.5 s) and 1,076 K (explicit, 4 s), truth 1,077 K; P 0.129-0.131 mbar, truth 0.135. At rest
  without radiation it stays at 1,316 K. So snapy's lid and implicit solve are right; the contraction under the lid is real
  and small.
- C48 with the corrected form, radiation every 300 s: at 10,500 s, 140 layers from equilibrium has T_min 1,092 K (old form
  281 K; single-column truth 1,095 K). Stopped on request at 19:34 before day 1, with no crash in any of the six:
  T_min / T_max 1,005 / 1,755 K at 28,000 s (140 layers, equilibrium start), 971 / 1,718 K (140, 1,500 K start), 1,025 /
  1,771 K and 1,004 / 1,732 K at 31,500 s (120), 1,037 / 1,797 K and 1,024 / 1,753 K at 35,000 s (100). Old against
  corrected form at equal time, equilibrium start: 100 layers T 1,019-1,803 against 1,037-1,797 K; 120 layers 867-1,777
  against 1,025-1,771 K. No day-1 dumps of the corrected runs, so no lid comparison yet.
- Equilibrium column (Newton on level temperatures, the same longwave and shortwave calls, F / 4 as F / 2 at mu0 = 0.5):
  T 1,316-1,590 K, lid 15 / 2.0 / 0.26 mbar at 100 / 120 / 140 layers. On it the two top-level choices change the net flux
  at the top by 80 / 1.6 / 0.03 W m-2, out of 2.5e5 absorbed.
- gcm10d.py and gcm_chem.py now use the corrected form (experiment files). The reruns are in todo.md.
- The share walk fixed in the previous entry: largest change 2.56e-6 over the chain, -2.53e-6 at call 2,000 (sulfur, one
  cell). The 2.6e-6 above and the 2.5e-6 in the code's description are these two numbers.

### 2026-11-05 22:38: code cleanup baseline at commit 3f92275 (experiments/2026-11-05_cleanup_baseline)

The per-step commits these cleanup entries name (through 2026-11-06) were later squashed, so their hashes are not in the history.
- Saved for the cleanup's proofs: compiled kernels (kernels.json: 28 compiles of the solver and derivative kernels on
  NCHO_photo, SNCHO_photo_network_2025 and CHO_photo, dark and lit; each compiled twice, all deterministic), float64
  outputs of the call, recovery, jvp and vjp at 4,096 cells (outputs_*.npz), table hashes and solver info
  (outputs_meta.json). NVRTC 12.8.93, torch 2.10.0+cu128, NumPy 2.4.6.
- Checks: parity 34 of 34 on NCHO_photo and on the 2025 sulfur network; derivatives 104 of 104 on all three networks;
  C++ export bitwise equal in all six cases; kintest quick 0 misses, 0 large misses, worst 0.014 dex on boxes.
- Speed (bench/gpu_speed.py, 1,382,400 cells, GPU not otherwise in use, the check runs done before it started): dark
  0.359 us per cell-call at 1.00 steps per call, lit 0.659 us at 2.58 steps per call.

### 2026-11-06 01:37: code cleanup, steps 1 to 13 (experiments/2026-11-05_cleanup_baseline)
- Done: tool config, Ruff clean, kernels in real files (`maxichem/kernels/`) with a generated data header, the solver modules split,
  a format version on the export, names and a glossary, named constants, one table of kernel arguments shared with C++, `docs/`,
  `close()` and context managers, typed package (mypy clean), lazy `paths`, pytest layout, known-answer and order tests.
- No result changed. Against the baseline, at the end of each series (commits 37fbdc0, b17631d, e1f43c4, 6f44358): float64 outputs
  of call, recovery, jvp and vjp bitwise equal on 6 of 6 cases; parity 34 of 34 on both networks; derivatives 104 of 104 on all three;
  C++ export bitwise equal in all six cases; kintest quick 0 misses, its log unchanged. All printed metrics equal, except the one below.
- Compiled kernels stayed byte-equal through every kernel step but one: `constexpr` constants (fdf226d) change the cubins, so that
  step was proved by bitwise outputs. Speed against the commit before it (gpu_speed, two rounds each, GPU otherwise idle): dark 0.3616
  to 0.3628 us per cell-call (+0.3 %), lit 0.6613 to 0.6620 us (+0.1 %). The same kernels varied by 1 % between runs (0.3590 to 0.3626
  dark), so the change is within the noise. Decision: `constexpr` stays.
- New known answers, on invented networks with constant-Cp NASA-9 data: A -> 2 B at constant pressure, relative error 7.9e-6 at
  rtol 1e-6; long calls at 800, 1500 and 2500 K against the closed-form equilibrium constant, mass-action ratios within 1.3e-15; element
  shares within 1.3e-16. Rodas3's order measured on the decay at 4 to 32 steps: slope 2.96.
- Decisions: the `REVERSE` table had no source and no user (no `# irreversible` heading in any of the 11 networks), so it is removed;
  every proof stayed equal. The kernels' explicit reverse path stays, and only a hand-built reaction dict reaches it now.
- A correction in a check's report: "FD J at exactly 0" counted only cells with a plateau as judged, so it always printed 100 %. It
  now prints the judged cells: 3,404, 3,358 and 3,463, of which 21 %, 28 % and 24 % have a plateau. Its pass rule is unchanged: it has no
  coverage bar. The tangent tests gained a limit on the worst cell (1.0; measured worst 0.77).
- Found, not changed: the table builder refuses a network with few reaction terms ("too many Jacobian pieces"), so an invented network
  needs about 40 species and 30 padding reactions to fit. The smallest real network has 35 species and 149 reactions.
- clang-tidy reports 128 findings on the kernels, all style: implicit widening 77, narrowing 27, unset variables 22, widening casts 2.

### 2026-11-06 03:53: code cleanup, steps 14 and 15: shared kernel arithmetic, refusals, the guard (experiments/2026-11-05_cleanup_baseline)
- No result changed on valid input. Float64 outputs of call, recovery, jvp and vjp bitwise equal to the baseline on 6 of 6 cases after
  every item; reader and table hashes 31 of 31 on all 11 networks.
- Small networks: the README was right and the table builder wrong. A network with few terms per lane cut its Jacobian slots into more
  extra pieces than Y holds. The even share now grows until they fit; the real networks fit at the first share. Supersedes the "Found,
  not changed" line of the entry above. A 6-species network runs in every kernel (equilibrium, recovery, tangent, adjoint).
- clang-tidy: implicit widening dropped (77 of 128 findings, each an index or size that cannot reach 2^31); narrowing, unset variables
  and widening casts fixed; 0 findings. The hook reads `build/kernels/network.h`: regenerate it when the header's macros change.
- Shared arithmetic: rates, the bad-input test, stream words, stage right-hand sides and the step error are forced-inline helpers. Compiled
  kernels change, bits do not. Speed, gpu_speed, 69 species, two rounds each: dark 0.3618 to 0.3587 us per cell-call, lit 0.6612 to 0.6544.
- Long kernels: 10 of 13 now meet the limits. Backed out: the `call` kernel's split, bits equal but +1.1 % against the commit before it
  (dark 0.3577 to 0.3618, lit 0.6537 to 0.6608); with `call` whole, msolve and check_cell still split, 0.3580 and 0.6535. `call`, `jvp_cells` and
  `vjp_cells` keep a written exception; a further split of the last two needs 30 or more arguments per helper (a count by hand).
- Final speed at 67e07e9, one round: dark 0.3570, lit 0.6547 us per cell-call. Derivative kernels, base against final, NCHO_photo, 4,096 cells, two rounds each: vjp 705.9 to 700.0 ms (-0.8 %), jvp 338.4 to 342.3 ms (+1.15 %). The jvp rise is above the 1 % rule, its cause is not found and it is not backed out: open.
- Removed: the explicit reverse law and its table support; a hand-built reaction with one is refused.
- New: count slot `nonfinite_h` (export format 3). `photo=True` raises on `J=None`, in Python and C++: `bench/kintest.py` relied on it on its dark
  paths and now passes zeros; the GCM drivers and tests pass J. `lm` screened in the kernel: a non-finite one takes no try, and comes back with
  flag 2. `cap` and `drift` checked. `launch` adds to `total`.
- Reader: a reaction above the first heading raises. After the first heading a comment must be a known heading, `# id` column titles, a
  commented-out reaction or empty, else it raises: none of the 11 networks' comment lines trips this, their free text sits above the first
  heading. `MAX_GIBBS_JUMP = 1.0` on G / RT at 1000 K: largest of the 114 species 0.143 (TiH), then 1.6e-6; the old HNO3 file 17.62. A factor e is
  0.43 dex in an equilibrium constant, well above the 0.1 dex of a miss. Three species' files (HO2, CH3O2, SH) hold 30 numbers, and read equal.
- The parity bar stays at 1e-6 dex.
- pre-commit passes on every file; the git hooks and the marker line are in. Parity, derivatives, known answers, C++ and CPU tests ran in parallel
  on one card at 67e07e9, the truth and CPU tests again at d16f41e; compiles of equal sources came from a disk cache keyed by the source text.
- `MAX_GIBBS_JUMP` tightened from 1.0 to 0.15 (0.065 dex, under the 0.1 dex of a miss): all 11 networks still load. TiH sets the floor at 0.143; without
  it the largest is 1.6e-6 (TiC). TiH's low range is not a sound fit: Cp / R is 40463 at 100 K, 819 at 200 K, -23 at 300 K, 18 at 700 K and 332 at 999 K,
  against 5.6 at 1000 K on the high range. The join check does not see this. Only TiSNCHO carries TiH.
- TiH checked against its sources (experiments/2026-11-06_tih_thermo). `NASA9/TiH.txt` is identical in VULCAN's three pinned clones, so this is VULCAN's data, not
  a reader error. Against Burrows et al. 2005 eq. A2, the formula it was fitted to: high range within 0.01 dex; low range within 0.07 dex at 500-1000 K, but 0.11 to
  0.73 dex at 250-450 K. On 200-1000 K its Cp / R runs from -36 to 819; of the 120 files only liquid water H2O_l_s, which the reader never reads as a species, is also this far off.
  Separately, eq. A2 itself sits 0.44 to 0.48 dex above log10 K computed directly from the paper's constants at 700-3000 K (0.36 at 500 K), and Barklem+ 2016 (A&A 588, A96) agree
  with the direct sum to 0.05 dex at 200-3000 K. The offset is nearly constant in T, so it is not the dissociation energy; the cause is not found. Burrows's own Q(TiH) is
  within about 6 % of the direct sum at 2000-3000 K, so it is not Q(TiH). The dissociation energy's error, 0.091 eV (Barklem), is itself 0.46 dex at 1000 K and 0.92 dex at 500 K.
  G is all the reverse rates use; the T derivatives use the fit's enthalpy. TiH is not one of the 12 scored gases, and TiO + 1.5 H2 = TiH + H2O at 1 bar (estimate: pH2 0.85, pH2O
  5e-4 bar, TiO2 ignored) gives TiH / TiO of 1e-32 at 300 K, 1e-9 at 1000 K, 1e-4 at 2000 K, 5e-3 at 3000 K. No effect on the scored gases is expected (inferred, no run made).
  The file is unchanged.
- Decision: refuse the bad data. `MAX_GIBBS_JUMP` is now 1e-5, six times the 1.6e-6 of TiC, the largest sound fit. TiSNCHO is refused with an error naming TiH.txt, the other 10
  networks load. A test pins this, using the real files. The fix is in todo.md. vulcan3 is not changed: its reader already refuses TiSNCHO by default for the duplicate reactions.
- Results review of the cleanup. Derivative speed, commit before step 14.3 against 964c347, NCHO_photo, 4 alternating rounds on an idle GPU 1 (`review_speed/pair.log`): jvp 339.33 to 342.45 ms (+0.92 %),
  vjp 706.86 to 700.80 ms (-0.86 %). This replaces the +1.15 % above and is under the 1 % rule, so nothing is backed out. Two further rounds (`two.log`) put most of the jvp rise at the helper split (34fa3ef to 1023e5a).
- The review found two reader regressions, now fixed: free comments past the thermal sections were refused, and so were 40-number NASA-9 files such as PO's (its ranges meet to 2.5e-8). A non-finite `lm`, -inf included, takes no try;
  -inf used to switch a reaction off. docs/api.md states this, and nothing here uses it. Extra bitwise comparison of the arguments the proof skipped (lm, dlm, rc, drc, dJ, adapt=False), 3962d00 against 964c347, `review_speed/bits*.log`:
  NCHO lit 54 arrays, synthetic dark and lit 117; falloff, small and decay networks were not compared.
- `vulcan-truth` keeps `nega_cut` -1 although VULCAN-JAX moved to -1e4: it is this project's reference, and changing it could invalidate the stored truth sets (inferred).
### 2026-11-06 06:57: the checks' holes, and a read of the code for what the rule-driven cleanup left (experiments/2026-11-06_slop_baseline)
- No result changed on valid input. Reader and table hashes equal on all ten networks that load (28 of 28); compiled kernels equal on the three networks (28 of 28 cubins); float64 outputs of call, recovery, jvp and vjp bitwise equal on the six saved cases; `bench/kintest.py quick` bitwise equal, start against end, except timings.
- Holes closed in the checks, with no failure exposed. test_ad's finite-difference cases dropped a cell with a NaN tangent from both errors and coverage: any non-finite tangent outside flag 2 now fails, and all the FD cases count 0. Injected check (`inject_nan.py`): 50 NaN cells of 1,024 fail the verdict, and the plain count of judged cells would have passed. A missing `VULCAN_THERMO`, reference file, CUDA toolkit or uv now fails instead of skipping; only a missing GPU skips. The only-TiH-refused test requires the 11 networks, and is marked slow because CI lacks the files. The small network's adjoint check now uses a weight per species, per cell: adjoint -2.748704976440e-05 against tangent -2.748704976440e-05, 8e-14 apart relative.
- Fixes. A non-finite rate coefficient is refused: a NaN prefactor passed the negative check and became a zero rate. close() drops the tables and scratch: ADSolver on the smallest network held about 265 MB after close, now 1 KB (its counts).
- API. `paths.DATA` and the others are now functions, `paths.data()` and so on. Breaking: `../reference/check_nets2.py`, `experiments/2026-11-05_gcm10d/gcm10d.py` and `experiments/2026-11-04_gcm-chem/gcm_chem.py` were updated; `../reference/make_libref.py` imports its own `old/lib/paths.py` and is unaffected; the other scripts under `experiments/` still use the old names. `ADSolver.status` is now `fallback`; `check_inputs` and `checked_drift` are private. The C++ caller takes its count slots from the export; the export drops `meta.json`, which nothing read, and its format is unchanged for `meta.txt`.
- Kernels. Removing a `constexpr double` changes the cubin, 128 bytes on the one looked at, since it is emitted as data, so `NASA_SPLIT_T` stays a constexpr set from the header. Forced-inline helpers split out of vjp_cells and jvp_cells gave equal cubins, so these splits cost nothing, unlike the call kernel's split (+1.1 %, above); vjp_cells falls from 290 to 185 lines.
### 2026-11-06 08:24: second read: the differences' half of the NaN hole, scratch sizes through the header, kernel names (experiments/2026-11-06_slop_baseline)
- No result changed on valid input, at 55eee2d against the starting commit 08d905f: reader and table hashes 28 of 28, cubins 28 of 28, float64 outputs 6 of 6 cases bitwise, `bench/kintest.py quick` 47 arrays bitwise except timings. Checks: truth 0 misses (worst 0.004 dex on paths, 0.014 on boxes), `test_ad.py` 72 passed, parity and guard 39, C++ 7, known answers and order 9.
- Scope of the equal-cubin proof, superseding "cost nothing" in the entry above: it holds at production sizes on this card (shared memory 101376 B). The forced-inline helpers of 4bb13dd do change the cubin at other block sizes: CHO_photo's adjoint at 49152 B (2 warps per block) and synthetic_small at 2 warps per block differ, same size, deterministic. Whether the arithmetic differs there is not known, so repeat the bitwise proof on the first new card; the NAS GH200s may be one.
- test_ad: a NaN in a finite difference made a cell's scale S NaN and took it out of errors and coverage. It now counts as non-finite, and any flag-2 cell at the default max_steps fails an FD case. No failure exposed: 114 FD verdicts, non-finite 0 and flag 2 0. Injected (`inject_nan_fd.py`): 50 cells of 1,024 with NaN differences fail the verdict, and one flag-2 cell does too.
- Found, not changed: one flag-2 cell makes the whole batch's rate-table cotangent NaN (`flag2_rc.py`, CHO_photo, 64 cells): with a NaN lm in cell 0, or with cell 0 out of tries, 1,863 of 2,187 entries are NaN, every column of all 207 thermal rows, against 0 with that cell left out. The other cells' x and lm cotangents stay finite. `nan_cotangents` writes NaN to gA and `arrh` skips only g == 0. docs/api.md and docs/method.md say only the cell's derivatives are NaN. Proposed: skip a NaN g in `arrh`, or set gA to 0 for flag 2, so the sum covers the good cells; this changes the output only for batches with a flag-2 cell. Needs a decision.
- Open, for a decision: the `call` kernel stays whole because its split cost +1.1 %, and jvp keeps helpers that cost +1.15 % (then 0.92 % in four rounds, above). The same kernels vary by 1 % between runs, so one rule should settle both, after more rounds on an idle GPU. Not timed here.
- API. `CudaSolver` and `ADSolver` share `_solver.py`. `K_COLL`, `CAP`, `DMAX`, `SPECIAL` and `formula` left `__all__`. The export's `meta.txt` lost `nr`, `t_min` and `drift`, which nothing read. `pytest -m "slow and not gpu"` is the documented run of the TiH inventory test, the wheel and the derivative kernels' compile.
### 2026-11-06 16:40: launch and check(), two refusals back, the bench's names, the checks not yet run on this code (experiments/2026-11-06_slop_baseline/round3)
- For a caller: `launch` now adds its counts to `total`, which `check()` reads (deliberate; `test_launch_adds_to_the_total`). On the pushed code (adc88de) `check()` saw only `__call__`. So a caller mixing `launch` and `check()` can now get a raise where it got counts, or a larger cold count. No x, h, done, steps, flag or derivative moves. docs/api.md states it.
- Fixes. The reader refuses a name given twice in `species`: it took the last occurrence, where adc88de took the first, so a reaction's rate could move with no message. The table builder refuses a hand-built reaction with a `rev` entry again: the kernels take every reverse from NASA-9, so the entry was ignored. Neither fires on a shipped network.
- No result changed, adc88de against this code, on NCHO_photo, SNCHO_photo_network_2025 and CHO_photo, dark and lit: 2,139 arrays and 50,476,237 numbers of the call, recovery, `jvp` and `vjp` bitwise equal on hard batches (`dump.py`: 39,041 cell-calls recomputed, 10,230 still failed, 4,428 bad input, 2,177 cold), only `nonfinite_h` added; 504 of 504 reader and table items equal; `bench/kintest.py quick` 45 arrays bitwise equal, all but its timings, and its report equal line for line apart from timings. NaN `lm` was left out: its h is a known difference. Compiled kernels against adc88de differ, 30 of 30, since the kernel text changed on purpose (the count slot, the lm screen), so they prove nothing there; against the commit before this work they are equal, 30 of 30.
- Checks. Truth: 904 paths, 1,016,411 states, 0 misses, worst 0.004 dex; 32,768 boxes, 1,540,096 states, 0 misses, worst 0.014 dex. C++ 7 passed, bitwise on 6 cases, the refused format too. `test_ad.py` 72 passed, 277 PASS, 0 FAIL, non-finite 0 on all 258 verdict lines that print it. Parity and guard 39 passed, worst 7.67e-8 dex. Known answers and order 9 passed, slope 2.961. CPU 64 passed, coverage 83.8 %; `-m "slow and not gpu"` 4 passed.
- `bench/kintest.py full` was not run to its end: it estimates 7.70 h (8.02 ms per tick, 3,456,000 ticks; 8.42 h on adc88de, other jobs running) and stops at its 5 h cap. Its box pass did run: all 131,072 boxes, 6,160,384 states, 0 misses, worst 0.066 dex. The first 1,000 ticks print the same on both versions, except timings.
- Found: on all 131,072 boxes the guard flags 2 cell-calls for drift and recomputes them, both recovered, on adc88de and on this code alike. The 2026-11-03 record above says guard 0, and the README said none; the README is corrected. Which commit between moved it, and which boxes they are, not looked at.
- Moved: the 12 gases, their floors, the solar start and its H2S ratio are in `maxichem/_checks.py`, since bench/ and tests/ can only share the package. `quickstart.py` keeps its own line, so a reader sees the gas without a private import. `_solver.py` stays in coverage: 68 % on the CPU, from the input checks and the failed-constructor tests, so it is not GPU-only.

### 2026-11-06 17:22: the guard's 2 recomputes on the NCHO boxes (experiments/2026-11-06_guard_recompute)
- "Guard 0" above was true when written and went stale at 3eb0586, which added the element-share correction. All 131,072 boxes, by commit: af0d9ed, 1f2cf97, cd3a1fd, f4014d9, 36a0f10, 548f608: all counts 0; 3eb0586, adc88de, b6b0583: drift 2, recomputed 2, recovered 2. The cleanup did not move it. The 3eb0586 entry's "guard counts stay bitwise" was on its saved cases, which do not include this pass.
- The two cell-calls, both the last call of a box (47, from 4.375e7 to 1e8 s, dt 5.62e7 s), both one step at h = dt, and only the drift count fires: box 33162 (2740 K, 85.9 bar, lit, unreacted start): C share off by 1.12e-5 relative, 11.2 times the allowance, N by 1.2 times. Box 98666 (2973 K, 43.4 bar, lit, hot-H start): O share off by 5.11e-5 relative, 51.1 times the allowance. The allowance here is `DRIFT`, 1e-6 relative; the atol term is far smaller. Over all 6,160,384 cell-calls the next worst is 0.31 of the allowance (box 121342, C, call 47); 186 are above 0.03, all from call 42 on.
- The recovery fixes both, in one step: worst ratio then 2.7e-10 and 1.8e-10. Against VULCAN, without the recovery 0.00002 and 0.00007 dex, with it below 0.00001 dex in the 12 gases; the largest change of any species 2.1e-4 dex (C6H6). So no accuracy depended on it.
- It is rounding, not an anomaly. Each flagged cell repeats alone (B = 1) and in a batch of 2,048 copies, so it is deterministic. On 548f608, where x still shows the drift, the same ratios are 0.0001 and 0.086. Relative noise of 1e-15 on the input flags box 33162 in 581 of 2,048 draws and box 98666 in 885 of 2,048; moving dt by up to 0.1 % flags 1,057 and 889 of 2,048. 3eb0586 moved x by up to 6.2e-13 relative (its entry). 84 noisy repeats of call 47 on all boxes (noise 1e-15 to 1e-12): 0 to 4 flags per pass, mean 1.4. The 36 repeats that list boxes (replay.log) flag 24 boxes, all at 2,517-2,996 K and 16.6-99.8 bar, 10 of them dark. So 0 on 548f608 and 2 now are two draws from the same lottery. An earlier run also shows it: boxes_ncho_final2.log in experiments/2026-11-03_overnight/logs, before atol 1e-25 went in, has drift 4, recovered 4.
- Inferred, not measured: at h = 5.6e7 s the factorized matrix tends to -J, singular along the element vectors, and in hot dense gas h times the fastest rate nears 1 / rounding, so the unpivoted solve loses the conservation. In line with that, the same state at dt 4e7 s or below flags nothing, and the worst ratio per call rises with dt from about 5e-9 at 1e-6 s to 0.23 at 2.5e7 s. A GCM's 70 s calls are far from this (worst ratio 1e-4 to 4e-4 at dt 20 to 80 s on these boxes).
- Open, for Isaac: the recovery can lose the same lottery. In the 48 repeats that count it (noisy_passes.log) 1 of 61 flagged cells ended flag 2 (box 106586, 2517 K, 58.2 bar, noise 1e-13: recovery's drift 12.7 times the allowance, done 0, state finite, error against VULCAN 0.00005 dex); in the draws above, 29 of 6,144 on box 33162 and 0 of 6,144 on box 98666. `kintest full` exits 1 on any still-failed cell, so a change that only moves rounding has about a 2 % chance per run of turning that pass red with no accuracy lost (estimate from those 48 passes). Nothing changed: `DRIFT`, the kernels and the bench's rule stay.

### 2026-11-06 19:25: kintest full within the 90-minute limit (experiments/2026-11-06_kintest_parts)
Isaac's rule: no test, tier or check run takes over 90 minutes of wall time.
- Result, at f217ef4, GPU 1, one unit: 3,245 s (54 min; loop 3,206 s, 691,200 ticks, 4.64 ms per tick, 415,299,269 cell-calls), exit 0. Misses first: 131,072 boxes, 6,160,384 states: 0 misses, 0 large, worst 0.066 dex, drift 2, recomputed 2, recovered 2 (entry above). 1,096 truth paths, 1,326,198 states: 0 misses, 0 large, worst 0.004 dex (QH family, 1500-s holds, path 4240060011). All blocks: unfinished 0, non-finite 0, negative 0; loop's guard all 0.
- Why 7.7 h: it never was. The estimate took the first 1,000 ticks' rate (8 ms, 1,012 cells with the QC family and the holds, which end by tick 10,965) x 3,456,000 ticks. Ticks run 4.5-4.7 ms after that, so an estimate of about 4.4 h, not run to the end (stopped at tick 100,000: 467 s, 0 misses, worst 0.004 dex). Neither the run record (once, at the end) nor the read-back (every 100,000 ticks in full) costs anything measurable; adc88de read the counters from the host every tick and was slower (8.76 ms against 8.02 ms, other jobs running).
- Where the time goes: a chain is sequential, one launch per tick, and a tick costs about its slowest cell. Kernels with a sync after each call, 256 cells: 3.4 ms (slowest cell 23 steps, mean 1.6), 64 cells 1.9 ms, 16 cells 1.3 ms, 1 cell 0.36 ms. So faster read-back gains nothing, a larger batch cannot be had in time, and fewer paths gain little: 256 of P's cells at dt 25 s alone took 4.31 ms per tick against 4.62 ms for all three P blocks (397 per tick). Only fewer launches help.
- The change: every box and every VULCAN path at full length, unchanged. P at dt 25 s for 200 days, dt 50 s for 400 days (both every tick, 69 cycles of the 10,000-row forcing), dt 500 s for 1000 days (every 2nd tick, 17 cycles). Before: 1000 days each, 346, 173 and 17 cycles. Stride is only a place on the schedule, so sharing launches gave dt 50 s twice the days at no cost. `CAP_H` 1.5; the estimate at tick 12,000 from the last 1,000 ticks' rate (printed 0.97 h); a stop when the run reaches 1.5 h, since a stuck cell after the estimate could overrun; `--force` gone. A split-by-ticks `--part` and `--resume` (88c365e), bitwise equal to an unbroken run on a subset, was removed unused (67a9b49): one check split in parts is still one check.
- `kintest quick` against b6b0583: 45 arrays, 999,482 numbers bitwise equal, all but timings and `run`; printed report equal but timings (at f217ef4). Both stops tried: exit 2 with a small cap.
- Given up against 1000 days at all dt: P calls 1.37e9 to 3.98e8 (29 %), no truth in either. States older than 200 days at dt 25 s and 400 days at dt 50 s. They are not settled: from the snapshots, over the last cycle 4 of 256 paths still move over 0.01 dex at each (largest 0.016 and 0.014 dex), and at dt 500 s, 28 paths over 0.01 and 1 over 0.1 dex (0.18) at 1000 days. A dt-specific failure on such an old state, or an event rarer than about 1 in 1.8e8 calls at those dt, would go unseen. Wrap jumps of the forcing fall from 346 and 173 to 69 each.
- The limit on the rest, measured with `--durations`: `pytest -m gpu -s` 128 passed in 31 min 57 s; `-m "slow and not gpu"` 4 passed in 100 s, longest 77 s (`test_derivative_kernels_compile[vjp]`); CPU 64 passed in 17 s. kintest quick 85 s. CI sets no `timeout-minutes` and runs the CPU tier.

### 2026-11-06 20:37: a failed cell no longer spoils its batch's rate-table cotangent (experiments/2026-11-06_rc_cotangent)
- The bug: `nan_cotangents` set `gA` to NaN for a flag-2 cell, and `arrh`'s skip of `g == 0` does not catch NaN, so one failed cell made the batch's `rc` cotangent NaN in all nine columns of every thermal row (1,863 of 2,187 on CHO_photo lit). docs said only that cell's derivatives are NaN.
- The fix: `nan_cotangents` writes `gA = 0`, which `arrh` already skips. Skipping a NaN in `arrh` instead was not taken: it would also hide a NaN from a cell that passed the guard, with no count. So a NaN in `rc` still means a real fault, and `still_failed` counts exactly the cells left out. Documented in docs/api.md and docs/method.md. `gA` is internal; the export holds no derivative kernel, so it does not move.
- No failed cell, against origin/main on NCHO_photo, SNCHO_photo_network_2025 and CHO_photo, dark and lit: the 12 jvp and vjp cases, 69 in all, 786 arrays and 24,321,374 numbers, bitwise equal. Over all 2,466 arrays, 12 differ, each `g_rc` of a vjp with failed cells (`steps25`, `cap10`), each from NaN (1,863 to 4,788 entries) to finite. Their jvp and everything else stay bitwise.
- A failed cell against the same batch without it, 60 cases on those networks, dark and lit, by a NaN lm and out of tries, with 1 or 3 failed cells at different places: the other cells' results and cotangents bitwise equal; its own NaN; `still_failed` right; `rc` finite. `rc` is bitwise equal when the failed cells come last; elsewhere the others change places, so the sum runs in another order, largest relative difference 9.7e-13.
- `test_failed_cell_is_left_out_of_the_rate_table` fails with `gA` back at NaN ("rc False", all 3 networks), and with a failed cell's x cotangent 0 ("its own cotangents NaN False").
- `pytest -m gpu -s`, which holds test_ad, C++, parity, guard and truth: 131 passed in 36 min 20 s, 0 FAIL lines; CPU 64 passed.

### 2026-11-07 21:49: snapy main 21a3e8a, #334 rebased, the coupling branch merged, the column heating tested (experiments/2026-11-07_pr334_rebase, experiments/2026-11-07_live_box)
- snapy main moved 2d9fa41 to 21a3e8a (#335, stretched x1, Cartesian stage 1). #336 (draft, on 21a3e8a) changes the default scalar
  transport: scalar mass and density reconstructed separately then divided (S177), donor positivity after the upper-bound blend, an
  implicit-transfer donor cap with a column sweep (S178) and spherical x1 volume rows, installed on every spherical WENO5 / cp5 deck.
  Once it merges, no fused-equals-default statement below holds. To match it the fused path needs: S177 in its reconstruction, the
  limiter order, the S178 sweep in its implicit transfer, and the x1 rows (or the default path) on the sphere.
- #334 rebased onto 21a3e8a (force-push of that branch only, a one-time exception): range-diff `=` on the reviewed commit. #335's
  stretched-x1 launch had the same cap, measured: 58, 90 fp64 and 116 fp32 variables on 106-cell lines (90 on 70-cell lines) fail with
  invalid argument; 57 passes. One more commit splits it the same way; the count per launch moved next to `stencil_kernel`. The many-
  variable test now checks right faces too, and a stretched twin fails before that commit and passes after. C64x100, 10 steps: 0, 16
  and 57 tracers bitwise main; main fails at 69 and 88; the head's 69 and 88 keep the 57-tracer flow and first 57 tracers bitwise.
  C++ ctest 109 of 110 (test_straka: no `ncrcat` here; torchrun tests collide on port 29500 under -j 24 and pass serially). Head
  1aa427b; CI waits for a maintainer's approval (action_required).
- isaac/maxichem-coupling: main merged (no force-push), pushed at 5be2af0. On a stretched x1 grid x1 takes the default path with
  per-face rows. The fused WENO5 guard checks cp as well as cm (found in review): its test fails with the cp check removed. Fused
  parity, 9 decks: WENO5 cases within 5.8e-16 of the local largest value, PLM bitwise; the new x1rat 1.1 box case fails at 1.7e-10 if
  the fused x1 kernel runs there. Fusion off, C64x100: 0, 16, 57 tracers bitwise main 21a3e8a; 69 and 88 bitwise the #334 head.
- `coupling/radiation.py`: heating -d(F_up - F_down)/dz and the per-column budget, tested (a height-constant flux on the sphere gives
  none; the area form gives some; the tests fail on the area form and on a sign flip). The three experiment drivers use it and log
  the worst column budget error. gcm_box.py also took `paths.vulcan_thermo()`, which the cleanup renamed.
- Box on the merged branch, from the 1-day spin-up, 300 steps of 70 s, radiation every 600 s: misses not scored (no VULCAN truth for
  these runs). Dark NCHO, 69 tracers: no unfinished cell, all guard counts 0, element budgets within 9.3e-16, column budget within
  4.1e-16 on 34 radiation calls; 1.61 solver steps per call (max 39), 0.52 us per cell-call; 1.10 s flow and tracers per step
  default, 0.34 s fused. 4,730 negative tracer entries clamped, the same with fusion on. Lit SNCHO 2025, 91 tracers, J from the
  WASP-39b reference: the same zeros, 1.99 steps per call, 1.33 us per cell-call, 0 negative entries, 19 GB peak.
  The same 300 dark steps fused with radiation every 3600 s (as 2026-11-04): 0 negative entries, guard counts 0. So the 4,730
  come with the 600 s radiation step, not the merge or the fused path (cause not tested; S178 in #336 caps the transfer).
- Single-tracer convergence, default path: the numbers of 2026-11-04 to the last digit (vertical orders 2.09, 1.24, 0.58; horizontal
  4.04, 4.32, 0.60), so the q-bar / rho-bar cap found in review is still there on main; #336's S177 is the planned fix.
- maxichem: max-args 12 enforced (jvp and vjp keep theirs, with a reason); numpydoc parameters for jvp, vjp and export.

### 2026-11-07 23:38: the lid comparison with the corrected heating (experiments/2026-11-07_lid)
- Setup: C48 hot Jupiter, flow only, 60 km layers, 70 s steps, radiation every 300 s, heating through `coupling.radiation`
  (column check at most 4.4e-16 on every call), snapy main 21a3e8a + #334; 100, 120, 140 and 160 layers, each from radiative
  equilibrium (eq, `re_start.py --mu 2.016e-3`; it reproduces re140_start.npz exactly) and from 1500 K (rest), 2 model days.
  Comparison: `lid_compare.py`, on the shared bottom layers by the lower lid's pressure bands (lid_compare.txt, .json).
- 100, 120, 140 layers: all six reach day 2 (140 crashed within 3 h with the area heating). Coldest cell 683-867 K. Lid tops at
  day 2 (eq / rest): 12.7 / 8.5, 1.8 / 1.2, 0.24 / 0.15 mbar. 160 layers: eq reaches day 2 (coldest 502 K, top layer, 0.003 mbar,
  night); rest stops at cycle 1,598 (day 1.3) at the density floor (top 0.002 mbar, T_max 2,310 K, CFL step 56 s).
- 0.1-1 bar rms differences, eq / rest: 100 vs 140 day 1 21.7 / 15.9 K, day 2 29.6 / 24.0 K and 147 / 124 m/s; 120 vs 140 day 1
  3.4 / 2.8 K, day 2 11.1 / 9.0 K and 70 / 57 m/s; 140 vs 160 day 1 1.0 / 0.9 K, day 2 (eq) 5.6 K and 34 m/s. Above 1 bar, 140 vs 160:
  0.2 K at day 1, 1.3 K at day 2. Each 20 layers higher about halves the day-2 change at 0.1-1 bar; every gap grows from day 1 to 2.
- Proposed: 140 layers (lid near 0.2 mbar) as the baseline; the snapy group's tolerance at 0.1-1 bar decides. Not separated: the lid's part
  of each day-2 gap against the flows' own divergence (the 6e-5 K chaos floor of 2026-11-05 was measured with the old heating).
- The sponge scan on the 140-layer lid (no sponge; tau 3e3, 1e4, 3e4 s over 600 and 900 km) runs on the snapy group's RTX 4000s from the
  bundle in this folder (bundle/sponge_scan.tar.gz); his probe measured 1.09 s per step and 4.2 GB.

### 2026-11-08 02:57: the 140-layer twin; #336's donor cap on the box (experiments/2026-11-07_lid, experiments/2026-11-08_pr336_donor)
- Decision in force from the snapy group: 140 layers is the working baseline, not final. Rule: if the twin floor at 0.1-1 bar is
  at least 2.8 K, 140 passes on T (u likewise against its own floor); under about 1 K, move to 160 layers with a sponge.
- Twin: eq140 again with the starting pressure times (1 + 6.7e-10 N(0, 1)), about 1e-6 K, and an unperturbed rerun (`twin.sh`,
  `twin_compare.py`, twin_compare.json). The rerun equals eq140 bit for bit at days 1 and 2. eq140 against the twin, rms at
  0.1-1 bar: day 1 1.7e-7 K, 3.6e-7 m/s; day 2 6.6e-6 K, 5.0e-5 m/s (max 1.1e-3 K); above 0.02 bar 1.6e-4 K at day 2.
- So the floor is under 1 K by about five orders: the day-2 gaps (140 vs 160 5.6 K / 34 m/s; sponge tau 1e4 vs 3e4 6.8 K
  / 45 m/s) are not round-off growth. By the rule this points to 160 layers with a sponge. Untested: whether a finite
  difference of about 1 K seeded near the top by day 1 grows to the day-2 gaps as weather; a twin seeded with about 1 K would
  separate that.
- Donor cap: the 2026-11-07 dark NCHO box (69 tracers, fusion off, 300 steps of 70 s, radiation every 600 s) on snapy #336
  d5b3adb merged locally with #334 1aa427b (not pushed): negative tracer entries 4,730 -> 0; guard counts 0 in both; solver
  steps 1.60 vs 1.61 per call (max 39 in both); element change per cell at most 2.6e-15 and the radiation column check 4.1e-16
  in both. No timing (the GPU was shared with the twin).

### 2026-11-09 16:01 to 16:32: the 1 K seed twins (at rest and on the day-1 flow); the box and tracer order on snapy main bd40fd25; the deck's gas against the tracers' gas (experiments/2026-11-07_lid, 2026-11-09_main_check, 2026-11-09_passive_check)
- Seed twins (`seed.sh`, `seed_compare.py`, seed_compare.json; `lid_run.py` gained `--perturb-top` and `--perturb-uniform`):
  eq140 again with the starting pressure times (1 + 7.6e-4 s) in the top 10 layers only, about 1 K at 1,316 K; s = N(0, 1)
  (seedn140) or s = 1, a coherent warming (seedu140). Same build as the lid runs (coupling branch 5be2af0, fusion off), 2 days.
  Against eq140, rms at 0.1-1 bar: day 1 3.3e-4 K and 7.5e-4 m/s (noise), 3.5e-4 K and 8.3e-4 m/s (coherent); day 2 0.23 K and
  2.0 m/s (noise; largest cell 15.7 K), 0.06 K and 0.39 m/s (coherent; 4.2 K). Day 2 at 0.02-0.1 bar: 1.2 K / 9.8 m/s and
  0.32 K / 2.1 m/s; above 0.02 bar: 3.1 K / 20 m/s and 1.6 K / 8.1 m/s.
- The seed set at rest has fallen to 3e-4 K at 0.1-1 bar by day 1, before there is a flow to grow it, so the snapy group asked
  for the seed on the day-1 flow. Day-1 seed twins (`seed_day1.sh`; `lid_run.py --perturb-day 1`): right after the day-1 dump
  (86,450 s) the running flow's pressure times (1 + 7.6e-4 s) in the top 10 layers, at fixed density and velocity (the energy
  plus cv/R times the pressure change); as applied 0.98 K rms there (largest cell 4.7 K noise, 1.26 K coherent). Both runs'
  day-1 dumps equal eq140's bit for bit. Against eq140 at day 2, rms at 0.1-1 bar: 0.17 K and 1.45 m/s (noise; largest cell
  9.2 K, 142 m/s), 0.22 K and 1.79 m/s (coherent; 11.7 K, 170 m/s); at 0.02-0.1 bar 1.06 K / 7.8 m/s and 1.28 K / 9.0 m/s;
  above 0.02 bar 6.1 K / 28 m/s and 6.7 K / 31 m/s; deeper than 1 bar 0.012 K / 0.02 m/s.
- So a 1 K seed near the top, at rest or on the day-1 flow, stays about ten times under the snapy group's 2 K line at day 2.
  By their rule (under 2 K: the gaps come from the setup; 3 K or more: weather) the day-2 lid and sponge gaps (5.6 K / 34 m/s,
  6.8 K / 45 m/s) come from the setup, which points to 160 layers with the weakest sponge (600 km, tau 1e4 s). Their call is
  pending, and the deck's gas (below) changes what a layer count means. Not tested: a larger seed, or a run past day 2.
- #334: CI ran at 6b2ee96 (run 42665017751, attempt 3: pre-commit, ubuntu 161 of 161, macOS 4 of 4), against main 3c891cb and
  with CUDA off, so it does not compile the patch. Four reviews approve at 6b2ee96. The branch is one behind main (#340); the
  snapy group's author asked for no push and keeps the update and the merge.
- snapy main is bd40fd25 (#336, #337, #338, #340 merged since 21a3e8a). Built: main merged with #334 (local head 7c0f54f of
  isaac/recon-many-variables in snapy-pr48, not pushed; `git diff bd40fd25 7c0f54f` equals `git diff 3c891cb 6b2ee96` byte for
  byte). C++ ctests there: 110 pass, 1 skipped (two GPUs), test_straka fails (no `ncrcat` here); the multi-process tests pass
  one at a time.
- Box on that build, the 2026-11-07 dark NCHO case (69 tracers, default scalar path, 300 steps of 70 s, radiation every 600 s):
  0 negative entries, no unfinished cell, all guard counts 0, 1.60 solver steps per call (median; max 39), element change per
  cell at most 3.2e-15, transport and chemistry budgets within 4.1e-16 and 9.0e-16, column check 3.8e-16. A second run repeats
  the first row for row. Misses not scored (no VULCAN truth for these runs). The sum of the tracers' mass fractions, logged in
  the second run (`gcm_box.py` now writes `qsum`): within -4.1e-5 and +5.4e-5 of 1 over the 300 steps (each species' WENO5
  weights differ, so the sum is not kept exactly; each species' mass is). No clean timing (other jobs on the CPUs): 1.39 s of
  flow and tracers per step against 1.10 s on the coupling branch's default path on 2026-11-07, to be timed properly.
- Single-tracer convergence on that build (`run_conv.sh`). The narrow blob (sigma 0.05 L, vertical, 25 to 400 cells): orders
  3.40, 3.79, 4.70, 4.95, against 2.87, 2.69, 2.22, 2.03 on main 2d9fa41 (2026-11-04); the largest error above 1e-3 of the peak
  at 50 cells 0.37 -> 0.25 dex, at 100 cells 0.060 -> 0.026 dex. So #336's separate reconstruction of tracer and density lifts
  the q-bar / rho-bar cap. Correction to the 2026-11-07 entry: its "2.09, 1.24, 0.58" was the wide blob (sigma 0.1 L), which has
  the wall's floor in its error (2026-11-04 02:00); on bd40fd25 that case gives 2.48, 0.62, 0.04 with L1 stuck near 1.4e-3. It is
  not an order test. Horizontal orders are unchanged (4.04, 4.32, then the x1 floor).
- The fused path of isaac/maxichem-coupling (5be2af0) no longer merges with main: conflicts in src/mesh/meshblock.cpp,
  src/scalar/scalar.cpp and tests/CMakeLists.txt, and its kernels compute the scalar path of 21a3e8a. Open upstream #326
  (fp32 foundations) changes four files of the tracer path again. Not rebuilt.
- The deck's gas is not the tracers' gas (found in review, measured in `mu_scan.py`, `mu_run.py`, `deck_mu_probe.py`). Every
  deck so far has one dry species H2 (2.016 g/mol, cv = 2.5 R); the tracers start as H2 0.84, He 0.16 plus metals,
  2.3625 g/mol. So the flow's R / mu and scale height are 17.2 % too large for the gas the chemistry carries, and its cp is
  11.9 % too large (14,435 against 12,903 J/kg/K at 1500 K with the NASA-9 heat capacities, `hot_cells.py`). The coupling
  itself is consistent (mass fractions per dry density, mass kept through the chemistry, T and P from the flow). A deck entry
  `composition: {H: 1.68, He: 0.16}` is accepted and gives 2.334 g/mol.
- What this does to the lid work (estimate, not run): with the mixture the same column holds 1.17 times more scale heights. At
  the same T(p), the lids of 100, 120, 140 and 160 layers (12.7, 1.8, 0.24 and 0.03 mbar on the H2 deck) would sit near 2.7,
  0.28, 0.026 and 0.0025 mbar. So layer counts do not carry over to a corrected deck (about 120 layers there reach the H2
  deck's 140-layer lid); the lid and sponge results hold as top pressures, to be confirmed by a run. Not decided: when to
  change the decks.
- How far the chemistry moves the molar mass, which a passive tracer cannot feed back: in the saved 10-day run (A, day 10,
  old heating) the tracers' molar mass is 2.3572 to 2.3648 g/mol, no cell more than 1 % from the median. At chemical steady
  state (dark, 1e10 s) it stays within 0.3 % up to 1500 K at every pressure and up to 2000 K at 0.1 bar and above; it falls
  by 7 % at 2000 K and 1e-4 bar, 12 % at 2300 K and 1e-3 bar, 28 % at 2300 K and 1e-4 bar (H2 dissociation).
- The energy matters earlier than the molar mass (`hot_cells.py`, hot_cells.json: the equilibrium cp, where the composition
  follows T, over the frozen cp, from the NASA-9 enthalpies at steady state): 1.01 at 1500 K and 0.01 bar, 1.20 at 1800 K and
  0.01 bar, 1.73 at 2000 K and 0.01 bar, 1.23 at 2000 K and 0.1 bar, 1.07 at 2000 K and 1 bar, 3.3 at 2000 K and 1 mbar
  (Bell & Cowan 2018, ApJL 857, L20; Tan & Komacek 2019, arXiv:1910.01622). Equilibrium is the upper bound; slow kinetics at
  low pressure give less. In the lid runs' day-2 states (flow only, H2 deck): eq140's hottest cell is 1,913 K at 0.38 mbar
  (eq160: 1,819 K at 0.33 mbar); the ratio is above 1.1 in 0.34 % of the cells above 1 mbar and 0.20 % at 1-20 mbar (largest
  3.3 and 1.9), in none below 20 mbar (largest 1.14 at 0.02-0.1 bar, 1.04 at 0.1-1 bar). So in this deck a passive
  treatment leaves out little below 20 mbar; a hotter planet, or the top millibar here, needs the heat of reaction in the flow.
- Review points on the box driver, not acted on: a cell that comes back unfinished is still written to the tracers (none
  occurred; the counts are logged); the photolysis table is a fixed 1-D profile times cos z; the 70 s first-order splitting
  has no convergence run with the corrected heating.

### 2026-11-09 16:50 to 17:15: a pass over the coupling's physics decisions (experiments/2026-11-09_science_pass)
Read: the drivers (gcm_box.py, gcm10d.py, lid_run.py, chem_glue.cu), snapy's scalar and implicit transfer at bd40fd25 + #334,
snapy's docs on tracers, and the papers below. One independent review of the same files, given no conclusions.
- Uniform ratio on snapy main (uniform_box.py; the box deck with implicit-scheme 9, the 1-day spun-up flow, 300 steps of
  70 s, no heating, winds up to 3 km/s): eight constant ratios stay to 8e-16 after 1 step and 1.8e-14 after 300, with 0
  positivity hits. A blob and its complement (sum 1 at the start) depart from 1 by 1.0e-4 after 1 step and at most 5e-4
  over the run. So the tracer flux is consistent with the mass flux through the implicit transfer (the donor cap's code
  ran; whether it bound is not logged); the closure drift of sum(q) comes from each row's own WENO weights.
- The single-tracer orders of the entry above were measured with implicit-scheme 0 (conv_tracer.py). The implicit
  transfer moves tracer at the donor cell's ratio (first order); its share of the vertical transport at 70 s is not
  measured.
- Drivers, read in the code: the sphere runs are dark (J = 0 in gcm10d.py); only the lit box runs use the J table times
  cos z. The longwave's edge temperatures are 2 T_0 - T_1, a full-cell extrapolation where the edge is half a cell away
  (1.5 T_0 - 0.5 T_1 is the linear value). The start mixture has O/H 8.32e-4, C/H 3.03e-4, N/H 7.19e-5, C/O 0.364, He/H
  0.095, against VULCAN-JAX's default 6.06e-4, 2.78e-4, 8.19e-5, 0.458, 0.097: 1.37 times its oxygen. The drivers hold
  one grey shortwave band and no code that makes J from a radiation field, so the planned J source is not built. The 10-day runs took the radiation hourly; the lid and seed runs every 300 s.
  The top sponge damps all three velocity components and leaves the total energy, so the kinetic energy it removes
  becomes heat.
- Published practice (papers opened; most through a text summary of the first part of each paper, so a number is as
  found there and an absence means not found). Passive mixing ratios with one global R and cp: the UM kinetics runs (Drummond et
  al. 2020: R 3516.6, cp 1.28e4 J/kg/K for HD 209458b, mass mixing ratios, chemistry every 3,750 s against a 30 s
  dynamics step, radiation every 150 s, opacities follow the chemistry, no photochemistry, no Kzz, equilibrium start,
  1,000 days), THOR (Mendonca et al. 2018: R 3714, cp 13,000, grey radiation with no feedback), Exo-FMS with mini-chem
  (Lee et al. 2023: chemistry every 1,500 s, no feedback, no eddy diffusion), MITgcm (Cooper & Showman 2006). Only Tan &
  Komacek 2019 let composition change R and cp, for atomic H, with onset near T_eq 2,200 K. The deck's R 4,124 and cp
  14,435 J/kg/K sit 17 % and 13 % above the UM's values; the tracers' own gas gives R 3,519 and cp 12,903 at 1,500 K.
- Photolysis and zenith angle: the direct beam's part of J is J_top exp(-tau / mu) with no cosine factor (Tsai et al.
  2021, eq. 8; Bais et al. 2003, JGR 108, 8543), so a J(P) table times cos z is too small aloft, by cos z itself in thin
  gas, and puts the photolysis layer at the wrong depth. No hot-Jupiter GCM found computes photolysis inside the model;
  the UM does for terrestrial planets, per column in its radiation (Ridgway et al. 2023).
- The independent review agreed on the passive representation, the dry-density convention, the flux and implicit
  transfer, the conversions and the time level, and on the deck's gas as the largest error. Its further points, checked
  in the code where stated above: the edge temperatures; the hourly radiation (its estimate of the thin-gas radiative
  time: 3,000 s at 1,500 K, 750 s at 2,500 K); the sponge's heat; the donor-cell implicit transfer; unfinished cells
  written back; the collision limit cuts the C2H + M pair by 47 times at 1,000 K and 100 bar (its number, not rerun).

### 2026-11-09 18:00 to 18:15: #334 brought up to main; the drivers' gas, radiation step, edge temperatures, photolysis stand-in, unfinished cells and start mixture changed (experiments/2026-11-09_science_pass)
- snapy #334: its branch now holds main bd40fd25 merged into the reviewed head 6b2ee96 (head 7c0f54f9, a plain push). The
  patch against main equals the reviewed one byte for byte (the same sha256 of the two diffs; 4 files, +282/-54). The
  result of the CI run on the new head is in the next entry that names it.
- The snapy group's calls after the deck-gas point was posted: the decks switch to the H2 and He gas before any new lid or
  sponge run; the lid target stays a pressure, about 0.03 mbar (about 140 layers of 60 km on the new gas, an estimate),
  against one level higher, to day 2, judged on rms T <= 5 K and rms u <= 20 m/s at 0.1-1 bar; the weakest sponge (tau 1e4
  s), its base quoted as a pressure; every lid, sponge and seed number on the H2 deck is provisional until rerun. How a
  dry H2 and He gas is best declared in snapy is being checked by snapy's author; no new run before that answer.
- The author's decisions on the drivers (gcm_box.py, gcm_chem.py, gcm10d.py, lid_run.py), applied:
  1. Gas: `composition: {H: 1.68, He: 0.16}`, `cv_R: 2.65` (2.334 g/mol, R 3,562 J/kg/K; the metals are left out, 1.2 %
     above the full mixture's R). re_start.py's cp follows (3.65 R). Not changed: conv_tracer.py and neg_repro.py (held
     flow), and the C++ box with its Python twin. The saved flows and start columns are of the H2 gas: re-spin first.
  2. Radiation every 300 s by default (was 3,600 s).
  3. The longwave's edge temperatures are 1.5 T_0 - 0.5 T_1 (were 2 T_0 - T_1). lid_run.py keeps --lid-fix and --lw-sub.
  4. gcm_box.py --jref: the table's J unscaled where cos z > 0, zero on the night side (was the table times cos z). Still
     a stand-in: no slant attenuation; the table holds no zenith angle and mixes layers of several VULCAN runs.
  5. An unfinished chemistry cell stops the run, in the start call and in every step, with the count and the first
     cell's T and P (was: the partial result written back and counted).
  6. The start's elements are VULCAN-JAX's default ratios per H nucleus (O/H 6.0618e-4, C/H 2.7761e-4, N/H 8.1853e-5,
     He/H 0.09692, S/H 1.3183e-5): H2 0.8366, He 0.1623, H2O 5.50e-4, CO 4.65e-4, N2 6.85e-5 (C/O 0.458, 2.361 g/mol),
     then the same 1e10 s call. Was O/H 8.32e-4, C/O 0.364.
- Smoke test of the edited box driver (smoke.sh; main bd40fd25 + #334; 20 steps from rest, then 5 coupled steps, dark;
  not a science run): both parts end normally; the start call takes 140 steps per cell on average, none unfinished; later
  calls 1.16-1.22; 0 negative entries, element change at most 2.3e-15, sum(q) - 1 within 2.4e-7, column check 2e-16. The
  100-layer box now reaches 2.4 mbar in its top cell (12.7 mbar on the H2 deck), as the estimate above said.
- Posted to the snapy group as points outside these drivers: the implicit transfer's donor-cell ratio (and that the
  single-tracer orders were measured without the implicit solve); the sponge's drag on all three wind components with
  the energy kept; the heat of reaction that passive scalars leave out near a higher lid; no code path yet for J.
- Open, not run: a 70 s against 35 s step comparison with the corrected heating and the new gas; the reruns.

### 2026-11-09 18:20 to 18:40: #334's CI; the implicit share of the vertical transport; atomic H at the lid; cv_R 2.62 (experiments/2026-11-09_science_pass)
- snapy #334 at head 7c0f54f9: CI passed (pre-commit; ubuntu 162 of 162 tests; macOS 4 of 4). CI builds with CUDA off.
- Implicit share (implicit_share.py; the box on main bd40fd25 + #334, implicit-scheme 9, the 1-day spun-up flow of the H2
  deck, 100 steps of 70 s, no heating; vertical acoustic Courant number 2.9-3.9, |w| up to 110 m/s). Per x1 face, the mass
  moved by the implicit correction that tracers ride on (hydro.icorr.mass_corr[IV2]) against the explicit flux's (dt F1
  A1), sums of magnitudes over all interior faces: 1.6-2.1 % in stage 0, 0.4-0.5 % in stage 1, 1.1-1.4 % in stage 2
  (top 20 layers 0.2-1.9 %), steady over the run. Divided by the RK weights (1, 1/4, 2/3) every stage gives about 2 %. So
  on this flow the donor-cell path carries about 2 % of the vertical mass transfer; the uniform-ratio test above ran on
  the same flow, so it went through that path. Not measured: the sphere with day-night forcing; the path's order.
- Atomic H at the lid (x_h_lid.py; the lid runs' day-2 states on the H2 deck; x_H from hot_cells.json's steady-state
  grid, an upper bound; cells below 1,400 K take the 1,400 K value). 160 layers, top layer (0.001-0.1 mbar): day side T
  1,211-1,508 K, x_H median 0.10 %, largest 0.18 %; night side largest 2.6 % (1,676 K). Top 20 layers: largest 4.6 % (160
  layers) and 4.1 % (140 layers, the 1,913 K cell); above 1 % in 0.30 % and 0.13 % of night-side cells, in no day-side
  cell; no cell above 5 %. A constant cp and molar mass hold for this deck at day 2; to be repeated on the new gas at
  the last day of each run.
- cv_R (the author's decision: the most reasonable value; cv_r_pick.py): 2.62. The new start mixture (2.3611 g/mol) has
  frozen cp 12,149 / 12,503 / 12,895 / 13,264 / 13,595 J/kg/K at 1,000 / 1,250 / 1,500 / 1,750 / 2,000 K (NASA-9); the
  deck's R is 3,562.5 J/kg/K, so cp is matched at 1,500 K by cv_R 2.6195 (cp/R by 2.662). With 2.62 the deck's cp is
  12,896 J/kg/K: 6.1 % too high at 1,000 K, 5.1 % too low at 2,000 K, and cp/R is 1.1 % low. The four drivers and
  re_start.py (cp = 3.62 R) use it; the 2.65 of the entry above is superseded.

### 2026-11-09 19:42 to 20:35: the drivers' shared code moved into coupling/, with tests; four driver bugs fixed (experiments/2026-11-09_coupling_package)
- What moved (one copy each, raising with the value and the fix; no snapy or pyharp import): `coupling/composition.py`
  (molar masses from pyharp's atomic weights, q to x and back keeping each cell's tracer mass, element moles, tracer
  totals, the gas constant 8.314462618, the start gas), `radiation.py` (+ face temperatures, the budget check,
  `RadiationClock`), `chemistry.py` (Pa to dyn cm^-2, `require_finished`), `flow_file.py` (flows and start columns that
  record their gas and grid), `photolysis.py` (the --jref stand-in, counted). coupling/ joined mypy (strict, as maxichem:
  0 findings; 0 cast, 0 type: ignore; `Any` only in the 7 annotations of the deck's geometry dict) and the coverage
  source (100 % of its 224 statements on the CPU tier).
- Bugs fixed in the drivers (gcm_box.py, gcm_chem.py, gcm10d.py, lid_run.py, re_start.py), each with a test that fails on
  the old form in a scratch copy (every new test was shown to fail on a broken copy: proof_of_failure/, 15 breaks):
  1. Radiation schedule: `next_rt = tsim + rt_dt` fired every 350 s at 70 s steps; now the first step at or after each
     multiple of rt_dt (0, 350, 630, 910, 1260, ... s; mean 300 s). The test fails on the old form (700 s for 630 s).
  2. The budget check kept `max(worst, residual)`, which drops a NaN; now a non-finite flux or heating raises with its
     column, and a residual above 2 nl (nl + 3) eps of the column's largest flux (the telescoping sum's round-off bound;
     measured 2.1e-16 to 2.2e-16) raises. The test fails on the old form (no raise).
  3. The unfinished-cell stop always advised max_steps or --dt; it now names the counted cause: cold (no solver setting
     helps), bad input (the driver's inputs), still failing after the recovery (a shorter --dt), with the call's counts.
     The test fails on the old message.
  4. Flows held only hydro_w and tsim, so a pure-H2 flow loaded into the H2 and He deck read about 16 % hotter. Flows now
     record the deck's species entry (composition, cv_R), the EOS molar mass and the geometry; a different record, other
     block shapes or no record is refused (re-spin). re_start.py's columns record the same, with the gas read from a
     deck YAML (`--deck`, replacing `--mu`); lid_run.py refuses another. The test fails when the record is not checked.
  Also: face temperatures (a linear T(z) is exact; the test fails on 2 T0 - T1; re_start.py's check used that old form
  and now uses the package's); the start gas (element ratios equal the inputs to 2e-16 on NCHO_photo, SNCHO_photo_2025
  and CHO_photo; a network with N but no N2 is refused, where the old `if "N2" in sp` dropped N silently; the test
  fails on that form and on an H count without H2S); --jref counts the day-side cells beyond the table (the test fails
  when not counted). The q, x round trip and element moles through a reaction are tested to 1e-13.
- Start ratios: O, C, N, S now the solar file's rows (6.0618e-4, 2.7759e-4, 8.1846e-5, 1.4099e-5); the drivers had C
  2.7761e-4 and N 8.1853e-5 from VULCAN-JAX's inactive default.yaml settings. He/H 0.09692 kept (author's decision;
  the deck's He/H is 0.0952). Start gas on NCHO_photo: H2 0.8366, He 0.1623, H2O 5.50e-4, CO 4.65e-4, N2 6.85e-5,
  2.36105 g/mol (start_check.py).
- Differences between the copies: kept, with a comment at each site: the spin's step (fixed --dt in the box, the CFL step
  on the sphere) and the start noise (1e-3 m/s seed 11 against 1e-6 m/s seed 5), reasons not recorded; the box and
  gcm_chem.py stop below the CFL step while gcm10d.py substeps (its long-run role); --jref is box-only. Removed: qsum now
  logged by the sphere drivers too.
- Smoke tests (smoke.sh; snapy-pr48 build, GPU1; not science runs), all as expected:
  - box: 20 spin steps, then 5 coupled steps, dark: start call 140 steps per cell on average, none unfinished; the first
    coupled call 14.5 steps (its h restarts at 1e-10 s), later calls 1.14-1.22; 0 negative entries; sum(q) within 2.4e-7 of 1; column check 2.1e-16; peak 16.59 GB allocated, as the
    pre-package driver's smoke test (2026-11-09_science_pass/smoke_box.json, 16.59 GB).
  - The old flow runs/box/spin1d.pt: refused at load with the re-spin message (exit 1).
  - gcm10d.py at C48x100, 69 tracers: 3 spin steps, then 3 coupled steps: none unfinished, 14.3 steps in the first coupled
    call, then 1.21-1.32, column check 2.2e-16, peak 23.27 GB allocated (27.21 GB reserved). The main checkout's
    unchanged gcm10d.py on the same flow: the same step counts and column check, peak 22.90 GB allocated (28.77 GB
    reserved), so the package's checks add about 0.37 GB (1.6 %) at the peak.
  - gcm_chem.py at C24x100: 2 spin steps, then 2 coupled steps: none unfinished.
  - gcm_box.py --jref at nx 24 on SNCHO_photo_network_2025 with jref_SNCHO25.npz (10 spin steps, 2 coupled steps): none
    unfinished; 12,672 of the 28,800 day-side cells lie deeper than the table's deepest layer (0.945 bar) and take its
    J, none above its top (1.05e-5 bar), printed once and logged per cycle.
  - lid_run.py: 3 steps from rest; re_start.py made a 100-layer column for that deck; lid_run.py took it for 2 steps; the
    old runs/re100_start.npz was refused (exit 1).
- Open: every saved flow and start column made before this entry is refused (23 run scripts under experiments/ pass
  --load): re-spin under the deck first. A budget error names the column within its panel, not the panel. The drivers'
  lint (ruff finds hundreds of style findings in each, as before; pyflakes clean). re_start.py repeats the deck's g, F
  and opacities. The C++ box twin is unchanged.
- An independent review's fixes (each test shown to fail on the old behaviour: proof_of_failure/review/, 14 breaks):
  - Clock: with `tsim += dt` and a fixed slack, 70.1 s steps missed the firing due at step 72,000 (summed time
    5,047,199.999999401 s) and 0.1 s steps the one at 39,900 s (verified on the committed code). The clock now keeps its
    own time as the exact sum of its steps (a fraction), allowing half an ulp per step and per multiple for decimal
    steps and periods; the drivers call `due()` and `advance(dt)`. Test: 200,000-500,000 steps of 70, 70.1, 0.1 and
    35.7 s (periods 300, 300.1 s) fire exactly at ceil(k P / dt), worked out in decimal fractions.
  - Photolysis: a NaN table entry gave NaN on the night side and an integer mask of 2 doubled J (verified); now refused
    (NaN or negative rates, empty table, non-boolean mask), and the night side is set by `torch.where`.
  - Conversions: a molar-mass vector of shape [S, 1] broadcast over the cells (verified); `mass` of shape [2, 1] was
    already refused (the review's example of it was wrong). Added: mw [S] finite and positive, per-cell arrays exactly
    [cells] and not negative, negative x refused in `mole_to_mass` (the call clamps accepted states at 0 and the guard
    scales by positive factors, so a returned x is never negative), non-finite atoms, one precision for q, mw and atoms.
  - Start columns: 2 levels with nz 4 loaded (verified); levels now checked as [nz + 1], finite and positive, on save
    and on load, with nz required in the grid.
  - Budget: dz [inf, 1] gave a NaN residual that passed (verified); thicknesses are checked, a non-finite residual
    raises, and inputs other than float64 are refused (the bound is for float64 without underflow).
  - Tests: hand-worked absolute element totals with unequal cell masses; the start's ratios pinned to the solar
    file's rows typed in the test (within five-digit rounding, 4e-5, which the drivers' old C 2.7761e-4 fails);
    composition, cv_R and molar mass each alone refused.
  - The dumps' signed conversion is `coupling.composition.signed_mole_fractions` (output only). The ghost refresh and
    wall mirror stay in each driver, with a comment: they drive snapy's Mesh, which coupling/ does not import.
  - CPU tier 87 passed; coupling/ 100 % of 276 statements; mypy and pre-commit clean.
  - Box smoke test again (smoke3.sh, started once GPU1 had no other process; 20 spin steps, 5 coupled steps, dark):
    both parts end normally; the same step counts, sum(q) range, column check (2.07e-16), radiation calls and peak
    memory (16.59 GB) as the first box smoke test above.

### 2026-11-09 19:16 to 22:06: four fixes from a code review: the equilibrium test's pressure, refused inputs, the upper temperature bound, kintest on both networks
Branch sweep-fixes (one commit per fix). Scratch data of the measurements are not kept in the repository.
- Equilibrium known answer. For H + H + M = H2 + M, K_p = (p_H2/P0)/(p_H/P0)^2 with p_i = x_i P, so x_H2/x_H^2 = K_p P/P0;
  the test had K_p P0/P and passed only because it ran at P = P0. Now at 10 bar with the right factor. A kernel whose reverse
  rates use the cell's P in place of P0 fails it in 6 of 6 cases (3.06e9 against 3.06e10); the old test passed on that kernel.
- Refused inputs. An input summing to 0.9 raised every density by 11 % unseen. |sum(x) - 1| > X_SUM_TOL = 1e-10 is now bad
  input. Measured |sum(x) - 1|: at most 1.3e-15 over every stored input the checks feed (boxes_ncho starts 6.7e-16; paths, mixes,
  panel P, yardstick, derivative boxes at most 8.9e-16; sulfur stored states 1.3e-15) and over the solver's outputs (60-call
  chains, both networks; export-check calls 8.9e-16 to 1.1e-15). Every refused cell (bad input of any kind, T out of range) now
  raises in the Python call, the derivative calls and the C++ call, naming the counts, the limits and the first cell; the
  kernels' counts and the other cells' results are still checked with the raise switched off. Valid-input outputs (x, h,
  done, steps, flags, counts, tangents, cotangents) bitwise equal to the base commit: 12,264,022 numbers, 0 differ.
- The read of the counts is the call's only host sync; before, a call read nothing back. Cost on an idle GPU, 1e6 settled
  cells, 20 alternating rounds of 3 calls with the read on and off: NCHO 359.04 against 358.90 ms per call, +0.050 % (round sd
  0.067 %); sulfur 690.24 against 690.16 ms, +0.012 % (sd 0.012 %). Below the 1 % rule; kept.
- The derivative test's x direction now keeps sum(x) = 1, since the call refuses a state off it. The part along x itself,
  which no difference can judge now, is checked apart: with the step sizes held the tangent along dx = x is 0 but for rounding
  (largest |dx_out| / max x_out 2.1e-14; limit 1e-12). No longer checked: the adaptive tangent along dx = x itself (the step
  control's response to scaling x through atol, which does not scale), which the central differences judged before. A
  difference along x moves x off sum(x) = 1, which the call refuses, and the derivative has no meaning for valid input, which
  must sum to 1 and so cannot move along x.
- Upper temperature bound. VULCAN's NASA-9 files hold only coefficients, no ranges. T_MAX = 3000 K (the owner's choice: the
  edge of what is checked against VULCAN; the fits themselves run to 6000 K). Every stored check input lies within 300-3000 K,
  both ends reached exactly (qual_ncho, panel P, yardstick: 300.0-3000.0 K). Cells above are counted as hot and refused;
  check() raises for cold and hot. EXPORT_FORMAT 5 (4 for the sum refusal alone). Outputs bitwise equal again (12,264,022
  numbers; the 44 new hot counts all 0).
- Checks on the upper-bound commit: CPU 66 passed; slow CPU 4 passed; parity and guard 39 passed in 219 s; known answers
  and order 9 passed; test_ad 81 passed in 1015 s (all 285 verdicts pass); kintest quick NCHO 0 misses (worst 0.004 dex on
  paths), 1 passed in 99 s.
- kintest takes --network; the truth sets per network are declared in bench/kintest_truth.py. The 2025 sulfur network runs
  boxes_sncho25 (1,930 scored), boxes_sncho25p (350 scored) and qual_sncho25 (QC 128, QCs 32, QC twins 32, QL 16; no QH, QHs,
  QLc), and scores 16 gases. A set declares the boxes whose VULCAN run ended before a stop (117 and 54; the manifests' errors,
  all "stop ... s not hit"), by count and ids' digest; any other error, a wrong count or other ids refuse the set (checked by
  hand on the three wrong declarations). The truth's condensates S8_l_s and H2O_l_s are 0 in every state and are dropped.
  Panel P on the sulfur network gets H2S = 1.41e-5 H2.
- Sulfur quick tier, reproducing the 2026-11-04 scores: boxes_sncho25 1 box with a miss, box 1939 at 4.341 dex in S2 (VULCAN's
  error); boxes_sncho25p 1, box 96 at 0.155 dex in S2 (maxichem's at atol 1e-25); next worst boxes 0.030 / 0.021 dex; paths 0
  misses over 245,328 stored states, worst 0.0068 dex; guard nothing recomputed; wall 112 s. Pinned point by point (see the
  review follow-up below).
- Not covered by kintest: the day-night path sets (daynight_ncho, daynight_sncho25). Their J changes every call, and the
  path calls here are dark; adding per-call J for 64 paths was left out. The one-off scoring of 2026-11-04 covered them.
- The floored error, the scored gases and the miss thresholds (0.1 and 1 dex) are one copy in maxichem/_checks.py for the
  bench and the tests. kintest quick on NCHO is bitwise the base commit's: 45 arrays, 1,007,852 numbers, all but timings.
- New stored reference: data/benches/kintest/quick_sncho25_ref1.npz, the sulfur quick tier on the final code. Final checks:
  test_vulcan_truth.py 2 passed in 207 s (both networks); test_export_cpp.py 12 passed in 384 s (the C++ call refuses with
  Python's message and counts: sum 0.9, NaN sum, T 3000.5 K; accepts sums 1 +- 0.9e-10 and T 300 and 3000 K, bitwise).
- Each new or changed test fails with its change reverted: without the raise, the 8 refusal cases of test_guard and
  test_parity, test_ad's negative x on 3 networks and the 3 refused C++ cases; without the kernels' sum and T_MAX tests, the
  bad-input and range cases (4) and the C++ sum and hot cases; with the tangent's normalization dropped, the scaling check on
  every case; with a pin moved by 0.02 dex or a pin removed, the sulfur quick tier exits 1.
- Open: the day-night sets in kintest; T_MAX is a constant, not each species' range (the files hold none).
- Review follow-up (two commits after this entry's first). (1) The call kernel and the guard judged sum(x) in different
  orders: a sum 1 + 1.000000082740371e-10 in the warp's order and 1 + 9.999978622943217e-11 in the network's (found on both
  networks' permutations) was refused by the call, counted by the guard as unfinished only, still failed in the recovery, and
  the call returned without raising. Now the kernel that steps a cell (call, jvp, vjp) screens it once and writes its status;
  the guard counts refused cells from it without screening again, the recovery steps only listed cells, and the host names the
  first refused cell of each kind from the status. (2) A refused cell at dt = 0 came back done and could not be named; a
  refused cell is now never done. New tests: the straddling sum is refused and its mirror accepted (Python, both networks; C++
  NCHO), T one float64 step outside each end at dt 0 and 500 s is refused and named (Python), T just above 3000 K at dt 0 is
  named in C++; all failed on the previous code (Python 4 of 4, C++ 2 of 2). The message's printed sum is the host's, added
  in another order (9.99998e-11 for that cell), and now says so. Valid outputs still bitwise the base
  commit's: 12,264,022 numbers, 0 differ. EXPORT_FORMAT 6. (3) Pins were per box: a new miss in another gas of box 1939
  would have passed under its larger S2 miss. KNOWN_MISSES now holds the gas and each missed stored state with its error
  (box 1939: S2 after calls 41-47, 1.647 to 4.341 dex, its other gases at most 0.018 dex; box 96: S2 after call 45, 0.155 dex);
  any other missed point in any gas of any box fails, as does a pinned point gone or moved by 0.01 dex. On the saved run
  with edits: a 1-dex CO miss in box 1939, its S2 misses moved to CO, and one more missed S2 state each fail; unchanged
  passes. Checks: CPU 66 passed; slow CPU 4 passed; parity and guard 43 passed in 219 s; known answers and order 9 passed;
  test_ad 81 passed in 1003 s; test_export_cpp 14 passed in 408 s; test_vulcan_truth 2 passed in
  205 s with both pins held.
- The guard's J parameter was left unused by the status (clang-tidy misc-unused-parameters, seen with the kernel header
  regenerated): removed; J is appended only to the call and recovery kernels (PHOTOLYSIS_KERNEL_NAMES). pre-commit on all
  files passes, clang-tidy included; CPU 66 passed; parity and guard 43 passed; test_export_cpp 14 passed; valid outputs
  bitwise the base's (12,264,022 numbers, 0 differ).

### 2026-11-10 00:22 to 02:41: the collision limit counts M as a collision partner; the old-rule truth gate fails (experiments/2026-11-10_collision_limit)
Branch collision-limit from main 0c7f7df. Pins: NCHO_photo 74d4f5c0, SNCHO_photo_network_2025 4e8d537a, the NASA9 folder's
sorted-file digest 13162af0 (120 files), libref8 9b27b5fd, libref9_SNCHO25 1dc74011, VULCAN-JAX 36419e5 (clean).
- The rule (VULCAN-JAX 1c6afe1). A direction counts when its molecules m plus M (b = 1) make two collision partners,
  q = k n_tot^(b + m - 2) with k without M; for a dissociation A + M, q = k. The kernels fold M into the stored
  coefficient, so such a side compares k_stored / n_tot; the derivative layer's density weight is
  wE = (wA - wG)(mf - 2) + wG (mr - 2), -1 for A + M. The reader now refuses a line whose M is not on both sides of a
  three-body law or on neither side of a two-body one (also a reaction changed by hand); every VULCAN-JAX network file and
  every invented one still loads (TiSNCHO is refused for TiH, as before).
- RED scan: the kernels' own final rates (a probe kernel around rate_pair and collision_limit), 901 T (300-3000 K, 3 K
  steps) x 65 log P (1e-5-100 bar), judged by the new eligibility. Old rule: NCHO_photo C6H5 (20,738.6x the ceiling at
  3000 K, 1e-5 bar), C6H6 576.1, C3H4 280.1, C4H5 47.3, C2H5 9.9; the 2025 sulfur network adds S4 (S2 + S2 + M, 1,441.0x),
  CH3CCH 131.6, CH3SH 19.6, S8 1.99. New rule: largest final q 1.0000000000000002 times the ceiling, no pair above. The
  numpy port of the same arithmetic agrees with the kernel to 2.3e-13.
- Against VULCAN-JAX's own rates (its environment, float64, CPU; remove_list empty, low-T caps off; 109 T x 29 P, 390 and
  532 pairs, M divided out of maxichem's coefficients): old kernel, the differences are exactly the 5 and 8 dissociation
  pairs above (up to 2.07e4 relative); every other pair within 3.48e-7. New kernel: every pair within 3.48e-7. The 3.48e-7
  is k_B: VULCAN-JAX's phy_const uses 1.38064852e-16, maxichem 1.380649e-16 (ratio - 1 = 3.4766e-7); pairs with no change
  of moles agree to 1.2e-13.
- Tests (each shown to fail): tests/test_collision_limit.py runs the derivative kernels' rate arithmetic through
  tests/rate_probe.cu on tests/data/synthetic_collision_network.txt at 3000 K: capped dissociations forward (H2 + M) and
  reverse (O2 + M) at K_COLL n_tot, the capped associations at 10 bar, a tie, the pair ratio, a first-order line and a
  photolysis branch at 1e9 s^-1 left alone, and the tangent and adjoint of every coefficient along T at fixed P (-1/T for a
  capped A + M) and along ln P (1). Largest errors 7e-16 (rates), 2.6e-15 (derivatives). It fails on the old code
  (H2 + M 7.5e7 against 2.4e6), with the old density weight (tangent 0 against -1/T; adjoint alone too) and with every
  one-molecule side counted (the first-order line capped). The associations at 10 bar and the photolysis branch pass on the
  old code; the tie checks the capped value, not the 1/2 split of the weights. The reader's 4 refusals and the hand-built
  refusal fail on the old code. CPU 92 passed; slow CPU 4 passed.
- Where the new cap binds in the truth sets (CPU port at each box's T, P): boxes_ncho 18,295 of 131,072 boxes (14.0 %),
  1879-3000 K; boxes_sncho25 912 of 1,930 (47.3 %), 1022-2999 K; boxes_sncho25p 103 of 350. Not in the pinned boxes 1939
  and 96. test_ad's random cases hold 128-139 such cells per case on NCHO_photo and CHO_photo, 482-483 on the 2025 network.
- Direct comparison at constant T and P, 47 checkpoints from 1e-6 to 1e8 s, every species above 1e-12: VULCAN-JAX's
  network-built chemistry right-hand side with its rates and imposed J, Radau at rtol 1e-10 (1e-8 agrees within 5e-5 dex)
  and the exact JAX Jacobian. Six hot C-rich NCHO cells (2582-2989 K, 1e-5 bar, dark and lit, the boxes' stored starts and
  two 1-s states with C6H5, C6H6, C4H5, C3H4, C2H5 raised 1000x): new maxichem at rtol 1e-7 agrees within 5e-6 dex, at the defaults within
  0.024 dex (C3H2 at 1e-6 s; scored gases 0.0036); old maxichem is off by 0.98-4.53 dex in transients (C3H4, C6H5, C2H5),
  scored gases 0.0048 dex at most, 0 at 1e8 s. Sulfur cells (1298-2981 K, 3e-5-5.2 bar; in two, S4, S8, CH3SH, C6H5, C6H6 raised 1000x): new
  maxichem at rtol 1e-7 within 5e-6 dex, at the defaults within 0.015 dex (S8 at 1e-6 s); old maxichem within 5e-5 dex of
  VULCAN-JAX except the hot dense cell (box 1130, 2981 K, 5.2 bar), 0.097 dex in C2H5. The lit 1299 K, 0.003 bar sulfur
  cell did not finish in VULCAN-JAX's Radau within 83 min and is not compared.
- Gate against the existing truth (old rule; kintest quick, saved runs of the old code compared): NCHO_photo 0 misses
  (worst box error 0.049 dex, was 0.014), paths unchanged (largest change 5e-5 dex), but 13 of 32,768 boxes' worst errors
  rose by more than 0.01 dex (2 by more than 0.03; largest +0.048, box 20725). The 2025 sulfur network: misses and pins
  unchanged, paths within 1e-4 dex, 2 boxes rose by more than 0.01 (box 117 of boxes_sncho25p +0.0169, box 1493 +0.0145).
  The gate fails.
- Diagnosis: the intended chemistry, not an implementation error. All 15 moved boxes are at 2772-2987 K and 2e-5-0.017
  bar, mostly C/O >= 1, the error in C2H2 (one in HCN) at stops of 1e-6 to 7e-4 s, where the new cap lowers a pair by
  97-14,100x (C2H2 + C4H3 + M -> C6H5 + M, S2 + S2 + M -> S4 + M). VULCAN-JAX integrated on their starts with the old rule
  agrees with the stored truth within 0.0004 dex in every box; with its current rule it differs from that truth by
  0.0105-0.0487 dex, box by box what new maxichem shows. New maxichem against VULCAN-JAX's current rule on these boxes:
  0.0008 dex at most; old maxichem against the truth: 0.001 dex.
- Not done, since the gate stopped the plan: new parity references (the torch generator's change of its eligibility masks,
  mf + tb >= 2, agrees with VULCAN-JAX within 3.48e-7 and is kept as a patch in the experiment folder, the generator left
  as it was), the loaders' rate-rule check (patch kept there), test_ad, the export test, speed, the final tiers. Parity and
  test_vulcan_truth would fail on this branch (inferred, not run): the stored references and truth carry the old rule.
- The truth sets, the truth generator (vulcan-truth/science.py) and the parity references libref8 and libref9_SNCHO25
  still carry the old rule. The new rule cannot be validated against them; it needs truth made with it.

### 2026-11-10 00:28 to 06:10: independent known-answer tests, two kernel hazards, the coupling stop's causes
Branch known-answers, from 0c7f7df. A review found that the tests catch regressions but prove little on their own: parity
compares with an earlier copy of the same formulas, CI runs no GPU test, and of eight deliberate breaks of the physics two
were seen by an independent test. The new tests take every expected value from a formula written out in the test, with k_B
(1.380649e-16 erg/K), the standard pressure (1 bar) and K_COLL (1e-9 cm3/s) typed there, so a wrong constant in the solver
fails them. Networks and NASA-9 files other than tests/data are written by the test (helpers.write_network) and read by the
public reader.
- Rate laws (test_rate_laws.py). Each law drives an isomerization CO_i -> CO_j beside spectators it does not change, so x
  relaxes to the pair's equilibrium at lambda = (k_f + k_r) times the spectators' densities, exactly; one cell per law and
  (T, P), dt = 1 / lambda, at (600 K, 1e4), (1500 K, 1e6) and (2600 K, 1e8 dyn cm^-2). Laws: Arrhenius; low-pressure only;
  low-pressure only with effi, traced in pure H2 so that k = eff k0 under any reading of the efficiency; Lindemann and Troe
  falloff, Pr 8e-4 to 20; the special law's two k0 terms and Visscher-Moses width (OH traced in CH3); the collision limit
  on a direction of two molecules (the reverse the larger, K < 1) and of three. The rows without Troe run again in a network
  with no Troe line, so both compiled falloff forms are seen. A direction of one molecule and M stays below 0.1 K_COLL under
  the current rule and the coming one (q = k), so that rule is not pinned. Worst error on lambda 1.0e-9 (limit 1e-6) at rtol
  1e-9. The Troe constants are those of Venot et al. (2012, Appendix C), which the code cites; the paper was not opened here. Every shipped network carries the special CH3OH law, so every one compiles the Troe form: the plain falloff arm, and fix A's edit of it, are seen only by these tests, not by the byte-identity dumps.
- NASA-9 ranges (test_known_answers.py). CO_1 has different low and high sets with every polynomial term non-zero, the high
  b1 and b2 set so that H and S meet at 1000 K. CO_2 / CO_1 after a 1e10 s call against exp(G_1 - G_2) from the NASA-9
  polynomials written in helpers.py: at 900, 1100 and 2500 K within 4e-15 (limit 1e-12).
- Photolysis mapping: three absorbers, one with two branches, J 0.7, 2.3, 0.4 and 1.1 s^-1 looked up by (absorber, branch)
  in net.photo; every species against the closed form at 0.3, 1 and 3 s, worst 6.3e-8 (limit 1e-6) at rtol 1e-8.
- Step control: the decay A -> 2 B with a first try h = dt at J dt = 5 and rtol 1e-6, within 2e-5 of the closed form; the
  same call with max_steps equal to its 118 accepted steps is unfinished, so a try was rejected (it finishes at 120).
- Constant pressure: H + H + M -> H2 + M from pure H, 1 / N - ln N = 1 + 4 k0 n_tot^2 t, x_H = 2 N / (1 + N); calls ending at
  N = 0.8, 0.5, 0.3 within 6e-9 (limit 1e-6). Rodas3's order on it with every try accepted: slope 2.92 (band 2.7-3.3), next to
  2.96 on the linear decay.
- The four copies of setup and tail (test_kernel_copies.py): one batch through call, jvp and vjp with h NaN, +inf and 10 dt,
  dt = 0, cold, hot and bad-input cells beside good ones and a first try rejected twice running; x, h, done, steps, status,
  flag and counts bitwise equal across the kernels, and h NaN, +inf and 10 dt bitwise the h = dt cell. At max_steps 10 the 7
  good cells go through recover, jvp_rec and vjp_rec, all recovered. A dt = 0 cell returns done, 0 steps and h = H_MIN.
- The guard's allowance (test_guard.py): the guard kernel alone on a state and results whose He share moved by 0.5, 0.99,
  1.01 and 2 times the allowance: flagged [0, 0, 1, 1].
- Mutants: each a fresh copy of the tests' commit (6b26b49) with one replacement found exactly once; every new test also
  passed on the unchanged copy. The review's eight, and what kills each (the failing assertion):
  kb_x10: test_forward_rate_coefficients (lambda off by 0.99 relative), the recombination and its order;
  constp_dropped: the recombination (and the decay, the equilibria, photolysis, step control, the order with w, and the
  kernel copies, whose derivative kernels keep the term);
  nasa_split_1200: test_equilibrium_follows_each_nasa9_range[1100] (1.21401 against 1.20948);
  kcoll_x10: test_forward_rate_coefficients, the capped rows (lambda off by 9.0);
  photo_shift: test_each_photolysis_branch_takes_its_own_rate (0.1114 against 0.1800);
  step_error_ignored: test_a_rejected_first_try_is_retried (x_A 3.327e-3 against 3.380e-3; it still rejects on negative
  states) and the rate laws (5.7e-5);
  guard_drift_off: test_drift_beyond_the_allowance_is_flagged (none flagged). Before it, only
  test_element_drift_is_recomputed failed here, and on an empty comparison of the stored data, not on its count;
  radiation_sign: test_heating_sign_and_column_budget (CPU).
  None survives. Further breaks, each killed: the high NASA-9 set ignored (1100 and 2500 K), Troe d 0.15, the second k0 term
  dropped, the efficiency dropped, the limit from the forward direction only, the plain falloff's Pr halved (the plain form
  only), the Jacobian's constant-pressure term w zeroed in the call kernel (order 1.11, and the recombination), prej left
  false in the call, recover, jvp and vjp tails, and the vjp's non-finite h taken as H_MIN.
- Fix A (common.cuh law(), ad_common.cuh rate_law_partials). A falloff row with k0 = k_inf = 0 gave Pr = 0 / 0, and the Troe
  factor at Pr = 0 (k_inf overflowing) or Pr = inf (k_inf = 0) took log10 of it: each made k NaN, and one such row made every
  cell of a call non-finite (12 of 12 cells not done in the new test). Now k0 = 0 gives no rate, and the factor is 1 where Pr
  is 0 or not finite: k = k0 at Pr = 0, k = 0 (the high-pressure rate) at Pr = inf. The partials take the same limits, so the
  tangents are finite; with the forward kernel fixed alone they were NaN. Test: k0 = 0, k_inf underflowing and k0 = k_inf = 0
  (x unchanged, tangent 0), k_inf overflowing (decay at k0 n_tot; the T tangent against -t k0 n_tot (B0 + C0/T - 1)/T), in
  both compiled forms. A k0 that overflows (above 1e308) still gives NaN, which the guard flags. Troe's form itself tends to
  Fc^(d^2/(1+d^2)), about Fc^0.019, as Pr -> 0 or inf, not to 1; F = 1 holds only at the end points, as in the partials before.
- Fix B (recover.cuh dense_lu). After the last shuffle every lane read A[k][k] for the exchange test while lane k & 31
  writes A[k][k] at j = k inside the exchange, with nothing between: under independent thread scheduling a late lane could
  read the exchanged entry, skip its share of the exchange and leave a mixed pair of rows. The search's reads come before the
  shuffles by their data, and every other cross-lane access is behind a __syncwarp, so this was the only race. The decision is
  now taken into a register, then __syncwarp, then the exchange. No failure was seen before or after, and no test can force
  the race. Test (test_dense_lu.py, harness tests/dense_lu_check.cu): 8 row-shuffled dominant 42 x 42 matrices (34-39
  exchanges) and 8 random ones with every other diagonal 1e-3 (6-11 exchanges, 25-33 rows kept where plain partial pivoting
  would exchange); row order equal to the rule written in numpy, |PA - LU| / (|L||U|) at most 1.0e-15, the solve within
  5.9e-13 of numpy's. It passes on the code before the fix; PIVTAU 1 and one lane skipping its share of the exchange fail it.
- Byte identity: valid-input outputs (x, h, done, steps, flags, counts, tangents, cotangents) dumped at 0c7f7df and at each
  fix: the two default networks' export, parity-start, lit, chain, jvp and vjp cases, and for every other network the reader
  takes (CHO_photo, CHO_photo_lowT, NCHO_full_photo, NCHO_photo_lowT_Jupiter, NCHO_thermo, SNCHO_DMS_Tsai2024, SNCHO_photo_C3,
  SNCHO_photo) its export cases and a jvp and vjp on 256 solar starts: 26,401,783 numbers, 0 differ after fix A and after B.
- Coupling: require_finished names the hot cause and the sum tolerance among bad inputs, and says that refused cells reach it
  only where the call's raise was switched off. Its test runs every count of COUNTS: a named cause gives only its own fix, the
  others are listed as leading to one. On the previous code its cold and hot cases fail; with the hot cause removed, hot fails.
- Checks on c3276ab: CPU 96 passed in 25 s; slow CPU 4 passed in 100 s; parity and guard 44 passed in 222 s; known answers,
  order, rate laws, kernel copies and dense LU 23 passed in 301 s; test_vulcan_truth 2 passed in 205 s (NCHO 0 misses, worst
  0.014 dex on boxes; sulfur pins held); test_ad 81 passed in 995 s (285 verdicts pass); test_export_cpp 14 passed in 391 s (with the project's CUDA toolkit on PATH); pre-commit on all files
  passes with the kernel header written.
- Not covered independently: the efficiency inside a falloff's Pr (the code multiplies the finished k, so at high pressure it
  gives eff k_inf; the convention is not pinned); the collision limit's rule for one molecule and M; the recovery's dense
  path inside a real call against an outside answer (the LU alone, and the copies' agreement); J changing within a call;
  anything above 3000 K.

### 2026-11-10 00:30 to 02:19: the call interface: keyword-only settings, named results, a read-only C++ solver
Branch call-interface, on 0c7f7df. An interface change; no number moves and no reference was regenerated.
- Python (breaking). `rtol`, `atol`, `max_steps` and `J` are keyword only in the call and `launch`; in `jvp` everything after
  `dh`, in `vjp` everything after `gx` (so `gh` too). `solver(x, T, P, dt, h, 1e-25, 1e-3)` had swapped the tolerances and passed
  every check; it now raises TypeError. The results are named tuples in the old order (`CallResult` x, h, done, steps;
  `LaunchResult` adds flag; `JvpResult` adds dx, dh; `VjpResult` adds the cotangents dict), so unpacking and indexing are
  unchanged. Every caller in the repository and the three GCM drivers already passed the settings by keyword.
- The flag. Both solvers keep the last per-cell flag in `solver.flag`; `launch` also returns it. Not taken: a flag field in the
  `jvp`/`vjp` results, which would change their arity (five 6-name unpackings in test_ad). The names `fallback` and `launch`'s
  counts are unchanged (2026-11-06 entries). The bench takes its guard counts as differences of `check()`, not `total[1:-1]`.
- C++ (breaking). `ns`, `nj`, `species`, `elements`, `default_atol`, `count` and `total` are private behind accessors; `count()`,
  `total()` and `atoms()` return copies. Assigning a one-slot `count` had pointed the kernels' atomics past its end. The
  constructor refuses an export whose kernels name an argument the call does not supply, naming kernel and argument; before,
  a renamed recovery argument stopped the call with a bare `map::at` after the call kernel ran. A defined J must be float64
  [B, nj] on the device; an undefined J only when nj = 0, so a thermal export takes [B, 0] (as the C++ box driver passes) and
  refuses [B, 1]. The C++ box drivers (2026-11-04_cpp, 2026-11-05_e2e) use the old members and will need `()`.
- Shown to fail without the change: the positional-settings test on the code without `*` (4 of 4) and on the base (import);
  the misuses (`chem.count = ...`, `chem.ns = 5`, `chem.nj = 0`, `chem.total = ...`, `chem.species.clear()`) compile on the old
  header and not on the new; on the old caller a renamed `didx` gives `map::at` and a [B, 0] J "this solver has no photolysis".
  In-place work on the returned copies leaves the members as they were (12 count slots, atoms sum 289).
- Checks: CPU 92 passed; parity and guard 45 passed in 216 s (worst 2.8e-8 dex); known answers and order 9 passed in 112 s;
  test_export_cpp 15 passed in 401 s, bitwise on every case, counts included; test_vulcan_truth 2 passed in 203 s (0 misses,
  worst 0.004 dex on paths, 0.014 on boxes); test_ad 81 passed in 992 s, 0 FAIL.
- Review follow-up (02:20 to 03:55, two commits). (1) `solver.flag` was set before the refusal raised, so a caught
  ValueError left flag 0 (passed) on the refused cells, and a call refused on the host kept the previous call's flag: it is
  None from the start of every call until the call returns (both solvers). (2) The constructors' `drift`, `photo` and `cap`
  are keyword only (`CudaSolver(net, device, True)` set drift to 1 with photolysis off); no caller here or in the GCM drivers
  passed them by position. (3) The C++ constructor checked names only against a whitelist: a missing `didx` failed in the
  driver after the call kernel ran, a repeated `atol` or a `Jin` on the guard ran with "bitwise identical" results, and Ta/Pa
  swapped fed the wrong slots. Each kernel's list must now equal `KERNEL_ARGUMENTS` (J last for call and recover with
  photolysis), and every meta.txt check runs before any CUDA work; all five edits are refused at construction, naming kernel
  and argument. (4) The interface test checks each optional parameter's kind; the C++ test changes what the accessors return
  in place and checks the solver is unchanged (a build whose accessors share the tensors aborts). Shown failing on the code
  before: the flag tests (guard 2 of 2, test_ad 3 of 3), the parameter-kind test (the constructors; one optional made
  positional again in a scratch copy). Checks: CPU 98 passed; parity and guard 47 passed in 215 s; test_export_cpp 19
  passed in 501 s, bitwise on every case; test_vulcan_truth 2 passed in 203 s (0 misses, worst 0.004 dex on paths, 0.014 on
  boxes); test_ad 81 passed in 993 s, 0 FAIL; pre-commit on all files passes.
- Not done: the C++ call still takes rtol and atol by position; `meta.at` lookups give no key name; Python's `count` and `total`
  are still mutable attributes; Python refuses a [B, 0] J without photolysis.

### 2026-11-10 06:30 to 09:11: the known-answer tests' review, four fixes; main merged
An independent review of branch known-answers confirmed the tests' derivations and the eight kills and found four holes;
each was checked first and is real. Each new test or expectation failed on a copy of the code before (1a7020f) or under
the break named, and passes now.
- Troe's end point. As Pr -> 0 or inf, (z + c) / (N - d (z + c)) -> -1 / d, so F -> Fc^(d^2/(1+d^2)): 0.98459, 0.97828,
  0.97772 for the fixture at 600, 1500, 2600 K, not the F = 1 of 9910b1f. The call and the partials now take that limit
  for the standard form (VM11 tends to 1), d ln F / dT = s Fc' / Fc; the limit test expects F k0 n_tot and the matching T
  tangent (fails on 1a7020f). A first form of the partials moved 17 dumped derivative arrays by rounding (at most 1.5e-15
  relative in dx); the finite-Pr block is the earlier text again, the limit a separate branch.
- The equilibrium oracle took G / RT from the reader and P0 from kinvulcan; it now reads the NASA-9 files and types 1 bar.
  With P0 x 10 the old tests pass and the six equilibrium cases fail.
- Rejection for error alone. The old case's first trial is negative (N_A = -712/7203, exact). New: J dt = 1/10, the exact
  Rodas3 trial positive; rtol 6.2e-6 puts the error RMS at 10, rtol 1.5e-4 the RMS at 0.42 and only the largest error,
  the first try's norm, at 2.4. The first-try max norm (common.cuh try_error) is a real path: removing it passes the old
  tests and fails the second case.
- The kernel copies must finish: every stepped cell done, every recomputed cell recovered, recovered states within 1e-6 of
  the ordinary call. With REC_STEPS 1 the old test passes and the new fails.
- A tie in the dense LU's column 0 across two lanes must go to the lower row; reversed, the old test passes and the new
  fails. docs/testing.md lists what no independent test covers.
- Merged local main (b1d2640, keyword-only call interface) with a merge commit; one test passed rtol by position and now
  passes it by keyword. Valid-input outputs at the merged head bitwise those of 0c7f7df: 26,401,783 numbers, 0 differ.
- Checks on the merged head (2a3daee): CPU 107 passed in 24 s; slow CPU 4 passed in 100 s; parity and guard 48 passed in
  225 s; known answers, order, rate laws, kernel copies, dense LU 25 passed in 336 s; test_vulcan_truth 2 passed in 204 s;
  test_export_cpp 19 passed in 532 s; test_ad 81 passed in 1006 s (285 verdicts pass, 0 fail); pre-commit on all files passes with the header written.

### 2026-11-10 09:15 to 13:10: the settings in one file, and no bare numbers in the kernels
Branch settings-file, from 5a032ea. A cleanup: no result moves. Scratch data of the measurements are not kept in the
repository.
- Inventory at 5a032ea: every numeric literal outside a module-level UPPER_CASE assignment in maxichem/, coupling/,
  bench/ and tools/ (syntax trees), and in the kernels and cpp/ with comments removed: 2,265. Classed by rules on file,
  value and line; every literal of a class other than arithmetic was read, and a sample of the arithmetic. No bare call
  setting, method setting or physical constant: all were named, the method settings as constexprs at the top of
  common.cuh, KB, P0 and NASA_SPLIT_T in kinvulcan.py. 56 bench scenario values (gpu_speed's batch, written out in its
  docstring; kintest's schedule and report bins); 81 coefficients of published formulas (the NASA-9 polynomial,
  Rodas3's four stages, K1's weight 2 and 1 / GAMMA = 2, the 3 of the step factor's cube root in its derivative,
  coupling's extrapolation and N2 per N); 352 layout numbers (201 of them the kernels' warp width: lane & 31, >> 5,
  shuffle offsets; 14 warp widths and 11 bytes per double in host code; table widths, flag fields, stride terms); 4 report
  bins of a check; 1,727 arithmetic, indices and shapes; 45 inside constexpr or #define lines. At the branch head 2,198.
- Moved: H_MIN through GMAX (16 method settings) from common.cuh to _constants.py; the kernels keep their constexprs, set
  from header macros NAME_VALUE = repr(value), each of which parses back to the same double. KB, P0, NASA_SPLIT_T from
  kinvulcan.py, which still exports KB and P0. _constants.py has a section each for call settings, method settings,
  physical constants and layout. Named: WARP_LANES, DOUBLE_BYTES, KIND_* (plain ints, so the reader's dicts and every
  digest are unchanged), FALLOFF_KINDS, THREE_BODY_KINDS, ARRHENIUS_COLUMNS, RATE_ROW_COLUMNS, the uint8 and int32
  limits; in C++ kWarpLanes. The step factor's derivative takes SECANT_MAX_ORDER for its 3, and a static_assert holds it
  to 3, the cube root step_factor takes. docs/method.md names STEP_SAFETY, ERR_FLOOR, H_MIN, SECANT_TARGET,
  SECANT_MAX_ORDER and T_MAX where it repeated their numbers.
- Proofs. Python-only step (e32cc92): the generated header text equal in 106 of 106 compiled variants; reader, table,
  derivative-table and size hashes equal on all 10 networks that load, dark and lit, at 101376 and 49152 B (146
  records); every constant's value equal; the package's syntax trees equal with each new name replaced by its value, but
  for 6 units read by hand (the moves, an f-string, two local names); the solvers' launch and scratch sizes and the
  exports of both default networks, dark and lit (96 files: cubins, tables, meta.txt), byte-equal (162 records); parity
  and guard 48 passed; slow CPU 4 passed. C++ step (da792eb): test_export_cpp 19 passed. Method settings through the
  header (c5f4b5d) and the cube root's name (5fa8d94): compiled kernels equal to 5a032ea's in 77 of 77 variants at the
  card's 101376 B (NCHO_photo, SNCHO_photo_network_2025 and CHO_photo, dark and lit; call at 1-4 cells per block, W in
  and out of shared memory; jvp, vjp) and in the test networks' forms at the sizes the solvers pick for them (synthetic
  call at 1, 4 and 5 cells, synthetic_small at 1 and 10, their jvp and vjp at 2, 5 and 32 warps, the plain falloff); the
  header gains exactly the 16 macros, every other line in place. Compiled again from the commits themselves (c5f4b5d,
  whose kernel file is the checked one with one comment realigned, and the head): 77 of 77 equal. Parity and guard 48
  and test_export_cpp 19 passed on the method step. No compiled kernel changed at the card's sizes, so no speed was measured.
- Found: at 49152 B NVRTC's output for an unchanged source varies from compile to compile. Of 40 variants at 49152 B
  (call at 1-4 cells with W in and out of shared memory, jvp, vjp; NCHO_photo and CHO_photo), 21 gave 2 or 3 cubins in
  3 compiles of 5a032ea, and in 6 PTX compiles each, 8 at a time, 22 gave 2 to 5 different PTX, so the spread starts
  before the assembler. One variant compiled one at a time gave 2 PTX in 4. The 16 call variants with W in global
  memory repeated exactly in every compile. The cause is not found; the load average was 14 to 33 throughout (these
  compiles and other jobs), so load is only a guess. The 2026-11-06 entries' equal cubins at 49152 B may have been
  lucky draws. These variants are compared by PTX instead: for each of the 40, 5a032ea, the method step and the
  cube-root step share at least one identical PTX among 6 compiles each (40 of 40 for each pair), which a change in the
  program would not allow. At 101376 B and in the test forms no compile here differed from another of the same program
  (the 77 variants in the compiles of 5a032ea and both kernel steps; one call 4 of 4 repeats).
- Not changed, with the reason. The kernels' 32 (warp width, 201 sites, with & 31 and >> 5): CUDA's warp, written as in
  CUDA's own idiom; a constexpr int would touch every kernel. Rodas3's 2 (K1's weight in Y3, Y4 and N') and 1 / GAMMA =
  2: naming one as a constexpr double changed the cubins of the call and derivative kernels by 16 to 144 bytes
  (measured on 6 variants), which then needs the bitwise and speed proof; docs/method.md states them and the test allows
  2.0. The guard's clamping allowance 2 atol per accepted step (guard.cuh) is a bare 2.0 of the same kind. Layout
  formulas (strides, work-array offsets, record fields, VOFF, the stream words' bit budget, warps_per_cell's 4 and 3),
  the header generator's bit-set macros and the NASA-9 file layout: the layout phase. gpu_speed's batch: written out in
  its docstring; kintest's schedule: the check's definition, left as written. coupling/: formula coefficients only.
- Kept from coming back: tests/test_constants_source.py (a new file, so the collision-limit branch's edits of
  test_tables_source.py and its docs/testing.md row merge as before) fails on a float in a kernel other than 0, 1/2, 1
  and 2 outside a constexpr or static_assert line, suffixed or not (shown on a bare 0.9 and 0.9f in the guard, and on
  5a032ea's 3.0); on a backticked UPPER_CASE name in docs/method.md, three letters or more, that neither _constants.py
  nor a kernel defines (shown with FAC_NONFINITE_X and CAPS); and on a C++ copy that differs from _constants.py (shown
  with kGuardBlockThreads 256).
- Merge with the pending collision-limit branch, tried in a scratch clone: one conflict more than main has, the import
  line of _tables.py (keep both); the merged tree passes the CPU tier (115). After the merge its _THREE_BODY_KINDS
  repeats THREE_BODY_KINDS.
- Checks on the final tree (c665d02's content; the two commits after it change a test's patterns and a doc page, and
  the CPU tier passes on them, 110), pre-commit on all files with the header written: CPU 110 passed;
  slow CPU 4 passed; parity and guard 48 passed in 225 s; known answers, order, rate laws, kernel copies and dense LU 25
  passed in 334 s; test_ad 81 passed in 1005 s (286 verdicts pass, 0 fail); test_export_cpp 19 passed in 534 s;
  test_vulcan_truth 2 passed in 214 s (NCHO 0 misses, worst 0.014 dex on boxes, 0.004 on paths; sulfur pins held; run
  in the worktree, since its run record reads the git commit). Valid-input outputs of the final tree, from a fresh
  process through the public calls (the two default networks' export, parity-start, lit, chain, jvp and vjp cases, and
  the export cases and a jvp and vjp on 256 solar starts for the 8 other networks that load), bitwise those of 5a032ea
  dumped today: 26,401,783 numbers, 0 differ; that 5a032ea dump is bitwise the one of 2a3daee made for the previous
  entry. No result moved.
- Corrections to this entry. The inventory's sub-counts come from coarse rules: the A, B, C, D and F lines were read,
  the 352 layout lines were classed by rule with a sample read, and some are mislabelled (_ad_tables' 2 npart as a kind
  code, the header generator's 32 two-bit fields per word as a warp width, the tuple-assigned _GT_* and _MAX_* constants
  as bare), so the layout sub-counts are approximate; the class totals stand. The 16 to 144 bytes of a named Rodas3
  weight come from one pilot that also renamed the cube root's 3, not a measured cost of the weight alone. The kernel
  commit's "compiled kernels unchanged" holds at 101376 B and in the test forms; at 49152 B it rests on shared PTX
  (above). Two test additions after the checks: each method setting's constexpr must read NAME_VALUE (fails on
  H_MIN = 1e-12 written back), and the names of docs/method.md are looked up in the kernels without their comments
  (fails on a GMAX left only in a comment).
- After an independent review (supersedes this entry's 49152 B argument, and c5f4b5d's message where it says the
  compiled kernels are unchanged without a size).
  - The 49152 B proof. A shared PTX among samples does not exclude other outputs with changed arithmetic, so it proved
    nothing; withdrawn. In its place, a compiler-independent check (tokens.py in the experiment's scratch): each
    variant's sources through `cpp -P` into C tokens, old against new, after folding each of the 16 moved initialisers
    to its binary64 bits (both sides), dropping the static_assert that holds SECANT_MAX_ORDER to 3 and putting the
    derivative's asserted 3.0 back. 5a032ea against c5f4b5d, c665d02 and 3a04ef7's head: token streams equal in 117 of
    117 variants each, the 40 at 49152 B included. It finds a one-ulp change of STEP_SAFETY and a wrong name in the
    derivative. The reviewer reached the same on 100 variants by its own route.
  - Found: NVRTC 12.8.93 is not deterministic at 49152 B. Of the 40 variants there (NCHO_photo and CHO_photo, dark and
    lit, call at 1-4 cells with W in and out of shared memory, jvp, vjp), 21 gave 2 or 3 different cubins over 3
    compiles of 5a032ea's unchanged sources, and 22 gave 2 to 5 different PTX over 6 compiles (8 at a time), so the
    spread starts in the front end; the 16 call variants with W in global memory repeated in every compile. At 101376 B
    and in the test forms no repeated compile differed. This contradicts the 2026-11-06 entries (cubins "deterministic",
    and equal at 49152 B). Cause not found; whether bits of a 48 KiB build can differ between two compiles is not
    known.
  - Moved too: MAX_GIBBS_JUMP (from kinvulcan.py, still importable there), NTAIL and the warps per cell (2, or 4 when at
    most 3 cells fit; from _tables.py) into _constants.py. Reader, table, size and header hashes equal (146 records),
    function trees equal with the names replaced by their values. Left beside their code, with the reason: MAX_SPECIES
    and the uint8 limit (follow from the tables' slot type), the adjoint's stream-word fields and limits (layout phase),
    MAX_WARPS_PER_BLOCK (the derivative kernels' queue), CUDA's limits and attribute numbers, the NVRTC options, the
    file formats (TROE_FIELDS, NASA_FILE_SIZES, SECTIONS), and coupling/'s constants (its own package for now).
  - The constants test now checks that the kernels' constexpr names equal a classified list (header settings, formula
    coefficients, kernel codes), over declarations wrapped onto several lines; shown on a new kernel-local
    NEW_SETTING = 0.37 (fails), a 1e-3 in the guard (fails), RODAS_C43 wrapped over two lines (passes).

### 2026-11-10 12:22 to 18:00: truth remade with the collision rule that counts M (experiments/2026-11-10_truth_v3)
data/truth_v2 is old-rule truth, kept only as a record: the generator then left out every direction of one molecule. The
checks now read data/truth_v3, and refuse a set or a parity reference that does not record RATE_RULE.
- Truth generator: vulcan-truth 0ea6d9c ("fix(science): count M as a collision partner"). VULCAN-0D's coefficients have
  M not folded in, so a direction counts when molecules + [M on that side] >= 2, at k M^(M + molecules - 2). RATE_RULE goes
  into every box set's manifest and part (merge refuses a part or set of another rule) and every path's spec and path
  set's manifest (history.py manifest). sobol.py adopt starts a set on a design copied byte for byte.
- Validation before generation. The generator's capped coefficients against VULCAN-JAX's on 109 T x 29 P: within 3.48e-7
  on all 390 NCHO and 532 sulfur pairs, the dissociations included (the k_B difference). check.py bitwise (fast against
  stock, 40 boxes): IDENTICAL on NCHO and on the sulfur network; the sulfur one finished at 14:29, 18 min after its sets
  had started. The settings were recovered by rerunning stored truth with the old rule, bit for bit: 26 sample boxes (10
  boxes_ncho, one of its step-cap reruns at 2e6, 10 boxes_sncho25, 5 boxes_sncho25p) and 6 paths (QC and QD of both
  networks, a twin QC and a QL of NCHO; no QH path). The NCHO sets
  were made with an earlier box.py (VULCAN's default atol, count_max 2e5; experiments/2026-11-03_atol/box_before_xatol.py),
  so NCHO ran a copy of the generator with that box.py (experiments/2026-11-10_truth_v3/gen_ncho); its build differs from
  the old stamp only in build.py's sulfur-network entry and in the staged NASA9 folder (VULCAN-JAX's HNO3 file, in neither
  network).
- Sets (new against truth_v2). boxes_ncho: 131,072 of 131,072 accepted, worst drift 9.87e-8 (the same), 97,500 CPU s
  (94,041), design 58ecd80d (the same rows); 6 boxes stopped at the cap 2e5, 4 finished at 2e6 and 2 at 2e7 (before: 3 at
  2e6, 3 at 2e7). qual_ncho 904 paths, 39,678 CPU s (39,471); daynight_ncho 64, 2,148 s (2,094); qual_sncho25 208, 42,202 s
  (42,677); daynight_sncho25 64, 12,941 s (8,745; its summed VULCAN steps are the same, 23,332,211 against 23,336,117, so
  the time is the machine's load); each with 0 missing, errors and forced steps. boxes_sncho25p: 350 of 404 accepted, the
  same 54 boxes without a result (step cap 2e6), 0 quarantined, worst drift 2.4e-8 (5.5e-10; now box 45, 2434 K, 97 bar,
  where the cap lowers S2 + S2 + M by 427x), 14,155 CPU s (14,695). boxes_sncho25: 1,931 of 2,048 accepted (1,930), 116
  without a result at the cap 2e6 (117: box 892, 425 K, lit, a reheat start whose donor the cap touches, now ends), the
  same quarantined box 810, worst drift 3.7e-8 (3.5e-8), 54,276 CPU s (56,189); its 8 workers ran 14:11-23:42, the
  slowest held 23 boxes that end at the cap (sobol.py records no seconds for those). The pinned boxes 1939 and 96 have
  bitwise the same truth in both versions (the cap touches neither nor their donors). A box set's CPU seconds count the boxes' own runs,
  not their donors' nor runs that ended at the step cap.
- Old against new truth on the scored gases at their floors. boxes_ncho: 57 boxes moved by more than 0.01 dex, none by
  more than 0.1; largest 0.067 dex (box 95573, C2H2, 2870 K, 5e-4 bar); all 57 at 2477-3000 K where the new cap binds
  (C2H2 54, HCN 3). boxes_sncho25p: 1 (box 117, 0.017 dex, C2H2, 2987 K). boxes_sncho25: 1 (box 1493, 0.015 dex, C2H2,
  2828 K), the next 0.0044 dex. Paths: qual_ncho 1e-4 dex at most, daynight_ncho
  0, qual_sncho25 4e-4, daynight_sncho25 1e-4. Boxes whose own T, P the cap does not touch still change by rounding when
  their start comes from a donor box at a T, P where it binds (boxes_ncho: 18,295 boxes where the cap binds, 6,281 through
  their donor, 106,496 bitwise equal, 0 unexplained; boxes_sncho25: 912, 178, 840, 0; boxes_sncho25p: 103, 43, 204, 0).
- Gate, kintest quick on NCHO against truth_v3 (db83a04): 0 misses; worst box 0.014 dex, paths 0.004. The 13 NCHO boxes
  that moved against the old truth are at 0.0003-0.0010 dex again (they were 0.0105-0.0487). New stored reference
  data/benches/kintest/quick_ncho_v3_ref1.npz.
- kintest full on NCHO against truth_v3 (db83a04): 3,483 s wall (boxes 10 s, loop 3,441 s, 4.98 ms per tick). All
  131,072 boxes, 6,160,384 states: 0 misses, worst 0.066 dex (box 87219, 487 K, CH4; the same as on the old truth). 1,096
  truth paths, 1,326,198 states: 0 misses, worst 0.004 dex; panel P: unfinished, non-finite and negative 0. Exit 1 for one
  box: 98666 (2973 K, 43 bar, lit, hot-H start), whose last call (4.4e7 to 1e8 s, one step) moves the element shares by
  5.5e-6, above the allowance 1e-6; the guard recomputes it and the recovery's step fails too (still failed, done 0). Its
  scored error is still 0.0003 dex. On main (the old rule) the same call is flagged and recovered; at rtol 1e-4 the old
  rule takes two steps, the new rule still one; at rtol 1e-5 both pass (share change 1.8e-16). A one-step call of 5.6e7 s
  at 2973 K and 43 bar is not a GCM call; left as it is, for the owner (experiments/2026-11-10_truth_v3/box_drift.log).
- Is it the rule or the lottery? Paired test (lottery.py, lottery2_98666.jsonl, lottery2_33162.jsonl, the per-cell flags
  in lottery2_<box>_<kernel>.npz, counts in lottery_paired.log). The state before that call was saved as reached under
  each rule (main cecbf8e and f2ab956), and each state was run on three kernels: old (cecbf8e), new (61fa409, its kernels
  f2ab956's) and pair-only (61fa409 with only H + C2H4 + M -> C2H5 + M under the new rule, lottery2_pair.diff). 2,048
  matched cells each: cell 0 as saved, the others x times 1 + s 10^U(-15, -12) per species and dt times
  1 + U(-1e-3, 1e-3), one seed for all. Box 98666, still failed after the recovery at the production drift allowance,
  old kernel to new: 0 to 4 of 2,048 on the old-rule state, 2 to 5 on the new-rule state. The excess points the same way
  on both states. Paired (failed on the new kernel only / old only / both): 4 / 0 / 0 (exact McNemar p 0.125) and
  5 / 2 / 0 (p 0.45); pooled 9 / 2 / 0, p 0.065. At 2, 5 and 10 times the allowance, pooled 7 / 1 (p 0.070), 4 / 0
  (p 0.125), 1 / 0. The pair-only kernel's flags equal the new kernel's, cell for cell. That is expected, not more
  evidence: at this T, P only that pair's capped rate changes (1.0987 times more). Control box 33162 (2740 K, 86 bar,
  where the cap does not bind): 7 still failed (180 flagged) on every kernel and state, the same cells. These are one
  sample of 2,048 a state; the 2026-11-06 17:22 entry's 29 of 6,144 (box 33162) and 0 of 6,144 (box 98666) used another
  setup and are not pooled with them. No new kind of failure: the control box fails 7 of 2,048 under the old rule on
  main, at 10 times the allowance at most 1 cell fails anywhere, and box 98666's scored error is 0.0003 dex. But the
  unperturbed call of box 98666 (cell 0) fails its recovery on every run under the new rule from the new state (flag 2 at
  1 and 2 times the allowance, passes at 5), so at this head kintest full on NCHO exits 1 on every run, not on about 2 %
  of runs. Verdict: inconclusive whether the rule raises the failure rate (pooled p 0.065); the failure of box 98666 is
  deterministic. No change made here.
- Gate on the sulfur network against truth_v3 (kintest quick, cce9052): boxes_sncho25 1,931 boxes, 7 missed states, all
  pinned (box 1939, S2, 1.647-4.341 dex, held); boxes_sncho25p 1 missed state, pinned (box 96, S2 0.155 dex, held); next
  worst boxes 0.030 and 0.021 dex; 208 paths, 0 misses, worst 0.007 dex. The pins keep their reasons: both boxes' truth is
  bitwise that of truth_v2. Boxes 117 and 1493, which moved against the old truth, are at 0.0003 and 0.0004 dex. TRUTH
  declares 116 sulfur boxes without a result (box 892 now ends). New stored reference
  data/benches/kintest/quick_sncho25_v3_ref1.npz.
- Parity against libref10 and libref10_SNCHO25 (made by the torch generator with its masks mf + M >= 2; its coefficients
  match VULCAN-JAX's capped ones within 3.48e-7): 48 passed; panel-P chains 1.3e-9 and 7.2e-9 dex, 0 and 2 step-count
  mismatches in 2,048,000 cell-calls. Checks on cce9052 (the merged settings file): CPU 122 passed; slow CPU 4 passed;
  parity and guard 48 passed; known answers, order, rate laws, kernel copies, dense LU, collision limit 26 passed;
  test_ad 81 passed in 1,040 s; test_export_cpp 19 passed.
- tests/test_rate_laws.py now caps a one-molecule-and-M row (fails on the old rule: the call does not finish its dt) and
  leaves a first-order row of 1e10 s^-1 alone (fails when every one-molecule side counts). tests/rate_probe.cu is kept: it
  checks what a call cannot show, each direction's coefficient apart and the derivative kernels' cap weights.
- test_vulcan_truth on both networks against the new quick references: 2 passed in 204 s.
- Speed, the old rule (main cecbf8e) against the new (cce9052), GPU 1 with nothing else on it, 4 paired rounds in separate
  processes, 262,144 cells over 800-3000 K and 1e-4-10 bar, 70-s calls, 10 timed calls a kind (the kernel-debt agent's
  CPU jobs ran on CPUs 8-23 meanwhile). Medians, old / new us per cell-call and the paired ratio's range: NCHO settled
  0.3631 / 0.3656 (+0.40 to +0.73 %), disturbed 16.487 / 16.440 (-0.18 to -0.32 %; steps 59.80 / 59.61); sulfur settled
  0.6730 / 0.6751 (+0.27 to +0.38 %), disturbed 37.18 / 36.62 (-1.1 to -1.8 %; steps 64.48 / 63.55). Settled gas takes
  one step a call both ways, so its +0.3-0.7 % is the k / n_tot of the new branch; disturbed gas takes fewer steps with
  the new cap. No cell recomputed, recovered or unfinished in any round. Below the 1 % rule for settled gas.

### 2026-11-10 14:25 to 2026-11-11 00:05: the kernels' design debt: argument structs, one owner of the layout, one step-control tail
Branch kernel-debt, from cecbf8e. A cleanup: no valid-input result moves. Scratch data of the proofs are not kept in the
repository. An independent review of the plan (go with changes on all three parts) added the derivative cases, the
checks of the old offsets and the 48 KiB compile-to-compile check below, and corrected two points of the plan: the
adjoint's stored attempt holds acceptance, not the step count, so its replays keep their partial restore; a shared
per-cell setup for the tangent and adjoint kernels would take about 18 arguments.
- (b) Arguments. The derivative kernels' helpers take five structs: `Streams` (the term streams and their rounds, the
  adjoint's gather), `Rates` (kf, kr, nd), `Tolerance`, `Attempt` (an attempt's error, Sherman-Morrison factor and
  acceptance, which `attempt`, `recovery_attempt`, `run_attempt` and `forward_step` return) and `StepState` (h, t, nok and
  the previous try's error, size and rejection; `forward_step` changes it through a reference). Counts with the macro
  lists expanded, before and after: forward_step 27 to 11, run_attempt 22 to 11, attempt 20 to 10, recovery_attempt 17 to
  9, jacobian_dot 21 to 12, rhs_dual 17 to 12, reverse_stages 16 to 12, rhs_vjp 15 to 11, rhs_vjp_n 12 to 8,
  save_attempt 8 to 4, tangent_step 22 to 17. Over 12: 24 functions before, 16 after, each with a written
  `size-limit-ok` reason: the six kernel entries (bound by name from KERNEL_ARGUMENTS; vjp_rec 49, jvp_rec 46, vjp 44,
  jvp 41, recover 28, call 27), jvp_cells and vjp_cells (the entries' lists), rhs 15 and rhs1 13 (`__noinline__`: a
  struct by reference would sit in local memory), cell_rates 13, rates 13, rates_tan 18, rate_cotangents 18,
  tangent_step 17, reverse_reaction 16 (six per-lane sums by reference). `tools/check_size_limits.py` now counts C and
  CUDA parameters (limit 12) with the folder's object-like macros expanded, the wider of two definitions (J_PARAM), and
  takes a reason from the line above a declaration's first line, so the three reasons wedged inside declarations moved
  above them; tests/test_size_limits.py fails on the old checker and on three mutants of the new one (no macro
  expansion, the old window, the narrower definition).
- (a) Layout. Each region is one ordered list of (array, doubles) in Python: `_tables.call_layout` (C_*, PERC),
  `recovery_layout` (R_*, PERWR), `scratch_layout` (S_*, SR_*, the strides), `rate_slots` (RATE_KR, RATE_ZERO, which the
  Jacobian stream's padding word also takes), `_ad_tables.scratch_layouts` (JS_*, VS_* and the stored attempt REC_*).
  The kernels carve with the names; VOFF, VR, `2 * NR + 2`, the vjp's `rho + 8 * NR` chain and `rp[NS + k]` are gone.
  Offsets only, no array moved.
- (c) One tail. `accept_try` and `next_try` (common.cuh, with StepState) end every try in call, recover, forward_step
  (adjoint) and jvp_cells. jvp_cells and vjp_cells keep their own per-cell setup (a shared helper would take 19
  arguments, counted with the checker on its signature); recover's differs (no screening, the block's barriers).
  tests/test_kernel_copies.py ties the copies on the edge inputs and needed no new case.
- Not done: one convention for the optional photolysis input (review finding 7). A nullable Jin on call and recover
  changes their parameter lists, KERNEL_ARGUMENTS, both launchers and the call kernel, so it is not a cleanup of this
  size.
- Proofs, compiled kernels (117 variants, sections compared one by one). (b): call, guard and recover equal in every
  variant; vjp and vjp_rec differ in 9 of the card's variants, jvp_rec in 4, jvp in none. (a): every section equal to
  (b)'s at 101376 B and in the test forms; at 49152 B only `.nv.info.recover`, which differs between two compiles of
  cecbf8e itself (12 of 40 variants), so NVRTC's spread, not the change. Every generated header equals cecbf8e's but
  for the 51 added offset macros and the dropped VOFF and VR (117 variants); every offset and size equals the old
  carving on 376 cases (14 networks, 2 and 4 warps, the cards' sizes, W in and out of shared memory, rates in and
  after the trailing block, odd and even species counts). (c) and the call's tail: recover in every variant, jvp and
  jvp_rec in 9, call in every variant; vjp equal.
- Proofs, host side. Reader, table, derivative-table, launch and scratch records and the exports of both default
  networks, dark and lit, equal at (a) (162 records; the 96 export files equal but for the 28 kernel sources).
  Registers, local memory, cells per block and warps per block of every kernel equal at (b), (a) and the final tree (44 records: both default networks, dark and lit).
- Proofs, outputs, bitwise against cecbf8e dumped today, 0 numbers differ in each: the final tree on the full set of
  the earlier entries (26,401,783 numbers: ten networks, call, recovery, jvp, vjp); the derivative half of it at (b)
  (3,065,911); the derivative cases the full set lacks (3,898,760 at (b) and at the final tree: photolysis on, 3
  directions with lm, rc and J tangents, adapt off, a vjp of up to 286 accepted steps against CAP 100, so the
  checkpoint replays run); the recovery's derivative path where it succeeds (1,867,982 at (b) and at the final tree:
  max_steps 20 and 40, 512 of 512 cells recomputed and recovered); the solvers built for a 48 KiB device (3,152,051,
  NCHO_photo and CHO_photo; cecbf8e compiled twice gave the same bits, and so did the final tree). The (a) step was
  not dumped apart: its compiled kernels are (b)'s at the card's size. Nor was (c) before the call's tail: its
  compiled kernels are the final tree's but for the call, whose code (c) left as cecbf8e's.
- Speed (12 paired rounds after 2 warm-ups, versions alternating, one process, GPU 1 with no other process for the
  139 samples taken over the run; 1,382,400 cells, 70 s calls continuing their chains; jvp and vjp on 4,096 unreacted
  cells over 500 s; the recovery on 16,384 cells with max_steps 1), each against cecbf8e, mean paired difference and
  its standard error:
  | | NCHO dark | NCHO lit | NCHO jvp | NCHO vjp | NCHO recovery | S25 dark | S25 lit | S25 jvp | S25 vjp | S25 recovery |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | cecbf8e | 0.3656 us | 0.6536 us | 464.8 ms | 946.0 ms | 335.0 ms | 0.6970 us | 1.0336 us | 1460 ms | 3084 ms | 2088 ms |
  | (b) | -0.06 +- 0.02 % | -0.04 +- 0.03 | +0.00 +- 0.11 | +0.05 +- 0.05 | +0.01 +- 0.02 | -0.08 +- 0.03 | -0.13 +- 0.04 | +0.67 +- 0.05 | -0.03 +- 0.05 | -0.01 +- 0.02 |
  | (c) | -0.24 +- 0.02 | -0.19 +- 0.03 | -0.07 +- 0.07 | +0.06 +- 0.07 | +0.03 +- 0.02 | -0.19 +- 0.04 | -0.22 +- 0.03 | +0.12 +- 0.05 | -0.18 +- 0.05 | +0.05 +- 0.04 |
  | (c) + call's tail | -0.22 +- 0.03 | -0.08 +- 0.03 | -0.17 +- 0.05 | -0.04 +- 0.05 | +0.07 +- 0.03 | +0.04 +- 0.03 | -0.01 +- 0.03 | +0.71 +- 0.06 | -0.02 +- 0.06 | +0.09 +- 0.03 |
  (a)'s kernels equal (b)'s, so it was not timed apart. The jvp kernel itself is the same code in cecbf8e and (b), and
  in (c) and (c) + the call's tail, yet those pairs differ by 0.67 and 0.58 % on the sulfur network: a paired standard
  error of 0.05 % understates the run's floor, which is near 0.7 %. Nothing reaches 1 %, so every step stays. The call
  with the shared tail against the call written out: +0.02, +0.11, +0.23 and +0.20 % (NCHO dark, lit, S25 dark, lit).
- Speed, second measurement (supersedes the jvp column above; asked for by an independent review of the branch, since
  +0.71 % with a 0.06 % standard error is near 1 % and unclear). 8 fresh processes, each building cecbf8e, a second
  copy of cecbf8e (`base2`) and the final kernels in a random order, then 6 rounds timing every version's jvp and vjp
  in a random order (4,096 unreacted cells over 500 s, the same inputs for every version); GPU 1 had no other process
  at any of the 9 checks between processes. Mean over runs of each run's relative difference against cecbf8e, its
  standard error and the range over runs:
  | | NCHO jvp | NCHO vjp | S25 jvp | S25 vjp |
  | --- | --- | --- | --- | --- |
  | cecbf8e | 460.96 ms | 940.03 ms | 1457.72 ms | 3067.53 ms |
  | base2 (the same code) | +0.24 +- 0.09 % (-0.12 to +0.68) | +0.02 +- 0.08 (-0.33 to +0.22) | +0.09 +- 0.09 (-0.32 to +0.53) | +0.07 +- 0.04 (-0.13 to +0.26) |
  | final | +0.10 +- 0.09 % (-0.24 to +0.40) | +0.11 +- 0.07 (-0.15 to +0.32) | -0.06 +- 0.07 (-0.31 to +0.32) | -0.01 +- 0.04 (-0.13 to +0.19) |
  The final kernels sit inside the copy-to-copy floor (up to 0.24 % in the mean, 0.68 % in one run), so the sulfur
  jvp's +0.71 % was the label artefact the first run's identical pairs showed. No cost; every step stays.
- After the review, three more commits. (1) The checker reads explicit specialisations (`f<int>`), operators
  (`S::operator()`, which it skipped before: 9 arguments) and commas inside `<...>`, and reports an unsupported
  declaration (a function returning a function pointer, a parameter list using a function-like macro) instead of
  passing it; each of the five forms is a test that the previous checker passes. (2) The last hand carvings went into
  the tables: the call's rates region (CR_*: v, the S v pieces at v + NR and the room G / RT may take beyond v), the
  retry copy (RSC_*: F, u, -Jn), the tangents' per-direction scratch (JD_*), and the recovery's dense W (W_DOUBLES,
  which the host's derivative scratch also takes). Generated values equal the old expressions in 376 of 376 cases;
  compiled kernels equal in 77 of 77 variants at the card's size and in the test forms (at 49152 B only
  `.nv.info.recover`, NVRTC's own spread). (3) Each remaining size exception says why its list stays flat; the
  derivative entries are bound by ADSolver, not by the C++ caller (which loads only call, guard and recover).
- Proofs added after the review, bitwise against cecbf8e: 2,261,235 numbers, 0 differ. The tangents' recovery with 3
  directions carrying lm, rate-table and J tangents at max_steps 20 (512 of 512 cells recomputed; recovered 511 on
  NCHO_photo and 327 on SNCHO 2025, the rest NaN as failed cells), the adjoint's recovery with lm, rc and gh, a vjp with
  adapt off over 1e8 s (234 and 286 accepted steps against CAP 100: the frozen checkpoint replay); NCHO_photo with the
  trailing block cut to 8 rows, so the call's rates sit after its arrays (rates_fit_in_block false), calls with h = dt
  = 1e3 s and of 1e6 s (many steps, the last cut to dt), and a recovery; the recovery with W in shared memory
  (NCHO_photo) and in global memory (SNCHO 2025). The recoveries at max_steps 2 all ended still failed, so their
  successful path in the solver is proved only by the earlier dumps' recovery cases.
- Checks on the final tree (4ac62a7; local main had not moved): CPU 115 passed in 26 s; slow CPU 4 passed in 100 s;
  parity and guard 48 passed in 226 s; known answers, order, rate laws, kernel copies and dense LU 25 passed in 333 s;
  test_vulcan_truth 2 passed in 208 s (NCHO 0 misses, worst 0.014 dex on boxes; sulfur pins held); test_export_cpp
  19 passed in 542 s; test_ad 81 passed in 1036 s (0 FAIL lines). The same tiers passed on 6204fa6 before the
  review's commits.
- A second review of the branch found three more gaps, fixed. (1) The checker passed three 13-argument definitions:
  a function-like macro reached through an object-like one (ARGS -> MANY(int)), a `noexcept(true)` qualifier, and a
  default argument holding `1 < 2`. It now reads the qualifier, looks for function-like macros after expanding, and
  reports a default argument with `<` or `>` as unsupported; the previous checker passes all three (shown on a copy).
  (2) The rate family's reasons: rates, its one group (the rate tables) would save 2 for an untimed change; rates_tan
  and rate_cotangents, a tangent or cotangent per rate-law input, the same group saving 2 of 18. (3) The last hand
  carvings: tangent_step and rhs_vjp_n step their stage loops by the table's spacing (A_K2 - A_K1, A_KD2 - A_KD1, with
  static_asserts that the stages are evenly spaced), and the recovery's ints have their own table (RI_PR, RI_NEXT in
  R_INTS). Machine code equal to 6827801's in every one of the 117 variants (the jvp variants' cubins byte-equal at
  the card's size; at 49152 B only `.nv.info.recover` differs, NVRTC's own spread); a first form that took a pointer
  per stage changed jvp's code in 4 variants and was not kept. The int region equals the old carving in 36 of 36
  cases. Checks after: CPU 115 passed, tests/test_size_limits.py 4 passed, pre-commit on the changed files passes. No
  GPU check was rerun, since no machine code changed.

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
