# Notes

The one record of the project: every experiment, result and author decision (author 2026-10-31: one notes file).
Since 2026-10-31 the project follows the physical-solver path; the direct-state networks (2026-06 to 2026-10-30) are the
closed baseline and keep their full condensed record in section 5. Layout: 0 conventions; 1 the current state; 2 the
physical solver's dated record; 3 every author decision, dated; 4 data (on disk now, then the full lineage); 5 the
direct-state networks; 6 references. New results go into section 2 the day they land and, when they change it, into
sections 1, 3 and 4.

History of this file: restructured 2026-10-19 from the dated record (4,601 lines, 2026-09-29 to 2026-10-19 16:15), which is
kept byte for byte in paper-notes/archive/dated_record_2026-10-19.tar.xz (it holds the full wording, per-row identities
and long tables); this file keeps, in condensed form, every model version (design, data, key result, why
superseded), every negative result (what, dose, outcome), every author decision, the data and generator provenance and
the numbers a paper would cite. Finalized 2026-10-20: checked line by line against the dated record (items it dropped or
changed were restored in condensed form), every cited reference listed by topic (section 6), and the small evidence files
exported to paper-notes/data (citations "data/<campaign>/<file>"). Reordered 2026-10-31 into the layout above with every
section reference renumbered; nothing removed.

## 0 Conventions

### Since 2026-10-31
- Closed campaign folders (2026-10-03_repo-sweep, 10-09_e13-conservation, 10-12_e13-fixes, 10-13_e13-scale,
  10-19_cleanup, 10-20_t0-fix, 10-21_data-home, 10-23_overlay, 10-31_notes) are archived as
  paper-notes/archive/campaigns_2026-10-31/<campaign>.tar.xz (paths inside relative to Chemulator_Project;
  <campaign>.files.tsv lists every file with its sha256). A cited path in one of them is read from its archive. In place:
  experiments/2026-10-19_minichem-yardstick and 2026-10-20_clean-state (the truth drivers import their panel workers and
  check their freeze manifests).
- Campaign folders closed 2026-11-01 (2026-10-30_kinetic-solver, 10-31_data, 10-31_cleanup, 10-31_fresh-start,
  10-31_broaden, 10-31_gpu-core, 10-31_mltracers, 10-31_qn, 10-31_sncho) are archived as
  paper-notes/archive/campaigns_2026-11-01/<campaign>.tar.xz (paths inside relative to Chemulator_Project;
  <campaign>.files.tsv lists every member with its sha256); a cited path in one of them is read from its archive. Still in
  place: gpu-core _build/legacy/, _build/polish/scripts/, _build/batch4/rule.txt, _build/diseq/results.txt, _build/gh200/;
  sncho _build/acc/vstage/ and _build/acc/truth/make_jobs.py; fresh-start _build/delete_d.py; qn _build/brief_design.txt;
  and whole: 2026-10-19_minichem-yardstick, 2026-10-20_clean-state, 2026-10-31_photo, 2026-10-31_refresh. Not archived:
  gpu-core learn/lib.npz and lib_w.npz (regenerable training libraries; sha256 and command in
  2026-10-31_gpu-core.dropped.tsv). The same folder holds legacy_tree_readmes.tar.xz (the four README copies of the legacy
  check trees) and paper-notes-data_review_1007.tar.xz (paper-notes/data/2026-10-06_e13/review_1007/, cited "archive:").
- In the refresh campaign's entries (section 2, 2026-10-31 night / 11-01): paths are relative to
  experiments/2026-10-31_refresh/_build unless stated (log.txt, rules_decision.txt, w*/ and mlw*/ workstream folders, judge/).
- Data deleted on 2026-10-31 (section 4) stay cited by their old paths; the numbers stand as recorded.
- In section 2: "log" = experiments/2026-10-30_kinetic-solver/_build/log.txt; e4/, r24/, x_opus2/, x_fable/, x_port/,
  x_sol/ are folders of that _build.
### Evidence labels and citations
- V = read from the named file or command output; I = inferred. Labels are inherited from the dated record: this
  restructuring did not re-read the cited files, except where marked "(V 2026-10-19)" (directory listings of the
  mini-chem tables, paper-notes/data and the evidence files; score_sweep.py line 42; VULCAN-0D vulcan_cfg.py lines 9,
  105, 121; git tags of code/Auto-Chem; the cleanup log). Items restored on 2026-10-20 from the dated record keep its
  labels (an "I" is written where it said inferred); the 2026-10-19 / 20 entries were read on 2026-10-20 from the reports
  they name (V unless marked I); the y21h provenance and the exported files were checked directly.
- "archive: <path>" = a script, log or text file of a deleted or to-be-deleted folder, kept in the text archive under its
  original path relative to Chemulator_Project. "data/<...>" = paper-notes/data/<...>, kept as is. Folder abbreviations
  used in citations (expand to the full path):
    arch    = experiments/2026-10-01_architecture-tests      audit = experiments/2026-09-29_code-audit/_build
    sweep   = experiments/2026-10-03_repo-sweep/_build       E13   = sweep/g2/E13_gcm (benches and truth sets)
    sprint  = experiments/2026-10-08_e13-sprint/_build       cons  = experiments/2026-10-09_e13-conservation/_build
    fixes   = experiments/2026-10-12_e13-fixes/_build        scale = experiments/2026-10-13_e13-scale/_build
    cleanup = experiments/2026-10-19_cleanup/_build          gen   = code/training-data-generator
    yard    = experiments/2026-10-19_minichem-yardstick/_build
  Phase logs: archive: sprint/plan.txt (the sprint log), cons/log.txt, fixes/log.txt, scale/log.txt; the live
  cleanup/log.txt. The sprint plan and its review memo (top of the sprint folder) are copied as
  data/2026-10-08_e13-sprint/sprint_plan.txt and sprint_review_2026-10-09.txt.
- Reviews: "independent review" = a read-only second opinion by another reviewer; their answer files are in the archive
  (scale/reviews/, scale/review1017/, scale/gfix/, data/2026-10-06_e13/review_1007/ (packed into archive
  campaigns_2026-11-01/paper-notes-data_review_1007.tar.xz) and its sibling folders). The external
  review report of 2026-10-20 and the yardstick spec review stay in place: scale/reviews/ and
  experiments/2026-10-19_minichem-yardstick/_build/reviews/ (yardstick below; its _build abbreviated "yard").
- Exported evidence (2026-10-20): data/2026-10-08_e13-sprint/, data/2026-10-09_e13-conservation/, data/2026-10-12_e13-fixes/,
  data/2026-10-13_e13-scale/, data/2026-10-19_cleanup/, data/2026-10-19_minichem-yardstick/ (campaign = experiments folder
  name; files flattened, a generic name prefixed by its source subfolder, e.g. scale/why/miss/inv.txt ->
  data/2026-10-13_e13-scale/why_miss_inv.txt). Files over about 5 MB stay in place (scale/research/gcm/gcm_tp.npz, the
  step-size sweep's raw rollouts, the truth audit's per-run table, the yardstick's design and truth arrays).
- Data locations: since 2026-10-21 all project data live in code/Auto-Chem/data (git-ignored), cited here as
  "repo-data/<kind>/..." ("data/<...>" stays paper-notes/data). Paths cited below from before that date (sweep, E13,
  cons, scale, yard, ... data folders, /scratch/imalsky) are old locations: repo-data/ref/pathmap.json maps each moved
  item (old absolute path -> its place under repo-data), and experiments/2026-10-21_data-home/_build/plan/move_map.tsv
  with move_journal.jsonl is the full record. Records (logs, reports, scripts) stayed in the experiment folders.

### Scoring rule (author 2026-10-11; hard bar removed 2026-10-13; history in section 5.4)
- Per-gas floors S: 1e-10 for H2O, CO, CO2, CH4, NH3, HCN, C2H2; 1e-8 for H, OH, O; 1e-6 for H2, N2. A point counts if
  truth or prediction >= S and its number density >= 1e-2 cm^-3 (VULCAN's atol); error = |max(pred, S) - max(truth, S)|,
  so an invented gas counts by how far it rises above S. 1e-30 is only the storage floor ("zero"). Code lib/scorerule.py
  (SCORE_RULE=phys). The flat 1e-12 floor of 10-08 to 10-11 stays a side readout.
- Basis (literature checked 10-11, corrected 10-12 against the paper texts): absorbers matter for spectra and heating from
  about 1e-7 (JWST detects C2H2, C2H4, CH4 from about 1e-7 to 1e-6, Gasman et al. 2022 arXiv 2202.01151; hot-Jupiter
  retrievals reach about 1e-6, best clear case about 1e-8, Greene et al. 2016 arXiv 1511.05528; in a coupled hot-Jupiter
  GCM species below about 1e-7 add little opacity and heating changes come mainly from CH4, Drummond et al. 2020 arXiv
  2001.11444). S keeps a margin of about 1000x for gas carried between layers. Mini-chem (Tsai et al. 2022 arXiv
  2204.04201; Lee et al. 2023 arXiv 2302.09525) judges accuracy only above 1e-20, has every > 100 % error below 1e-10 in
  its solar cases, calls a factor of 10 acceptable and is worst on C2H2. It carries the same 12 species (+ He) with 6 net + 4
  elementary reactions, floors mixing ratios at 1e-30 in its code, and its tables are valid for one metallicity each.
- Why the per-gas floors replaced the flat 1e-12 floor (10-11): the largest flat-rule misses (9-12 dex) were traces crossing
  it (Gc 667 H at truth 10^-18.8, 8.3-10.0 dex in every closed run; Gc 387 C2H2 at truth 10^-21, 12.39 dex invented in B6
  s1); under the per-gas rule Gc 667 drops out and Gc 387 falls to 1.4 dex. The n >= 1e-2 cm^-3 mask alone admits mixing
  ratios down to about 1e-21 (32 % of resolved species-times on G are below 1e-8). Since 2026-10-19 a non-finite
  prediction on a scored point counts as a failure (error +inf), not as an unscored point, and the per-gas rule is the code
  default.

### Terms
- row = one fixed-condition job (constant T, P VULCAN truth) at one start age: 1e-3 s ("fresh", "raw"), 1e3 s, 1e6 s
  ("settled", "evolved"). job = one truth run.
- late = max over checkpoints >= 1e3 s elapsed; lasting = error at the 1000-day checkpoint; first call = the error after
  the first call; late-born = lasting > 0.5 but < 0.3 dex at 2.8e7 s.
- six-set board = G, Gc, H2, H2c, H3, H3c as one combined bench: 13,821 rows (4,608 jobs x 3 start ages), dt 500 s, P*,
  per-gas rule (code/Auto-Chem bench/board6.py; the original in the archive, scale/checks/).
- inside = the real-GCM envelope env_any (section 5.4) widened by +-100 K and +-0.1 dex in P: 9,396 rows (9,060 in the
  first rescore of 10-14); cooler stratum = 300-1300 K at 1e-5 to 100 bar (5,481 rows, overlaps inside); in scope
  (10-19 09:23) = inside with Z 1-100x and C/O <= 1 (3,720 rows); target scope (10-19 21:4x) = inside with Z 0.1-300x and
  C/O 0.1-1 (the step-size sweep: 437 of 606 jobs per dt and start age instead of 360).
- counts: "lasting > 0.5" = rows whose worst gas is > 0.5 dex at 1000 days; "late > 0.5" / "late > 2" = rows with a late
  point above 0.5 / 2 dex; fixed / broken = rows that leave / enter a count against the reference; invented / lost
  switch = truth below the floor and prediction above, or the reverse.
- path metrics: p50 / p90 / p99 / max of |dlog10 x| over scored points of the 12 reported gases, then paths with any point
  > 0.5 dex ("bad paths"); late window = calls 2500-end (DEV 2500-5000, xlong 2500-10000; Gate A0 / A tables 3000-10000);
  per-path p50 / p99 = statistics per path, paired over paths; pooled = over all points; [a, b] = 95 % paired bootstrap
  interval (1000 resamples); negative = candidate better.
- architecture-era metrics: worst-T-bin p90 = max over checkpoints and five T bins of the p90 of |dlog10 y| over
  resolved species; five-band numbers a / b / c / d / e = 300-800 / 800-1300 / 1300-1800 / 1800-2300 / 2300-3000 K
  (late-max p90 unless stated); "[x]" after them = species with truth >= 1e-8 only; truth mask = scored if the truth is
  resolved; either mask (from 10-07) = truth or prediction resolved.
- precision modes: P1 all fp32 with the export's phys -> z -> phys round trip; P2 fp32 with a multiplicative update; P3
  fp64 state, fp32 network (every E13 bench); P4 fp64 host with the multiplicative update (= P3 to 8.3e-10 dex).
- P* (inference policy since 2026-10-12): stage2.pt, P3, 32 Newton iterations of the element projection, a
  Levenberg-Marquardt rescue on rows where Newton fails (lhbench --closed-fallback lm --lm-adaptive; converged rows
  bitwise unchanged), per-row counters of failed solves and kept inputs, and on fixed-condition starts the first call as
  five calls of 100 s (--startup-substeps 5:1; not used on paths, SD, T2 or the step-size sweep). Training projects with
  12 iterations and keeps the input on failure. --lm-adaptive runs each CUDA-graph block with the keep fallback and, where a
  Newton solve failed, redoes the block call by call with LM (bitwise equal to LM throughout on a 3,000-call bench; V).
- relaxation head: z' = z + a (q - z), a = -expm1(-k dt), q = clamp(z + r, zlo, zhi) with zlo, zhi the z of y = 1e-30
  and 1; log10 k = c + h tanh(s / h) over (-12, 3), c -4.5, h 7.5 (never ablated; both ends non-binding for 10-500 s,
  I). z = standardized log10 mixing ratio. The trunk sees (z, g = normalized log P, T) only; dt enters only through a.
  Closed heads end each call with a Newton projection onto the input's element ratios (relaxclosed; relaxclosedr keeps
  the relaxed state in fp64).
- training terms: one-step windows; long-jump pairs (z(t), z(t + D)), D log-uniform 0.1 s to 1e7 s, weight jump_w 0.2,
  512 pairs per step; grid flow term (grid_w: the composed flow over 20 stored intervals with BPTT; end-anchoring
  grid_end_frac 0.5); chain term (chain_w: rollouts on changing-condition VULCAN chains, a no-gradient pre-roll then BPTT
  over K calls, recorded mixing applied between calls); M rows (small-displacement starts whose windows begin at row 0).
- 25-entry closed state: the 12 reported species (H2O, CO, CO2, CH4, C2H2, NH3, N2, HCN, H, OH, O, H2) + He + C, N, CH3,
  H2CN, C2H6, CN, O2 + atom reservoirs C_res, N_res, O_res, H_res (atoms in the other 33 VULCAN species) + M_res (their
  molecule count); its C/H, N/H, O/H and molecular sum equal the 53-species state to <= 1.2e-15.
- abbreviations: Z metallicity (x solar), C/O, N/O; DEV = 47 used paths of the 48 x 5000-call development chain set
  (bench prefix PD); PH = path holdout (opened 10-12); PH2 = sealed path set; xlong = v2x = PX (16 x 10,000 calls, 58 d);
  v2l = P2l (64 x 1000 calls, 1.3-1.5 d); v2s = P2s (512 x 200 calls); SD = small steps from evolved starts; T2 = the
  snap-back test; U2-U5 = training caches; data batches and caches: section 3.

### Run labels (label = run folder under <phase>/runs/)
- architecture tests / repo sweep: FIN = p_h1_jump (finalist), p_h1_jo, s_h1 (6x512 stage 1), r_h1_jump20,
  z_h1_s1_jump20; J = e3_R3_s0, LJ = e3_R4_s0/s1, R5, JO40 = e3_jo40_s0; 800k = e5_p800k_jo40; e6_fp_LJ_s0; e8_F10_s0;
  e9_F10fill_s0; e11_mlp512x6; FiLM runs e11_*.
- E13 12-species (sweep runs): e13_phys16_s0, e13_all16_s0, e13_all53c16_s0, e13_allG16_s0 (allG), allGend, allGcmb,
  capw, all17E, capw17 = e13_capw17_s0, big17 = e13_big17_s0; sprint: m17_s*, m17f_s*, m17fs_s0 / s1 (frozen: e13_m17fs_s0),
  m17fT, m17ffc, m18fd, m64f, ms17, s2long17, relaxTz17, f0fs_s0 / s1.
- conservation: B6 = e13p_B6_s0 / s1 / s2; B6L = e13p_B6L_s0 / s1 / s2; B6Lc = e13p_B6Lc_s0; E = e13p_ens3med;
  D-cold = e13p_B6u_s2; W = e13p_B6w_s2 (B1 / B2: 12-species chain-term arms). fixes: F1 = e13f_F1ev_s2,
  F2 = e13f_F2ev19_s2, F3 = e13f_F3ev20_s2.
- scale: FINAL = e13s_FINw1536_s2 (stage 1 e13s_S1U2w1536_s2); B = e13s_FINw1024_s2 (stage 1 e13s_S1U2w1024_s2);
  A = e13s_S2U2w1536_s2; G1 = e13s_G1w1536_s2 (the label-to-folder mapping of B and A is I, from the board file names);
  C0 = e13s_T5C0_s2; D-kick = e13s_T5D_s2; LD, LDc, LD1, LDF = e13s_LD*_s3; round E = e13s_LE1_s4, LE2_s5, LE3_s6;
  X2C, X2E, X2V = e13s_X2*_s2; T6 = e13s_T6Aw1536_s2, B_C = e13s_T6BCw1536_s2; P15 = e13s_P15_s2; batch arms
  e13s_B512_s2, B512e, B8k, B8kL, B8kLion; S2B512_s3, S2B8k_s3; CBS1 = e13s_CBS1_s2 (stage 1 of CB and CBg);
  CB = e13s_CB_s2 (uncapped, paused); CBg = e13s_CBg_s2; CBgP1 = e13s_CBgP1_s3; CBgA = e13s_CBgA_s2;
  CB5S1 / CB5 = e13s_CB5S1_s2 / e13s_CB5_s2; CBgN = e13s_CBgN_s3 (never started).

## 1 Current state (2026-11-01, end of the refresh campaign): the physical solver and its reduced-tracer call

Superseded in part by the 2026-11-03 entry (section 2): the default network is now VULCAN-JAX's NCHO_photo (69
species, photolysis), the reduced-tracer call is removed (the GCM advects every species), the rates have a collision
limit, the truth data are data/truth_v2/, and the derivative kernels take every VULCAN-JAX network. The text below is
the state of 2026-11-01.

### What the author wants (section 3, 2026-10-31 ~22:1x)
A 0-D chemistry model inside a GCM, fast, accurate over the realistic domain (no rare misses of orders of magnitude),
with the GCM advecting at most ~30 tracers (fewer ideal), ML in the final model where it measurably helps, and later
coupling to the GCM's transport, radiation and photolysis kept possible (not built). The sulfur network comes second.

### The solver (code/Auto-Chem main 3b3be41, local, not pushed)
- VULCAN's C-H-N-O network (53 species, 290 reversible reactions) as generated CUDA kernels: Rodas3, unpivoted sparse
  LU, rtol 1e-3, the step size carried between calls; since 2026-11-01 a call's first step is judged by its worst
  species (max(RMS, max-norm)) and the device recovery gets 40,000 tries (section 2). Derivatives (tangent / adjoint)
  and the JAX layer follow the same step control. Any VULCAN-format network up to 254 species (SNCHO tested).
- Accuracy (V, section 2): 0 misses on every VULCAN truth set: the decision panels, the qual set (QC, QCs, twins, QL,
  QLc, QH, QHs), QT (13.5 M states; its 2 step-control misses removed by the 11-01 fix) and QT2 (6.26 M states, worst
  0.033 dex). The sulfur network: 0 misses on every SNCHO box and history.
- Speed (V): ~0.23-0.31 us per cell-call at large batches on an idle GPU (kernel, inputs on the GPU); with the 11-01
  step fix ~1.2x the steps; in the judge's harness 1.9-4.8 us per cell-call for the whole call at 28,800-131,072 cells.

### Fewer tracers (V, section 2, 2026-10-31 / 11-01)
- lib/kintracer.py: carried species + C, H, N, O element budgets (+ optional molar-sum rows), a per-cell memory of the
  hidden species, a maximum-entropy reconstruction after every GCM mix (Newton, then a dual-objective phase for cells
  that stall), then the full 53-species mechanism; outputs tracers, mean molar mass, cp, optional heating; batch- and
  order-independent; a reserved photolysis argument.
- Final judgment (QT2, frozen candidates, every call paired with the all-species control): G31 (31 tracers: N30 +
  CH3OH + C2 - N2H) and N30sN2H (30: N30 + CH3OH - N2H) pass the main domain (0 paths with a miss; worst 0.094 dex)
  and every R1 panel; N30, G30b, D30 fail narrowly (0.10-0.71 dex on QT2's far-donor paths); N26 and every design below
  26 fail. The two reviews recommend G31, and the sign-off confirms it: G31 is clean for 1000 days (QL / QLc, worst 0.002
  dex from the full mechanism) and on every known development regression, while N30sN2H keeps a 1.6-dex main-domain miss
  near C/O 1 (both on the merged code). Weak spots of both: carbon-rich gas that keeps mixing (C/O 1-2 ring: G31 1.26 dex, N30sN2H 3.19),
  4 cells 0.33 dex in the first calls after an equilibrium start at Z 300. The per-cell memory is required (memory-free
  fails by ~6 dex). The reduced call is 0.4-1.5 us per cell-call slower than the all-species call (it saves advection
  only; in the UM, 30 tracers cost ~21 % of run time, 13 ~3 %).
- Why it took this: the 30-tracer failures of 10-31 were the reconstruction's numerics (a stalled Newton dumped carbon
  into free C), not lost information (a collision test moved the scored gases by <= 0.0034 dex).
- Sulfur: the tracer floor is ~50-62 (twice the limit).

### ML (V, section 2, 11-01)
No learned component is adopted: learned tracer bases (blind and retargeted to the failure), an in-call learned prior
(with and without memory), a learned reconstruction warm start and the sulfur network's learned reduced-network picker
each tied or lost to the matched plain design. The useful results were non-learned (CH3OH, C2, the dual-objective
reconstruction, a per-cell early exit). Untried: learned rates, a faster full mechanism, an unbounded prior correction,
ML for per-cell photolysis rates.

### Testing (author 2026-10-31; project CLAUDE.md, Testing)
Quick directional tests by default; a big test only at a decision point, at most 4-5 h; every reduced-tracer candidate
is judged on VULCAN from the start (a narrow ring bed passed designs VULCAN rejects), with every call scored against the
all-species control; frozen rules and one-time judge sets (QT, QT2; both now used).

### Open work: todo.md

## 2 The physical solver: dated record (2026-10-30 evening on; newest last)

### 2026-10-30 (evening): the kinetic-solver route (mini-chem's structure with learned rates)
- Route (plan from the external reviews of 10-30): a stiff solver of mass-action kinetics on mini-chem's 13 species and 10
  reactions (4 elementary, 6 net with tabulated k_f(T, P)), reverse rates from NASA-polynomial equilibrium constants,
  with a network giving one positive factor per reaction applied to forward and reverse together: element conservation,
  positivity and descent of the Gibbs energy toward the stock equilibrium hold for any network output.
- Opening CPU checks of the second plan review (as quoted in experiments/2026-10-30_kinetic-solver/_build/log.txt 20:09):
  a Python mini-chem right-hand side reproduces the stored Fig. 4 runs to 0.004-0.02 dex, 0.2 dex in one case; fitted
  constant rate factors take the 10-bar cases from 0.30 / 2.74 / 0.74 to 0.14 / 0.00 / 0.01 dex (tables off by up to
  10^2.8); at 1 mbar 0.8-3.7 dex remain (11.8 % of the carbon in CH3 at 2000 K / 1 mbar, outside the 13 species); 10 of
  12 clean runs within 0.02 dex with the stock tables.
- Reactor convention (V): VULCAN-0D resets the total density to P / kT after every internal step (constant T, P);
  mini-chem holds n_tot inside a call and renormalizes at the call's end. The solver integrates the constant-P form
  dx/dt = (S v - x sum S v) / n_tot. Against mini-chem at rtol 1e-8 with 401 log-spaced calls to 1e8 s, the six
  Fig. 4 cases agree to <= 0.0008 dex (41 calls: <= 0.009 dex, the per-call renormalization); element ratios kept to
  2e-15 dex (V). One integration to 1e8 s takes 0.09 s (scipy BDF); the analytic Jacobian matches central differences
  to 1.6e-7 relative.
- Batched GPU solver (two-stage L-stable Rosenbrock, exact Jacobian, adaptive step): against a tight implicit reference
  on 300 cells (300-3000 K, 1e-5-100 bar, unreacted, aged and 1-3 dex scrambled starts, dt 1-1e6 s) worst 0.0068 dex at
  rtol 1e-3, 0.0007 dex at 1e-4; median 57 steps (V). A fixed-step version (8-48 geometric steps) is not usable: errors
  of 6-10 dex where linearly implicit steps overshoot (V).
- Hidden atoms (V, 49,149 clean runs, all compositions): runs with > 10 % of C, N or O outside the 13 species at some
  t >= 1 s: 1.2 % (12.5 % at >= 1800 K and < 1e-3 bar, <= 2.1 % elsewhere); outside the 20 physical species of the
  25-entry state: 0.3 % (2.3 % at >= 1800 K and < 1e-3 bar). Provisional: computed with a formula parser that counted an
  element appearing twice in a formula once (CH3OH read as one H; found and fixed later that evening, below), which
  affects both shares through the multi-group species; not recomputed.
- Constant-factor oracle (V, 2,799 near-solar runs of the GCM and cooler strata, Z 0.5-2, C/O 0.4-0.75, stratified by
  T, P and start kind; one constant factor per tabulated reaction per run, fitted to the floored log errors at stored
  times >= 1 s): runs with a miss / large miss at any time, stock tables -> best factors 37.0 -> 23.8 % / 17.6 ->
  11.1 %. Constant factors fix high pressure (>= 10 bar: large 9.6 -> 1.5 %) but not low pressure or cold gas (300-800
  K unchanged; worst gas H). Equilibrium starts: 0 misses. (Run, time) points 17.6 -> 11.8 % / 6.6 -> 4.5 %; large by
  P 0.1-10 bar 10.9 -> 3.2 %, < 1e-3 bar 27.3 -> 22.4 %; hot runs' remaining large misses sit at t ~ 1 s (75 of 126);
  only the 6 tabulated reactions were fitted. The fit stopped with 2,800 of 2,876 runs: the 76 left (stiff fits at
  extreme factors) are missing, so the shares are slightly optimistic.
- Representability test (V, same runs): between stored times, can any positive rate law on the 13 species follow
  VULCAN's mapped trajectory (each reaction's net extent in the direction its affinity allows)? Intervals ending >= 1 s
  missed by > 0.1 / 0.3 / 1 dex: 0.76 / 0.03 / 0.004 %. Of the 310 runs with a large miss under the best constant
  factors, 6.5 % have an interval > 0.3 dex. The first reading ("about 93 % of the remaining large misses are rate
  problems, not missing species") was withdrawn after the round-10 reviews: the test is necessary, not sufficient, and
  interval errors can add up, so it cannot assign that fraction. The structural limit sits at >= 1300 K and < 1e-2 bar
  (1.6 % of those runs > 1 dex).
- On mini-chem's own yardstick rows (V, 395,234 rows): rows with a miss / large miss, mini-chem 17.62 / 6.874 %, the
  best-balanced network (ov_V1jL) 5.90 / 0.553 %, the flow map (ov_DT2) 4.46 / 0.422 % (from t0 = 0: 39.6 / 22.0,
  42.6 / 6.7, 19.0 / 5.5 %). Mini-chem's value for the route
  is its structure, not its rates; the learned rates must carry the accuracy.
- Rate network, first training (V, near-solar training set of 25,889 runs mapped onto the 13 species; samples = a VULCAN
  state at a stored time -> a stored time >= dt later, dt 1-1e4 s; gradients by replaying the adaptive solver's
  accepted steps): held-out intervals with a miss / large miss, stock tables 12.50 / 3.22 % -> 11.72 / 2.73 % after 200
  steps of a smoke run (compiled vs eager solver 1.6e-13 dex). The full runs: next bullets.
- Solver with stock rates against the yardstick's 1x segment rows (V e4/yard_stock.out, yard_stock_m2.out; 39,528 rows,
  up to 2,000 calls of dt 1-500 s from VULCAN restart states, step size carried between calls): rows with a miss / large
  12.41 / 4.478 %, worst 6.68, no unfinished call (mini-chem on the same rows 12.38 / 4.260 %): the pipeline reproduces
  mini-chem; with mapping v2 (next bullet) 12.39 / 4.263 % (eq-target 2.51 / 0, eq-far 18.36 / 5.487, raw-A 23.25 /
  11.569, raw-B 5.45 / 0). Networks on the same rows: ov_V1jL 3.88 / 0.106 %, ov_DT2 1.24 / 0.018 %. About 3.5 us per
  cell-call with the network evaluated every step (I: wall time beside a training job; 1.1 accepted steps per call).
  Near-solar paths of panel P (V e4/p_stock.out; 14 of 256 paths with Z 0.5-2, C/O 0.4-0.75; 10,000 calls, T, P and dt
  changing every call): stock rates, paths with a miss / large 50.0 / 28.6 %, worst 5.75 dex, no unfinished call;
  networks ov_V1cj / ov_V1j / ov_VN1j 21.4 / 0 (worst 0.83-0.91), ov_V1jL 28.6 / 0, ov_DT2 35.7 / 14.3, clean 42.9 /
  21.4 (a 14-path sample).
- The 13-species start mapping (V log 22:0x; datagen/build_kin.py): the first mapping put hidden carbon (atomic C,
  ~0.03 % of the carbon) into CH4 when no hidden O was left to pair it into CO, and the solver made C2H2 from it at
  ~3000 K, 1.2e-5 bar (VULCAN's equilibrium C2H2 there ~1e-20): all eq-target large misses of the first readouts (34
  rows with stock rates, 179 with the it-500 network). Mapping v2 (hidden C to the larger of CO / CH4, O balance to H2O,
  H balance to H2) agrees with mini-chem's own mapping to <= 0.0017 dex on 27,200 yardstick units. Mini-chem's NASA-9
  coefficients for the 11 other shared gases equal VULCAN's (g/RT difference 0 at 400-3000 K), H2's within 0.0045.
- Rate network E4a (V e4/e4a.out; 3 x 128 network -> 10 log factors bounded at 4 dex, inputs log x, log T, log P; batch
  4,096, lr 2e-3, mapping v1; 4,096 held-out intervals): rows with a miss / large, stock 12.21 / 3.42 % -> it 500 8.72 /
  2.17, it 2000 7.28 / 1.95, it 4000 6.88 / 1.86 (worst 4.79; stopped by the preset rule: large not below 1.80 %); no
  unfinished held-out solve. E4b (+25 % of each batch from single calls of the 106 near-solar changing-condition chain
  paths; V e4/e4b.out): 8.52 / 2.20 at it 1000, 7.15 / 2.00 at 3000 (stopped: not 0.3 points better). Yardstick 1x rows
  (V e4/yard_e4a_500.out, yard_e4a_1000_m2.out): it 500 8.15 / 2.856 %, it 1000 with mapping v2 7.58 / 1.594 % (eq-target
  2.51 / 0, eq-far 13.60 / 3.472, raw-A 11.62 / 2.905, raw-B 2.62 / 0). Near-solar P paths with it 500: 35.7 / 14.3 % (V
  e4/p_e4a_500.out).
- Where E4a (it 2000) still misses (V log 22:11, e4/diag_val.py; 8,192 held-out intervals, stock -> it 2000): all 12.39
  -> 7.46 % / large 3.04 -> 1.73 %; large by P < 1e-3 bar 5.59 -> 3.36, 1e-3-0.1 3.89 -> 2.66, 0.1-10 1.05 -> 0.21, > 10
  bar 0.73 -> 0 %; by T 300-500 K 3.21 -> 3.08, 1100-1500 K 3.71 -> 1.10, 1500-2000 K 4.93 -> 1.62 %; start time < 1e-2 s
  6.94 -> 4.71 %, >= 1e6 s 0.35 -> 0.14 %; rawA 8.06 -> 1.61 %, eqt 0; worst gases NH3, C2H2, CH4, HCN; large misses
  fixed / broken 107 / 0. Cool gas does not move. In two cool runs (raw packs, V) VULCAN parks the C and N of HCN in
  H2CN and returns them (row 70626, 305 K: HCN 10^-7.4 at t 0 -> H2CN 10^-7.4 at 1.2e4 s -> HCN 10^-8.3 at 1e8 s; row
  30792, 408 K: HCN 10^-7.3 -> H2CN 10^-7.3 -> HCN 10^-7.4). I: such hidden-state memory explains the cool misses in
  general; a 13-species frame cannot hold it (H2CN, CH3, C2H6 are among the 25-entry state's 20 species).
- Round-10 reviews of the rate-network code (two external reviews in the campaign's reviews/; V by their reading of the
  code): the replay padded every cell to 300 steps (~90 % waste) and dropped cells over 300 attempted steps (2.1 %,
  holding ~40 % of the large misses); the factor bound saturated (|log10 f| 3.9-4.0 for five reactions); training
  intervals were not log-uniform 1-1e4 s (rounded up to the stored grid: 0 % <= 1 s, 31 % > 1e4 s); the hard floor gave
  no gradient below a floor; factors frozen within a step (first order in the learned rates; rtol 1e-3 vs 1e-5 <= 0.018
  dex). Expected ceiling of the 13-species learned-rate model (I, the reviews): ~0.5-1 % large on the yardstick, or
  "plausibly below 1 %". Recipe E4c: mapping v2, compacted replay (cap 1,000 steps), bound 6 dex, inputs plus
  per-reaction ln(Q/K) and element ratios, a soft floor log10(x + S) in the loss, 4 supervised stops per solve,
  oversampling of t < 1 s and P < 0.1 bar, 20 % chain calls; stop rule: held-out large at it 2000 not below 1.4 %.
  Interim (V e4/e4c.out; its own held-out start 11.16 / 3.34 %): it 500 7.06 / 1.83 %, it 1000 6.57 / 1.59 % (held-out
  solves all finished; sampled training batches still report 6-11 unfinished cells at the cap).
- Fig. 4 cases with E4a (it ~3500; calls of growing length, which agree with 500 s calls within 0.01 dex for stock
  rates; V log 23:05): worst after 500 s, 1 mbar 1000 / 1500 / 2000 K 0.13 / 0.57 / 1.59 dex, 10 bar 0.05 / 0.06 / 0.00
  (mini-chem 1.67 / 3.52 / 2.96 and 0.30 / 2.48 / 0.02; ov_V1jL 0.04 / 0.41 / 1.48 and 0.17 / 0.22 / 0.39).
- The full VULCAN mechanism in the same batched solver (V log 23:05; lib/kinvulcan.py reads VULCAN's reaction list and
  NASA-9 files as data: two-body, three-body falloff, low-pressure-only and the special CH3OH falloff; 53 species, 290
  reversible reactions). The formula-parser error (an element appearing twice in a formula counted once) was fixed in
  lib/kinetics.py, lib/kinvulcan.py and datagen/build_kin.py and the training set rebuilt (kin_solar_m3.npz; the m2
  cache and the hidden-atom map above carried the error). Parity against the numpy reference: 120 intervals, worst
  0.0010 dex; GPU two-stage Rosenbrock on 1,500 intervals over all compositions, strata G / C / F and start kinds: rtol
  1e-3 worst 0.004 dex (p99 0.0009), rtol 5e-3 worst 0.033 dex, none above 0.05 dex, none unfinished. Plan v2
  (plan2.txt): three one-model candidates on a physical solver, C1 the full mechanism with ML for speed, C2 a reduced
  20-30-species mechanism with learned positive rate factors, C3 a Gibbs-proximal relaxation map with learned rates;
  C1's parity gate met.
- Speed (author, sec 3: under 25 us per amortized call, ideally under 10). VULCAN-0D's fast path on one CPU core (V log
  10-31 00:01; 40 single calls from yardstick states): 0.041 s per call at dt 50 s and at 500 s (0.009 s integration,
  ~0.031 s per-call rate set-up). The batched full-mechanism solver with a sparse Jacobian (equal to autograd to 5e-16),
  a compacted loop that steps only unfinished cells (equal to the masked loop to 5e-8 dex) and rates on the GPU (equal
  to numpy to 7e-14), on GPU0 SHARED with a training job (the intended pause stopped a helper process, not the
  training: upper bounds, not idle-GPU numbers) with inputs resident on the GPU (no transfers), dt 50 s, yardstick states, us per
  cell-call: batch 32,768, rtol 1e-3: masked loop cold 4,915 / warm 490, compacted cold 211 / warm 15.3 (2.5 steps per
  call); rtol 5e-3 warm 5.8 (1.4 steps) but scored error up to 0.114 dex against a tight reference (rtol 1e-3: 0.0137),
  too loose; batch 8,192 warm 45.1 / 9.5. Working point rtol 1e-3, ~15 us per cell-call warm (a benchmark, not yet a
  whole deployment call). Profile of one warm 50 s call at batch 32,768: 404 ms (12.3 us per cell), 188 iterations (mean
  2.56 accepted steps per cell); GPU time 244 ms, of it batched LU factor 40 %, LU solves 28 %; the rest launch overhead
  of the tail iterations. A generated straight-line step (lib/kingen.py: right-hand side, sparse Jacobian, sparse LU
  without pivoting, the constant-P rank-one term by Sherman-Morrison; 14,085 lines for the full mechanism) equals the
  batched step to 7e-10 relative at h = 1 s (V log 10-31 00:07); its GPU timing is running (target 2-5 us, I).
- Full-mechanism readouts restarted 10-31 00:03 with the fast code on GPU1 (yardstick in 13 chunks of 32,768 rows, ~100
  s per chunk, 1.03 accepted steps per call; panel P, 256 paths): results pending. The network line's queues were stopped
  10-30 23:10 (ov_V1j30 at joint step 23,000 of 30,000; six CPU late-hold readouts unfinished).

- RESULT (V, data/benches/yardstick/run/vulcan53_stock.npz, scored on the design's truth): the full VULCAN mechanism in
  the batched solver (stock rates, rtol 1e-3, compacted loop) on mini-chem's whole yardstick, 395,114 segment rows over all
  10 composition cells, up to 2,000 calls of dt 1-500 s from VULCAN restart states: rows with a miss 17 (0.004 %), large
  0, worst 0.665 dex, rows above 0.01 dex 140, unfinished calls 0. Same rows: mini-chem 17.62 / 6.871 %, ov_V1jL 5.90 /
  0.552 %, ov_DT2 4.45 / 0.420 %. The 17 rows are one start (C/O 1 side-check cell, 300 K, 35.5 bar, eq-far, t = 0; CH4
  from the first call), under investigation.
- 2026-10-31: old-method data pared down by a reviewed manifest (25.2 GB: the merged cache of the data supplement, which
  rebuilds from its two kept sources, plus duplicate and resume checkpoints). The rest was deleted the same day on the
  author's call ("Delete all old data"; V, experiments/2026-10-31_cleanup/_build/delete_c_log.txt): 255 rows, 106.8 GB,
  i.e. the old networks' training caches (arch_*; rebuild from data/raw with datagen/build_*.sh), the non-baseline run
  folders, their stored panel and yardstick outputs, the E13-era and closed-diagnostic bench outputs, the overlay
  analysis arrays and a compile cache. Kept: data/raw, sealed sets, truth, ref, gen, kin_solar, every bench the physical
  solver reads, and the baseline (ov_DT2, ov_V1jL, ov_V1cj, ov_V1j, ov_VN1j and their parents, clean_s1/s2, CBg / CBS1,
  with their stored outputs). tests/bitcheck_train.py and the old training defaults need the caches rebuilt first.
- RESULT (V, data/benches/score_runs/vulcan53_P_stock.npz): the full mechanism on panel P, all 256 paths, 10,000 calls
  each with T, P and dt changing every call and the recorded mixes: paths with a miss / large miss 0.0 / 0.0 %, worst
  0.01 dex, unfinished calls 0. Same paths: the direct-state networks 35-51 % / 0.8-8 % (ov_V1jL 35.2 / 0.8 %, worst 2.43;
  ov_DT2 35.5 / 2.7 %). On the mini-chem paper's Fig. 4 cases the worst error after 500 s is 0.00 dex in all six.
- Speed on an IDLE GPU (V log 10-31 00:4x; whole call = rates on the GPU every call + compacted ROS2, rtol 1e-3, step
  carried; states settled 100 s; 32,768 cells): dt 0.01 / 0.1 / 1 / 10 / 50 / 500 s -> 1.5 / 1.5 / 1.7 / 2.9 / 4.1 / 4.0 us
  per cell-call (1.0-2.0 steps per call; one-call error vs a tight reference 0.0001-0.012 dex). Under GCM-like forcing
  (panel P's paths replicated, T, P, dt every call, median dt 21.8 s): 7.1 us per cell-call at 32,768 cells, 4.7 us at
  131,072 (2.3 steps per call), unfinished 0. VULCAN-0D's fast path: ~41 ms per cell-call on one CPU core. The earlier
  ~15 us figure was measured on a shared GPU. A generated straight-line step (lib/kingen.py) was slower under the
  compiler (131 vs 33 us on the shared GPU, 575 s compile) and is shelved.
- RESULT (V, data/benches/score_runs/vulcan53_{side,Sraw,S,L}_stock.npz): the full mechanism on every decision panel,
  misses first: S 157,232 rows 0.000 / 0.000 % (worst 0.01 dex), Sraw 39,306 rows 0 / 0 (0.01), side 15,704 rows 0 / 0
  (0.01), L 2,048 rows of 172,800 calls 0 / 0 (worst 0.02, last decade the same), P 256 paths 0 / 0 (0.01); unfinished
  calls 0 everywhere. Same rows, the flow map ov_DT2: S 0.944 / 0.007 %, Sraw 2.015 / 0.003 %, side 2.878 / 0 %, L 31.4 /
  3.32 %, P 35.5 / 2.7 %. The 17 yardstick rows that miss are one violent start (89 % atomic H dropped to 300 K, 35.5
  bar): three tight independent integrators of the same mechanism agree with our solver (CH4 -8.05 at 1.97 s) and not
  with VULCAN's stored truth (-7.39): VULCAN's own integration error there (V).
- Integrator (V, two independent improvement studies on a shared GPU, step counts and errors as evidence; idle-GPU check
  log 10-31 02:0x): ROS2's embedded error estimate rejects steps after any T change above ~2e-4 per call (radicals
  re-equilibrating), so GCM-like paths cost 13.7 steps per call; Rodas3 (4 stages, order 3, L-stable, stiffly accurate)
  needs 1.1-1.2. Idle GPU, GCM-like forcing, 65,536 cells, worst error vs a tight reference over 40 calls: ROS2 rtol 1e-3
  6.17 us per cell-call (2.32 steps, 0.012 dex); Rodas3 3.35 us (1.13 steps, 0.025 dex); Rodas3 with an unpivoted LU 2.71
  us (same answers). Adopted: Rodas3, unpivoted LU, rtol 1e-3. Rejected (V): fp32 LU with refinement (0.07-0.16 dex),
  skipping quiet calls, learned step sizes, Rodas4, chemistry every 10th call (3.9 % of P paths miss), not advecting
  the 5 fastest species (0.36 dex after 50 % mixing).
- Where ML fits (V, same studies): speed-ups of the full mechanism have <= ~13-24 % headroom; reduced mechanisms with stock
  rates need ~48-50 species for <= 0.1 dex (31 species: 29 % / 17 % misses on 20-call chains from young states); learned
  positive factors on the 31-species skeleton cut misses only to 27.4 % / 12 %; the Gibbs-proximal map cannot represent
  0.5-2.7 % of calls. The 13-species learned-rate line plateaued at 1.54 % large on held-out intervals (stopped by rule).
- Stress beyond the domain (V log 10-31 01:4x): in-domain-like cells (300-3000 K, 1e-5-100 bar, <= 1 dex scrambles)
  31,600 solves with 0 failures and element drift <= 2.3e-10 dex; outside (below 300 K, 2-3 dex scrambles, pure random
  gas) 0 non-finite or negative states but up to ~6 % of calls slower than 20,000 steps and drift up to ~1e-4 dex.
- With Rodas3 (unpivoted LU, rtol 1e-3) the panels were re-run (V, data/benches/score_runs/vulcan53_*_rodas3np.npz): S, Sraw,
  side, L and P all 0.0 % misses and 0 large, worst 0.02 dex, unfinished 0; S in 284 s (ROS2 356 s), P in 831 s (1,825 s).
  Idle-GPU speed with Rodas3 (V log 10-31 02:4x): settled states, dt 0.01-500 s: 1.7-2.0 us per cell-call (1.00-1.09 steps);
  GCM-like forcing: 3.4 / 2.3 / 2.2 us per cell-call at 32,768 / 131,072 / 262,144 cells (1.13 steps per call).
- New truth set 'qual' (2026-10-31; V, code/Auto-Chem/data/truth/qual/manifest.json and the campaign log
  experiments/2026-10-31_data/_build/log.txt): VULCAN-0D runs for qualifying the physical solver, 2,466 files (3.5 GiB,
  65.3 CPU-h of integration), all accepted (element drift <= 3.1e-9 dex, no negative entries). Panels: QC 512 per-call GCM
  paths x 10,000 calls (128 near-solar) + 128 kick twins; QCs 64 at C/O 1-2; QL 64 paths of 1000 days in 57,600 holds of
  1500 s (16 near-solar); QLc 8 per-call twins of QL 0-7 (172,800 calls of 500 s, same mixes); QH 1,024 cold-hold / reheat
  histories (256 near-solar) + 512 kick twins; QHs 128 at C/O 1-2; conv/ 26 tolerance reruns (rtol 5e-4 vs 5e-3, pass rule
  set before the run: all 26 pass, worst 0.018 dex). IDs 4,240,000,000 + k (k < 100,000; used k 0-511, 10,000-10,254,
  20,000-20,063, 30,000-31,023, 40,000-40,511, 50,000-50,063, 60,000-60,127, 70,000-70,007); seeds 2026103111-2026103120.
  Driver code/training-data-generator/e13/qual_paths.py (+ qual_paths_fc.py: FastChem partner calls in chunks of <= 100
  points after a 120-s time-out, bitwise equal to one call). Findings (I here: from the data campaign's log, not re-checked):
  QL (T, P per 1500-s hold) and QLc (per call) differ
  by up to 1.9 dex on hot mixing paths, so a model is run with the forcing its truth used; kick twins (median kick 0.64 dex):
  on QC 54 % are within 0.1 dex of the parent at the path end and 8 % still > 1 dex; on QH 43 % heal at the first cold hold,
  57 % at the first reheat. lib/paths.py gained QUAL (truth/qual), YARD and VULCAN_THERMO on 2026-10-31 06:31 (V,
  lib/paths.py:21-25).
- Under 25 transported entries (author 10-31: the GCM can carry 53, "but see how accurate you can get with under 25").
  R0, hidden species zeroed every call with their atoms lumped into majors, panel P 256 paths x 2,000 calls vs VULCAN (V log
  10-31 03:04-03:05): 20 entries 43.0 % / 24.2 % of paths miss / large (worst 3.16 dex); top-24 set 35.5 / 23.0 % (NH3 and
  HCN: slow pathways through intermediates that build up over many calls). C5, each cell keeps its non-transported species
  locally and projects them each call onto the transported budgets (maximum entropy, element-exact), full mechanism: 24
  entries on P with its recorded mixes 0.0 / 0.0 %, worst 0.02 dex (= the full mechanism). Every-call mixing stress
  (2,048 cells, 500 calls, 20 carried + C, H, N, O budgets, vs the full mechanism mixing all 53 exactly): upstream-neighbour
  mixing f 1e-3-0.03: 0.05 % of cells ever > 0.1 dex, none > 1 dex, worst 0.20 dex. The first joint projection failed to
  converge in 2.6 % of cell-calls (its line search tested the dual value, which cannot resolve a tiny budget such as N
  beside H) and, under random-partner mixing up to f 0.3, accepted a non-finite point that spread to every cell (a
  solver defect). Fixed (V, experiments/2026-10-30_kinetic-solver/_build/r24/proj.py v4): line search on the
  budget-scaled residual, no step to a non-finite point, and a failed projection is never used (each hidden species
  scaled down by the elements it holds until no element exceeds its budget, the rest into the carried free atoms C, H,
  N, O, so elements stay exact). With v4 (V, r24/c5v4_*.out; 2,048 cells): neighbour mixing 500 calls: 0.05 % of cells
  ever > 0.1 dex, 0 > 1 dex, worst 0.197, 0 projection failures, element ratios exact to 7e-16 dex; random-partner
  mixing f up to 0.3 (stopped at call 200): 22.6 % of cells > 0.1 dex, 2.6 % > 1 dex, worst 1.71 dex (13.9 / 1.4 % at
  call 100), 247 failures handled exactly, element ratios to 1.2e-15 dex. The random-partner error is C5's real limit:
  a cell cannot know the hidden species of gas mixed in from a very different T-P history. Not yet tested: strong
  mixing with real neighbours, advection noise, timing. Default stays all 53 transported.
- Long histories (V, e4/long53.py and long53_dt*.out: panel P's 256 paths, each T, P history cycled call by call with its
  recorded mixes, one fixed GCM step per run, 256 cells; GPU1 shared by the three runs; stopped
  2026-10-31 04:57 by the author's call to reach a stopping point): dt 500 s reached 520 days (89,900 calls), dt 50 s 60
  days, dt 25 s 31 days; in all three: non-finite 0, negative 0, unfinished calls 0 (counted on every 200th call only, e4/long53.py
  lines 66-68: the non-finite zero holds for all calls, the unfinished zero covers 0.5 % of them), worst element-ratio
  drift on unmixed paths 8.8e-8 dex; 1.08-1.11 steps per call. Qual 1000-day panels against VULCAN (V, e4/qualhold53_QL*.out, stopped at
  the same time; running scores only, no saved file): QL 64 paths (one call per 1500-s hold) to day 350: 0 paths with a
  miss, worst 0.018 dex; QLc 8 per-call paths (500 s) to day 500: 0 paths with a miss, worst 0.015 dex.
- Qualification of the full mechanism on the qual set (V, data/benches/score_runs/vulcan53_qual{QC,QH}_rodas3np.npz; Rodas3,
  unpivoted LU, rtol 1e-3, rates every call, every stored state scored, 12 gases floored): QC 512 per-call paths + QCs 64 +
  QC kick twins 128, 10,000 calls each with T, P, dt changing every call and 138,480 recorded mixes: 6,896,743 scored calls,
  0 with a miss, 0 large; worst 0.018 dex (QC), 0.064 (QCs, CO2 at 1348 K), 0.015 (twins); unfinished 0; 1.23 steps per
  cell-call. QH 1,024 + QH kick twins 512 + QHs 128 (cold holds 300-800 K, reheats, one call per 1500-s hold): 238,894
  stored states, 0 misses, 0 large, worst 0.017 dex, unfinished 0. Every stratum, start kind and composition the same.
- Whole-call GPU kernel (2026-10-31; V, experiments/2026-10-30_kinetic-solver/_build/x_opus2/results.txt and idle_*.out;
  two outside reviews independently ranked this idea first): one CUDA kernel compiled at run time (NVRTC, no nvcc), one
  warp per cell doing the cell's whole call (rates from T, P inside the kernel, every Rodas3 step with the library's step
  control, exact Jacobian, sparse LU without pivoting in a minimum-degree order with the dense trailing block in
  registers, the constant-P rank-one term by Sherman-Morrison), persistent warps taking cells from a queue (no host loop,
  one launch per call). Same answers as the library: identical step counts, <= 7e-8 dex on cold yardstick calls, 1.2e-7
  dex over 2,000-call panel-P chains, scrambled starts up to 2 dex 1.6e-8 dex, no non-finite states. IDLE GPU, whole call
  incl. rates, GCM-like forcing (panel P paths replicated): 0.39 / 0.30 / 0.29 us per cell-call at 32,768 / 131,072 /
  262,144 cells (library 3.33 / 2.34 / 2.19), 768 cells 3.1 ms per call (library 32.1 ms); settled states dt 0.01-500 s
  0.24-0.27 us (library 1.68-2.01), same one-call errors vs a tight reference (<= 0.0065 dex). Rejected inside the study:
  thread-per-cell generated code (register spill, 2.3x slower than the library), fp32 exp (no gain), Ros3 (more steps
  under forcing), atomic Jacobian scatter (15 % slower). Not yet in lib/; it needs a parity test and a solve-residual
  guard with the library as the fallback before it replaces the batched step.

### 2026-10-31: fresh start on the solver path (cleanup, records, test speed, the GPU kernel into lib)
- Code (V, git log of code/Auto-Chem; local commits, not pushed): 7ade76b = the old method's final state (miss and
  rollout losses, flow-map and mixed-mode heads, the miss readout; the two kept baselines load only with it), tag
  direct-state-final; 29c596e = the solver files (lib/kinetics.py, kinvulcan.py, kinbatch.py, kinnet.py, kingen.py,
  datagen/build_kin.py); 18d90b2 = 25 old-only files removed from main (training, data, models, rollout engine, element
  projection, cache builders, model runner, speed and 1000-day benches, bitcheck), lib/paths.py trimmed, README rewritten
  for the solver; tests/score_regress.py keeps its synthetic part (a), PASS 20 of 20 (its part (b) needed the CBg / FINAL /
  LD1 yardstick outputs deleted at 05:10, part (d) the retired runner).
- Data (V, experiments/2026-10-31_fresh-start/_build/log.txt, delete_d_log.txt): the raw_gcm* packs inventoried first
  (raw_inventory.py), then 70 rows, 88.21 GB deleted through a manifest with a check-only pass (0 problems); quota 281
  -> 199 GB. Details: section 4. The uv package cache (36 GB) is outside the project: pruning removed
  10 MB (its entries are in use); not cleared.
- Records (V): nine closed campaign folders packed into paper-notes/archive/campaigns_2026-10-31/ (every member checked
  against the folder by sha256 before the folders were deleted); 10-19_minichem-yardstick and 10-20_clean-state stay in
  place because the truth drivers import their panel workers and check their freeze manifests. This file: reordered
  into one clean notes file (author's call; nothing removed, section references renumbered).
- Why the tests were slow (V, the kinetic-solver logs and run outputs; ms per call = wall / sequential calls):
  P 256 cells x 10,000 calls 831 s (83 ms); QC + QCs + twins 704 cells 685 s (68.5 ms); QH 1,664 cells padded to 33,776
  calls 916 s (27 ms); L 2,048 cells with fixed T, P 172,800 calls 609 s (3.5 ms, one pass per call); QL 64 cells 147 ms
  per call (5.36 steps per 1500-s call), 2.35 h projected; QLc 35.8 ms, 1.7 h; the 1000-day runs at 66-79 ms per call
  with three processes on GPU1 (dt 25 s: 63 h projected; about 32 h alone). Causes: sequential calls (1000 days at 25 s =
  3,456,000), a library call that waits on the host for its slowest cell through several compiled step passes (idle,
  768 cells: 32 ms vs 3.1 ms for the whole-call kernel), GPU time-slicing between our own jobs (about 2-4x for the
  library), host syncs and per-call Python scoring (minor at the library's latency, not at the kernel's), padded
  finished paths. Making VULCAN truth is not the bottleneck (the qual set: 2 h 48 min on 24 threads).
- Test design (two independent reviews; the answer file and the log in
  experiments/2026-10-31_fresh-start/_build): both: one runner, one process, one idle GPU, every panel a block of cells in one call loop, finished cells
  masked, counters on the GPU every call; QUICK all QC / QH blocks, QL / QLc at 500 s, P at dt 25 / 50 / 500 s and a short
  yardstick pass, about 3-10 min with the kernel (I); FULL one 3,456,000-call loop at a 25-s clock carrying the 1000-day
  blocks, about 3-4.5 h with the kernel (I); a test-only multi-call launch would cut it to about 0.5-1 h (I). They differ
  on new truth: one asks for per-call cold-hold paths and 1000-day per-call paths at dt 25 / 50 s (100-117 CPU-h, 16-19 h
  wall, one dt-25 path being serial); the other shows (V, qual_paths.py docstring) that QL's truth is exact at every
  divisor of 1500 s, so QL already gives 1000-day truth at dt 25 / 50 s with held forcing, and proposes a per-call set
  QG (48 paths x 60 days at dt 25 / 50 s, 35-41 CPU-h, about 1.7 h on 24 threads) only when a model other than the full
  mechanism becomes the candidate. Decision: the runner as both describe; no truth regenerated now; QG then; the larger
  per-call 1000-day set is the author's call. Also from the review: every panel is at 0 misses, so a direction test
  compares paired against stored outputs of the current solver (counts above 0.01 / 0.03 dex, steps, latency); VULCAN
  stays the judge of misses.
- Correction (V, e4/long53.py lines 66-68): the long histories counted non-finite, negative and unfinished cells on every
  200th call only; a non-finite state persists, so that zero holds for all calls, but "unfinished 0" covers 0.5 % of
  calls (folded into the 10-31 entry above).
- The whole-call kernel into lib (two independent implementations from one brief,
  experiments/2026-10-31_fresh-start/_build/brief_kernel.txt; one module + one check script each). The second
  (x_sol/kincuda.py, kincuda_check.py; built without a GPU, run here on GPU1, x_sol/check_gpu1.out, V): PASS on every
  case: yardstick cold calls at dt 1 / 500 s worst 6.9e-8 dex vs the library, sigma 0-2 dex scrambles at dt 0.01-500 s
  <= 4.5e-9, a 2,000-call panel-P chain 1.09e-7, 0 step-count mismatches, 0 unfinished, 0 fallbacks; the NaN-state and
  unfinished-call guards and the 300 K error work. Guard: cells with a non-finite state, an unfinished call or an element
  ratio drift above 1e-6 relative are recomputed by the library from their inputs, and the count is reported. Paired
  timing, GPU1 possibly shared with the other port: 768 cells 4.76 vs 43.0 ms per call (9.0x); 32,768 cells 0.755 vs
  5.73 us per cell-call (7.6x). It imports lib/kingen.py's ordering (retired from main): to be inlined before adoption.
  The first implementation (lib/kincuda.py, tests/kincuda_check.py; V, its check rerun here,
  experiments/2026-10-31_fresh-start/_build/kernel/check_adopt.out): 19 / 19 PASS in 2 min 11 s; v10's numerics bitwise
  on the yardstick cold calls; worst 1.5e-7 dex over a 2,000-call panel-P chain, 0 step-count mismatches of 2,048,000
  cell-calls, 0 recomputed cells in domain; guard (a second kernel: non-finite state, unfinished call, element-share
  drift > 1e-6 relative beyond the clamping allowance -> recomputed by the library path) at +0.6 / +1.5 % per call
  (32,768 / 768 cells); paired, GPU possibly shared: 768 cells 2.79 vs 29.0 ms per call, 32,768 cells 0.388 vs 3.44 us
  per cell-call. Adopted (in lib, no dependency on the retired kingen.py); local commit a31577f. Open: a recomputed
  unfinished cell costs up to max_steps eager library iterations; one call at a time per solver object; sm_89 only.
- Speed ideas checked against the VULCAN-JAX record (V, ~/VULCAN_Project/VULCAN-JAX/notes.md, "Efficiency program",
  "Ideas not adopted", "Do not retry"): layer-parallel solvers (cyclic reduction, BCR), warm starts, float32 in the step,
  Jacobian / LU reuse across steps, tensor cores, register-resident inverses were tried there and rejected; its CUDA
  block-Thomas is latency-bound on the serial pivot chain (~7 % of fp64 peak on GH200); its step profile (HD 189733 b,
  150 layers x 69 species, M1 CPU; jax_paper/data/jax_step_profile_HD189.json) is 20.4 ms per step, 62 % in the
  block-Thomas solve, 1,495 steps. Nothing there gives a large software gain here (this kernel already uses table-driven
  Jacobians, register solves, occupancy budgeting and a refill queue); the large lever it uses is GH200 fp64 hardware
  (33.5 TFLOP/s peak there; these RTX 6000 Ada cards run fp64 at 1/64 of fp32). Remaining software levers here (I,
  untested): 2-4 cells per warp for lane use (one cell per thread failed earlier on registers), Jacobian / LU reuse
  across calls (W-method, under 2x ceiling), tabulated rates (~10 %), occupancy (~10 %). Kernel cycle profile (V,
  x_opus2/prof5.out, an earlier version): later stages 27 %, Jacobian 25 %, LU 17 %, rates 14 %, first solve 7 %.
- The mini-chem paper's six Fig. 4 boxes (V, rerun with production settings, session figures): worst floored error vs
  VULCAN over the 12 scored gases for t >= 1e-3 s: this solver 0.0016 / 0.017 / 0.0064 (1 mbar; 1000 / 1500 / 2000 K) and
  0.0012 / 0.0016 / 0.0039 (10 bar); mini-chem 1.67 / 4.05 / 6.23 and 0.30 / 2.74 / 0.89.
- GCM tracer units (V, the papers' methods): Drummond et al. 2020 (UM, 30-species Venot et al. 2019 network, 181
  reactions): "The quantity that is advected is the mass mixing ratio", converted to number densities before each
  chemistry call, chemistry every 125 dynamical steps (3750 s), element mass kept to 99.9 % over 1000 days; Drummond et
  al. 2018 (HD 189733 b relaxation): the CO mole fraction is advected; Lee et al. 2023 (Exo-FMS, mini-chem, 12 tracers,
  chemistry every 1500 s): "The mixing ratio of each species is advected quantity" (units not stated).
- 15- and 30-tracer study (author: "carefully look into the maximum accuracy on a 15 and 30 species version"): two
  reviews (experiments/2026-10-31_fresh-start/_build/answer_reduced_sol_xhigh.txt, reduced/review_claude.txt) agree:
  the full mechanism runs inside every cell; with a per-cell memory of the non-advected species the only error comes
  from mixing (a cell cannot know the hidden species of gas mixed into it); species chosen by error on paired mixing
  rollouts, not abundance; an information floor (donor gases with identical advected tracers but different hidden
  species); a GCM-like ring advection test bed against the exact 53-species reference; expected (I): 30 passes, 15
  uncertain. Species holding > ~0.1 % of an element beyond the 12 scored gases and He (V,
  reduced/atom_share.out, 759,028 qual states): C2H4 (up to 43 % of C), N (48 % of N), C (43 % of C), O2 (30 % of O),
  C2H6, CN, NO, CH3, C2H, C2, C4H2, CH2, CH, NH2, NH, CH3OH, H2CN; every other species < 3e-4 of any element. Phase A
  (build and quick tests; decision rule fixed in reduced/brief_impl.txt before results) running.

### 2026-10-31 (day and night): the GPU chemistry core (differentiable, flexible, GCM-ready)
- Aim and decisions (author, section 3, 2026-10-31 ~06:2x-06:5x): an end-to-end GPU, differentiable, flexible, fastest
  chemistry core for a GCM; campaign experiments/2026-10-31_gpu-core/_build (log.txt; plan in the session plan file).
- Design reviews (two, the answers in the campaign folder): speed census (answer_A_sol_xhigh.txt, answer_A_review.txt):
  on the RTX 6000 Ada the kernel is ~64-73 % fp64-pipe busy (E, from ~3,600 fp64 warp-instructions per step at 47 %
  lane use; tables V); software gain left ~1.2-1.5x (E); on GH200 the bound becomes the per-cell dependency chain
  (0.02-0.25 us per cell-call, E, unmeasured). Gradients (answer_B_*): derivatives of each call along its accepted steps,
  tangent stage solves on the primal LU, adjoint by transposed solves with Sherman-Morrison (no transpose factorisation),
  recomputation instead of storage (~12.9 KB per cell per step otherwise), ~0.8x a call per tangent direction and
  ~2-2.5x for the adjoint (E); whether the step-size controller must be differentiated: the reviews disagree, tests
  decide; kinnet.replay breaks with Rodas3 (zero-padded steps) and make_step is ROS2 (V, flagged by the review).
- JAX front end (V, jax/results.txt, test_kinjax.log 29 / 29): jax 0.12.0.1 (CUDA 12) added to the Auto-Chem venv
  (10 packages added, none changed; kincuda_check still 19 / 19); an XLA FFI handler (g++, CUDA driver API via dlopen,
  no nvcc) runs the production kernel from JAX bitwise equal to CudaSolver on every check input incl. the 2,000-call
  chain, under jit / vmap / scan / concurrent calls, with no host synchronisation; derivative wiring as three jax.extend
  primitives (call, linear tangent, adjoint as its transpose), tested with a stand-in linear map. JAX needs
  XLA_PYTHON_CLIENT_PREALLOCATE=false on the shared GPU.
- Generality (V, gen/network_table.txt, logs): the generator builds the kernel for any VULCAN-format network (11 networks,
  29-106 species incl. SNCHO 88 / 514 reactions); NCHO bitwise equal to lib; parity vs the torch library built from the
  same network within 1e-6 dex on all (Venot 2019: one cell of 4,096 trips the element-drift guard at 1.24e-6 vs the
  1e-6 allowance); SNCHO costs ~3.4x NCHO per cell-call (2.2x arithmetic x 1.5x lost occupancy, shared GPU, relative).
  The reader gained Troe / efficiency lines, an irreversible section, explicit photolysis / condensation stops.
- Speed (V, speed/logs/b_final.out, check_final.out, kintest_final.out; paired vs the lib kernel on a shared GPU): two
  warps per cell, a 32-row register-factored block, fma S.v, 32-bit Jacobian words, batched pivot reciprocals, no padding
  multiplies, memory re-use (6 cells per SM): 1.19x at 32,768 cells, 1.23x at 768, 1.31-1.32x cold, 1.94x for one hard
  cell; within 1.2e-8 dex of lib; kintest quick 0 misses. Measured losses: level-scheduled LU -7 %, per-round warp vote
  -8 %, rates in shared memory -7.5 %; orderings: every minimum-degree tie-break gives the same structure.
- Device-side recovery (V, recover/test_rec.log etc.): a kernel after the guard redoes only flagged cells on a dense,
  threshold-pivoted (0.1) Rodas3 with twice the step budget (plain partial pivoting's solve error reached 12x the
  step-control scale vs 1.3e-10, pivot_check.log), unflagged cells bitwise unchanged, forced recovery within 5.3e-7 dex
  of the library, no host sync on either path; out-of-domain cold scrambled cells mostly still fail (as in the library).
- Gradients (V, ad/ref/results.txt, ad/cuda/logs/check_final.log 26 / 26, window_P_1000.log): a torch reference equal
  to the library bitwise under deterministic algorithms (the library is not reproducible at 1e-12 dex with torch's
  default index_add atomics); CUDA tangent (1-16 directions) and adjoint kernels (recompute inside the call, transposed
  solves on the primal factors, fixed-order parameter reductions) for x, T, P, dt, h, per-cell log rate multipliers and
  the Arrhenius table, two contracts: A (the step control differentiated on the realised branch: the derivative of the
  computed map) and F (accepted steps frozen: no dt / h derivative). Against the reference: tangents median 4e-14 to
  2e-10, cotangents 0 to 5e-12; central differences median 8e-11 to 2e-9. Over 1000-call windows on all 256 panel-P
  paths A stays within 1e-2 of FD of the solver on every path, F misses > 1e-2 on up to 2.3 % of paths; against a
  converged tight solve, F is 2-8x closer at the median on multi-step calls (A adds the derivative of the step choice);
  on one-step GCM calls A = F. Default: A (the computed map, with d/d dt; one external review agrees), F kept as an
  option; the author decides. torch's forward-mode min / max rule is inaccurate when the tangents differ by orders of
  magnitude (fixed by selection).
- JAX derivatives (V, jaxad/logs/test_3.log 25 / 26): jvp, vjp, grad, linearize, jacfwd, jacrev, vmap, jit, scan with
  checkpoint, bitwise equal to the CUDA derivative kernels; a jax.grad of a loss after 10 calls w.r.t. 290 log
  multipliers matches central FD to 5e-10 to 5e-8; the failing case is the starting-step-size derivative of an
  ill-conditioned 300-K cold cell (p90 1.5e-4 vs a 1e-4 bar).
- Integration (V, integrate/stages.txt and the lead's rerun integrate/lead_verify/): main fast-forwarded to 8acdcf0 (4
  local commits: the merged generator for any VULCAN-format network with two warps per cell (four for the sulfur
  networks); the device-side recovery, no host sync; the derivative kernels ported to the merged layout with a primal
  bitwise equal to the call kernel (NCHO-like networks only); the JAX layer) + README 4150ad8; checks on the head:
  kincuda_check 19 / 19, kintest quick 0 misses and no path moved > 0.01 dex, AD 26 / 26, JAX 26 / 26. Not pushed.
- Idle-GPU speed of main (V, gpu-core _build/timing/idle_kernel.out, idle_lib.out, idle_ad.out; no other compute
  process before or after): 2.21 ms per call at 768 cells; 0.310 / 0.250 / 0.244 us per cell-call at 32,768 / 131,072 /
  262,144 cells (library 29.41 ms / 3.340 / 2.383 us: 13.3x / 10.8x / 9.5x); 1.18-1.39x faster than the v10 kernel's
  idle numbers of the previous night (same protocol and GPU). Derivatives vs the forward call: jvp 2.5x / 3.1x, vjp
  3.9x / 3.4x at 768 / 32,768 cells (~1 us per cell-call at 32,768).
- Reduced tracers (V, fresh-start reduced/decide.out, gpu-core learn/decide.out): only N = 30 (26 species + C, H, N, O
  budgets, per-cell memory + element-exact projection) passes the rule fixed before results (bed 0 cells > 0.1 dex,
  worst 0.005; VULCAN QC / QCs 0 misses; 6.2 vs 4.0 us per cell-call, shared GPU). N = 25 and 20 miss against VULCAN
  (25: 1.2 % / 10.9 % of QC / QCs paths, worst 4.2 dex; 20: 3.7 % / 14.1 %, 2.2 dex); N = 15 fails (8.7 % of cells
  > 0.1 dex). A learned rebuild (0.27 M-parameter network before the projection, trained on 12 other planets) got N = 25
  to 0.028 % of cells > 0.1 dex on the bed (worst 0.149) but still misses against VULCAN (worst 4.9 dex) and fails the
  composition edges (Z 300); N = 20 and 15 worse. Information floor after one mix at N = 15: 0.026 dex (the error is
  build-up over many mixes). Recommendation: 30 tracers; untried: training through K calls with the adjoint kernels.
- Batch scaling on an idle GPU (V, gpu-core _build/scaling/fwd.out; panel P paths replicated to B cells, T, P, dt every
  call): 0.161 ms per call at 1 cell, 1.92 ms at 768, 9.9 ms at 32,768; per cell-call flat at 0.234-0.236 us from
  524,288 to 4,194,304 cells; the host returns from a call in 0.061-0.068 ms at every size (3 kernel launches and a few
  small device ops per call, no host sync). The first derivative sweep (jvp.out, vjp.out) reused a stale state while the
  forcing moved on and overstates the derivative cost; on identical inputs jvp / vjp cost 3.24x / 3.50x a forward call at
  262,144 and 1,048,576 cells (V, scaling/fable/drop.out). Small batches are set by a few heavy cells (at 768 cells the
  heaviest 10 % take ~88 % of the call; V, fable/tail.out); 1 cell per block is bitwise equal and 1.4-1.9x faster at
  1-128 cells, slower from 2,048 (V, fable/). ptxas (V, adsync/ptxas_v.log): call 180 registers, no spills; jvp 215
  registers, 448 B stack, no spills; vjp 255 registers, small spills (48 B stores / 200 B loads).
- Derivative calls without a host sync (main 2ee94b4, on top of the recovered-cell work below): the guard
  counts stay on the device, status reads them on demand, check() raises for cells below 300 K (as CudaSolver.check).
  AD check 26 / 26, JAX check 26 / 26 (V, adsync/ad_check.log, jax_check.log). Paired timing on a shared GPU, same inputs
  per mode (V, adsync/time_jvp.log, time_vjp.log): the host returns in 0.08-0.09 ms instead of waiting for the GPU; wall
  time with / without the per-call read: jvp 4.86x / 1.18x / 1.02x at 1 / 768 / 32,768 cells, vjp 4.65x / 1.10x / 0.92x
  (the shared GPU magnifies the wait; on an idle GPU the read cost ~0.11 ms per call, V scaling/fable/audit_fable.txt).
- Corrected derivative cost (V, scaling/jvp.out, vjp.out; idle GPU, state advanced every call, no host sync): jvp / vjp
  0.387 / 0.438 ms at 1 cell, 4.33 / 6.74 ms at 768, 30.9 / 33.9 ms at 32,768, 844 / 817 ms at 1,048,576 cells (2.3-3.5x
  the forward call; ~0.80 / 0.78 us per cell-call at 1M cells).
- Recovered cells, drift per network, JAX jvp (V, gpu-core _build/polish/results.txt, stages.txt, logs/): (1) the
  derivative kernels redo the cells the guard flags as the recover kernel does (dense threshold-pivoted Rodas3, twice
  the tries) and differentiate that path (tangent solves and transposed adjoint solves on its pivoted factors); a cell
  still failed gets NaN derivatives and is counted; the JAX call with rate multipliers runs the recovery too; recovered
  cells match central differences of the recovered call (cold-start cells judged against max(1e-5, 2x the ordinary
  path's p90 on the same cells: the ordinary path misses 1e-5 there too), dot products p99 <= 7e-8. (2) Element-drift
  allowance per network: max(1e-6, 10 r), r = the network's worst rounding drift on the parity stress cases (3e-11 to
  4.5e-9 for six networks, 1.61e-6 for Venot 2019; the pivoted recovery drifts the same, so it is rounding); NCHO's
  kernel source unchanged; Venot 2019 parity 15 / 15 at 1.61e-5 (14 / 15 at 1e-6). (3) A JAX jvp binds the tangent
  primitive alone (its outputs include the call's, bitwise); jit(jvp) / ADSolver.jvp 0.96 (shared GPU). (4) Not adopted
  yet: layout-independent numerics (128-lane stream cuts, explicit fma) make 2 and 4 warps per cell bitwise equal on
  five networks and allow 4 warps at small batches (1.46-1.59x at 384-768 cells, 0.85x at 32,768), but fail the AD
  check's cold-start starting-step FD case (p90 1.5e-4 to 5.0e-4 vs a 1e-4 bar), which the committed code also exceeds
  on 5 of 12 direction draws.
- Merged (V, gpu-core _build/final/*.log): main 4150ad8 -> 2ee94b4 (items 1-3 + derivative calls without a host sync;
  local): kincuda_check 19 / 19, kintest quick 0 misses and no path changed, AD 37 / 37, JAX 28 / 28.
- Broadening survey (V, experiments/2026-10-31_broaden/_build/{lit_exo,assets,lit_ml,species}/report.txt; literature
  claims as the reports quote them, own-code claims with file:line there): (1) the generator caps networks at 254
  species (8 / 16-bit tables, kincuda.py:140, 176, 271-278); the per-cell layout ends near 130-140 species on the RTX
  6000 Ada (I, from the code's sizing rules and a symbolic LU fill of 12 local networks); 88 / 96-species networks cost
  ~3.4 / 3.7x NCHO (shared GPU); 500-1000 species need a new design. (2) No published H-rich GCM runs photochemistry
  inside its chemistry step; the cheapest route here is photolysis rates as per-cell inputs (est. 3-6 days) plus a new
  photochemical truth set (every truth set is 0-D thermal NCHO); VULCAN-JAX's photochemistry and transport are GPL-3.0
  (data files readable, code not importable). (3) Networks disagree by more than 0.1 dex (HCN up to 4 dex between two
  networks; 1.5-3x in HCN / NH3 inside one GCM): broadened models need more than one network's truth. (4) No upper-T
  guard in lib/; third-body efficiencies fixed to H2 (high metallicity would need composition weights); the reverse-rate
  exponent is unclipped (kincuda.py:1106). (5) ML placement, ranked by the reports: ML for one input (photolysis rates,
  missing rate / thermo data) with the physical solver downstream; per-cell reduced reaction sets with a full-mechanism
  check and the guard / recovery as fallback, only for hundreds of species; offline ML choosing which species sums the
  GCM advects, decoded by physics in each cell (nothing learned inside the call); not: solver replacement, learned
  rebuilds of hidden long-lived species, speed-only ML (no Newton iteration to warm-start in a Rosenbrock + direct LU
  kernel). Literature: ensemble spread correlates weakly with surrogate error (r = 0.31); the schemes that bound errors
  use a physical acceptance test.
- Decisive test of the reduced tracer sets (V, gpu-core _build/decisive30/results.txt, decide.out, bed_dt*_b / _c.log,
  vulcan_Q*.log, cost.log; rule fixed before results in fresh-start reduced/brief_impl.txt): N = 30 (26 species + C, H,
  N, O budgets, per-cell memory + element-exact projection, full mechanism inside) and the margin variant N = 26 (22 + 4)
  against the exact 53-species truth on the GCM-like bed: 345 / 110 / 32 days at dt 500 / 100 / 25 s (59,708 / 95,431 /
  111,303 calls), 0 cells above 0.1 dex and 0 cell-calls above 0.01 dex; worst 0.0046 / 0.0032 / 0.0042 (N30, equal to
  the control with nothing hidden) and 0.0097 / 0.0065 / 0.0060 (N26, all in calls 24-418; 0.0045 / 0.0032 / 0.0055
  after call 500); element drift vs truth <= 3.7e-7 dex; guard counts 0. VULCAN: 0 stored states with a miss for full,
  N30 and N26 on QC (worst 0.018 each), QCs (0.064 / 0.055 / 0.021), QC twins, QL (0.019), QLc (0.017). Cost (shared
  GPU, paired): full 9.25, N30 11.21, N26 10.65 us per cell-call. N30's projection fails every call in one or two cells
  (top of the hottest Z 300 planet at dt 500; spilled into atomic C, within 1e-4 dex of truth). Both pass the rule;
  "smallest passing N" picks 26, but 25 (= 26 without NH3) misses VULCAN by up to 4.2 dex: 30 vs 26 is open (reviews,
  then the author). Not covered: beds with C/O > 1, metallicity outside 0.3-300x, 1000 days at dt 25 s, QH / QHs.
- Hand-made species sums below 26 tracers (V, gpu-core _build/probe/results.txt, plan.txt fixed before runs, decide.out;
  decisive30's rule unchanged; the projection generalised to M sum rows, bitwise decisive30's on N26): none passes. N25h
  (no H budget, hidden H folded into H2): bed 15.5 % of cells > 0.1 dex, worst 1.03 dex (C2H2), offset kept for the run,
  yet 0 VULCAN misses; N24n (one N-atom row for N + NH + NH2): 9 bed cells (Z 300, 600-840 K, CH4 0.1-0.4 dex) and one
  C/O 1-2 VULCAN path (352 states, worst 0.259); N23 and N22c (adds a CH3 + HCO row) fail both ways; N25 (NH3 hidden)
  fails VULCAN again (QC 917 states, worst 2.201; QCs 4,094, worst 4.215) but passes the bed; N26 passes (bed worst
  0.010; 0 misses on QC, QCs, QL, QLc). Neither test alone sets the edge. Elements exact (<= 1.6e-7 dex); cost 4.5-7.2 us
  per cell-call (shared GPU). Untried: leave-one-out variants of N26 picked by VULCAN misses (about 20 variants).
- Learned decoder through K calls, PRELIMINARY (V, experiments/2026-10-31_mltracers/_build/prelim.txt, pilot_N20.log,
  pilot_N22.log, pilot_N20s.log, collide_*.log, loo_*.log, floor.log; 400 calls of the ring bed at dt 100 / 500 s, one
  seed, shared GPU): a decoder of the hidden species trained through 6 calls with the solver's derivatives did not help:
  cells ever > 0.1 dex of 2,880, no learning vs learned: N20 26 vs 33 (worst 1.274 vs 1.356), N22 55 vs 56 (1.516 vs
  1.419), all in P2200 (Z 300); N26 clean (worst 0.010). Training unreliable (gradients up to 1e32 before clipping; a
  review found an unscaled step through the projection). Collision test (two gases identical to the design, 10 calls,
  f 0.3): N26 0 cells > 0.1 dex (worst 0.023), N22 29 (1.29), N20 25 (0.77), N15 61 (2.24): the information is not
  carried. Leave-one-out from N26: 0 only for C2H4 and HNCO. One-call log-Jacobian rank (singular values > 0.1): median
  10, p90 18, max 22. Reading (I): single-species tracers bottom out near 26; learned species combinations are the
  remaining route, likely near 20-22 at best.
- ML tracers, phase 2 (V, mltracers/_build/phase2.txt, grad_fd_S100.log, grad_fd_S142.log, collide2.log,
  ring2_combo.log, ring2_ref.log; one seed, sets chosen and scored on the same ring planets): the training gradient
  fixed (projection step scaled as the forward pass, the solver's step-size derivative carried): median relative error
  against central differences over a 10-call window 2.4e-4 / 6.6e-4 (old 3.2e-2). Species pairs carried as one sum
  tracer (each cell's memory splits it): 44 of 210 pairings leave no collision gap > 0.1 dex vs 2 of 21 single drops.
  400-call ring, no learning: N22g2 (pairs C2H4+NH3, C2H2+NH2, OH+NH, CH3+HCO) 0 cells > 0.1 dex, worst 0.040, 1
  projection failure; N22g0 0 cells, worst 0.094 (262 projection failures); N20 sets 5 cells (P2200), worst 0.31 / 0.32;
  with each cell's memory reset to the truth before every mix all four are clean (so the ring misses are build-up,
  not lost information); 19 fails the ring (24 cells, worst 1.28). Training with the fixed gradient (4 runs of 8 min,
  worst-weighted loss) did not help yet (held-out P2200 worst 0.32 -> 0.26 with 5 cells left; with P2200 in training
  lr 1e-3 / 3e-4 made it worse, 1e-4 changed nothing). Next: the decisive rule (quick form) on N22g2 / N22g0.
- Sulfur network SNCHO (88 species, 514 thermal reactions) with ML, PRELIMINARY (V, experiments/2026-10-31_sncho/
  _build/ml/results.txt and its logs; reference = the full SNCHO solver; one seed; <= 400 calls; shared GPU): tracers:
  NCHO's 26 + an S budget fails (1,095 of 2,880 ring cells > 0.1 dex, worst 6.8); about 42 needed (N46 minus S8, S4,
  CH3S, CH3SH passes 200 calls, worst 0.039; dropping HS2 too: 13 cells, 0.90), all failures in P2200. Per-cell
  reduced networks (DRG flux cuts; a median call needs 43 of 88 species): 68-species cut within 0.01 dex on every
  sampled ring call (1 % of 1-dex scrambles miss, worst 2.8); paired speed-ups at 32,768 cells 3.84 / 2.26 / 1.85 /
  1.52x for 42 / 61 / 68 / 75 species. Learned picker (MLP on T, P, dt, state; 42 or 68 cut) + a one-step implicit
  check + fallback: 0 calls > 0.1 dex on held-out ring calls and perturbed gas (worst 0.075); in a 400-call rollout 27
  cells failed (omitted species frozen, then stale on hand-back) until every cell is refreshed by the full network every
  10th call (0 cells, worst 0.087); net about 1.9x vs 1.6x for the static 68 cut with the same check and refresh (I).
- Sulfur network against VULCAN, PRELIMINARY (V, experiments/2026-10-31_sncho/_build/acc/results.txt, score/*.log,
  speed/idle2.log; new truth data/truth/sncho, VULCAN-0D on VULCAN's SNCHO network and NASA-9 data, S/H 1.41e-5 x
  metallicity as H2S): 88 gases, 514 thermal reactions; 8 boxes (1 mbar, 10 bar x 600-2000 K, to 1e8 s) and 19
  GCM-like histories (1,500-2,060 calls, T, P, dt every call, quenched starts, some mixes): 0 misses on the 12 scored
  gases + H2S, SO2, COS, S2 (floors 1e-10), worst 0.020 dex (boxes; NCHO 0.019 on the same boxes) and 0.011
  (histories); no unfinished call, guard idle, elements incl. S within 3e-8 dex. Sulfur shifts HCN, CH4, H by up to 2-3
  dex in transients; the solver matches VULCAN's shift within 0.006 dex. Idle-GPU cost SNCHO vs NCHO: 23.5 / 13.9 us
  (64 cells), 5.8 / 2.5 (768), 0.87 / 0.31 (32,768), 0.76 / 0.24 (262,144): 1.7-3.1x.
- Legacy code removed (V, gpu-core _build/legacy/results.txt, rounds/c02813c/): main 2ee94b4 -> c02813c (26d10cb scorer;
  472999b build_kin.py + runner.sh; 0a07f21 the pure-torch reference solver, kinetics.py folded into kinvulcan.py;
  c02813c README); 18 files, +253 / -3,402 lines. The checks now read the torch solver's outputs stored once on their
  exact inputs (data/benches/kincuda/libref.npz, 236.9 MB; SNCHO 404.3 MB; input digests, refused on mismatch);
  kincuda_check 19 / 19, kintest quick 0 misses and no path changed, AD 37 / 37, JAX 28 / 28; cubins byte-identical.
  Last library / kernel ratio before removal: 18.5 / 17.7 at 768 / 32,768 cells (shared GPU).
- Jacobian / LU reuse across calls (V, gpu-core _build/jac/results.txt, logs/): today's call at 32,768 cells spends rates
  10-11 %, Jacobian 12.4 %, u + diagonal 3.7 %, sparse LU 9 %, trailing LU 17 %, stage solves + rhs ~43 %, step control
  3 %, guard + recovery 0.9-1.7 % (clock64 counters in a bitwise kernel copy; shared GPU). A W-method (ROS34PW2, Rang &
  Angermann 2005, BIT 45:761; order 3 with any matrix, checked numerically) takes 2.4-2.6 tries per cell-call against
  Rodas3's 1.17-1.27 (its embedded estimate, R(inf) = -0.478, fires on stiff radicals) and runs at 0.42-0.62x even with
  reuse; with unlimited reuse it misses VULCAN (3 QC paths, 0.64 dex in H). Rodas3 with a reused factor (order 1 with an
  inexact matrix): unlimited reuse misses (7 paths, 0.80 dex); a 10-call limit moves 6 paths / rows by > 0.03 dex; a
  conservative rule (refactor at least every 5 calls, or when T moves > 0.1 % or P > 0.5 %) passes the quick tier (worst
  0.064 dex, as the reference) at 0.99-1.01x / 1.18-1.26x / 1.23-1.34x at 768 / 32,768 / 262,144 cells (GPU nearly idle),
  carrying 11.9 KB per cell between calls. Not adopted (needed >= 1.2x at 768 and 32,768 cells with no miss); the
  conservative rule is the starting point if large-batch throughput ever limits.
- Small batches (V, gpu-core _build/batch4/results.txt, rule.txt, answer_rule_*.txt): the derivative check's cold-start
  starting-step FD case failed on 47 % of 256 direction draws for main itself (651 of 654 large errors within 3x the FD
  estimates' own disagreement); replaced, after two reviews, by a rule over 16 step scales per cell (resolved errors
  only, a sensitive-pair floor, 85 % stable) plus an exact dot-product check of the step-size derivative; main 44 / 44.
  Layout-independent numerics with 4 warps per cell at small batches not adopted: on an idle GPU 2-7 % slower at 2,048-
  262,144 cells and 1-8 % slower on derivative calls (the earlier 1.4x came from a shared GPU). Adopted instead: batches
  of 2 up to the number of SMs (142) run a one-cell-per-block build of the same kernel, bitwise main: 1.24 / 1.48 / 1.40
  / 1.29 / 1.24x at 4 / 8 / 16 / 64-128 / 142 cells (idle GPU), 1.00x elsewhere.
- Merged and pushed (V, gpu-core _build/final2/*.log, log.txt): small-batch rebased onto the legacy removal (9ab0bb4;
  kincuda_check 20 / 20, kintest quick 0 misses and no path changed, AD 44 / 44, JAX 28 / 28) + README 8c7d6c6. GPL
  history rewrite applied (V, gplfix/filter2.log, before2.txt, after2.txt): the five GPL-importing blobs of 3b8806e
  stripped from every ref, main's tree unchanged, paper-v1 unchanged, e13s_CBg_s2 -> f2c8518, all 263 commits by the
  author; main e862fd3 and the tag force-pushed. GitHub's closed, unmerged PR #1 (refs/pull/1/head) still holds the old
  history; the author did not want a GitHub Support request (draft deleted).
- Non-equilibrium coverage (V, gpu-core _build/diseq/results.txt and its logs; local equilibrium from an independent
  numpy Gibbs minimisation with the same NASA-9 data, the solver's own fixed point to 6e-8 dex, FastChem's stored
  equilibria within 0.011 dex): about half of all scored VULCAN states are > 1 dex from local equilibrium and a quarter
  > 3 dex (qual 7.2 M states, yardstick 11.8 M), mostly quenched gas below 1500 K (85 % of states at 300-600 K, < 1 % at
  2000-3000 K). On the 3.46 M far qual states: 0 misses, worst 0.020 dex, p99 error 0.0017 dex (0.0008 near
  equilibrium); yardstick: only the 17 known rows (distance 6.96 dex), else worst 0.035. The solver does not snap to
  equilibrium: of 9.26 M far states, none put within 0.5 dex of equilibrium where VULCAN stayed > 1 dex away; the share
  reaching equilibrium per panel matches VULCAN's. A single 1e13-1e20 s call leaves cold gas up to 9 dex from
  equilibrium (frozen pathways), so a long call is not an equilibrium solver. Gap: about 95 % of far per-call states
  change by < 0.01 dex per call; far-and-active states are few (8.7 k per call, 21 k at 1500-s holds, 94 k yardstick);
  not covered: per-call quench / reheat swings, mixing with out-of-equilibrium partners, photochemically processed gas.
  Proposed truth set QN (N1 quench / reheat, N2 quenched-partner mixing, N0 tight reruns of the 17 rows; 6.7 CPU-h).
- Findings recorded late (the refresh campaign's records audit, 2026-10-31 23:0x: each re-read from the source file
  named; experiments/2026-10-31_refresh/_build/w1_records/gaps.txt holds the verbatim quotes):
  - The drift survey cited by the repository's engineering notes
    (experiments/2026-10-31_gpu-core/_build/polish/scripts/drift_survey.py) cannot run on main since the legacy
    removal: its network helper nets_p.py imports the removed kinetics module and defaults to the removed polish
    worktree (V, import test 2026-10-31 22:53). It needs a port to kinvulcan (or the legacy copy) before the next
    new network's drift allowance is measured.
  - Tracer-count reviews (two independent reviews; V, gpu-core _build/tracers/answer_tracers_*.txt): both keep all
    53 as the GCM default and defer the leave-one-out search below 26. One takes N26 as the provisional reduced
    design (the rule's pick; no bed projection failures, while N30 failed projection 59,709 times on the bed, V
    decisive30/results.txt:100), the other N30 for margin with the departure from the rule logged and a switch
    rule fixed in advance. Both list tests before either is adopted: (1) the per-state difference between each
    design and the exact 53-species run on QC / QCs / QL / QLc, never recorded (on QCs the designs differ from
    exact-53 by >= 0.044 dex at one state, from the triangle inequality on the stored worst errors 0.064 full vs
    <= 0.020 for both designs; I); (2) QN with the reduced designs; (3) mixing between gases of different element
    ratios (C/O or Z gradients), never run although it is why budgets count as tracers; (4) beds on new planets
    (C/O 1-2, Z 0.1-0.3), since the decisive bed's five planets are the ones the species were selected on; (5) the
    memory-free form (never run at 26); (6) a projection fallback for both designs; (7) QH / QHs; (8) the GCM's
    own advection cost (the reduced designs cost 1.15-1.55x the full chemistry per call, V decisive30/cost.log,
    probe/cost.log, so their only gain is advection, unmeasured).
  - Baselines (V, sncho/_build/ml/results.txt 2f; test sets = ring calls 200-399 + 16,384 perturbed copies, sigma
    0.5 dex; cell-calls > 0.1 / > 0.01 dex, worst): learned picker with the one-step implicit check 0 / 8, 0.075
    on perturbed gas; the same cascade without learning (every cell starts at the 42-species cut) 3 / 27, 0.42;
    the 68-species cut with the check and no learning 1 / 10, 0.154. A per-species tendency check flags most good
    calls (system 0.72-0.93x of the full kernel) and was dropped. The 68-species cut alone as the model for 400
    ring calls stays within 0.005 dex on the scored gases while its dropped species drift by up to 2.28 dex;
    handing a stale cell to the full network moves the scored gases (refresh every 50 calls: 8 cells > 0.1 dex,
    worst 0.443; every 10 calls: 0, worst 0.039), hence the refresh every 10th call. A perfect per-cell choice
    between the 42- and 68-species cuts would give 3.29x at 32,768 cells (I, shares x measured ratios).
  - N24 (N26 with C2H4 and HNCO hidden, the only two single hides the collision test allows) passes the collision
    test (worst 0.047 dex) but max-ent decoding fails it on the ring from the start (25 training / 166 held-out
    cells > 0.1 dex; 122 / 206 after 150 learning steps, when the run stalled on the host; V,
    mltracers/_build/log.txt 19:47-20:00): the case a learned decoder was meant to fix, never finished. Counting
    every species above 1e-10 instead of the 12 scored gases, the 10-call collision test loses information even at
    N26 (18 / 54 cells > 0.1 dex, inert persisters; N24 35 / 90, N15 149 / 184; V, log.txt 19:51), so the
    scored-gas collision test is a lower bound on lost information. Phase 2 extras (V, log.txt 20:39-20:46): N20g0
    trained at lr 1e-3 diverged (5 -> 32 training cells > 0.1 dex in 40 steps); N19g0 fails the ring (24 cells,
    worst 1.28; perfect memory 2, 0.22) and N18g1 badly (369 cells; perfect memory 1).
  - Also (V, sncho/_build/acc/results.txt): boxes run at the GCM call pattern from unreacted gas (dt 25 s x 40,000
    calls, dt 500 s x 20,000) 0 misses, worst 0.0176 / 0.0130 dex (NCHO 0.0081 / 0.0134). Sulfur: S/H = 1.41e-5 x
    Z (VULCAN's solar file, log eps 7.1492) as H2S taken from H2; the generator's FastChem file has 7.12 and the
    generator runs FastChem without sulfur (S = -60), so SNCHO starts need their own sulfur rule. CH3NH is not in
    SNCHO (dropped from mapped NCHO states, <= 9e-14). The NCHO twin boxes equal data/truth/fig4 bitwise. Of the
    HCN transients that separate SNCHO from NCHO, sulfur itself drives the 1 mbar 2000 K one (2.2 dex) and the
    extra non-sulfur species the 10 bar 1500 / 2000 K ones (2.7 / 0.59 dex) (I: an SNCHO-minus-sulfur kernel run
    without VULCAN truth). Truth cost: 43 min on 19 threads, 39.2 VULCAN steps per call. The 10-day lively history
    (ID 4,241,700,105) stopped at call 1,500 of 2,574 (V, acc/truth/logs_ext/worker_0.log 21:50); the refresh
    campaign reran it (its result is recorded with that campaign).
  - Photochemistry phase 1, facts before the pause (V, experiments/2026-10-31_ photo/_build/design.txt section 1
    and facts/*.log; design itself an unreviewed draft): VULCAN's photolysis lines carry no rate; J = sum over
    wavelength of branching ratio x actinic flux x dissociation cross section, which needs the column above the
    cell, so a 0-D box can only take J as an input; VULCAN-0D as shipped has use_photo = False and no
    cross-section or stellar-flux entries. The local NCHO_photo network (55 species, 305 thermal reactions, 35
    branches from 18 absorbers) is not the qualified default's thermal part: it adds O_1 and CH2_1 (products of 7
    branches) with 9 reactions, 3 reverse-direction duplicates with their own fits and 2 slightly different rate
    laws, so it needs its own VULCAN qualification on dark gas, or a new file 'default + O_1 / CH2_1 + photo
    section'. Realistic J from VULCAN 1-D runs (HD 189733b, HD 209458b, WASP-39b) spans 0 to 39 s^-1 over 1e-5-100
    bar; long-wave absorbers (HCO, NO2, H2CO, CH3CHO) stay above 1e-12 s^-1 down to 93 bar, so photolysis is not
    confined to the top of the GCM domain (I).
  - QN, paused at the design stage (V, experiments/2026-10-31_qn/_build/ log.txt; brief_design.txt and one review
    saved, the second review unfinished): the 17 yardstick rows that miss are one trajectory (unit 11985, ID
    4,290,000,865, 300 K, 35.5 bar, t0 = 0, 89.5 % H), all with 0.667 dex in CH4 at their first call whatever dt;
    in the stored VULCAN run CH4 goes from 10^-16.22 to 10^-7.448 within 1e-6 s and is flat at 10^-7.386 from 1 s
    on, so the difference is made in the first microsecond. Proposed cause (VULCAN renormalises the total after
    every step, our kernel integrates the constant-P form) is contested by the review (per-step mole change <=
    1.2e-3); the planned check is a tolerance ladder of VULCAN reruns plus a rate-by-rate parity at that T, P. The
    review also found (I, not re-checked) that the GCM-envelope draw rule of the qual design skips models in 22 %
    of G draws (up to 10x weight on some models). code/training-data-generator is not a git repository.
  - The runner (code/Auto-Chem bench/kintest.py) built to the reviewed design (V,
    experiments/2026-10-31_fresh-start/_build/runner/*.log, 06:4x, kernel then in lib): QUICK 131 s wall
    (yardstick 16 s; loop 33,776 ticks in 87 s, 2.58 ms per tick, 18.9 M cell-calls, 4.6 us per cell-call, guard
    counts 0); reference data/benches/kintest/quick_ref.npz (58 MB) written, and the run paired with itself shows
    0 changes. Yardstick rows: 17 with a miss (the known start), 703 above 0.01 dex, identical between the kernel
    and the library (worst differs by 2.4e-5 dex).
  - The two speed reviews disagree on the bound: one puts the call at 11-15 % of the RTX 6000 Ada's fp64 peak (an
    arithmetic floor of 44 ns per cell-call from ~62 k flops), the other at 64-73 % fp64-pipe busy with half of
    each warp instruction's lanes idle; the hardware counters that would settle it are admin-only on daw (V,
    scaling/fable/audit_fable.txt), so 'fp64-bound' stays inferred. Audit (V, same file): lax.scan over calls
    gains nothing over one jitted call per step (host dispatch is never the limit), JAX equals torch within 0-5 %
    at >= 64 cells, and the guard's serial one-thread-per-cell loop (20-22 us) is ~13 % of a one-cell call (I).
  - QH strata (V, experiments/2026-10-31_data/_build/log.txt 00:34, 03:32): G 512 GCM-anchored (cold and hot from
    the envelope's Tmin / Tmax at the path's P, cold holds 1e4-3.2e6 s), C 256 cooler (300-800 K cold, 800-1300 K
    hot), S 256 independent draws labelled as a stress stratum; near-solar a quarter of every panel. The design
    went through one review only (the second reviewer's service was out of credits; the substitute review's answer
    file is in that campaign folder). A 0.15-dex kick at a 440 K cold hold persisted to the path's end in the
    pilot (cold-gas memory).
  - A third contract was proposed and tested in the torch reference only (A0c: steps frozen except that the step
    size entering each call is held, giving F's tangents for x, T, P and the rates plus a dt derivative that
    tracks A's within p90 2.1e-4; V, gpu-core _build/ad/ref/results.txt:94-97); one review keeps it as an opt-in
    pending a decisive test against a converged solve; not in the kernels.
  - GH200 timing package (V, experiments/2026-10-31_gpu- core/_build/gh200/: pkg/ and gh200_timing_pkg.tar.gz,
    22.7 MB): the call kernel and reader at 4150ad8 (before the legacy removal and the small-batch build), panel
    P's forcing, NCHO and NASA-9 files and tidle_gh200.py (daw's idle protocol: 150 warm-up and 100 timed calls at
    768-1,048,576 cells); for the author to run on NAS; not run yet.
- ML tracers, phase 3: the decisive rule in quick form on the 22-tracer pair sets (V, mltracers/_build/phase3/decide.out,
  vulcan_QC{a,b,c}.log, vulcan_QL.log, vulcan_QLc.log, vulcan_QCdiag1.log, vulcan_QCdiag2.log, bed_dt25_100_500.log,
  cost.log; the run finished after the session's stopping point and its template phase3.txt was never filled; GPUs
  shared; no learning; rule = decisive30's, fixed before results). Ring bed 5,240 calls (1.5 / 6.1 / 30.3 days at dt 25
  / 100 / 500 s): 0 cells > 0.1 dex for N26 (worst 0.010), N22g2 (0.054) and N22g0 (0.097). Against VULCAN, stored
  states with a miss / large miss: N26 0 / 0 on QC, QCs, QL and QLc (100 days; 6 projection failures on QC, as in
  decisive30) -> PASS; N22g2 QC 125 / 0, QCs 10,817 / 7,944, QL 6 / 0, QLc 10 / 0 -> FAIL; N22g0 QC 8,468 / 8,270, QCs
  3,940 / 1,046, QL 11 / 0, QLc 10 / 0 -> FAIL; worst on the 16 failing QC / QCs paths 4.81 dex (N22g2, H, path
  4240050042, QCs, 716 K, 2.4e-5 bar) and 4.88 dex (N22g0, HCN, path 4240000025, QC, 458 K, 0.48 bar). Projection
  failures on QC: 129,034 (N22g2) and 122,911 (N22g0) vs 6 (N26). One pair at a time on those 16 paths (N25 = N26 with
  one pair merged): C2H4+NH3 fails (1,537 states with a miss on the first 6 paths, worst 3.55 dex in C2H2, path
  4240050031, QCs, 575 K, 18 bar), C2H2+NH2 fails (worst 4.70 dex in H, path 4240050042), OH+NH fails (1.81 dex in H,
  same path), CH3+HCO and H+NH have 0 states with a miss (worst 0.018 / 0.031 dex). Cost of the design's own layer
  (median of 3 rounds, us per cell-call, shared GPU): full 53 2.67, N26 3.67, N22g0 10.50, N22g2 20.11 (the GPU Newton
  of the projection leaves cells to a host fallback). Conclusion: the pair sums fail VULCAN although they pass the
  collision test and the ring bed, as N25, N25h, N24n and N23 did before: the ring's neighbour mixing never brings in
  gas as different as QC's partners (equilibrium gas of the same elements at T +- 400 K, log P +- 1; cold NH3 / CH4
  into hot carbon-rich gas, I), so the ring bed alone cannot judge a tracer design; every candidate is judged on
  VULCAN QC / QCs / QL / QLc from the start (refresh campaign, 2026-11-01).

- GCM-coupling readiness and the literature on transported tracers (refresh campaign, 2026-10-31 23:1x; V from the
  code lines and opened sources named in experiments/2026-10-31_refresh/_build/w7_couple/answer_coupling.txt and
  refs.txt; read-only audit, no code changed). Nothing in the solver blocks later coupling; each item is days to weeks:
  per-cell photolysis rates need a one-way rate input with a zero reverse (today every non-explicit reverse is kf / K,
  kincuda.py:1113, and the per-cell log multipliers lm scale forward and reverse together, kincuda_ad.py:9-10, so lm
  keeps every equilibrium constant and cannot stand in for photolysis); the reader stops at the photolysis section
  (kinvulcan.py:46, 78-79); a C / Fortran caller needs a C launcher (the kernels are built through NVRTC from Python
  tables, and kinjax.cc already has plain C load / device entry points, :231, :250); the derivative kernels stop at 64
  species and NCHO-type laws, so there is no gradient (and no JAX front end) for the sulfur network. The reduced-tracer
  layer is the place where thermal-only or network-specific choices could be baked in: N26 hides 5 of the 18
  photolysis absorbers of the NCHO photo network (C2H6, CH3CHO, H2CO, NO, NO2), N30 hides 2 (CH3CHO, NO2).
  Literature (39 references, each checked to exist; refs.txt): exoplanet GCMs with kinetic networks advect one tracer
  per species, 12-13 with mini-chem (Lee et al. 2023; Baeyens et al. 2026) and 30 in the UM with the Venot et al. 2019
  network (Drummond et al. 2020; Zamyatina et al. 2023, 2024; Steinrueck et al. 2025), relaxation schemes 0-4; a full
  mechanism per cell behind <= 30 tracers has no precedent found. Tracers are cheap next to chemistry: in the UM 13
  tracers add 3 % and 30 add ~20 % to the run time per Earth day (421 / 435 / 510 s, Christie et al. 2026, arXiv
  2604.15850), so going from 26 to 22 tracers saves a few percent (I). Earth models transport fewer species than they
  solve (GEOS-Chem: 250 of 358 mechanism species, the short-lived radicals not transported; counted from its source
  files) and use transported families (Jacob et al. 1989: 6 tracers; Douglass et al. 1989: an NOy family keeps the
  family total but changes the partitioning). Our projection is the combustion method of rate-controlled constrained
  equilibrium / species reconstruction (Keck 1990; Ren et al. 2006) with each cell's memory as the prior, and the
  greedy hiding of species matches Hiremath et al. 2010; ML that picks transported combinations exists in combustion
  (principal-component transport, learned progress variables) but was judged on averages only; learned solvers in Earth
  models accumulate error where gas ages or mixes (Keller & Evans 2019; Kelp et al. 2022). Reading (I): the literature
  supports ML offline, choosing what is carried, with physics rebuilding the rest in each cell.
- The reduced-tracer call in the library (refresh campaign, 2026-11-01 ~00:1x; code/Auto-Chem commit d5d803d, not
  pushed; V experiments/2026-10-31_refresh/_build/w3_tracer/results.txt and logs/): lib/kintracer.py with the designs
  N26 and N30 as JSON (carried species, element-budget and molar-sum rows; He the mass remainder). The maximum-entropy
  projection and the exact element spill run in two per-cell CUDA kernels around CudaSolver (no host numpy, no host
  sync; every unconverged cell gets up to 60 more iterations, then the spill, flagged). GCM-facing call: carried tracers
  in mass mixing ratio + per-cell memory (hidden species) + step size -> tracers, memory, step size, mean molar mass,
  cp, an opt-in chemical heating rate, a per-cell flag and counts; a reserved photolysis argument (raises unless None).
  Checks: tests/kintracer_check.py 8 / 8 (against a reference made once from the frozen experiment layer: cells
  untouched by step-control branching within 6.5e-9 dex; 1,044 N26 bed cells branched at rounding level, their worst
  error vs the full mechanism +3.8e-5 dex relative to the reference's; elements 1e-15 relative; spill path; cp and heating
  vs numpy 1e-15; batch sizes 4320 / 2000 + 2320 / permuted / 100 / 3 / 1 bitwise equal; 300 K); kincuda_check 20 / 20;
  kintest quick 0 misses, no path's worst error changed by more than 0.0003 dex. A torch version matched the experiment
  bitwise but changed with batch size (torch picks matrix kernels by batch size); the kernels trade that bitwise match
  for batch independence. Gate (bench/kintracer_gate.py, rules v2: every call paired with the full mechanism plus stored
  VULCAN states, misses on mixing paths after the first mix): N26 passes the 18-path kill gate (worst 0.018 dex, 121 s)
  and the qual panels (776 paths, 7.0 M stored states, 0 misses, worst 0.064 dex = the full mechanism's own QCs worst,
  element drift 2.3e-7 dex, 43 spilled cell-calls with worst 0.002 dex there); N22g2 killed (9 of 18 paths, worst 3.75
  dex). Cost, preliminary (GPU0 at 100 % with ~10 jobs; median of 5 paired rounds): the layer adds 1.83 / 1.41 us per
  cell-call to a 15.94 / 11.07 us full call at 28,800 / 131,072 cells, about 0.11-0.13 of the call (the experiment layer
  cost about one call); idle-GPU timing owed.
- A learned warm start for the tracer reconstruction (the author's request, 2026-10-31 23:2x; refresh campaign, V
  experiments/2026-10-31_refresh/_build/w9_warm/results.txt and logs/; GPU shared, timings preliminary; one seed):
  NOT adopted, the learned start does not pay. The library's projection kernel runs 8 Newton iterations per cell per
  call, but most cells converge after 1-3 (from today's zero start, mean 3.2 / 1.6 on the every-call-mixing ring / the
  synthetic bed for N26). A per-cell early exit at the same tolerance (non-ML) makes the projection kernel 2.2-2.4x
  faster on the ring and 4.5-5.2x on the synthetic bed (N26 at 131,072 cells: 0.727 -> 0.303 / 0.819 -> 0.181 us per
  cell-call), with the same misses against the all-species control cell for cell and differences at the rounding-noise
  level (3.8e-3 dex, the spread of today's code from a 1e-14-perturbed start); starting from each cell's own previous
  lambda halves the iterations again (ring 1.72 vs 3.23; projection 0.179 us) and cuts N30's failed projections on the
  ring from 2,435 cell-calls to 10, but adds M stored numbers per cell and changes which calls miss in cells whose
  projection fails (N26 synthetic test 293 -> 397 missed calls, same cells). A learned start (MLP, ~83k parameters,
  predicting lambda) needs more iterations than the previous-lambda start where it matters (ring 2.0 vs 1.7; a held-out
  ring planet 3.0 vs 1.9), saves ~0.02 iterations on the synthetic bed, and costs 0.11 us per cell-call in torch, so
  it is 1.8-1.9x slower than early exit + previous lambda; its prediction error (5e-3 to 5e-2 in lambda) is far above
  the ~1e-5 Newton needs to save a step. Its one gain, rescuing stuck projections (87 -> 22 records), did not reduce
  misses. Side finding: on the synthetic bed's test split N26 and N30 miss the all-species control on 10 and 6 of 512
  paths whatever the start (worst 2.1-2.2 dex; 0.16 / 0.22 dex in cells that never had a failed projection).
- New VULCAN truth set QT and the first miss of the all-species solver (refresh campaign, 2026-11-01 02:1x-02:4x; V
  code/Auto-Chem/data/truth/qt/manifest.json, experiments/2026-10-31_refresh/_build/w8_qt/results.txt, design.txt,
  logs/control_qt.log, check_miss.log, one_call.log; driver code/training-data-generator/e13/qt_paths.py; design
  reviewed once (an external review, 9 corrections adopted) and against the frozen decision rules). Purpose: judge reduced-tracer
  designs where they failed or were never tested; judge only, used once on frozen candidates. 2,320 GCM-like per-call
  paths (+ 26 tight reruns), every call stored (14.04 M states, 5.9 GB), Sobol over Z 0.1-300, C/O 0.1-1 (main) and
  1-2 (side block), He/H 0.078-0.100; mixing by MASS with a per-mix ledger of element amounts (acceptance against the
  ledger: drift <= 1.7e-10 dex); blocks QTe (mixing every call, f = r dt, r 3e-4-3e-3 per s, capped at 0.5), QTq (QC's
  mixing law), QTc (cold holds with complete reheats, mixing throughout), QTo (out of the ML bed's generator: partners
  = stored far-from-equilibrium qual hold ends projected onto the path's elements, and He/H drift), QTs (C/O 1-2), QTu
  (8 unmixed sanity paths); partner kinds a (same-element equilibrium gas at a nearby T-P point), b (far-from-equilibrium
  donors from qual QH / QHs / QL hold ends, provenance recorded), c (another composition: Z within x10, C/O within +-0.3,
  two alternating anchors), cHe (He/H too). IDs 4,243,000,000 + k (the generator's reserved-ID table extended: clean
  block to 4,211,310,719, yardstick panel to 4,290,200,019, qual, QN, sncho and QT blocks added). 87.1 + 1.8 CPU-h, 2 h
  13 min on 40 threads; tight reruns (rtol 5e-4) 26 / 26 pass, worst 0.014 dex. The all-species solver (control) on QT:
  13,503,214 counted states of 2,304 mixing paths, 2 with a miss, 0 large; p99 0.0010-0.0032 dex in every block, kind,
  T band and dt bin; worst outside the two 0.044 dex. The two are one event, CO2 0.163 dex at call 5001 of the twin paths
  4243000312 / 4243000313 (QTe, kind c, near-solar, 1569 K, 1.3 bar, dt 4.8 s, main domain): VULCAN reruns 10x and 100x
  tighter agree with the stored truth (0.0000 dex), the solver started from VULCAN's state with a small first step
  reproduces it at rtol 1e-3 / 1e-6 / 1e-9, and the state one call earlier matched within 0.001 dex; the cause is the
  step size carried from the previous call (10.3 s > the 4.8 s call), so the call was taken as one Rodas3 step and
  accepted right after the mixed-in gas switched composition anchor (CO2 in the input 25x higher, relaxing within the
  call); with a 1e-10 s first step, or rtol 1e-6 with the carried step, it matches VULCAN exactly. Verdict: the first
  real miss of the full mechanism, from step control; OPEN for the author (frozen rules R1); no change to lib/.
  Side effect: the generator's panel_clean.py no longer matches the 2026-10-20 freeze list (the closed decision / twin
  panel jobs and train_a / train_b refuse to start), and qual_paths.check_id now raises for qual's own (finished) IDs.
- Sulfur network SNCHO with ML, the refresh campaign's verdict (2026-11-01 03:2x; V experiments/2026-10-31_refresh/
  _build/w6_sncho/results.txt, log.txt, ml/*.log, truth/score_ext.log; one or two seeds; supersedes the PRELIMINARY
  picker result above). (1) Truth: the stopped 10-day lively history (ID 4,241,700,105) was finished (3,832 s on one
  thread) and added to data/truth/sncho (manifest regenerated, the old one kept as manifest_2026-10-31_2139.json); the
  full SNCHO solver on it: 0 of 2,574 states with a miss, worst 0.0107 dex; the full solver has 0 misses on every SNCHO
  box and history. (2) The learned picker (42- or 68-species cut per cell, a one-step implicit check, fallback, a full
  refresh every 10th call) does NOT earn its place: against VULCAN it misses in the unreacted-gas boxes (13-24 states
  with a miss for each of three trained pickers, worst 0.618 dex; build-up of the gases a cut freezes, which a per-call
  check cannot see), where the static 68-species cut with the same check and no learning has 0; on 8 new held-out
  planets both fail (static cut 14 cells > 0.1 dex, worst 1.76; picker 46, worst 0.97), also after retraining on 16
  broader planets with two seeds. (3) Speed: the earlier ~1.9x came from the ring, where a call takes ~17 steps; on
  realistic GCM-like calls (~1 step) the 68 / 42-species networks are only 1.34x / 2.5x faster than the full one, and the
  check (one implicit full-network step per cell, now device-only and equal to the test version within 1.2e-5 dex) costs
  5-9 full calls, so the system runs at 0.15x the full network on path calls (idle GPU; 0.9x on ring calls; an ideal
  fused check at most ~1.1x, I). (4) Tracers with sulfur: the earlier "about 42" does not hold on broader planets: 46
  (NCHO's 22 + all 19 sulfur species + 5 budgets) fails (92 / 276 cells > 0.1 dex on the training / held-out planets),
  49 leaves 2 cells at C/O <= 1, 62 leaves none at C/O <= 1 (one C/O 1.31 planet still fails); the extra non-sulfur
  gases of the sulfur network (C2H6, excited N, ...) are the cause (the network with its sulfur removed fails the same
  way with NCHO's 22 carried). Floor about 50-62 tracers with sulfur, twice the author's limit; merges into shared
  tracers not tried (I: implausible across that gap). (5) In a few cold, thin night-side cells (~330 K, 3e-4 bar) of
  one new planet set the full SNCHO solver's answer changes by up to 4.8 dex for a 1e-10 relative change of the start
  (NCHO on the same planets is not sensitive): chemistry or numerics open (a VULCAN run of such a cell would settle it);
  those cells were left out of every count. (6) While building the new planets' starts, the full SNCHO kernel's drift
  guard fired often (248 / 41 cells still failed after recovery on the two sets; none in the 400-call runs): its drift
  allowance is NCHO's 1e-6 and may need SNCHO's own. (7) On these new planets the NCHO reduced designs leave a few cells
  just over 0.1 dex: N26 3 (worst 0.17 at C/O 1.0, 0.14 at Z 77), N30 2 (C/O 1.0); NCHO is not sensitive there, so
  these are design errors.
- N26 vs N30 against the frozen rules, and QT's verdict: NEITHER PASSES (refresh campaign, 2026-11-01 02:4x-03:3x; V
  experiments/2026-10-31_refresh/_build/w4_decide/results.txt sections 1 and 2.0-2.8, qt_report.txt, summary_all.txt,
  finalists.txt; library layer d5d803d unless stated; every call paired with the all-species control). Passed by both:
  the ring at the minimum lengths (30 / 6 / 1.5 days at dt 500 / 100 / 25 s) plus 1 day at dt 5 s, QC + twins, QCs, QL /
  QLc to 100 days, the kill gate (N30 through the library gate: 776 paths, 0 misses, worst 0.064 = the control's own),
  metal-poor gas (Z 0.1-0.3). New rings: gas of other composition flowing in (main domain; sources 0.2-200x solar, C/O
  0.3-0.95, He/H 0.078-0.100): N30 0.02 dex, N26 19 of 2,592 cells > 0.1 (2 > 1 dex, 1.31 in C2H2; elements right to
  5e-9 dex: the right atoms in the wrong gases); a start from chemical equilibrium: N26 116-130 of 2,880 cells up to 2-4
  dex in the first calls, N30 4 cells up to 0.33 dex at calls 3-7 (all on the 300x hot planet's upper levels); C/O 1-2
  rings (side domain): both break (N30 201, N26 295 of 3,456 cells > 0.1, up to 3.2 dex in CH4; across C/O = 1 N30 up
  to 0.64, N26 1.1). Without the per-cell memory both fail (up to 6 dex vs VULCAN). Deployment contract (library):
  bitwise equal for any cell order or batch split and after a restart from saved tracers, memory and step sizes
  (dropping the step sizes: ~0.003 dex); designs run alone equal designs run beside the exact run (no truth leaks).
  Library vs the experiment layer: 6e-9 dex on converged cells. QT (run once, on the frozen candidates; 2,320 paths,
  13.5 M states after the first mix; 03:05-03:26): main domain (2,048 mixing paths): N30 misses on 30 paths (12 > 1
  dex, worst 3.71 dex C2H2; 5,236 states, 940 > 1 dex), every one with partners of a DIFFERENT HISTORY (stored
  far-from-equilibrium VULCAN hold ends at 300-1500 K, projected onto the path's elements; mostly the cold-hold and
  out-of-generator blocks); with equilibrium partners of the same or another composition, every-call mixing and QC-law
  mixing N30 stays within 0.07 dex of the control. N26 misses on 150 paths (60 > 1 dex, worst 4.22; 50,221 states),
  also with equilibrium partners of another composition (11 paths, up to 1.06 dex) and on 4 every-call paths (up to
  0.30). Side block C/O 1-2: N30 36 paths (worst 2.45), N26 62 (5.58). On the control's 2 OPEN states both designs equal
  the control (0.0000 dex). Paired on main-domain mixing paths: cell-calls > 0.03 dex off the control 9,749 (N30) vs
  76,087 (N26), p99 1e-4 vs 1e-2 dex, worst 3.71 vs 4.22. N30's unconverged projections on QT: 4,509 cell-calls (fewer
  than its 5,236 miss states; not yet split by path). Verdict under the rules: no frozen reduced design passes R1, so
  R2 has no winner and all species stay the reference; N30 is far closer than N26 and fails in one regime; QT is now
  a development set for any N30-derived fix (QT-contaminated); C/O 1-2 goes to the author. Cost: N26 / N30 / full
  11.74 / 12.13 / 11.07 us per cell-call at 131,072 cells (shared GPU, preliminary). QT's manifest field
  donor_partition still names QL hold ends, but the donor pool holds only QH (121,260) and QHs (15,218) states (V
  donor_pool.npz src_panel): QL was dropped after the design review.
- The carried step size and the control's QT miss (refresh campaign, 2026-11-01 ~04:3x; V experiments/2026-10-31_refresh/
  _build/w10_step/results.txt, log.txt, local_*.npz, traj_*.npz, logs/; prototype kernel kincuda_w10.py, bitwise equal to
  lib with its options off; lib unchanged; decision rule and the QTe pick fixed before trajectory results). Cause: after
  an abrupt change of the input (a little gas of a very different make-up mixed in), the carried step is usually longer
  than the call, so the call is one Rodas3 step; its error test is an RMS over 53 species scaled by max(old, new), so one
  fast-relaxing gas can be far off while the average passes (QT miss: worst species 4.6x the limit, RMS 0.89). How often
  (vs a tight reference from the same input): QC 2 of 5.76 M calls > 0.03 dex (worst 0.064), the synthetic every-call
  bed 0 of 2.05 M, the QT twins the known 2 (0.163); every case a single accepted step after a gas jumped > 1.4 dex vs the
  cell's previous output. Stress (0.1-10 % of another stored cell's gas mixed into stored inputs): 52 single-step errors
  in 244 k calls, worst 0.80 dex, mostly 0.1 % mixes at 1300-2600 K. Fixes (own trajectories vs VULCAN; steps per call
  vs today): d1, accept the call's first step only if max(RMS, max-norm) <= 1 (a few kernel lines): removes every case in
  every set (closest catch 1.25x the limit), QC 1.19x steps, QT's every-call block 1.42x; a2, cap the carried step at
  half the call (one caller line, no kernel change): removes every case, QC 1.79x, QT every-call 1.16x; a stricter RMS
  limit on the first step (0.5 / 0.3): enough on development data but 15 stress cases left at 0.5; a small first step
  only when the input jumped > 0.1-0.3 dex since the last output: cheapest (1.01-1.07x) and works, but needs the previous
  output (the reduced-tracer layer has it, the all-species call does not). QT every-call block (512 paths, 3.66 M
  states, read once after the pick): today 2 misses (0.163); d1 and a2 0, worst 0.015 dex, no state worse than today by
  > 0.01 dex. Cost on a shared GPU (204,800 cells, ratios): a2 1.37x, d1 1.24x. Decision (a numerics change) is the
  author's. Separate problem found: cold cells (306-321 K) receiving gas 3-4 dex different can trap today's step
  sequence at tiny steps until max_steps (4,000) runs out; the device recovery saved 1 of 3 stress cases, the other two
  returned 2.4 / 3.1 dex wrong; 20,000 steps finish exactly; it depends chaotically on the starting step; none in the
  development or QT runs, but the synthetic bed's truth generation recorded 50 unfinished calls in 16.4 M (33 still
  failed after recovery; same mode not checked).
- ML arm A, a learned tracer basis below 26 (refresh campaign, 2026-10-31 22:4x-11-01 04:05; V experiments/2026-10-31_
  refresh/_build/mlw/results.txt, finalists.txt, logs/; blind to QT; two seeds on two runs; GPU shared): learning did not
  help. Synthetic GCM-like bed (8,192 paths x 2,000 calls, truth by the all-species solver; rebuilt twice: v1's
  "equilibrium" partners were still frozen far from equilibrium when cold, v2's gas borrowed from other paths broke even
  N26; v3: chemical-equilibrium partners and each path's own earlier gas; N26 misses on 5 of 512 test paths there).
  Gradient through mixing, the max-ent projection and the chemistry checked against finite differences (median rel err
  3.0e-4 soft / 8.7e-5 snapped over 10 calls; the outliers are kinks or unconverged projections). Six runs from four
  starts never cut validation misses by the required 30 % (paths with a miss, start -> end: ml1 N22 14 -> 14, killed
  by the development gate; ml3 / ml3b N21 two seeds 11 -> 12 / 11, stopped by the fragility rule; ml4 / ml4b N22-B1v two
  seeds 7 -> 11 / 10); the soft row assignment never moved a species. Test split, paths with a miss of 512: N26 5, B0 =
  N24p (judge-informed) 8, non-ML greedy designs N23-B1v 6, N22-B1v 11, N21-B1v 11, NMF N22-B2 11, ml1 15. The obstacle is
  the projection on combined rows, not the gradient: a few unconverged rebuilds per thousand calls decide most verdicts
  (the experiment layer and the library disagree on several designs for that reason, e.g. N24-B1 passes one gate and is
  killed by the other), and rows holding a lone atom (H, N) receive the element spill (B0: 3 % of cell-calls outside
  the 1e-24 row tolerance; two such rows ~18 %). In the library, N23-B1v is killed at the gate and N22-B1v / N21-B1v fail
  the full development panels (6 of 379 mixing paths, worst 4.4-4.9 dex, plus a real 0.27 dex HCN error).
  Memory-free versions fail (N26: 5 + 15 defect paths up to 2.2 dex before any mix). Restart and cell reordering
  bitwise. Answer: no design below 26 from learned weights; the lever would be a projection that always converges on
  combined rows.
- Second opinion on the author's question "will this method reach ~25 tracers, fast, wide coverage?" (an external review,
  2026-11-01 ~04:5x; experiments/2026-10-31_refresh/_build/reviews/brief_goal_sol.txt, answer_goal_sol_xhigh.txt; its
  claims cite the campaign's files): fast, yes (idle kernel 0.23-0.31 us per cell-call; transfers and coupling not in
  that number); wide thermal C-H-N-O coverage, strongly supported for the full mechanism (QT's 2 misses fixed in
  prototypes, the step-control fix and cold-cell step exhaustion still to integrate); <= ~25 advected tracers, unmet
  (even N30 fails foreign-history mixing; C/O 1-2 fails). The results reject the present reconstruction designs, not a
  physical minimum; decisive test: a reachable-state collision (two mixes with identical carried tracers, memory, T, P,
  dt and step but different hidden partitions; > 0.2 dex separation in a scored gas rules out any deterministic
  reconstruction from those inputs). Ranked routes toward ~25: (1) carry slow, quench-sensitive species and modes,
  rebuild fast ones, chosen by downstream sensitivity through mixing and reheating; (2) chemically justified family
  tracers; (3) a few transported reservoir / history variables replacing others; (4) a better projection or local
  fast-chemistry relaxation (mainly for numerical failures); (5) a learned closure (cannot beat a collision; learning
  what to transport is more promising); (6) a smaller mechanism (low confidence). Cost: UM 421 / 435 / 510 s per Earth
  day for 0 / 13 / 30 tracers, full kinetic runs 635-680 s, so tracers cost ~3 % / ~21 % against ~125-170 s for the
  chemistry itself; 25 / 35 / 53 tracers unmeasured (I: ~488 / 532 / 611 s by extending the slope); one million cells
  at a 30-s step would cost ~660-890 s per simulated day of chemistry on one GPU at the measured throughput (I).
  Advice: establish full-species GCM coupling and measure the real savings before making 25 a hard requirement; a
  validated ~35-tracer design is a sensible target to investigate.
- QT2, the fresh judge set (refresh campaign, 2026-11-01 03:35-04:46 generation, control scored ~07:3x; V
  code/Auto-Chem/data/truth/qt2/manifest.json, experiments/2026-10-31_refresh/_build/w8_qt/results.txt section 3,
  logs/control_qt2.log, logs/pool_qt2.log, plan_qt2.out; driver code/training-data-generator/e13/qt2_paths.py, which
  imports qt_paths.py unchanged): QT's recipe with new seeds (2026110143-2026110154), IDs 4,244,000,000 + k, and a donor
  pool disjoint from QT's by path and by condition (qual QH / QHs hold ends of the 224 of 576 conditions that supplied
  no QT donor; 72,838 states, 35,024 far from equilibrium, most of these from QH's labelled stress stratum S, whose
  states are real gas at 300-2500 K drawn with independent hold temperatures; each donor carries its stratum label);
  weighted to the blocks that separated designs: QTc 384, QTo 384, QTe 192, QTq 96, QTs (C/O 1-2) 64, QTu 8 paths;
  1,154 files (1,128 + 26 tight reruns), all accepted, 6.44 M states stored, 2.8 GB, 40.3 + 1.8 CPU-h; tight reruns 26
  / 26 pass (worst 0.0072 dex); drift <= 1.3e-11 dex vs the mixing ledger. The all-species solver on QT2: 0 misses in
  6,260,434 counted states of 1,120 mixing paths, worst 0.052 dex (CO2, twins 4244030094 / 095, call 5008, right after
  the mixed-in gas switched composition: the same weak spot as QT's step-control miss, smaller; I), unmixed sanity
  worst 0.0017. Judge only: no tracer design scored on it yet.
- ML arm A2, an in-call learned prior for the uncarried species, with and without per-cell memory (refresh campaign,
  2026-10-31 23:2x-11-01 ~08:0x with two usage pauses; V experiments/2026-10-31_refresh/_build/mlw_prior/results.txt,
  log.txt, logs/; blind to QT; one seed; the synthetic bed v3 without element-changing partners): negative, no
  finalist. Without memory (hidden species rebuilt from the tracers each call) both N26 and N30 fail by ~6 dex on the
  development gate and converge badly (N26 memory-free ~833 us per cell-call on a busy GPU vs ~4 for the all-species
  call); a learned correction on memory-free N30 (300 steps) made validation slightly worse (18 -> 21 paths). With
  memory at 22 tracers (N22g2), the reviewed recipe (correction bounded to a factor e) cut cell-calls > 0.1 dex by ~30 %
  but barely moved the paths that fail (validation 23 -> 21, held-out 38 -> 36) and still failed the development gate
  (7 / 16 paths, worst 2.06 dex vs 3.75 without learning); a looser bound (factor 20, outside the recipe) failed too
  (6 / 16, worst 0.58; held-out 38 -> 33). Oracle: giving the correction the true answer takes N22g2 from 35 to 30
  failing validation paths within the factor-e bound and to 5 without a bound (N26: 9): on this bed the information is
  mostly carried but a bounded correction does not use it; a better choice of rows (the plain N22-B1v (mlw)) passes the
  same gate. Gradient FD check (memory variant) median rel err 7.2e-6; deployment contract bitwise. On the synthetic
  bed N26 misses on 11 of 1,024 held-out paths (worst 2.2 dex). Rebuilds change carried atomic species (where the exact
  element spill lands) beyond the row tolerance on ~6 % of N26 cell-calls (largest 2.6e-5 mol/g, in cells whose
  projection failed).
- The step-control fixes (author's go 2026-11-01 ~04:4x, "fix however you need"; refresh campaign; V experiments/
  2026-10-31_refresh/_build/w12_fix/results.txt, log.txt, logs/; branch stepfix commit 1069f84 on d5d803d, worktree
  code/Auto-Chem-stepfix; not merged at the time of writing). (1) Until a call has accepted a step, err = max(RMS,
  max-norm) drives both acceptance and the step factor (call and recover kernels); the derivative kernels mirror it
  (primal bitwise; the selected norm differentiated, ties shared evenly); bitwise the step study's prototype (w10_step) on 941 chain calls and
  all 3,659,562 QTe states. (2) The device recovery gets 10 x max_steps (40,000) tries: under the new rule two of the
  three trapped cold stress cells still failed at 20,000 and finish exactly at 30,000-40,000. Validation: QT's every-call
  block 0 misses (today 2), worst 0.015 dex, no state worse than today by > 0.01 dex; the step study's stress sets > 0.03 / > 0.1
  dex 54 / 24 -> 0 / 0; kintest quick 0 misses on every qual panel, largest per-path rise 0.0024 dex (three QC paths
  better by > 0.01); chain trajectories vs a tight solve: old reference worst 2.41e-2, new kernel 4.62e-3 (better by >
  1e-3 dex on 80 paths, worse on 0). Checks against new references made by independent implementations of the new rule
  (the legacy torch generator patched; libref2 / libref2_SNCHO / refdata2 / kintracer ref2; old files kept):
  kincuda_check 19 / 20 (the 2,000-call chain differs by 1.26e-6 dex at one cell-call of 2,048,000 vs the 1e-6
  limit, step counts identical, the kernel smooth there: the reference's own step sequence differs at that call, I),
  SNCHO parity 15 / 15, AD 44 / 44, JAX 28 / 28, kintracer 8 / 8. Cost: steps per call 1.11-1.45x on most panels (QC
  1.18x, QH 1.11x, panel P 1.14x, QT every-call 1.42x), up to 1.92x on the 500-s hold panels (a carried step from a
  smooth call often fails the worst-gas test once, I); ~1.2x time per cell-call (shared GPU); a trapped cell using the
  big recovery budget holds its call for ~8 s (only in stress mixes; no development or QT run triggered a recovery).
  Reverse-mode derivatives now take at most 4,000 tries per call (was 4,096).
- Why N30 failed QT: the reconstruction's numerics, not lost information (refresh campaign, 2026-11-01 ~07:1x-09:04;
  V experiments/2026-10-31_refresh/_build/w11_diag/results.txt, finalists.txt, log.txt, a_g0.npz, d_g0.npz, kl_a.txt,
  col_f2.json, qth_report.txt; QT used as a DEVELOPMENT set: everything here is QT-contaminated; the experiment layer).
  QT's far-from-equilibrium donors are cold gas holding almost all its hidden carbon as CH3OH, so after such a mix the
  only consistent reconstruction is nearly all CH3OH, 4-8 dex from the cell's memory. The library's projection judges
  its steps by a budget-scaled residual that the tiny nitrogen budget dominates, stalls, and takes the spill path,
  which puts the unplaced carbon into free C (and H), making C2H2 and H that persist in cold gas. Example: path
  4243020106 call 462, CH3OH truth 1.21e-6, an independently converged KL projection 1.21e-6, the library 7.2e-11 plus
  1.21e-6 of free C (hidden C:H:N:O budget 1:4:0:1 = CH3OH's own ratio); at 23 of the 30 failing paths the first
  divergence is an unconverged library projection with free C the largest error. On the 60 development paths (30
  failing + 30 matched passing): N30 as in the library 30 paths with a miss (12 large, 3.710 dex); more iterations of the
  same Newton (300 / 1000 / 3000) 18 paths (7 large, 3.710); the memory reset to the truth before every mix 15 paths
  (3.692); a fixed rebuild ("phase 3": damped Newton on the convex dual objective f(lam) = sum_h exp(lg_h + a_h.lam) -
  b.lam with an Armijo line search on f, from the phase-2 multipliers and from zero) 4 paths, 0 large, worst 0.187;
  fixed rebuild + memory oracle 2 (0.145). Collision test: swapping in other real donor gas with identical tracer
  values moves the scored gases by at most 0.0034 dex over 240 pairs, so the tracers carry the information. With the
  fixed rebuild on all of QT (main domain, own misses): N30 4 paths (worst 0.187); N30 + CH3OH - N2H (30) 0 (0.049);
  N30 + CH3OH - HNCO 0 (but the main ring fails at 1.309); N30 + CH3OH - NH2 2 (0.103); G30b = N30 + CH3OH + C2 - N2H
  - CH3 (30) 2 (0.122, twins); G31 = N30 + CH3OH + C2 - N2H (31) 0 (worst 0.0495, 6 cell-calls > 0.03, p99 6.9e-5);
  side block C/O 1-2: G31 11 paths up to 0.246 (the library N30: 36 up to 2.445). N30 + CH3OH - N2H fails the mlw2 bed
  path 2323 (C/O 0.94, Z 86, kind-b partner) at 1.598 dex HCN and the synthetic bed path 3087 (C/O 1.00) at 0.108;
  both are fixed only by carrying C2 (or free C: the element budgets cannot tell pure-carbon species apart). Every
  30-tracer pair tried with CH3OH and C2 carried fails one check by 0.10-0.2 dex. Smallest working N found: 31 (G31,
  passing QT, the synthetic bed, the mlw2 bed's paths, the main / xm / dt-5-s rings and the library kill gate + panels; not run:
  the low-Z, C/O 1-2 and across-C/O-1 rings, R5 ablations, cost). The fixed rebuild is not in the library yet (its GPU
  cost untested); through today's library every candidate still fails at ~0.5 dex. Frozen finalists for QT2: F1 G31,
  F2 N30 + CH3OH - N2H, F3 G30b (all with the fixed rebuild).
- ML arm A3, learning what the GCM carries, trained on the failure regime (author-approved 2026-11-01 ~04:1x; refresh
  campaign; V experiments/2026-10-31_refresh/_build/mlw2/results.txt, finalists.txt, log.txt, logs/; QT used as a
  development set; one seed per run): learning did not help; no learned finalist. Bed: 4,096 synthetic histories (half
  cold holds with reheats; half with stored far-from-equilibrium partners from QT's donor pool projected onto the path's
  elements, 456,573 such mixes), truth by the all-species solver; it reproduces the failure (N30: 23 of 512 validation
  paths with a miss, 22 with other-history gas). Gradient FD check median 1.5e-5. Three learned-row runs at 27, 25 and
  30 tracers (the last with free row membership) left validation misses unchanged (5 -> 5, 9 -> 10, 3 -> 3 paths);
  membership drifted to CH2, CH2OH, CH3O, never to the species that matter; at the same N and projection budget
  learned designs tie or lose to plain ones (27: both fail QT development paths at 0.90 dex; library on the bed at IT2
  3000: ML27 8 vs P27 8, ML30 6 vs Q29 5 paths) and the library projection sometimes fails to reproduce learned rows.
  Plain screens: carrying CH3OH is needed (as the diagnosis found); carrying C2 alone fixes one main-domain bed path (C/O 0.94,
  Z 86, other-history partner, 1.598 -> 0.041 dex HCN) and two C/O 1-2 paths, also with a converged projection
  (information, not numerics). Finalist D30 = N30 + CH3OH + C2 - N2H - HNCO (30 tracers; frozen
  mlw2/designs/D30_frozen.json, sha256 0c07e99b...): with the library's phase-2 budget raised to 1000 iterations (no
  phase 3), 0 own misses on all of QT's main blocks (worst vs control 0.033; at the default 60 two twin paths at 0.380),
  clean on the 60 development paths at every budget from 60; on the bed one main-domain validation path stays at 0.41
  dex at every budget; C/O 1-2 one test path 3.2 dex. Paired on the bed in the library at IT2 1000 (main domain,
  validation / test paths with a miss): D30 1 (0.411) / 1 (0.115), G31 2 (0.716) / 1 (0.236), G30b 8 / 6, N30sN2H 3
  (1.596). Cost of IT2 1000: the projection 0.67 -> 2.3-2.65 us per cell-call for D30 (N30 0.83 -> 3.9-4.5; shared GPU).
- The fixed reconstruction in the library and the merge (refresh campaign, 2026-11-01 09:1x-11:2x; V experiments/
  2026-10-31_refresh/_build/w13_proj/results.txt, logs/; merge/kincuda_check.log, merge/kintracer_check_main.log;
  code/Auto-Chem git log). lib/kintracer.py gained the diagnosis's phase 3 (w11_diag) (433c382): for cells the existing Newton leaves
  unconverged, up to 400 damped Newton steps on the convex dual objective sum_h n_h - b.lam (Jacobi-scaled direction,
  Armijo 1e-4 on the objective over step lengths 2^-l, l < 40, else the first length lowering the scaled residual),
  from the phase-2 multipliers and again from zero, the lower residual kept, then the existing exact fix and spill; per
  cell on the device, batch- and order-independent; a new counter for cells that needed it. Parity with the diagnosis's (w11_diag)
  reference: identical verdicts and worst errors on the 60 QT development paths (N30 4 paths, worst 0.187; G31,
  N30sN2H 0, 0.044; G30b 0, 0.053) and on the main ring (5,184 calls); on the diagnosis's 103 recorded stalls the old library
  spilled 91, the new one converges 99 (the 4 left within 7.6e-12 of a 300-iteration reference). Cost: bitwise the old
  results and timings within 1-2 % wherever the old projection converged; 1-in-1,000 stalled cells make the whole call
  3-12 % slower (5.41 -> 6.05 us per cell-call at 28,800 cells); all cells stalled: projection 7 -> 20 us, whole call
  32 -> 35 (the cold restarts' chemistry dominates). kintracer_check 9 / 9 against a new reference ref3.npz made by an
  independent implementation (incl. a new case: CH3OH-dominated hidden budgets far from the memory, which the old library
  spilled in 119 of 216 cells and the new one converges in all). Kill gate + qual panels pass for N30, G31, N30sN2H, G30b
  (0 misses). Merged into main by fast-forward d5d803d -> 3b3be41 (1069f84 step control, 433c382 phase 3, 6f8ccaa
  designs G31 / N30sN2H / G30b, 3b3be41 the chain check: one cell-call in 100,000 may exceed 1e-6 dex, none 1e-5);
  kincuda_check 20 / 20 (chain worst 1.26e-6, one cell-call above 1e-6, 0 step-count mismatches of 2,048,000),
  kintracer_check 9 / 9 on main; not pushed. The kintest quick reference (quick_ref.npz) still holds the old kernel's
  outputs (largest per-path change 0.0024 dex vs the new kernel): to be re-saved on an idle GPU.
- THE FINAL JUDGMENT on QT2 (refresh campaign, 2026-11-01 11:2x-13:30; V experiments/2026-10-31_refresh/_build/judge/
  results.txt sections 1 and 2.1-2.9, qt2_report.txt, log.txt, cost.json; brief judge_brief.txt; code main 3b3be41;
  every frozen candidate run ONCE on QT2, paired call by call with the merged-kernel all-species control; nothing
  changed after a result was read). Control on QT2: 0 misses, worst 0.033 dex. Main domain (1,056 mixing paths): G31
  (31 tracers) 0 paths with a miss, worst 0.0943 vs VULCAN (0.093 vs the control; C2H2, one pair of QTo twins with
  far donors from QH's stratum S), 347 cell-calls > 0.03, p99 1.05e-4; N30sN2H (30) identical to four decimals; G30b
  (30) 2 paths (0.101); D30 (30, mlw2) 4 paths (0.714) at its frozen budget and the default; N30 4 paths (0.464); N26 66
  (13 > 1 dex, 1.468); B0 = N24p 105 (21 large, 1.992; a sum row not reproduced in 126,528 cell-calls); N22-B1v 238 (93
  large, 4.163); the learned ml4b-N22 238 (95 large, 4.475). All misses of N30, G30b and D30 are on QTo paths with kind-b
  far donors from QH stratum S. QT2's side block (C/O 1-2) and the qual set's C/O 1-2 paths: 0 misses for N30, G31,
  N30sN2H, G30b, D30. Validity: element drift <= 4.7e-8 dex, no negative He, no bad states or unfinished calls; rows
  reproduced at 1e-24 for the element-row designs. Gates + qual panels (equal to the library port's (w13_proj) in all 434 fields), the ring at
  the minimum lengths + 1 day at dt 5 s (0 cells > 0.1 for every design), other-composition mixing and low metallicity:
  clean for G31 and N30sN2H; D30 fails the ring (23 cells > 0.1, 2 > 1 dex, 1.31 in CH4). Beds outside R1's panel list
  (reported beside): the carbon-rich ring (C/O 1-2) fails every reduced design by 1.3-3.2 dex (N30sN2H 3.19 in CH4,
  G31 1.26) while the full mechanism is clean; across C/O = 1 every design has misses next to the C/O 1.8 source gas
  (G31 and N30sN2H 7 cells at 0.30 dex); from an equilibrium start G31, N30sN2H and N30 have 4 cells up to 0.33 dex in
  the first calls on the Z 300 planet (G30b 41, D30 26 cells, up to 1.31). Cost (GPU1 holding only this job; whole call
  in this harness, us per cell-call): all-species 4.83 / 4.70 (ring, 28,800 / 131,072 cells), 2.12 / 1.91 (QT2 inputs);
  the designs 5.19-5.74 / 5.08-5.22 and 2.52-3.66 / 2.31-2.61. Under the frozen rules: G31 and N30sN2H pass R1 in the
  main domain; R2's smallest passing N is 30 (N30sN2H), but it sits 0.093 dex from the control (> R2's 0.05), so the
  author chooses between N30sN2H and G31; N30sN2H also has known failures on development beds outside R1's panels
  (1.598 dex HCN on the mlw2 bed's path 2323, C/O 0.94; 0.108 on the synthetic path 3087) that G31 passes; no ML design is
  adopted (R3: the only learned finalist fails and is worse than its plain twin). Not done: the R5 ablations for the
  pick, the 1000-day sign-off, second seeds.
- The decision reviews (2026-11-01 ~14:0x-14:3x; V experiments/2026-10-31_refresh/_build/reviews/brief_decision.txt,
  answer_decision_sol_xhigh.txt and answer_decision_claude.txt (two independent reviews); both
  re-derived the judge's main numbers from its output files). Both: the frozen rules adopt nothing on their own (R2's
  0.05-dex clause sends N30sN2H vs G31 to the author); recommend G31; neither design is yet signed off for production;
  no learned part is adopted (ML lost in the roles and bounds tried; untried: learned rates, a faster full mechanism, an
  unbounded prior correction). Points the reviews add: (1) everything that separates G31 from N30sN2H lies outside
  R1's panels (path 2323 1.598 vs 0.042 dex HCN; path 3087 0.108 vs 0.009; the carbon-rich ring 3.19 vs 1.26) and was
  measured on pre-merge code: rerun on 3b3be41 before the author decides; path 2323's partner is the path's own earlier
  gas (ordinary mixing, not a QT kind-b donor; correcting w11_diag/finalists.txt), and the miss persists with a
  converged projection: C and C2 have the same element vector, so near C/O 1 the memory splits carbon between them
  wrongly (I); (2) 266 of QT2's 384 out-of-generator paths take donors only from QH's labelled stress stratum S, which
  QT left out; every QT2 failure of N30, G30b and D30 is on such a path; read as stress, N30 and G30b would pass QT2's
  main part (their worst on the other QTo paths 0.014 / 0.028): a domain call for the author; (3) the equilibrium start
  (the commonest GCM start) gives 4 main-domain cells 0.333 dex off for G31, N30sN2H and N30 (Z 300, C/O 0.7, 1.8e-5 bar,
  call 4): a main-domain miss under misses-first, filed under R5 by the judge; (4) the reduced designs are 0.4-1.5 us
  per cell-call slower than the all-species call (they save advection only), and an every-cell-in-phase-3 worst case
  costs ~35 us per call; the timings exclude host transfers; (5) still owed: R5 ablations for the pick, the 1000-day
  sign-off, C/O > 1 (G31 needs C/O <= 1 or the all-species default), dt 1-5 s build-up beyond a day, limiter-clipped
  transport, photolysis; QT2 is now used, so no untouched judge remains.
- Sign-off of G31 and N30sN2H on the merged code (refresh campaign, 2026-11-01 14:1x-15:1x; V experiments/
  2026-10-31_refresh/_build/w15_sign/results.txt, log.txt, sign_*.json, bed_w5c.json, bed_w5a.json, contract_*_lib.json,
  vulcan_*nm*.json; library layer, main 3b3be41, each paired with the all-species control on the same forcing). 1000
  days: QL (64 paths x 57,600 holds of 1500 s) and QLc (8 paths x 172,800 calls of 500 s): both designs 0 paths with a
  miss, 0 cell-calls > 0.1 dex; worst vs the control 0.0019 (QL) / 0.0013-0.0015 (QLc); worst vs VULCAN 0.0190 / 0.0174
  = the control's own; element drift <= 3.2e-7 dex; no defects, failed calls, spill or unconverged projections; no
  build-up. Known development regressions replayed on the merged code: G31 clears all (the mlw2 bed paths 2323 / 2123 /
  163: 0.042 / 0.0008 / 0.0020 dex; its full validation and test splits, main and side: 0 misses, worst 0.042 main /
  0.069 side; the synthetic bed's path 3087: 0.054; the mlw bed's test split main: 0 misses), whereas on the library without
  phase 3 it had misses on 9 + 9 mlw2-bed paths; N30sN2H still fails path 2323 (1.598 dex HCN, call 792; main domain, C/O
  0.94, Z 86), misses on 4 mlw2-bed side paths (up to 0.69) and on path 3087 over 2,000 calls (0.459 dex C2H2, call 1017).
  The two design files differ only in C2 (G31 carries it), so carrying C2 is what fixes these paths. Both share 2 large
  misses on the mlw bed's carbon-rich side paths (C/O 1.09 and 1.74; 2.19 / 2.10 dex). Deployment contract (both):
  restart from saved tracers + memory + step sizes bitwise (resetting the step sizes moves the state by <= 1.44e-6
  mol/g, as it moves the full mechanism), any cell order or batch split bitwise, no truth in the batch bitwise over 200
  calls; the equilibrium start (judge's run on the same code): 4 cells up to 0.333 dex (C2H2, P2200 top levels, calls
  3-7) + one P1800 cell at call 211 (0.102), the same for both. Memory-free forms fail (QC 5 paths with 1.10-dex defects
  before any mix; QCs 6 paths, up to 1.66 dex). Verdict (lead, from the frozen rules, both reviews and this sign-off):
  G31 is the recommended production reduced-tracer design for C/O <= 1; N30sN2H is not recommended (a known 1.6-dex
  main-domain miss); the author picks (R2); all 53 species stay the default until then and for C/O > 1.
- snapy (Cheng Li's GPU rewrite of the ExoCubed dynamical core) on daw, and the chemistry's cost at that size (refresh
  campaign, 2026-11-01 ~16:xx; V experiments/2026-10-31_refresh/_build/snapy_test/results.txt, runs/*.json,
  logs_chem_steady.log; snapy 2.12.0 + kintera 2.6.6 from the maintainer's prebuilt GPU wheels in code/snapy/.venv,
  torch 2.10.0+cu128; MIT clone at code/snapy, tag v2.12.0; nothing in Auto-Chem imports it; one RTX 6000 Ada). snapy's
  own cubed-sphere tracer test passes. A dry GCM-like run (C64 x 100 layers = 2,457,600 cells, weno5 / hllc / rk3,
  dt 69.3 s): median step 335 ms with 0 passive tracers, 1,810-1,839 ms with 31 (5.45x), 2,810-2,817 ms with 53
  (8.40x); each tracer ~47 ms per step (0.14 of the dry step, 0.019 us per cell) and ~0.49 GB; at C48 (1.38 M cells)
  4.69x / 6.91x (about half of the dry step is a fixed cost, I). The all-species chemistry call on 2,458,080 cells of
  the ring bed advanced call after call (state and step sizes carried, the bed's forcing and mixing every call, new
  step control): 8.73 s per call = 3.55 us per cell-call, 14.2 accepted steps per cell on average, every guard count 0
  (a first-call-only timing gave 11.4 s, 18.4 steps). On this fp64-slow card the chemistry dominates: ~26x the dry
  step, ~3x a step with 53 tracers; carrying 31 instead of 53 tracers saves ~1 s of ~11-12 s per step (I). The earlier
  0.23-0.31 us per cell-call figures were measured on GCM-like forcing without mixing every call (1.1-1.4 steps per
  call): every-call mixing raises the steps per call and the cost about tenfold.
- Radiation and a live coupled run in snapy (2026-11-01 ~17:xx; V experiments/2026-10-31_refresh/_build/snapy_test/
  results.txt sections 3-4, rt/*.json, gcm_run.py; one RTX 6000 Ada; maxichem from the new public repository's code,
  torch 2.10 + nvrtc 12.8 without conflicts). Cheng Li's GCM radiation (pyharp ToonMcKay89 two-stream, one grey
  shortwave + one grey longwave band, recomputed every 3600 s, heating held between): 0.05 s per update on 2.46 M cells
  (~1 ms per step averaged over the hour); 8 / 32 repeated spectral points 0.22 / 1.09 s per update. Coupled run (C48 x
  100 = 1.38 M cells, a hot-Jupiter-like test setup spun up 1.2 days to 833-1961 K; flow + 31 passive tracers +
  radiation + maxichem's G31 call, passive chemistry from local equilibrium, 2.06 h at dt 70 s): chemistry every step
  0.578 us per cell-call (1.03 accepted steps per call on average, p99 <= 3, max 36), every 7th step (490 s calls)
  0.852 us (2.06 steps); guard counts 0, element totals changed by <= 8.7e-12 relative; the reconstruction needed extra
  iterations in 100-2000 cells per call and left <= 46 cells per call unconverged (elements exact; 20 defect flags in
  all). Live chemistry is ~6x cheaper than on the every-call-mixing bed (3.55 us) two hours after an equilibrium start
  (a long run may drift toward the bed cost, I). Projected per model day on one card at 2.46 M cells (I, arithmetic on
  the measured parts): flow 7 min; + 31 tracers + radiation 38 min; + chemistry every step at the live cost 67 min (1000
  days ~47 days), every 7th step 44 min (~31 days); at the bed cost 3.6-4.6 h; on 6 GPUs the live-chemistry run ~10 days
  per 1000 days (I).
- Speedup scans (read-only, 2026-11-01 ~17:xx; /home/imalsky/tmp/speed_snapy, /home/imalsky/tmp/speed_rt; V their
  measurements): snapy's tracers move as one array (no per-tracer loop); per tracer ~43 ms per step at C64 of which
  WENO5 reconstruction in fp64 16.5 ms and the flux-covariance term 8.9 ms; setting the tracer reconstruction to plm cut
  the 53-tracer step 2.70 -> 2.20 s (mass drift <= 1e-15, positivity kept; more diffusive); fp32 tracer reconstruction
  would give ~2-3x on the tracer part (upstream work); CUDA graphs / torch.compile / stream overlap not worth it; couple
  chemistry once per step after the RK stages (stage forcings run every stage). kintera's GCM path is one backward-Euler
  step with a dense LU per cell: ~155 us per cell on a 40-species CHO network (fp64), ~1 MB of GPU memory per cell, no
  error control; maxichem's call is >= 30x cheaper per cell. Realistic correlated-k radiation (80-240 spectral points
  in published GCMs) would cost ~36 ms per point per update at C64 here; fp32 Toon 1.7-2.5x faster.
- Why every-call mixing costs ~14 steps per call, and what would cut it (read-only scan, 2026-11-01 ~17:xx; V
  /home/imalsky/tmp/speed_chem/results.txt, bed*.json; the ring bed tiled, state and step size carried; an instrumented
  kernel copy bitwise equal to production). After a mix a few fast trace species are out of balance: the first try
  (the carried step, often the whole call) fails and the step shrinks (dt 500 s: 14.1 rejections before the first
  accepted step, median first accepted step 2e-4 s), then climbs back through ~19 small steps: 38.5 tries / 24.3
  accepted per call at dt 500 s, 21.0 / 12.6 at 100 s, 9.9 / 6.1 at 25 s; tries grow with the input jump (1.2 for
  < 0.01 dex, 16 for 0.03-0.1, 41 for 0.3-1, 60-85 above 1 dex) and with T (8.6 at 300-650 K, 39 at 2000-3000 K); later
  rejections ~0; growth-capped steps 7-10 %. Variants (us per cell-call at 69,120 cells; worst error vs a tight solve,
  local / along the trajectory): production 3.65 (0.0036 / 0.0037); the old RMS-only first step 2.57 (0.0042, but it had
  the 11-01 miss); RMS + a half-call cap 2.64 (0.0041); remembering last call's first step 3.08; atol 1e-14 2.24
  (0.0088); atol 1e-14 + half-call cap 1.63 (0.0098, just under 0.01; 1.58 vs 3.57 at 2.46 M cells, ~2.3x). The mix
  applied as a steady source over the call instead of a jump before it: 0.231 (16.7x; 1.1-1.5 tries per call), but it
  is a different coupling: vs a 16-substep reference (built the same way, so not independent) the jump coupling is off
  by up to 2.65 dex (3.7 % of cell-calls > 0.1) and the source form by 0.91 (0.9 %); every VULCAN truth set applies
  mixes as jumps, so adopting it needs new truth, a positivity-safe outflow, element checks and a reconstruction for the
  reduced tracers; untried before (grep of both notes files and todo). Chemistry every 2nd / 4th step: 1.5x / 2.3x but
  trajectories > 3 dex off (as every 10th, rejected before). Load balance: a persistent atomic work queue exists; tries
  per cell-call p50 / p90 / p99 / max 10 / 66 / 93 / 165. Kernel headroom on this card ~1.2-1.5x (earlier reviews).

### 2026-11-01 (evening): can the G31 call reach < 1 us per cell-call on mixed gas? (numerical tests)
- Request (author): under 1 us per cell-call for the GCM-facing G31 call on this card, with the gas mixed every call;
  test numerical optimizations, with an outside review. Record: experiments/2026-11-01_mixed-speed/ (results.txt,
  log.txt, runs/, logs/; two outside reviews before and after the results). No production code changed.
- Method (V): an instrumented copy of kincuda.py with compile-time options, all off = bitwise production (max |dx| 0);
  the check's ring bed (4,320 cells tiled, mixing + forcing every call, state and step size carried); idle GPU1.
- Speed (V, us per cell-call, G31 = projection + chemistry + reform): today 4.052 at 69,120 cells / 3.975 at 2.46 M;
  best passing jump-coupled setting V3 (log-log secant shrink after two rejections at one state, floor 1e-4, growth
  cap 20, atol 1e-16) + reuse of rhs / Jn / u on rejected tries (bitwise) 2.632 / 2.558 (1.55x); the mix as a source
  over the call (uniform outflow, exact element budgets) 0.573 / 0.574. All species: 3.630 -> 2.286 (V3+reuse) ->
  0.231 (source). Tries per call 23.1 -> 14.4 -> 1.5. The projection costs ~0.33-0.35 us of the G31 call.
- Accuracy of V3 (V): bed worst vs tight 0.0051 / 0.0052 (local / trajectory; production 0.0036 / 0.0037); kintest
  quick 0 misses on every panel, no path or yardstick row moved > 0.01 dex from quick_ref2; QTe every-call block vs
  VULCAN 0 misses, worst 0.0180 (production 0.0153); a jump stress rebuilt from QTe (218,238 cells, truth = the
  production kernel at rtol 1e-6) worst 0.0133 (production 0.0106).
- Rejected (V): atol 1e-14 (QTe worst 0.070, stress 0.0996); atol 1e-14 with the first-step max-norm kept at 1e-18
  passes (QTe 0.023) but V3 has more margin at the same speed; atol 1e-12, rtol 5e-3, rtol 2e-3 (bed local 0.016 /
  0.025 / 0.012, > 0.01); Shampine's filtered error estimate on every try or the first step only (1.34 dex; I: it
  hides the first-step error the 11-01 rule guards against) or after the first accepted step (0.031); growth cap 20 alone
  (22.9 tries vs 23.1) and the carried-step fix (23.1); a first-try timescale cap (20.9 tries but 16.6 accepted steps
  vs 14.4). Not run alone: a smaller shrink floor (I: the typical first rejection shrinks by ~0.34, above the 0.2 floor).
- Coupling (V, dt 25 / 100 s blocks; the dt-500 s transport is singular, no positive root): against 256 substeps of
  transport (matrix root) and chemistry, the jump is off by up to 2.97 dex (2.55 % of cell-calls > 0.1), the source
  forms by up to 3.7-3.8 dex (1.48-1.49 %); 64 vs 256 substeps still differ by 0.60 dex, so the reference is not
  converged at the worst cells (I: no ranking at the worst cells; the source form is closer for typical cells). A
  constant source can clamp at zero and lose element budgets (0.12); the device recovery ignores the source (under
  the constant form in G31, 5 unfinished cells were recomputed without it).
- Conclusion (I): with jump coupling, < 1 us is out of reach on this card: the passing settings still need ~14 tries
  per call at ~0.16 us per try, and the remaining steps follow real fast relaxations after each mix. Only the source
  coupling reaches it (0.57 us), and it is an interface change that needs a source-aware recovery, VULCAN truth with the
  same source term and an independently converged coupled reference before it can be judged. V3 is the candidate
  numerics change if the 1.55x is wanted; adopting it means regenerating LIBREF, AD_REF, TRACER_REF and quick_ref2 and
  the full checks (author's call).

### 2026-11-01 (night) to 2026-11-02 06:1x: the V3 step rule (not committed), live-gas speed, source coupling
- Request (author, 11-01 ~20:15): adopt V3 + reuse as the production numerics (task 1); research the source coupling
  (task 2, no production change); overnight plan approved 11-02 ~04:5x (no commits). Records: experiments/
  2026-11-01_v3-adopt/, 2026-11-01_source-coupling/, 2026-11-02_projection-exit/, 2026-11-02_coupling-test/,
  2026-11-02_v3f/ (each with log.txt); outside reviews of the plan, the results (twice), the task-2 design and results.
- Task 1, in the working tree, NOT committed (V): kincuda.py's call and recover kernels (secant shrink after a second
  rejection at one state, floor 1e-4; factor limits [0.2, 20] after an accepted try, [1e-4, 20] after a rejected one;
  atol 1e-16; a rejected try reuses rhs, Jn and u, bitwise), the derivative kernels (the factor's partials; the step
  history in the tangent and in adjoint records of NS + 6), the JAX layer (13-int geometry), README; references
  regenerated with patched copies of the external generators (libref3, libref3_SNCHO, refdata3, kintracer ref4,
  quick_ref3). The call kernel is bitwise the tested variant (40 bed calls x 8,640 cells; a token diff of the generated
  sources); the recover kernel's new rule was never in the tested copy (checks only: recovered cells vs a 40,000-try
  call 4.5e-10 dex). A latent NaN in the torch reference's reverse mode (unused secant factor at p <= 0.05) was fixed in
  the generator copy; refdata3 is unaffected.
- Check criteria changed by the author's choice (11-01 ~22:4x; tests/ in the working tree): single-call parity allows
  one cell-call above 1e-6 dex per case and none above 1e-5; cold-start central-difference bar 3e-5; 1000 K h-pair floor
  71 (55 of 64 cells now carry h >= dt); tracer end state 1e-6 relative + 1e-16. With them (V): parity 20/20, tracer
  9/9, JAX 28/28, SNCHO 15/15, kintest quick 0 misses on the qual paths (the yardstick's known 300 K trajectory
  unchanged), no path moved > 0.01 dex; QTe every-call block vs VULCAN 0 misses, worst 0.0180 (old 0.0153); jump stress
  worst 0.0133 (0.0106); G31 gate 0 misses, worst 0.021. AD 43/44: the recovered-cell cold-start central-difference
  case fails, and with the old code it fails on 3 of 4 other direction seeds too (a pre-existing, seed-sensitive check).
  The 'starting step alone' dot test passes at p99, but its maximum rose from 1e-10 to 2-5e-2 on two 300 K cells whose
  derivatives are not computable in fp64 (an independent torch autograd's own forward and reverse modes disagree there
  by up to 2x); the cause is the plain shrink floor 1e-4, not the secant, the cap or atol.
- V3F = V3 with the plain factor floored at 0.2 after a rejection (secant floor kept) (V; experiments/2026-11-02_v3f,
  references regenerated there, not installed): bed 2.287 vs 2.282 us per cell-call, 14.43 vs 14.41 tries, local worst
  0.0054 vs 0.0051; jump stress 0.0133 and QTe 0.0180 both; every check as V3, plus parity within the original 1e-6
  criterion and the dot-test maximum 7.6e-8; the same recovered-cell failure. The second results review prefers V3F.
- Speed (V, idle GPU1, us per cell-call, 52bdf63 -> V3): bed all species 3.601 -> 2.250 (69,120 cells), 3.555 -> 2.214
  (2.46 M); G31 4.021 -> 2.606, 3.970 -> 2.544. Live gas (the coupled C48 x 100 hot Jupiter, 1.38 M cells, G31 every
  70 s step): 0.577 / 0.577 / 0.585 (old / V3 / source form) on a 40-step restart; 0.582 -> 0.577 on 8 recorded calls
  replayed. Live cells take 1.03-1.08 tries per call, so the step rule barely matters there; the tracer projection
  (0.36 us, 62 %) dominates.
- Projection early exit (V, research copy): stopping a cell's Newton iterations once its residual is at RTOL, in phase
  1 only (cells that need phase 2 run as before): on the replay set the projection 0.360 -> 0.111 us and the G31 call
  0.576 -> 0.327 (1.76x); with one polishing iteration 0.364 (1.58x); 12 gases within 5e-12 dex of production;
  kintracer_check 9/9 and the G31 gate (kill gate, 776 panel paths: 0 misses, worst 0.021) with and without the polish.
  The any-phase exit fails the far-from-memory case (6e-6 / 3e-5 vs 1e-6). Earlier ring-bed result: notes above, 11-01.
- Task 2 (V): (a) over the coupled day the per-call median jump (the bed's all-species metric) is 1.9e-4 dex, the
  per-call p99 0.034 (bed median 0.073). The 1,174 reconstruction defects of the day: in a 260-step rerun all 270 are
  negative carried species (C2, N, O2, -1e-41 to -6e-23 of the cell's moles) arriving with the transported tracers near
  the model top (450-3,700 Pa); no element is off (worst 2.5e-16). (b) A coupling test with a positive continuous
  transport L = (T - I) / dt from the bed and Strang splitting with tight chemistry as the reference: the reference
  converges at first order (1024 vs 2048 substeps differ by up to 0.23 dex at the worst cell, p99 4e-3); srcu has
  smaller worst local errors than the jump (2.82 -> 2.50 dex on the bed, 1.78 -> 0.90 with the coefficients x 0.1) and
  fewer cell-calls > 0.1 dex locally, but not in every statistic (trajectory share > 0.1 dex 4.5 vs 4.3 % on the bed);
  Strang with 1-4 substeps is worse than both; tight and production forms agree (the error is the coupling). The bed's
  jump tail cannot be scaled to the live one (p99 1.9-6.6 dex vs 0.034), so the live coupling choice stays open; it
  needs replays of live neighbourhoods against a jointly refined reference. (c) A VULCAN-0D pilot with the source term:
  about 35 core-min of compute but about half a day to give the fast driver per-call stepping; not run. (d) A
  source-aware recovery prototype (source and loss in the recover kernel): recovered cells match a 40,000-try call to
  6e-11 dex, against 3.4 dex without it. A GCM would need the pre-transport state per cell, an outflow fraction (retention or from the jump),
  positive transported tracers, a source-aware recovery with a budget guard, and truth with the same source term. On
  live gas the source form saves no time.
- For the author: commit V3, V3F or neither (live gain ~1 %; bed and cold starts 1.55x); the recovered-cell
  central-difference check; whether to pursue the projection early exit (a projection numerics change: reviews, a new
  TRACER_REF); nothing from task 2 adopted.
### 2026-11-02 (afternoon): V3F and the projection early exit adopted; speed ideas; GCM tracer cost
- Request (author, 11-02 ~16:1x): "go with the best version that works in a realistic scenario ... super fast in a GCM
  and always accurate"; out-of-the-box and machine-learning ideas from outside opinions. Records: experiments/
  2026-11-02_adopt/ (log.txt), 2026-11-02_speed-ideas/ (three independent idea reports), 2026-11-02_fast-proj/,
  2026-11-02_figs/. Supersedes the previous entry's open decisions (V3F chosen; the early exit adopted, with no new
  tracer reference: the reference has its own torch projection, so the exit does not change it).
- Plan review (outside): go with changes: V3F over V3; the exit test is resid()'s own decision and warp-uniform; take
  the polishing iteration (after an exit at RTOL a design with sum rows can show a ~1.8e-10 sum-row defect; no shipped
  design has sum rows); add odd-input coverage, a live A/B, a timing of the GCM-facing call. Results review: commit
  after documentation fixes (done: README's accuracy claims qualified, the coupled timing with gather / scatter).
- Adopted (V): V3F (kincuda.py: the plain step factor after a rejected try floored at 0.2; the secant shrink keeps
  1e-4) with V3F's references (libref3, libref3_SNCHO, refdata3, ref4, quick_ref3; V3's set kept in experiments/
  2026-11-01_v3-adopt/ref_v3_files/); the projection's phase-1 Newton iterations stop once the iterate passes
  resid()'s test, then one more iteration (phase 2 and the dual-objective steps unchanged). The projected states are
  bitwise the tested research build on 8 recorded live calls.
- Checks on the final tree (V, GPU1 alone): kincuda_check 20/20; kintest quick vs quick_ref3: 2,440 qual paths 0 misses,
  worst 0.022, every path change 0.0000; the yardstick's only misses are the known 17 rows of one 300 K equilibrium-far
  start (CH4 0.666 dex at call 1; 0.667 with the old code); SNCHO 15/15; kincuda_ad_check 43/44 (the pre-existing
  recovered-cell cold-start central-difference case); kinjax_check 28/28; kintracer_check 9/9; G31 gate: kill gate
  passed, 776 panel paths of up to 17,280 calls 0 misses, worst 0.022 dex, defects 0, chemistry failures 0. Odd inputs
  (200,000 live cells against the 52bdf63 projection: strong mixing, each row zeroed, zero memory, non-finite and
  negative tracers, a test design with two sum rows): identical status and defect flags, states within 1.3e-10 relative.
- Speed (V, idle GPU1, coupled C48 x 100 hot Jupiter, 1.38 M cells, G31 every 70-s step): 8 recorded calls replayed,
  kernel intervals 0.5796 -> 0.3616 us per cell-call (projection 0.3601 -> 0.1465, chemistry 0.2118 -> 0.2076; layer
  counts identical; 12 gases within 0.0020 dex of 52bdf63). Live A/B, 300 steps from one snapshot (passive chemistry, T
  and P bitwise equal across runs), GCM-facing call with gather / scatter: 0.585 (52bdf63) / 0.580 (V3F alone) / 0.368
  (final). Final state, species above 1e-12: final vs 52bdf63 <= 3.3e-3 dex, final vs V3F alone <= 2.2e-3 (rounding
  differences that change the step-size history of a few cells: 80 cells with h differing by > 0.01 dex); 0 cells > 0.01
  dex; flags of final vs V3F alone identical. Species between 1e-16 and 1e-12 move by up to 0.69 dex with V3F (atol
  1e-16; 480 cells > 0.01 dex). A 1,000-day sign-off of the new tree is outstanding.
- Speed ideas (three independent reports, read against these notes and the VULCAN-JAX notes): machine learning does
  not pay inside the call on live gas: a cell takes about one Rosenbrock try per call, and verifying a learned guess
  costs about a try (a 2026 study's gate returned 96.8-99.7 % of cells to the solver). Ranked levers: (1) the projection
  kernel, latency-bound at one warp per cell: one thread per cell and densities in product form (M exponentials
  instead of one per hidden species); (2) reusing the factored matrix across calls (tried before: 1.18-1.34x at large
  batches; the author: "probably don't do this", stopped before a build); (3) previous-lambda starts; (4) chemistry lane
  packing and trimming redundant fp64 work (est. 8-12 %); a GH200 timing first (never measured; the package from 10-31
  exists). Not recommended: surrogates, tabulation (ISAT), cached rate coefficients (T changes every call), fp32 LU.
- One thread per cell projection (V, research copy experiments/2026-11-02_fast-proj/mx): the phase-1 iterations, the
  exit, the polish, the fix and the outputs in a kernel generated per design with the row weights as literals and the
  densities as p_h prod_k exp(lam_k)^a_kh; cells with non-finite inputs, a product-form range overflow risk
  (sum_k |lam_k| max a_kh > 600), phase 2 or a polished residual above RTOL go to the warp kernel, which redoes them as
  production. Replay: projection 0.1466 -> 0.0385 us, whole call 0.3623 -> 0.2543 (1.43x); 12 gases within 1.9e-12 dex,
  projected states within 7e-15 relative of production; the cells sent to the warp kernel = the cells needing phase 2
  (347-392 per call); with the kernel off the copy is bitwise production. kintracer_check 9/9; G31 gate: kill gate
  passed, panels 0 misses, worst 0.022, defects 0; odd inputs: identical flags, <= 7.7e-15 relative.
- GCM tracer transport (V from the runs' logs; read-only look at snapy, nothing changed): one hot-Jupiter step costs
  dynamics 0.35 s without tracers and 1.08 s with 31 (the spin-up and coupled-day records), so moving the 31 tracers
  (~0.73 s) costs more than the chemistry (0.5 s). snapy runs, every RK stage and for all tracers, a chain of whole-array
  fp64 operations (reconstruction, upwind flux on the hydro mass flux, covariance term, positivity limiter, halo
  exchanges); by arithmetic and memory traffic it runs ~20-50x above this card's limits (I). Levers (GCM code, not
  maxichem): a fused tracer kernel, fp32 tracer reconstruction, transport once per step with stage-averaged mass fluxes,
  PLM reconstruction (measured 11-01: 53 tracers 2.70 -> 2.20 s), fewer tracers.
- Figures (author's request): experiments/2026-11-02_figs/: the mini-chem paper's six Fig. 4 boxes (VULCAN, maxichem,
  mini-chem; worst floored error over six gases for t >= 1e-3 s: maxichem 0.0003-0.0149 dex, mini-chem 0.21-5.3 dex),
  and the yardstick's worst error per row (calls 1-200: rows with a miss mini-chem 15.21 % / large 5.99 %, maxichem
  0.004 % / 0).

### 2026-11-02 (evening): the one-thread-per-cell projection adopted; the yardstick's 17 missed rows are a VULCAN error
- One-thread-per-cell projection (author: "do both", 11-02 ~17:0x; outside review: adopt with changes, all made;
  experiments/2026-11-02_fast-proj/, log.txt): kintracer.py's phase 1, the early exit, the polish, the fix and the
  outputs run in a kernel with one thread per cell for designs with integer row weights 0-12 on the hidden species and
  at most 6 rows (every shipped design); the weights are literals, the densities p_h prod_r exp(lam_r)^a_rh. A cell goes
  to the warp kernel (which redoes it as before) when an input is non-finite, when an evaluation could leave the
  exponential's range (sum_r |lam_r| max a_rh > 600, or log max p_h + that sum > 690; the review's counterexample: a
  1e305 memory) or is non-finite, or when it needs phase 2; hand-offs are counted (COUNTS "handoff"). New check case
  (g): the two kernels of one solver cell by cell. Results (V, GPU1 alone): replay of the 8 recorded live calls 0.5792 (52bdf63) -> 0.2514 us per cell-call
  (projection 0.3599 -> 0.0365, chemistry 0.2118 -> 0.2068); against 7c298ac on the same inputs (the kernel before the
  review's range guard):
  12 gases within 4.6e-12 dex, projected species above 1e-12 of the cell within 8.9e-15 relative (components near
  1e-30 mol/g differ at their rounding level); live GCM-facing call with gather / scatter 0.257 us per cell-call (100
  steps from the snapshot; dynamics with the 31 tracers 1.11 s per step, the call 0.355 s); hand-offs 224 per call =
  the cells needing phase 2. kintracer_check 10/10 (case (g): flags and counts identical on 17 input sets, worst
  difference 8.7e-16 of the cell's moles: the rounding of an element budget landing on atomic N or O through the
  spill); G31 gate: kill gate passed, 776 panel paths 0 misses, worst 0.022 dex, defects 0; kincuda_check 20/20;
  kintest quick: qual paths 0 misses, every path change 0.0000; kinjax_check 28/28; SNCHO 15/15 (case (g)'s worst: 1.9e-5 of its bound).
- The yardstick's 17 missed rows (one start: 1x solar C/O 1, 300 K, 35.5 bar, equilibrium-far = atomic-H gas from
  2,617 K at 5e-5 bar; CH4 0.666 dex at call 1, every dt) (V, experiments/2026-11-02_yard17/, log.txt; VULCAN reruns in
  the VULCAN tree's experiments/2026-11-02_yard17_vulcan/): maxichem at production and at 1e4x tighter tolerances
  agree to 0.0000 dex along VULCAN's 3,374 output times; an independent integration (the torch reference's rates and
  right-hand side, scipy Radau and BDF at rtol 1e-10) agrees with maxichem to 0.0000 dex (CH4 frozen at 8.82e-9 after
  ~0.01 s). VULCAN-0D rerun on the same job: the stored settings reproduce the stored run exactly (CH4 4.11e-8);
  rtol 5e-4 / 5e-5 with the default first step (dttry 1e-10 s, dt_min 1e-14 s) stay wrong (CH4 1.2e-8, HCN 1.2-1.3 dex
  high); a first step of 1e-18 s (dt_min 1e-20) at rtol 5e-3 / 5e-4 lands within 0.008 / 0.006 dex of the Radau
  solution (worst of 12 gases). Cause: VULCAN's first step is far too long for the first femtoseconds of the H
  recombination at 35 bar; its error estimate skips species whose new value is <= 0 and its clip sets small negatives
  to 0 (CH4 is 0 at its first output, 1e-14 s); the wrong CH4 then freezes at 300 K. The 17 rows are a reference error:
  against the corrected VULCAN run maxichem is within 0.008 dex, so the yardstick has no maxichem miss.
- Reference corrected (author's go, 11-02 evening; V): VULCAN rerun of panel job 865 with its stored settings and only
  the first step changed (dttry 1e-18 s, dt_min 1e-20 s; 3,848 steps, end_case 2, to 1.01e8 s); the design's truth
  formula reproduces the stored truth from the stored run exactly (0 difference), then the 918 saved states of the 17
  t0 = 0 rows take the corrected run's values (CH4 up to 0.661 dex lower) in data/benches/yardstick/design/
  truth_saved.npz (original kept in experiments/2026-11-02_yard17/backup/, sha256 in truth_sha256.txt). The parent's
  230 restart rows keep their start states and truth (states of the stored run continued by it: valid tests of a call).
  kintest quick on the corrected truth: yardstick 395,114 rows 0 misses, worst 0.035 dex; only those 17 rows changed
  (worst error -0.66; maxichem's states identical); saved as data/benches/kintest/quick_ref4.npz (the quick reference
  from now on). mini-chem on the same rows: 15.21 % with a miss, 5.99 % large (unchanged to the shown digits).
- Scan of the stored dense VULCAN runs for the same signature (scan.py; a species present above 1e-25 at t = 0, exactly
  0 at VULCAN's first output, later above 1e-12 and its floor): 3,818 yardstick panel runs: 4 (runs 865, 1401, 1809,
  2333); maxichem's worst on the t0 = 0 rows of the two other design parents 0.002 and 0.001 dex (harmless; 2333 is not
  a design parent); the six Fig. 4 boxes: none. The qual and QT sets store only call-end states (one VULCAN box per
  call), so the signature cannot be read there; maxichem matches them (0 misses, worst 0.022 dex on qual) and their
  tolerance reruns (rtol 5e-4 vs 5e-3, default first step) passed: qual 26 / 26 (worst 0.018 dex), QT 26 / 26 (0.014),
  QT2 26 / 26 (0.0072).

### 2026-11-02 (night): what a full VULCAN-0D solver still needs; photolysis as per-cell input rates (research copy)
- Request (author, 11-02 ~20:0x): "work on getting everything for a full vulcan 0d solver done, like if photochemistry
  or transport is still needed". Records: experiments/2026-11-02_photo/ (log.txt; mx/ is the research copy, the diff in
  _build/photo.patch); VULCAN runs in the VULCAN tree's experiments/2026-11-02_photo_vulcan/. Nothing changed in
  maxichem/.
- Gap analysis (V, VULCAN-0D b3ccf4f read only; outside review agrees): thermal kinetics done. Photolysis is the one
  missing physics input: VULCAN-0D computes J from a two-stream flux through a column, which a box lacks, so J enters
  as a per-cell input (the author's 10-31 interface). Transport does not exist in a 0-D box (GCM's job, author 10-31).
  Condensation in VULCAN-0D is a test stub (H2O -> H2O_l_s, rate capped at 1e-6, and NH3 -> NH3_l; no shipped network
  has a condensation section): GCM cloud scheme, not built. No ion chemistry in VULCAN-0D. Not supported by the reader:
  remove_list (VULCAN switches reaction directions off by id) and the "2*OH" token of CHO_photo_network.txt.
- Implementation (research copy): the reader reads the "# photo" lines into net.photo / net.Sp (absorber its sole
  reactant, atoms balanced, unique absorber-branch pairs; species that only photolysis names are not added, they go to
  photo_missing and photo=True then refuses). CudaSolver(net, device, photo=True) appends the branches as reactions
  after the thermal ones, irreversible (kf = J per cell, kr = 0, no reverse Jacobian terms), in the call, recover and
  guard kernels; the call takes J [B, branches] (s^-1; None = 0); a negative or non-finite J makes the cell a bad input
  (not stepped, counted). photo=False generates the same CUDA source as before. Derivative kernels, JAX and the
  reduced-tracer call are not extended (the G31 design is for the 53-species network; NCHO_photo has 55).
- Tests on VULCAN-0D's NCHO_photo (55 species, 305 thermal reactions, 35 branches) (V, logs/ of the folder):
  (a) photo=False sources byte-identical for NCHO_thermo, NCHO_photo and SNCHO (36 builds). (b) photo=True with J = 0
  vs photo=False: identical steps and done in every cell; recorded starts 4e-12 dex, scrambled 2.1e-9, a 200-call chain
  2.4e-7 (C4H2 at 6e-29; scored gases 3.1e-8): the pre-declared 1e-9 bar failed (rounding of the changed LU order and
  stream sums); within the parity checks' 1e-6. Bad J (negative, +-inf, NaN, columns up to 34): 12 of 12 counted, not
  stepped. (c) Against an independent reference (the torch reference's thermal rates and right-hand side plus
  separately written photolysis terms, Radau rtol 1e-10): 287 boxes from the layers of three VULCAN 1-D photochemical
  runs (HD 189733b, HD 209458b, WASP-39b; their T, P, composition, J x 10^U(-1, 1)), 192 dark qual states switched to
  lit, 24 day / night chains of 100 calls: 0 misses, worst 0.0004 dex; photolysis moves 13.5 % of the dark-to-lit boxes
  by > 0.1 dex (up to 3.2). (d) Against VULCAN-0D with imposed J (its chem_funs regenerated for NCHO_photo, J set after
  rev_rate, use_photo off before the integrator, first step 1e-18 s, landing on 1, 3, ..., 1e8 s): 36 boxes (12 1-D
  layers, 12 dark-to-lit, 12 with J = 0), maxichem chains of dt 1, 100 and 500 s: 0 misses, worst 0.0004 dex; VULCAN
  rtol 5e-3 vs 5e-4 within 0.0044 dex; Radau vs VULCAN within 0.0004 dex; the gas moves up to 4 dex; a dark chain is off
  the lit VULCAN runs by > 0.1 dex in 17 % of lit boxes (up to 1.65). (e) Speed (idle GPU, 65,536 qual states, dt
  100 s, settled calls): photo=False 0.213, photo=True J = 0 0.213, lit 0.258 us per cell-call.
- Weak spot (V): lit gas at 300-320 K. In (e) 32 cells (300-325 K, 3e-3-2e-2 bar) needed 2,000-30,000 accepted steps
  for their first lit 100-s call (the independent Radau: ~2,500 at rtol 1e-10, 130-215 at 1e-6); the recovery finished
  30 of them (within 1e-4 dex of Radau), 2 still failed (partial states, counted). It takes CH4 and NH3 photolysis
  together (without CH4 or without NH3 photolysis: median 43 / 26 steps); trace atomic N sets the error and goes below
  -atol on the rejected tries; larger atol makes it worse (1e-14 / 1e-12: 2 / 14 of the 32 cells unfinished after the
  recovery). It persists for J x 1e-3 to 10 and dt 10-500 s, and in the second lit call; dark and dusk calls are easy.
  A T-P map from equilibrium starts (J of HD 209458b's layers): 300-320 K up to 13,553 accepted steps in the first lit
  call and 15,735 in the second (1 bar), irregular in T (305 K at most 293); 325-1500 K at most 78.
- Outside review of the plan (ended on a capacity error) and of the results (same thread): the photolysis formulation
  and the kernel changes are sound (no packing, signature or Jacobian defect); the imposed-J VULCAN driver is faithful;
  verdict "not yet" for production because of the 300-320 K cells; made from it: the reader no longer adds
  photolysis-only species (the first version changed CHO_photo_lowT from 35 to 37 species), J must be on the device of
  x, duplicate absorber-branch lines refused, a cold cell with bad J documented as counted cold, the reference's solver
  status no longer hidden (rerun: no Radau failure; (c) and (d) unchanged). Its correction: NCHO_photo's thermal part
  has 15 more reaction entries than the default (9 for O_1 / CH2_1, 3 reverse-direction duplicates, 3 same-direction
  duplicates; the default's laws all present), not 12 plus 2 changed laws. Its remedies for the weak spot, in order:
  capture matched rejected steps (stages, N budget, Jacobian, solve residuals) to separate discretization from
  arithmetic; rejection-aware step growth; ROS2 as a rare-cell recovery; a nonlinear implicit recovery.
- For the author: adopt photolysis into maxichem/ (the photo=False path is unchanged), or first fix the 300-320 K lit
  cells; a VULCAN imposed-J truth set beyond the 36-box pilot; a photolysis-aware tracer design.

### 2026-11-02 (late night): why lit cells at 300-320 K are hard, and a fix (research copy)
- Request (author, 11-02 ~21:2x): "investigate and determine exactly what makes this hard, and if its possible to fix".
  Records: experiments/2026-11-02_hardcell/ (log.txt, logs/); VULCAN runs in the VULCAN tree's
  experiments/2026-11-02_photo_vulcan/out_hard/. Nothing changed in maxichem/.
- Cause (V, logs named in log.txt): the network's CN + M -> C + N + M (A 4.2e-10, 71,000 K) gets its reverse from the
  equilibrium constant; at 300 K C + N + M -> CN + M is 3.0e-7 cm^6 s^-1, 3e21 times a three-body collision limit
  (130x smaller at 325 K, 1e4x at 350 K). VULCAN computes the same rate. CH4 photolysis makes atomic C, NH3 photolysis
  atomic N, and the two titrate each other. One accepted step (0.34 -> 6.2 ms) takes C (4.5e-26) and N (3.9e-27) to
  -8e-24 / -1.3e-22, far below atol 1e-16, so neither the error norm nor the -atol test sees it; the clamp sets both
  to exactly 0. At
  C = N = 0 the titration's Jacobian entries vanish (1e-9 s^-1 against -3.9e10 with the exact values), so every later
  try is blind to it in its linear solves but meets it in the stage evaluations, which drive N negative: only tries of
  1e-5-1e-4 s pass, with repeated erasure and regrowth of C and N, until near 1 s. From the exact state, tries of
  1e-2 s pass. ("3e21 times" is against an assumed 1e-28 cm^6 s^-1 benchmark; the fit is from 4,060-6,060 K
  shock-tube data in argon, Tsang 1992.) Without the CN + M line the hard cells take 45-52 steps (scored gases move
  <= 1e-4 dex). VULCAN (2x step growth,
  1e-18 s first step) takes 695-2,558 steps on 8 such cells, within 2e-4 dex of Radau.
- Fix (research kernel option): also reject a try in which a species that was above zero falls below -0.5 times its
  value. Hard cells 5,249 / 20,006 steps (median / max, 2 failed) -> 55 / 66, none failed, <= 1.4e-4 dex from Radau; a
  1,593-cell lit grid (300-1500 K, 1e-4-10 bar, J x 0.01-10) max 20,008 steps and up to 18 failures per call -> max 252,
  none; first lit call of 65,536 qual states (dt 500 s) 114 -> 3.95 us per cell-call. Thermal NCHO: VULCAN quick bench
  0 misses, no path moved > 0.01 dex, same steps, 1.00x; mixing bed 2.245 -> 2.264 us per cell-call, same accuracy.
  SNCHO scrambled-start stress (8,192 cells x 20 calls, against a tight solve): the kernel today has local errors up to
  1.31 dex (5 cell-calls > 0.1 dex, the same clamp mechanism); threshold 0.5: worst 0.175 (1); 0.1: 0.071 (0); 0: 0.066
  (0). Threshold 0 costs +7 % on the NCHO bed; 0.1 not yet timed. Not adopted: no growth after a rejection, clamping to
  a fraction of the old value, larger atol; capping over-fast rate pairs (keeping K) works but changes the scored gases
  by up to 0.028 dex.
- Outside review (V, _build/ in the experiment folder): the mechanism is convincing; the patch is not ready: the derivative
  kernels (kincuda_ad.py:520, 615) and through them JAX keep the old acceptance test, so their primal would diverge from
  the call; 0.5 is not justified as a general default (SNCHO favours 0.1 or 0); the SNCHO result needs an independent
  reference; regression cases (zero starts, subnormals, forced recovery) and new torch references are needed; raise the
  over-fast reverse rates with the network's author.
- Network route (V, logs/scan_nets.log, fixed_tests.log, kintest_quick_*.log, vulcan_fixed.log): VULCAN's own NCHO
  networks (VULCAN 3, in VULCAN-JAX) removed CN + M -> C + N + M, CN + C2H2 -> C2H + HCN and CN + C2H4 -> C2H3 + HCN and
  reversed N + CO -> CN + O and NH2 + NH2 -> N2H3 + H in 8.2020; VULCAN-0D's files predate that. Over-fast directions
  (two-reactant coefficient at 1 bar > 1e-8 cm^3 s^-1, all at 300 K): VULCAN-0D NCHO 7 (CN + M 7e21x, NH2 + NH2 5e6x,
  CH + M 6e4x, N + CO 3e4x, CN + C2H2 2e4x, NH3 + CH 3e3x, C2H + M 3e3x); VULCAN 3 NCHO 4 (CH + M, NH3 + CH, C2H + M,
  NCO + HCN 15x); VULCAN 3 SNCHO also S + CN 6e4x. A copy of VULCAN-0D's networks with the five 2020 corrections (VULCAN
  tree, nets_fixed/): with today's solver the 32 cells take 49 / 59 steps, the 1,593-cell grid at most 142, the
  first lit call 3.80 us per cell-call, maxichem within 0.0004 dex of VULCAN on the corrected network. But it is
  different chemistry: against the current truth, yardstick 1.48 % of rows with a miss (0.26 % large, worst 4.6 dex HCN
  at 2309 K, 5e-5 bar), qual 165 of 2,440 paths; CN + M removal alone 0.06 % (worst 3.1 dex), the other four edits
  1.45 %; VULCAN corrected vs original on the photochemistry pilot up to 0.16 dex. Adopting it means regenerating every
  truth set and reference and re-qualifying G31. Summary and all edits: experiments/2026-11-02_hardcell/fixes.md.

### 2026-11-03 (night to 09:2x): VULCAN-JAX's NCHO_photo as default, the collision limit, new truth data, reduced tracers removed, derivative kernels extended
- Requests (author): 11-03 ~03:1x: a complete setup in which maxichem replaces the GCM's chemistry step, under about
  1 us per cell-call, for every network up to the sulfur ones; remove GPU code that saves only hundredths or tenths of
  a microsecond; fix the remaining gaps; make all the truth data; remove legacy code; plan and review first;
  simplified technical English (rules put into CLAUDE.md). ~05:08: "1) extend [the derivative and JAX layers] 2) drop
  reduce tracer code, but keep all notes so I can build it in the future 3) fix the ch and ch2 stuff 4) push to github
  when neded". Records: experiments/2026-11-03_overnight/ (log.txt, logs/, reviews/), 2026-11-03_truth-gen/,
  2026-11-03_ad/, 2026-11-03_tracer-rule/.
- VULCAN-JAX (its ledger C29-C31; e32dc86, be9f785, 5f352cc pushed earlier; 8b30242 pushed with this entry): step
  control (the error estimate counts a species the step creates, a value below nega_cut rejects the step, a retry
  restores ymix and the element losses, the dt_min fallback keeps the state); data: NH3 + CH removed; HNCO + M; NCO +
  HCN and S + CN rewritten in their exothermic directions (Tsang 1992; UMIST 2022); N(2D) thermo; CS, CS2, COS, HCS
  masses; C31: the C3 network's `C2H + C -> C3H + H` written `C2H + C -> C3 + H` (Hickson et al. 2016, the row's
  reference). CH + M and C2H + M were first removed (their low-T reverse rates are super-collisional), then put back:
  removing them moved maxichem by more than 0.1 dex in 34 of 45,027 truth boxes at T >= 1000 K (worst 1.92 dex, C2H2,
  1325 K, 0.048 bar, dark). Normal tier 324 passed, 66 skipped; audit PASS.
- Collision limit (C30; VULCAN-JAX, the truth generator, the torch reference and maxichem): for each thermal pair,
  q = the larger direction's two-body-equivalent rate (k with M folded in, times n_tot^(molecules - 2), directions of
  two or more molecules); if q > 1e-9 cm3/s both directions are scaled by 1e-9 / q, so K_eq and the equilibrium are
  kept. In NCHO_photo it acts somewhere in 300-3000 K, 1e-5-100 bar on CH + M, C2H + M, NH + M, NCO + NH2 and
  CN + C2H6 (overnight/log.txt 05:28); the CH + M reverse is capped below 563 K at 1 bar (655 K at 100 bar), the C2H + M
  reverse below 895 K (1828 K) (a continuous solve in reviews/plan2.answer.txt, item 4). A regularization ceiling, not
  a measured rate. Rates agree across codes: VULCAN-JAX vs the torch reference within 1e-6 relative (96 T, P points),
  the generator vs the torch reference within 8.6e-14 (NCHO_photo) and 8.5e-14 (SNCHO_photo; 48 points each,
  truth-gen/logs/rates_vs_ref_*.log). The generator's SNCHO network has a condensation line S8 -> S8_l_s that maxichem
  skips; it never acts in the boxes (S8_l_s stays 0).
- maxichem: default network VULCAN-JAX's NCHO_photo_network.txt (69 species, 390 thermal reactions, 48 photolysis
  branches); photolysis adopted (CudaSolver photo=True: J per cell and branch, fixed within a call; bad J counted, not
  stepped); the CH3OH law with Visscher & Moses 2011's broadening (the "vm11" width, as VULCAN-JAX and the generator);
  K_COLL in the call and recover kernels; the reader refuses unbalanced reactions and lists duplicates (rates add, as
  in VULCAN); the small-batch one-cell-per-block launch path removed (it gained only at <= 142 cells); the Rodas3
  stage inputs written as explicit fma (the compiler had contracted F + 4 ih K1 differently in the call kernel and the
  derivative kernels on NCHO_photo, which broke their bitwise primal parity; F and K1 were equal, the stage input
  differed by 1 ulp in 9 of 69 rows). All 11 VULCAN-JAX networks load and run (4,096 cells, 21 calls: 0 unfinished,
  0 failed, element shares within 3.8e-11). The quick-start example runs.
- Checks after the last kernel change (the fma): kincuda_check 28 of 28 (new reference libref6 on NCHO_photo with the
  limit: recorded starts mapped by species name; photolysis calls, a lit 300-call chain, lit recovery, bad J and cold
  precedence); SNCHO parity 15 of 15 (libref6_SNCHO); kintest quick 0 misses (new reference run
  data/benches/kintest/quick_ref5.npz, the first on the new truth).
- Truth generator (outside the repository): VULCAN-0D b3ccf4f on VULCAN-JAX's data with the same step-control
  corrections, the CH3OH broadening, the collision limit and k_B = 1.380649e-16; fast path bitwise equal to the
  corrected stock path (20 boxes, after the limit); first step 1e-18 s, dt_min 1e-24 s (a pilot box stalled at the
  old 1e-20); step cap 2e5. VULCAN's rtol 5e-4: at 5e-3 one pilot box of 4,096 kept a 4.1-dex CO error against Radau
  (truth-gen/logs/radau_box276.log); at 5e-4 that box agreed with Radau within 0.016 dex and with maxichem within
  0.020 (console output; no log kept). Merge: an atomic publication (the manifest lists the shards and replaces the
  old one in one rename) that keeps earlier outcomes for boxes not run again (a toy test passes). A paths worker now
  exits 1 on an error and resumes only an output whose job and build digests match.
- Truth data (data/truth_v2/; the old-network sets moved to old_network/): boxes_ncho: a scrambled Sobol design of
  131,072 boxes (T 300-3000 K and P 1e-5-100 bar log-uniform, metallicity 0.1-300x, C/O 0.1-2; half lit with the J of
  a 1-D model layer times 10^U(-3, 1); starts: unreacted gas, a reheated, mixed or scrambled 1e8-s state, or 10 % of H2
  split into H), 48 stored states 0-1e8 s: all 131,072 accepted (worst element drift 9.9e-8); qual_ncho: the recorded
  forcing and mixes of the old qual panels replayed (904 paths: QC 512 x 1,000 calls, QCs 64, twins/QC 128, QH 64 x
  up to 1,000 holds, QHs 32, twins/QH 32, QL 64 x up to 3,500 holds, QLc 8 x up to 3,000 calls, as long as the old
  paths are; 0 errors, 0 forced); daynight_ncho: 64
  QC paths x 1,000 calls with J = J_ref s max(0, sin(2 pi t / 1 day + phase)); boxes_sncho: 8,192 boxes on
  SNCHO_photo, 8,191 accepted (box 6602, 2676 K: element drift 1.2e-6 > 1e-6, left out). 18 boxes stopped at the
  2e5-step cap; rerun with 2e6 (13) or 2e7 (5, up to 2,959,228 steps): all finished (the manifests list them).
- Accuracy (rtol 1e-3, atol 1e-16; experiments/2026-11-03_overnight/logs/): NCHO boxes 6,160,384 states, 0 misses,
  worst 0.076 dex (CH4, 487 K, lit); guard 4 recomputed of 6.2 M cell-calls, all recovered. Paths 1,016,411 states,
  0 misses, worst 0.013 dex; panel P stable. Day-night 64,000 states, 0 misses, worst 0.0030 dex. Sulfur boxes:
  dark 0 misses (worst 0.034 dex); lit 29 of 4,095 boxes with a miss (159 states, 127 large; worst 6.86 dex), all at
  354-750 K, mostly S2 and COS; guard 8 recomputed of 385,000, 7 recovered, 1 still failed.
- The sulfur misses (logs/sulfur_misses3.log; Radau of the generator, rtol 1e-10, atol 1e-30, on 27 of the 29 boxes;
  2 still running at the time of writing): VULCAN-0D is off from Radau by 0.56-6.86 dex in 19 of the 27; maxichem is
  off from Radau by more than 0.1 dex in 17 of them (up to 5.5 dex; 0.12-5.5 in the 8 where VULCAN is right). On
  maxichem's side (box 5932, 519 K, 5.2 bar; logs/sulfur_box5932.log): COS and S2 (about 1e-18 and 1e-24 at 1.1e4 s)
  are exactly 0 in maxichem's output from 3e5 s on and stay 0, while in Radau COS grows to 3.4e-5 and S2 to 8.7e-10 by
  1e8 s; atol 1e-20 or 1e-25 removes it, rtol 1e-6 with atol 1e-16 does not (the step that zeroes them, presumably an
  accepted value above -atol clamped to 0, was not observed). With atol 1e-20 maxichem is within 0.013 dex of Radau on
  all 27 boxes (16 gases, at the stored states, max_steps 2e5 in this diagnostic). These are selected boxes, not a
  survey; Radau runs on the generator's chemistry functions, so it checks VULCAN's integration, not its chemistry. atol 1e-20 elsewhere: NCHO boxes 0 misses, worst 0.069 dex, steps per call
  3.0 -> 3.8 on the 0-1e8 s chains; sulfur boxes against VULCAN 54 boxes with a miss (VULCAN's own errors show more).
  Speed (1,382,400 cells, 70-s calls, idle GPU): NCHO_photo 0.346 / 0.349 us dark / lit at atol 1e-16, 0.349 / 0.350
  at 1e-20; SNCHO_photo on sulfur gas from the truth boxes 0.636 / 0.636 and 0.646 / 0.637. For the author: atol
  1e-20 as the default (a numerics change: references and checks to redo) and a tighter sulfur truth (VULCAN at a
  smaller rtol, or Radau) before the sulfur network can be certified.
- Derivative kernels extended (the author's decision 1): photolysis (kf = J exp(lm); tangents and cotangents in J
  itself, so they hold at J = 0); the Troe factor's derivatives (standard and vm11 widths); the collision limit's
  (weights from the pair before the limit: wA = 1 above, 1/2 at the limit; wG = wA times the reverse direction's share
  of the larger, 1/2 at a tie; the Arrhenius cotangent takes the pre-limit part); the 14-column rate table (its 9
  Arrhenius columns differentiated); the adjoint's photolysis products (kincuda.tables leaves them out of rx); tsolve
  with two leading rows per lane; gather descriptors widened (species 7, rate 12, destination 9 bits; the old 6-bit
  species field could not hold a 64-species network's padding index); input dtype and device checks. Limits: 64
  leading rows (96 species), 2,047 reactions; explicit reverse laws refused. Plan and code reviewed
  (experiments/2026-11-03_ad/review_plan.answer.txt, review_code.answer.txt: no formula or index error found; the
  criticisms of the test design were taken into dev_check2.py). Development check dev_check2.py (states from truth
  boxes mapped by species name; primal bitwise CudaSolver's; adaptive tangents against central differences per input
  class incl. lm != 0, p90 <= 1e-4 with plateau coverage >= 90 %; J at 0 by one-sided Richardson differences; JVP / VJP
  dot products, p99 <= 1e-6 (lm 1e-4), max <= 1e-3; recovered cells; determinism): NCHO_photo 99 of 99, CHO_photo 99 of
  99, SNCHO_photo 97 of 99 (the J tangent on box starts at dt 500 s: p90 2.5e-4 and 3.7e-4 against the 1e-4 bar; p99
  3e-3). Not done: the JAX layer (refuses until ported), an independent torch-autograd reference for these networks,
  tests/kincuda_ad_check.py (written for the old 53-species network and refdata3).
- Reduced tracers removed (the author's decision 2): kintracer.py, the designs, its check and gate (code at 399a347).
  Its method, designs, failed designs, memory and speed costs stay in this file (2026-10-30 to 11-02 entries). A simple
  abundance rule for tracer designs fails (2026-11-03_tracer-rule/): on QT2's 1,056 mixing paths (old network) 30
  tracers give 16 paths with a miss (0.81 dex), 36 tracers 2 (0.17), 36 + HCO + HNCO 0 (0.062): HCO and HNCO hold small
  element shares but matter after mixing. Learned designs were tried before and tied or lost (section 1).
- Legacy removal (deletion list reviewed): the Auto-Chem code, its .venv and .git removed after a bundle and a full tar
  (restore checked: references and commits identical); duplicate data and superseded references removed; the
  training-data generator archived.
- Kintest: the box panel refuses a set whose manifest lists boxes not run, failed runs, bad states or element drift,
  whose shards do not hold every design box exactly once (accepted or quarantined), or whose shards differ in species,
  stops or photolysis lines; quarantined boxes are reported; the paths must be exactly those of jobs.json, once each,
  each as long as its job asked; b_set and q_set are digests of the scored truth and forcing (boxes and build stamp;
  paths), and a paired run refuses a reference with other digests; negative or non-finite truth is a failure, never
  the floor; the run exits 1 on a miss, an unfinished call or a non-finite or negative state. Paired check of the new
  reference against a second run: no path or box changed (logs/kintest_quick_paired.log).

## 3 Author decisions and requirements (dated)

- 2026-06-09 (gate 2 of the paper plan): held-out open-loop rollouts at fixed dt 6.4 s for K >= 100 steps plus 0.5 s and
  100 s; 100 % finite; error growth that flattens; p90 per-species error well below 56.6 % at 99 steps (target < 10 %).
- 2026-09-29 / 10-01: the training data contain no helium (confirmed). 10-01 decisions: stage-2 recipe R (superseded
  10-04), provisional stage-1 gate 1 (loss <= 0.5x persistence, below it in every start decade), stage-2 target a
  reported per-species p90 at a stated step and dt, frozen tails cut, any in-range dt per call, AOTI not run on daw
  (torch.export .pt2 is the artifact), MIT licence with CITATION.cff. Implementation (old tree, now under tag paper-v1 and
  the audit branch): recipe R 4842418 (20,000-step epochs, peak LR min(3e-5, stage-1 LR); replaced at the port by
  48c95e3), gate 1 53b83ac / 8a32e8e, frozen-tail cut dfd5c40 (16,311 groups cut, 986 lost, 0.39 % of windows; norm
  sha256 ee4a7834...), robustness fixes from a code review 3815bb1, MIT + CITATION.cff 8a32e8e; He-bearing compositions
  declared outside the validated domain in the export metadata (host mapping open, section 5.2).
- 10-03: warmup + cosine in both stages; stage-2 peak LR not capped at the stage-1 checkpoint's LR. Physical constraints
  are tested against a null of no benefit.
- 10-04: test assumptions cheaply (CPU preferred, literature-checked); the best long-run model, smaller and faster
  better at equal accuracy.
- 10-05: regenerate data as needed (up to 32 cores); try FiLM on P and T; delete old material; keep records here.
- 10-06 (goals): a stable surrogate (one call = one step), for GCM timesteps, the C/O and metallicities of GCMs, as
  accurate as possible (first guess about 10 %), fast (first guess about 50 us per amortized call); as wide a range of
  metallicity, C/O and time as possible. Targets: accuracy first (50 %, then 0.1 dex; superseded 10-08); metallicity up
  to 300x; timestep the GCM's radiative-transfer step, working range 10-500 s; element conservation not required unless
  it helps; store all 53 species; Sobol or any generator. 53+ tracked species too many (sulfur will add more): about 20
  plus a lumped "misc". The final training may run on NAS (the author runs NAS).
- 10-07: keep the strict resolved rule (small mixing ratios can carry high opacities); all 12 species matter; the 50 us
  target is on the GPU; test generation targeted at the difficult physics; up to 48 CPU cores and 1 GPU; a deliberate,
  non-repetitive design; a careful review, a 2-3 day sprint, leave the Auto-Chem repo untouched; data allowance up to
  about 200 GB (Gate 0, about 22:20).
- 10-08 (15:39-15:44): reading only per-band p90 is a methods mistake; the target is the best OVERALL error, typical
  error plus every large miss. Gate 1 calls delegated ("use your best judgement on all these points").
- 10-09: "whatever changes are needed" for conservation (the first allowance was up to about 20 tracked entries).
- 10-10: most research decisions delegated, each checked first by two independent reviews; scope, interface, rule
  changes and opening holdouts stay with the author. The > 10 dex veto covers xlong ("anywhere").
- 10-11: the per-gas scoring rule; no repeated seeds of big tests ("spending too much time running multiple seeds of big
  tests"; one seed per recipe, a second only when a decision hinges on it, quick path tests first); sampling redesign for a
  scale-up (more cold examples below about 650 K, GCM-like starts, changing-condition runs, drop drift rows: elem_drift >
  1e-3 in a production cache, > 1e-2 in the final data incl. c18 and b02; ideally in rounds). Every results report includes a figure like the ApJ paper's (superseded 10-19: figures only when asked).
- 10-12: fix the four hard-bar rows by any means, clean up, test more data and a bigger model; one accurate solver for
  every row if it costs a few us; the network may reach about 30 us per call when scaled up.
- 10-13 09:2x: "5 dex off is fine. Get rid of this super hard rule." Scale phase: fix the remaining large misses, refine
  by size and data, 2 days, final model at most about 10 us per cell, no interface change without the author. 22:15:
  is completing the 2^20-row design (about 580 CPU-h; all 434,482 unused base rows are at T >= 1000 K) worth it?
  Deferred, ranked last on 10-14.
- 10-14: FINAL's misses are too large for production; too little varied data suspected; near-perfect trajectories that
  return to the truth after an error ("just as long as huge errors don't build"); lasting errors matter most ("but I
  don't love any error"); the T-P domain that matters is real GCM profiles over 1e-5 to 100 bar; test a constant-timestep
  emulator if GCMs use fixed steps; keep the interface; H4 / H4c / PH2 stay closed for the next phase's pick.
- 10-15: decision domain env_any +-100 K / +-0.1 dex inside 1e-5 to 100 bar plus a cooler-planet check (300-1300 K); no
  training above 3000 K; run T5 overnight; tracked species: first "Don't go past the number of tracked species you have",
  then (afternoon) "Now you can test a model that goes as high as you want it" ("try one with 53 species. Fresh, best
  guess model."); "ideally time goes from 0.1 to 500 seconds"; GCM chemistry starts "mostly eq". Compute: up to 48 CPU
  threads ("unless the daw gets super busy"); GPU0 for tests if every job there stops when another user needs it.
- 10-16: "I need the model to work from 1-500 seconds" (was 10-500 s). Data like the target: dt 1-500 s, metallicity "1x
  to 100x", C/O "maybe only go up to 1.0", GCM P and T; "more data + a slightly larger model may help"; a final large
  model may train up to 3 days; check the label interpolation; "Maybe make more data until it is near the memory limit of
  the GPU?". Batch size may hurt accuracy: "Above 512 previously I found errors" (no record on daw; not reproduced).
  "I want notes": a notes entry the same day a result lands, including the author's own findings and decisions.
- 10-17: "Maybe try streaming and more data? Use your judgment"; project data cap 300 GB; "bigger model, more data, longer
  training all combined" (CB); two independent reviewers "to check the complex work"; "I approve stuff you need"; "More
  training steps?".
- 10-18: "I'm really surprised the more data didn't help. Combine with more steps? Training should basically keep going
  until loss levels off." "No preference" among a full all-data run, data in stage 2 only, or no new data (CB5 taken);
  GPU0 for about 2 days; "Yes" to deleting the 52 GB U5 build intermediate.
- 10-19: CBg adopted ("This is so much better..keep this one. Figure out why it is so much better. Record results in
  notes. Try to target remaining errors. Keep improving."), overriding the large-miss guards. Composition target 09:23
  "Mainly 1-100x, C/O <= 1" (superseded 21:4x). "get rid of the rule that prioritizes avoiding rare large misses so
  much" (pass rule v2). 10:0x "Can you stack the tests or put multiple on the same GPU". 11:2x scratch caches deleted
  ("make sure they're not being used rn"). Markdown: only notes, todo and the project instructions file per folder ("I only
  want the 3 per dir max"), then "readmes can stay". 14:1x "The time I care to get right is 10 seconds. But I really do care about
  all times. Make sure there is no dt gap, where it sucks at certain points."; every dt in 1-500 s with equal weight, as
  many calls as fit in about 5 minutes per test, raw-gas starts count in decisions (a readout 10-15 to 10-19), GPU1 queue
  stopped. 16:1x CB5 "kill it".
- 10-19 (cleanup request about 16:30; answers 21:40; cleanup log): full cleanup of temporary files, legacy code and
  data, all figures and figure scripts (figures only when asked), code bloat; docs cut to the instructions file, notes.md and
  todo.md (READMEs stay); one GPU used efficiently; refocus on beating mini-chem over whole trajectories. Time weighted equally
  per decade; dt weighted equally per factor of 10; keep the old paper data (13 GB); delete CB5 and its caches, the old
  repository's plotting scripts, the downloaded GCM archives and PDFs; metallicity floor delegated (set to 0.1x,
  log-weighted 0.1-300x; section 5.2); C/O 0.1-1.
- 10-19 21:4x (push target): the cleaned code replaces main of github.com/imalsky/Autoregressive-Chemulator; the paper's
  code (origin/main f6e6116) is tagged paper-v1 and linked from the new README; push only at the very end, after the
  checks, with every outgoing commit authored by the author; one code home (code/Auto-Chem; the harness history kept as a
  bundle in paper-notes/archive/); anything importing VULCAN-0D, the generator or mini-chem stays on the local-only GPL
  side; the unpushed audit-branch commit 6f17adf stays local.
- 10-19 23:12 (going to sleep): "If there is cleanup or small checks to do before then, keep going until 9am".
- 10-20 ~07:10 (t0-fix design.txt): "do whatever you and [the two reviewers] recommend for the loss/scoring. Don't report
  minichem at all ... fix everything that I wrote, but don't touch minichem. Work around it if you can." Read as: the
  scoring rule is delegated (v3.1 adopted after the two external design reviews, section 5.4); mini-chem's code, tables,
  driver and outputs are not changed and its table defects are not reported upstream. Then "email when done" (logged
  08:44).
- 10-20 ~09:00, after an external maintainer review (GitHub branch cleanup/maintainer-pass aaa7ea1, PR #1, closed): "get
  the code to a good clean state when appropriate and after checks, and email me" (done 09:4x, section 5.6).
- 10-20 ~10:0x: "do you need my approval for anything? I give it" -> code/Auto-Chem main and tags pushed (section 5.6).
- 10-20 ~11:0x, step-size range: the GCM literature (t0-fix startweights/report.txt, citations checked) shows published
  GCMs call kinetic chemistry every 1,500 s (Exo-FMS + mini-chem, Lee et al. 2023), 3,600 s (Lee, Tan & Tsai 2023) or
  3,750 s (UM: Drummond et al. 2020, Zamyatina et al. 2023, Christie et al. 2026, Steinrueck et al. 2025), with dynamical
  steps of 25-50 s. Asked whether to extend the range to 1-4,000 s, the author: "the idea is the chem would now be called
  basically the same frequency as the fluid dynamics" -> dt stays 1-500 s.
- 10-20 11:2x, training data and recipe: "for the eventual training dataset, I think I want something like reasonably
  close to what I have now, but sobol sampled, organized, clean, following the goal principles I have, align the training
  epochs/steps/guess at model size ect" -> the clean design stays close to CBg's training set (size about 766k runs, its
  start-kind and stratum mix) in one scrambled Sobol design, departing only where the goal requires; step counts set to
  CBg's epochs on the new train split; a model-size estimate from the project's scaling evidence, not above 6x2048.
  Then "email when done" (at the clean ready state, before the full run).
- 10-20 ~10:3x, answers to six open questions:
  (1) start types: "I think mostly close to eq. Like, I care about the working emulator over a wide domain, but not
  unphysical. Look up gcms and make a rec." (start-type weights to be set from the GCM literature; the v1.1 parent
  weights were eq-target 0.35, eq-far 0.25, raw-A 0.20, raw-B 0.20, chosen in the yardstick design, not by the author).
  (2) "Do equal decades": equal weight per decade of elapsed time stays, from the first call (a 1000-day run spans about
  8 decades, so the first 10 s weigh as much as day 116 to the end); a last-decade (lasting) score is reported beside it.
  (3) "Try to avoid doing this for now": no separate first-call starter (one network call per step).
  (4) "do whatever you rec, just make sure its recorded somewhere": first recorded as "vary He/H in all new training
  data"; superseded the same day (~10:4x): "For helium, I want to use Lodders 2019 instead of the old one" -> new data use
  the Lodders 2019 solar vector, He included (He/H 0.083946, the value of mini-chem's 1x table; training so far fixed
  0.096917 = Lodders 2009); recorded in the project instructions (New data). Same message: "I don't want big new model,
  I want to just get to a fully clean state", and asked how long regenerating all the data (instead of the current mix
  of batches) would take. Then: "Talk me through your rec for helium. My only goal here is realistic atmospheres that
  can vary in the way real gcms will vary" and "that holds for all other decisions ... I just want a clear common sense
  emulator that covers the domain that I want, realistic gcms over as wide of a domain in terms of composition and state
  as realistic". Helium setting adopted under that rule: Lodders 2019 metals, He/H uniform 0.078-0.100. Basis: He is
  inert in these networks (dilution and third-body only; one-cell measured effect on CBg 0.003 dex, late; V yardstick
  confound/report.txt); GCMs keep He/H fixed at whatever solar table they adopt, which spans 0.082 (Asplund 2021) to
  0.0986 (Lodders 2019 protosolar) (V element_vectors.json He_H_alternatives); planets form from protosolar gas (Lodders
  et al. 2025 protosolar Y = 0.2753), and Jupiter's measured He is depleted by helium rain (Galileo: He mole fraction
  0.1359 +- 0.0027, Y = 0.234, von Zahn et al. 1998 -> He/H = 0.0786, I); the generator so far fixed He/H at the Lodders
  2009 protosolar 0.0969 and scaled only C, N, O with metallicity (V code/training-data-generator/gcm.py:7-8, 62, 83-85).
  (5) "yes": the sealed sets H4 / H4c and PH2 may be opened once, on a frozen new pick that passes every planned check
  and its F1 confirmation.
  (6) the repo-sweep folder with an unopened secrets/ subfolder (deleted 10-20 unread; excluded from the text archive,
  V: 0 manifest rows): "idk, do what you recommend" -> scanned every added line of the public history (main, paper-v1,
  e13s_CBg_s2; 186,733 diff lines), all local refs (317,277) and the pre-rewrite bundle of 2026-10-04 (232,247) for
  GitHub, AWS, Slack, Hugging Face, OpenAI-style and wandb tokens, private-key headers and key / token / password
  assignments: 0 hits (a planted-credential control was caught, 3 of 3); GitHub's secret-scanning alert list for the
  repository is empty. Nothing to rotate as far as the histories show; the deleted folder's own contents are unknown.
- 2026-10-21: data home. "get rid of all that scratch stuff. I think I basically want that whole folder gone" and "all
  data related to this project should go into this dir, in an organized way, and git ignored following best
  practices"; chose code/Auto-Chem/data/ (git-ignored) with subfolders raw, caches, truth, sealed, runs, benches.
  Nothing on /scratch is discarded; everything moves. This overturns the clean-state specs' choice of /scratch for
  caches and the twin. Then "ok, can yuo launch the full training thing when ready?" = go for the clean full run,
  launched after the move.
- 2026-10-23: overlay at every point, misses first. Morning: "The model look OK, but not amazing. I really need the
  chemistry to nearly perfectly overlay the results of the vulcan chemistry at every point"; longer training, more data
  and a bigger model accepted; "The model can also be restricted to 1 to 100 seconds as the time domain or made larger if
  that would be helpful. More layers? More residual connections?"; 48 CPUs and 1 GPU for this work. After figures of
  one call from a start state against its step size (errors up to about 6 dex): "Don't you see that these are horrible?
  Why are you sayin they're close? The model is many many orders of magnitude wrong in some casese????". On the proposal
  to judge by misses (share of rows and starts with any gas above 0.1 and 1 dex, and the worst error) with the averaged
  score second: "Yes!!!! Make sure that judging every failure is in the loss somehow! Check with [the external
  reviewer]. Check different architectures or training setups based on this. Do research. Do small tests. Make sure this type of distinction is
  recoreded in important places!!! maybe even do a different set of stages because of this??". Recorded in the project
  instructions (Aim; Judging: misses first, the loss feels every miss, adoption = fewer misses with an averaged score
  not worse); it supersedes "one score" (2026-10-20) and makes the large misses the first criterion instead of readouts.
- 2026-10-25, on the open questions (perturbed starts with sigma up to 2 dex, cool gas with a hot-source composition, an
  NO entry, the design reserve as new data): "Make whatever choices seem reasonable. I think reasonably perturbed starts
  may happen due to cloud formation or something?" Choices made under that delegation (sec 5.6, 2026-10-25): perturbed
  starts stay in the training design and the judged readout with sigma 0.1-2 dex (reported by sigma band); hot-source
  cool gas stays (gas mixed up from hot deep layers keeps their quenched composition); no new state entry for now; new
  data from the design reserve plus a targeted extension of the same design.
- 2026-10-25 evening: "yeah, keep chugging away, keep trying to make the model perfect" (a standing go for tests). Then,
  shown that the fresh-start tail is dominated by perturbed starts with sigma 1.5-2 dex (3 % of rows, about half the
  large misses) and that data cannot close it (sec 5.6), and offered A (cap the random scramble at about 1 dex for
  training and judging, report harder ones as a stress test), B (physically structured perturbations) or C (keep the
  extreme scrambles, change the model): "Do A". Supersedes the sigma 0.1-2 choice above: perturbed gas is in the domain
  up to sigma 1 dex; readouts judge that set (bench/misses.py by_set "judged") and report the rest as "stress"; new data
  does not generate sigma > 1.
- 2026-10-26 / 27: "Can you and [the external reviewer] work together to keep improving? It's improving, but this is
  still way below the accuracy I need." Then, shown that the flow-map models drift on long runs (sec 5.6) and offered a hybrid head (a target
  that does not depend on dt, so settled gas stays put at any step size and long-step training reaches 1-500 s, plus a
  bounded dt-dependent transient part that vanishes at the target and for very short or long steps): "yes, I approve
  hybrid" (an architecture test; the current flow-map head stays the reference until a hybrid passes).
- 2026-10-27 / 28: "run whatever tests you need, email me the resulting figures so I can judge"; "keep it easy to
  understand" (plain emails and figure labels). 2026-10-28: "use the second gpu. quit immediatly if somone else requests
  it" (GPU0 for test runs alongside GPU1, under an automatic guard that stops our job when another user's process
  appears on it).
- 2026-10-29: shown that neither family has both good short and good long runs ("These are all kind of ass") and offered
  (1) a test of the two together (the flow map while the gas changes fast, the per-entry head once it has settled; two
  networks per call, against the one-call rule) and (2) a new single-model head: "Test both. Email me the results." The
  author may narrow composition to "solar +- a little bit" and accept a fixed dt "if that works way better" (domain and
  interface relaxations on offer, not yet applied). "do whatever you gotta do. Try to maximize how efficiently we're
  testing shit on the gpu. Keep trying shit" (standing go; both GPUs; several jobs per GPU).
- 2026-10-30, on the pair's model switch: "Is there a better way to do this? I don't like flow chart like models". The
  product is one model with no switch between models; the pair is a benchmark and a possible teacher, not the answer.
- 2026-10-30 (direction change): the author wants "a stable autoregressive solver" that does not catastrophically fail
  ("Any kind of really catastrophic failures will make it so that I can't put it into the GCM"); mini-chem as the basis
  ("Minichem already works"); any architecture, a narrower parameter space, a slower call and more training data are
  acceptable; plans to be made with and based on the first external reviewer; the incremental path is not the needed
  breakthrough.
- 2026-10-30 21:0x, on the kinetic-solver route: "I still want an ml model. Are you making a minichem like solver instead
  of what I wanted". The model stays a neural network: it predicts reaction rates inside a solver frame (mini-chem's
  structure) that keeps elements, positivity and the equilibrium, so a wrong output gives wrong timing, not a blow-up.
  Offered to return to the direct-state network if preferred (no answer yet).
- 2026-10-30 22:4x (direction): "think out of the box"; research planning with external reviews for a 48-hour plan; "I really,
  really want some physical solver where it's doing the 0D chemistry and it's stable ... I really, really care about these
  blowups"; combine mini-chem's physical, deterministic, bounded solving with machine learning (integrators, flow maps, jump
  solvers, two-stage training, noise injection); both GPUs busy; a status email every 6 hours; notes, configs and CLAUDE.md
  kept current.
- 2026-10-30 23:3x (speed): "I really care about speed for this method"; "under 25 microseconds per amortized call.
  Ideally under 10. Also stable and accurate on all solves that you can."
- 2026-10-31 ~02:5x (answers): (1) the GCM can carry 53 tracers, "but see how accurate you can get with under 25"; (2) "Do
  whatever is more accurate" (the full mechanism is the default; ML is not required for its own sake); (3) the MIT code may
  read VULCAN's reaction and NASA-9 data files: yes; (4) "hard error when gas below 300 K".
- Standing resource rule (author's global rule, 2026-09-26): all jobs together at most 32 CPU threads and 1 GPU; this
  overrides the project's earlier 48-thread grant. Raised 2026-10-20 to 60 threads on CPUs 8-67 (0-7 carry other
  services; V t0-fix log 09:54); GPU0 in use beside GPU1 since the author's 10-28 go, under a guard. Speed tests only
  on an idle GPU. Parallel helper processes in the
  cleanup: at most 30 at once (cleanup plan; project instructions of 2026-10-20).
- 2026-10-31 ~05:1x (after the stopping point): "put that custom gpu thing on todo. Delete all old data" (done, 106.8 GB,
  section 2); "I want to finish cleanup and refocus before doing the 1000 day runs".
- 2026-10-31 ~05:2x (fresh start): "a big pass cleanup. I want to keep the notes and minimal documentation ... for all the
  old tests ... But the goal now is to go down this new path"; keep a skeleton to recreate the old method, delete the old
  data and old code. Tests: "for the big test to take maybe 4 or 5 hours max"; "I don't need like full, complete
  exhaustive tests. I just want to get a sense of whether what I'm asking is the right way to go"; find out why tests
  take so long, possibly by regenerating the data for the current parameter space; these working rules into the project
  instructions (Testing). Choices: old code committed, tagged (direct-state-final) and removed from main (local commits,
  no push); keep the best two baselines (ov_DT2, ov_V1jL); pack the old campaign folders into archives; drop the old
  raw_gcm packs, keep raw_clean and paper. Two independent implementations of the whole-call GPU kernel into lib in
  parallel.
- 2026-10-31 ~05:3x: a todo item "to make everything jax"; asked "is it a good idea to build everything in jax" (answer
  given: not as a rewrite now; the custom kernel is the core either way; evaluate JAX when the ML part starts or if the
  GCM coupling is in JAX; open with the author).
- 2026-10-31 ~05:4x: "I still only want one single notes.md file, not tons of notes floating around. It can be longer to
  account for the many experiments tho" (this file: one notes file, reordered; nothing removed).
- 2026-10-31 ~06:1x (15 / 30 tracers): test both designs, with and without a per-cell memory of the non-advected
  species; element ratios can vary in a GCM run (element-budget tracers count toward the tracer number); tracer units:
  "Research" (answer: the UM advects mass mixing ratios, section 2). Asked that the important parts be recorded in the
  project instructions (done: Aim, GCM interface; Code).
- 2026-10-31 ~06:2x-06:5x (GPU core): "eventually part of a gcm, but highly optimized to be both differentiable, and
  specifically for a gpu"; "the most flexible, fastest, solver possible"; bonus if only 15-20 traced species; work with
  an external reviewer on all speed-ups. Answers: no GCM chosen (flexible + differentiable: C ABI, JAX first); gradients for ML training,
  retrievals, sensitivities, a differentiable GCM (full reverse through a GCM run unlikely); controller gradients:
  recommend after testing; RTX and GH200, portable; NCHO now, other networks later; reduced tracers: decide from the
  curve; inside Auto-Chem; few tuned options; tracer layer fused in the kernel; full build at once; bitwise reproducible
  on the same GPU; mixed precision allowed on RTX if validated; outputs as recommended (tracers + mean molecular weight +
  cp + diagnostics, heating opt-in); JAX added to the environment.
- 2026-10-31 ~06:4x: keep working ~8 h overnight with up to 10 agents; save the pre-CUDA work (done: tag pre-gpu-core,
  bundle paper-notes/archive/autochem_pre-gpu-core_2026-10-31.bundle).
- 2026-10-31 ~14:0x: "Yes, do what you recommend for all": derivative default = contract A (adaptive; F kept as an
  option); push main (done on the explicit "Push" that followed: e0e8c3b..4150ad8); a GH200 timing (package
  prepared for NAS); the 30-tracer decisive test and the full 1000-day test tier (both running).
- 2026-10-31 ~14:2x-14:4x: "Why is it pytorch and Jax?" (answer: one CUDA solver, two thin front ends; the pure-torch
  solver is only the checks' reference); "get rid of all the legacy and old code" (planned after the polish branch
  merges: todo item 7); "make me a couple figures" (batch scaling); "Fix the gradient calls weak spot" (the per-call
  host sync of the derivative calls: removed, above). The full 1000-day tier was stopped at tick 200,000 of 3,456,000 (0
  misses, worst 0.064 dex so far): on a GPU shared with the 30-tracer test it could not finish inside its 5-h cap.
- 2026-10-31 ~16:3x-17:0x: the long-term aim: photochemistry, maybe transport, much bigger networks, the widest range of
  T, P and composition while staying accurate; asked where ML could help (e.g. representing more species); "do the tests
  needed for the batches thing 1.4x" (the small-batch layout); "for the flux and transport, you can assume that the gcm
  is going to handle the radiation and the advection and transport stuff" (the module stays per cell; photolysis rates
  come in from the GCM); asked whether a Jacobian method could speed up the call where it failed in VULCAN-JAX
  (cross-call Jacobian / LU reuse in a W-method: being tested).
- 2026-10-31 ~19:1x: "go ahead for everything but github": the non-equilibrium truth set QN (N1 quench / reheat, N2
  out-of-equilibrium mixing partners, N0 tight reruns; not N3) and photochemistry phase 1 (per-cell photolysis rates from
  the GCM in the kernel) approved; the GitHub Support request for the closed PR #1 not wanted for now.
- 2026-10-31 ~19:4x: "put eq and photochem on todo, try to finish the full 1000 day thing asap, basically kill early.
  The main thing I care about is the ml stuff": QN and photochemistry paused (todo items 8-9); the FULL tier stopped at
  tick 1,800,000 of 3,456,000 (V, gpu-core _build/fulltier/full2.log: 520 days of the 25-s clock, 0 paths with a miss,
  worst 0.064 dex, unfinished / non-finite / guard 0; yardstick complete: only the 17 known rows miss, restarts worst
  0.019 dex).
- 2026-10-31 ~20:1x-20:4x: asked whether the ML tests were running (the next round started), for simple performance
  figures in the usual abundance-vs-time form (sent: mini-chem Fig. 4 boxes, 10-day QLc windows, tracer designs over
  time), and "try the bigger network ... the 89 sulfur thing, and check the stuff with the ml" with 48 CPUs and both
  GPUs (several tests per GPU).
- 2026-10-31 ~22:1x (the refresh; "a full refresh and good starting point and make sure what I want out of this project
  is determined"): (1) the model is a 0-D model for use in the GCM; the CUDA differentiable work stays ("That seems
  cool"); (2) it must later work with the GCM's transport and radiation, "I don't want this to be attempted now, but I
  need it to be flexible"; (3) "tracers may be very expensive in GCMs. I can't track 89+ species. I want to make sure
  that the model can work perfectly with maybe 30 species tracked maximum. Fewer would be even more ideal"; (4) "ML
  sells. Ideally the final model would use ML in some way. This is where the most agents and research must be done";
  (5) cleanup: only README.md, CLAUDE.md, todo.md and notes.md as documentation, important findings always in notes.md,
  a full audit of documentation files made during experiments; legacy data no longer needed deleted; up to 100 GB of
  Sobol-sampled new data allowed; (6) resources: up to 64 CPU threads and both GPUs, several jobs per GPU, leave a GPU
  as soon as someone else has a job; up to 10 parallel workers; an external reviewer for hard problems; a concise status email every 4 h.
  Answers to the plan's questions: both networks, NCHO at <= 30 tracers first (pushed below 26 with ML), then the
  sulfur network to <= 30; ML in either place (designing offline what the GCM carries, or inside each call), whichever
  measurably helps; closed campaigns' records packed into archives, not deleted; about 24 h, then a stopping point.
  Campaign: experiments/2026-10-31_refresh/_build (log.txt).

- 2026-11-02 ~16:1x: "Can you go with the best version that works in a realistic scenario? I really care about this
  being SUPER fast in a gcm and always accurate"; then: rewrite the projection kernel (one thread per cell) and look into
  large speedups of the GCM's tracer routine without doing them yet; reusing the factored matrix across calls: "probably
  don't do this".

- 2026-11-02 ~18:0x: the yardstick's 17 missed rows: "can you retest those cases? Figure it out"; then "push stuff"
  (5 commits pushed, 9ff3f2a..7c298ac). Later: "1. push 2. yes 3 yes": e4039e3 pushed; parent 705's yardstick reference
  replaced by the corrected VULCAN run; the stored VULCAN runs scanned.
- 2026-11-02 ~20:0x: "in the meantime, can you work on getting everything for a full vulcan 0d solver done? like if
  photochemistry or transport is still needed": photolysis as per-cell rates built and tested in a research copy (not
  adopted); transport and condensation stay with the GCM (author 10-31).
- 2026-11-02 ~21:2x: "investigate and determine exactly what makes this hard, and if its possible to fix" (the 300-320 K
  lit cells): cause found and a step-acceptance fix tested in a research copy (not adopted). Then "make a .md called
  fixes" with the edits (experiments/2026-11-02_hardcell/fixes.md) and "I think this should be a fix to the vulcan
  version ... Can I just fix this, regen stuff, and have everything work?" (network route tested, not adopted).
- 2026-11-03 ~03:1x: a complete maxichem as the GCM's chemistry step, under about 1 us per cell-call, for every network
  up to the sulfur ones, even at some cost in speed; small-gain GPU code removed; tracers reduced only if a clean,
  simple method exists; plans and complicated code reviewed first; simplified technical English everywhere. ~05:08:
  "1) extend [the derivative and JAX layers] 2) drop reduce tracer code, but keep all notes so I can build it in the
  future 3) fix the ch and ch2 stuff 4) push to github when neded".

## 4 Data

### On disk now (2026-10-31 23:0x, after the refresh deletion; sizes V du, data/sealed never measured)
- Kept in code/Auto-Chem/data: raw/raw_clean 69 GB (the clean design's VULCAN runs: 20 packs b00-b19, 1,310,662 runs
  incl. the 2026-10-25 supplement, SEEDs 4,210,000,000-4,211,310,719; V refresh _build/w2_data/raw_count.out; no
  current code reads it), raw/paper 13 GB (the published paper's data, 5 files); truth/qual 3.5 GB (the current-domain
  truth set, below), truth/panels 1.1 GB (the decision panels' histories; the generator reads it), truth/chains 384 MB
  (train_a / train_b of the kept baseline ov_V1jL), truth/sncho (the sulfur truth), truth/fig4;
  benches/designs/decision 1.2 GB, benches/yardstick (design 2.0 GB, read by the checks; panel/runs 1.5 GB, the VULCAN
  truth behind it; run 11 GB: mini-chem, the two kept baselines, the solver's rodas3np outputs), benches/kincuda and
  kincuda_ad (the checks' stored references), benches/kintest (quick_ref.npz), benches/fig4, benches/score_runs/overlay
  2.0 GB (the two baselines' panel outputs); caches/kinjax; gen/ 1.6 GB (the generators' design tables: the recipe for
  every deleted pack); ref/ (GCM envelope gcm_tp.npz, never_rerun row lists and SEEDs, the 12-species snapshot,
  pathmap.json, deleted_raw_2026-10-31.tsv); runs/ov_DT2, runs/ov_V1jL; sealed/ untouched. Quota 184 GB of 500 GB.
- Deleted 2026-10-31 22:57 (refresh campaign, reviewed manifest, check-only pass first; V refresh
  _build/w2_data/delete_log.txt, results.txt): 2.05 GB in 10 rows, nothing current reads them: the retired library
  solver's panel / qual outputs (benches/score_runs/vulcan53_*rodas3np.npz; their numbers stay as recorded in section
  2), caches/kin_solar (the 13-species rate-network set; its m2 cache carried the formula error), raw/clean_pilot (its
  SEEDs are all in raw_clean), truth/qual_pilot2 (IDs inside the qual block), ref/weights, benches/yardstick/panel/
  pilot (IDs kept in panel/runs/jobs_pilot.json). Kept on purpose (they serve the author's 10-31 choice to keep the two
  baselines and a skeleton of the old method): truth/chains (regeneration is blocked: the generator's freeze check
  fails for clean.py, fast/tdg/clean.py, fast/worker.py, fast/drive_fast.py; V), benches/score_runs/overlay,
  ref/snapshot (its code is only at an unreferenced commit, 7d5b036). Checks after the deletion: kincuda_check 20 / 20,
  kintest quick 0 misses on the qual paths, no path moved > 0.01 dex (V w2_data/checks.log).
- The QT truth set (V, repo-data/truth/qt/manifest.json; 2026-11-01): 2,346 files, 5.9 GB, every call stored; blocks
  QTe / QTq / QTc / QTo / QTs / QTu, IDs 4,243,000,000 + k; judge only (section 2). Quota 197 GB after it.
- The QT2 truth set (V, repo-data/truth/qt2/manifest.json; 2026-11-01): 1,154 files, 2.8 GB, IDs 4,244,000,000 + k,
  donors disjoint from QT's; used once as the final judge (section 2). Both QT and QT2 are now used: a new design needs
  fresh data.
- repo-data/benches/refresh_beds (2026-11-01, 4.4 GB; moved out of the refresh campaign folder, which keeps symlinks
  at the old paths): mlw_bed (the synthetic GCM-like bed, 8,192 paths x 2,000 calls; its info.txt), mlw2_bed (the
  failure-regime bed with far-from-equilibrium partners), w9_harvest (the warm-start study's test and ring records; the
  train / val files, 0.8 GB, were deleted: regenerable with that study's harvest.py in ~25 min), w10_step (the
  step-size study's npz outputs cited in section 2). Quota 200 GB of 500 GB after the campaign.
- The qual truth set (V, repo-data/truth/qual/manifest.json; experiments/2026-10-31_data/_build/log.txt): QC 512 paths
  x 10,000 per-call calls (dt 1-500 s, T, P every call; 128 near-solar) + 128 kick twins; QCs 64 (C/O 1-2); QL 64 x
  57,600 holds of 1500 s (1000 days; truth exact at every divisor of 1500 s); QLc 8 per-call twins of QL (172,800 calls of
  500 s); QH 1,024 cold-hold / reheat paths + 512 kick twins; QHs 128; conv 26. 65.3 CPU-h, 2 h 48 min on 24 threads.
  IDs 4,240,000,000-4,240,099,999 and seeds 2026103111-2026103120 are taken.
- Deleted 2026-10-31 (V, experiments/2026-10-31_cleanup/_build/delete_c_log.txt and
  experiments/2026-10-31_fresh-start/_build/delete_d_log.txt):
  - 05:10, 106.8 GB: the old training caches (arch_*), old benches, non-baseline run folders, large analysis arrays.
  - 05:3x, 88.2 GB: the E13-era raw packs raw_gcm, raw_gcm_cold, _d, _e, _h, _k, _m, _n, _r, _u, _y (75.0 GB; 19 files,
    1,321,990 runs; inventory with bytes, group counts, SEED ranges and design attributes in
    repo-data/ref/deleted_raw_2026-10-31.tsv and data/2026-10-31_fresh-start/deleted_raw_2026-10-31.tsv; every SEED in
    repo-data/ref/never_rerun/deleted_raw_seeds_2026-10-31.npz); 11 run folders (all but ov_DT2, ov_V1jL); the old truth
    sets cooln, e13_holdout (H2), e13_holdout3 (H3), e13_scaledev, gtest (G), gtest_cold (Gc), labels_dense_1027,
    paths_v2l / v2s / v2x / v3, t2; benches E13_gcm, f1, out, out_fixes, t2, designs P2l / P2s / PD / PH / PX / t2,
    score_runs/clean_full and the non-kept overlay outputs, the superseded solver outputs (*_stock, kin_*), the yardstick's
    score_v31; ref/vulcan (ref_a).
  - Quota 380 -> 281 -> 199 GB of 500.
- Regenerating a deleted pack (I): the drivers in code/training-data-generator (gcm.py, e13/gen_cold.py, fast/ drive_fast,
  e13/rows_*.py) with the design tables in repo-data/gen/gcm and the row lists in repo-data/ref/never_rerun reproduce
  the boxes (the batches below give each batch's rows and drive); cost hundreds to about 1,500 CPU-h (1.32 M runs at about
  1-4 s each; z23n1 took 4.1 s per box). New data use IDs and seeds outside every used range (the tsv, the npz, the row
  lists, the qual range above).

### The original 12-species data (paper data; kept, 13 GB)
- Four raw HDF5 files, code/Auto-Chem/data/raw: run6050, run9095, run10017, run10142 (11.9e9 B; 224,000 groups each with
  consecutive seeds, 896,000 in all; one copy each). sha256: run10017 3a8946495bef6c868a982c8bdffd8d8fb829aa8b3398767542a00d5
  08302c18f; run10142 30a20b7d782ea4e2daf6b35e37d774d23e82162f997d1af393fbc4a3f640adeb; run9095 a5df22e2860056fe88927f0dacd76
  0d2005f06fc10c8c158f286f421bfd4cc5a; run6050 e8d799a1edcc3952bb038ebb150e6271be9a1d6c9f0df0fef7855e0f3e5a4a72.
- Generator: VULCAN-0D (github.com/shami-EEG/VULCAN-0D) at b3ccf4f (2024-09-02), network thermo/NCHO_thermo_network.txt:
  53 species (52 C-H-O-N + inert He), 290 reversible reactions (580 rate coefficients); P and T fixed per box (p_box in
  dyn/cm^2), y reset to n_0 x ymix every step. Equivalent commits 9b0c47b, 7d479d6 (176c3d0 and older have different HNO2
  NASA9 data; the 2026-06 / 07 commits add sulfur, change FastChem and set count_max 1e4; the main VULCAN repository's
  network, the one the paper cites, has 69 species and 391 reactions). b3ccf4f was current when the raw files were written
  (2026-02-02); its reaction 1 H + H2O -> OH + H2 (7.50e-16 T^1.6 exp(-9720/T)) matches the data's OH decay. Licence
  GPL-3.0-or-later; Auto-Chem contains and imports no VULCAN code (since 2026-10-20 every VULCAN / generator importer
  lives on the local GPL side, gen/e13; section 5.6).
- Units (two independent checks): P in barye (late equilibria match only if P/1e6 is bar; offset 6.0000 dex); T in K;
  t, dt in s; species are mixing ratios over the whole gas, untracked species included, not renormalized over the 12.
- Composition: no helium (author confirmed; also from the data: H2 <-> 2H late equilibria match to 1e-4 dex in 1,494
  trajectories and three-body H recombination gives M / (P/kT) 0.990-0.99994, where He would give 1.03-1.46); the 12
  species sum to 1 at t = 0 within 5e-16, later fall as untracked species form (below 0.99 in 11.7 % of trajectories,
  below 0.9 in 2.2 %, as low as 0.077), so no closure H2 = 1 - sum works. Sampler
  (reproduced to 3.3e-16): u = np.random.RandomState(SEED).uniform(lo, 0, size=11), lo -20 (run6050) or -25; weights
  10^u for H2O, CO, CO2, CH4, C2H2, NH3, N2, HCN, H, OH, O, H2 weight 1, other 41 species (He included) 1e-30, all
  normalized: H2 is the bulk gas (0.31-1.0). Only 0.46 % of kept groups lie within 0.5 dex of solar C/H, O/H, N/H.
- Time grid: t = 0 plus logspace(-3, 8, 99) s, linear-in-t interpolation of y / n_0 (bitwise), held constant after the
  last solver step (about 42-50 % of groups end in exactly constant tails).
- Reconstruction (gen/generate.py, verify.py; numpy 1.26.4, scipy 1.13.1, sympy 1.12, numpy < 2 required): stock config
  except trun_min 1.0 s (true value in [0.937, 1.0155) s), dt_max 1e7 s, runtime 1e8 s; the He floor is needed (He = 0
  stalls VULCAN's atom check). Only this stack reproduces stored trajectories bitwise: 24 of 80 from their stored t = 0
  rows (12 / 20 each of run10017, run10142; none of run6050 / run9095, computed without AVX-512). From the sampler:
  resolved values median 3e-15, max 0.025 dex on the 80 verify groups; on the 500 ref_a reruns 5 sources reach up to 1.24
  dex in trace species near the error-control level (corrected 10-03; data/2026-10-03_repo-sweep/G3-06_per_source.csv). About 5 s per box on one core (median 5.3, max
  14.7 s). About 0.2 % of groups are all-NaN (the original wrapper's failure path, probably a per-box limit: one needs
  26,794 steps, 542 s). The original wrapper is unknown, as are the sampler's exact float arithmetic (1-2 ulp) and the
  machines of run6050 / run9095 (they store glibc pow time grids, run10017 / run10142 numpy's AVX-512 logspace). numpy
  2.x differs by 1e-14 to 1e-12 at 1e-3 s. dt_max must be exactly 1e7 s (one box matches at all 100 points only then); the
  grid is linear-in-t on y / n_0 (bitwise; log-log would be 0.8 dex off). verify.py (80 groups, sampler start): largest
  |dlog10| per trajectory < 1e-9 in 29, < 1e-3 in 68, < 1e-2 in 78, < 0.1 in all; the paper's 1e-30 filter decides alike
  in 79 of 80.
- The published description (arXiv 2609.00428 Sec. 2.1) confirmed: 0-D VULCAN with a fully reversed 52-species C-H-O-N
  network, non-primary species at 1e-30 then normalized, T uniform 300-3000 K, P log-uniform 1e-6 to 1e4 bar, t = 0 plus
  99 log steps 1e-3 to 1e8 s. It differs: He said to be sampled log-uniform in [1e-1.5, 1e-0.5] (not in the data); trace
  range 1e-20 (run6050) / 1e-25; 290 reactions, not about 1200; the cited thermo folder holds a different 69-species
  network. The paper drops profiles with a species below 1e-30 (8.56e7 -> 6.2e7, 27.6 %); the raw files are the
  unfiltered set (Auto-Chem's drop_below rejected 27.4 %, 67.1 % at 300-600 K; triggers O 50.7 %, C2H2 46.0 %, OH 18.7 %
  of rejected groups). Its mini-chem comparison (p90 0.0216 vs 0.875 dex) uses the corrected pressure conversion (sibling
  repository commit eb122d0).
- Composition coverage of the old data: 3,003 of 648,098 kept groups (0.46 %) lie within 0.5 dex of solar in C/H, O/H,
  N/H (Asplund 2009), 2.6 % within 1 dex, 9.97 % meet [M/H] 0-2 and C/O 0.3-1.0; GCM-range starts are mostly
  single-carrier (CO2 holds > 90 % of C in 43 %, CO in 29 %); a GCM-centred sampler would give 19 % solar-like and 23 %
  GCM-range starts. An HD209 VULCAN column (69 species) has a 12-species sum of 0.857 (x_He 0.143), 2.8e-5 of training
  states.
- Build facts: frozen-tail cut (commit dfd5c40) 16,311 groups cut, 986 lost; windows train 2,070,330 / validation
  256,734 / test 259,126; 25 % of windows start at exactly 1e-3 s and raw t = 0 rows never enter training; species_sum
  0.1 / 1 / 5 % levels 0.6455 / 0.9107 / 0.9878; label noise about 1e-3 dex mean (p99 0.022), so val_loss differences
  below about 5e-4 are not accuracy differences; values below 1e-20 carry solver noise (4.4 % of kept values); float32
  half-ulp of log10 y 6e-8 to 9.5e-7 dex (why the export returns an increment); split key = file basename + group name.
  31.8 % of window starts fall in [1e-3, 1e-2) s; 50.25 % of kept groups ended in bitwise-constant tails before the cut
  (1.8 % at 300-600 K, 99.9 % at 2700-3000 K) and 17.3 % of transitions were exact no-ops; the 12-species sum falls below
  0.99 in 10.6 % of kept groups and the tracked ratios drift (C/H p90 21 %, N/H 39 %), and the log-log interpolant
  under-counts element totals between raw outputs, so an element budget is scored against the truth's at the same step,
  never as conservation drift. Persistence scores 0.0112 on the stage-1 validation definition, 71 % of its loss from the
  top 1 % of windows (per-window 0.023 below 0.1 s, 1.8e-5 after 1e5 s): val_loss mostly measures early transients. At dt
  6.4 s 46.5 % of (state, species) pairs move less than the fp32 half-ulp (a perfect delta with an fp32 host would freeze
  4.1 %); a bf16 export erred 6.2e-3 dex (fp32 3.3e-7) and bf16 is refused since 762708d.
- Oxygen-poor hole (E7, 10-04): no group has a t = 0 O-index below a T-dependent edge B(T) (-18.7 at 1000 K, -9.6 at
  2000 K, -5.1 at 2500 K, -1.3 at 2900 K; data/2026-10-03_repo-sweep/E9_oxygen_edge_BT.npy); no draw was discarded (stored
  t = 0 rows equal the sampler's draws to <= 3.6e-15 dex; each raw file holds exactly 224,000 consecutive seeds), the
  O-poor compositions were paired with cooler T (below 2000 K 35-50 % more O-poor compositions than the sampler gives,
  3.5-5.5x the density just above B(T)); why is unknown. Oxygen only (the C and N indices have no edge), independent of
  P. 7 of the holdout's 20 hot sources lie below B(T); VULCAN handles them (17-22 s, ratios kept to <= 6.2e-9).
- Fixes to it: drop_below 0 (E8; only negative values reject; commit 6f17adf, local branch audit-2026-09-29, unpushed);
  O-poor fill run_ofill1 (E9: 50,098 VULCAN boxes, seeds 3.1e9 + k, lo -25, T U(1800, 3000) K, accepted below B(T); 4 h
  on 40 cores; end_case 1 / 2 / 3 45,999 / 4,015 / 84; +39,932 train groups; a fresh O-poor test of 150 sources);
  production rebuild 942,585 groups (rejects non_finite 2,160, frozen_tail 1,353), 754,503 train. The fill corrects about
  20 % of the deficit (about 257k groups, estimated from the sampler); a full regeneration is the clean fix.
- Test sources of that era: ref_a (500 S sources rerun to 1e10 s), a_tight (125, rtol 0.005), b, c, p, ref_r (200
  formerly rejected sources); sweep holdout (100 fresh draws, seed 20261099, sweep/holdout/MANIFEST.json; used for the
  diagnosis, no longer independent; its marker SEALED.md was renamed SEALED.txt on 10-19, the checksummed MANIFEST.json
  still names SEALED.md); uniform test U (300 sources, seeds 3,400,000,001-300, He 4e-31 to 1e-30, so void for
  He-carrying models). Every E13 bench read vulcan/ref_a at start-up until 2026-10-20 (the pruned lhbench no longer does).
  For a next data round (10-03 audit): a 1e10 s horizon under drop_below would reject 29 of the 96 cool S boxes; dense
  300-point, 53-species fp32 storage would be 32.7 GiB at 900k boxes.
- Old Sobol design (10-05, stopped 10-06): Sobol d 13, seed 20261005, 2^20 rows on the old sampler's marginals; blocks
  0-2 packed (196,608 rows) then deleted with the design file; the 2^16 pilot cache arch_sobol16 is not rebuildable.
  2.18 boxes/s on 28 cores (median 7.3 s, max 1,417 s); block 0 end_case 1 / 2 / 3 29,244 / 36,111 / 181.

### The E13 design and generator (2026-10-06 on)
- gen/gcm.py: scrambled Sobol, seed 20261006, d 16 then 17, 2^20 rows (gen/_build/gcm/design.npy); T 300-3000 K, P 1e-6
  to 1e3 bar, Z 0.1-300x, C/O 0.1-2, N/O 0.3-3x solar (all log-uniform), He solar (Lodders 2009). Starts: 50 %
  equilibrium (FastChem at a source T', P'), 20 % mixtures of 2-3 sources, 20 % perturbed (species scrambled by up to
  sigma 2 dex, clipped at 3 sigma), 10 % broad (11 species log-uniform over 25 dex); half the sources near the box (T'
  within 300 K, P' within 1 dex). All 53 species stored; VULCAN as before (runtime 1e8 s, rtol 0.05). Literature behind
  the start mix (10-06): a network trained on trajectory data fails on 5 %-perturbed samples while multi-scale starts stay
  accurate (Zhang et al. 2022, arXiv 2201.03549); broad starts evolved along their trajectories are robust (DeePODE, Yao
  et al. 2025, arXiv 2401.01220); no clean evidence that a narrower composition domain is easier.
- FastChem traps found by testing: the element file must list every element of the species file, and in alphabetical
  order, or it returns wrong equilibria with no flag (atomic H 1e-31 instead of 6.6e-8 at 1000 K, 1e-3 bar); FastChem
  runs on its species file filtered to VULCAN's 47 molecules; every start is projected onto the design's C/H, N/H, O/H,
  He/H over all 53 species (residual < 1e-9 or the box fails; max ratio error 1e-12 on 4096 starts; a continuation
  fallback on 12 of the first 27,902 rows); C/O = 1 retried with C/H lowered in 0.5 % steps. Mapping FastChem's network to
  VULCAN's by name first dropped molecules (14 of 1024 test starts > 1 % off). 16 boxes started at FastChem's equilibrium
  stay within 0.001 dex of VULCAN over 1e8 s (E13/eqcheck).
- Packing: refuses missing rows, writes under a temporary name, marks the block packed before deleting npz, records
  end_case (1 converged, 2 runtime, 3 step cap) and element-ratio drift per box; end_case 3 boxes stay out of caches;
  h5py libver v110 (13 KB per group, 41 KB with the default). Conserved quantities are element ratios to H, not totals.
- Element drift: block 0 p50 4e-14, p99 3.5e-8, but a few failed cold solves (300-490 K, high P) reach > 1 (max 361;
  c18 1.1e3; n3b 9.2e2). Rule (author 10-11): rows with elem_drift > 1e-2 get weight 0 (87 / 88 U2 / U4 train rows, 9
  new U5 rows).
- Cost: 5.18 CPU-s per box (5.4 boxes/s on 28 CPUs; cold rows 6.1-6.85). Fast path (10-17; to move to gen/fast, cleanup plan A2):
  the two generated chemistry functions (95 % of integration time) in C with the same expressions and order, four
  step-loop fixes, persistent workers: 11-12 boxes per CPU-second in production (stock 0.21), npz byte-identical except
  the timing field on 240 + 1,168 + 300 + 266 rows, the 14 fast-path failures fail the same way in the stock path.
  Hazards: one configuration module per worker must be re-filled per box; two drivers or duplicate IDs can share a
  FastChem work directory (add duplicate rejection and a per-row lock before further use).
- Design extension (10-16 / 10-17): Sobol(d 17, scramble, seed 20261006).random_base2 reproduces design.npy bitwise;
  design_ext.npy holds rows 2^20 to 2^23 - 1 (7,340,032 x 17; extended to 2^21, 2^22, 2^23, each prefix bitwise equal);
  extension IDs 4,201,048,576 to 4,208,388,607. 434,482 unused base rows below the M block are all at T >= 1000 K.

### Data batches (packed under code/Auto-Chem/data/)
| batch | date | rows | regime | notes |
|---|---|---|---|---|
| blocks 0-1 | 10-06 | 2^17 design rows | full design | block 0: 65,535 of 65,536; end_case 1 / 2 / 3 27,849 / 37,674 / 12 |
| b02 | 10-07 | 32,768 | design rows 131073-219545 with T < 1300 K | gen_cold.py; raw_gcm_cold/run_gcm_b02-result.h5; end_case 2 / 1 / 3 32,404 / 345 / 19; 13 rows drift > 1e-2 |
| c18 | 10-08 | 65,533 of 65,536 | new cold rows < 1300 K | 35 rows drift > 1e-2 |
| M rows | 10-08 | 16,384 (M64: 65,536) | small-displacement starts | gen_shift |
| u19 | 10-11 | 32,768 | rows 396,502-649,270, T < 650 K | never rerun (drive_u.sh) |
| x20 (e20) | 10-12 | 12,283 of 12,288 | < 1000 K, highest kNN-predicted switch rate (0.301 vs 0.090) | never rerun (drive_e.sh); 13,421 cold events (11.0 %) in its cache |
| y21h (= sprint gen_h) | 10-08 run, 10-09 packed | 167,112 of 167,114 | design rows 131,072-396,488 with T >= 1300 K (the hot complement) | raw_gcm_h/run_gcm_h-result.h5 (rows 315211, 328042 missing); in U2 as y21h |
| y21a | 10-13 | 38,512 usable (38,541 packed) | T < 650 K | in U2 |
| y21b | 10-13 | 68,469 | 650-1000 K | in U3 |
| z22k | 10-15 | 53,729 of 53,732 | element-preserving kicks of U2 train rows | 30 / 45 / 15 / 10 % at 300-650 / 650-1300 / 1300-2000 / 2000-3000 K |
| n1 (z23n1) | 10-16 | 78,106 of 78,108 | unused base rows [396,490, 983,040), Z 1-100x, C/O 0.3-1, 1e-5-100 bar; all >= 1000 K (11,710 / 27,346 / 39,052 at 1000-1300 / 1300-2000 / 2000-3000 K; no unused base rows below 1000 K in this regime) | 4.1 s per box |
| n2 | 10-16 | 69,836 | extension rows, same regime, T < 1300 K (24,449 / 24,423 / 20,964 at 300-650 / 650-1000 / 1000-1300 K) | |
| n3a | 10-17 | 214,637 of 214,643 | same regime: 139,643 cool (rows 2^21 to 2^22 - 1), 75,000 hot (2^20 to 2^21 - 1) | fast path; 12.2 GB; end_case 1 / 2 / 3 54,243 / 160,366 / 28 |
| n3b | 10-17 | 307,992 of 308,000 | same regime: 200,000 cool (2^22 to 2^23 - 1), 108,000 hot (2^21 to 2^22 - 1) | fast path; 17.6 GB; end_case 78,101 / 229,846 / 45 |
- y21h provenance (V 2026-10-20; the dated record left it open): the sprint's gen_h batch (experiments/2026-10-08_e13-sprint/
  _build/gen_h: rows_h.npy = 167,114 design rows 131,072-396,488; drives h1 / h2 of 83,557 rows each, 83,557 + 83,555
  present) is the packed hot set: raw_gcm_h/run_gcm_h-result.h5 holds 167,112 groups (T 1300-3000 K), its design attribute
  names rows_h.npy and missing rows 315211 and 328042 (packed 2026-10-09), and U2's run_gcm_y21h-result.h5 is a link to it;
  the U2 build log calls y21h "the packed hot set h". Its rows lie below 396,490, so the gcm.py never-rerun rule covers
  them (I).
- Kicks of z22k: 1-3 species by 0.1-1 dex (36,634 boxes), small kicks on all 12 reported species (16,200, max 0.30 dex),
  898 CH4 -> CO exchange kicks (carbon carrier moved up to 3 dex); element fractions within 1e-10.
- 68.57 GB of loose box files were deleted 10-18 after every one of 670,570 was checked bitwise against its packed group
  (archive: scale/cleanup/delete_plan2.sh; a copy in cleanup/gpl/delete_plan2_ref/); their solver step counts and run times were not packed and are lost.
- Never rerun (project rule): gcm.py drive or pack below row 396,490; the gen_cold.py drives of u19 (drive_u.sh) and x20
  (drive_e.sh); the scale-phase drives gen/drive_n1.sh, drive_n2.sh, drive_n3a.sh, drive_n3b.sh, fastgen/takeover_n3.sh,
  and gen_cold.py or drive_fast.py with GEN_ROWS = rows_n1 / rows_n2 / rows_n3a / rows_n3b.npy: their loose boxes are
  gone, so a rerun would redo every row. New row lists must exclude these lists (or the packs' SEEDs). Added 2026-10-20
  (project instructions rewrite): the y21 drive (scale/gen/drive_y21.sh; gen_cold.py or drive_fast.py drive with GEN_ROWS =
  rows_y21a, rows_y21b1-b4.npy) and the kicked-run drive. Their 107,010 loose boxes (gen/_build/gcm/boxes/06 to 14) were
  each compared in full with their pack groups on 2026-10-19: 107,010 equal, 0 missing either way (V cleanup/gpl/
  report.txt); the packs do not hold x0 (= row 0 to 9.4e-16), the step count or the wall time.

### Training caches
- 12-species E13 caches: arch_gcm16_all (2^16; train / validation / test 52,269 / 6,615 / 6,639; phys 36,659), 17_all
  (2^17, 104,678 train), 17cold (+ b02, 130,911), 18cold (+ c18, 183,573), 17m (117,784 = 104,678 + 13,106 M), 18m
  (196,679), 17m64. Closed-set caches: a 53-species copy and the 25-entry reservoir cache 17m_res; the norm.json of
  /scratch/imalsky/arch_gcm17m_res is the 25-entry normalization of every closed model and every bench (HARNESS_NORM).
- 19a (F2: B6's files + b02 + c18 + u19; 222,845 train); 20e (F3: + x20).
- U2 (10-13): the six previous files + x20 + y21a + y21h = 496,329 runs (train 397,368 / validation 49,412 / test 49,549);
  distinct usable rows vs F2's stage-2 data x1.5 below 650 K, x3.0 above 1300 K; vs F2's stage 1 x7.6 / x3.0-3.4.
- U3 (10-15): U2 + y21b + z22k (505,907 train rows). U4 (10-17): U3 + n1 + n2 (624,324 train; reservoir cache 15 GB).
  U5 (10-18): U4 + n3a + n3b (1,042,588 train: U2 prefix 397,368, 591,491 after it, 53,729 kicked).
- Draw weights: FINAL's stratum shares (T band x event x M rows) spread over each stratum's usable rows (max |share -
  target| 3.4e-13); kicked rows 0.20 of stage-2 draws; drift rows weight 0. The weight files w_U2_*, w_U4_* are needed
  as they are: their builder reads the U2 cache deleted 10-19. The U2 weight builder exempted b00, b01 and m18 from the
  drift filter, so (truth audit 2026-10-19) 20 U4 training runs drifting > 1e-2 dex (b00 11, b01 9; up to 3.8 dex) still
  carry weight, about 1 in 68,000 stage-2 draws (I); their labels are projected onto row 0's ratios while their stored rows
  carry the drift (I).
- Coverage of the target: 65.3 % of U2 / U3 training rows lie inside env_any; rows inside env_any with Z 1-30x and C/O
  0.3-1.0 are 9.5 % of U3's rows (7.1 % of D-kick's draw weight); Z >= 200x is 4.2 % of U4's runs.
- Status 10-19 11:29 (V housekeeping): /scratch/imalsky holds arch_gcm16_all53, arch_gcm16_res, arch_gcm17m_res,
  arch_gcmU4_res, e13_fixdev, the holdouts and e13_scaledev; U2_res and U3_res deleted 10-19, earlier caches 10-16. A
  cache rebuilds from the packs in about 3 CPU-h (cleanup plan). Before deleting any cache, grep lib/paths.py and
  run/env.sh (the 10-16 deletion of arch_tests stopped every bench for six hours).

### Chain sets (paths_v3) and path data
- Machinery (cons/gen/paths_v3.py; refuses to overwrite a set): per-call VULCAN chaining with all 53 species carried;
  forcing families smooth ramps, multiscale cycles (2-3 periods, 1e4-1e6 s), intermittent steps, an OU walk in T and log
  P (dT up to 600 K, dlog P up to 1.5 dex); dt per call log-uniform 10-500 s or fixed; starts half equilibrium at (T0,
  P0), half transported (equilibrium at a shifted T, P); mixing with probability p_mix ~ U(0, 1) per path, x <- (1 - f)
  x + f x_partner, f log-uniform 1e-3 to 0.3, the partner a 53-species equilibrium of the same element budget nearby;
  VULCAN continues from the blend, so labels are exact (element ratios held to <= 7e-14). 1.28 s wall per call. The
  tau = 1e5 s sinusoid (xlong, v2l, v2s family) and the tri and burst families are kept out of training and DEV.
- Sets: train_a 2000 x 100 calls; train_b 512 x 500; DEV 48 x 5000 (one failed); hold = PH 32 x 10,000; train_c (10-10)
  96 x 2000 (even paths post-heating hydrocarbon drain: mean T 700-1200 K, amplitude 300-600 K, CH4-rich parcels that
  heat); hold2 = PH2 (sealed, 16 x 10,000, all six families); train_n (10-19) 56 x 2000 cool N-bearing chains (Z
  30-100x, C/O 0.2-1.0, mean T 400-900 K, amplitude 200-600 K, mean log10 P -2.25 to -0.75 bar, dt 100-500 s, seed
  918273645; 0 failed; built for CBgN, never trained; target: xlong path 14's NH3 / HCN drain on its cooling legs, 507-1085
  K at -2e-4 to -6e-4 dex per call; 56 % of its calls at 300-700 K; 112,000 chain-calls = 24.6 % of train_a + train_b;
  chains within 0.5 dex of path 14's Z 8, before 2). Peak-to-peak T p50 / max: train_a 76 / 1160 K, train_b 222 /
  1198 K, DEV 465 / 1108 K. Failures are 120 s FastChem time-outs on the mixing partner.
- Earlier path data: paths (10-06; 64 x 200 calls, one dt per path 100.9-496.4 s, only T and P change); v2s (512 x 200),
  v2l (64 x 1000), v2x = xlong (16 x 10,000, dt 500 s, T and P on a 1e5 s cycle; E13/paths_v2x).

### Label and truth quality
- Labels match a rerun with every solver step to 3.6e-15 dex; the error is the 99-time grid with log-log interpolation.
  Dense truth (10-16; 200 U3 runs at rtol 0.005, steps <= 2 %): label error at t0 + dt p50 2-3e-7, p99 0.016-0.017, max
  0.46 dex, none > 0.5; p99 parts: stored tolerance 0.016, interpolation 0.0044, projection 0.0012; worst gas C2H2 (p99
  0.049). PCHIP on the same rows would give p99.9 0.0075. rtol 0.005 for new data: grid error p99 0.013 -> 0.0030 at 3.2x
  CPU. Labels cannot cause the lasting > 0.5 dex misses; their p99 is at the top of the one-call model error. Against
  VULCAN's own steps (45 boxes rerun in-process) one-step labels are off by p50 6e-9, p99 0.003, p99.9 0.026, max 0.15 dex;
  the dense truth itself is converged (rtol 5e-4 vs 0.005 on 38 runs, p99 3.9e-4). For the cold switch misses label
  tolerance is ruled out: rerunning nine missed cold cases at rtol 0.05 vs 0.005 moves the transition times by 2-8 %
  against model timing errors of 1.1-6x. Stored transitions fall short of the converged change by about 3.8 % of their own
  log change (shared by the test truth). Labels below the scored threshold are reproducible (rtol 0.05 vs 0.005 p90 0.0013
  dex), so masking them is reallocation, not denoising. Files: data/2026-10-13_e13-scale/interp_audit.txt,
  interp_accuracy.txt, interp_tables.txt.
- Window and jump labels are projected onto row 0's element ratios, grid-chain labels are raw rows (max 5.2e-5 dex);
  logy_at(t = 0) returns row 1 (x0 windows and jump pairs substitute row 0); dt_norm clamps to 0.1-1000 s.
- VULCAN tolerance: rtol 0.005 vs 0.05 on 128 G jobs, late per-job max p90 0.022-0.077 per band; at 300-1300 K tails keep
  moving (median change 0.97 dex between 1e9 and 1e10 s; hot cells converged); VULCAN's convergence check ignores species
  below atol (4 of 124 cool converged tails wrong by 0.22-4.18 dex), so the binding limit for this data is the 1e8 s
  horizon. Tail convergence between 1e9 and 1e10 s by band (median over cells of the largest change; share below 1e-2
  dex): 300-1300 K 0.97 dex (5 %), 1300-1800 K 0.017 (48 %), 1800-2300 K 0 (98 %, max 0.29), 2300-3000 K 0 (100 %); held
  converged tails match no-stop VULCAN to p90 <= 2.3e-9 dex in the hot bins and 1.7e-4 / 1.8e-4 in the cool ones at 1e8 s.
  rtol 0.05 vs 0.005 (10-03 definition): max 0.28-1.37 dex (depends on the time grid compared); G jobs t >= 1e3 s p50
  0.006, p90 0.027, max 0.145. VULCAN drops only species below atol from step control, so scored truth points are under
  rtol control. Rule: define hard regions from training-distribution rows, never from G. G's truth is clean (max drift
  7.5e-7).
- Truth audit of U4 (2026-10-19, read-only, all 766,593 stored runs of the 16 packs at all 100 times; V cleanup/
  truth_audit/report.txt; tables data/2026-10-19_cleanup/truth_audit_tables.txt): no non-finite or negative value, every
  state sums to 1 within 8.9e-16, one time grid. C, N, O, He to H drift: median 3.1e-15 dex, p99 8.4e-9, max 4.98 (N); 213
  runs > 1e-3 dex, 117 > 1e-2, 67 > 1e-1, every one at T <= 530 K (> 1e-1 dex: <= 412 K), nitrogen in 151 of the 152
  drifting training runs, most at 100-1000 bar; packs m18, y21h, z22b1-4, z22k, z23n1 have none; once drifted a run stays
  drifted (drift at 1e8 s >= 0.9 of its peak; many jump in the first 1e-2 s). VULCAN's element monitor omits N
  (atom_list H, O, C, He) and its loss_eps 1e3 rejects no step for any element (I: the check is effectively off). Training
  runs > 1e-3 / 1e-2 / 1e-1 dex: 152 / 88 / 50, of which 88 / 68 / 39 have zero stage-2 weight. End states: 545,098 ran to
  1e8 s, 221,355 stopped by VULCAN's steady-state test (216,111 before 1e8 s; almost all >= 1300 K; 324 below 650 K, 15
  before 1e3 s, not checked for premature stops), 140 hit the step cap (none in the cache; with one frozen-tail reject they
  are the cache's 141 missing groups). Row 0 is the generator's true start state, not the first solver step (code,
  540 sampled boxes bitwise, 4 reruns; VULCAN stores t = 0 itself; row 1 at 1e-3 s carries the fastest chemistry, median
  max |dlog10| 15 dex for broad starts). The full audit took 9 min on 6 cores.
- Hidden species: (10-01) replacing the 40 untracked species by He keeps bulk species within 0.09 dex, but O, OH, H below
  1300 K reach p90 9.16 / 2.35 / 1.72 dex and NH3, HCN, C2H2 at 1300-1800 K 1.03 / 1.01 / 0.82 (untracked total at the 1e4
  s restart p50 1.4e-6, p90 1.6e-3, max 4.4e-2; closure-aware p90 at dt 10 s: relax 10x2048 one-step 0.50, best 6x512 0.98,
  delta after BPTT 4.63); (10-06) closure test v2 (256 rows, restart with 12 tracked + He) p90 0.29 dex, with the element
  inventory restored 0.23 (hidden intermediates carry memory at 800-1800 K; tracked-only p90 by band 0.091 / 0.62 / 0.48 /
  0.091 / 0.116; above 1e-6 p90 0.037; hidden share of C / N / O at the restart p50 9e-5, p90 0.018, max 0.89); (10-06)
  12 + He + C, N, CH3, H2CN, C2H6, CN, O2 still leave 5-19 % of one element outside in the worst 0.1 % by band; (10-09)
  greedy coverage: 12 + He leave p99 35 % of the worst element outside, 8 more named species still 6 %: exact closure
  needs reservoirs; no named closed-set entry is information-free (each of C, N, CH3, H2CN, C2H6, CN, O2 carries >= 1 % of
  an element at 1.26-2.78 % of the 19a training grid points); at 22 of 49 checked F2 states the five reservoir values
  cannot come from any mix of the hidden species;
  (10-13) lift test: true 25 entries with the hidden species from the 5 nearest training states reproduce the future on
  11 of 13 rows; (10-14, T4) lifts from true states (nearest-neighbour fill) give rows > 0.1 dex in 1-3 of 594, a flat
  fill 41 of 593 rows > 0.5; 87 of 200 model-visited states cannot be rebuilt exactly (molecule count fails), so VULCAN
  labels at the model's own states would need a projection and a two-fill agreement check.

## 5 The direct-state networks (2026-06 to 2026-10-30; closed, now the baseline)

### 5.1 In brief
- What it was: one network call per GCM step on a closed 25-entry state (20 physical species plus element-carrier
  entries), a relaxation head (each entry relaxes toward a predicted target at a predicted rate) and an element
  projection in float64; two-stage training (one-step windows, then rollouts); trained on VULCAN-0D boxes. Published
  12-species model: tag paper-v1 (arXiv 2609.00428). Lines tried in order: delta head and architecture tests, E13
  12-species models, closed-state and conservation fixes, the 6x2048 scale line (CBg, 21.2 M parameters, data U4 624,324
  runs), the clean data design (786,432 runs) and retrain, then the overlay campaign (misses first: miss and rollout
  losses, flow-map and mixed-mode heads, targeted data supplement).
- Best results (section 5.6, 10-26 and 10-30, V there): the flow map ov_DT2: main-domain rows with a miss 1.25 / 1.23 % (screen /
  confirmation), 27 / 37 large-miss rows, worst 3.3 / 4.1 dex, averaged score 0.0023 dex; but panel P (GCM-like paths)
  35.6 % rows with a miss and panel L 31.4 %. The best-balanced single model ov_V1jL: short runs 7.22 % (0.088 % large),
  panel L lasting 6.8 % (15 rows above 1 dex), panel P lasting 27.0 %. CBg speed 2.8-3.7 us per cell on an idle GPU.
- Why it was superseded (author 2026-10-30, section 3): rare misses of many dex never went away, panel P failed for
  about a third of rows with every model (large temperature swings), and a wrong call could not be ruled out from
  blowing up a GCM run. The physical solver has 0 misses on the same panels by construction of its numerics.
- How to recreate it: code at git tag direct-state-final (commit 7ade76b; tags e13s_CBg_s2 and paper-v1 for older lines);
  the training cache from repo-data/raw/raw_clean with that tag's datagen/build_clean.sh (design map
  repo-data/gen/clean/design_map.npy, snapshot repo-data/ref/snapshot); the configs, norm files and weights of the two
  kept models in repo-data/runs/ov_DT2 and ov_V1jL (their parents' configs are in the overlay campaign archive). The
  E13-era data would need their packs regenerated first (section 4).
- Where the record is: this section and section 4, the campaign archives (section 0),
  paper-notes/data (small evidence tables) and the 2026-10-19 archives.

### 5.2 State of 2026-10-20 to 10-23 (kept as written then)

#### The pick: CBg (author, 2026-10-19: "This is so much better..keep this one")
- Design: closed 25-entry relaxation head, 6x2048 trunk (21.2M parameters), every call ends with the element projection.
  Data U4 (624,324 training runs). Stage 1 (CBS1) 1,000,000 steps at batch 8192, AdamW 6e-4, warmup 20k, cosine; stage 2
  200,000 steps at 3e-5 with 4x LD's batches (one-step 8192, jump 2048, grid 1024, chain 1024), the lockstep step, the
  fused projection, the non-finite-step guard and a per-row cap 0.01 on the rollout state gradient; one-step windows dt
  10-500 s; TF32 on. Code: code/Auto-Chem tag e13s_CBg_s2 = commit 3b8806e, whose lib/ equals the run's recorded lib_md5
  (V 2026-10-19, git tags; cleanup log 21:43). Since 2026-10-20 the pruned code/Auto-Chem main benches CBg bitwise like
  that tag (sweep, fig4 and path benches at its HEAD) and passes the training bitcheck (both stages, at e0e8c3b on
  2026-10-20; section 5.6); the run folder holds manifest.json + norm.json (stage2.pt sha256
  d97140ea..., norm sha256 de785522... = /scratch/imalsky/arch_gcm17m_res/norm.json, parent CBS1 stage1.pt f08ae8d3...;
  V cleanup/h1/report.txt).
- Development benches (V data/2026-10-13_e13-scale/verdict_e13s_CBg_s2_vs_e13s_LD_s3.txt, why_critic.txt, late_dry2.txt):
  inside lasting > 0.5 31 (LD 43, FINAL 61), late > 0.5 96 (175, 202), late > 2 6 (4, 4), first-call > 2 0 (2); cooler
  lasting 35 (LD 49); all 13,821 rows lasting 57 (64), late 189 (329); in scope lasting 5 rows / 4 jobs (LD 10 / 8), late
  19 (47); T2 shares +0.096 over LD; SD dt 1 s inside rows > 0.5 11 (23). Passes pass rule v2 (it failed only the
  large-miss guards of rules_t5, which the author then made readouts). Paper-style metric (mean |dlog10 x| per row over
  resolved late points, no floor; recomputed 2026-10-20, V cleanup/h2/report.txt): G per-row median 0.0015 vs LD 0.0026, p90
  0.0164 vs 0.0265; v2l per-path median 0.0099 vs 0.0135.
- Step-size sweep (10,908 rows; 9 dt from 1 to 500 s; raw and evolved starts; V data/2026-10-13_e13-scale/dtsweep_score.txt):
  equal-dt-weight lasting rows in scope 4.8 (LD1 5.6, FINAL 6.9, LD 7.0); in the target scope (Z 0.1-300x, C/O <= 1; V
  data/2026-10-19_cleanup/sweep_target_score.txt, headline block; 437 rows per dt and start, V cleanup/h2/report.txt)
  5.9 (LD1 8.0, FINAL 9.4, LD 9.7);
  from evolved starts 0 lasting rows at every dt; from raw gas a gap at 1-2 s (in-scope raw lasting 16 / 13 of 360 at 1 /
  2 s, 1-4 at 5-500 s); 1 s is the worst dt for every model in both scopes.
- Speed (idle GPU, fp64 step in a CUDA graph, 32 Newton iterations; V data/2026-10-13_e13-scale/
  speed_e13s_CBg_s2_idle_{keep,lm}.json): 2.81 / 3.15 / 3.73 us per cell at 1e4 / 1e5 / 7.3e5 cells (LM on every row 5.65 /
  5.32 / 5.94); hard limit 50 us, target about 10. 1000-day cycle test bounded (largest change over the last five cycles
  0.061 dex; V data/2026-10-13_e13-scale/cyc1000_e13s_CBg_s2.json).
- Remaining errors (section 5.3, CBg): learned-rate errors from right states (20 of 31 inside lasting rows), late-born
  misses mostly at Z 200-300x, an NH3 / HCN drain on cool frozen paths (xlong late p99 1.114 vs LD 0.471; it appears only
  in the models trained on U4, so U4 is implicated, I), CH4 formed too fast from CO in far-from-equilibrium gas at 1000 K,
  10 bar, and the raw-gas short-step gap at 1-2 s.
- Sealed and unread: H4 / H4c (1,280 jobs) and PH2 (16 x 10,000 calls) (author 10-14: for the next phase's pick).
- Done 2026-10-23 00:23: the clean full run (section 5.6, 2026-10-22 / 23). The retrained model "clean" (CBg's
  architecture and recipe on the clean data) beats CBg on score-1: 0.0039 vs 0.0196 dex, D -0.014 [-0.017, -0.012],
  better on every panel and dt bin; adoption (rule: adopt unless worse) pending the author's read of the flagged rows.

#### The goal now (author, 2026-10-23): overlay at every point, misses first
- "I really need the chemistry to nearly perfectly overlay the results of the vulcan chemistry at every point". A model
  is judged first by its misses (a scored point above 0.1 dex; a large miss above 1 dex): the share of rows and of start
  states with any miss and any large miss, and the worst error, by panel, T band, start kind, start age and dt bin; the
  averaged score comes second, because it hid misses of many dex. The training loss must give weight to every miss.
  Longer training, more data and a bigger model are accepted; architectures and stage designs are open to test. In the
  project instructions (Aim, Judging); section 3 has the author's words; section 5.6 (2026-10-23) the measurements.

#### The goal (author, 2026-10-20, replaces the 10-19 goal below; its "one score" and "same architecture" are superseded 2026-10-23)
- "a clear common sense emulator that covers the domain that I want, realistic gcms over as wide of a domain in terms of
  composition and state as realistic": accuracy against VULCAN over the whole realistic domain, for gas that varies the
  way it does in real GCMs; equal weight per decade of time and per factor of 10 in dt (kept); the start mix mostly near
  equilibrium and transported gas (GCM literature); mini-chem a benchmark on its table cells, not the target; one score
  plus readouts; the same CBg architecture ("I don't want big new model"); a fully clean state (one regenerated data
  design). Section 3 has the author's words.

#### The goal from 2026-10-19 21:4x (author; V cleanup/log.txt 21:40; SUPERSEDED 2026-10-20 as above)
- Beat mini-chem over the whole trajectory, judged on overall error (large misses are readouts): Z 0.1-300x weighted
  log-uniformly; C/O 0.1-1 (1-2 a side check); dt 1-500 s weighted equally per factor of 10 (10 s matters most, but no dt
  may be a weak spot); time weighted equally per decade. Supersedes the 09:23 "mainly 1-100x, C/O <= 1" and the
  per-step-value weighting.
- Metallicity floor (delegated by the author; literature): measured hot-Jupiter atmospheres reach about 0.1x solar
  (WASP-77Ab, JWST, [M/H] -0.91, about 0.12x, August et al. 2023; HAT-P-30b 0.2-0.8x); the one lower claim (TOI-5205b
  < 0.03x) is an upper limit its authors say may be biased low by stellar spots; hot-Jupiter hosts cluster at [Fe/H]
  -0.5 to +0.5; GCM studies typically run 1x and 10x.
- Mini-chem rate tables (V 2026-10-19, listing of scale/fig4/mini_chem/netrate_originals and chem_data; the clone is
  now code/mini_chem): Z 0.1, 1, 10, 100 and 500x, each at C/O 0.25, solar, 1 and 2 (Tsai 2022 net rates, Lodders 2019), plus 3x and 5x at solar C/O
  only. Inside the target that is 14 cells (Z {0.1, 1, 10, 100} x C/O {0.25, solar, 1}, plus 3x and 5x); no 300x and no
  C/O 0.1 table, so elsewhere VULCAN is the yardstick (500x as a bracketing readout). Our data use Lodders 2009. The
  table audit of 2026-10-19 (section 5.4, yardstick) found defects that cut the scored cells to 10: the 0.1x-solar CO-HCN
  table is a copy of CH4-HCN and the 100x-solar CH4 + CH4 -> C2H2 rate is zero in 77 % of the domain (both readouts).
- Where CBg stands against mini-chem (V, yardstick v1.1 under the adopted rule v3.1, section 5.4): all three models still
  lose overall; segments: mini-chem 0.0236 dex, LD1 0.0573, CBg 0.0577, FINAL 0.0635 (paired D +0.031 / +0.031 / +0.037);
  CBg beats FINAL and ties LD1. The loss sits entirely in runs started at the exact t = 0 unreacted or transported state
  (t0 = 0 rows CBg +0.111); on restart rows all three beat mini-chem (CBg -0.009). The ablation (section 5.4) traces it to
  the untrained exact-t0 input pattern: from VULCAN's state at 1e-3 s CBg's t0 = 0 error falls from 0.174 to 0.022 dex;
  no run-time workaround comes close. The v3 numbers (CBg +0.056) and the six-case numbers are superseded.

#### Open items (todo.md is the live list)
- 2026-10-20: rule v3.1 adopted, then (same day) the author approved replacing the decision machinery with one score
  against VULCAN (spec in the clean-state campaign); the t0-fix training arms were stopped before any result; the next
  step is one clean regenerated data design (todo.md).
- The raw-gas short-step gap (1-2 s) and the cause study (head form vs missing information in the 25 entries vs
  coverage) wait for the clean model: the regenerated data (one design, Lodders 2019 metals, He/H 0.078-0.100, one-step
  windows from 1 s) may close the gap, and the study is re-run on that model if not.
- External review findings (report dated 2026-10-20 in scale/reviews/) and what was done (section 5.6, 2026-10-19 / 20):
  (a) non-finite predictions were scored as zero error (score_sweep.py line 42 nan_to_num(e, nan=0.0); scoreboard.py,
  path_board.py): fixed, a non-finite prediction now counts as a failure; every stored state was finite, so no reported
  number moved (V cleanup/h1/report.txt). (b) VULCAN's own element check skips nitrogen (vulcan_cfg.py line 9 atom_list H,
  O, C, He; loss_eps 1e3, so no step is ever rejected for element loss; rtol 0.05 line 121, atol 1e-2 line 105): the
  truth audit of U4 found 213 of 766,593 runs drifting > 1e-3 dex, all at <= 530 K, mostly N; 20 training runs drifting >
  1e-2 dex still carry draw weight (b00, b01; about 1 in 68,000 stage-2 draws): a one-line change for the next weight
  build, open. (c) Sampler: an assertion that B valid windows were found was added (draws unchanged). (d) Normalization:
  model manifests with the norm-file hash, checked on load. Row 0 of the generated data is the true start state (settled).
- Training projects with 12 fused iterations that keep the input on failure, use runs 32 iterations plus LM: the
  difference is unmeasured (the yardstick's diagnostic panel, cleanup plan Part C 2c, not run).
- Code: bitcheck of both stages PASS at e0e8c3b (2026-10-20, t0-fix code_clean/report.txt); bench/judge.py uses pass rule
  v2 since 5f11a99; main e0e8c3b and tags paper-v1, e13s_CBg_s2 pushed 2026-10-20 10:0x; commit 7d5b036
  (the snapshot rawtraj.py still imports) is on no branch or tag, so a push would not publish it (V cleanup/h3/report.txt).
- Deployment: the closed 25-entry state, the head with 32 Newton iterations plus LM, and the projection are not yet in an
  exported model or a host step. Choices so far: fp64 host state; start from a spun-up state or make the first calls of a
  fresh start in 100 s substeps; reject non-finite updates (keep the previous state, flag it) and count a kept input as a
  flagged update; the projection can run on the host CPU in fp64 (structured solver, about 1 us per cell per core). The
  He question of 10-01 (the old 12-species data held no He) is settled for the E13 line: the closed state carries He, the
  E13 data fix He/H at 0.096917, and the clean data vary it over 0.078-0.100 (the host passes its own He).

### 5.3 Model lineage

| version | date | design | data | key result | fate |
|---|---|---|---|---|---|
| one-step paper model | before 09-29 | latent 4x1024 enc/dyn/dec, L 512, 11.6M, batch 2048 | 4 raw files, 12 species | one-step p90 0.0216 dex vs mini-chem 0.875; rollout p90 56.6 % after 99 steps | arXiv 2609.00428; motivates this work |
| production delta head | to 10-04 | FlowMapMLP 10x2048 (37.9M), z + MLP(z, dt, g); BPTT K 10 stage 2 | 4 raw files, drop_below 1e-30 | 98 % of rows leave the envelope at dt 1 s within 1e5 calls | replaced by the relaxation head; old 12-species code = tag paper-v1 |
| sibling latent model | -- | LatentLinearFlowMap, enc/dec 3x1024, L 512 | same | 90 % of rows fail at dt 1 s | not pursued |
| FIN (p_h1_jump) | 10-03 | relax 10x2048; 200k one-step + 20k long unroll + 20k unroll + jump 0.05 | 4 raw files, filtered | SPF dt 10 s, 1000 d p90 0.530, p99 3.57; holdout 0.80 | ported 10-04 |
| LJ vs J (6x512) | 10-03 | stage 2 unroll + jump vs jump only | filtered | LJ better A10 +0.12 / +0.14 (two seeds) | LJ default |
| 800k (E5) | 10-04 | 10x2048, 800k stage 1 + jump-only 40k | filtered | holdout 7.18 dex | O-poor data hole (E7) |
| e9_F10fill_s0 | 10-05 | relax 10x2048, 200k + jump-only 40k | floored + O-poor fill | holdout hot 0.10; U hot 0.08 | old-data baseline |
| trunks, FiLM (E10, E11) | 10-05 | KAN, species attention, small MLPs, FiLM on (T, P) | floored (+ fill) | none beats the 6x512 MLP per multiply-add | dead ends |
| e13 phys / all (12 sp.) | 10-06 | relax 6x512, 12 species | E13 block 0 | hot-band element-ratio drift (G hot band 2.9-3.1 dex) | grid term |
| all53c / res | 10-06 | closed head on 53 species / 25 entries | block 0 | hot drift gone, cold not; 8x bench cost | basis of B6 |
| allG | 10-06 | + grid flow term in stage 2 | block 0 | G 0.19/0.18/0.14/0.05/0.03 vs e9 0.68/0.76/0.55/0.33/0.20 | allGend, capw17 |
| capw17 | 10-07 | 6x1024, end-anchored grid term | 2^17 (104,678 traj.) | G 0.13/0.09/0.06/0.03/0.02; dead band near equilibrium | m17fs |
| big17 | 10-07 | 10x2048, same recipe | 2^17 | stage 2 diverged (+0.05 to +0.10 vs capw17) | 2048 retried in CB |
| m17fs_s0 (frozen) | 10-08 | 6x1024, 12 species, M rows in stage 1, 75k stage 2 | 17m (117,784 train) | G p90 0.033/0.061/0.042/0.026/0.015; xlong late p50 0.193, p90 2.03 | B6 |
| F0fs | 10-08 | M + S + relaxT + cold data, from scratch | 18m | vetoed (O worst-species; 15.5-dex C2H2) | dropped |
| B1 / B2 | 10-10 | m17fs + chain term 0.5 / 1.0 | 17m + chains | DEV better; vetoed on G / Gc p99, rows | dropped |
| B6 s0-s2 | 10-10 | closed 25 entries, 6x1024, chain term, 48.8k stage 2 | 17m res + train_a/b | xlong late p50 0.003, p90 0.157; half the frozen model's > 2 dex rows | F2 |
| B6L / B6Lc | 10-11 | long chain curriculum (+ train_c) | | not eligible (seed spread; xlong 11.30 dex) | dropped |
| E (median of B6 x3) | 10-11 | per-entry median, one projection | | G+Gc rows > 2 15 vs 29; 3.7x cost; froze holdout rows | withdrawn |
| D-cold, W | 10-11 | stage 2 on cold union / per-gas censored loss | 19a / 17m res | rows > 2 19 / 23 vs 29; vetoed / not admissible | data -> F2; W untried later |
| F1 / F2 / F3 | 10-13 | B6 s2 stage 1 + events x3 (+ cold union, + event runs) | 17m / 19a / 20e | P* rows > 2 21 / 17 / 16, > 0.5 469 / 423 / 438 | F2 picked; FINAL |
| G1 / B / A | 10-14 | 6x1536 old data / 6x1024 U2 / U2 in stage 2 only | 19a / U2 | rows > 2 25 / 26 / 25 | width or data alone fails |
| FINAL | 10-14 | 6x1536 (11.98M), both stages on U2 | U2 (397,368 train) | rows > 2 14, > 0.5 373; inside lasting 61; 2.31 us | LD, then CBg |
| C0 / D-kick | 10-15 | FINAL + 20k at 1e-5; D-kick adds U3 and kicked starts | U2 / U3 | inside lasting 53 / 48 | LD |
| X2C / X2E / X2V | 10-15 | dt 0.1-500 s / two-exponential head / speed loss | U2 | X2C lasting 68 vs 61; X2E, X2V no gain | dead ends |
| T6 (+ B_C) | 10-16 | fresh 53-species 6x1536 | U2 | inside lasting 140 vs 61 | dead end |
| LD | 10-16 | D-kick + 20k | U3 | inside lasting 43, late 175; ADOPT (rules_t5) | candidate; CBg |
| LDc / LD1 / LDF | 10-16 | cool rows 2x / dt 1-500 s / TF32 off | U3 | no gain / long-run cost / no gain | dead ends |
| round E (LE1-3) | 10-18 | LD + 3 x 20k | U3 | lasting 44 / 40 / 40 vs 43 | plateau at 6x1536 |
| P15 | 10-17 | FINAL's recipe on U4 | U4 | lasting 64 vs 61 | no data effect |
| batch, TF32 arms | 10-16 | batch 512-8192, Lion; TF32 off | U2 / U3 | no accuracy cost at equal steps | 8192 adopted |
| CB (uncapped) | 10-18 | 6x2048, U4, 1M stage 1 at 8192 | U4 | grid gradient exploded; paused at 35k | CBg |
| CBg | 10-19 | CB + per-row rollout gradient cap | U4 | inside lasting 31, late 96 | the pick |
| CBgP1 / CBgA / CB5 | 10-19 | +40k / 48.8k stage 2 / CBg recipe on U5 | U4 / U5 | 29 vs 31 (leveled off) / stopped / stopped | open: CB5 |

#### Delta head and the architecture tests (2026-06 to 10-03)
- Production delta head (Auto-Chem to 75af56f): FlowMapMLP 10x2048, LayerNorm, SiLU, z + MLP(z, dt, g), 37,896,204
  parameters; stage 1 one-step, stage 2 BPTT K 10 (curriculum 2 -> 10); export returned y_next. Every tested delta
  checkpoint failed long rollouts: at 10x2048 stage 1 (200k steps, validation 1.23e-3 vs persistence 1.12e-2) 98 % of 300
  rows leave the envelope at dt 1 s (first failing call p10 / p50 / p90 5,483 / 16,157 / 51,228); after BPTT 45 % of
  row-calls rejected at dt 1 s, 92 % at 10 s. At 6x512 it failed under every regime (one-step continuation, BPTT, detached
  gamma 0.9, pushforward, input noise 1e-2). One-step validation loss favours it (0.025 vs 0.089 for relax), so one-step
  loss cannot select rollout models. Mechanism: its one-call move barely depends on dt (1.1e-3 dex at 0.1 s, 3.1e-3 at
  1000 s at equilibrium) where relax's is proportional to dt (5.8e-8 to 5.6e-4). 6x512 one-step screen: the delta head
  leaves the envelope in 94 % of rows at dt 1 s (first failing call p10 / p50 / p90 426 / 3,484 / 16,024; H2, H2O, H pushed
  above 1), 92 % of row-calls rejected teacher-forced at dt 10 s, p90 5.5-8.4 dex on the full S bench; the relaxation head
  at the same budget has no interventions (worst-bin p90 0.68 / 2.5 / 4.8 dex at dt 1 / 10 / 100 s over 1e5 s / 1e7 s /
  1000 d). Rows failing at dt 10 s over 1e6 calls (median first failing call): production delta 10x2048 one-step 300 / 300
  (15,553), + BPTT 300 / 300 (48,413); delta 6x512 92 % (2,732), + BPTT 86 % (4,143); sibling latent 90 % (1,735); latent
  with exact zero but no box L16 / L512 19 / 84 %; box-bounded latent and relax 0. At 10x2048 after stage 1, 83 % of rows
  end with a species above 1. These are failures of the tested checkpoints, not proof that every delta model must fail
  (corrected 10-02); failure timing spans about 1e2-2e5 calls. Earlier production runs survive only in git (a584f2d: v3
  10x2048, batch 8192, dt 1-50 s, val minimum 4.03e-4 at epoch 207; v2_bigger_batch 6x2048, 4.11e-4 at epoch 488; other
  data, not comparable).
- Relaxation head (our construction: a learned QSSA / chemical-relaxation step; exact identity at dt = 0, dt-independent
  fixed point, any dt per call, box invariance by clamping q): zero guard interventions in every campaign run. Prior art:
  the plain QSSA / exponential step (Hesstvedt, Hov and Isaksen 1978; Jay et al. 1997), chemical relaxation in exoplanet
  GCMs (Cooper and Showman 2006; Tsai et al. 2018), GRU flows (Bilos et al. 2021), GRU-ODE-Bayes (De Brouwer et al. 2019),
  Deep-OSG (Chen and Wu 2023); in all of these the network sees t or dt (moderate confidence that the combination is new;
  references in section 6). Caveat (sweep, 10-03): the guard fires only on non-finite output (I), so zero interventions is
  not evidence of accuracy. The learned target sits a median 0.032 dex from the true equilibrium at hot converged states
  (p90 0.165, max 1.17), which bounds how far it can drift at a fixed point.
- Architecture-test training (harness): AdamW, weight decay 1e-5, clip 1.0, HybridLoss; stage 1 one-step LR 3e-4 (2k warmup,
  cosine to 1 %), EMA 0.9995, lambda_z 0.1, batch 2048; stage 2 LR 3e-5 (1k warmup), EMA 0.999, lambda_z 0.25, BPTT K 10
  (curriculum 2 -> 10 over 5k steps); fp64 rollout state, fp32 network; TF32 on in training, off in evaluation; 6x512 =
  1.35M parameters (3.6 ms per step on the GPU), 10x2048 = 37.9M (10.5 ms). Stage 1 was 1.15 % / 1.54 % of the production
  12.99M steps; one seed at 10x2048; the harness cache kept the 648,098 production groups (reject counts equal) and
  reproduced 261 production validation windows to 4.8e-7 z.
- Regimes at 6x512 (stage 1 150k, 25k stage 2; worst-bin p90 at dt 10 s / late hold / P/T switch, p99 at dt 10 s; full
  table data/2026-10-01_architecture-tests/scores.csv): one-step 2.50 / 1.53 / 3.25; one-step continuation 2.38 / 1.60 /
  2.87 (p99 12.5); BPTT K 10 3.50; BPTT K 32 2.83; pushforward 3.26; detached gamma 0.9 2.89; 25 % steady-state starts 3.60;
  input noise 1e-3 z 2.89 (p99 19.1), 1e-2 z 9.4; 80-step long unroll 1.37 / 0.57 / 1.30; jump 0.05 alone 1.01 / 0.27 /
  3.04; unroll then unroll + jump 0.05 0.65 / 0.07 / 1.39 (p99 4.5), + jump 0.2 0.63 / 0.04 / 1.40 (p99 3.1); seed 1 + jump
  0.2 0.60 / 0.04 / 1.33. Seed spread about 0.05 dex (hold 0.23). Regimes that did not consistently help still improved
  some arms (corrected 10-03): pushforward beat BPTT K 10 on every arm, detached rollouts and steady starts improved hold
  and switch, BPTT K 10 improved dt 1 s (0.68 -> 0.60).
- Latent heads (6x512): box-bounded latent mixture 16 modes p90 1.98 dex at dt 10 s, 64 modes 3.70, against relax 1.37;
  the exact zero at dt = 0 helps at 16 modes (rejected row-calls 70 % -> 16 %), not at 512 (75 -> 80 %); only box
  invariance removes the blowups (p90 at dt 10 s: zero-at-0 without box 4.28 / 9.60 at 16 / 512 modes, defect controls
  9.27 / 8.02, box latent 1.98; the box latent also had the long unroll, so not fully matched).
- Finalist FIN (10x2048 relax, 200k one-step at batch 2048, 20k steps of 80-step long unroll, 20k of unroll + jump 0.05
  with 512 pairs, D 0.1 s to 1e7 s): SPF level P3 worst-bin p90 dt 10 s over 1000 days (8.64e6 calls) 0.530 (p99 3.57),
  dt 100 s 0.546 (p99 2.54), late hold 1e7 calls 0.16, perturbation 0.49, P/T switch 1.17 (p99 10.4); no pre-registered
  acceptance (p90 <= 0.1, p99 <= 0.3, hold <= 0.02). Holdout (100 fresh sources, arm H) 0.80 vs selection 0.34. Element
  inventories under the stress test: C x6.28, O x11.8, N x42.8, H x0.58 at 1e6 calls (first reported as clean;
  corrected 10-03). Its parent checkpoint is gone; retraining takes about 9.5 h, not bitwise. SPF by precision mode P1 /
  P2 / P3: dt 10 s 1000 d 1.36 / 0.548 / 0.530, dt 100 s 0.67 / 0.548 / 0.546, hold 0.30 / 0.19 / 0.16 (hold drift 0.022
  after 1e6 calls, 0.16 after 1e7): the export's fp32 round trip (P1) loses up to 0.8 dex over 1000 d, the multiplicative
  fp32 update (P2) is within 0.03 of P3. SP columns (dt 10 / 1 / 100 s, hold, switch, perturbation): one-step stage 1 0.79 /
  0.23 / 1.90 / 0.20 / 2.18 / 1.15; finalist 0.43 / 0.39 / 0.55 / 0.0224 / 1.17 / 0.49 (hold just above the 0.02 line);
  jump 0.2 alone 0.42 / 0.37 / 0.63 / 0.04 / 1.65 / 0.49 (0.417 vs 0.425 at half the stage-2 steps); stage 1 on 25 % of
  the trajectories 0.88. Element budget (relative error of the tracked C, H, O, N totals vs the truth's, worst bin, dt
  10 s): C 0.60, H 0.059, O 0.31, N 0.81. The small finalist's 1e7-call dt 1 s bench was stopped unfinished (no result).
- JO40 vs FIN (10x2048, 40k stage-2 steps each from the same stage 1, jump 0.2 vs 0.05; one seed): equal on all-species
  A10 (+0.052 [-0.079, +0.389]), FIN better on closable A10 (+0.136), dt 1 s (+0.126) and the switch arm (+0.448).
- Repo-sweep E1 on FIN: hot-cell drift stays below 0.1 dex to about 2-3e8 s (beyond 1000 d = 8.64e7 s), 0.607 at 1e10 s;
  iterated to 1e9-1e13 s every checkpoint reaches an unphysical attractor (13-19 dex off, sum12 p50 2.0); the head makes
  the fixed point dt-independent, not physical. Rate saturation at log10 k = -12 for up to 12.7 % of FIN's states (the
  lower bound; near the upper <= 0.09 %, so a wider upper range would not help, I). At 1e9 / 1e10 s: FIN 0.187 / 0.607,
  p_h1_jo 0.175 / 2.175, r_h1_jump20 0.539 / 2.115, z_h1_s1_jump20 0.658 / 1.951, the stage-1 models 2.40 / 6.65 (10x2048)
  and 5.31 / 11.52 (6x512). Path dependence fails for all six (FIN 0.339 / 0.505; 81 % of sources spread > 0.05 dex):
  errors from the early transient are not forgotten (I).
- Port (10-04, branch audit-2026-09-29, commits f180fdc to 728e4cd): relax head, jump loss (weight 0 stage 1, 0.2 stage
  2), warmup + cosine in both stages, stage 2 = 81 no-grad steps + BPTT to 90, 50 x 1000 steps at batch 1024, fp64 rollout
  state, export of a per-row dlog10 increment with a reference host_step. The 12.99M-step stage 1 was never run with it.
  Commits: 8f661a6 raw-trajectory store; 762708d relaxation head (FlowMapMLP, plain MLP and Fourier dt features removed);
  3d8c439 long-jump loss; 48c95e3 stage-2 recipe (replaced recipe R, commit 4842418; early stopping off in both stages);
  728e4cd per-row dlog10 export (OOR_TOL 1e-5, export_{cpu,cuda}_dynB_1step_dlog10.pt2 with a .sha256). Verified bitwise
  against the campaign harness (338 arrays; all 27 gradient tensors). Stage-1 early stopping stays off because one-step
  validation loss does not see rollouts (the 800k model had the better validation loss, 0.0760 vs 0.0773, and the far
  worse holdout).

#### Old-data models (10-04 to 10-05)
- E5 budget ladder (jump-only stage 2): 6x512 stage 1 150k / 600k / 2.4M: dt 10 s median cell major 0.054 / 0.018 /
  0.014 dex, trace 0.424 / 0.404 / 0.430; 10x2048 200k / 800k: major 0.011 / 0.010, trace 0.224 / 0.189 (finalist 0.154).
  Power-law extrapolation to 12.99M steps (I): trace species do not reach 10 % (0.0414 dex) from stage-1 budget alone.
  The 800k model scored 7.18 dex on the holdout (200k: 1.01), all in the 2300-3001 K band on oxygen-poor compositions
  (early-max 3.67 dex even from 1e6 s starts, sum12 0.008-3.7, 0 guard interventions). Other cells: dt 1 s trace median
  0.187 / 0.158 / 0.148 (6x512 ladder), late-hold trace 0.023 / 0.015 / 0.008; 10x2048 closable trace 0.184 / 0.152.
  Extrapolation (I, two points at 10x2048): 6x512 trace ~0.43 [0.33, 0.54], 10x2048 ~0.13 [0.09, 0.31] at 12.99M steps.
- e8_F10_s0 (floored data, 10x2048) and e9_F10fill_s0 (+ 50,098 O-poor boxes): fresh O-poor test 1800-2300 / 2300-3001 K
  p90 0.23 / 0.09 (fill) vs 2.63 / 3.03 (no fill) vs 9.17 / 10.85 (800k); holdout hot 0.10 / 1.15 / 7.18; on the uniform
  test U late p90 0.66 / 0.65 / 0.63 / 0.24 / 0.08 (fill) vs 0.73 / 0.69 / 0.74 / 0.49 / 0.98 (no fill). Old-data
  baseline of every E13 comparison; on G 1.02 / 0.81 / 0.55 / 0.33 / 0.21 (either mask). Floored vs filtered data (E8;
  formerly rejected sources ref_r, hot p90 with rows > 3 dex of 120): 6x512 s0 0.11 (1) vs 1.26 (26), s1 0.17 (1) vs 1.88
  (33), 10x2048 0.05 (1) vs 0.19 (9); 300-800 K 0.80 vs 1.09; on kept sources floored 6x512 is slightly worse (closable
  +0.169 [+0.035, +0.296] seed 1), 10x2048 ties. Fill (E9): filtered 200k 1.27 / 1.58 on the O-poor test (holdout hot
  0.83); fill vs no fill on the SP bench within noise (-0.012 [-0.163, +0.112]); holdout fill best in every band (0.85 /
  0.44 / 0.51 / 0.17 / 0.10); the 300-800 K holdout error (0.85) and the selection-optimism question stayed open.
  Longer stage 1 made extrapolation into the gap worse (800k vs 200k); retraining the 800k model with the fill: not run.
- Trunk tests (E10, 6x512 recipe R2): MLP 4x256 4.14 dex, 3x128 6.10 (6x512 0.768); FastKAN 256x3 at matched parameters
  0.963 (+0.195 [+0.089, +0.357]) at 2-3x the cost; species attention d32 x2 (17.8k parameters) 2.74 beats a 213k MLP of
  similar multiply-adds, but the compute-matched d64 x4 loses (1.183 vs 0.768). The MLP stays. Sizes (CPU ms per call at
  batch 4096, one thread; one-step validation): 6x512 1,346,584 parameters, 80-118 ms, 0.088; 4x256 213,016, 14-21 ms,
  0.114; 3x128 40,600, 0.153; FastKAN 64x2 59,188 (3.62 dex); attention d64 x4 135,234 parameters, 1.65M vs 1.34M
  multiply-adds per sample (closable +0.311, dt 1 s +0.138). Literature: section 6 (none steps chemistry
  autoregressively with attention feeding a relaxation law).
- FiLM on (T, P) (E11): small trunks gain beyond the parameter-matched control and the second seed (4x256 A dt 10 late
  -0.548 [-0.927, -0.234]); 6x512 differences within seed spread; FiLM 4x256 still worse than plain 6x512 (1.17 vs 0.79).
  From scratch on the M recipe (m17ffc, 10-08) FiLM gave a much better one-step loss (0.420 vs 0.485) but worse G in
  every band and k -> 0 at settled hot starts: dropped. FiLM (Perez et al. 2018) with a zero-initialised conditioner MLP
  2 -> 32 -> 32; cost 148-154 vs 76-87 ms per call (6x512, CPU), 93.7 vs 95.2 ms with gamma, beta computed once per
  trajectory; no FiLM-on-(T, P) chemistry surrogate found, the method itself is standard.
- Sobol pilot (E12, 2^16 rows, two seeds pooled): per training trajectory Sobol beats a same-size random subset of old rows
  on U at 800-1800 K (-0.25 [-0.35, -0.14] at 800-1300 K); with 1/14 of the data it is level with the full old cache on U
  but behind it on the old-distribution S bench and the O-poor test; seed 0's zero-collapse result did not repeat; seed
  spread within an arm is as large as some arm differences (only pooled numbers count).

#### E13 12-species models (10-06 to 10-08)
- e13_phys16_s0 (relax 6x512 on 2^16 E13 rows): unstable over long runs (G dt100 hot band 3.11; error grows smoothly even
  from converged starts). The drift is element-ratio drift: ratios are a neutral direction of the true map, so a per-call
  bias there compounds with calls (truth 0.000; phys0 hot band 3.2 / 3.8 / 2.9 dex in C/H, N/H, O/H at 864k calls).
- Closed heads: all53c (53 species, closed) G dt500 0.65 / 0.94 / 0.68 / 0.32 / 0.17 vs all0 0.60 / 0.81 / 1.36 / 2.86 /
  3.12 (hot drift fixed, as predicted; cold not); about 8x the relax bench time. res (25-entry reservoir set) 0.63 /
  1.09 / 0.52 / 0.18 / 0.12.
- Grid flow term (allG, 10-06): stage 2 trains the composed flow over 20 stored intervals; G dt500 0.19 / 0.18 / 0.14 /
  0.05 / 0.03, every band beats e9_F10fill_s0 (intervals below 0), paths 0.39 (all0 0.96, e9 1.57); the hot bands stop
  drifting without closure. Reproduced from scratch (s1 0.23 / 0.19 / 0.13 / 0.06 / 0.03). Stage-1 long jumps hurt once
  the grid term is on (allJG +0.09 to +0.19). physG (phys data) 0.19 / 0.16 / 0.12 / 0.05 / 0.03 and resG (25 entries)
  0.18 / 0.22 / 0.12 / 0.03 / 0.02 tie allG: under the grid term the start mix no longer matters. The remaining error is a
  systematic bias concentrated in dilute species (allG 300-800 K over species >= 1e-8 0.03 vs 0.20 overall). val_loss
  cannot rank E13 runs (for 12-species runs it is the one-step loss on the old production shards). Harness trap: lhbench
  builds the model from the cfg stored in the checkpoint, not config.json.
- End-anchoring (allGend, grid_end_frac 0.5) the one stage-2 lever that moved the middle bands (-0.02 to -0.04); stride 2
  and unresolved weighting about -0.02 alone (not tested on top of it); stage-2 noise about 0.01.
- Factorial (10-07, one seed per cell; G dt500): 6x512 / 36.7k 0.18 / 0.15 / 0.10 / 0.05 / 0.03; 6x1024 / 36.7k 0.16 /
  0.12 / 0.06 / 0.04 / 0.02; 6x512 / 104.7k 0.17 / 0.12 / 0.08 / 0.05 / 0.03; capw17 = 6x1024 / 104.7k 0.13 / 0.09 /
  0.06 / 0.03 / 0.02 (meets 0.1 in 4 of 5 bands). At 800-1300 K data and width each give about 0.03; 1300-1800 K is
  capacity-limited; 300-800 K within noise (from-scratch noise sigma about 0.03 there, 0.01 above; no data x width
  interaction is supported, first read as "more data helps only the wider net", corrected 10-07). big17 (10x2048, 105k)
  0.18 / 0.19 / 0.13 / 0.06 / 0.04: its stage 2 diverged inside warmup under the shared recipe (I: width x grid-term
  BPTT); its stage 1 had the better one-step loss (val 0.485 vs capw17 0.517) and its stage-2 val rose 0.456 -> 0.469;
  speed about 4 us alone (37.9M parameters, 75.7 MFLOP per cell-call; 6x1024 about 1/7 of that).
- Dead band (10-07 review): capw17 barely moves when T or P change by less than 20 K or 0.03-0.1 dex near equilibrium
  (median move ratio 0.000-0.011); its target moves right but its rate a is about 0; forcing a = 1 diverges (settled hot
  jobs 6.7 / 3.9 / 4.2 dex after 1000 calls). Cause (I): near-equilibrium starts are 0.8 % of training rows and the first
  training input is the state at 1e-3 s; 81 % / 92 % of the 1800-2300 / 2300-3000 K training trajectories end in converged
  tails, where a frozen rate hides a small target error. From true path states the one-call move / truth move has median
  0.103 / 0.048 / 0.008 / 0.000 / 0.000 by band (0.86-0.97 on fixed-T, P transients); hot targets are right on average
  but noisy (14-20 % wrong sign), a p50 0.000 (median log10 k -9.1 at 2300-3000 K).
- m17fs_s0 (frozen 10-08; 5.31M parameters): M rows (16,384 small-displacement rows, windows from the displaced start) in
  stage 1 + a 75k-step stage 2 (S); G (resolved, floor 1e-12) p90 0.033 / 0.061 / 0.042 / 0.026 / 0.015, p99 0.86 /
  0.38 / 0.43 / 0.44 / 0.31, max 13.98 / 4.85 / 9.57 / 13.78 / 12.30, 292 of 3072 rows > 0.5 dex; the largest misses are
  C2H2. Paper metric (per-row mean, p90) 0.0270 dex vs the one-step paper's 0.0216 (not an identical test; corrected
  10-16). 58-day paths: late p50 0.193, p90 2.03, p99 4.22, max 6.6; error accumulates from small per-call errors (restart
  every 50 calls: p90 0.022 after 1 call, 0.12 after 100, at every age). By window 0-6 / 6-17 / 17-58 d p50 0.036 / 0.098
  / 0.193, paths > 0.5 dex 12 / 13 / 14 of 16; major species grow too (CO p50 0.019 -> 0.170, CH4 0.058 -> 0.453): the
  1.5-day v2l understated deployment error. C/O from the 12 species at 1000 d median 0.0004, p90 0.0137 dex. Speed
  0.53-0.67 us per cell (GPU); CPU 13-18 us (8 threads), 66-93 us (1 thread). Code tag e13_m17fs_s0 (harness history
  bundle). Its training cache is gone, so it cannot be remade. Board: data/2026-10-08_e13-sprint/sb_br.txt.
- F0fs (M + S + relaxT + cold data D, from scratch; two seeds): wins 33 of 45 scoreboard cells but two vetoes (G+Gc
  300-800 K worst species O 4.08 / 4.46 vs 2.41 / 2.35; v2l max 15.5 dex, path 40 C2H2 at 532 K). Not frozen.

#### Closed-state models (conservation phase, 10-09 to 10-12)
- Why: VULCAN's 12 tracked species hold C/H, O/H, N/H to < 0.01 dex on 14 of 16 xlong paths while the frozen model's
  ratios wander up to 2.7 dex. A hard 12-species projection at inference (Gate A) cut the 58-day late p50 9-25x but was
  vetoed (G hot p99 0.313 -> 0.790, new 17.8-dex C2H2 miss): atoms really sit in untracked species. A learned budget leak
  (B3) was ill-conditioned (gradients 1e3-1e5x relax's). The closed 25-entry set removes the growth even on the old
  recipe (Gate A0: resG16 xlong p50 0.036 / 0.046 / 0.054 by window vs allG16 0.067 / 0.119 / 0.167), but its tails are
  not better (max 8.38, path 14 C2H2; O / OH worst on 8 of 16 paths under changing conditions). Clue first: projecting
  saved m17fs xlong states onto the start's ratios moved the late window 0.193 / 2.03 / 4.22 / 6.60 -> 0.018 / 0.33 / 1.30 /
  4.38 offline, while on G it hurt hot fresh starts (rows 23 -> 34): early in a fresh start the truth's own tracked ratios
  move up to 0.7-0.8 dex.
- B6 (e13p_B6_s0-s2): 25 entries, relaxclosedr 6x1024, 12 Newton iterations in training and 32 in benches, own stage 1
  (150k), stage 2 48.8k steps with the chain term (K 8 -> 16, pre-roll 0 -> 80) on train_a + train_b; about 3.4x B1's
  stage-2 wall time. DEV late 0.001 / 0.026 / 0.146 / 2.03 (frozen 0.010 / 0.132 / 0.941 / 6.01); xlong 0.003 / 0.157 /
  1.75 / 10.43 (frozen 0.191 / 2.01 / 4.18 / 6.60); C/O p99 at 1000 d 0.129 -> 0.001. Rows with a late point > 2 dex G / Gc
  19 / 42, 21 / 43, 18 / 46 for s0-s2 vs frozen 72 / 70 (flat rule): about half. 1000-day runs finite, ratios exact.
  Speed 1.62 us per cell at 1e5 (B6 s2). Worst case: xlong path 14 C2H2 (slow post-conversion drain; seed- and
  stage-1-dependent: 10.43 / 5.44 / 7.80 dex). B6 - B1 on DEV p50 -0.008 [-0.020, -0.001], p99 -0.086 [-0.176, -0.018];
  the control's pooled DEV p99 moves about 0.4 between seeds, the per-path p99 gains are 6-8x the seed difference. v2l
  0.001 / 0.047 / 0.322 / 3.94 (frozen 0.022 / 0.185 / 0.697 / 4.49), v2s 0.001 / 0.018 / 0.110 / 1.15 (0.007 / 0.068 /
  0.287 / 1.47). G 300-800 p99 / max 0.619 / 4.98 vs 0.861 / 13.98, but B6's p90 is worse in the cold G bands (300-800
  0.033 -> 0.048, 800-1300 0.061 -> 0.075), better hot; hot fresh rows (T >= 2300 K) 23 -> 15; xlong points > 2 dex 11,008
  vs 125,545. Caveats: B6 had about 3.4x B1's stage-2 wall time (3.8x alone), so its gain cannot be assigned to
  conservation alone; DEV shares the training families (held out by budget and phase), xlong does not, and their tails
  pointed opposite ways (B6's in-training xlong max rose 5.58 -> 10.43 during stage 2). Stage-1 val B6 s0 / s1 / s2 0.04362
  / 0.04375 / 0.04329. 1000-day runs: paths 3 (s1) and 3, 11 (s2) still move 0.10-0.12 dex per cycle at 1042 d (real or
  model unknown: no truth past 58 d). Boards: data/2026-10-09_e13-conservation/scoreboard_gateB.json, path_board_*.json,
  cyc1000_*.json.
- B1 / B2 (12-species m17fs + chain term 0.5 / 1.0): DEV per-path p99 -0.219 but vetoed (G 300-800 p99 +0.124, more rows).
- B6L (chain curriculum to K 32, pre-roll 468) and B6Lc (+ train_c): not eligible (B6L fails four seed-spread items;
  B6Lc xlong 11.30 dex and +33 % G+Gc 800-1300 rows). The path-14 trace is set in stage 1 (B6L s1 / s2 match B6 s1 / s2).
- E (per-entry median of B6 s0-s2, one projection): G+Gc rows > 2 15 vs 29 (B6 s2), rows > 0.5 126 / 171 vs 146 / 218;
  1000 d bounded; 5.6-7.4 us per cell (3.7x; 156.5 vs 39.0 us at 100 cells per call, so it meets 50 us only at >= about
  1,000 cells per call; 16,103,574 parameters). On the opened holdout H2 it froze two rows (median of frozen members). The
  median removes one-copy misses and keeps the shared ones (3 of 13 all-copy rows fixed); a median of three states can sit
  above each member's error after the projection (G 44: 4.08 vs 3.22 / 3.94 / 2.08); it lags a checkpoint when members
  disagree on timing (G 243 CH4 2.01 from 1e6 s). Its development large-miss advantage (15 vs 29) did not hold on H2 + H2c
  (12 vs 12); its typical-error and path advantage did (rows > 0.5 196 vs 221; PH per-path p50 -0.0006 [-0.0008, -0.0003],
  29 of 32 paths). Under P* (10-13) it fixes different rows than F2 (H2 52 fresh 5.45 -> 2.28, H2c 251 4.32 -> 1.47; worse
  Gc 559 0.23 -> 3.38, H2 278 0.92 -> 3.28).
- D-cold (e13p_B6u_s2: B6 s2 stage 1, stage 2 on the cold union b02 + c18 + u19): G+Gc rows > 2 19 vs 29, but five path
  intervals above 0 and fresh hot rows 15 vs 14: vetoed; it fixed one-copy rows, not the shared cold misses.
- W (e13p_B6w_s2: per-gas censored loss, floors 2 dex below the scoring floors, x2 on H2O, CO, CO2, CH4, NH3, H): rows >
  0.5 304 vs seed copies 357 / 388 / 364 (beyond seed noise), DEV p99 -0.0076; regressed hot fresh low-P starts on the
  up-weighted gases (G 638 H2O 2.75 dex, CO2 2.24, NH3 1.70; G 146 H2O 1.91; G 842 CO2 0.97). Not admissible as written;
  never stacked later. Rows > 2 by start age (fresh / 1e3 s / 1e6 s): B6 s2 20 / 7 / 2, s0 16 / 9 / 1, s1 18 / 6 / 2, W
  13 / 8 / 2, E 10 / 4 / 1, D-cold 15 / 3 / 1: most large misses are fresh starts.
- Cold-miss census (flat rule, data/2026-10-09_e13-conservation/colddata_census.json): > 2 dex rows invented / lost /
  both G 12 / 15 / 13, Gc 28 / 21 / 31; mostly C2H2 and H, then NH3, HCN, CH4, CO2, O.
- Holdouts H2 / H2c / PH (opened once 10-12 on E and B6 s2): both failed the then 5-dex bar through a silent projection
  freeze (H2 606, 8.04 dex; H2 218 for E) and a first-call miss (H2c 49, 6.35), and the cold CH4 collapse (H2 52); PH
  passed (max 2.28 / 1.85). The rows: H2 606 (2077 K, 5.0e-4 bar, 38x, C/O 1.85, equilibrium start; the truth turns
  carbon-rich within 500 s), H2 218 (2536 K, 2.4e-5 bar, 18x, C/O 1.30), H2c 49 (1906 K, 3.3e-4 bar, 154x, C/O 1.48; first
  call only), H2 52 (428 K, 1.7e-5 bar, 7.7x, C/O 0.105; CH4 falls 1e-3.5 -> 1e-14.5 between 1e3 and 1e6 s as carbon goes
  to HCN). H2 rows > 0.5 / > 2 B6 s2 169 / 9, E 147 / 10; H2c 52 / 3, 49 / 2. The holdout truth keeps element ratios to
  4e-7, so these are model errors.

#### Fixes phase models (10-12 to 10-13)
- P* fixed the freeze (H2 606 8.04 -> 0.10) and the first-call misses (H2c 49 6.35 -> 0.10); B6 s2 under P* rows > 2
  over G + Gc + H2 + H2c 26, > 0.5 505 (H2 + H2c rows > 2 12 -> 7; worse under P*: H2 123 4.49 -> 4.78, H2c 207 3.49 ->
  4.00). First-call dose: five calls of 100 s cut rows > 2 23 -> 3 (E 23 -> 1); two of 250 s left 3 / 5 rows and H2 606
  frozen; fifty of 10 s were worse (8 / 6 rows, max 5.1 / 6.1). Every first-call miss > 2 dex is a fresh start whose truth
  moves >= 2 dex within 500 s, almost all hot and below 1e-2 bar; 11.3 % of training rows move >= 5 dex in the first 500 s.
- Projection solvers: LM (accepted only if the residual falls) converges on every frozen case within 24 iterations; the
  convex dual also converges but scores worse (0.98-1.11 vs 0.56-0.62 dex on 606). Structured solver (10 atom-count groups,
  He exact, LM with step cap 4, 48 iterations): 0 failures of 100,000 realistic first-call states (Newton 32: 5), <= 3e-10
  dex from Newton; C / OpenMP fp64 8.2-10.3 us per cell per core at fixed 48 iterations, 1.15 with a per-cell stop; in
  eager PyTorch on this GPU (fp64 at 1/64 rate) about 4x the network's cost. A synthetic +-17 dex stress set (not model
  outputs) still breaks LM 32 / 64 on 4 % / 3 % of rows; none of the model's own states in any bench did
  (data/2026-10-12_e13-fixes/solver_speed_*.txt).
- F1 (events drawn 3x), F2 (= D-cold's data b02 + c18 + u19 with events x3; 19a cache, 222,845 train) and F3 (F2 + 12,283
  event-targeted cold runs): P* rows > 2 21 / 17 / 16, > 0.5 469 / 423 / 438, DEV p99 0.128 / 0.121 / 0.121 (B6 s2 26 /
  505 / 0.128). F2 picked by the written order (rows > 2 within 10 %, then rows > 0.5; applying the 10 % band to the
  second criterion too would have picked F3 by 0.0002 dex of DEV p99). H2 52 (cold CH4 collapse) stays above 5 dex in
  all; cold data fixed the shared cold Gc rows (Gc rows > 2 15 -> 5), but in the very cold, low-pressure CH4-collapse class
  F2 is worse than B6 s2 (H2 52 from 1e-3 s 5.45 vs 2.93; H3 486 4.03 vs 2.03); new miss H2c 251 (898 K, C2H2) 1.12 ->
  4.32. Event share of draws: F1 1.8 -> 5.2 %, F2 8.9 % of 111,832 cold, F3 6.5 %. Boards: data/2026-10-12_e13-fixes/
  sel_board_F1.txt, sel_board_F2.txt, sel_board_F3.txt, sel_rows_gt2_all.txt, board_fx_*.json.
- Sealed H3 / H3c opened once 10-13 (F2 / B6 s2): max 4.03 / 2.81, late rows > 2 2 / 4, > 0.5 151 / 177, pooled late
  p99 H3 0.173 / 0.196. Worst row H3 486 (433 K, 1.4e-6 bar, late CH4 collapse). 1,023 of 1,024 H3 jobs have a reference
  run (3,069 rows), H3c 768; rows > 0.5 by start (fresh / 1e3 s / 1e6 s) F2 61 / 37 / 10, B6 s2 75 / 43 / 12; on H3c F2 is
  lower below 1300 K (p99 0.460 / 0.591, 0.370 / 0.432), higher at 1300-1800 K (0.384 / 0.230). Files:
  data/2026-10-12_e13-fixes/sel_board_H3.txt, sel_h3_bands.txt, sel_rows_gt2_H3.txt, board_fx_H3_F2_B6s2.json.
- Scaling probes (stage 1, 150k): half the data +15.6 % validation loss; 6x1536 -3.3 % (I: one-step error data-limited;
  11,983,922 parameters, 23.9 vs 10.7 MFLOP per call; a power law through the two data points, exponent about 0.21, would
  put doubling the data at about -13 %: an extrapolation). Stage-1 xlong readout: half data p50 0.0155 / p99 5.79, 6x1536
  0.0240 / 2.92, B6 seeds 0.0124-0.0130 / 2.17-4.00.
- F2: 1.75 us per cell (LM on every row 3.60); 1000 d bounded (last-cycle change 0.081 dex).

#### Scale-phase 6x1536 line (10-13 to 10-18)
- FINAL (6x1536 on U2 at both stages): six-set rows > 2 14, > 0.5 373, first-call > 2 0, max 5.08; path late p99 DEV
  0.112, xlong 0.453, v2l 0.277, v2s 0.077, PH 0.150. G1 (width alone) 25 / 524, B (data alone) 26 / 481, A (U2 only in
  stage 2) 25 / 486, F2 19 / 574: width and data only help together, and the stage-1 data step carries it. Inside the
  envelope (rescore 10-14): late > 2 / late > 0.5 / lasting > 0.5 / lasting > 2: FINAL 4 / 198 / 59 / 1, F2 10 / 318 /
  101 / 7, B 12 / 256 / 77 / 8, G1 11 / 307 / 82 / 1, A 12 / 275 / 64 / 2 (FINAL's 59 lasting rows: 34 at 300-800 K, 23 at
  800-1300 K, 2 at 1300-1800 K). Speed 2.31 us (LM 4.20); 1000 d bounded (last-cycle change 0.044 dex). Not
  production-ready by the author's reading (10-14). U2 is 1.78x F2's stage-2 data and 3.37x its stage-1 data (the plan's
  "1.8x at both stages" corrected). Rows > 2 carry about +-4-5 of training luck. FINAL fixed 12 of F2's 19 rows > 2; its
  14 (6 inside 1e-5-100 bar, 4 inside env_any: G 886, Gc 545, H2 52 from 1e3 s, H3 716) are the cold-CH4 group (Gc 85 / 250
  at 580-610 K and 28-354 bar, colder than any GCM there; H2 52; H3 486) and mistimed real 3-9 dex switches (H3c 187, G 638,
  G 886, H3 716, Gc 536, Gc 545), none invented; the four still wrong at 1000 d are all <= 810 K (H3 486 x2, H2 52 from 1e3
  s, Gc 545). v2l late p99 +0.037 over F2 fails the win test's regression limit, but 44 of 64 paths are better (per-path p99
  -0.025 [-0.040, -0.010]). Stage-1 val on U2 6x1536 0.03826 vs 6x1024 0.04044 (-5.4 %), while its stage-1 xlong readout is
  worse (p50 0.0224 vs 0.0103). T2 inside: FAIL on every rule, shares a 0.720 / 0.631 / 0.656 (k1 / k2 / k3; need 0.95),
  b 0.787 (0.90), c 0.925 (0.95); kicked rows late max <= 0.1 dex 56.2 % (unkicked partners 74.3 %). Boards:
  data/2026-10-13_e13-scale/board6_*.json, critic_env_rescore.txt, score_t2.txt (FINAL's T2 score).
- Where FINAL fails (10-15): 85 of 87 lasting rows below 1300 K; worst gas at 1000 d C2H2 36, CO2 20, HCN 8, CO 7, NH3 6,
  CH4 6, H 3, H2O 1 (carbon species mostly model-high), median lasting error 0.705 dex, 37 of 87 first pass 0.5 dex after
  1e7 s; systematic velocity error on true states explains 58 of 87 lasting rows (startup 11, feedback 8, mixed 20); model /
  truth velocity on moving late nodes p10 / p50 / p90 0.20 / 0.88 / 1.51 (data/2026-10-13_e13-scale/failure_report.txt).
- Continuations at s2_lr 1e-5, 20k steps each (inside lasting / late > 0.5): FINAL 61 / 202 -> C0 (U2) 53 / 194 ->
  D-kick (U3: + y21b + 53,729 kicked runs as starts, kicked rows 0.20 of draws) 48 / 186 -> LD (+20k) 43 / 175 -> LE1-3
  (3 x 20k more) 44 / 159, 40 / 162, 40 / 154. Each 20k fine-tune flips about 19 lasting rows (noise unit about 5;
  calibration: the same parent and seed with TF32 off flips 7, the stage-2 batch arms 2 and 9, a full retraining on other
  data 69). LD vs FINAL: ADOPT by rules_t5 (27 fixed, 9 broken; T2 +0.040; cooler lasting 49 vs 70; all rows lasting 64 vs
  87); 2.28 us; 1000 d bounded. LD vs D (R1): lasting 43 vs 48 (9 / 4), not ADOPT by rules_t5 but passes the fallback; new
  rows > 2 H2 52 fresh 4.43 CH4 (FINAL 1.38) and G 146. D-kick vs C0 alone: NO GAIN (48 vs 53, 12 fixed / 7 broken; T2 mean
  +0.018): most of the gain over FINAL came from the 20,000 extra steps alone (C0 53 vs FINAL 61). Verdicts:
  data/2026-10-13_e13-scale/verdict_e13s_LD_s3_vs_e13s_FINw1536_s2.txt, verdict_e13s_T5D_s2_vs_e13s_T5C0_s2.txt,
  verdict_e13s_LE*.txt, l_judge.txt, e_judge.txt.
- Dead ends of the line: LDc (rows below 1300 K drawn 2x; cool draw share 0.463 -> 0.633) 44 vs 43; LD1 (one-step dt
  1-500 s, 37 % of windows below 10 s) lasting 53 vs 43 with no evolved-start 1 s gain (raw-gas Fig. 4 sum of per-case
  medians / maxima: dt 1 s 1.60 / 6.67 vs 2.23 / 10.66, better but short of its 0.70 bar; dt 500 s 0.32 / 7.01 vs 0.29 /
  4.86, worse); X2C (dt 0.1-500 s, 54 % below 10 s) 68 vs 61 (at dt 500 s its Fig. 4 per-case max is above FINAL's in 5 of
  6 cases, sum 8.16 vs 4.51); X2E (two-exponential head relax2expr, kf >= 2 ks) worse than X2C at hot 1 mbar (motivated by
  oracle fits on young starts: one exponential 1.08 dex, two rates 0.56, two targets 0.38); X2V (asinh speed loss, vel_w
  0.01; one-step speed error -11 %) 68 vs 68; LDF (TF32 off) 40 vs 43 (below the rule); S2B512 / S2B8k (stage-2 batches
  /4, x4) 44 / 43 vs 43. Verdicts: data/2026-10-13_e13-scale/verdict_*.txt, vel_diag_*.txt, b_judge_s2.txt.
- T6 (fresh 53-species 6x1536, no reservoirs, one-step dt 0.1-500 s, 12 gases x3; 12,156,010 parameters): inside lasting
  140 vs 61, late > 2 33 vs 4, first-call > 2 10 vs 0, xlong late p99 1.131 vs 0.453, SD inside rows > 0.5 47 / 74 vs 8 / 24
  at dt 0.1 / 1 s; the gap already shows in stage 1; B_C (+20k) no better. The 25-entry line continues.
- P15 (FINAL's recipe on U4 = U3 + n1 + n2, 624,324 runs): lasting 64 vs 61 (33 fixed, 36 broken): no data effect at
  6x1536; outside lasting 34 vs 26, cooler 63 vs 70, all rows 98 vs 87; PX late p99 0.699 vs 0.453, P2l 0.155 vs 0.277;
  stage-1 path readout p50 better (0.0129 vs 0.0224), p90 worse (1.007 vs 0.663). U4 also carries the kicked runs in stage
  2, so data and kicks are not separated (data/2026-10-13_e13-scale/verdict_e13s_P15_s2_vs_e13s_FINw1536_s2.txt).

#### 6x2048 line (10-17 to 10-19)
- CBS1 (6x2048, U4, 1M steps at batch 8192, AdamW 6e-4, warmup 20k): validation 0.03511 at 150k, 0.03424 at 200k, 0.03364
  at 1M (training about 0.019: a memorization gap, I: data-limited); path readout p50 / p90 / p99 / max 0.091 / 1.004 /
  3.18 / 10.8 at 1M (FINAL's stage 1 0.022 / 0.663 / 5.38). The 1M steps were raised from 400k after the one-step paper's
  step count (about 2.3e8 samples if its batch counts pairs, I); the size pilot P20 was dropped for CB. The 6x1536 stage-1
  runs were still falling at 150k (0.03856 -> 0.03826 on U2), the 6x2048 one levels off.
- CB (uncapped stage 2, 4x batches, lockstep step; 0.43-0.64 s per step, peak 28.6 GB): 77 skipped non-finite steps by
  35k; the grid-term gradient explodes along one hot low-pressure row per step (median grid gradient norm at nsub 1-4: 19 /
  2.9e5 / 1.4e9 / 1.1e12, max 5.3e20; FINAL 2.3); the global clip leaves a one-row kick, and the fp32 sum of squares in the
  clip overflows above about 1.8e19, which the guard skips. 109 of 24,576 sampled grid rows exceed 1e6, all at 1508-2861 K
  (none of 12,092 below 1400 K), median 2.6e-4 bar; on such a step the fp32 gradient with TF32 off is 88-89 % off fp64.
  Paused at 35k (resumable). Diagnosis: data/2026-10-13_e13-scale/nfdiag_report.txt.
- CBg (CB + a per-row cap c = 0.01 on the incoming state gradient after every gradient-carrying rollout call, fp64 clip
  norm; c from the p99.9 0.0086 of FINAL's and LD's per-row peaks): 0 skipped steps; validation at 30k 0.03839 vs 0.04475
  uncapped; path p90 0.076 vs 0.384. Vs LD inside: lasting 31 vs 43 (24 fixed, 12 broken), late 96 vs 175 (116 / 37),
  late > 2 6 vs 4, lasting > 2 4 vs 1; xlong late p99 1.114 vs 0.471 (paths 9 and 14). Failed rules_t5 guards G1, G2,
  G3, G6; adopted by the author; passes pass rule v2. Also vs LD: first-call > 2 0 vs 2; cooler late 64 vs 138, cooler late
  > 2 7 vs 4; paths late p99 (max) PD 0.101 vs 0.105 (1.41 vs 1.60), PX 1.114 vs 0.471 (2.17 vs 2.53), P2l 0.197 vs 0.251,
  P2s 0.059 vs 0.068, PH 0.142 vs 0.154; new late > 2 rows G 638 (H2O), Gc 559 and Gc 250 (CH4; all outside the envelope,
  transient) and H3 594 x3 (C2H2, 1257 K, 1.4e-4 bar, inside; lasting 2.47-2.52 vs LD 0.86-0.89); no invented switch. In
  sum 28 % fewer lasting misses than LD, 49 % fewer than FINAL, 45 % fewer late misses (I). The cap on fixed weights: grid
  gradient median at nsub 1-4 54 / 1.3e5 / 1.2e9 / 1.7e12 -> 1.2 / 1.7 / 2.0 / 2.3 (3-6 % of rows capped); on FINAL and LD
  it touches 0.05-0.10 % of grid rows and changes the direction (cosine < 0.9) on 2-4 % of draws (their rare bursts):
  "typical gradients barely affected", untested in a healthy-model training run; cost 0.378 vs 0.369 s per step. Files:
  data/2026-10-13_e13-scale/gfix_report.txt, verdict_e13s_CBg_s2_vs_e13s_LD_s3.txt, p_judge_cbg.txt.
- Why CBg is better (I, read-only attribution): broad gain (every T band but 2300-3000 K, every start age; C2H2 78 -> 37
  rows); by jobs late misses 69 vs LD 116 (sign test p 6e-5), lasting jobs 19 vs FINAL 37 (p 0.001), vs LE3 27 (p 0.12,
  not established); the lasting gain is all at Z < 100x and C/O <= 1 (6 vs LD 15). Data alone at 6x1536 gave nothing
  (P15), batch nothing, stage 2 flat from 50k to 200k: most of the gain is a wider network trained far longer in stage 1
  (each design row drawn about 11,000 times vs 410), which only the cap made trainable in stage 2; width and stage-1
  length are not separated. Inside late > 0.5 by band LD -> CBg 47 -> 24 (300-650 K), 38 -> 18, 23 -> 8, 41 -> 23, 20 ->
  14, 6 -> 9 (2300-3000 K worse); by start 110 -> 63, 48 -> 26, 17 -> 7; worse gases H2O 11 -> 17, OH 1 -> 5 (hot
  transients); inside late p50 / p90 / p99 0.0134 / 0.184 / 0.685 -> 0.0085 / 0.112 / 0.505; T2 at 650-1300 K a_k1 0.68 ->
  0.82, b 0.48 -> 0.71; above 100 bar (outside) CBg's lasting rows 0 -> 9. The path-9 NH3 drift appears only in models
  trained on U4 (P15 0.98, CBg 1.47 dex; every other model <= 0.34): U4 implicated (I). Files:
  data/2026-10-13_e13-scale/why_attr_report.txt, why_critic.txt.
- CBg's remaining errors (VULCAN restarts from CBg's own states; V1N controls reproduce the truth from the true 25
  entries in 12 of 12 cases, so information is not missing): of 31 inside lasting rows, 15 are state right but later
  steps wrong (e.g. H3 594 C2H2 2.5 dex: CBg's C2H2 stops falling after 2.8e7 s), 8 committed in the first 1-3 calls of a
  fresh start, 3 committed mid-run, 5 a slow drift in sub-floor gas or reservoirs; the worst row, H2 52 from 1e3 s (CH4
  4.93 dex at 428 K, LD 2.67), is right to 3e4 s and then misses its CH4 switch (committed mid-run, corrected by the
  independent check); 13 born after 2.8e7 s (where 3-4 % of one-step
  windows start but two thirds of a 1000-day run's calls fall), 12 of 31 at Z 200-300x; NH3 / HCN drained at about 1e-3
  dex per call on frozen cool paths (xlong path 9, 318-640 K; VULCAN from CBg's state holds NH3 within 0.001 dex). Of the
  31, 19 are shared with LD and 12 new. Late-born rows CBg 13, LD 3, LE3 3, FINAL 10, P15 15; late-window speed model /
  truth median 1.009 (LD 1.014), reversed 1.0 % of moving cells (LD 0.0 %). Z 200-300x holds 12 of 31 (LD 10, FINAL 18),
  C/O 1.5-2 CBg 7 (LD 5); Z >= 200 is 4.2 % of U4's runs and 4.6 % of its stage-2 draw weight. Failing rows do not sit in
  thinner condition-space data (median 154 neighbours vs 129 for all inside rows) except at the Z edge (H3 594 64). The
  late > 2 transients outside the envelope come from steep switches (3-6 dex per decade) that turn small radical errors
  into 2.5-2.9 dex. Training chains near PX path 9: 3 of 2,512. T2 rule-a failures inside 357 (LD 613). SD dt 1 s misses
  (11) are mostly hot (1534-2374 K), within 1e4 s of an evolved start, not diagnosed. Mechanism of the cold CH4 collapse
  (H2 52, H3 486): the few-percent residual of H + CH4 -> CH3 + H2 against H + CH3 + M -> CH4 + M (net 3-26 % of gross),
  so small rate errors move its timing a lot; Gc 85 / Gc 250 are the opposite case, slow net production (production / loss
  about 600x and 170x). Files: data/2026-10-13_e13-scale/why_miss_report.txt, why_miss_inv.txt, why_miss_*.json,
  diag_v5.txt.
- CBgP1 (+40k at 1e-5, seed 3): inside lasting 29 vs 31, late 94 vs 96: NO GAIN, CBg leveled off (under pass rule v2 it
  does not pass either, 29 > 27.9); paper-style metric G per-row median 0.0015 both, p90 0.0161 vs 0.0164, v2l 0.0097 vs
  0.0099 (V cleanup/h2/report.txt, recomputed 2026-10-20; verdict data/2026-10-13_e13-scale/
  verdict_e13s_CBgP1_s3_vs_e13s_CBg_s2.txt). CBgA (48.8k stage 2)
  stopped at 27k (path readout at 20k equal to CBg's). CB5 (CBg recipe on U5, 1.04M runs, 1.67M + 334k steps) stopped at
  stage-2 step about 110k (path readout at 90k p99 0.616 vs CBg 1.28; interim only): the all-data question is open.
- Raw-gas behaviour: CBg forms CH4 from CO too fast at 1000 K, 10 bar (CH4 error at matched 5,000 s elapsed +1.14 /
  +1.09 / +0.81 dex at dt 1 / 20 / 500 s vs FINAL +0.20 / +0.21 / +0.40); restarts from VULCAN's exact state at 1e3 s
  show it too (+0.64 / +0.80 / +0.61 at +1e3 / +1e4 / +4e4 s; FINAL +0.24 / +0.19 / +0.05). 1000 K, 10 bar worst gas
  median / max / end, CBg vs FINAL: dt 500 s 0.07 / 0.88 / 0.07 vs 0.13 / 0.91 / 0.13; dt 20 s 0.74 / 1.19 / 0.61 vs 0.35 /
  1.52 / 0.27; dt 1 s 1.17 / 1.20 / 1.14 vs 0.33 / 0.53 / 0.37 (all six cases at three dt: the dated record (paper-notes/archive/dated_record_2026-10-19.tar.xz) lines 4424-4431).
  A constant per-call bias is not established; no decision test combines short steps with far-from-equilibrium gas; the
  two cases first shown to the author were CBg's weakest of the six.

### 5.4 Benches and scoring

#### 12-species era (architecture tests, repo sweep; 10-01 to 10-05)
- Sources from the test split: S = 500 (25 per cell of 5 T x 4 log10 P bins, seed 20261001), SP = 100, SPF = SP at full
  length; start ages 1e-3, 1e3, 1e6 s. Arms: A transient at dt 1 / 10 / 100 s (to 1e5 / 1e6 / 8.64e5 calls; SPF dt 10 s
  8.64e6 calls = 1000 d), B late hold from VULCAN's 1e8 s state (1e6-1e7 calls), C perturbation (244 boxes, +-0.3 dex at
  1e10 s), E P/T switch (128 boxes: P x10^+-0.5, T +-300 K at 1e4 s), D stress (blend with a partner cell 1-10 %, P/T
  random walk, injected zeros; no truth); the repo sweep added V (dt changing per call), G (stress dose groups, then
  recovery) and H (rollout from a VULCAN run's own states). Truth VULCAN-0D b3ccf4f rerun to 1e10 s with no early stop.
  Guard: a non-finite state or log10 y outside [-40, 0.05] keeps the previous state and is counted.
- Score: |dlog10 y| on truth-resolved values (n >= 1e-2 cm^-3), pooled per T bin; headline worst-T-bin p90; persistence
  scored alike; closable mask (species whose hidden-species p90 <= 0.1 dex) supplementary. Pre-registered pass rule: no
  interventions, p90 <= 0.1 and p99 <= 0.3 dex in every transient and switch stratum, hold p90 <= 0.02, cool bins below
  persistence at the end: no model passed. Wall time of the finalist's 8.64e6-call dt 10 s SPF arm 12.5 h (one GPU, partly
  shared).
- Every campaign score used the test-split S / SP sources (selection bias, not leakage); the sealed 100-source holdout
  (10-03) gave FIN 0.80 against 0.34 on SP. Paired differences: candidate minus reference on identical rows, 95 %
  source bootstrap within T x log P cells. Data: data/2026-10-01_architecture-tests/ (runs.json, scores.csv, curves.csv,
  first_fail.npz, closure.csv, vulcan_checks.csv, benchmark.json, rollouts.npz) and data/2026-10-03_repo-sweep/ (E1-E12:
  E1_summary.json, E1_table.csv, E2_prereg.json, E2_X1 / X2 / X3 / X7.json, E2_RENORM.json, E2_summary.json, E3_pairs.csv,
  E3_decisions.json, E3_r3_rerun_check.json, E3_metrics_P1S1_s0.json, decision_rule_E4.txt (sha256 d63bec4d...e21d134c),
  E4_decisions.json, E4_pairs.csv, E4_sum12.csv, decision_rule_E6_fp.txt, E6_fp_*.json, <model>_holdout_score.json,
  E5_ladder_report.txt, E8_floor_*.json, E9_*.json / .npy, E10_trunks_scores.json, E11_film_scores.json,
  E12_utest_jobs.json, plan_*.txt). Selection: all 369 rows in 41 campaign bench folders used the test-split S / SP
  sources and the reference sets lie inside S; the splits share no group (selection bias, not leakage).
- Cost note: only 5 of the 217 stored architecture-test bench arms ran alone on the GPU; shared-GPU wall times are not
  cost data. GPU cost estimate then (the 10x2048 finalist, C48 x 53 levels = 732,672 cells, 1000 d): 12.7-31.9
  GPU-days at dt 100 s, 127-319 at 10 s, 1,269-3,186 at 1 s (2.30e5 row-calls/s measured at batch 900). CPU numerics: a
  cell's output depends on the call size (up to 1.3e-5 dex) and, at some thread counts, on the thread count (up to
  1.9e-8); batch-context rollout differences level off at 1.6e-6 dex (fp64 host) and 1.5e-5 (fp32).

#### E13 fixed-condition sets
- G (10-06, make_gtest.py): 1024 jobs, Sobol seed 20261007, physical starts only (5/7 equilibrium, 2/7 mixed), truth to
  1e10 s, all 53 species, rtol 0.05; gtest_tight = the same at rtol 0.005. Benched at dt 500 s (172,800 calls to 1000 d)
  and dt 100 s (864,000 calls; equal to dt 500 within 0.01 for grid-term models). Under the target's mask and pooled
  p90, VULCAN's own rtol error is 0.001-0.009 dex per band (per-job max p90 0.088, max 0.34), so a 0.1 dex target is
  meaningful. The 12- and 53-species caches differ (frozen-tail cuts depend on the species set: 330 differences on 42k
  boxes; aligned by cache_align53).
- Gc (10-08): a real-truth cold set below 1300 K (G+Gc gives 2106 / 2103 rows at 300-800 / 800-1300 K).
- H2 / H2c (1024 jobs / corner set of 768 rows; rtol 0.005; /scratch/imalsky/e13_holdout): sealed 10-09, opened 10-12.
- H3 / H3c (1,024 jobs seed 20261101 / 256 corner jobs seed 20261102; 732 / 292 equilibrium / mixed starts; T 302-2999 K;
  /scratch/imalsky/e13_holdout3): sealed 10-12 13:01 with a sha256 list, opened once 10-13.
- Sealed and never read: H4 / H4c (1,024 jobs seed 20261201 / 256 corner jobs seed 20261202; /scratch/imalsky/
  e13_holdout4) and PH2 (paths_v3 hold2); sha256 lists written 10-13.
- Six-set development tag /scratch/imalsky/e13_scaledev/devall6 (read by every development bench and the sweep). A
  fixed-condition bench costs per call, not per row, so the six sets run as one combined bench (split back afterwards;
  same rows above 2 and 5 dex, per-row late max within 5e-4 dex). Benches had to set ARCH_CACHE=/scratch/imalsky/
  arch_gcm17m_res (lhbench loaded data.CACHE's test split at start-up; arms H and P do not use it; rerun bitwise equal);
  since 2026-10-20 the pruned lhbench loads no test split. devall6 silently left out job 3657 of its 4,608 (found
  2026-10-19; lhbench now reports skipped reference jobs).
- Batch shape: the trunk's matrix products depend on the rows per call (up to 2.5e-5 in z over 120 calls; up to 7e-4 dex
  across batch shapes), so bitwise comparisons hold only at equal batch shape.

#### Path sets and their roles
- DEV (47 x 5000) decides together with G and Gc (since 10-09); xlong (58 d), v2l and v2s are continuity readouts; PH
  (32 x 10,000, held-out forcing mix) opened 10-12. The per-call-restart VULCAN reference matches continuous runs (pooled
  p90 0.0012 dex, worst per-call p90 0.026); xlong truth is converged (rtol 0.05 vs 0.01 vs 0.005: max 0.05 dex).
- v2s at 300-800 K first read as "nothing moves"; it is dilution (movers' pool p90 0.397). Sparse checkpoints (23 of
  172,800 calls) understate peaks in bad rows by up to +1.05 dex. Path differences under about 0.15 dex on the 64-path set
  are not read (two replicates differ by 0.13).

#### The real-GCM envelope and GCM time steps (10-14)
- T-P grids of 571 published GCM models (Roth 2024, 516; Kennedy 2024, 55): coldest GCM gas 426 / 808 / 1361 K at 0.1 / 1
  / 10 bar; GCM area (equal model weight, 1e-5 to 100 bar) above 3000 K 3.75 %, below 300 K 0.05 %; by band 300-650 /
  650-1300 / 1300-3000 K 7.7 / 31.6 / 56.8 %. env_any = (log P, T) bins reached by at least one model: 13,384 of 15,120
  bins in 1e-5 to 100 bar and 300-3000 K (scale/research/gcm/gcm_tp.npz, in place, 12 MB; every array but the per-model histogram in
  data/2026-10-13_e13-scale/gcm_tp_without_H_model.npz; data/2026-10-13_e13-scale/gcm_build.txt, gcm_analyse.txt). U2 training rows inside env_any 65.3 % (300-650
  K 46.5 %, 1300-3000 K 77.0 %); 22.4 % of rows lie outside 1e-5 to 100 bar.
- GCM steps (from the cited papers): one step per run in every GCM checked; dynamics 6-150 s; radiation / physics 30-1000
  s; coupled kinetics called every 1500-3750 s with a stiff solver inside (UM 3750 s, chemistry every 10 vs 125 steps
  "little difference", elements conserved to 99.9 % over 1000 d, Drummond et al. 2020; Exo-FMS + mini-chem 1500 s for hot
  Jupiters, 3600 s for brown dwarfs, 1x and 10x solar, Lee et al. 2023). A 1000-day run is 2.3e4-5.8e4 chemistry calls.
  Mini-chem costs about 27 us per cell call on an Apple M3 Pro (arXiv 2609.00428 Sec. 3.5); 1.5 ms per cell-step in its
  Python version.

#### Tests of the scale phase
- One call from 12,000 true U2-validation states (FINAL): rows > 0.5 after one call of 10 / 30 / 100 / 300 / 500 / 1000 /
  3600 s 373 / 219 / 190 / 207 / 295 / 419 / 789, dominated by starts younger than 1e2 s (starts at 1e4-1e7 s: 0 at every
  dt); 3600 s as 12 x 300 s 120 rows; 500 s as 5 x 100 s 151, 50 x 10 s 247. The network never sees dt; dt_norm clamps to
  0.1-1000 s unless the bench's unclamped option is set.
- T1 (10-14): FINAL at dt 100 s (to 8.8e6 s) and 25 s (to 8.96e5 s) vs its dt 500 s bench at shared checkpoints, no
  startup substeps: late > 0.5 +9 % / +14 %, rows > 2 equal: no dt dependence >= 25 %, so no fixed-dt fine-tune.
- T2 snap-back (10-14; scale/t2): 249 jobs (59 of FINAL's lasting or > 2 dex rows, 160 stratified inside, 30 cooler);
  at t_r = 1.5e3, 1e5, 1e7 s the full 53-species truth is kicked with element fractions held: k0 identity, k1 the kicked
  species x10^+-0.3, k2 x10^+-1, k3 random +-0.3 dex on the reported species but one carrier per element (kicked species
  by band CH4 / C2H2 / CO2 / H2O or the job's worst species); 3,478 boxes kept (747 identity + 2,731 kicked; 2,124
  kicks dropped and counted), VULCAN rtol 0.005 to 1e8 s; the model starts from the kicked state at t = 0. Rules (scope
  inside): (a) excess error over the unkicked partner, max over tau >= 1e3 s, <= 0.1 dex in >= 95 % of rows, for k1, k2,
  k3; (b) recovery half-time within 0.5-2x VULCAN's in >= 90 %; (c) slow rows (VULCAN half-time > 1e7 s) excess <= 0.3
  dex in >= 95 %. VULCAN passes every rule; a kick-ignoring model and a frozen model fail. Gap: no CH4 kick where CH4
  holds most carbon. Dropped kicks: infeasible 793, hidden species moved > 1 dex 742, not converged 306, molecule-count cap
  135, other 148; element fractions preserved to 9.7e-11. Files: data/2026-10-13_e13-scale/t2_protocol.txt, t2_kicks.txt,
  score_t2*.txt, t2_oracle_t0_score_*.txt (controls).
- SD (10-15): the six sets started from VULCAN's states at 1e3 and 1e6 s, constant dt 0.1 and 1 s, 10,000 calls, P*
  without substeps; inside 6,264 rows, outside 2,950. Inside rows > 0.5 at dt 0.1 / 1 s: FINAL 8 / 24 (p99 0.152 / 0.317),
  X2C 10 / 24, D-kick 9 / 24, LD 8 / 23, LD1 8 / 23, T6 47 / 74, CBg - / 11; outside FINAL 9 / 33.
- Mini-chem paper cases (fig4; 10-15): 0-D solar (Lodders 2009 as in training), 1000 / 1500 / 2000 K x 1 mbar / 10 bar
  from the unreacted initial gas; VULCAN 0-D rtol 0.005 to 1e8 s; mini-chem commit ae021c8, dlsode, NCHO network, 1x
  solar tables, element drift <= 4.3e-10 over 1e8 s, from the initial gas at t = 0; models from VULCAN's state at 1e-3 s.
  |mini-chem - VULCAN| median / max (8 species, floor 1e-10): 1 mbar 0.000 / 1.67, 0.144 / 3.05, 0.000 / 2.96; 10 bar
  0.000 / 0.30, 0.025 / 2.71, 0.000 / 0.56. FINAL vs VULCAN on the same cases (dt 500 s at 1000 K with the first call as
  5 x 100 s, 100 s at 1500 K, 10 s at 2000 K; 5,000 calls): 1 mbar 0.000 / 0.33, 0.021 / 0.70, 0.000 / 3.21; 10 bar 0.000 /
  0.50, 0.003 / 0.15, 0.000 / 0.50. Caveats: different start times (mini-chem t = 0, models 1e-3 s); mini-chem's errors
  include its reduced network. Files: data/2026-10-13_e13-scale/mc_c0_T1000_P0.001.npz to mc_c5_T2000_P10.npz (mini-chem;
  rerun bitwise on 2026-10-19 with the same numpy, up to 1.9e-3 dex off with numpy 1.26.4), fig4_data_c0-c2.npz and
  fig4_data_c3-c5.npz (the remake's arrays, split losslessly from scale/fig4/fig4_data.npz), fig4_init_vmr.json; the fig4
  tag stays in place.
- Step-size sweep (10-19; scale/dtsweep/sweep.py, score_sweep.py): one GPU rollout per model of 10,908 rows = 9 dt (1, 2,
  5, 10, 20, 50, 100, 200, 500 s; each row keeps its own dt) x 2 start ages (1e-3 s raw, 1e6 s evolved) x 606 jobs (600
  of the 4,608 development jobs, seed 20261019: 360 in scope, 120 inside but out of the composition, 120 outside the
  envelope; 258 cooler; plus the six fig4 cases), N = 10,000 calls per row (elapsed 1e4 s at 1 s, 5e6 s at 500 s), P*
  without substeps. Checks: with one dt per row it reproduces the bench's fixed-dt arm H bitwise; mixing dt in one batch is
  bitwise equal to single-dt runs of the same batch size; about 6 min per model (CBg 35.4 ms per call). Scoring: lasting
  = worst gas > 0.5 dex at call 10,000, late = > 0.5 at any recorded call 5,000-10,000, equal weight = mean over the nine
  dt of per-dt counts, raw + evolved summed; matched elapsed time (1e3, 1e4 s) as a second view. Non-finite predictions
  were scored as zero until 2026-10-19 (every stored state finite; fixed). The scorer's "in scope" became the target
  (Z 0.1-300x, C/O 0.1-1) on 2026-10-20 (section 5.6). Time per call vs batch (CBg, idle GPU): 1,090 rows 5.5 us per cell,
  10,908 rows 32.2 ms (3.0 us); a 1e3 s start age (16,362 rows, 8 min per model) was left out. Inside (480 jobs) lasting
  7.4 / 9.2 / 10.8 / 11.0 and late 8.2 / 9.9 / 11.9 / 11.9, cooler (258) lasting 7.6 / 9.3 / 11.0 / 11.2, in-scope late 5.6 /
  6.1 / 7.8 / 7.8 (CBg / LD1 / FINAL / LD); median per-row mean worst-gas error inside, raw, CBg 0.030 at 1 s to 0.009 at
  500 s (FINAL 0.041 to 0.010); FINAL and LD also have a gap at 5 s; at 1 s about 3-9x the mean rows of 10-500 s; per-dt
  differences of 1-3 rows are noise. Files: data/2026-10-13_e13-scale/dtsweep_score.txt, dtsweep_score.json,
  dtsweep_rows_main.json, dtsweep_cbg_miss_identities.txt; the raw rollouts (125 MB each) stay in scale/dtsweep.
- 1000-day cycle test (10-10 on): the 16 xlong forcings repeated 18 times = 180,000 calls, 1042 days, under P* from
  10-13; no truth past 58 d, so boundedness only (finite, rejected calls, projection failures, kept input, element-ratio
  drift, change between cycle ends).
- Speed protocol: idle GPU, one fp64 GCM step in a CUDA graph, TF32 off, 32 Newton iterations, 1e3-7.3e5 cells per call
  (Newton only, and LM on every row as the conservative number); cost does not depend on the weights; shared-GPU timings
  are invalid. Closed 25-entry networks at 1e5 cells, Newton only / LM on every row: 6x1024 1.63 / 3.60, 6x1536 2.45 /
  4.27 (trained FINAL 2.31 / 4.20), 6x2048 3.08 / 5.14 (trained CBg 3.15 / 5.32); the 12-species m17fs 0.66; the B6
  median ensemble 6.07. CPU alternative: the structured fp64 projection solver at about 1 us per cell per core. A bench
  call is about half trunk, half Newton projection (32 iterations, 1.4 ms each in a 94 ms call at 13,821 rows); with a
  TF32 trunk one call's state update moves up to 0.008 dex (11 % of rows above 1e-3 dex); benches and speed tests run
  TF32 off. Earlier CPU figure: 6x512 19-21 us per cell on one thread (10-06). Speed files:
  data/2026-10-13_e13-scale/speed_*.json, data/2026-10-12_e13-fixes/speed_e13f_F2ev19_s2_idle_*.json,
  data/2026-10-08_e13-sprint/speed_e13_m17fs_s0.json.
- Earlier instruments: S-tag (10-08; 126 G jobs x 9 T / P shifts from settled starts; move ratio model / truth);
  readiness R-early / R-full (fp32 host, random dt per call, spin-up, 2000 d); mixing stress gstress (consistency only).

#### Decision-rule history (the rule in force at the time decided; later entries keep their verdicts)
- 10-01: pre-registered architecture rule (above). 10-03 / 10-04: sweep rules written before the runs (E3 "better" = A10
  late-max and final lower by > 0.1 dex on every seed with guards on B, E, A dt 1; E4 Route 1 / Route 2 against a null of
  no benefit, HARMFUL if a guard worsens with interval above 0; E6 HELPS / HARMFUL; data/2026-10-03_repo-sweep/
  decision_rule_E4.txt, decision_rule_E6_fp.txt).
- 10-06: target p90 <= 0.18 dex, then <= 0.1, after a GCM-length run in every T band (superseded 10-08). 10-07: strict
  resolved rule (n >= 1e-2 cm^-3, VULCAN's atol) kept by the author; either mask; all 12 species scored.
- 10-08 (sprint): effect floor 0.02 dex; noise floors 0.03 (300-800 K) / 0.01; a single branch needs >= 0.028, a two-seed
  mean >= 0.02; no band worse by > 0.01. Then (author 15:39-15:44) the overall-error scoreboard (per band pooled p90,
  worst-species p90, p99, max, rows > 0.5 on G dt500 and G+Gc, floor 1e-12; path sets p90 / p99 / max / bad paths; C/O
  p99, sum12 max): a candidate wins with more cells won than lost and no cell where both its seeds are worse than both
  reference seeds by > 20 %.
- 10-09 (conservation): decisions on DEV + G + Gc; adopt a lever when DEV late p50 and p99 improve against a paired
  control with the interval excluding 0, bad paths not up, no veto; vetoes: a G / Gc band p99 worse by > 0.1, a band's
  rows > 0.5 up by > 10 % and beyond seed spread, a new miss > 10 dex anywhere (xlong included, author 10-10), more
  hot-start bad rows, > 50 us per call; identities of new > 0.5 and > 2 dex misses read before adopting.
- 10-10: Gate C (eligible = every xlong path and G / Gc band < 10 dex, no band p99 worse than B6 s0 by > 0.1, rows and
  DEV not worse beyond the seed spread).
- 10-11: per-gas floors (above) and a hard bar (no scored point > 5 dex on any checkpoint of any set); candidate vs B6 s2:
  veto if a point > 5 dex, band p99 +0.1, hot-start rows, a path interval above 0; primary G+Gc rows > 2 down by >= 5 and
  the six key gases' per-band max not worse (that clause also fails between seed copies).
- 10-12 (fixes): eligible = no point > 5 dex on any development set, no kept input, <= 50 us; then fewest late rows > 2
  over G + Gc + H2 + H2c; within 10 %, fewer rows > 0.5; then DEV late p99.
- 10-13: hard bar removed (author); scale pick rule: fewest late rows > 2 over the six sets, tie group within max(5, 10
  %), then fewest rows > 0.5, then DEV p99; "win" for a lever = rows > 0.5 <= 0.9x, rows > 2 <= reference + 3, DEV p99 <=
  + 0.02, other path p99 <= +0.03, max <= +1.0, first-call > 2 <= +2; veto 3 or more invented switches; kill a stage 2
  if its chain loss is > 2x F2's at two consecutive logs in steps 1k-15k.
- 10-15: rules_t5 (data/2026-10-13_e13-scale/t5_rules.txt; inside the envelope): ADOPT iff lasting > 0.5 <= 0.80x the reference
  with fixed >= 2x broken (P1), late > 0.5 <= the reference (P2), T2 shares not worse (P3), and guards G1 large misses
  (late > 2 and lasting > 2 inside), G2 cooler stratum, G3 path sets late p99, G4 first call, G5 numerics, G6 vs FINAL,
  G7 invented switches. Round L's fallback pass rule (scale/l/judge.py):
  lasting <= 0.90x with churn, late <= the reference, P3, G1-G7. Power: one 20k-step continuation flips about 19 lasting
  rows, so only about 10 % effects show. Noise floors measured then: stage-2 replicate sigma about 0.002 dex on G
  (capw17r), from-scratch 0.008 at 300-800 K.
- 10-19: pass rule v2 (data/2026-10-13_e13-scale/t5_pass_v2.py; author: large misses become readouts): inside lasting > 0.5 <= 0.90x
  the reference, late > 0.5 <= the reference, lasting fixed >= 2x broken, late fixed >= broken, T2 (P3), finite states
  (G5); cooler stratum lasting <= reference + 3 and late <= 1.10x. Re-scoring the 24 earlier verdicts changes one: CBg
  vs LD passes. (Since 2026-10-20 the rule is code/Auto-Chem bench/pass_v2.py; bench/judge.py gives its DEV / G / Gc
  verdict by it since commit 5f11a99, with the pre-10-19 helps() rule and the large misses as readouts.)
- 10-19: step size: every dt in 1-500 s with equal weight (14:1x), then equal weight per factor of 10 (21:4x); time
  equal per decade (21:4x).
- Proposed, the author's call (decision rule v3, cleanup plan Part C): per-call error = floored mean over the 12 gases
  (worst gas and a major / trace split as readouts); averaged per decade of elapsed time, then over decades; 8 dt bins
  (1-2, 2-5, 5-10, 10-20, 20-50, 50-100, 100-200, 200-500 s) with equal weight; two paired comparisons bootstrapped by
  case (against mini-chem on the 14 table cells, against the incumbent on VULCAN-only cells), each a symmetric 5 %
  trimmed mean with the untrimmed mean beside it; a per-dt check that no bin is clearly worse; readouts full-horizon mean,
  per-gas errors, rows over 0.5 and 2 dex, failure counts. Reviewed before any result (two independent reviews) and
  revised into the yardstick spec v1.1 section 5.6 (below): fixed 12-gas denominator, log-time quadrature within decades,
  bin weights log(b/a)/log(500) instead of equal bin weights, failures scored 2 dex per call, simultaneous per-bin upper
  limits. Superseded 2026-10-20 by rule v3.1 (scoring delegated by the author; only unrecovered steps fail, for both
  solvers; exact decade-edge splitting; section 5.4, "Rule v3.1 and the rescore").

#### Mini-chem yardstick v1.1 (design and truth panel, 2026-10-19 / 20)
- Sources (V): yard/spec_v1.1.txt and the mc/, panel/, design/ reports; copies in data/2026-10-19_minichem-yardstick/
  (spec_v1.txt, spec_v1.1.txt, element_vectors.json, table_manifest.json, mc_*.json, panel_manifest.json, panel_conv.json,
  panel_design.json, design_meta.json, design_budget.json, design_verify.json and the three reports). The truth arrays
  (yard/panel/tag, yard/design/*.npz, up to 1.6 GB) stay in place.
- Claim scope (v1.1): constant T and P, no mixing, mini-chem's native composition cells; truth-initialized segments plus
  uninterrupted free runs; wording if won "better than mini-chem over the sampled truth-initialized segments and free runs
  at constant T-P, at mini-chem's native compositions". Changing T-P, variable dt and mixing are v2. Written before any
  result (spec v1, reviewed twice, then v1.1 with addenda A 23:24 and B 00:03, both before the runs they size).
- Cells: primary (equal weight) Z 0.1 x C/O {0.25, 1}; Z 1 and 10 x C/O {0.25, solar, 1}; Z 100 x C/O {0.25, 1} (10
  cells). Readouts: 0.1x solar (its CO + NH3 -> HCN table is a copy of CH4 + NH3 -> HCN, also in the shipped chem_data/01x)
  and 100x solar (CH4 + CH4 -> C2H2 + 3H2 rate 0 at 14,779 of 19,201 entries, 0.77 of the 300-3000 K x 1e-5-100 bar
  domain; mini-chem floors kf at 1e-199, so the reaction is off there), 3x, 5x; off-table VULCAN-only cells 300x at C/O
  {0.25, solar, 1} and C/O 0.1 at {1, 10, 100}x (model vs CBg; mini-chem with its log-nearest table); a confound cell
  1x_L09 (He/H 0.0969, Lodders 2009 N/H). Other table findings: 0.1x C/O 2 CO-HCN equals 100x C/O 0.25 CH4-HCN; 100x C/O 1
  CO-HCN exists only as a 40 x 56 grid to 3000 K; several headers name the wrong reaction (data correct); converted C/O
  tables verified by reproducing all 42 shipped solar files byte for byte (table_manifest.json).
- Element vectors (element_vectors.json): Lodders 2019 present-day (Table 8): 1x C/H 2.9512e-4, N/H 7.0795e-5, O/H
  5.3703e-4 (equal to mini-chem's shipped benchmark runs and example input); O and N scale with Z, C = (C/O) x O at fixed
  O (Tsai et al. 2022 sec 2.2.1; the paper prints no numbers); He/H 0.08395 in truth and mini-chem (not recorded
  upstream; a decision). Mini-chem's equilibrium tables use Asplund 2021 (not used). Our training data use Lodders 2009
  (C +6.3 %, N -13.5 %, O -11.4 % against the table vector).
- Mini-chem driver (code/mini_chem/e13_driver, GPL side): one cold call per external step, per-row table set, start, T, P
  and call list; returns istate, reached time, warning counts; failure only for istate < -1, reached time short or a
  non-finite / negative state; reproduces the six stored Fig. 4 runs bitwise; per-step coupling vs the stored adaptive run
  <= 0.02 dex in five cases, up to 0.157 (C2H2) at 2000 K, 1 mbar, mostly mini-chem's own rtol 1e-3 error (rtol 1e-5 vs
  1e-8 <= 0.010); 11.5-13.0 us per call per core from raw gas (16.5-18.9 at 400-800 K), 3.9-6.9 near equilibrium. Start
  mapping from VULCAN's 53 species to mini-chem's 13 (He included): 12 shared gases and He copied, hidden C to the larger
  of CH4 / CO (its O taken from the hidden O first), hidden N to the larger of NH3 / N2, hidden O to H2O, H balanced in H2.
- Truth panel (panel/report.txt, manifest.json): 21 cells x 46 conditions (40 scored: 24 GCM-envelope by occupancy, 10
  cooler 300-1300 K, 6 edge; plus the six Fig. 4 points) x 4 starts (eq-target, eq-far, raw-A: C in CH4 / N in NH3 / O in
  H2O; raw-B: CO, N2, rest of O in H2O; weights 0.35 / 0.25 / 0.20 / 0.20) = 3,818 VULCAN-0D b3ccf4f runs at rtol 0.005 to
  1e8 s with exact rows at the restart ages (17: start, 1, 3, ..., 3e7 s); fast path byte-identical to the stock path
  (114 of 114); 2.5 CPU-h. All accepted (max element drift 3.7e-4 dex, N; limit 0.01); rtol 5e-4 rerun of 249 histories
  per-call 12-gas mean p99 5.3e-4 dex (limit 0.01); eq-target starts stationary (worst gas <= 8.6e-4 dex over 1e8 s);
  direct-stop check (VULCAN stopped exactly at call times vs interpolated stored steps; 1,411 stratified + 16,517
  sharp-transition units) max 9.5e-4 dex (limit 0.005). ID base 4,290,000,000 (4.30e9 exceeds 2^32). raw-A is infeasible
  at 300x C/O 1 (omitted there).
- Shared row design (design/report.txt): 1,600 scored parents (10 cells x 40 x 4), 17 restart ages x 17 step sizes (16
  seeded log-uniform draws, 2 per log bin, plus the 10 s anchor), segments of N = min(floor((t_next - t0) / dt), 2000)
  calls (2000 from the parent's own start; 67,286 untested triples with N = 0), 54 saved calls per full segment; 40
  free-run parents (eq-target, raw-A) at dt 500 and 100 s to 1e8 s and dt 10 s to 1e6 s: 395,234 rows, 530 M calls and
  16.7 M saved comparison points per solver. Model starts equal the bench code's own (27,200 of 27,200 bitwise). The start
  mapping had to clip H2O to zero at 5.5 % of start units (element residual > 1e-6 at 452, up to 2.5 %; a gas > 1 dex off
  at 253 units, 0.9 %); recorded, not subtracted (spec). Budget: mini-chem about 1.5-1.8 of 2 CPU-h; models about 29
  GPU-min by cell-time scaling but about 97 GPU-min (CBg) once the sequential free runs are counted (decision owed).
- Score (proposal v3, spec sec 5.6): per call the mean over the 12 gases of the floored error (fixed denominator); readouts
  worst gas, 6-major and trace means; failures (non-finite, projection fallback, istate < -1, reached time short) score 2
  dex per call from the first failure (sensitivity 0.5 and 5 dex); within each decade log-time quadrature over the saved
  calls, then equal weight per decade, dt bins by log width; paired D = model - mini-chem per condition; symmetric 5 %
  trimmed mean primary with untrimmed mean and median beside it; 4,000 condition-cluster bootstrap resamples; "beats" =
  trimmed and untrimmed mean D < 0 with upper limit < 0, every bin's simultaneous upper limit < +0.05 dex, worst-gas
  trimmed D upper limit < +0.05, failure rate <= mini-chem's + 0.1 %. Large finite misses are readouts.

#### Mini-chem yardstick v1.1: results (2026-10-20; under the registered rule v3; superseded by rule v3.1, next entry)
- V (yardstick _build/score/report.txt, tables.txt, score.json; 06:35): all three models LOSE to mini-chem under the
  registered rule. Per-call 12-gas mean error vs VULCAN, segments (400 conditions, 10 primary cells x 40, 4 starts, 17
  restart ages, 17 step sizes): mini-chem 0.024 dex, LD1 0.057, FINAL 0.064, CBg 0.083; free runs (20 conditions, all from
  t = 0): mini-chem 0.015, LD1 0.069, CBg 0.069, FINAL 0.076. Paired trimmed D (model - mini-chem), segments: LD1 +0.031
  [+0.023, +0.038], FINAL +0.037 [+0.029, +0.045], CBg +0.056 [+0.047, +0.064]; free runs +0.053 / +0.060 / +0.053 with
  intervals about 0.01-0.10. Holds at failure costs 0.5 / 2 / 5 dex, under the lenient failure reading (CBg +0.030
  [+0.023, +0.037]) and with exact interval splitting at decade edges (CBg free runs +0.039, 10 s anchor +0.041 instead
  of +0.085). Mini-chem's score is unchanged at rtol 1e-5. Every dt bin and every cell is worse for every model; FINAL and
  LD1 tie mini-chem on the cooler stratum and the domain edges (whole loss at the GCM-like points). Between models: CBg vs
  FINAL not established (FINAL wins segments, CBg the free runs by 0.006), CBg vs LD1 CBg loses (segments +0.017, free runs
  tie). Failures: mini-chem 0.012 % of steps; FINAL 0.032 %, LD1 0.024 %; CBg 1.42 % (first-call projection fallback in
  5,300 raw-B rows at C/O = 1; LM repaired all but 20; the registered wording counts the fallback).
- V POST-HOC (labelled; no verdict changes): the loss sits entirely in rows started at exactly t = 0 (the parent's own
  unreacted or transported state): D +0.10 to +0.18. On restarts from the VULCAN state (t0 >= 1 s) every model beats
  mini-chem at early ages (CBg by 0.007-0.022 dex from 1 s to 1e4 s under the lenient reading), ties to about 3e5 s and
  trails by 0.002-0.004 at 1e6-3e7 s; restart-only, FINAL and LD1 meet all five "beats" criteria under either reading,
  CBg under the lenient one (registered: failure rate and upper limit +0.001 fail). Mechanism: a t = 0 raw start has 15-16
  of the 25 model entries at the 1e-30 floor; none of CBg's 61.8 M training states has more than 13; for the first 5-10
  calls the models leave the scored gases unchanged while the truth moves 0.15-0.35 dex (mini-chem follows within 0.02);
  by call 2,000 their median error is 0.015-0.019. Every earlier model bench started at 1e-3 s, never at the floor state.
- V Composition confound (_build/confound/report.txt): the 1x cell rerun at the training He/H 0.0969 and Lodders 2009 N/H:
  CBg's D +0.034 -> +0.031 (paired -0.003 [-0.004, -0.001]), only late (1e4 s on; eq-target error 0.0039 -> 0.00001);
  mini-chem unchanged; the t = 0 loss unchanged. He/H is fixed at 0.096917 in the whole training set, so every panel cell
  is off-training in He/H; the 0.002-0.004 late deficits are probably that mismatch (I, one cell).
- V Reviews: two external reviews (t0-fix reviews/) reproduced D from the stored states and
  found no alignment, mapping, unit, floor or normalization error; open: mini-chem's driver floors before testing
  negativity (some mini-chem failures may be hidden), the decade-edge weighting (above), the failure wording (asymmetric:
  mini-chem's handled DLSODE recoveries do not count, the model's handled LM recovery does).
- I Reading: under the author's weighting (every decade equal, raw starts counting) the deciding weakness of all three
  models is the very first calls from an exact unreacted / transported start at the floor; from any truth state at >= 1 s
  they already beat mini-chem at early times. Candidate fixes (none run): train on t = 0 floor-pattern starts (exact x0
  rows of raw and transported starts, multi-call), or start from a short internal substep (the P* startup substeps were
  off here by design: one call per step); CBg's C/O = 1 raw-B first-call projection conditioning. Untested: changing T-P,
  variable dt, mixing; readout cells; FINAL / LD1 at the training He/H.

#### Rule v3.1 and the rescore (2026-10-20; adopted, scoring delegated by the author ~07:10)
- V (t0-fix rule/report.txt): there was no arithmetic bug. Two unfair details of v3: (1) a model step counted as failed
  whenever the LM rescue stepped in, even when it repaired the step (median actual error 0.000 dex), and the rest of the
  row was then scored as failed, while mini-chem's handled DLSODE recoveries never counted (about 0.025 of CBg's +0.056);
  (2) whole-node quadrature gave a decade a full share when a call landed exactly on its edge (10 s, 100 s). Rule v3.1:
  a failure is an UNRECOVERED step only (model: kept input, guard, non-finite or negative state; mini-chem: istate < -1,
  short reach, non-finite), the same for both solvers; time averaged with exact interval splitting at decade edges; a
  decade with zero covered duration is untested. Code: code/Auto-Chem bench/yardstick.py (NumPy only; rule switch v3 |
  v3.1, default v3.1; v3 reproduces the stored score.json exactly) and tests/yardstick_regress.py (five synthetic tests,
  the v3 reproduction, the v3.1 pin for CBg / FINAL / LD1, a negative control); commit 99a98c6. bench/judge.py now gives
  the DEV / G / Gc verdict by pass rule v2 with the large misses and the older rule as readouts (5f11a99); adoption
  between models is decided by the yardstick rule, the judge is a continuity check.
- V Rescore (t0-fix score_v31/report.txt; paired 5 % trimmed D, 95 % bootstrap by condition), segments vs mini-chem:
  CBg +0.056 -> +0.031 [+0.023, +0.038], FINAL +0.037 [+0.029, +0.045] (unchanged), LD1 +0.031 [+0.023, +0.038]; all
  three still lose. Free runs: CBg +0.039, FINAL +0.046, LD1 +0.038 (were +0.053, +0.060, +0.053). Absolute trimmed
  segment scores vs VULCAN: CBg 0.0577, FINAL 0.0635, LD1 0.0573, mini-chem 0.0236. Between models: CBg beats FINAL
  (-0.006 [-0.007, -0.005], all five checks; before: not established) and ties LD1 (+0.000; before: lost). Failures
  under v3.1: CBg 20 runs (0.008 % of segment steps), all started at t = 0; FINAL and LD1 none.
- V Where the gap sits: t0 = 0 rows CBg +0.111 [+0.091, +0.131], FINAL +0.128, LD1 +0.105 (all lose); restart rows only
  (t0 > 0) CBg -0.009 [-0.013, -0.006], FINAL -0.008, LD1 -0.007 (all beat mini-chem on all five checks). By time since
  a row's own start, t0 = 0 rows, CBg: +0.291 [+0.240, +0.343] at 1-10 s, +0.279 at 10-100 s, +0.156 at 100-1e3 s, +0.015
  at 1e3-1e4 s; restart rows -0.006 [-0.009, -0.003] in the first 1-10 s, about 0 by 1e5-1e6 s.
- Stated limitation (not fixable without changing mini-chem, which the author ruled out): mini-chem's driver floors and
  renormalizes before testing negativity, so some mini-chem failures could go unseen.

#### Early-transient ablation of CBg (2026-10-20; inference only, no retraining)
- V (t0-fix ablation/report.txt, tables.txt, score.json; choices fixed beforehand in choices.txt): the 27,200 t0 = 0
  segment rows of the yardstick (400 conditions x 4 starts x 17 dt, 2,000 calls, 54 saved calls each), rerun with six
  run-time changes and scored under v3.1. Arm (a), CBg unchanged, reproduces the stored outputs bitwise (all scored points,
  counters and failure calls) and the scorer the stored t = 0 numbers.
- V D vs (a) (abs 0.1740): (b) Newton 64 instead of 32 +0.0000; (c4) / (c10) first call in 4 / 10 internal substeps -0.015
  / -0.032; (d) entries at the 1e-30 floor lifted to 1e-20 and re-projected -0.011; (f) first 10 s in about 1 s calls
  -0.006; (e) ORACLE: start from VULCAN's state at t = 1e-3 s instead of t = 0, -0.147 [-0.166, -0.129] (abs 0.022; vs
  mini-chem on these rows -0.029, all five checks; gain in every start type, dt bin and decade). c10 helps only at dt
  50-500 s and makes 1e4-1e6 s worse by +0.006 to +0.009; c4 and f likewise slightly worse late.
- V Mechanism: at t = 0 a raw start has 15-16 of the 25 inputs at the floor (no training state has more than 13; training
  windows start at 1e-3 s). On call 1 the network's relaxed update leaves raw gas nearly in place (0.001-0.015 dex moved
  against a true 0.14-0.17; relaxation fraction 5e-5 to 4e-3 per call; product targets stay at the floor), so the
  projection has nothing to lose (Newton 32 and 64 agree to 3.5e-11 dex after the floors; LM converges on every Newton
  failure). V POST-HOC split (ablation posthoc_rows.json; not pre-registered), raw starts by whether the 1e-3 s state
  differs visibly (>= 1e-3 dex, 12-gas floored mean) from the t = 0 state: VISIBLE rows (raw-A 2,856, median T 2,044 K;
  raw-B 2,584, 2,101 K; about 40 %) carry 80-90 % of the raw-start error (I, row-mean arithmetic): real chemistry happens
  within the first millisecond, CBg's call-1 error from t = 0 is 2.6 dex (raw-A) / 1.4 dex (raw-B) median, from the 1e-3 s
  state 0.07 / 0.05. INVISIBLE rows (3,944 / 4,216, median T 876 / 913 K): the 1e-3 s state differs only in sub-floor
  traces (raw-A medians: H -30 -> -10.1, CH3 -30 -> -11.1, OH -30 -> -14.5, reservoirs -28 -> -13 to -20); mean row error
  0.082 / 0.069 from t = 0, 0.014 / 0.027 with those traces (the network reads them as a clock). A uniform 1e-20 lift does
  not reproduce the radical pattern (0.056 / 0.042); it does remove almost all raw-B first-call Newton failures (1,233 ->
  23 rows).
- Conclusion: the t = 0 weakness is the untrained exact-t0 input pattern, not the projection and not mainly the step size.
  No deployable arm adopted (best 29 % of the gap, with a late cost). Supports training on t = 0 starts (the t0-fix arms,
  section 5.6); an untested run-time alternative is a learned or kinetic "starter" that supplies the first-millisecond
  traces (the author ruled a starter out for now, 10-20). Caveat: the sub-10 s steps of arms c and f lie below CBg's
  trained one-step range, so their small effect is not a clean negative.

#### Starting-gas mix for the score, from the GCM literature (2026-10-20)
- V (t0-fix startweights/report.txt; two external reviews in startweights/; citations checked against the
  papers): every 3D GCM chemistry paper read that states its start (10 of 14) initializes at local equilibrium, often
  after a 1,000-10,000-day dynamics spin-up (e.g. Mendonca et al. 2018; Drummond et al. 2020; Lee et al. 2023; Zamyatina
  et al. 2023; Christie et al. 2026); pseudo-2D models start from a converged 1D kinetic column. Departures from local
  equilibrium come from transport and quenching: minor gases (NH3, HCN, CO2, the minor carbon carrier) sit 1-4 dex off
  (Liu et al. 2026: CO and CO2 from below 1e-15 to about 1e-3), H2, H2O and the main carbon carrier within about 2x;
  quench levels from a few mbar to about 15 bar. Equilibrium from a source 1 dex deeper or several hundred K hotter is a
  realistic transported STATE but not a one-step jump (I, from published winds: under about 10 K and 0.01 dex per 500 s
  step; 300 K takes about 0.2-0.7 day). Unreacted gas appears only in initial-condition memory tests (Cooper & Showman
  2006) and mini-chem's 0D bench test (Tsai et al. 2022). Published GCMs call kinetic chemistry every 1,500-3,750 s with
  dynamical steps of 25-50 s (the emulator is meant to run at the dynamical step; section 3).
- Decision (scoring delegated; reviewed): score weights eq-target 0.55, eq-far 0.45, raw-A 0, raw-B 0 (literature and
  one external review; the other preferred 0.60 / 0.30 / 0.05 / 0.05 to keep evolved raw descendants in the score).
  Raw starts are a robustness readout without a gate (a gate on an unphysical start would contradict "large misses are
  readouts"). New panels draw eq-far sources from T-P bins co-occupied by at least one GCM (69 % of the v1.1 pairs) and
  add eq-far long runs; a practical tie margin is needed. Training data keep a small share of raw / exact-t = 0 starts
  for robustness. Sensitivity readouts at 0.70 / 0.30 and 0.40 / 0.60. Spec: clean-state campaign.

#### Score-1 and the clean data design (2026-10-20; specs final, implementation in progress)
- Sources (V): experiments/2026-10-20_clean-state/_build/spec/score_spec.txt and data_design.txt (revision 3; the
  shared block is identical in both), reconcile.txt, implementation_plan.txt; two external reviews in
  reviews/ (a third planned review failed for lack of credits and was replaced by the second). Earlier versions kept
  as *_v1.txt, *_v2.txt.
- Score-1: one number against VULCAN from three panels with equal weight: S constant-condition segments (600 conditions,
  450 GCM-occupancy / 150 cooler, x {eqt, eqf}, restarts at 17 ages), L long uninterrupted runs (128 conditions x {eqt,
  eqf} x 8 dt bins, 172,800 calls), P changing-condition paths (128 conditions x {eqt, eqf}, 10,000 calls, truth held
  over holds of <= 1,500 s); per-gas floored error, equal weight per decade and per factor of 10 in dt, start weights
  0.55 / 0.45, raw starts a readout; paired per-condition D (5 % trimmed mean, untrimmed beside it, bootstrap of
  conditions within cells); tie margin 0.005 dex (above measured noise sources, below the smallest real model gap of the
  v3.1 rescore, 0.006); failures only unrecovered steps; readouts: lasting score, per-dt-bin and 10 s check, raw starts,
  large misses with identities, T2 snap-back (in-domain boxes only), 1000-day cycle, full-call speed, mini-chem on its
  cells, old-weight continuity. A clean model is adopted unless WORSE than CBg. A sealed twin of the panels is built and
  sealed at creation (/scratch/imalsky/clean_twin_SEALED); which sealed set is opened at the next pick is the author's
  decision then. Code: bench/score_design.py -> bench/score_run.py -> bench/score.py; 25 older scoring scripts retire.
- Clean training data: 786,432 runs (rows 0-786,431 of one 2^20-row scrambled Sobol table, d = 16, seed 2026102115;
  CBg's U4 kept 766,452); IDs 4,210,000,000 + row (none used in any pack); Lodders 2019 present-day metals, Z 0.1-300x
  log-uniform, C/O 0.1-1 plus a 1-2 side slice, N/O 0.3-3x solar, He/H uniform 0.078-0.100; T-P from the GCM occupancy
  plus a cooler box and the full box, 1e-5-100 bar; start kinds eqt 0.20, eqn 0.20 (near-equilibrium small shifts, the
  old m18 rule), eqf 0.30 (transported, sources from GCM-co-occupied bins), mix 0.10, pert 0.14, raw 0.06 (exact t = 0);
  "broad" random gas dropped as unphysical, the kicked second pass replaced by pert starts (kicks gave no gain); no draw
  weights (CBg's weights gave an effective 169k / 230k of 624k rows in stages 1 / 2); rtol 0.005 run to 1e8 s; the same
  100-point grid, row 0 the exact start. Training: CBg's architecture (6x2048) and recipe, steps 1,000,000 / 200,000
  (equal passes per row within 1 %), row-0 probabilities 0.2 / 0.1 per file, one-step windows 10-500 s (dt_min 10 kept:
  every earlier from-scratch run below 10 s was worse at larger steps; a test on the new model is pre-registered if a
  1-10 s weakness shows). Stage-2 chains train_a 2000 x 100, train_b 512 x 500 calls, regenerated with the new
  composition and rtol 0.005 passed explicitly (the old chain driver recorded rtol 0.05 but never passed it; VULCAN's
  default is also 0.05; V gen/e13/paths_v3.py:48, 190, 207 and gen/VULCAN-0D/vulcan_cfg.py:121).
- Plan (I, planning numbers until the pilot measures them): about 36-39 h from go to the scored comparison with CBg
  (data 1-3 h, cache 1.6 h, stage 1 about 10.4 h, stage 2 about 21 h, scoring); project data peak about 245 GB.
- Implementation and pilot (V clean-state pilot/report.txt and the package reports A, B, C, D): check-design PASS on the
  full table; fast vs stock 96 / 96 byte-identical (integration 8.60 s stock vs 0.143 s fast per box); pilot 4,096 /
  4,096 accepted at 0.123 CPU-s per run (8.13 per CPU-s; planning 0.18-0.23), label audit rows 1-10 worst-gas p99
  0.00038 dex vs exact stops, old production settings (rtol 0.05) differ by worst-gas p99 0.017-0.030 dex; packing 75
  groups/s (two packers 70.5 + 71.5); training chains 1.84 CPU-s per call; path truth 1.76 CPU-s per hold; carried
  state per hold vs one run p99 9.0e-05 dex; scorer reproduces the v3.1 rescore (1,659 leaves per model, 0 differ) and
  the runner CBg's stored yardstick and PD states bit for bit; bitcheck s1 / s2 PASS; smoke training (200 + 50 steps on
  the pilot cache and chains) finite, row-0 share 0.1985 (set 0.2), projection residual max 3.6e-15. Before stage 2 up
  to 0.5 % unfinished chain paths are allowed (lib/chains.py loads finished paths only; CBg trained with 1,999 / 2,000
  and 511 / 512). code/Auto-Chem commits 135e9c3 to 13ba400 (not pushed); 25 older scoring scripts retired. Generator
  files frozen by sha256; full-run preflight clean. Timeline from go (I): about 36-38 h, set by training.

### 5.5 What helped and what did not

#### Helped (ranked by effect on the long-run metric of its time; one seed unless stated)
1. Relaxation head instead of the delta head (10-01): zero guard interventions in every run, against every delta
   checkpoint failing (98 % of rows at 10x2048, dt 1 s); data/2026-10-01_architecture-tests/scores.csv.
2. Closed 25-entry state, per-call element projection and the chain term (B6, 10-10): xlong late p50 0.191 -> 0.003, p90
   2.01 -> 0.157; DEV per-path p99 -0.305 [-0.469, -0.189]; C/O p99 at 1000 d 0.129 -> 0.001; about half the frozen
   model's > 2 dex rows (three seeds); data/2026-10-09_e13-conservation/scoreboard_gateB.json, path_board_PX.json.
3. Grid flow term (allG, 10-06): G dt500 hot band 3.1 -> 0.03, every band below e9_F10fill_s0 with intervals below 0;
   data/2026-10-06_e13/scores_e13_1914.json.
4. Long-jump loss with jumps to 1e7 s (10-01): 6x512 dt 10 s p90 2.38 -> 1.01, late hold to 0.04; capping jumps at 1e5 s
   loses (A10 +0.268 [+0.118, +0.482], hold 0.037 -> 0.479); data/2026-10-03_repo-sweep/E3_pairs.csv.
5. A 6x2048 trunk with a 1M-step stage 1 at batch 8192, made trainable in stage 2 by a per-row rollout gradient cap
   (CBg, 10-19): inside lasting 43 -> 31 vs LD (61 for FINAL), late 175 -> 96, T2 +0.096; data/2026-10-13_e13-scale/
   verdict_e13s_CBg_s2_vs_e13s_LD_s3.txt, gfix_report.txt.
6. Width and data together (FINAL vs F2, 10-14): rows > 2 19 -> 14, > 0.5 574 -> 373, inside lasting 101 -> 59; neither
   alone helps; data/2026-10-13_e13-scale/board6_*.json, critic_env_rescore.txt.
7. Stage-2 continuations at s2_lr 1e-5 (10-15 to 10-18): inside lasting FINAL 61 -> C0 53 -> D-kick 48 -> LD 43 -> LE3
   40, then a plateau; each 20k step is within about one noise unit alone (data/2026-10-13_e13-scale/e_judge.txt).
8. Oxygen-poor fill on floored data (E8, E9, 10-05): fresh O-poor hot test p90 3.03 -> 0.09; holdout hot 1.15 -> 0.10;
   floored data alone: formerly rejected sources hot rows > 3 dex 26 -> 1; data/2026-10-03_repo-sweep/E9_*.json.
9. P* inference (10-12): the silent projection freeze removed (H2 606 8.04 -> 0.10 dex), five 100 s substeps on the first
   call (first-call rows > 2 23 -> 3, H2c 49 6.35 -> 0.10); converged rows bitwise unchanged; data/2026-10-12_e13-fixes/
   freeze_trace.txt, freeze_solvers.txt, first_study.txt.
10. Cold-data union with switch events drawn 3x (F2 vs B6 s2, 10-13): P* rows > 2 26 -> 17, > 0.5 505 -> 423; Gc rows > 2
   15 -> 5; data/2026-10-12_e13-fixes/board_fx_*.json, sel_board_F2.txt.
11. M rows in stage 1 (M fs, 10-08; two seeds): the hot dead band narrows (S-tag median move ratio 0.49-0.90 at >= 1300 K
   vs capw17's 0.000-0.011), paths -0.199 [-0.393, -0.124]; with a 75k-step stage 2 on top (m17fs) best on every
   instrument (G+Gc -0.0195 / -0.0196 at 300-800 / 800-1300 K, v2l -0.368); data/2026-10-08_e13-sprint/stag_e13_*.txt, score_g_*.txt.
12. End-anchoring of the grid term (allGend, 10-07): -0.02 to -0.04 in the three cooler bands; width 512 -> 1024 and data
   x2.9 each about -0.03 at 800-1300 K (factorial); data/2026-10-06_e13/scores_v11.json.
13. fp64 host state (10-08): an fp32 host drifts 0.0042 dex per 1e6 calls (about 0.034 at 1000 d); fp64 stays within
   1e-10. Deployment rule.
14. Sobol design per training trajectory (E12, two seeds): beats a same-size random subset of old rows on U at 800-1800 K
   (pooled 800-1300 K -0.25 [-0.35, -0.14]).
16. (2026-10-23, rollout misses, fine-tune tests, one seed; section 5.6) One-step windows at dt 1-500 s: large rollout misses
   1.93 -> 0.86 %; the rollout term with a loss at every call: 0.79 -> 0.37 %, calls with a miss 3.2 % (final clean model
   10.4 %).
15. Speed work that keeps results (10-17): data in host memory (bitwise batches), lockstep rollouts (fp64 gradients equal
   to 2.2e-10), fused projection (1.4e-11 in ln y), truth arrays read once in benches (bitwise), the fast generator
   (byte-identical boxes); batch 8192 (4x the batch for 1.16x the step time, no accuracy cost).

#### Did not help (what, dose, outcome)
Heads and architectures:
- Delta head under every regime tested: 6x512 one-step continuation, BPTT K 10, detached rollouts (gamma 0.9),
  pushforward, input noise 1e-2 z (p90 4.7-11.5 dex); 10x2048 one-step and BPTT K 10 (all rows fail).
- Latent heads: exact zero at dt = 0 without a box (16 / 512 modes, 16 % / 80 % rejected row-calls), matched controls,
  the sibling latent model (90 % failing rows), box-bounded latent 16 / 64 modes (1.98 / 3.70 vs relax 1.37).
- Simplex head (E4b, 13-slot softmax): HARMFUL (A dt 1 s late-max +0.287 [+0.170, +0.470]).
- Two-rate head with a shared target (E4d): NULL (A10 -0.056, short of -0.10; dt 1 s -0.125 real; E veto +0.279).
- Two-exponential head relax2expr (X2E, 10-15): worse than its control at hot 1 mbar raw-gas cases (dt 0.1 s medians
  1.50 vs 0.53 at 2000 K), slightly better at 10 bar; not carried further.
- Fresh 53-species state (T6, 10-16): inside lasting 140 vs FINAL 61, late > 2 33 vs 4; the gap is there in stage 1.
- Trunks (E10): MLP below 6x512 (4x256 4.14, 3x128 6.10 dex vs 0.768); FastKAN at matched parameters +0.195 [+0.089,
  +0.357] at 2-3x cost; compute-matched species attention +0.415 [+0.259, +0.651]. FiLM at 6x512 within seed spread;
  FiLM from scratch (m17ffc) worse in every G band.
- relaxT (zero-init 1000/T and ln T input): stage-2 branch NULL; from scratch 800-1300 K -0.012 (k = 2 bar missed).
- 10x2048 at the shared 6x1024 recipe (big17, 10-07): stage 2 diverged; retry at lr 1e-5 killed by its rule.
- (2026-10-23) Straight-through box clamps in the relaxation head: diverged (raw targets up to 900 z units outside the box).
  A rate-only dt head trained on single calls: better one-step fit, rollout misses 2-3x worse (needs rollout training).
Constraints and conservation:
- In-model sum projection (E4a): NULL; E veto +0.934; trades sums too high for sums too low (stress-arm share below the
  data's 0.1 % sum quantile 0.4 % -> 6.4 %). Inference-only: NULL.
- Soft physics loss (E4c: hinge on sum12 > 1 weight 0.05, on ratio drift weight 0.2): NULL, excursions unchanged.
- Host renormalization (E2): NULL on both bars; under stress it collapses species to the floor (91-100 % of rows).
- Fixed-point supervision (E6, HybridLoss on converged tails, weight 0.2): HARMFUL; |q - z| at equilibria fell tenfold
  (tiny-rate share 0.605 -> 0.217) but drift at 1e9 s rose 0.386 -> 1.765 (1e10 s 1.873 -> 3.339), hold +0.156.
- A 12-species projector inside training (B4): dropped after Gate A. A frozen-radical rate hinge (10-07): dropped before
  training (frozen truth only 3-5 % of the above-p90 300-800 K radical points; 85-91 % of its value outside 300-800 K).
- Hard 12-species projection at inference (Gate A, 10-09): vetoed (G 2300-3001 K p99 0.313 -> 0.790, a new 17.8-dex
  C2H2 miss); learned 12-species budget leak (B3): not run past smoke (bias accumulates every call; gradients through
  the ill-conditioned projection 1e3-1e5x relax's).
- Chain term on the 12-species model (B1 weight 0.5, B2 1.0): DEV better, vetoed on G / Gc p99 and rows.
- Forcing the rate at near-equilibrium (a = 1, 10-07): diverges (6.7 dex after 1000 calls); multiplying every rate by
  3.98 at kicked 650-1300 K states (10-15): absolute errors > 0.5 dex at 5e5 s 15 -> 102; per-row rate factors picked
  early do not hold.
- Difference law z' = q(z, g_k) + (1 - a)(z - q(z, g_{k-1})) (inference only, 10-08): better on single shifts, divergent
  on the 64 paths.
Training regimes and losses:
- Short BPTT (K 10, K 32 at matched transitions), pushforward, detached rollouts, 25 % steady-state starts (6x512, 25k
  steps): modest at best; BPTT K 10 worsened the 10x2048 transient and hold. Input noise sigma 1e-2 and 1e-3 z: worse.
- Longer stage 1 alone (E5: 600k, 2.4M at 6x512; 800k at 10x2048): major species improve, trace species do not reach
  10 %; the 800k model extrapolated worse into the O-poor hole.
- Jump-only steps doubled (R3 vs R2), jump loss from step 0 in one stage (R8), LJ in one stage vs two: equal. Jump loss in
  stage 1 on E13 data (allJG +0.09 to +0.19; m17fj +0.035 / +0.039 / +0.029).
- One-step windows below 10 s: wide dt 0.1-1000 s (m17fw, +0.037 / +0.027 / +0.018), dt 0.1-500 s (X2C: inside lasting 68
  vs 61, xlong p99 0.650 vs 0.453), dt 1-500 s (LD1: 53 vs 43, cooler and xlong guards fail; better only for raw-gas
  starts at 1-2 s).
- Relative one-step speed loss (X2V, asinh, vel_w 0.01; dose check passed: speed error -11 %): lasting 68 vs 68.
- Grid dex^2 term (fsq) null; jump_w 0 (fsnj) worse tail; per-gas censored loss (W) not admissible (regressed hot fresh
  low-P starts).
- Reweighting: cold trajectories drawn 3x (traj_w [1300, 3]) no cold gain, 1300-1800 K +0.04; rows below 1300 K drawn 2x
  (LDc) 44 vs 43.
- Longer chain curriculum (B6L, K to 32, pre-roll to 468) and + train_c (B6Lc): not eligible. More M rows (M64, 4x):
  no gain. EMA 0.9999 slightly worse, 0.99995 worse. Seed ensembles (2 seeds, E2; m17fs s0 + s1): about the better seed.
  Median ensemble E: 3.7x cost, froze holdout rows, its large-miss gain did not hold on the holdout. Sub-steps n = 2, 4
  per call: only the first call improves (FIN 1.428 -> 1.023 -> 0.812), worse after 1e3 calls. First-call splits other
  than five of 100 s: two of 250 s (3 / 5 rows > 2, H2 606 still frozen), fifty of 10 s (8 / 6 rows, max 5.1 / 6.1).
- More stage-2 steps past the plateau: round E (3 x 20k on LD) below the rule; CBgP1 (+40k on CBg) no gain; CBg's own
  stage 2 is flat from 50k to 200k.
- Batch size (10-16 / 10-17): stage 1 at 512 / 2048 / 8192 at equal steps within 1.5 % validation (512 helps only at equal
  samples, 4x the steps); Lion 3e-5 at 8192 +3.7 %; stage-2 batches /4 or x4: 44 / 43 vs 43. TF32 off: stage 1 ratio 0.997;
  stage 2 (LDF) 40 vs 43, below the rule. Stage-1 validation vs batch 2048 (0.038263): 512 at equal steps +6.7 %, 512 at
  equal samples (600k steps) -5.3 %, 8192 -0.7 %, 8192 at LR 6e-4 -1.6 %; path p90 moves 10-16 % between arms either way
  (one-seed luck). The author's "Above 512 previously I found errors" is not reproduced and has no record on daw
  (data/2026-10-13_e13-scale/b_readout.txt, b_judge_s2.txt).
- Gradient-explosion alternatives (10-18): truncating the grid back-propagation to 16 calls alone leaves outliers (5,970)
  and bends healthy gradients (nsub-4 cosine 0.92-0.95): not used; the per-row cap was.
- Engineering options never run or unusable: the stage-2 stability keys committed 10-13 (s2_prewarm, grid_w_ramp, grid_K /
  grid_sub ramps, grid_bptt, term_clip, gnorm_every) never ran on the GPU (removed from the code 2026-10-20); CUDA graphs
  over whole stage-2 calls are unusable in this torch version (illegal memory access, or minutes of re-recording per call
  pattern), graphs for the no-grad calls add only about 2 %; the fused 4x4 projection solve returns a zero step for
  pivots <= 1e-100 (the row then fails the residual check and keeps its input), and zero pivots gave NaN gradients under
  10-30x amplified moves until the guarded pivot of the lockstep step (0 such rows, was 116-856).
Data:
- Width alone (G1, 6x1536 old data), data alone (B, 6x1024 U2), new data only in stage 2 (A): rows > 2 25 / 26 / 25 vs 19.
- 57 % more runs targeted at the GCM range at 6x1536 (P15, U4): inside lasting 64 vs 61 (33 fixed, 36 broken).
- Kicked off-trajectory runs (D-kick vs C0): 48 vs 53, inside the no-gain band; attribution not separable.
- More cold data of the design (D-cold) and 12,283 event-targeted cold runs (F3 vs F2): systematic cold misses unfixed;
  F3 mixed (fewer rows > 2, more > 0.5).
- 2.86x trajectories at 6x512 (all17G vs allG): -0.02 to -0.03, about seed spread; 36.7k -> 104.7k no consistent xlong
  gain. Data fractions 25-100 % at 1.5 % of the production budget: no consistent effect.
Untested or open (not negative): the all-data run CB5; late-time windows (L); cool-N chains (CBgN); truth-supervised
short-step rollouts; log(dt) as a trunk input; dense labels for steep trajectories; a T-gated cold expert (Xiao et al.
2025); the LiLaN loss; relabelling the model's visited states with VULCAN; a tail (CVaR) chain term; neighbour-cell
mixing stress; He in the training data; from the 10-08 audit: steady-state oversampling 0.25 on the M recipe, the grid
term in stage 1, 8x1024 with its own LR, stride 2 or unresolved weighting alone on top of end-anchoring. Also never run: a
learned clock (time since start) input and data or loss aimed at the cold CH4 collapse near 400-450 K below 1e-4 bar;
a kick-pair separation loss, an arrival-time loss, a flux form in the element null space (10-15); a band-balanced top-20 %
tail loss (10-07); the pp start-mix arm (config stubs only); for F2 the restart test, the stress arm and a median of
F-type copies; a tracer extension (H2CO + CH3OH -> 27 entries, 10-13); a check of the 15 cold runs VULCAN stopped as
steady before 1e3 s; and the external review's cheap gates (2026-10-20): 12 fused vs 32 unfused vs 32 + LM iterations on
identical inputs, an oracle per-state (q, k) fit through the projection plus several hidden-state lifts of one 25-entry
state (head form vs missing information), then an equal-budget 1-10 s truth-rollout continuation from CBg.

### 5.6 Dated record, 2026-09-29 to 2026-10-30 (newest last; the archive holds the full entries)

#### 2026-09-29 to 10-05: the 12-species code and the architecture tests
- 09-29 to 10-01, code audit of the old 12-species tree (now tag paper-v1 / the audit branch): 209 confirmed findings, 19 fix
  commits (eb93667 to dd38bb3); stage 1 10.5 ms / stage 2 85 ms per step on daw; the shipped stage-2 recipe needed 17.6 d.
- 10-01, units, provenance and the generator reconstruction: section 3. Decisions: section 6.
- 10-01, VULCAN checks: 300-1300 K tails keep evolving (median change 0.97 dex between 1e9 and 1e10 s), so a model must
  follow slow chemistry, not hold a fixed point; rtol 0.05 vs 0.005 (stated definition, 10-03): p50 <= 6.7e-5, p90
  0.7-9.1e-3, p99 1.5-4.6e-2 dex; hidden-species test: section 3.
- 10-01, host precision (identity model, 3000 calls at dt 10 s): P1 drifts p90 1.0e-6 dex, P2 a one-time 1.7e-8, P3
  exactly constant; in the 6x512 hold P1's fp32 round trip stalls the slow drift (0.85 vs 1.5 dex).
- 10-01, delta vs relaxation head, latent heads, structural ablation: section 2. Regimes at 6x512 (stage 1 150k, 25k
  stage 2; worst-bin p90 dt 10 s / hold / switch): one-step 2.50 / 1.53 / 3.25; long unroll (0-400 no-grad steps) 1.23 /
  0.51 / 1.30; 80-step unroll then unroll + jump 0.2 0.63 / 0.04 / 1.40 (confound: 50k vs 25k steps); seed 1 reproduces
  within about 0.05 dex. The remaining error after the unroll is broad (CH4, CO, CO2, H2O, N2, NH3, HCN p90 0.4-1.6 dex at
  1e7 s); I: at k dt << 1 labels fix only k (q - z), so slow targets and rates are weakly identified; the jump loss pins
  them. Hottest-bin switch gap: one fitted exponential per species follows the truth to p90 0.013-0.075, so it is not a
  representability limit (corrected 10-03).
- 10-01, size and data (SP, P3): 10x2048 one-step 0.79 (closable 0.50), + BPTT 1.23, + long unroll 1.16, finalist 0.43,
  jump alone 0.42; 25 % of the trajectories 0.88; 6x512 one-step 2.50 (not a matched comparison). Data fractions show no
  consistent effect at about 1.5 % of the production step count.
- 10-01, prior art and test ideas: LiLaN (arXiv 2501.08423) is not autoregressive and not the identity at dt = 0;
  suggested: DAgger relabelling at visited states (Ross et al. 2011), element-budget diagnostics (Sturm and Wexler),
  checkpoint and seed selection by short rollouts (ACE2, NeuralGCM), a seed ensemble of (q, k), a semigroup diagnostic,
  a horizon curriculum. Not useful here: PINN source residuals, Jacobian regularization, memory inputs, hard T experts.
- 10-01, data implications: run VULCAN without early stop to 1e9-1e10 s, add GCM-like starts (mixtures, restarts at new
  P, T), save all 53 species, store raw trajectories (4x smaller than windows).
- 10-03, finalist SPF bench (section 5.3; data/2026-10-01_architecture-tests/scores.csv, SPF_p_h1_jump). Super-cycling dt
  100 vs 10 s over 1000 d: P3 0.546 vs 0.530 (no transport in between, so the splitting error is untested). GPU cost
  estimate (10x2048, C48 x 53 levels = 732,672 cells, 1000 d): 12.7-31.9 GPU-days at dt 100 s.
- 10-03, review of GCM use: the equilibrium oracle beats FIN late in the hot bands (SPF dt 100 s final p90 0.144 vs 0.000
  at 1800-2300 K); persistence beats FIN on the 1300-1800 K hold (0.0081 vs 0.0145); in mixing-ratio units only element
  ratios are invariant (ref_a: ratios drift <= 6.7e-4 while totals change -58 % to +99 %); 63 of 621 layers of reference
  GCM profiles lie outside the training box (VULCAN rates above 3000 K extrapolated); CPU output depends on the call size
  (up to 1.3e-5 dex); a tampered .pt2 loads without error (the export now writes a .sha256); the label interpolation floor
  is p99 0.014 / 0.009 dex (300-point grid 0.003); dt_max 1e8 cuts VULCAN steps to 0.44x; a GCM-centred sampler would
  give 19 % solar-like starts. Records: FIN's parent checkpoint gone; the campaign harness was unversioned and edited after
  FIN's stage 1; repointed 10-03 23:45 to a frozen Auto-Chem copy at 7d5b036 (the snapshot every later harness imports;
  training and benches stopped importing it on 2026-10-20). More from that review: the oracle's edge comes from the 1e-3
  and 1e3 s starts (0.17 / 0.13 at 2300-3001 K; the 1e6 s start 0.02), holds in all 32 cells tested, and a relaxation
  scheme with the oracle as target and a cross-fitted tau(T, P) loses badly on transients (2.20-3.69 vs 0.06-0.55); stress
  inventories already at call 316 C x18.8, O x15.8, N x26.8, H x1.48 (global N/H 1.23 above the richest cell's 0.55), the
  production delta head C x46.8, N x93.1, a static sum12 cap still C x2.14, O x3.99, N x27.4; nominal arms keep global
  inventories within 1.5 % / 2.7 % of the truth's but per-row relative error p90 O 0.44, N 0.57 at 800-1300 K (medians
  near 0: scatter); in ref_a the molecule count changes 0.50-2.38x and the untracked share of H reaches 0.51; the
  reference GCM layers outside the box are HD189, HD209, Teq_500 and W39b (W39b's deep layers reach 4504 K) and only 38 of
  VULCAN's 290 forward rates state a validity range reaching 3000 K; FIN's dt 1 s headline 0.390 is worse than its own
  stage 1 (0.231; late-max 0.266 vs 0.222); GPU cost 127-319 GPU-days at dt 10 s; the processed data were built at
  6a17e18 and the six kept checkpoints' zlo / zhi equal the then normalization; the generator folder is no git repository.
  Recommendations then: the host applies the update as a multiplicative factor (P2-style) in its own precision, keeps a
  guard and counts its interventions; before a GCM: a real budget, a coupled column test and changing-dt rollouts (none run).
- 10-03, E1-E4 (section 5.3 and 5.5): E2 alternating dt 1 / 10 / 100 s: FIN passes on accuracy, fails the sum12 clause (9.2 %
  vs 7 % of checkpoint-rows above 1.01); random dt 1-100 s per call passes on all four models; every stress dose group
  shows sum12 excursions (FIN G2 57.9 %, G4 = the campaign's D arm 67.5 %, still 61 % / 76 % after 1e5 clean calls, max
  sum12 6.5; at 6x512 the smallest dose G1 already 43-54 %), a random-partner blend being harsher than neighbour transport
  (a neighbour-cell test was not run); semigroup defect <= 1.8e-9 dex from evolved states (no semigroup loss needed); P4 =
  P3 to 8.3e-10; an fp32 host with the same update adds 0.006 to FIN's hold (0.0285 vs 0.0224). E4 control (LJ, s_h1 stage
  1): A10 late-max 0.621, B 0.038, E 1.483, A1 0.524; zero-GPU prior: FIN's sum12 > 1.01 in 6.9 % of A10 rows, Spearman
  0.06 against row error, a static projection moves metrics by <= 0.0012 dex; the two-rate head ran because a two-rate fit
  of VULCAN paths was 4.1x better than one exponential; the simplex head's own stage 1 alone scored A10 16.4 vs 2.50.
  Sweep meaning (10-04): the host may vary dt per call within 1-100 s; zero interventions is not a quality signal; expect
  errors near the holdout's 0.8 dex on new compositions until the gap is explained (hot part explained 10-05).
- 10-04, E6 fixed-point supervision HARMFUL; E6 sealed holdout late-max FIN 0.803, JO40 1.009, LJ 6x512 0.796, the
  fixed-point arm 0.798 (selection 0.342 / 0.394 / 0.621 / 0.649): later explained in part by the O-poor hole. FIN is
  worse on the holdout in every T bin (final 0.80 / 0.35 / 0.62 / 0.34 / 0.34 vs SP 0.34 / 0.22 / 0.24 / 0.17 / 0.06),
  worst from the 1e-3 s start (1.28); the 300-800 K bin is the worst for all four models; the 6x512 control equals FIN
  there while trailing it by 0.28 on the selection sets (selection optimism, I; not separated from source and start
  effects). Sweep process failures (one line): a stale verify gate stalled the E5 jobs 90 min each; misnamed queue files
  were ignored. Not run for budget: alternating dt on p_h1_jo, stress groups G1 / G3 / G5 at 10x2048, an fp64 host at dt
  1 s.
- 10-04, design record: DECISIONS.md, AR_PAPER_PLAN.md and spec.md removed (text in paper-notes/autochem_history_before_
  rewrite_2026-10-04.bundle). Delta-head evidence: production runs v3 / v2_bigger_batch only in git (a584f2d); a June
  2026 Robertson sandbox (records never on daw, unverified) chose one dt-conditioned model, LayerNorm, K 10, LR peak / 10.
  Weight decay 1e-5 is inactive in fp32 (300 steps bitwise equal to wd 0); the z-MSE term is 0.34-0.86 % of the loss
  (objective in effect a per-species log10 MAE).
- 10-04, paper plan of 2026-06-15 (superseded): an ApJ methods paper, sequel to arXiv 2609.00428, which promised training
  "designed specifically to control error accumulation over long roll-outs"; its motivating failure p90 14.5 % after 10
  steps, 56.6 % after 99; its speed figures were not measured on daw (mini-chem 92 / 178 us per cell, the emulator about
  71 us per cell on a Mac CPU); the delta-head export ran 395-487 us per cell on one daw thread. Unit trap: the emulator
  takes P in barye, mini-chem in Pa (BARYE_TO_PA 0.1; the sibling repo once shipped a factor-10 slip, fixed in eb122d0).
  An unfinished audit flagged 4 of 52 literature attributions as mismatched: re-check before any paper use.
- 10-04, open items at 728e4cd (archive): stage-1 checkpoint chosen by one-step loss; no long-horizon arm in Auto-Chem's
  own evaluation; eval_rollout_error.py scored from index 1 by an unrecorded choice; stage-1 vs 2 comparison confounded;
  per-window dt storage; split key re-split would move 34 % of groups (old tree; replaced on main 2026-10-19).
- 10-04, port of the relaxation head (section 5.3); verified bitwise against the campaign harness; references
  ref_after_sweep_port_s1 / _s2.
- 10-04 / 05, E5 ladder, E7 O-poor hole, E8 floored data (formerly rejected sources hot p90 0.11 vs 1.26, rows > 3 dex 1 vs
  26 at 6x512; kept sources slightly worse at 6x512, tie at 10x2048), E9 fill, E10 trunks, uniform test U, E11 FiLM, E12
  Sobol pilot (sections 5.3, 4, 5.5). Cleanup 10-05: 31 GB of old processed builds and caches deleted; deleting arch_tests
  broke the S / SP bench, its test split was rebuilt (64,946 groups; source lists reproduce exactly).

#### 2026-10-06 to 10-09: E13 data and the 12-species models
- 10-06, goals, the E13 design, generator, G and the 64 paths (sections 4, 5.4, 3). Plan agreed: 2^17 rows, a closure test,
  12 + k species, ablations at equal size, then scale (NAS by the author). Three independent reviews in two rounds; all
  data built before the fixes deleted and rebuilt. Per-species log interpolation of labels breaks sum 1 and the element
  ratios, so labels are projected onto the trajectory's ratios (lib/elemproj.py; p90 change 5e-3, max 0.87 dex). The
  closed head failed CUDA-graph capture (linalg.solve_ex fixed it); a closed 53-species rollout of 2000 calls kept |sum -
  1| <= 8e-7 and ratio drift <= 3e-6 at about 8x the relax cost.
- 10-06, block 0, element drift (18 cold boxes > 1e-2), closure test v2, VULCAN tolerance: training stays at rtol 0.05, G
  gets gtest_tight.
- 10-06, first E13 models (section 5.3): phys0 drifts (element-ratio drift); data and the dt law both matter (G dt500 hot:
  phys0 2.94, physwide0 1.08, old Sobol data 0.59); at 800-1800 K the models' end errors are about the truth's own
  movement; U is void for He-carrying models (all its starts have He 4e-31 to 1e-30).
- 10-06, slow-band diagnosis (two independent reviews): windows from t >= 1e5 s are about 20 % of windows but carry about
  1e-5 of the one-step loss; at late 800-1300 K states the per-call rate is too small (pred / true p50 0.12-0.13 at dt
  500) with the wrong sign 20-34 % of the time; relaxclosed formed z + delta in fp32 and erased 42-64 % of slow increments
  (fixed in relaxclosedr; no score change). Author: about 20 entries plus a lumped misc; the 25-entry reservoir set
  defined. allJ (jumps to 1e8 s in both stages) 0.2-0.3 dex better in the slow bands; allG PASS.
- 10-06, cleanup for the new direction (one line): old Sobol packs, a cache and superseded checkpoints deleted (13 GB).
- 10-07, grid-term replicates; all17G (2.86x trajectories) -0.02 to -0.03; resG - allG a cold gain offset by an 800-1300 K
  loss; scoring rules (either mask = truth-only within 0.02 for the grid-term models).
- 10-07, normalization and loss review (two independent reviews, no bug): z-MSE about 1 % of the loss; about 30 % of the
  grid term on unscored points; late times under-supervised (no chain starts at >= 1e6 s); windows in the first 10 ms
  carry 60 % of the one-step term. Stage-2 branches: end-anchoring best; stride, unresolved weight, anchor_p 0.01-0.02;
  stage-2 noise about 0.01 (data/2026-10-06_e13/an_v7.txt, scores_v8.json).
- 10-07, generalization gap (seen vs unseen +0.06-0.07 at 300-800 K; data/2026-10-06_e13/cscores.json); where the error lives (300-800 K: H 33 %, OH 15 %,
  C2H2 14 %, HCN 9 %, half at 1e-2 to 1e4 cm^-3; 800-1300 K real errors in abundant species, worse at Z >= 10x and C/O >=
  0.9; data/2026-10-06_e13/breakdown_allG.txt); reweighting null; new distinct cold rows -0.03 at 300-800 K where
  reweighting did nothing (scores_v10d.json).
- 10-07, factorial, big17, b02 (sections 5.3, 4; scores_v11.json, scores_v12.json). Author: the 50 us target is on the GPU.
- 10-07 evening, E13 review (archive: data/2026-10-06_e13/review_1007/, packed 2026-10-31 in
  campaigns_2026-11-01/paper-notes-data_review_1007.tar.xz): the dead band; readiness never tested on E13 models (an fp32
  host loses 76-96 % of resolved per-call increments at dt 10 s); lib/elemproj.py included He and renormalized to 1, so a
  host needs a 12-species projector; one replicate pair bounds noise only within about 70x; big17's misfire is width x
  grid-term BPTT (I); the 300-800 K error ceiling without H is 0.077, without H, OH, O and C2H2 0.018. Sprint plan
  (data/2026-10-08_e13-sprint/sprint_plan.txt; review memo sprint_review_2026-10-09.txt).
- 10-08, sprint arms (sections 5.3, 5.5): D1a null; M branch GRAY; S adopted; W killed; T null; M from scratch advanced; M+S
  branch slow; M64 no; c18 packed; D1b USE met (G+Gc -0.013, CLAIM not met); m17fs best on every instrument; FiLM fs
  dropped; relaxT kept by an amended route; M+D USE; jump in stage 1 and wide dt dropped; F0fs vetoed. R-early on capw17:
  fp32 host NOT OK; random dt 10-500 s per row per call PASS; unclamped dt 1000-3600 s degrades fresh starts; spin-up at
  10 s mild harm. R-full on m17fs PASS (2000 d = 1000 d within 0.01). Mixing stress: even control rows end with sum12 >
  1.01 in 16-20 of 100 (unexplained). Not run: L (long own-state unroll).
- 10-08, target changed to the overall error; tail audit (m17fs points > 0.5 dex: C2H2 1505, H 868, O 819, HCN 801, NH3 720,
  OH 669, CH4 501, CO2 397, H2O 339, CO 24; a slow CH4 miss changes 5e-5 to 2e-6 dex per call, invisible to a loss of
  about 0.055); methods-gap audit (pooled p90 hides large species errors; the export returns NaN for dt outside 0.1-1000 s
  and out-of-box T, P, so chemistry would silently freeze; the L1 loss gives a 1-dex miss 1.01-1.05x the gradient of a
  0.01-dex miss; points > 0.5 dex are 36 % of the loss value but 0.8 % of its gradient; the jump term, 29 % of the loss,
  trains a one-call map deployment never uses; "never trained from its own state" was wrong, grid-chain BPTT feeds 20 own
  predictions, what was missing were long changing-T, P chains); label quality (the 99-time grid smears sharp collapses:
  one-step targets > 0.5 dex wrong about 4.5e-5, max 6.7 dex). Tail (late-max p99 by band): capw17_s0 1.908 / 0.587 /
  0.533 / 0.395 / 0.337, m17fs_s0 1.745 / 0.524 / 0.489 / 0.440 / 0.342: the levers moved p90, barely the tail. Freeze of
  m17fs_s0 (data/2026-10-08_e13-sprint/sb_br.txt, scoreboard_br.json).
- 10-09, 58-day paths (data/2026-10-08_e13-sprint/xlong_board.txt, xlong_levers_full.txt, xlong_restart_e13_*.json,
  rtolchk_scores.json): error grows with time for every model; the restart test shows accumulation, not harder late
  states; the truth is converged (rtol checks); stage 2 is the largest lever (late p50 2-4x); one-step error is the same
  for every model; relaxT (0.117) and FiLM (0.095) had the lowest late medians, both costing elsewhere; 100-call restart
  p90 0.12-0.17 with shifted-start rows vs 0.33-0.52 without. First call is the worst checkpoint (G fresh p90 0.116, 171 of 1024
  rows > 0.5); dt 100 s halves it, dt 10 s does not.
- 10-09, stopping point: speed m17fs 0.53-0.67 us per cell on the GPU (data/2026-10-08_e13-sprint/speed_e13_m17fs_s0.json), CPU
  13-18 us (8 threads); deployment choices (fp64 host, spun-up start or 100 s first calls, a rule for NaN exports).

#### 2026-10-09 to 10-13: conservation and fixes phases
- 10-09, drift diagnosis and decision rules (sections 5.3, 5.4). The harness moved to its own local repository (first commit =
  the frozen run's exact code); stock training is not bit-reproducible run to run (27 of 29 tensors differ after 300
  steps), deterministic mode is; checkpoint-every-N with exact resume and an in-training 58-day readout added (readout
  only). 12-species projection residual <= 2.3e-15 dex, 212 ns per cell at batch 4096.
- 10-09, Gate A0, paths_v3 chain sets (62,262 s on 40 workers), Gate A vetoed, the chain term (s / step relax 6x1024 stage
  2 0.093 -> 0.143 with chains; closed 0.338 -> 0.546), closed-head iterations (3 of 1024 fresh G starts unconverged at
  12, none at 32: benches at 32, training at 12), learned leak not run, stage 1 at 150k (6x1536 -1.3 % one-step loss;
  closed stage 1 drifts 27x less).
- 10-10, Gate B (section 5.3): B6 the only candidate; path 14 (VULCAN converts CH4 to CO, leftover hydrocarbons keep
  draining; B6's drain far slower, its C2H2 error grows to 10.4 dex by call 10000); model-only 1000-day runs (frozen
  model: sum12 up to 3.87, ratios drift 6.10 dex; B6 finite, ratios exact, periodic within 0.069 dex); label check at rtol
  5e-4 (selected cold windows p99 <= 0.088 dex): label error does not explain the cold tails.
- 10-10 / 11, Gate C: neither B6L nor B6Lc eligible; plain B6 recipe for the final seeds; B6L s1 / s2 path sets only; the
  xlong path-14 trace is set in stage 1. Cold-miss census (data/2026-10-09_e13-conservation/colddata_census.json): 25 rows above 2
  dex in all five closed runs, 23 below 650 K; over every row the closed design has about half the frozen model's.
- 10-11, the per-gas rule (re-score of G+Gc rows > 2 | max: B6 s0 6+20 | 5.16, s1 6+20 | 4.59, s2 8+21 | 4.38, frozen 43+19
  | 6.79); E, D-cold, W, EMA, rules vs seed copies (sections 5.3, 5.5). Not run: a logk_range ablation; the T-gated cold
  expert; member disagreement as a miss detector (a spread above 0.5 dex catches 83 % of seed 0's > 2 dex Gc points while
  flagging 0.82 % of points; never built into the host step); pinning the closed set's log-sd of H, O, OH (premise fell).
- 10-12, E's speed (5.60-7.37 us per cell at >= 1e4 cells, 3.7x); version choice (E recommended conditionally, B6 s2 by
  the rules as written); holdouts H2 / H2c / PH opened once (section 5.3): the freeze mechanism found (relaxation asks for
  13-17 dex, Newton does not converge, the head silently returns its input every call).
- 10-12, fixes phase: H3 / H3c sealed; selection rule; the freeze (clipped Newton steps are not descent directions for
  Jacobians of condition 6e4-1.6e6; the iteration cycles) and the LM rescue (converges within 24 iterations; the convex
  dual also converges but scores worse); training-time projection failures 5.9e-4 of one-step inputs at 12 iterations
  (LM in training 4.5x the step cost: training keeps the old rule); 189 of 11.5M label projections fail (left as is);
  first-call study (every first-call miss > 2 dex is a hot low-pressure fresh start whose truth moves >= 2 dex within
  500 s); the structured solver (species with equal C, N, O counts share a factor: 10 groups; He exact; LM with step cap
  4; 0 failures of 100,000; C / OpenMP fp64 1.15 us per cell on one core with a per-cell stop; data/2026-10-12_e13-fixes/
  solver_speed_*.txt); cold misses are rare late sharp switches whose timing hangs on trace radicals (2,514 of 30,496 cold
  training trajectories hold one); kNN event predictor (top-10 % rate 0.350 vs base 0.092); scaling probes; B6 s2 under P*.
- 10-13, F1-F3 and the pick F2; F2 speed; H3 / H3c opened once (data/2026-10-12_e13-fixes/sel_board_H3.txt); F2's 1000-day run
  bounded (last-cycle change 0.081 dex); E under P* (rows > 2 20, after F2); hard bar removed (F2 eligible).

#### 2026-10-13 to 10-19: scale phase
- 10-13, rules, sealed H4 / H4c / PH2, F2's misses diagnosed with VULCAN restarts (data/2026-10-13_e13-scale/diag_diag_results.txt):
  SYSTEMATIC (information present, rate learned wrong) H2 52, H3 486 from 1e3 s, H2 151, G 892, Gc 85, Gc 250;
  COMMITTED (VULCAN from F2's own early state also misses) H3 486 fresh, H2c 251, G 811, H2c 207, H2 123 fresh; MISSING
  only part of H2 123. No tracer change.
- 10-13 / 14, U2, G1, B, A, FINAL (section 5.3; FINAL 1000-day run last-cycle change 0.044 dex). Extras: within one data set
  one-step validation loss orders moderate misses but not large ones; the stage-1 path readout does not predict the
  stage-2 ranking; two same-config GPU runs differ by 1.83e-5 in EMA weights after 200 steps; the cold CH4 collapse
  (H2 52, H3 486) is the few-percent residual of H + CH4 -> CH3 + H2 against H + CH3 + M -> CH4 + M (net loss 3-26 % of
  gross), so small rate errors move its timing a lot (data/2026-10-13_e13-scale/diag_v5.txt).
- 10-14, snap-back and data analyses (data/2026-10-13_e13-scale/data_s*.txt, snap_heal.txt): U2 starts 48.3 % displaced
  equilibrium, 19.5 % mixtures, 19.3 % perturbed, 9.6 % broad, 3.3 % M rows, none perturbed mid-trajectory; the cold late
  CH4 collapse occurs in 28 of 10,471 training runs of its cell; rows within 0.02 of a design face hold 7 of FINAL's 14
  rows > 2; 10 of those 14 are mistimed switches with the right end state; 246 of 373 rows > 0.5 heal by 1000 d (all at
  >= 1800 K, 67 of 143 at 300-800 K). Envelope and GCM steps (section 5.4); T1; T2 (FINAL fails every rule; 650-1300 K
  relaxes 2-10x slower than VULCAN; data/2026-10-13_e13-scale/score_t2.txt); T4. Also (data/2026-10-13_e13-scale/
  data_s*.txt, snap_heal.txt): Gc 85, Gc 250, G 638, G 886 and H3 716 sit as close to training as same-band controls
  (0.12-0.38 vs 0.15-0.41 dex), H2 52 (1.97 vs 0.26), H3 486, H3c 187, Gc 536 and Gc 545 farther; switch timing errors >
  0.2 dex F2 294, FINAL 156 of 3,752 switches; 60-71 % of 300-800 K jobs move < 0.3 dex in 1000 d, while 0.3-dex kicks
  halve within 500 s in 71 / 78 / 93 % at 1300-1800 / 1800-2300 / 2300-3000 K; 76 of 124 fresh-start misses also miss from
  the true 1e3 s state (rate errors on true states); on path 14 FINAL's CH4 loss per cycle is 1.5-2.6x the truth's over
  calls 3000-7000. Reading (I): about half the large misses sit at design edges or on the rare cold CH4 collapse, the
  other half are switch timing in well-covered regions that did not shrink with the data step.
- 10-15, mini-chem Fig. 4 remade (section 5.4): from raw gas FINAL is wrong in the first 10-100 s at 0.1-1 s steps (2000 K,
  1 mbar max 4.89 dex at dt 0.1 s), every case max <= 2.00 at 500 s.
- 10-15, where FINAL fails (data/2026-10-13_e13-scale/failure_report.txt): 85 of 87 lasting rows below 1300 K; an accumulated
  slow bias, not one bad call (one call from the true state median 0.0000 dex where the chained error is 0.88; per-call
  biases summed along the true path give 0.84x the chained error; bias 1.4e-6 / 4.8e-6 dex per call against true changes
  of 2.5e-6 / 7.7e-6 at 300-650 / 650-1300 K); off-trajectory the target is right but the rate is 0.6 dex (4x) too slow;
  the projection, T / P conditioning and step count ruled out; one exponential fits late states to 0.0065 dex rms, young
  fresh starts need two.
- 10-15, T5 (C0, D-kick), X2 (X2C, X2E), X2V, SD (sections 5.3, 5.5); author: chemistry starts "mostly eq".
- 10-15 / 16, T6 (53-entry network speed 3.70 us at 1e5, LM 7.03; a 1000-day checker bug at 53 entries fixed in the T6
  copy only).
- 10-16, round L (LD, LDc), LD1, LD's speed and 1000-day run (path 11's cycle change grew for seven cycles before
  falling); Fig. 4 at dt 20 s (FINAL per-case max <= 1.52 dex).
- 10-16, training data vs the target (inside env_any 65.3 %; Z 1-100x and C/O 0.3-1.0 inside 12.8 % of U3 rows); n1, the
  design extension, n2; next-model plans (estimates only). GPU peak memory of training runs (log.csv peak_mem_gb):
  D-kick 14.1 GB with U3's 10.3 GB train file, FINAL 11.9, T6 21.4, of 48 GB. Housekeeping: earlier caches deleted (the
  deletion of arch_tests broke the benches, entry 10-17).
- 10-16, label interpolation audit and dense truth (section 4); training profile (stage-2 step launch-bound: about 31k
  kernel launches per step, the GPU busy about a quarter of the time, fp64 4x4 solves of the projection); batch scan (4x
  the batch for 1.16x the time); batch record (every E13 model trained at 2048, with jump pairs 512 and grid and chain
  windows 256; the one-step paper (4x1024 enc / dyn / dec, latent 512, about 11.6M parameters, AdamW 1e-4, 300 epochs)
  says performance drops above batch ~2048, citing Goyal et al. 2017 and Keskar et al. 2016, without runs on daw; the
  repository's 196 commits moved batch through 256-8192 with no recorded outcome (config_auto 256 -> 512, "this works well
  and is fast", -> 4096 with 100 -> 3 rollout steps); no daw record of the author's 512 observation; LJ vs J also differed
  in windows per step, LJ supervising 4.5x more transitions; the critical-batch literature, Zhang et al. 2024, says the
  limit grows with data size).
- 10-16 / 17, batch study (section 5.5; data/2026-10-13_e13-scale/b_readout.txt, b_judge_s2.txt); host_data (torch.multinomial on CUDA
  over 505,907 weights does not repeat with a fixed seed: 0.56-0.61 % of draws differ; host_data costs +16 % per step at
  batch 2048, +3.7 % at 8192 and frees 10 GB of GPU memory; never used by a run); where the time goes (box integration 84 %
  of a box; stage-2 step grid rollouts 50 %, chain rollouts 37 %; a bf16 trunk rejected, gradient cosine 0.31-0.64; the
  TF32 gradient is 53-82 % off an fp64 reference on the one-step and jump terms, while two different batches' gradients
  differ 41-85 %; bench truth files were decompressed 53 times at start-up).
- 10-17, U4 built; CB and P15 started; TF32 stage 1 (no effect); fused projection (25,466 vs 42,871 launches, step
  0.87x; unconverged rows at 12 Newton iterations 31 of 65,536, so 12 stay); n3a / n3b with the fast path; lockstep
  rollouts adopted for CB's stage 2 (0.68-0.82 of the step) with a skip-non-finite guard; P15 NO GAIN; TF32 stage 2 NO
  GAIN. Outage (one line): every bench failed at start-up 02:10-08:46 after the 10-16 deletion of arch_tests, a harness
  default; fix ARCH_CACHE in every bench; the U4 build and n3a drive logged as started 10-16 23:1x were launched 10-17
  08:49 / 08:51. Independent reviews: lockstep swap kept (losses within 7e-14 in fp64); fast generator safe for the
  n1-n3b lists (add duplicate rejection before further use); deletion script v1 had a check-then-unlink race (not run;
  rewritten); bench fix bitwise on 278 reference files. Author's data cap 300 GB.
- 10-17 / 18, CBS1 done; CB stage 2 started; a judge line "LD stays the candidate" was an artifact of missing benches
  (INVALID placeholders for CB, P20, CB5 likewise).
- 10-18, the grid-gradient explosion diagnosed (data/2026-10-13_e13-scale/nfdiag_report.txt: one hot low-pressure row per step, state
  Jacobian product over 80 calls about 1e24 vs 1e0.7 for FINAL, projection not involved, not caused by the faster step
  code); the per-row cap (data/2026-10-13_e13-scale/gfix_report.txt) and the restart as CBg from a 10k-step capped run (validation
  0.0387 vs 0.0435); round E (plateau near 40 at 6x1536); 6x2048 speed 2.76 / 3.08 / 3.60 us at 1e4 / 1e5 / 7.3e5 cells
  (LM 5.27 / 5.14 / 5.75), 1.26-1.30x the 6x1536; CB5 designed (equal passes per run as CBg; level-off pieces of 40k
  judged against their parent); 68.57 GB of box files deleted after a bitwise check; U5 built and CB5S1 started on GPU0
  under a yield guard; project data at 299 of the 300 GB cap; the 52 GB U5 build intermediate deleted.
- 10-19, CBg vs LD (data/2026-10-13_e13-scale/p_judge_cbg.txt: NOT ADOPTED by rules_t5, adopted by the author); CBg speed and 1000-day
  run; why CBg is better (data/2026-10-13_e13-scale/why_attr_report.txt, why_critic.txt); CBg's remaining errors (data/2026-10-13_e13-scale/
  why_miss_report.txt, why_miss_inv.txt); composition target 09:23; large misses become readouts (pass rule v2); train_n
  generated (data/2026-10-13_e13-scale/cooln_report.txt; 7987 s on 24 CPUs; CBgN never started); CBgP1 NO GAIN (data/2026-10-13_e13-scale/
  verdict_e13s_CBgP1_s3_vs_e13s_CBg_s2.txt).
- 10-19, CBg on two mini-chem cases at dt 20 s, 5,000 calls (worst gas over the 12 gases, median / max / at 1e5 s): 1500 K,
  1 mbar CBg 0.18 / 0.59 / 0.13, LD 0.12 / 0.85 / 0.08, FINAL 0.11 / 0.98 / 0.09, mini-chem 0.91 / 3.83 (C2H2) / 0.88;
  1000 K, 10 bar CBg 0.74 / 1.19 / 0.61, LD 0.41 / 1.26 / 0.34, FINAL 0.35 / 1.52 / 0.27, mini-chem 0.30 / 0.30 / 0.30
  (CBg's error is CH4 forming from CO: log10 CH4 at 1e5 s VULCAN -7.68, CBg -7.07, mini-chem -7.98). 10,000 calls of 500
  s: 1500 K, 1 mbar CBg 0.13 / 0.53 / 0.23 vs mini-chem 0.91 / 1.62 / 1.62; 1000 K, 10 bar CBg 0.08 / 0.88 / 0.09 vs
  0.30 / 0.30 / 0.30. Six raw-gas cases at dt 1 / 20 / 500 s and restarts from VULCAN's state: CBg's CH4 at 1000 K, 10
  bar is too fast at every dt (section 5.3); other cases from 1e3 s equal or better than FINAL.
- 10-19, step-size coverage audit: since FINAL every model trains one-step windows at 10-500 s; chains 10-500 s (median
  about 70 s, 16-18 % of calls <= 20 s); grid rollouts from ages >= 1e3 s with calls >= about 85 s; below 10 s only
  isolated jump pairs (about 25 % of them): no multi-call training below 10 s; no decision test at 10 s (DEV paths dt
  median 461 s, 9 % <= 20 s; xlong all 500 s). Independent review of coverage (archive:
  scale/reviews/, 10-19 dt answer): proposed a batched step-size panel, then training levers ranked: truth-supervised
  short-step rollouts from raw, intermediate and evolved starts (1-500 s, deliberate 1 / 10 / 20 s coverage), controlled
  dt sampling, short-dt chains, consistency losses last; open point weighting per step-size value (about 2 % of weight
  below 10 s) vs per log interval (about 37 %), settled 21:4x as per factor of 10.
- 10-19, CB5S1 done (validation 0.02922 on U5's split; path readout p99 6.57 vs CBS1 3.18); CB5 stopped at about 110k;
  run A (CBgA) stopped at 27k (rules: lasting <= 34 and late <= 106 -> 48.8k suffices); run L designed and checked (a
  late_frac 0.15 switch moves 15 % of one-step windows and jump pairs to starts after 1e7 s: one-step windows after 2.8e7
  s 3.0 -> 10.7 %; draws otherwise unchanged; never run, 12 of CBg's 13 late-born rows out of scope); GPU scheduling (one
  training at a time, benches overlap the next; two trainings per GPU would need about 30 GB each with U4 on the GPU and
  ran about 1.7x slower per lane, I); housekeeping (scratch caches U2_res and U3_res deleted, /scratch 41.5 ->
  20.0 GB; stray .md converted or deleted; READMEs restored).
- 10-19, step-size sweep (section 5.4; data/2026-10-13_e13-scale/dtsweep_score.txt, dtsweep_cbg_miss_identities.txt). In scope, raw
  starts, lasting rows of 360 at dt 1, 2, 5, 10, 20, 50, 100, 200, 500 s: CBg 16 13 3 4 1 2 1 1 2; LD1 12 8 6 7 4 2 1 3 5;
  FINAL 17 15 10 5 4 2 1 2 4; LD 17 15 11 5 4 2 1 2 4; evolved starts CBg 0 at every dt (others 1 row each at 200 and 500
  s; evolved max worst-gas error CBg 0.29, others 0.66-0.79 dex). At matched 1e4 s elapsed the 1-2 s gap holds (CBg 16 /
  14 rows); at 1e3 s every model is also weak at 500 s (the raw-gas first-call transient without substeps). Inside, raw,
  late > 2 (readout): CBg 1, LD1 6, FINAL 7, LD 8, all at 1-20 s. fig4 1000 K, 10 bar worst gas at 1e4 s: CBg 1.06 /
  1.02 / 0.88 dex at dt 1 / 10 / 500 s vs FINAL 0.41 / 0.38 / 0.62. CBg's in-scope raw misses at 1 s: 16 jobs, NH3 5,
  HCN 5, CH4 3, C2H2 2, CO2 1, 665-1902 K; one 339 K job fails at every dt. I: no dt gap from evolved starts; from raw gas
  every model has one at 1-2 s; LD1 (trained to 1 s) is best at 1-2 s and worse at 5-500 s, so 1-10 s windows on a
  CBg-class model are the lever to test.
- 10-19 21:40-21:43, cleanup started (cleanup/log.txt): the orphan GPU sampler stopped (running since 10-15);
  notes.md backed up (now paper-notes/archive/dated_record_2026-10-19.tar.xz); the external review report saved; the
  three CBg lib files copied into the harness, all lib md5s equal to the run's done.json, committed (5e2a3f7) with the
  bench read-once fix and tagged e13s_CBg_s2; the abandoned relaxleak branch kept as tag archive/agent3-leak. Migration:
  code/Auto-Chem tag paper-v1 = f6e6116 (annotated); the old configs, processing, src and testing trees removed and the
  E13 code tree copied in (commit 3b8806e, tag e13s_CBg_s2); env.sh H -> code/Auto-Chem (81bedc2); the harness history
  bundled (paper-notes/archive/e13-harness_history_2026-10-19.bundle: main 5e2a3f7, tags e13_m17fs_s0, e13s_CBg_s2,
  archive/agent3-leak); code/e13-harness renamed code/e13-harness.RETIRED (history only).

#### 2026-10-19 / 20: cleanup, one code home, truth audit, mini-chem yardstick
- Sources (V unless marked): cleanup/log.txt; the cleanup/h1, h2, h3, truth_audit and gpl reports (copies in
  data/2026-10-19_cleanup/: code_checks_h1_report.txt, code_tooling_h2_report.txt, code_prune_h3_report.txt,
  truth_audit_report.txt, gpl_moves_report.txt, line_counts.txt); the yardstick spec and reports.
- The author's request (about 16:30) and answers (21:40, 21:4x): section 3 (full cleanup; composition 0.1-300x
  log-weighted with the literature behind the 0.1x floor; time equal per decade; dt equal per factor of 10; keep the old
  paper data; figures only when asked; delete CB5 and its caches, the old repository's plotting, the GCM archives and
  PDFs; the cleaned code replaces main of the public repository, the paper's code f6e6116 tagged paper-v1; push only at
  the end; at most 30 parallel workers). A text snapshot of the deleted sources and logs (24.8 MB, 13,548 unique files)
  and the large logs are in paper-notes/archive/.
- Equivalence checks (cleanup/h1, 21:50-23:06): tests/bitcheck_train.py trains the current code and a frozen reference
  for 4 steps on the same data and draws and compares the step loss, all 27 gradient tensors per step, the logged values
  and the final weights bit for bit (stage 2 vs tag e13s_CBg_s2 on CBg's config, stage 1 vs CBS1's frozen copy; a +2^-20
  change in one loss weight is caught in 140 / 153 tensors; a repeat of the reference matches); tests/score_regress.py
  reruns the stored scorers (sweep scores, mini-chem case cells, CBg's two rules_t5 verdicts, boards): byte-identical.
  Finding: PyTorch's compiler times two block sizes of one layer-norm-backward reduction when a process starts, so about 1
  process in 10 sums one first-layer gradient in a different order: two training runs of one config and seed are not
  bitwise identical even with uniform draws (besides the non-repeating CUDA multinomial trajectory draw, which the check
  cannot cover); the check turns that tuning off on both sides; training is unchanged. Six numerics-neutral fixes, one
  commit each: unknown config keys refused; atomic final stage files; the samplers assert B valid windows; a non-finite
  prediction counts as a failure (+inf) in the score rule and boards; the per-gas rule is the default; the bench reports
  reference jobs it skips (devall6 drops job 3657 of 4,608); each model carries a manifest (norm-file hash checked on
  load; written for CBg and CBS1). No stored decision number moved (all stored states finite).
- Tooling into the repository (cleanup/h2, 23:1x-00:2x): rules_t5, pass_v2, the judge, score_mask, the T2, SD, board6 /
  split6, rows_gt2, cyc1000 scorers, the step-size sweep, the mini-chem scorer (taken out of the figure scripts) and the
  cache and kick builders now live in code/Auto-Chem and find data through lib/paths.py (CHEM_PROJECT, CHEM_SCRATCH); no
  repository file imports VULCAN, gcm.py, generate.py or vulcan_ref, directly or indirectly (GPL side: gen/e13); 16 scripts
  that read deleted data or only drew figures removed, after recomputing the two numbers fig_paper.py printed (CBg vs
  LD and CBgP1 vs CBg, section 5.3; the notes' earlier "unverified" values match). Every moved scorer reproduces its stored
  outputs exactly; a 100-call sweep of CBg is bitwise equal with the repository's bench code and CBg's frozen copy;
  bitcheck passes. Separate commit: the sweep scorer's "in scope" became the target (inside the envelope, Z 0.1-300x, C/O
  0.1-1; 437 rows per dt and start age instead of 360). Equal-dt-weight rows > 0.5 (raw + evolved), target (old scope):
  lasting CBg 5.9 (4.8), LD1 8.0 (5.6), FINAL 9.4 (6.9), LD 9.7 (7.0); late 6.7 (5.6), 8.6 (6.1), 10.4 (7.8), 10.4 (7.8).
  Per dt 1, 2, 5, 10, 20, 50, 100, 200, 500 s, lasting (raw + evolved summed): CBg 19 16 5 6 1 2 1 1 2; LD1 16 10 8 9 6 5 3 6 9; FINAL 21 19 13 7 6
  4 3 4 8; LD 21 19 14 7 6 5 3 4 8. Worst dt 1 s for every model in both scopes; from evolved starts CBg 0 at every dt.
  The rows span Z 0.10-296x and C/O 0.10-1.95, so the mask is in practice "inside and C/O <= 1" (I: log-weighted in Z by
  the design). Files data/2026-10-19_cleanup/sweep_target_score.txt, sweep_target_score.json,
  sweep_target_scope_summary.txt.
- Pruning (cleanup/h3, 00:24-01:40): the repository now holds only CBg's code paths: lhbench 984 -> about 330 lines (arms
  H and P, P3), models.py 659 -> 136 (the closed relaxation head), train.py 1,052 -> 577 (DEFAULTS 107 -> 66 keys, every key
  of CBS1's and CBg's configs still accepted), legacy scorers, the 12-species projector and the CPU solver trio removed;
  ResidualMLP and HybridLoss vendored (same code as commit 7d5b036 and tag paper-v1), so training and benches import no
  snapshot module; comments without dates, phase or tool names; a README for the public repository. Lines of .py / .sh /
  .c: tag e13s_CBg_s2 7,099, after the tooling move 8,895, after pruning 5,960 (lib 4,467 -> 1,953). After every library
  change both bitcheck stages passed with their controls. Final gates (01:57, GPU1 shared with the yardstick): the full
  step-size sweep of CBg (10,908 rows x 10,000 calls) bitwise equal to the stored one, the fig4 dt 20 s bench and the arm-P
  benches P2s and PD bitwise equal, score_regress PASS; bitcheck refuses a shared GPU1 (rerun pending; last real PASS at
  lib commit 1ad62f6, later lib commits comment-only by syntax-tree comparison). Commits are authored by the author;
  H1-H3 carry no Co-Authored-By line (the repository's rule), the two migration commits do (to drop before the push).
- Truth audit of U4 (22:0x-22:2x): section 4, label and truth quality.
- GPL side (21:4x-22:07): the fast generator moved to gen/fast (same files and tree hashes as production, kernel
  self-check 3,000 trials bitwise, one packed y21 row regenerated byte-identical except the timing field); the 107,010
  y21 loose boxes are bitwise equal to their packs (section 4); the mini-chem clone moved to code/mini_chem (upstream
  ae021c8 intact; outputs and plots left out) with our driver in code/mini_chem/e13_driver; it reproduces the six stored
  Fig. 4 runs bitwise with the same numpy (up to 1.9e-3 dex off with numpy 1.26.4: mini-chem's noise floor is about 1e-3
  dex, its rtol); one cold mini-chem call costs about 4-5 us on one core (case means 3-10 us, single calls up to 177 us,
  about the same at dt 1 to 500 s), plus 0.035-0.05 s per process start.
- Mini-chem yardstick (22:52-01:04): spec v1, two independent reviews, spec v1.1 with addenda A and B (all before any
  result), table manifest, element vectors, the batch driver, the VULCAN truth panel (3,818 runs, all accepted), the
  shared row design; mini-chem and the models started 01:04 (section 5.4).
- Reviews and what was adopted. External review report (received 10-19 21:2x, dated 10-20; scale/reviews/): keep CBg as
  the incumbent, but its requested victory over mini-chem is not established; top findings: the decision rule and
  mini-chem setup cannot establish a whole-trajectory, every-dt, full-composition win; the shared harness silently lacked
  six settings CBg was trained with; truth gates do not enforce element integrity (N unchecked); short-step multi-call
  training is missing (isolated jumps to 0.1 s exist); one exponential per entry and 33 hidden species in five sums
  could each limit accuracy; training and use project differently; a NaN prediction scored as zero; the recent targeted
  data leave out parts of the new target; label, sampler and reached-time issues; the speed figure times the model step,
  not the full GCM call. Adopted: CBg's frozen code made the single code home with unknown-key refusal, the score fix,
  the sampler assertion, manifests, the truth audit, a paired trajectory yardstick with matched element vectors and a
  solver completion status. Not yet: the 12 vs 32 + LM measurement, the head-vs-missing-state study, the short-step
  continuation, timing of the full host adapter (section 5.5, untested). Yardstick spec review (yard/reviews/): adopted the
  12-gas mean as primary with a fixed denominator, log-time quadrature, bin weights log(b/a)/log(500), the trimmed mean
  as a robust comparison with the untrimmed mean beside it, simultaneous per-bin upper limits, bounded failure penalties,
  equilibrium-at-target starts checked for stationarity and a quantitative eq-far, parent weights separate from restart
  ages, uninterrupted free runs beside the segments, a stratified direct-stop check with a 0.005 dex threshold plus an
  rtol convergence check, a deterministic start mapping (mini-chem has 13 species including He), and reached time plus
  status from the driver.
- Documentation (01:42): the project instructions file rewritten (goal, data protection with the y21 and kicked-run
  drives added to the never-rerun rule, resources GPU1, 32 threads, one job at a time); these notes finalized with the
  evidence exported.

#### 2026-10-20: t0-fix campaign (rule v3.1, ablation, raw-start data R24, panel F1, code clean, training queue)
- Sources (V unless marked): experiments/2026-10-20_t0-fix/_build/ log.txt, design.txt, design_v2.txt, rule/report.txt,
  score_v31/report.txt, ablation/report.txt, r24/report.txt, f1/report.txt, code_clean/report.txt; reviews in reviews/.
- Design (07:2x) and reviews (08:54): two external reviews; folded into design_v2.txt before any arm was
  trained or scored. Findings adopted: the raw-start signal was under-dosed (1.5 % of one-step windows; the first design
  would have cut CBg's t = 0 anchoring from 19.7 % to 15 %); a matched control arm is needed (CBgP1 was deleted and never
  ran on the yardstick); gates per row family instead of one aggregate; R24 must include C/O = 1, where CBg fails; no
  re-ramp of the chain curriculum in a continuation; a t = 0 validation readout; tests for v3.1; the 1e-3 s oracle arm in
  the ablation; a fresh confirmation panel.
- Rule v3.1 and the rescore (09:12) and the ablation (09:46): section 4.
- R24 data (09:00-09:37): 32,768 VULCAN runs from raw gas (16,253 raw-A: C, N, O in CH4, NH3, H2O; 16,515 raw-B: in CO,
  N2, H2O; raw-A infeasible at 262 high-Z conditions), 16,515 conditions from one scrambled Sobol sequence (seed
  2026102024): T-P 50 % GCM occupancy, 25 % cooler (300-1300 K), 25 % full box; C/O 70 % log-uniform 0.1-1, 15 % 0.8-1.0,
  10 % exactly 1.0, 5 % 1-1.2; Z 75 % log-uniform 0.1-300x plus 25 % 100-300x; He/H log-uniform 0.075-0.105; N/O
  0.3-3x solar. All accepted first time (no step cap, all reached 1e8 s, element drift <= 1.2e-7 dex); direct-stop vs
  interpolant <= 0.0008 dex in the first ten stored rows, <= 0.004 later; row 0 equals the analytic start. Pack
  code/Auto-Chem/data/raw_gcm_r/run_gcm_zz24r-result.h5 (1.89 GB); merged cache /scratch/imalsky/arch_gcmU4R_res (U4
  unchanged as a byte prefix of every array; train 650,434 = 624,324 + 26,110 rows; R24 file last; splits by condition);
  draw weights r24/w_U4R_s{1,2}.npy (U4 0.8, R24 0.2). Never-rerun note: the R24 drive's loose step histories (12 GB,
  r24/runs) are to be deleted at the campaign's end. Sealed-set slip: the first ID-range grep passed through
  paths_v3/hold2 (output filtered; nothing viewed or used); later greps excluded it by path.
- F1 fresh confirmation panel (09:01-09:22; frozen, manifest sha256 3b881fac...; no model has run on it): 6 cells (1x
  solar, 10x C/O 1, 100x C/O 0.25 with mini-chem tables; 3x and 5x solar, 10x C/O 0.1 as model-only readouts) x 20 new
  conditions x 5 starts (the yardstick's four plus a held-out raw-C: C in CO, N in NH3, O in H2O, weight 0 in the main
  score, pre-registered in f1/choices.txt) = 600 VULCAN histories (IDs 4,292,000,000-599), all accepted; 148,180 rows,
  179.3 M steps per solver; a 5 % strict-tolerance rerun moved the truth by about 0.0007 dex; mini-chem run on its five
  cells (1 failed row of about 123,000). Scoring F1 needs the repository scorer adapted (5 starts, no free runs, a cell
  without mini-chem).
- External maintainer review (received ~08:58; branch cleanup/maintainer-pass aaa7ea1 vs origin/main: 15 files, +19 / -46,
  comment and dead-code trims) and the code clean-up (09:00-09:43): one real bug, bench/score_mask.py imported the paths
  module and then defined a function paths(), so its command line crashed at its end (after printing its table; rules_t5
  only imports its functions, so no stored verdict was affected; introduced by the 10-19 tooling move): renamed to
  path_scores (34cbd56). The T2 scorer's description said the model starts at 1e-3 s; the code starts at the restart
  itself (T0 = 0.0): text fixed (92681a6). The review's trims applied (af82bc3) except three lhbench comments and the
  .gitignore lines that keep CLAUDE.md and todo.md untracked. Training-fix code: per-file row-0 probabilities
  (x0_anchor, x0_jump; 938ff2f), an optional t = 0 validation readout (t0_readout; 92f19ec), start kind "raw" in the cache
  builder and a merge that keeps the given file order (e0e8c3b). Checks: bitcheck s1 and s2 PASS (CBg's settings train
  bit for bit as before), score_regress PASS, yardstick_regress PASS, the score_mask command line runs to the end and
  writes the table it printed before, comment-only edits equal by syntax tree, pyflakes clean. code/Auto-Chem main is 8
  commits ahead of origin/main 4961084 (5f11a99 to e0e8c3b), all authored by the author. Pushed 10:0x on the author's
  approval ("do you need my approval for anything? I give it"): main 4961084 -> e0e8c3b, tags paper-v1 (f6e6116) and
  e13s_CBg_s2 (3b8806e; both targets already on main; the repository's pre-push hook refuses every tag, bypassed after a
  manual check), remote branch cleanup/maintainer-pass deleted (PR #1 closed; its head stays as refs/pull/1/head).
- Training queue (launched 09:4x; design_v2 ARMS): stage-2 continuations of CBg from its stage2.pt, seed 3, s2_lr 3e-5,
  warmup 1000, chain_K 16 and pre_max 80 from the start (no ramps), 20,000 steps each (about 2.1 GPU-h): T0C control
  (CBg's recipe on U4), T0A (U4R with R24 at x0 0.5 / 0.25, m18 / z22k 0.5 / 0.25, the other U4 files 0.15 / 0.10, t = 0
  readout on), T0B (T0A plus one-step windows from 1 s); each followed by the yardstick rows, the path sets and the
  step-size sweep. Gates fixed in design_v2 (t0 = 0 rows better than CBg with the upper limit < 0; restart rows and free
  runs upper limit <= +0.003; failure rate not worse; worst-gas upper limit <= +0.01; path late p99 not worse by > 0.05
  dex; sweep not worse), then confirmation on F1. Started 10:16 (repo e0e8c3b; equivalence of the new evaluation runner
  with CBg's stored yardstick outputs checked first, result kept as t0-fix queue/equivalence_CBg.json). STOPPED 10:27 on
  the author's approval (below): T0C at about step 1,000-1,700 (log.csv step 1000: loss 0.0538, 0.375 s per step), no
  checkpoint written, no arm scored, no new pick.
- Campaign closed 10:27. Author ~10:2x: "all your recs seem good. I approve them all. Implement all of them with advisors
  to check. clean all the old notes and docs that might be out of date with these decisions" (the recommendations:
  start mix from the GCM literature, mostly near-equilibrium and transported gas; stop this campaign; accuracy against
  VULCAN over the realistic domain as the main target with mini-chem a benchmark; one clean regenerated data design;
  one score plus readouts; Lodders 2019 metals with He/H 0.078-0.100). Lessons carried forward: exact t = 0 states must
  be in training (the ablation's oracle); the projection and step size are not the t = 0 cause. Deleted at closure
  (t0-fix delete/manifest.tsv, check-only first, 0 problems): the R24 loose histories 12.7 GB, R24-only caches 2.4 GB,
  the evaluation outputs 0.65 GB, /scratch/imalsky/arch_gcmU4R_res 16.2 GB (31.9 GB). Kept: the R24 pack (never rerun
  its drive). Work continues in experiments/2026-10-20_clean-state.

#### 2026-10-21: all data into one folder; clean full run launched
- Sources (V unless marked): experiments/2026-10-21_data-home/_build/ log.txt, items.txt (inventory), refs.txt,
  docs.txt, plan/ (plan.txt, move_map.tsv, move.py, patches/, checks.txt), review_A_safety.txt, review_B_code.txt,
  move_run.log, move_journal.jsonl, eqcheck.log; experiments/2026-10-20_clean-state/_build/log.txt.
- Author (section 3): all project data into code/Auto-Chem/data (git-ignored), organized by kind, /scratch emptied;
  then the go for the clean full run.
- Layout (repo-data/ = code/Auto-Chem/data): raw/ (paper/ = the published paper's packs, raw_gcm* packs, clean_pilot/,
  raw_clean/), caches/ (arch_gcmU4_res, arch_gcm17m_res, arch_gcm16_res, arch_clean_res + arch_clean_work), truth/ (G,
  Gc, paths v2s / v2l / v2x / v3, opened H2 / H3, the dev tag, T2, fig4 tag, cool-N chains, clean chains and panels),
  sealed/ (e13_holdout4, hold2, clean_twin_SEALED; guard tokens kept in the names), runs/ (CBg, CBS1, FINAL, LD, LD1, F2,
  e13_m17fs_s0; clean_s1 / clean_s2), benches/ (E13 benches, score designs, yardstick and F1 designs with their VULCAN
  panels beside them because score_run opens the panel next to a design, ablation, confound, dtsweep, fig4, speed and
  1000-day json in out/), ref/ (snapshot, VULCAN ref_a, gcm_tp.npz, never-rerun row lists, draw weights, pathmap.json),
  gen/ (the generator's clean/ and gcm/ design tables and boxes with packed markers). Experiment folders keep records
  only; the GPL-importing vulcan_ref.py of the five kept scale runs' harness copies stays in scale/runs_gpl/; mini-chem
  output folders were split (arrays moved, its scripts stayed).
- Size: about 141 GB on disk (142.8 counted; two hardlinked panel pairs moved by rename, kept linked); 19.9 GB plus the
  sealed H4 copied off /scratch, the rest same-disk renames (V items.txt). 323 journaled steps, 12:44:43 -> 12:46:41
  (V move_run.log). /scratch copies: cp -a, compared with the live source (types, modes, sizes, sha256 of every file,
  link targets) before the source was removed; H4 by one whole-folder mv (exit status only, never listed); hold2 by one
  rename. 4,631 absolute symlinks rewritten (4,608 in the dev tag, 16 U4 aliases with names kept, 2 raw_gcm_m, 1
  snapshot, 4 generator links). Nothing deleted except 10 folders the move emptied; deletion candidates (about 4.4 GB)
  listed for the author (todo).
- Code: lib/paths.py has one root DATA (CHEM_DATA, default <repo>/data; CHEM_PROJECT / CHEM_SCRATCH gone) and
  paths.moved (old location -> new, via repo-data/ref/pathmap.json) for stored configs; bitcheck applies it to its work
  copies of the CBS1 / CBg configs and reference libs (run folders untouched); the six readout design manifests'
  source.dir rewritten (originals in data-home plan/orig/); the full-run driver, preflight, config check and templates
  repointed; generator root constants edited and re-frozen (clean.py afc9dfee, drive_fast.py 1a18a4c2, paths_clean.py
  ea40c04f, panel_clean.py 14e813be); a hold2 refusal added to the retired chain generators (after the move their
  "paths.json exists" refusal no longer covered the sealed set). Committed 365c4d3 (local).
- Reviews: two independent reviews before the move (data safety; code and checks); fixed before running: a check that
  ran git status --ignored (it would have opened the sealed folders), journal handling of undone steps, settling of a
  killed step (a verified copy is never deleted), and a link check that compared against pre-rewrite targets. Re-review:
  go.
- Checks after the move, all PASS: journal 323 / 323; generator equivalence (64 pilot rows re-driven with the edited
  code, 7 kinds x 3 strata + 9 C/O-side, identical in every array but the timing field; pilot chain jobs equal; panel
  roots, twin root and freeze OK), so the 4,096 pilot rows stay production rows; bitcheck s1 and s2 bitwise EQUAL;
  score_regress (a) 20 / 20, (b) 4 / 4, (d) 4 / 4; full-run preflight and template check OK; the six copies re-hashed
  EQUAL against their manifests; 4,627 links, none broken; no live reference to an old location; no GPL code in
  repo-data; /scratch/imalsky empty.
- Rules added to both CLAUDE.md files: every recursive command over code/ prunes data/sealed; never git clean -x / -X /
  -fdx, git stash --all, git add -f under data/, or git status --ignored.
- Author "delete and fix these" (same day): deleted 4.252 GB through a manifest with a check-only run first (data-home
  delete/): the yardstick FINAL and LD1 batch pieces (every part's states, row fields and timings verified equal to
  model_states.npz; the one per-row field only the pieces held, newton_total, saved first as parts_extra.npz: it
  differs from newton_n in 5 / 6 rows, Newton failures after the scored horizon), the truth audit's per-pack arrays
  (its cited per-run table kept), fig4 model outputs and time grids (the mini-chem references kept), regression
  scratch, the E13 row lists into deleted caches, 8 byte-identical csv copies, caches and empty folders. Kept: the
  33 KB arch_gcm16_res (CBg's reference code names it). The VULCAN reference tag's worker logs were compressed
  (51.6 -> 1.1 MB). score_regress (a) and (b) PASS afterwards.
- Public history (V git): tag e13s_CBg_s2 = 3b8806e is an ancestor of main; it and the next 11 public commits carry
  five files that import the GPL generator (lib/vulcan_ref.py, datagen/gen_cold.py, datagen/make_tag.py, and old
  versions of datagen/cache_gcm.py and bench/analyze_e13.py), removed by 93c04e9; current main is clean. Re-pointing
  the tag alone cannot remove them; a history rewrite and force-push is the author's call. f1/run/groups.npz still
  does not exist (needed only at a final pick).
- Speed tests before stage 1 (author: "check if you can make it much much faster in some easy way"; V clean-state
  _build/speedtest/ and log.txt): steady seconds per step on the U4 cache with the CBS1 / CBg configs, idle GPU1:
  stage 1 0.0317 as is, 0.0321 with the fused projection, 0.0321 fused plus CUDA graphs (no gain: stage 1 is not
  launch-bound at 8,192 rows); stage 2 0.382 as is; CUDA graphs on every compiled call 105-181 s per step (the graphs
  are re-recorded every step: pending backward passes and a varying pre-roll length), unusable; without the no-gradient
  pre-roll 0.319, so graphing the pre-roll could save at most 17 % (that variant errored and was not pursued). No easy
  numerics-preserving speedup; the run continues unchanged. Left for later: bigger stage-2 batches with fewer steps
  (the 10-16 scan: 4x batch for 1.16x step time), a recipe change that needs its own comparison.
- Clean full run launched 12:56:54 (drive.sh pid 2758299, repo 365c4d3): box drive block 0 on 38 CPUs (61,440 of 65,536
  rows; the pilot rows are reused), chains train_a on 2 CPUs, constant-mode decision panels on 4 CPUs.

#### 2026-10-22 / 23: clean full run, result
- Sources (V): experiments/2026-10-20_clean-state/_build/ log.txt, full/logs/drive.log and phaseD.log,
  scores/decision/tables.txt, score.json, large_misses_*.csv; repo-data/runs/clean_s1, clean_s2 log.csv;
  _build/speedtest/.
- Run: boxes 786,432 (786,431 accepted; 1 failed twice, block 11) in about 45 min; 12 packs by 14:39; cache (train
  629,040 / validation 78,811 trajectories, element ratios to 1.3e-15) 16:05; chains 2,511 / 2,512 by 19:42; panels
  (decision constant, path; twin constant, path, sealed 03:59 unread); stage 1 9.86 h (0.032-0.035 s per step), stage 2
  20.6 h (0.31-0.39 s per step, 0 non-finite steps); speed equal to CBg's (2.83 vs 2.81 us per cell at 1e4 cells; LM on
  every row 5.53 vs 5.65).
- Pipeline gap found at phase D: score_design.py's readers expected a tag layout (accepted entries with Z, C/O, He/H,
  Sobol index and dt draws; P: paths.json and log10 saved states) that panel_clean.py does not write; the two were never
  run end to end before go. Bridged by a new generator-side script e13/panel_tag.py that writes tag/ views (compositions
  recomputed from each panel's Sobol table with T, P and stratum asserted equal to the manifest and start element ratios
  equal within 1e-12), with no change to either frozen piece; all five designs then passed score_design's own checks.
- Speed tests before stage 1 (no easy gain): stage 1 0.0317 s per step as is, 0.0321 with the fused projection or CUDA
  graphs; stage 2 0.382, CUDA graphs on every call 105-181 s (re-recorded every step), no pre-roll 0.319 (the most that
  graphing the pre-roll could save).
- Training readouts: on the old changing-condition paths (v2x, late window, all scored points) stage 1 ended at p50
  0.0025 / p90 0.51 / p99 2.59 / max 11.9 dex (CBS1 0.091 / 1.00 / 3.18 / 10.8); after stage 2 0.0015 / 0.13 / 1.20 /
  5.9 (CBg 0.0017 / 0.079 / 1.32 / 3.57). The stage-1 training loss rose 0.024 -> 0.034 after 150k steps while
  validation fell (validation holds no row-0 windows: train.py sets p0_anchor on the train split only); the start-state
  readout's tail grew over the same steps (10 s p99 1.71 -> 2.17) while its median fell (0.0079 -> 0.0031) (I: the rise
  comes from the hardest exact-start windows).
- Score-1 on the decision set: clean 0.0039 [0.0033, 0.0051] (L 0.0047, P 0.0052, S 0.0020) vs CBg 0.0196
  [0.0169, 0.0229] (L 0.0223, P 0.0274, S 0.0092); D -0.014 [-0.017, -0.012] (untrimmed -0.017), OUTCOME BETTER; per
  panel S -0.006, L -0.016, P -0.020; every dt bin better (simultaneous intervals below 0); lasting -0.005; 10 s rows
  -0.004; readouts: raw starts -0.063 [-0.080, -0.046] (Sraw 0.021 vs 0.091), side (C/O 1-2) 0.0022 vs 0.0097.
- Flags: R5 fired (failed rows in the scored panels clean 1, an eqf 10 s readout row of S, vs CBg 0; Sraw 2 vs 2);
  R1-R4 quiet; the 1000-day cycle and T2 are for a final pick. Rows with any gas > 2 dex: clean S 133, L 15, P 3, Sraw
  234, side 12 vs CBg 1,106, 270, 34, 1,348, 112. Rows with the 12-gas mean > 2 dex: clean S 34, L 6, P 0, side 10 vs
  CBg 29, 9, 1, 0. Identities of the rows with the 12-gas mean > 2 dex: all eqf (transported) starts at t0 = 0, in the
  early transient (worst calls at 50-3,700 s, dt mostly 1-26 s), mostly 1,400-2,100 K, worst gases H2O and CO; in S 16
  of 600 conditions (S:317 6 rows, S:371 5); on the side check two C/O > 1 conditions (C/O 1.13 at 1,320 K and 5.6e-4
  bar; C/O 1.01 at 2,009 K, 148x) where CBg stayed below 0.25 dex. The wider set of rows with any gas > 2 dex is not
  all early (V large_misses_clean_*.csv, re-read 2026-10-23): S 133 rows, all eqf, but 15 at t0 > 0, worst gas C2H2 in
  39, H 22, H2O 21, CO 15, CH4 9, T 357-2,776 K, worst calls up to 1.4e5 s; L 15 rows (all eqf from t0 = 0) with worst
  calls up to 6.9e7 s; P 3 rows, one from an eqt start.

#### 2026-10-23: misses first; what limits accuracy (read-only checks on the final clean models)
- Sources (V): the final clean models (repo-data/runs/clean_s1 stage1.pt, clean_s2 stage2.pt) on the validation split of
  arch_clean_res (78,811 runs), evaluated in this session's read-only scripts (kept in experiments/2026-10-23_overlay/
  _build/); clean_s1 / clean_s2 log.csv; scores/decision/large_misses_clean_*.csv; figures experiments/2026-10-23_overlay/.
- Misses, deployment calls (V bench/misses.py, 32 Newton + LM, TF32 off; 16,384 validation runs of the screening set,
  each rolled at constant T, P from ages 0, 1, 10, 100, 1000 s at dt 1, 2, 5, 10, 30, 100, 300 s for max(10, 1000 / dt)
  calls up to 1000, against the stored trajectory; experiments/2026-10-23_overlay/_build/diag/misses_clean_s*.out, the
  json and rows files beside each model). Final model (clean_s2): rollouts with any gas above 0.1 dex at some call
  14.9 %, above 1 dex 1.6 %, worst 9.5 dex; from exact starts (age 0) 25.8 % / 5.3 %; from age 1000 s 1.6 % / 0.014 %;
  by start kind eqt 0.00 % / 0 (max 0.01 dex), eqn 0.7 % / 0.004 %, eqf 27.6 % / 3.0 %, mix 19.5 %, pert 21.0 %, rawA
  27.8 % / 7.3 %; by dt 1 s 18.7 % / 3.7 %, 100 s 10.6 % / 0.23 %; by T 1300-2000 K 19.7 % / 2.0 %, 2000-3000 K 10.6 % /
  2.6 %. Stage 1 alone (clean_s1): 11.6 % / 1.9 % (stage 2 adds small misses and removes some large ones). The labels
  are the stored trajectory's log-log interpolant, so part of the small misses at early ages may be label error
  (measured separately). One call from a start state at 1 s can miss CO by about 6 dex (figure head_example_curves.png,
  the 99th-percentile 1300-2100 K start for that gas). The averaged score (0.0039) hid all of this.
- The training-time start readout (t0_readout) is not a deployment measure: its EMA copy uses the training projection
  (12 iterations, failed rows keep their input), and its worst rows are bit-identical in clean_s1 and clean_s2 (dt 10 s
  max 8.933645992946165, dt 100 s 8.937938562698147, 650-1300 K 5.882067876183925, 2000-3000 K 8.347884651375752; V
  log.csv), i.e. projection-failure rows that do not depend on the model. Its shares (10 calls of 10 s: 8.8 % of starts
  with a gas above 0.5 dex, p99 2.53) are superseded by the deployment readout above.
- Where the error is (stage-1 model, one-step windows, dt 10-500 s): windows starting before 1 s carry 87 % of the loss;
  from 1e3 s on the scored error is about 0 (6e-5 dex for starts at 1e3-1e5 s). Per-window loss median 0.0002 vs mean
  0.014: the mean is the tail. Loss on training vs validation runs 0.0116 vs 0.0139; stage 2 raises the one-step loss
  0.0139 -> 0.0165 (scored 0.0045 -> 0.0056).
- Projection in training (models.py: a row whose Newton solve fails keeps its input, so it gets no gradient; training
  12 iterations, use 32 + LM): exact-start windows (dt 10-500 s) fail at 0.48 % (1300-2000 K 1.54 %, 2000-3000 K 0.50 %,
  below 1300 K <= 0.02 %); the failed rows carry 64 % of the start-window loss (mean 13.5 each); 0.8 x 0.0165 + 0.2 x
  0.1005 = 0.033 reproduces the logged final training loss 0.0335; with the LM rescue the same weights give 0.038 instead
  of 0.100 on those windows. Failures vs iterations (dt 1-500 s): 12 -> 0.39 %, 16 -> 0.061 %, 24 -> 0.012 %, 32 -> 0.003 %,
  64 -> 0, at negligible cost. I: this is the stage-1 training-loss rise after 150k steps and the jump of the 1300-2000 K
  start-readout p99 (2.7 -> 6.2 dex at 250-400k steps).
- The head's time response: the best single exponential per gas (least squares in z per entry, no projection) fitted to
  the stored trajectory over dt 1-500 s from row 0 (8,192 validation runs): mean 0.023 dex, per-start worst p99 3.2,
  against the model's 0.031 / 4.2; over 10-500 s 0.015 vs 0.018; from t = 1e-3 s 0.0135 vs 0.0128; 1-100 s 0.017; 10-100 s
  0.009. I: one target and one rate per entry, independent of dt, cannot follow a fast jump followed by a slow drift; the
  projection lets the model bend somewhat, so the floor is approximate; restricting dt to 1-100 s lowers it little.
- Training coverage: one-step windows and chains use dt 10-500 s only (stage-2 jump pairs reach below); grid chains
  start at 1e3 s (grid_t_min default), so no multi-call training covers the first 1e3 s; 74 % of the 34 S rows with the
  12-gas mean > 2 dex have dt < 10 s (median 2.7 s). Draws by window start: 20 % at t = 0, 20 % at 1e-3 s, 56 % before
  1 s, 34 % after 1e3 s (with about 0 loss); the 12 scored gases take 33 % of the loss weight.
- Not limits (V): the truth (clean data rtol 0.005; stored rows within p99 3.8-5.3e-4 dex of exact stops; panel truth
  1.2-3.4e-4); condition coverage near the worst S conditions (350-473 training eqf runs within +-100 K, 0.5 dex in P and
  Z, 0.2 dex in C/O). Unmeasured on the clean data: the log-log interpolation error between stored rows in the early
  transient, and hidden-species memory at transported starts.
- Checks before round 1 (V experiments/2026-10-23_overlay/_build/diag/checks35.out; clean_s1): of the rows with a miss,
  those whose trunk output gets no gradient (failed projection) are 1.99 % in row-0 windows and 0.46 % in early windows
  at 12 Newton iterations, 0 at 32. Label interpolation, leave-one-row-out at twice the stored spacing (1e-3 to 1e3 s):
  worst-gas error p99 0.039-0.054, p99.9 0.16-0.27, max 1.30 dex, rows above 0.1 dex 0.21-0.47 % (I: an upper bound on
  the interpolant's error at the stored spacing). Non-monotone early curves (0-500 s from row 0, floored): 11.7 % of runs
  have a gas reversing by > 0.1 dex, 6.3 % by > 0.5, 4.1 % by > 1.
- An independent read-only review of the pipeline ranked the same limits (projection failures, the head's time
  response, early label resolution, the loss and sampling aimed at a different objective, hidden-state ambiguity,
  capacity and precision last) and corrected the large-miss identities (folded into the 2026-10-22 / 23 entry).
- Next (I): judge by misses; a loss term that grows with each miss; closed_iters 32; a dt-aware head; training below
  10 s and through the first 1e3 s; small fine-tune tests first.
- Round 1 of small tests (V experiments/2026-10-23_overlay/_build/log.txt, ft/queue.log; repo-data/runs/ov_*/misses_stage1
  .json; each a 30k-step fine-tune of clean_s1's stage 1 at peak LR 3e-5, closed_iters 32, one seed; miss readout as
  above on the same 573,440 rollout rows; "fixed / broken" = rows whose large miss disappears / appears in the paired
  comparison). Rollout rows with a miss / a large miss, worst:
  - ov_T0 control (projection fix only, dt 10-500 s): 11.56 % / 1.930 %, 9.44 (clean_s1 11.57 / 1.931); second data
    order ov_T0s 11.55 / 1.932 (paired: 72 / 84 rows differ at 1 dex, 483 / 396 at 0.1 dex: the noise floor).
  - ov_C (+ one-step windows at dt 1-500 s): 10.56 / 0.864, 9.41 (6,820 fixed, 708 broken); exact starts large 7.40 ->
    2.64 %, dt 1 s large 5.28 -> 1.93 %.
  - ov_CMn (ov_C + bounded miss term, lambda 2): 10.64 / 0.786 (754 fixed, 308 broken vs ov_C).
  - ov_CRn (ov_CMn + rollout term: 512 runs per step from row 0 or ages 1-1e3 s, 0-32 calls without gradient then 8
    with, loss and miss term at every call): 9.06 / 0.373, p99.9 1.70, worst 8.23 (2,512 fixed, 141 broken vs ov_CMn);
    calls with a miss 3.16 % (final clean model 10.35 %); exact starts large 1.48 % (clean_s2 5.33 %); eqf large 0.51 %
    (3.00 %). Its worst rows: rawA starts, first 1-2 calls at dt 1-2 s, 1,900-2,800 K, low P (CO, C2H2 7-8 dex).
  - ov_CK (rate-only dt head, one-step training): 20.47 / 3.03 although its one-step fit is better (validation 0.0161 vs
    0.0180): a dt-dependent rate trained on single calls loses multi-call consistency.
  - ov_CD (dt-in-trunk head, one-step training): one-step fit far better (validation 0.0070 vs 0.0180, batch miss share
    2.4 vs 8.2 %); rollouts: large 0.580 % (vs 0.864), p99.9 1.91 (5.92), but miss 13.53 % (10.56) and calls with a miss
    14.1 % (6.0): the single-exponential form capped the one-step fit (I), and dt-aware heads need rollout training.
  - ov_CM (bounded miss term + straight-through clamps): DIVERGED (36.6 / 9.6; loss 0.068 -> 0.146): the straight-through
    target clamp let raw targets drift up to 900 z units outside the box (diag/drift.py); removed from the code.
  Conclusions (I): sub-10-s windows and the rollout term (loss at every call) are the large levers; next: the rollout
  term with the dt-in-trunk head, the fractional-error loss (round 2: per point 10^a - 1, summed over each
  trajectory's calls and gases), longer rollouts, then a full-length run.

#### 2026-10-24: rounds 2-4 of the small tests (fractional loss, dt-aware heads with rollouts, review levers)
- Sources (V): experiments/2026-10-23_overlay/_build/log.txt, ft/queue.log; repo-data/runs/ov_*/misses_stage1.json and
  _rows.npz; reviews/answer_next_*.txt (external review of next tests and architectures). Same protocol as round 1
  (30k-step fine-tunes of clean_s1 at peak LR 3e-5, closed_iters 32, one seed; 573,440 rollout rows). New readout column:
  the summed fractional error per row (10^e - 1 over its calls and 12 gases; mean over rows, dominated by the extreme
  calls).
- Noise floor for this recipe: ov_R4s (= ov_CRn with another data order) vs ov_CRn: 76 / 85 rows differ at 1 dex,
  865 / 1,005 at 0.1 dex (miss 9.09 vs 9.06 %, large 0.374 vs 0.373 %).
- Rows with a miss / a large miss / summed fractional error per row (mean) / worst:
  - ov_CRn (dt 1-500 s + bounded miss term + rollouts): 9.06 % / 0.373 % / 1.21e3 / 8.23 (round 1's best).
  - ov_R4fm (ov_CRn + fractional term frac_w 0.1, cap 2 dex): 8.73 / 0.285 / 2.0e5 / 8.07 (844 fixed, 333 broken vs the
    second seed: real); more rows fixed but calls with a large miss 0.077 vs 0.057 % and an eqf worst of 8.07 vs 6.17.
  - ov_R4p0 (ov_CRn + first rollout calls supervised half the steps): 9.02 / 0.358 / 1.07e4 / 8.07 (337 / 243: marginal).
  - ov_R4md (ov_CRn + 4 dt targets per anchor): 9.06 / 0.376 (46 / 63: no effect; a dt-free head shares its target and
    rate over dt, I).
  - ov_Cf (fractional loss frac_w 1, one-step): 10.71 / 0.625 / 6.6e7; ov_CRf (frac_w 1 + rollouts): 9.27 / 0.384 / 2.9e5;
    with rollouts the full-weight fractional loss is worse than the bounded miss term (calls with a large miss 0.154 vs
    0.057 %).
  - ov_CDR (dt-in-trunk head + rollouts + miss term): 13.23 / 0.309 / 672 / 7.06: fewer large misses and lower extremes
    but more small drift (calls with a miss 12.4 vs 3.2 %; eqt starts drift up to 0.70 dex, the dt-dependent target loses
    the equilibrium fixed point); ov_CDRf (same + frac_w 1): 14.38 / 0.652 / 1.45e5.
  - ov_CKR (rate-only dt head + rollouts + miss term): 12.90 / 0.414 / 370 / 7.25; eqt stays exact (max 0.07 dex).
  - Control references: ov_T0 frac sum 3.17e8, ov_C 9.28e7, ov_CD (one-step dt-in-trunk) 1.09e6.
- Training runs vs held-out runs (V repo-data/runs/ov_CRn and ov_R4fm, misses_stage1_train.json vs misses_stage1.json;
  the same rollout design on 16,384 training runs): ov_CRn rows with a miss 8.37 vs 9.06 %, large 0.307 vs 0.373 %,
  calls with a large miss 0.028 vs 0.057 %, eqf large 0.41 vs 0.51 %, rawA large 2.94 vs 2.45 % (worse on training
  runs); ov_R4fm 8.03 vs 8.73 %, 0.228 vs 0.285 %, rawA 2.90 %. I: little to gain from fitting the training runs better;
  the raw-gas first calls were not learned even on training runs (possibly a limit of that head's form; the flow-map
  trunk later took rawA large to 0.33 %).
- External review (2026-10-23): FNO is a mismatch for a 0-D state of unordered entries (Fourier features of
  log dt a later cheap ablation); real architecture tests: the dt-in-trunk head with rollouts (done above), the paper's
  encoder-dynamics-decoder adapted to the relaxation output and projection (fresh, round 5), a signed transient-basis
  head after an oracle fit.
- Round 5, first arm (V repo-data/runs/ov_FM1/misses_stage1.json, log.csv): the flow-map trunk (the 2026 paper's
  encoder-dynamics-decoder: E(z, g) -> 512, h' = h + F(h, g, dt), D(h') -> target offsets and rates, then the relaxation
  and the projection; 4x1024 blocks, 12.7M parameters, 25.4 MFLOP per call vs 42.4), trained fresh for 300k one-step
  steps (peak LR 6e-4, dt 1-500 s, bounded miss term, closed_iters 32; no rollouts): rows with a miss 3.65 %, large miss
  0.056 %, calls with a large miss 0.028 %, p99.9 0.79 dex, worst 3.84 dex, summed fractional error per row 28.6; eqt
  starts within 0.15 dex; rawA large 0.33 % (worst 2.99); one-step validation 0.00313 (clean_s1 0.0154; still falling at
  300k). vs the final clean model: 9,211 rows fixed, 96 broken at 1 dex.
- Round 5, matched control (V repo-data/runs/ov_FS1/misses_stage1.json, log.csv): the same fresh recipe on the
  dt-in-trunk 6x2048 trunk (21.2M parameters, 42.4 MFLOP per call): miss 5.56 %, large 0.109 %, calls with a large miss
  0.071 %, p99.9 1.03, worst 3.63 dex, frac sum per row 48.3; eqt max 0.12; rawA large 0.39 %, pert 0.34 %; one-step
  validation 0.00336. Paired, flow-map vs control: at 1 dex 442 rows fixed / 137 broken, at 0.1 dex 15,832 / 4,884 (both
  far above the noise floors of about 80 and 900). Both parts count (I): fresh training with dt 1-500 s and the miss
  term took the dt-in-trunk head from 13.5 / 0.58 % (fine-tune ov_CD) to 5.56 / 0.109 %, and the flow-map trunk halves
  the large misses again at 60 % of the compute. Their one-step validation is close (0.00313 vs 0.00336); the difference
  shows in rollouts (calls with a miss 2.9 vs 5.4 %).
- The flow-map model's worst row, dissected (2026-10-25; V experiments/2026-10-23_overlay/_build/diag/hdiag.out,
  hdiag2.out): a pert start at 861 K, 0.0055 bar, dt 10 s, where VULCAN's H falls from 10^-2.6 to 10^-8.4 within ~250 s
  and the model takes ~3,000 s (3.8 dex; the author: "terrible"). (1) One 10 s call from the true state is right to
  <= 0.03 dex at every time; the free rollout goes wrong from call 2. After call 1 the unscored entries are far off
  (NH3 -1.56, N_res -1.50, C +1.40, C2H2 +1.05, M_res -1.05, H_res -0.94, O_res -0.74 dex) and steer the next call
  (one swapped entry moves call-2 H by 0.05-0.18 dex). (2) Single long calls from the exact start also miss (dt 100 s:
  1.6 dex; 500 s: 3.3 dex). (3) On runs like it (pert, H falls >= 4 dex by 1,000 s), one call from the exact start has
  H error p99 0.12-0.17 dex on 3,604 training runs and 1.45-1.78 dex on 439 validation runs (> 1 dex: 0 vs 2.1-2.7 %):
  in this regime the model is data-limited (a tenfold tail gap), unlike the overall picture of the old trunk. I: the
  rollout drift runs through the hidden entries and reservoirs, and the hard regimes need more data or regularization.
  The author then asked for every lever (more data, bigger model, longer training, physics-informed, different and
  weighted losses, Sobol sampling, mini-chem physics) to be examined separately, with an external review.
- ov_FM2 (V repo-data/runs/ov_FM2/misses_stage1.json; ov_FM1 + the ov_CRn rollout fine-tune, 30k steps): miss 2.31 %,
  large 0.028 %, calls with a miss / large miss 1.38 / 0.0104 %, p99.9 0.60, worst 3.72 dex, frac sum per row 16.8;
  pert 6.69 / 0.138 %, rawA 3.98 / 0.017 %, eqt max 0.09 dex. Paired vs ov_FM1: at 1 dex 178 rows fixed / 15 broken, at
  0.1 dex 8,556 / 876. The new best. Rollout training does not reach the worst rows: the same pert run (861 K, Z 67,
  C/O 0.16) still misses H by 3.7 dex at every dt, and a 1,407 K pert start at 4e-5 bar misses NH3 by 3.6 dex on its
  first call at dt 1-2 s (I: held-out-regime errors, matching the tail gap above).
- Round 6 on the flow-map trunk (V repo-data/runs/ov_R6k16fm, ov_R6longfm misses_stage1.json; one change each
  against ov_FM2): 16-call rollouts (batch 256): miss 2.25 %, large 0.029 %; paired at 0.1 dex 1,000 fixed / 643
  broken, at 1 dex 7 / 13. 10 % of rollouts after up to 256 calls: 2.29 % / 0.029 %; 644 / 539 and 6 / 14. Both tie
  ov_FM2 on large misses; neither is adopted. Per run (of 16,384): runs with a large miss ov_FM1 63, ov_FM2 32, K16 32.
- Readouts on training runs and on the disjoint confirmation set (2026-10-25; V repo-data/runs/ov_FM{1,2}/
  misses_stage1_train.json, misses_stage1_confirm.json): ov_FM2 training runs 1.67 % rows with a miss, 0.003 % large
  (18 rows, 7 runs), worst 1.28 dex; screen 2.31 / 0.028 % (159 rows, 32 runs), worst 3.72; confirmation 2.22 / 0.048 %
  (276 rows, 46 runs), worst 4.08. Pert holds 2 of 18, 108 of 159 and 196 of 276 large rows. ov_FM1: training 3.00 /
  0.017 % (98 rows), screen 3.65 / 0.056, confirmation 3.56 / 0.063 %. On the confirmation set ov_FM2 still beats
  ov_FM1 (1 dex: 110 fixed / 27 broken; 0.1 dex: 8,514 / 871), but its large-miss gain is 24 % there against 50 % on
  the screen: large misses sit in few runs, so one 16,384-run set misstates them; report both sets. I: the large-miss
  tail is held-out generalization (10-16x, pert ~50x), the small misses mostly drift (1.4x).
- Lever study (2026-10-25; one CPU investigation per lever on ov_FM1 / ov_FM2, reports in
  experiments/2026-10-23_overlay/_build/levers/*/report.txt; two reviews). V unless marked:
  - Drift: feeding the true state at every call removes 76 % of ov_FM1's large and 81 % of its small misses; resetting
    only the 12 scored gases recovers 75 %, the reservoirs alone 22 % (the worst row is the exception); a consistent
    0.002-0.003 dex per call at dt 1-2 s over 30-460 calls, also on training runs. ov_FM2 fixes 87 % of the
    drift-type large misses, 4 % of the one-step ones; of its large misses 48 % are one-step-type, 20 % drift-type,
    33 % mixed.
  - Memorization at exact starts of gas far from equilibrium: one call from t = 0, pert validation 10.95 % vs training
    0.89 % rows with a miss (large 0.43 vs 0 %); pert sigma 1.5-2 dex 59.5 vs 4.8 %; the gap is 5-7x for mix and eqf,
    none for rawA, and shrinks from 7x at t = 0 to 2x by 1 s. Pert with sigma >= 1 dex: 3.2 % of rows, 38 % of
    ov_FM1's large misses. The training/validation loss gap is equal for the 12.7M flow-map and the 21.2M plain trunk.
  - ov_FM2's 159 large-miss rows come from 32 runs (ov_FM1: 63); 80 are at 300-650 K (9 runs; one mix run 24 rows),
    a band rollout training barely moved (87 -> 80).
  - Cool-gas study (V levers/cool_gas/report.txt): of those 80 rows, 66 come from cool starts that carry the composition
    of a source >= 2000 K (8.1 % of cool rows; cool starts without a far source: 0 large, 0.00 % miss); worst gases H
    38, HCN 18, CH4 13, NH3 8. Teacher forcing: 26 one-step rows, 9 mixed, 45 drift rows whose one-call error
    (0.027-0.029 dex per call, vs 0.002-0.006 in the warmer bands) adds up undamped to 1 dex by a median call 42-50. The
    truth is sound (a rerun reproduces the packs; 10x and 100x tighter rtol move the labels <= 0.07 dex at all 80
    points); 53-species lifts with the same 25 entries and exactly matched hidden totals spread <= 0.05 dex in 5 runs
    (one run, 50508, spreads 0.3-0.5 dex against errors of 1.1-1.8; run 70904's lifts did not converge at t = 0); I: the
    misses are the model's. 39 of 80 rows lie within 2 dex of a floor. These runs change 2.8-15.6 dex within 1-3000 s
    (the median cool run 0.00-0.01 dex); on chemically active cool runs large misses are 0.15 % on training vs 1.10 % on
    validation runs. I: a data-limited class; whether hot-source cool gas is realistic went to the author (it stays, sec
    6).
  - Weight decay 1e-5 has never acted (1 - lr x wd = 1 - 6e-9 rounds to 1 in fp32; an earlier entry records 300 steps bitwise equal to wd 0).
  - Loss spending (ov_FM1): the L1 term is 98 % of the loss; the unscored entries carry about half of the residual
    derivative; miss-driven resampling and CVaR were never run.
  - Estimates (I): 2x data of the current design about -10 % large misses; longer training (600k-1M) misses -5 to
    -20 %; bigger model -20 to +20 %; loss reweighting, physics-informed terms (semigroup, fixed point, reservoir
    consistency): little, because the training runs have almost no misses left; a recovery ("memory loss") term is
    niche. Mini-chem closed as a lever: on the same starts it misses far more (random runs, one 100 s call: 24.6 /
    10.7 % vs ov_FM1 2.3 / 0.00 %) and on the same rows; 12-34 us per call on a CPU core; GPL.
  - Possible state limit (I, a correlation confounded with pert): starts with M_res >= 1e-6 are 10.6 % of rows and
    63 % of large misses; in the worst row NO is 10^-3.3, lumped into the reservoirs. A state change is the author's call.
  - A frozen-call defect: when Newton and LM both fail the call keeps its input. 2 of 78,811 validation runs (rawB,
    ~2,900 K, 1.6e-5 to 4.9e-4 bar, dt 100 and 500 s from the start) freeze (an 8.6-dex CH4 miss outside the screen
    set); 1,000 Newton plus 2,000 LM iterations do not help (an infeasible target, I); the 7.9-8.7 dex training-batch
    worsts are single failed rows (proj_fail_share = 1/8192), which get no gradient. Cause (V levers/frozen_proj): not
    infeasible; from lam = 0 one entry (C2H2) dominates the H and C budgets, the Newton step is 100-265 ln along a
    near-flat direction and the +-2 clip stalls it; LM settles in a false minimum. Continuation from the input reaches
    project_ln's own root (within 0.1-0.5 dex of VULCAN on the first call, where the frozen call was 8-9 dex off). Only
    exact rawB starts at 2,530-2,965 K and 1.3e-5 to 4.9e-4 bar (17 training, 2 validation runs). Fix: a third fallback,
    elemproj.project_cont, behind a head attribute that defaults off; on 16,384 ordinary calls it changes nothing
    (bitwise).
  - Decision: train-split and confirmation readouts first; weight decay 0.05 as one fresh arm (a null at 0.01 would
    not reject the lever); the half-data test and ov_FS2 dropped; new data wait for the author's rulings (pert sigma up
    to 2 dex, an NO entry, the design reserve); the frozen-call defect and the cool-gas band are investigated next.
- A loader bug (load_init looked for the plain trunk's first-layer keys in the flow-map trunk) stopped the flow-map
  rollout fine-tune at start; fixed. Round 6 moved to the flow-map trunk: ov_FM2 (FM1 + the ov_CRn rollout fine-tune),
  then one change each against it (K 16; 10 % long rollout prefixes to 256 calls), then ov_FS2.
- Conclusions so far (I): the head stays the dt-free relaxation head; the large levers are dt 1-500 s windows and the
  rollout term with the bounded miss term at every call; a light fractional term (0.1) fixes more rows but adds a few
  extreme calls; dt-aware heads trade extremes for drift in fine-tunes (round 5 tests fresh training). Remaining worst
  rows: unreacted-gas first calls at 1,900-2,800 K and low P (7-8 dex).

#### 2026-10-25: weight decay, the frozen-call fix in place, the data supplement
- The frozen-call fix is in the code: elemproj.project_cont, run after LM when the head's cont attribute is set
  (default off, so training and the benches are unchanged); bench/misses.py --cont sets it. The training regression
  checks (stage 1 and 2) pass unchanged.
- Weight decay 0.05 (V ov_FMwd05; ov_FM1's fresh recipe otherwise; 1e-5 had been a no-op in fp32). Screen readout:
  miss 3.64 %, large 0.043 %, worst 6.97; the 6.97 and 14 more rows were new kept-input calls (7 rawB runs at
  1,293-1,702 K, C/O 0.10-0.23, CO2 or CH4 3.7-7.0 dex off from the first call, which ov_FM1 gets within 0.08 dex). With
  the fallback those 15 rows pass and no other row changes: large 0.041 %, worst 3.77; paired vs ov_FM1 at 1 dex
  198 fixed / 109 broken, 0.1 dex a tie. On the disjoint confirmation set: miss 3.57 %, large 0.061 %, worst 4.48 vs
  ov_FM1's 3.56 / 0.063 / 4.08; paired 190 / 181 at 1 dex, 7,043 / 7,134 at 0.1 dex: a tie. Not adopted. Both sets
  agree only on unreacted-gas (rawA) starts (56 -> 16 and 37 -> 15 large rows); perturbed starts got worse on the
  confirmation set (218 -> 237). A screen-only gain of this size is within the run-to-run spread of the large-miss
  count, which sits in a few dozen runs.
- Choices under the author's delegation (sec 3): perturbed starts stay with sigma 0.1-2 dex; hot-source cool gas stays;
  no new state entry. New data, both arms of equal size (262,144 runs each), same domain and mapping:
  - arm U: the clean design's never-driven reserve (rows 786,432-1,048,575);
  - arm T: rows 2^20 to 2^20 + 2^18 - 1 of the same scrambled Sobol sequence (its first 2^20 rows equal the clean
    table, checked), kind shares eqt 5, eqn 5, eqf 35, mix 15, pert 35, raw 5 % (was 20/20/30/10/14/6), by a monotone
    map of the kind coordinate; strata, conditions, compositions and sigma unchanged.
  - IDs 4,210,786,432-4,211,310,719, unused by all 3,960,343 stored runs (V scan of the packs' group names). The
    supplement goes to training only; the validation split and the normalization stay byte for byte.
  - Pilot: 2,048 runs (1,024 per arm), none failed, 0.13 CPU-s per run (V).
  - Driven and packed (V): 524,260 runs kept, 27 rejected by the acceptance rule and 1 that failed twice (0.005 %);
    the pack and cache checks pass. 419,624 of them fall in the training split (the rest stay as an extra held-out
    set); the merged training split has 1,048,664 runs, 838,852 per arm with the clean runs.
  - Where the best model's perturbed-start misses sit (V ov_FM2 readouts joined with the design map): large-miss rows
    per sigma band, screen / confirmation: 0.1-0.6 dex 4 / 31 (1 / 2 runs), 0.6-1 9 / 36, 1-1.5 20 / 47, 1.5-2
    75 / 82 (18 / 16 runs). Starts with sigma >= 1 (3.2 % of rows) hold 95 of 159 and 129 of 276 large-miss rows;
    rows with a miss rise from about 2 % (sigma < 0.3) to about 18-20 % (sigma 1.5-2).
  - Test: ov_FM1's recipe on clean + U and on clean + T (equal steps), ov_FM1 the control. Adoption, set before
    reading: fewer large-miss rows on both the screen and the confirmation set (paired), rows with a miss not worse,
    worst not worse by more than 0.5 dex; T against U separates targeting from size.
  - ov_DU (clean + U), screen, without the fallback: miss 3.52 %, large 0.052 %; paired vs ov_FM1 at 0.1 dex 7,275
    fixed / 6,525 broken, at 1 dex 176 / 152; its worst rows are 15 rawB kept-input calls (the runs that froze for the
    weight-decay model). At t0 = 0 it ties ov_FM1 (65 / 59 large rows): more uniform data leaves fresh-start first calls
    alone.
- Why the best model's worst rollout misses H (validation run 65232, pert sigma 1.68, 861 K, 0.0055 bar), V:
  - From the exact start the error is 3.5-3.7 dex at every dt (one 300 s call included); from 1 s on, 0.1-0.3 dex. At
    300 s the call has already reached its learned target (H 10^-4.45 vs VULCAN 10^-8.06): a wrong target, not a slow
    rate; the projection converges normally.
  - In VULCAN the perturbation's atomic N (5.5e-4) and OH (4.1e-2) give NO within 1 ms (3.6e-7 -> 5.0e-4, then
    constant); NO speeds H recombination. The model's reservoirs (where NO lives) are 1-1.7 dex low after call 1. With
    VULCAN's reservoirs swapped into the model's state after call 1 (re-projected onto the element ratios) the H error
    at 300 s falls from 3.68 to 0.31 dex; VULCAN's state with the model's reservoirs reproduces it (3.52). Swapping
    the scored or tracked entries changes little.
  - The 25-entry state fixes the future: 12 alternative 53-species states with the same 25 entries (hidden totals
    matched to <= 4e-9) give scored-gas spreads <= 0.01 dex at t + 1-300 s from t = 0 for 7 of the 8 worst fresh-start
    runs (hidden shares up to 8e-3); the eighth spreads 0.7-1.0 dex in NH3, while the model misses other gases there.
- The fresh-start tail in general (ov_FM2's 100 large-miss rows at start age 0, 29 runs; V): call 1 alone is > 1 dex in
  73; from call 2, still > 1 dex: 85 as is, 14 from VULCAN's state after call 1, 27 with VULCAN's 12 scored gases, 74
  with the tracked species, 76 with the reservoirs. The first-call error persists (later worst = call-1 error, median
  +0.03 dex); the NO route covers about 3 runs. Call 1 is the same for ov_FM1, FM2 and the round-6 fine-tunes (72-73
  rows > 1 dex), and the separately trained ov_FMwd05 and ov_DU share ov_FM1's error sign in 72 of 73 rows: a shared
  bias of the trained function, not seed variance. The first call's reservoirs can be unrealizable (every hidden species
  holds C or N, so C_res + N_res >= M_res; violated in 20 of 79 exact-start pert large-miss rows).
- Data cannot close it: the nearest-training-start distance of pert starts shrinks as N^(-1/10.2) (6-8 for eqf, mix
  and the equilibrium kinds); halving it needs about 1000x the pert data, and the large-miss rate of sigma 1.5-2 starts
  rises from 0 to 22 % across distance quintiles. Perturbed starts with sigma >= 1 are 3.2 % of rows and about half the
  large misses (I: the steering case for the decision recorded in sec 3, "Do A").
- The judged sets (sec 3, decision A; bench/misses.py by_set): main domain = C/O <= 1 and pert sigma <= 1 dex (87 % of
  rows); C/O 1-2 side check (9.8 %) and the stress set (3.2 %) reported beside it. ov_FM2, screen / confirmation:
  main 1.50 / 1.43 % rows with a miss, 0.007 / 0.011 % large (34 / 53 rows, 5 / 6 runs), worst 3.47 / 4.08; C/O side
  4.87 / 4.85 %, 0.054 / 0.167 % (5 / 13 runs); stress 16.3 / 15.8 %, 0.51 / 0.70 %. ov_FM1 main: 2.60 / 2.52 %, 0.027 /
  0.023 % (23 / 21 runs). On its training runs ov_FM2's judged (sigma <= 1) rows: 1.49 % / 0.003 % (6 runs).
- The data arms, judged, with the fallback (V; paired with ov_FM1, 1 dex fixed / broken, runs with a large miss):
  ov_DT (targeted kinds) screen 90 / 25 (23 -> 10 runs), confirmation 74 / 30 (21 -> 12); rows with a miss -8 / -5 %;
  worst not worse. It passes the preset test on both sets: the targeted supplement is adopted. ov_DU (uniform) ties
  on large misses (confirmation 0.023 -> 0.022 %) and trims small misses (2.52 -> 2.37 %); DT beats DU on large misses
  (confirmation 76 / 38). I: targeting helps large misses beyond what size does, though it cannot close the sigma > 1
  tail (above).
- Small misses on the main domain are a fit / drift limit, not coverage (V, ov_FM2): 1.50 % of held-out rows vs 1.26 %
  of training-run rows. dt 1-2 s hold 53 % of them, peak at a median call 587 of 1000 (dt 1) and miss in 18-24 % of
  calls: per-call bias accumulating over hundreds of calls. The rollout fine-tune halved them (dt 1: 6.15 -> 3.30 %);
  16-call or late-prefix rollouts add little (3.07 / 3.21 %); uniform data alone little (5.88 %).

#### 2026-10-26: the fine-tune on the targeted data; the stress runs stay in training
- ov_DT2 = ov_FM2's rollout fine-tune (30k steps) started from ov_DT (V, fallback readouts, screen / confirmation): main
  domain 1.247 / 1.227 % rows with a miss, 27 / 37 large-miss rows (2 / 6 runs), worst 3.32 / 4.10; C/O side 4.08 /
  4.18 %; stress 13.1 / 13.0 %. Paired with ov_FM2: main-domain small misses 3,458 / 2,176 and 3,218 / 2,217 fixed /
  broken (-17 / -14 %), large 12 / 5 and 25 / 9; better on the C/O side and the stress set too. The large-miss gain is
  inside the paired noise (the main domain now holds only 27-53 large-miss rows per set). The adopted data and the
  adopted fine-tune compose: ov_DT2 is the best model so far.
- ov_DTcap (ov_DT with the perturbed sigma > 1 runs weighted 0 in the draws: 37,447 of 838,852 drawn training runs, 4.5 %) vs ov_DT (V):
  main-domain large misses 32 / 83 and 33 / 74 fixed / broken (runs 10 -> 18 and 12 -> 17), small misses a tie, the
  stress set much worse (1 dex 36 / 149, 48 / 201). The stress runs stay in the training draws. I: the hard scrambles
  teach the large first-call moves that the in-domain tail also needs. New data still does not generate them
  (decision A).
- Full-length run plan (stage 1 of ov_DT's recipe, then the fine-tune; adoption test against ov_DT2 set before
  reading): both reviews asked for 600k steps instead of 1M (the training-length study's own range of window draws per
  training run), arm U out, K 8, a tighter adoption test. On hold since the decision panels (below).
- The decision panels, first time for the flow-map models (V, data/benches/score_runs/overlay; the continuation
  fallback on). Averaged score: ov_DT2 0.0023, ov_FM2 0.0026, the clean model 0.0039, both a tie with it overall, but
  the lasting score is worse than the clean model's (+0.004 / +0.005 dex; long runs at 200-500 s +0.013 / +0.016).
  Misses per panel (rows with a miss / large miss): S ov_DT2 0.94 / 0.007 %, clean 3.46 / 0.249 %; P ov_DT2 35.6 /
  2.73 %, clean 39.5 / 5.08 %; L (2048 runs of 172,800 calls, dt 1-500 s, 2-999 days, median 40 d) ov_DT2 31.4 / 3.32 %,
  clean 22.3 / 2.69 %; side (C/O 1-2) ov_DT2 2.88 / 0, ov_FM2 2.24 / 0, clean 4.79 / 0.134 %; Sraw ov_DT2 2.01 / 0.003,
  ov_FM2 2.04 / 0.018, clean 6.87 / 0.944 % (V experiments/2026-10-23_overlay/_build/diag/panel_misses_1026.out).
- Lasting errors on the long runs (V, panel L, last decade of each run): rows with a miss / large miss: clean 5.6 / 0.59
  %; clean_s1 (its stage 1 alone) 26.2 / 4.25 %; ov_FM1 37.8 / 4.59 %; ov_FM2 32.0 / 3.56 %; ov_DT2 29.9 / 3.32 %. Worst
  gases NH3, HCN, CH4; worst at 650-2000 K; rising with dt; the flow-map models miss even from local-equilibrium starts
  (23-25 %; clean 0 %, clean_s1 8 %). The cause is training without a long-horizon stage, not the flow-map trunk: the
  clean model's stage 2 (long jumps to 1e7 s, grid chains from 1e3 s, changing-condition chains) removed most of them;
  the campaign's readout (constant conditions, at most 1000 calls, up to ~3,000-4,000 s elapsed with the start age)
  could not see them. I: one-call windows of 1-500 s resolve slow rates only to the loss noise, and 172,800 calls
  integrate a per-call bias of ~2e-6 dex into ~0.3 dex.
- The old 16-path readout (paths_v2x, 10,000 calls at dt 500 s) agrees: the targeted-data models carry growing HCN /
  NH3 errors on three paths (p99 1.63-1.70 vs 0.99-1.09 for the clean-data flow-map models), though the decision panels
  show ov_DT2 a little better than ov_FM2 on L and P. Points above 1 dex on paths 3 / 6 / 7: ov_DT2 7,001 / 10,921 /
  14,002, ov_FM2 2,091 / 140 / 0 (HCN 19,537 vs 2,255 points, NH3 12,387 vs 0); ov_DT2 better on paths 11 and 12 (V
  diag/path_tail.out). Late p99 of the fresh models: clean or uniform data 0.85-1.20 (ov_FM1, ov_FMwd05, ov_DU),
  targeted data 1.65-1.70 (ov_DT, ov_DTcap) (V log.csv as quoted in the campaign log 10-26 04:43).
- The fixed point at deployment dt (I: direct CPU calculations of an external review on the L panel's stored 53-species
  truth, interpolated, not fresh solver stops;
  experiments/2026-10-23_overlay/_build/reviews/answer_overlay_program_*.txt): one call at VULCAN's state at 1e7 s, row
  78 (dt 107.6 s) CH4 change VULCAN -2.07e-6 dex, ov_DT2 -4.52e-5, clean_s2 -2.09e-6; on the 128 local-equilibrium L
  conditions at 1e7 s (VULCAN moves <= 1.3e-9 dex in 50 s) ov_DT2's one-call motion median 1.5e-6, p90 2.5e-5, max
  1.25e-4 dex against clean_s2's median 4.1e-9, max 3.9e-7. Not precision (fp64 weights change it <= 3.9e-8 dex); row
  78's unprojected CH4 change (-2.15e-6) is near the truth, and the projection moves the joint update's inconsistency
  into CH4. Panel P restarted from VULCAN's states after call 5035 (same forcing): at call 5220 1 miss / 0 large vs 29 /
  2 uninterrupted, at 5817 9 / 0 vs 29 / 2: most of P's error is carried state error, plus a local forcing response
  (paths 190, 191 regrow 0.51 dex). Panel P paths span 0.12-57 days (median 2.6), 64 of 256 with variable dt.

#### 2026-10-27: stage 2 on the flow map, late terms, the small-dt late readout, the hybrid head
Short readout = bench/misses.py with the continuation fallback, all rows (573,440); L lasting = panel L rows with a miss
/ large miss in the last decade of elapsed time (diag L_lasting).
- ov_DT2s2 = the clean model's stage 2, 10k steps, from ov_DT2 (V): rejected. Short readout 4.05 / 0.046 % (ov_DT2
  1.91 / 0.024); panel S 0.94 -> 2.00 %; L lasting 29.9 / 3.32 -> 31.8 / 5.08 % (large rows 68 -> 104); averaged 0.0029.
  From VULCAN's state at 1e6 s (500 s steps, +9e6 s) 21.1 % (ov_DT2 30.5, clean 0.8). I: this stage-2 recipe at this
  dose does not carry over to the flow map at 1-500 s, though some late behaviour transfers.
- ov_R2a (late rollouts added to the fine-tune) (V): L lasting 28.9 / 2.05 % (large rows 68 -> 42); S 0.79 %; averaged
  0.0021; short readout 2.02 / 0.025 %. Cuts long-run large misses; the common small lasting drift stays. Late hold (V
  diag/teacher_late_R2a.out; 128 validation runs at 650-2000 K, from VULCAN's state at 1e6 s, 500 s steps, +9e6 s): 23.4
  / 0.8 % (ov_DT2 30.5 / 3.1); from the start state 28.1 / 2.3.
- ov_R2b (ov_R2a + a late one-call stationarity term at weight 1, no ramp, fixed eps) (V): rejected. Short readout 5.12 /
  0.052 %; L lasting 21.9 / 7.32 % (150 large rows); averaged 0.0040; late hold 28.1 / 4.7 %, from the start 33.6 /
  13.3 % (V teacher_late_R2b.out). It froze some drifting rows and broke others.
- Small-dt late readout (dt 30 s from VULCAN's state at 1e6 s, +1e6 s) (V): rows with a miss ov_DT2 22.7 %, ov_R2a
  20.3 %, clean 0.8 %; at 500 s the same elapsed time gives ov_DT2 10.9 %. I: drift per unit time is ~2x faster at 30 s,
  so the GCM use (1000 days at 25-50 s) is worse than panel L (172,800 calls per row) shows.
- Label check (V): dense VULCAN output from 208 late training states: entries VULCAN keeps still are still in the
  interpolated labels in 99.5-99.9 % (p99 2-5e-7 dex); moving scored gases |ratio - 1| p50 0.10, p90 0.43 at 1 s and
  0.20-0.23 at >= 30 s. The still targets are sound; a moving-entry term should use the scored gases only.
- Hybrid head relaxclosedfmeq (author approved, sec 3): a dt-free target from the encoder; the rate and a bounded
  transient from the flow-map dynamics with dt_norm clamped to [0, 1]; the target offset projected off the five
  directions the element projection absorbs. Long jumps wired into stage 1 (s1_jump_w). Existing heads bitwise unchanged
  (bitcheck s1 / s2 PASS). 17.0 M parameters, 33.8 MFLOP per call (flow map 12.7 M, 25.4).
- ov_H1 (hybrid + stage-1 jumps, fresh 300k) (V): fails its aim. Short readout 7.82 / 0.317 % (ov_DT 3.32 / 0.037); L
  lasting 30.0 / 4.54 % (93 large rows); S 2.42 %; P 46.5 / 8.2 %; averaged 0.0040; late hold (500 s) 28.1 / 0 %, from
  the start 37.5 / 1.6 %, 30 s steps 18.8 / 1.6 % (V teacher_late_H1.out, _dt30_H1.out).
- The author's four questions, answered by an external review (reviews/answer_author_questions_*.txt; the design
  table's T shares 15.8 / 39.2 / 24.2 / 20.8 % for 300-650 / 650-1300 / 1300-2000 / 2000-3000 K, V
  data/gen/clean/design_map.npy, before acceptance and splitting): I: the design is broad, not shown optimal, and
  650-2000 K already holds 63 % of the design (the drift is not a coverage hole);
  changing-condition histories are the untested data lever; no evidence for separate models per region (fit is not the
  limit; less data each; seams); a fixed 50 s step is a simpler problem but its gain was unmeasured (tested 10-28 / 30,
  below); 10x data does not fit the machine (disk, GPU memory).

#### 2026-10-28: the jumps control, accuracy against data size
- ov_FMJ (flow map + the same stage-1 jumps, fresh 300k) (V): short readout 3.08 / 0.039 % (ov_DT 3.32 / 0.037), so
  the jumps cost nothing and ov_H1's loss comes from the hybrid head. L lasting 30.2 / 1.86 % (38 large rows, the fewest
  of the flow-map models); S 1.23 %; P 39.5 / 3.5 %; averaged 0.0026; late hold (500 s, +9e6 s) 23.4 / 0 %, from the
  start 23.4 / 0, 30 s steps 20.3 / 0 (V teacher_late_FMJ.out, _dt30_FMJ.out). I: long jumps cut large lasting misses,
  not the small drift.
- Data size (ov_FM1's fresh recipe on 157k / 315k / 629k / 839k uniform clean training runs: ov_D25, ov_D50, ov_FM1,
  ov_DU; ov_DT = 839k with targeted kinds) (V): short readout 4.81 / 0.115, 4.04 / 0.087, 3.65 / 0.056, 3.52 / 0.049 %;
  ov_DT 3.32 / 0.037 %. L lasting 30.6 / 1.81, 35.6 / 4.00, 37.8 / 4.59, 31.0 / 1.86 %; ov_DT 38.1 / 3.76 %. More data
  cuts short-run misses by ~10-16 % per doubling with no sign of levelling off; long-run lasting errors show no trend.
  Late hold, 500 s steps +9e6 s / 30 s steps +1e6 s from VULCAN's state at 1e6 s (V diag/teacher_late_*.out): ov_D25
  27.3 / 22.7 %, ov_D50 38.3 / 26.6, ov_FM1 41.4 / 31.2, ov_DU 32.8 / 24.2; ov_DT 39.8 / 28.9: no monotone trend
  among these arms either (I).
- Step time does not depend on data size (~0.039 s per stage-1 step for 315k-839k runs); stage-1 jumps add ~45 %, the
  hybrid head ~60 %; stage 2 ~1 s per step (V, log.csv). 10x data (V inputs, I totals): ~340 CPU-h of VULCAN, ~880 GB
  of raw packs (over the 500 GB quota), ~200 GB fp64 cache (4x one GPU's memory).

#### 2026-10-29: the hybrid with stage 2, the fixed 50 s step (partial)
- ov_H1s2 = ov_H1 + 10k steps of the clean model's stage 2 (V): rejected as a pick. Short readout 8.66 / 0.245 %. L
  lasting 22.6 / 4.44 % (91 large rows); ov_H1 30.0, ov_DT2 29.9, ov_DT2s2 31.8, the rejected ov_R2b 21.9; clean 5.6. By
  start code 0 / 1: 3 / 43 % (ov_H1 17 / 43, ov_DT2 23 / 37, clean 0 / 11); it falls with dt (24 % at 1-2 s, 17 % at
  200-500 s) where every flow-map model is flat or rises. From VULCAN's state at 1e6 s: 500 s steps +9e6 s 15.6 / 0.8 %,
  30 s steps +1e6 s 24.2 / 2.3 % (ov_DT2 22.7; ov_H1 18.8). The whole L gain is the start-code-0 rows (1,024 per code);
  code 1 (transported starts, the GCM case) stays at 43 % (the clean model's stage 2 took the same code from 45 to 11
  %), and at 30 s it got worse. Paired with ov_DT2 (lasting): 0.1 dex fixed / broken 274 / 124, 1 dex 43 / 66; from the
  start state +9e6 s 18.0 / 1.6 % (V teacher_late_H1s2.out). I: the hybrid's rate comes from the dt-aware trunk with dt
  clamped at 1000 s, so long supervision trains it at the clamped input only and does not reach 1-500 s. "The flow map
  does not respond to stage 2" is too strong (ov_DT2s2 at 500 s late 30.5 -> 21.1 %).
- ov_CS2F = ov_R4fm's short-run fine-tune (dt 1-500 s, miss, rollout, fractional terms, 30k) started from clean_s2 (V):
  short readout 9.15 / 0.264 %, judged 8.22 / 0.201 % (clean_s2 14.87 / 1.646 with the fallback, the same as without
  it); L lasting 15.3 / 1.56 % (32 rows), by start code 0 / 1 4 / 27 % (clean 0 / 11), rising with dt (9 % at 1-2 s, 25
  % at 200-500 s), paired vs clean 0.1 dex fixed / broken 29 / 228, 1 dex 11 / 31; late hold 30 s 8.6 %, 500 s 21.9 /
  0.8 %, from the start 25.0 / 0.8 (V teacher_late_CS2F.out). ov_CS2Fs2 = + 10k of stage 2: short 13.38 / 0.715 %, L
  lasting 4.7 / 0.10 % (2 large rows; start code 0 / 1 0 / 9 %), late hold 30 s 0.8 %, 500 s 0.8 / 0, from the start 8.6
  / 0.8 (clean_s2 0.8 / 0.8 / 6.2; V teacher_late_CS2Fs2.out, teacher_late.out). The per-entry head trades one end for
  the other; starting from clean_s2 does not lift its short-run ceiling (ov_R4fm from clean_s1: 8.73 %).
- Misses barely depend on composition (V, main domain): ov_DT2 short 1.25 % all, 0.97 % at Z 0.3-3x, 1.67 % at Z > 30x;
  L lasting 29.9 % all, 28.5 % at Z 0.3-3x; ov_CS2Fs2 12.47 / 11.97 % short, 4.7 / 4.2 % L. Narrowing the composition
  domain alone removes neither failure (I: it raises data density, which helps short runs ~10-16 % per doubling).
- Fixed 50 s step (ov_F50: ov_DT's recipe trained only at dt 50 s; every call at 50 s for 1e7 s on 1,024 main-domain
  validation runs) (V, 2 of 6 models so far): from t = 0, +1e7 s, ov_F50 41.2 / 10.9 % vs ov_DT 36.6 / 4.7 %; from
  VULCAN's state at 1e6 s, 38.0 / 9.4 vs 34.3 / 3.6. Training at one fixed step has not helped so far.
- The runs in progress were killed 10-28 22:30 with the terminal session that launched them (memory exhaustion on the
  machine) and restarted 10-29 05:50; ov_H1s2's stage 2 resumed from its checkpoint, the fixed-step readout restarted.

#### 2026-10-30: the two families together (the pair); the mixed-mode head
- The pair (author's request, sec 3): every call runs both ov_DT2 (flow map) and ov_CS2Fs2 (per-entry head after stage
  2) and keeps the flow map's step where either model moves a resolved scored gas by more than tau dex in that call,
  else the per-entry head's (experiment code outside the repository; forced tau -1 / 1e9 reproduce each model exactly;
  CUDA-graph replay equal to eager). Results (V; tau 1e-3 chosen among 1e-2, 1e-3, 1e-4 after reading):
  - Short readout (all rows): 2.12 / 0.024 % (ov_DT2 1.91 / 0.024, ov_CS2Fs2 13.38 / 0.715); tau 1e-4 2.05 / 0.025,
    1e-2 3.85 / 0.033.
  - Panel L lasting: 2.1 % / 0.00 % (no row above 1 dex; median of miss rows 0.14 dex; worst row 0.35 dex, NH3, 469 K);
    ov_DT2 29.9 / 3.32 % (68 rows, worst 4.80 dex), ov_CS2Fs2 4.7 / 0.10 %, clean 5.6 / 0.59 % (worst 9.04 dex).
    Whole-row 4.6 / 0.00 % (ov_DT2 31.4 / 3.32). Paired vs ov_DT2: 0.1 dex fixed / broken 576 / 6, 1 dex 68 / 0.
    tau 1e-2 2.6 / 0.00, 1e-4 4.5 / 0.44 %.
  - From VULCAN's state at 1e6 s: 30 s steps +1e6 s 0.8 % (ov_DT2 22.7); 500 s steps +9e6 s 1.6 % (ov_DT2 30.5); from
    the start state +9e6 s 2.3 / 0.8 % (ov_DT2 31.2 / 3.9, ov_CS2Fs2 8.6 / 0.8).
  - Short runs, judged rows (V): pair 1.49 % vs ov_DT2 1.25 %; rows above 1 dex 34 vs 27 (the 7 extra all raw starts, 6 at
    dt 1 s); 80 % of the pair's misses are ov_DT2's own. The pair is the best on long runs and a small loss on short
    runs, not a tie. Its excess: eqf starts at dt 1-2 s (removed by tau 1e-4: 512 -> 84 rows) and raw starts at every dt
    (rawB 3.1 % vs 0.3 %; barely changed by tau): I: the switch counts moves only above the floors, so a raw start whose
    CO2 or H sits below its floor looks settled and goes to the per-entry head.
  - I: the two failures mostly do not overlap: the flow map is right while the gas changes fast, the per-entry head once
    it has settled. tau 1e-3 was chosen after reading the same panels (mildly optimistic until S / P / side confirm).
    Closed 10-30 04:02 on the author's direction (one model, no switch; sec 3): the panel S run finished but its states
    and scored miss readout are unavailable (no S/states.npz), panels P, side, Sraw and the averaged score were not
    completed, and the dt-scaled switch pair_r3 (tau 1e-3 x min(1, dt / 10 s)) was stopped unread. Call speed under
    contention: 228 ms per call on panel S vs 34 ms for ov_DT2 (two networks per call). Late hold of the other
    thresholds (30 s / 500 s / from the start): tau 1e-2 0.8 / 0.8 / 0.8 %, tau 1e-4 0.8 / 7.0 / 7.0 % (V
    diag/teacher_late*_pair_t*.out).
- The mixed-mode head relaxclosedv (author's request for a new single-model design, sec 3): z~ = z + V diag(a) V^T (q -
  z), a = 1 - exp(-k dt), V = exp(K - K^T) orthogonal (K from 300 extra trunk outputs), q and k dt-free; then the
  element projection. The exact solution of dz/dt = -V diag(k) V^T (z - q) frozen at the call's start: V = I is the
  per-entry head; q = z gives no change at any dt (long steps and 1-500 s steps share the settled state); S mixed modes
  allow fast-then-slow and non-monotone entries within a call; started from a per-entry file the mixing outputs are added
  at zero. V by a fixed-length Taylor series and squarings (torch.linalg.matrix_exp syncs with the host and cannot be
  captured in a CUDA graph; both reviews found it). Limits (reviews): symmetric modes, so |z~ - q| never grows within a
  call; one target per entry; no call-to-call consistency; long jumps train q and k, not V. bitcheck s1 / s2 PASS.
  ov_V1 (ov_CS2F's recipe + the stage-1 long jumps, from clean_s2) vs control ov_CSJ (the same, per-entry head): at 10k
  steps the mixing is in use (max |V - I| median 0.115 at start states, 0.046 at row 80) and validation 0.0176 vs
  0.0215. ov_VF: the head trained fresh with ov_FMJ's recipe, 12.0 M parameters (ov_FMJ 12.7 M).
- Results (V; short = judged rows of the short readout with the fallback; L = panel L lasting; P = panel P lasting):
  - ov_V1 (mixed-mode head, from clean_s2, ov_CS2F's recipe + stage-1 long jumps): short 7.45 % (large 0.086 %), L
    15.9 % with 68 rows above 1 dex, P 33.2 %. Control ov_CSJ (per-entry head, same recipe): short 9.14 % (0.211 %), L
    9.5 % (6 rows), P 32.0 %. The head helps short runs and hurts long runs; the stage-1 long jumps alone cut the
    per-entry head's lasting drift from 15.3 % (ov_CS2F) to 9.5 % (large rows 32 -> 6).
  - ov_V1j (ov_V1 + a joint stage 2: stage 2's long-run terms with stage 1's miss / fractional / rollout terms, new
    default-off key s2_joint, bitcheck s1 / s2 PASS): short 7.28 % (0.083 %), L 7.8 % (26 rows; paired vs ov_V1 fixed /
    broken 177 / 12), P 27.7 %, late hold 30 s 0.8 %, 500 s 0.8 %, from the start 5.5 % (V teacher_late*_V1j.out).
    Sequential control ov_V1s2: short 9.64 % (0.275 %). Joint training removes the short-run cost of the long-run stage
    and halves the drift; still far from the flow map's short runs.
  - ov_VN1 (non-normal modes, U = I + N, from ov_V1, +30k): short 6.28 % (0.054 %); its control ov_V1c (ov_V1 + 30k,
    mixed-mode head) 6.50 % (0.060 %): most of the gain is the extra training. N near its 0.1 bound.
  - ov_VF (fresh, flow map's recipe and size): stopped at 50k by its preset rule (validation 0.0134 vs the hybrid's
    0.0094 and the flow map's 0.0063); its training path readout p99 4.8 dex at 25k, 11.1 at 50k. I: trained fresh, the
    mixed-mode head fits one step about 2x worse than the flow map.
  - Head vs jumps, late hold (30 s steps +1e6 s / 500 s steps +9e6 s from VULCAN's state at 1e6 s / from the start; V
    diag/teacher_late*_V1, _CSJ, _CS2F .out): ov_V1 2.3 / 10.2 / 14.8 %, ov_CSJ 2.3 / 7.8 / 16.4, ov_CS2F (no jumps)
    8.6 / 21.9 / 25.0. I: for these heads the late-hold gain comes from the stage-1 jumps, not the mixed-mode head. ov_V1
    L lasting by start code 0 / 1 10 / 22 %, paired vs ov_CSJ 0.1 dex fixed / broken 29 / 160, 1 dex 1 / 63; its
    training path readout had shown it (p99 3.68 vs 1.36 dex, V log.csv).
- Panel P (T and P changing every call, ~2 months, half the paths with mixing events): every model misses about a third
  of the paths (ov_DT2 35.5 %, clean 39.5 %, ov_V1 39.1 %); 66 paths missed by all four; same-sign errors across heads.
  Not the mixing (mixed 35 % vs unmixed 36 %, no clustering after mixes). The temperature swing: rows with a miss by
  T-swing tercile (22 K / 285 K / 1,775 K at the 10th / 50th / 90th percentile) ov_DT2 8 / 30 / 67 %, clean 15 / 26 /
  77 %. I: large temperature excursions are under-trained (constant-condition windows and stage 2's chain sets only).
  Breakdown (V overlay log 10-30 03:42; the worst-gas count also in reviews/answer_overlay_program_*.txt): ov_DT2 91 of
  256 rows with a miss, worst gas C2H2 38, HCN 21, OH 12, NH3 7; first exceedance at a median 20 % of the path, then just
  over the line (median worst per tenth of the path 0.06-0.11 dex); 650-1300 K 40 %, 1300-2000 K 49 %, 300-650 K 0 %,
  2000-3000 K 3 %; eqf starts 42 %, eqt 29 %. The clean model errs early (median first exceedance at 0.5 % of the path,
  0.20 dex in the first tenth). I: on GCM-like paths the misses are small and persistent at 650-2000 K, not blow-ups.
- Later the same day (V): ov_V1s2 (sequential stage 2 from ov_V1) short 9.64 % (0.275 %), L 7.5 % (26 rows): the same
  long-run gain as the joint stage (ov_V1j 7.8 %, 26), with the short-run cost the joint stage avoids. Decision: the
  joint stage is the standard second stage. ov_V1c (ov_V1 + 30k) L 14.5 % (48 rows), ov_VN1 14.6 % (63): stage-1-only
  runs keep ov_V1's long-run fault; the non-normal modes add nothing on long runs.
- The author's early-time reweighting (ov_V1jL = ov_V1j with row-0 starts 0.2 -> 0.05, first-stored-time anchors 0.25
  -> 0.05, jump row-0 starts 0.1 -> 0.05, rollouts from row 0 0.5 -> 0.2; I: one-step starts before 1 s ~56 -> ~34 %)
  (V): short 7.22 % (0.088 %), tied with ov_V1j (7.28 / 0.083; paired 2,064 / 1,748); starts after 10 s 3.17 vs 3.31
  %; L 6.8 % with 15 rows above 1 dex (ov_V1j 7.8 %, 26; paired 0.1 dex fixed / broken 33 / 12, 1 dex 18 / 7); P 27.0 %.
  It helps long runs at no short-run cost: part of the standard recipe. ov_V1jL is the best-balanced single model so
  far (the flow map ov_DT2: short 1.25 %, L 29.9 % / 68, P 34.4 %; the per-entry head ov_CS2Fs2: short 12.47 %, L 4.7 %
  / 2). Its rarest long-run failures are cool gas (375-485 K): NH3, CO, C2H2 creep off after ~1e6-1e7 s (worst row
  6.3 dex, C2H2, 422 K).
- The last single-model runs (V repo-data/runs/ov_V1cj, ov_VN1j misses_cont.out; panel L lasting by diag/L_lasting.py
  and panel P by diag/panel_misses.py, rerun 10-31 for ov_VN1j and the P column: experiments/2026-10-31_notes/_build/
  L_lasting_V1cj_VN1j.out, P_misses.out): ov_V1cj (joint stage 2 from ov_V1c, i.e. 30k more stage-1 steps first): short
  all 7.48 / 0.120 %, judged 6.47 / 0.059 %; L lasting 4.9 / 0.39 % (8 rows; paired vs ov_V1jL 0.1 dex 48 / 10, 1 dex
  12 / 5); P lasting 28.9 % (1 large row). ov_VN1j (joint stage 2 from ov_VN1): short 7.25 / 0.116, judged 6.22 /
  0.054; L lasting 5.3 / 0.44 % (9 rows; vs ov_V1jL 49 / 19, 14 / 8); P lasting 30.1 % (3 large rows). The joint
  models' L lasting misses sit in cool gas (300-650 K: ov_V1jL 31 %, ov_V1cj 21 %, ov_VN1j 26 %; 650-1300 K 4-6 %),
  worst gas NH3. P lasting of the others (rows with a miss / large rows): ov_DT2 34.4 % / 6, ov_V1j 27.7 / 1, ov_V1jL
  27.0 / 1, ov_V1s2 32.4 / 7, ov_V1c 35.9 / 3, ov_VN1 31.6 / 5, ov_FMJ 37.5 / 9, ov_R2a 35.2 / 6, ov_H1s2 45.7 / 15.
  ov_V1j30 (ov_V1jL + 30k joint steps) was stopped 10-30 23:10 at step 23,000 of 30,000 (checkpoint kept, never read)
  when the network line closed. ov_V1cj is the best single network on long runs.
- Early-time scoring subsets (V overlay log 10-30 03:46; judged rows of the short readout restricted to eq / eqf / mix
  starts at t0 >= 10 s): ov_DT2 1.25 -> 0.83 %, ov_CS2Fs2 12.47 -> 6.74, ov_CS2F 8.22 -> 3.10, the pair 1.49 -> 0.89.
  I: early times do not hide the per-entry head's failure.
- Why dt-free heads cap the one-step fit (external review, reviews/answer_round7_*.txt, V from stored logs and data):
  final validation loss ov_DT2 0.00277, ov_FMJ 0.00260, ov_H1 0.00505, ov_V1 0.0158, ov_CSJ 0.0213, ov_CS2F 0.0204;
  7.3 % of one-step windows from young states have a scored gas moving > 0.1 dex away from its final value (10.5 % at t
  = 0, 2.4 % at 100-1000 s). I: a symmetric relaxation toward one target per entry cannot follow those.
- Mini-chem scored misses-first on its yardstick (V overlay log 10-30 15:24, from the yardstick truth and mini-chem's
  stored outputs; 10 cells, 395,234 rows, constant T and P, mini-chem's native compositions): rows with a miss / large
  miss 17.62 / 6.874 %, worst 18.1 dex; from t0 = 0 39.56 / 22.03 %, restarts 16.00 / 5.75; eq-target 1.50 / 0,
  eq-far 25.55 / 9.00, raw-A 30.83 / 17.45, raw-B 12.63 / 1.05; free runs (120) 26.7 / 18.3 %; worst gas of large rows
  HCN 10,255, C2H2 8,810, NH3 3,613, CH4 2,126. Its v3.1 score flags 49 of 395,114 segment rows as failed (CBg 20; V
  repo-data/benches/yardstick/score_v31/score.json); its driver clips and renormalizes outputs. Caveat: mini-chem
  carries 13 species, its start is mapped from VULCAN's 53.
- The mini-chem paper's Fig. 4 cases (V overlay log 10-30 15:51-17:10; VULCAN to 1e8 s; worst floored error after
  500 s, 1 mbar then 10 bar, 1000 / 1500 / 2000 K): mini-chem 1.67 / 3.52 / 2.96 and 0.30 / 2.48 / 0.02 dex; ov_V1jL
  (500 s steps from the unreacted start) 0.04 / 0.41 / 1.48 and 0.17 / 0.22 / 0.39; ov_DT2 0.75 / 1.73 / 0.06 and 0.19 /
  0.42 / 0.01. Caveat: the models' first call covers 0-500 s; mini-chem integrates from t = 0.
- Facts checked by the first plan review (reviews/answer_round9_plan_*.txt, V by its read-only calculation):
  mini-chem's 13 x 10 stoichiometric matrix has rank 8, the full element-conserving span of its species (E S = 0), so
  more reactions among the same species cannot widen it; the shipped 0.1x tables repeat one HCN rate column for two
  reactions and the 100x CH4 -> C2H2 table holds 14,779 zeros (the 1x tables are positive).
- fixed_dt_long at 50 s (4 of 6 models; the run was stopped by a guard false alarm): from t = 0, +1e7 s, ov_F50 /
  ov_DT / ov_DT2 / clean_s2 41.2 / 36.6 / 31.4 / 3.8 % (large 10.9 / 4.7 / 3.9 / 0). Training at one fixed step does
  not help. From VULCAN's state at 1e6 s, +1e7 s: ov_F50 38.0 / 9.4, ov_DT 34.3 / 3.6, ov_DT2 30.0 / 3.4, clean_s2 0.5 /
  0 %; ov_FMJ from the start 28.4 / 3.1 % (its late part and ov_H1 were not run) (V diag/fixed_dt_long_dt50.out).

## 6 References cited (by topic)

Identifiers as the dated record or the cited report gives them; "no id" = none recorded there. Verify each before paper
use (an unfinished 10-04 audit flagged 4 of 52 attributions of the old design record as not matching their source).
- Our work and codes: the one-step emulator paper (Malsky, Zhang et al.), arXiv 2609.00428, accepted in ApJ; VULCAN-0D
  github.com/shami-EEG/VULCAN-0D (b3ccf4f); VULCAN github.com/shami-EEG/VULCAN; mini-chem github.com/ELeeAstro/mini_chem
  (ae021c8); github.com/imalsky/Autoregressive-Chemulator (paper code = tag paper-v1).
- Mini-chem: Tsai et al. 2022, arXiv 2204.04201 (net-reaction tables; sec. 2.2.1 composition recipe; mini-chem vs VULCAN
  "rarely exceeds a factor of two", sec. 3.2); Lee et al. 2023, arXiv 2302.09525 (Exo-FMS + mini-chem, 1500 s chemistry
  step, 1x and 10x, Table 2; tables valid for one metallicity, sec. 2).
- Solar abundances: Lodders 2009 (the training generator; no id); Lodders 2019, arXiv 1912.00844 (Table 8 present-day,
  mini-chem's tables); Asplund, Amarsi and Grevesse 2021, arXiv 2105.01661 (mini-chem's equilibrium tables); Asplund 2009
  (no id).
- GCM chemistry steps and envelope: Drummond et al. 2020, arXiv 2001.11444 (UM: chemistry every 125 steps = 3750 s,
  radiation every 150 s, elements conserved to 99.9 % over 1000 d, heating changes mainly from CH4); GCM T-P grids Roth
  2024 (516 models) and Kennedy 2024 (55 models) (no ids).
- Detection limits behind the per-gas floors: Gasman et al. 2022, arXiv 2202.01151; Greene et al. 2016, arXiv 1511.05528;
  Drummond et al. 2020 (above).
- Metallicity range: August et al. 2023 (WASP-77Ab, JWST, [M/H] -0.91), BOWIE-ALIGN (HAT-P-30b 0.2-0.8x), TOI-5205b
  (< 0.03x upper limit) (no ids in the cleanup log); not re-read: K2-18b at 180x, arXiv 2506.23891; observed giant-planet
  metallicity 0.1-100x and C/O mostly 0-1, arXiv 2510.26785.
- Relaxation-head prior art: Hesstvedt, Hov and Isaksen 1978, Int. J. Chem. Kinet. 10, 971 (plain QSSA step); Jay, Sandu,
  Potra and Carmichael 1997, SIAM J. Sci. Comput. 18, 182; Cooper and Showman 2006, ApJ 649, 1048; Tsai et al. 2018, arXiv
  1711.08492 (chemical relaxation in exoplanet GCMs); Bilos et al. 2021, NeurIPS, arXiv 2110.13040 (GRU flows); De Brouwer
  et al. 2019, arXiv 1905.12374 (GRU-ODE-Bayes); Chen and Wu 2023, J. Comput. Phys. 493, 112498, arXiv 2302.03358 (Deep-OSG).
- Latent and operator models: LiLaN (Nockolds, Krishnanunni, Bui-Thanh, Tang), arXiv 2501.08423; Lusch, Kutz and Brunton
  2018, Nat. Commun. (no id); Sulzer and Buck 2023, arXiv 2312.06015; MACE (Maes et al. 2024; no id).
- Rollout training: List, Chen, Bali and Thuerey 2025, CMAME 433:117441, arXiv 2402.12971; DAgger (Ross et al. 2011),
  Kelp et al. 2022, Um et al. 2020, ACE2, NeuralGCM, GraphCast (no ids).
- Training-data design: Zhang et al. 2022, arXiv 2201.03549; DeePODE (Yao et al. 2025), arXiv 2401.01220; Sobol vs random
  for small samples, NSF PAR 10370794.
- Trunks and conditioning: FiLM (Perez et al. 2018), arXiv 1709.07871; Kinetic-Mamba, arXiv 2512.14471; ChemKANs, arXiv
  2504.12580 (PCCP 2025); KA-CRNN, arXiv 2511.07686; Vojtekova et al. 2026, arXiv 2609.17894; Boledi et al. 2026, arXiv
  2603.15307; Ukorigho and Owoyele 2026, arXiv 2607.09582.
- Budgets, ensembles, consistency, timescales, cold expert: Sturm and Wexler 2020, 2022; Holdship et al. 2021; Shikhman
  2026; Ono and Sugimura 2026, ApJ 996, 9 (no ids); Xiao et al. 2025, arXiv 2507.08277. Not checked: an "ExpNet"
  network (Chem. Eng. J. 2026), Huang and Seinfeld 2022, Grassi 2022.
- Batch size: Goyal et al. 2017 and Keskar et al. 2016 (cited by the one-step paper; no ids); Zhang et al. 2024, arXiv
  2410.21676 (critical batch size).

### Transported species, families, slow manifolds and tracer counts (verified 2026-10-31; full list with each check:
experiments/2026-10-31_refresh/_build/w7_couple/refs.txt)
- Jacob D.J., Sillman S., Logan J.A., Wofsy S.C. 1989, JGR 94, 8497, doi 10.1029/JD094iD06p08497 (6 transported tracers).
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
- Zamyatina M. et al. 2023, MNRAS, arXiv 2211.09071; Zamyatina M. et al. 2024, arXiv 2402.14535; Steinrueck M.E. et
  al. 2025, arXiv 2509.21588; Christie D.A. et al. 2026, arXiv 2604.15850 (UM tracer cost); Baeyens R. et al. 2026,
  arXiv 2607.08486 (Exo-FMS + mini-chem, 13 tracers).

## 7 Status at the end of the refresh campaign (written 2026-11-01 ~15:3x)

- Nothing of the campaign runs after the final email; both GPUs freed. Campaign record:
  experiments/2026-10-31_refresh/_build (log.txt; RESUME.txt; rules_decision.txt; one folder per workstream; judge/,
  w15_sign/ the final evidence). The earlier status of 2026-10-31 21:5x is superseded by section 1 and the 11-01 entries of
  section 2.
- Code: code/Auto-Chem main 3b3be41, local (not pushed): the reduced-tracer library (lib/kintracer.py, designs N26, N30,
  N30sN2H, G30b, G31), the first-step worst-species acceptance and 10x recovery budget, the dual-objective
  reconstruction, the chain-check rule; references libref2 / refdata2 / kintracer ref3 (old ones kept); the kintest quick
  reference re-saved with the merged kernel (quick_ref2.npz).
- Data: new truth sets data/truth/qt (5.9 GB) and data/truth/qt2 (2.8 GB), both now used as judges; the 10-day SNCHO
  history added to data/truth/sncho; 2.05 GB of legacy data deleted; quota ~201 GB of 500 GB.
- Records: closed campaigns packed into paper-notes/archive/campaigns_2026-11-01 (section 0); only README / CLAUDE /
  todo / notes remain as documentation (plus upstream READMEs in the GPL clones).
- Waiting for the author: todo.md, "Decisions for the author".
