# To do

Tasks put off to return to, with the date deferred.

- 2026-11-03. Reduced advected tracers for the current networks (notes.md, "Reduced tracers").
  - Idea: advect fewer tracers than species and rebuild the rest in each call. The removed call did this with 31
    tracers for the old 53-species network (code at commit 399a347).
  - Why: with snapy's fused tracer path (2026-11-04) transport costs about 11 ms per tracer per step at C64 x 100
    (44 ms before) and about 0.38 GB per tracer. 69 tracers and chemistry use 41 GB of a 48 GB card in the Python
    driver; 91 (the 2025 sulfur network) take 38 GB for transport alone at C64 (2026-11-04), so with chemistry they
    likely do not fit on this card (they would on a GH200).
  - Known: it ran in the coupled snapy test with exact element totals. Its weak spot was carbon-rich gas that keeps
    mixing (C/O 1-2: 1.3 to 3.2 dex misses).
  - Needed: a tracer design for each network (69 and 91 species), the rebuild kernel and its own reference. Adopt it
    only if the method stays simple.
- 2026-11-03. Benchmark on the NAS GH200s.
  - The call is fp64 throughout. The RTX 6000 Ada runs fp64 at 1/64 of its fp32 rate; the GH200's fp64 peak is far
    higher, so the cost per cell-call should drop a lot (not measured).
  - Run the speed tables of README.md (quiet gas and the mixing bed, NCHO_photo and SNCHO_photo_network_2025) and the
    checks there.
- 2026-11-05. Sphere heating bug (notes.md, 2026-11-05 18:53).
  - Bug: heat = -(A_top F_top - A_bot F_bot) / V with the column's plane-parallel fluxes gives a false 2 F / r per volume on
    the sphere. It cools the night side and heats the day side, most near the lid, and crashed 140 layers.
  - Correct form: heat = -(F_top - F_bot) / dz.
  - State: `maxichem/coupling/radiation.py` holds the correct form and the per-column budget check, with tests that fail on the old form
    (notes.md, 2026-11-07 21:49). The three experiment drivers use it; only the box driver has run with it.
  - Still needed: run the sphere drivers (the rerun below); use the same form in the C++ sphere coupling when it is written.
- 2026-11-05. Rerun the sphere work with the corrected heating formula (notes.md, 2026-11-05 18:53), after the code cleanup.
  - The 2-day runs at 100, 120 and 140 layers from both starts, then the lid comparison on 0.1-1 bar and a lid choice.
  - The 10-day runs with and without chemistry and their figures, then the radiation-step and sponge tests as needed.
