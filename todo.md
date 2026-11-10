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
- 2026-11-04. One snapy branch after PR #334 merges.
  - The fork has two branches: isaac/recon-many-variables (the launch-split fix alone, PR #334) and
    isaac/maxichem-coupling (the same fix plus the fused tracer transport).
  - Done 2026-11-09: #334 is in snapy main (fe2edaa3); isaac/recon-many-variables is deleted on the fork and the fork's
    main is level with snapy main.
  - Still needed: bring isaac/maxichem-coupling up to the new main (merge it in and port the fused tracer path to main's
    new tracer code in src/mesh/meshblock.cpp, src/scalar/scalar.cpp and tests/CMakeLists.txt).
- 2026-11-05. Sphere heating bug (notes.md, 2026-11-05 18:53).
  - Bug: heat = -(A_top F_top - A_bot F_bot) / V with the column's plane-parallel fluxes gives a false 2 F / r per volume on
    the sphere. It cools the night side and heats the day side, most near the lid, and crashed 140 layers.
  - Correct form: heat = -(F_top - F_bot) / dz.
  - State: `coupling/radiation.py` holds the correct form and the per-column budget check, with tests that fail on the old form
    (notes.md, 2026-11-07 21:49). The three experiment drivers use it; only the box driver has run with it.
  - Still needed: run the sphere drivers (the rerun below); use the same form in the C++ sphere coupling when it is written.
- 2026-11-05. Rerun the sphere work with the corrected heating formula (notes.md, 2026-11-05 18:53), after the code cleanup.
  - The 2-day runs at 100, 120 and 140 layers from both starts, then the lid comparison on 0.1-1 bar and a lid choice.
  - The 10-day runs with and without chemistry and their figures, then the radiation-step and sponge tests as needed.
- 2026-11-09. Design debt in the kernels (found in a code review; no result is wrong because of it).
  - The memory layout has two owners. Python sets the call kernel's array order (`VOFF = nnz + 10 * ns + 3`,
    `maxichem/_kernel_header.py:132`); the scratch offset `kf + 2 * NR + 2` is written again in `call.cuh:31` and
    `recover.cuh:141`, with its own forms in `jvp.cuh:229` and `vjp.cuh:565`, and in `_tables.py` / `_ad_tables.py`. Needed:
    named offsets from the generator, as the `A_*` ones already are.
  - Argument counts are far above the limit of 12: kernels take 26 to 48 parameters (`call` 26, `recover` 28, `jvp` 40,
    `vjp` 43, `jvp_cells` / `jvp_rec` 45, `vjp_cells` / `vjp_rec` 48), helpers 15 to 21. `tools/check_size_limits.py` does
    not count arguments and no exception is written. Needed: a struct for the per-cell streams and one for the step state,
    then the count in the checker.
  - The per-cell setup and the accept / step-control tail are each written four times (`call.cuh`, `recover.cuh`,
    `jvp.cuh`, `vjp.cuh`) and no test ties the copies together (the derivative kernels' copies of the step itself are tied
    by `test_primal_is_the_call`). Needed: one shared form, or a test that the four agree.
  - Any of these changes the compiled kernels: each needs the bitwise proof on the saved cases and a speed check.
