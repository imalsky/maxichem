# To do

Tasks Isaac put off and wants to return to, with the date deferred.

- 2026-11-03. Reduced advected tracers for the current networks (notes.md, "Reduced tracers").
  - Idea: advect fewer tracers than species and rebuild the rest in each call. The removed call did this with 31
    tracers for the old 53-species network (code at commit 399a347).
  - Why: with snapy's fused tracer path (2026-11-04) transport costs about 11 ms per tracer per step at C64 x 100
    (44 ms before) and about 0.38 GB per tracer. 69 tracers and chemistry use 41 GB of a 48 GB card in the Python
    driver; 88 (sulfur) likely do not fit at C64 on this card (they would on a GH200).
  - Known: it ran in the coupled snapy test with exact element totals. Its weak spot was carbon-rich gas that keeps
    mixing (C/O 1-2: 1.3 to 3.2 dex misses).
  - Needed: a tracer design for each network (69 and 88 species), the rebuild kernel and its own reference. Adopt it
    only if the method stays simple.
- 2026-11-03. Benchmark on the NAS GH200s.
  - The call is fp64 throughout. The RTX 6000 Ada runs fp64 at 1/64 of its fp32 rate; the GH200's fp64 peak is far
    higher, so the cost per cell-call should drop a lot (not measured).
  - Run the speed tables of README.md (quiet gas and the mixing bed, NCHO_photo and SNCHO_photo) and the checks there.
