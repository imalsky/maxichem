# To do

Tasks Isaac put off and wants to return to, with the date deferred.

- 2026-11-03. Reduced advected tracers for the current networks (notes.md, "Reduced tracers").
  - Idea: advect fewer tracers than species and rebuild the rest in each call. The removed call did this with 31
    tracers for the old 53-species network (code at commit 399a347).
  - Why: transport costs about 47 ms per tracer per GCM step at C64 x 100 (about 67 min per model day for 69 tracers) and
    about 0.49 GB of GPU memory per tracer. 69 tracers nearly fill a 48 GB card; 88 (sulfur) do not fit.
  - Known: it ran in the coupled snapy test with exact element totals. Its weak spot was carbon-rich gas that keeps
    mixing (C/O 1-2: 1.3 to 3.2 dex misses).
  - Needed: a tracer design for each network (69 and 88 species), the rebuild kernel and its own reference. Adopt it
    only if the method stays simple.
- 2026-11-03. Faster tracer advection in snapy (notes.md, "GCM coupling"); a change to the GCM, not to maxichem.
  - snapy already moves the tracers as one array. Per tracer at C64: about 43 ms per step, of which the WENO5
    reconstruction takes 16.5 ms and the flux-covariance term 8.9 ms.
  - PLM reconstruction for the tracers cut the 53-tracer step from 2.70 to 2.20 s (mass drift <= 1e-15, positivity
    kept; more diffusive).
  - fp32 tracer reconstruction: about 2-3x on the tracer part (an estimate; not measured).
  - Not worth it: CUDA graphs, torch.compile, stream overlap.
- 2026-11-03. Benchmark on the NAS GH200s.
  - The call is fp64 throughout. The RTX 6000 Ada runs fp64 at 1/64 of its fp32 rate; the GH200's fp64 peak is far
    higher, so the cost per cell-call should drop a lot (not measured).
  - Run the speed tables of README.md (quiet gas and the mixing bed, NCHO_photo and SNCHO_photo) and the checks there.
- 2026-11-03. Benchmark the chemistry on realistic transport.
  - The cost per cell-call depends on the solver steps per call: about 1 on quiet gas (0.35 us, NCHO_photo), about 27
    on the mixing bed (7.5 us). Real transport sits somewhere between; the one coupled test (old 53-species network,
    atol 1e-16) took 1.03 steps per call over 2 h of model time.
  - Measure steps per call and cost in a coupled snapy run on the current networks at the default atol, over many
    model days.
