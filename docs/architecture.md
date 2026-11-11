# Architecture

How the code is laid out, and why the kernels look as they do. [method.md](method.md) has the equations.
[glossary.md](glossary.md) lists the short names.

## Modules

| Module | Job |
| --- | --- |
| `kinvulcan.py` | Reads a VULCAN reaction list and NASA-9 files into a `VulcanNetwork` |
| `_tables.py` | Builds the kernels' tables from a network: species order, rate laws, term streams, LU pattern |
| `_ad_tables.py` | The derivative kernels' extra tables and shared-memory layout |
| `_kernel_header.py` | Writes one network's numbers as the header `network.h` |
| `kernels/` | The kernels themselves, as `.cu` and `.cuh` files |
| `_cuda.py` | NVRTC and the CUDA driver through ctypes: compile, load, launch |
| `_export.py` | Writes a solver's kernels and tables for the C++ caller |
| `_solver.py` | What the two solvers share: the kernels' lifetime, the counts, the input checks, the shared tables |
| `kincuda.py` | `CudaSolver` |
| `kincuda_ad.py` | `ADSolver` |
| `_constants.py` | Settings and tables shared by the modules above |
| `paths.py` | Data locations of the checks and benches |
| `_checks.py` | What the checks and benches share: the scored gases and their floors, the start gas |

The subpackage `maxichem.coupling` (`maxichem/coupling/`) holds the pieces that the snapy coupling drivers (scripts in
`experiments/`) share, each held once and tested on the CPU. `import maxichem` does not import it. The drivers stay scripts and may print; these modules raise, with the offending value and the
fix, and never fall back silently. They import torch and numpy, and `chemistry.py` imports maxichem's constants; none imports
snapy or pyharp. The GCM side is in SI units; `chemistry.PA_TO_DYN_CM2` converts the pressure at the solver's edge.

| Module | Job |
| --- | --- |
| `composition.py` | Molar masses from a network's atoms table and pyharp's atomic weights; mass mixing ratios (kg per kg of gas) to mole fractions and back, each cell keeping its tracer mass; signed mole fractions for output only (the dumps keep the transport's negatives); element moles and the tracers' totals; the gas constant; the start gas from element ratios per H nucleus (`START_PER_H`), all C in CO, the rest of O in H2O, N in N2, S in H2S, refusing a network that lacks a carrier |
| `radiation.py` | The heating of each layer from two-stream fluxes, $-\partial(F_\uparrow - F_\downarrow)/\partial z$ per unit volume, also on a sphere (weighting plane-parallel fluxes by face areas would add a false $2F/r$); each column's budget, which takes float64 only and refuses non-finite fluxes, heating or residuals, bad thicknesses and a residual above round-off; the face temperatures for the longwave ($1.5 T_0 - 0.5 T_1$ at the ends of a uniform grid); `RadiationClock`, a fixed schedule whose mean cadence is the period, keeping its own time as the exact sum of the steps it is given, so a long run accumulates no rounding |
| `chemistry.py` | The solver's pressure unit, and `require_finished`: a call that leaves cells not done stops the run, with the cause from the call's counts (cold, hot, bad input, still failing after the recovery) and its fix |
| `flow_file.py` | Spun-up flows and start columns saved with the gas (composition, cv_R, the model's molar mass) and grid they were made for; loading refuses another gas or grid, and a file with no record or a malformed one |
| `photolysis.py` | The stand-in J of a stored table: the layer nearest in log pressure on the day side, zero at night, with the day-side cells beyond the table's range counted |

## From a network to a kernel

The kernel files never change with the network. `_kernel_header.py` writes sizes, offsets, masks and rate-law switches as macros
in `network.h`, which reaches NVRTC as an in-memory header. The kernels select their forms with `#if`, for example `HAS_TROE`
and `NJ`. NVRTC compiles the result once, at construction, with the compiler pinned in `pyproject.toml`: another compiler can
fuse other floating-point operations and change the bits.

`kincuda.cu` holds the call, guard and recovery kernels. `jvp.cu` and `vjp.cu` include it and add the derivative kernels, so
every helper of the call is shared.

| File | Holds |
| --- | --- |
| `common.cuh` | Constants, the arithmetic every kernel shares as forced-inline helpers, the right-hand side `rhs`, the triangular solve `msolve` |
| `call.cuh` | The `call` kernel |
| `guard.cuh` | `check_cell` and the `guard` kernel |
| `recover.cuh` | The dense factorization and the `recover` kernel |
| `ad_common.cuh` | The call's arithmetic on one warp, rates and their derivatives, attempts |
| `jvp.cuh`, `vjp.cuh` | The tangent and adjoint kernels |

## Layout of the call kernel

A cell has `W` warps: 2, or 4 when the shared memory holds at most 3 cells per block. At 4 warps x 32 lanes and 160-190 registers
per thread, the register file already caps a block at 2 or 3 cells, so the extra lanes cost no cells.

- **Trailing block.** The last `NT = min(NTAIL, NS)` rows and columns of the elimination order are dense or nearly so after the
  fill. They are held as a dense array after the other slots and factored in registers of the cell's first warp. Lane l owns
  block row l, and rows l, l + 32, ... of the sparse leading part in the solves. While the block is in registers, its
  shared-memory slots hold the reaction rates and the S v pieces, and they pass the factors to the second warp.
- **Streams.** The S v sums, the Jacobian terms and the LU updates run as balanced streams over the cell's 32 W lanes. A term is one
  packed word, with bit widths set by the network's sizes. A slot longer than an even share is cut into pieces added in order.
- **Reciprocals.** Those of the sparse pivots are taken in batches of consecutive independent pivots.
- **Solves.** The four stage solves run from one copy of the triangular solve. Stage 1 solves K1 on the first warp and M0^-1 w on
  the second. Each stage ends with the Sherman-Morrison update on the first warp.
- **Cells per block.** As many as the shared memory, the named barriers and the registers allow, with no more local memory than a
  one-cell build. They change no arithmetic.
- **Queue.** Persistent cells take work from an atomic queue, so no cell waits for the stiffest one. Results do not depend on the
  batch size or order.

Shared arithmetic is `__forceinline__`: a noinline helper changes which operations fuse, and so the bits. `rhs`, `rhs1` and
`check_cell` stay out of line, and their sums that reach the state use explicit `fma`. What stays repeated says
why beside it: the kernels' `rhs` and `rhs1` differ in barriers and lanes, the call's LU reads its update words ahead, and the
derivative forms of the rate law (`rate_law_partials`) need the terms the primal `law` folds. `test_ad.py` holds them to the call's bits and to
finite differences.

The derivative kernels use one warp per cell. Each lane walks `W` of the call kernel's lane streams, so every sum of the primal
is the call kernel's. They take networks up to 96 species (64 leading rows and the 32-row trailing block) and 2,047 reactions.

## Tables shared by Python and the kernels

Some facts must stay equal in Python, the kernels and the C++ caller. Each is defined once in Python and reaches the
kernels through `network.h`; the C++ caller keeps copies of a few (`kExportFormat`, `kGuardBlockThreads`, `kWarpLanes`,
`kPhotolysisArgument`, `kDefaultRtol`, `kDefaultMaxSteps`), held equal by `tests/test_constants_source.py`. `_constants.py`
holds the solver's settings; the coupling's few constants are named in its own modules (`maxichem/coupling/`), and hardware
limits and file formats beside the code that meets them (`kincuda.py`, `kinvulcan.py`).
The kernels' memory has one owner: each region (the call's shared memory per cell and its rates region, the
recovery's, the scratch slots and the copy kept for a retry, a cell's rates, the tangents' directions, the adjoint's
stored attempt) is an ordered list of arrays and sizes in `_tables.py` (`call_layout`, `call_rates_layout`,
`recovery_layout`, `scratch_layout`, `retry_layout`, `rate_slots`, `dense_w_doubles`) or `_ad_tables.py`
(`scratch_layouts`, and `layout` for the derivative kernels' shared memory). `network.h` names each offset after its
region (`C_DINV`, `CR_PART`, `R_PART`, `S_RSC`, `RSC_JN`, `RATE_ZERO`, `JD_NG`, `VS_RHO`, `REC_PEN`, `W_DOUBLES`, ...)
and the launchers take the sizes from the same lists. `_ad_tables.py` also holds
the fields of the adjoint's stream words (`GATHER_FIELDS`), and `_constants.py` the rest, in sections:
the call settings, the data the reader accepts (`MAX_GIBBS_JUMP`), the method settings (step control, the recovery's
pivoting, the guard's correction, the dense trailing block `NTAIL`, the warps per cell), the physical constants (`KB`, `P0`, `NASA_SPLIT_T`, `K_COLL`) and the layout. The kernels keep a `constexpr` of each method setting's
name, set from the header's `NAME_VALUE` macro. Among the layout:

- `WARP_LANES`, `DOUBLE_BYTES`: the units of the launch sizes and the shared memory.
- `KIND_*`: VULCAN's numbers for a reaction's law, which the reader's dicts hold.
- `RATE_LAW_FLAGS`, `RATE_LAW_SHIFTS`: the bits of a reaction's flag word.
- `COUNT_SLOTS`: the device's count array.
- `KERNEL_ARGUMENTS`: each kernel's parameters, in order. `CudaSolver.launch`, `ADSolver` and the C++ caller build their launches
  from it. A test compares it with the kernel files.
- `EXPORT_FORMAT`: the export's version. The C++ caller refuses any other.

## The C++ caller

`CudaSolver.export` writes the kernels' source, the cubin for this GPU, every table as a raw file and `meta.txt`. `cpp/maxichem.hpp`
loads the export and launches the same kernels with the same arguments, on torch's current stream, so a call gives the same bits
as in Python. A cubin is for one GPU architecture.
