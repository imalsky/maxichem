# API

Tensors are float64 CUDA tensors on the solver's device, one row per cell. Units: T in K, P in dyn cm^-2 (1 bar = 1e6), dt and h in
s, J in s^-1, x as mole fractions in the order of `net.species`. Importing the package does not import torch.

## `VulcanNetwork(network_file, nasa_dir, species=None, bulk="H2")`

Reads a VULCAN reaction list and a folder of `<species>.txt` NASA-9 files. `species` sets the order; the default is the order of
first appearance in the list, with He appended. `bulk` is the gas whose efficiency an `effi` line applies.

| Attribute | Meaning |
| --- | --- |
| `species`, `elements` | Names |
| `atoms` | [species, elements] |
| `lo`, `hi` | NASA-9 coefficients a1..a7, b1, b2 of the two ranges |
| `reac` | Thermal reactions, as dicts with their law |
| `S` | [species, reactions]: products minus reactants |
| `photo`, `Sp` | Photolysis branches and their stoichiometry |
| `photo_missing` | Species that branches name and the list lacks; `photo=True` refuses such a network |
| `skipped` | Condensation lines skipped |
| `duplicates` | Pairs of entries with the same species on both sides; their rates add, as in VULCAN |

It refuses a non-finite rate coefficient, a reaction whose atoms do not balance, a reaction above the first heading (it has no rate law), a reaction after
`# reverse stops` outside photolysis or condensation, a photolysis line whose reactant is not its absorber, a repeated absorber and
branch, a name given twice in `species`, and an `# irreversible` reaction: every reverse comes from NASA-9. Building a solver's tables
refuses a hand-built reaction with an explicit reverse law (`rev`), for the same reason.

From the first heading to the end of the thermal sections (`# reverse stops`, condensation, photolysis or `# re_end`), a comment must be a known
heading (`KNOWN_HEADINGS`), a line of column titles (`# id`), a commented-out reaction or empty: reactions under an unknown heading would
keep the law of the section above. Free text goes above the first heading or past those sections. A NASA-9 file must hold 20, 30 or 40
numbers (two to four ranges of 10; those past the second are unused), and G / RT may jump by at most `MAX_GIBBS_JUMP` at 1000 K. VULCAN's
`TiH.txt` fails this, so `TiSNCHO_photo_network.txt` is refused until that file is corrected.

## `CudaSolver(net, device, *, drift=None, photo=False)`

Builds the kernels on the device. `drift` is the element-drift allowance (default `DRIFT`). With `photo=True`, the network's photolysis
branches enter the call, which then takes J. Calls run one at a time on torch's current stream.

| Member | Meaning |
| --- | --- |
| `solver(x, T, P, dt, h, *, rtol, atol, max_steps, J)` | Returns a `CallResult` (x, h, done, steps); raises if it refused a cell |
| `launch(...)` | The same, as a `LaunchResult` with the per-cell flag added: 0 passed the guard, 1 recomputed and passes (a recomputation that drifts is retried with a smaller first try, which changes the computation only where the cap binds, [method.md](method.md#guard-and-recovery)), 2 recomputed and still fails (x, h, done 0 and steps are the last recovery pass's; x may be non-finite) |
| `flag` | The last call's per-cell flag (int32 [B]), set when `launch` or a call returns; None before the first call and after any call that raised (a refused cell has flag 0, so a flag of a refused call would read as passed) |
| `fallback` | The last call's counts, as a dict |
| `check()` | The counts since construction; raises if any cell was below `T_MIN` or above `T_MAX` |
| `export(folder)` | Writes what `cpp/maxichem.hpp` loads |
| `count`, `total` | The counts on the device: the last call's (int32), and those since construction (int64) |
| `info` | The kernel's layout |
| `close()` | Unloads the kernels and drops the tables and scratch; the counts stay readable |

J is [B, len(net.photo)], columns in `net.photo`'s order. It is required with `photo=True` (pass zeros for dark cells) and refused
without it.

The results are named tuples, so `x, h, done, steps = solver(...)` and indexing work as before:

| Result | Fields |
| --- | --- |
| `CallResult` | `x` [B, S] mole fractions; `h` [B] the next call's starting step, s; `done` [B] bool; `steps` [B] accepted steps, float64 |
| `LaunchResult` | `CallResult`'s fields, then `flag` [B] int32 |

Interface change: `rtol`, `atol`, `max_steps` and `J` are keyword only, in the call, `launch`, `jvp` and `vjp`. A call that passed
them by position, as `solver(x, T, P, dt, h, 1e-3, 1e-25)`, now raises TypeError: write `solver(x, T, P, dt, h, rtol=1e-3,
atol=1e-25)`. Before, swapping the two tolerances passed every check. Unpacking and indexing the results is unchanged. The
constructors' optional arguments are keyword only too: `CudaSolver(net, device, True)` had set drift to 1 with photolysis off;
write `drift=` and `photo=` (and `cap=` for `ADSolver`).

A cell is refused when it is bad input (a negative or non-finite mole fraction, |sum(x) - 1| above `X_SUM_TOL`, no mole fraction
above 0, a non-finite T, P <= 0, a negative or non-finite dt, or a negative or non-finite J) or outside [`T_MIN`, `T_MAX`] (counted
as cold or hot). The kernel that steps the cell decides this once (method.md, "Refused cells"); it does not step the cell, and the
guard counts it. After the kernels, `launch` and a call read the counts back, the call's one host sync, and raise ValueError when any
cell was refused, naming the count of each kind, the limits and the first refused cell of each kind with its values. The counts are
added to `total` first, and `fallback` reads them after the raise. `ADSolver`'s calls refuse the same cells; a non-finite log
multiplier is not a refusal there (the cell fails, flag 2). Inputs of the wrong shape, dtype or device raise before any launch.

A non-finite h acts as dt and is counted in `nonfinite_h`. `launch` and a call both add their counts to `total`, which `check()` reads.

## `ADSolver(net, device, *, cap=CAP, drift=None, photo=False)`

| Member | Meaning |
| --- | --- |
| `jvp(x, T, P, dt, h, dx, dT, dP, ddt, dh, *, lm, dlm, rc, drc, adapt, rtol, atol, max_steps, J, dJ)` | Returns a `JvpResult` (x, h, done, steps, dx, dh) |
| `vjp(x, T, P, dt, h, gx, *, gh, lm, rc, adapt, rtol, atol, max_steps, J)` | Returns a `VjpResult` (x, h, done, steps, cotangents) |
| `fallback`, `check()`, `flag` | As `CudaSolver` |
| `rc0` | The network's rate table [R, W], which a call uses unless given another |
| `close()` | As `CudaSolver` |

Everything after the required tensors (after `dh` in `jvp`, after `gx` in `vjp`) is keyword only. The results are named tuples:
`JvpResult` holds `CallResult`'s fields, then `dx` ([B, S], or [D, B, S] with a leading axis of directions) and `dh` ([B] or [D, B],
s); `VjpResult` holds `CallResult`'s fields, then `cotangents`, a dict: x [B, S], T [B], P [B], dt [B], h [B], lm [B, R], rc [R, 9]
and, with `photo=True`, J [B, branches], each per unit of its input. The per-cell flag is `flag`, as in `CudaSolver`.

A non-finite log multiplier in `lm` takes no try: its cell comes back with done 0, flag 2 and NaN derivatives, counted as unfinished,
recomputed and still failed.

A cell with flag 2 has NaN in its own tangents and in its cotangents x, T, P, dt, h, lm and J. The rate-table cotangent `rc` is a sum over the
batch, so it leaves such a cell out and stays the sum over the other cells: read `fallback["still_failed"]` before using it, since it counts
the cells left out.

`cap` is a positive integer; `drift`, in both solvers, is finite and at least 0.

With a leading axis of at most `DMAX` directions on every tangent, the directions share one call. See
[method.md](method.md#derivatives) for `adapt`.

## Constants in `kincuda`

`ATOL`, `DRIFT`, `REC_STEPS`, `T_MIN`, `T_MAX`, `X_SUM_TOL`, `COUNTS`. The result types are `kincuda.CallResult`,
`kincuda.LaunchResult`, `kincuda_ad.JvpResult` and `kincuda_ad.VjpResult`.

## C++: `maxichem::Solver(folder, device)`

Loads an export from `cpp/maxichem.hpp`. `chem(x, T, P, dt, h, J, rtol, atol, max_steps)` returns the new mole fractions, the next
step sizes, done and the accepted steps. A negative `atol` takes the export's default. It refuses the cells `CudaSolver` refuses,
with the same message, as `std::invalid_argument`; the export's `meta.txt` carries the limits. A defined J must be float64
[B, `nj()`] on the device; an export without photolysis (`nj()` 0) takes an undefined J or an empty [B, 0] one, and refuses any
other. Before any CUDA work, the constructor checks each kernel's argument list in `meta.txt` against the one the call passes
(`KERNEL_ARGUMENTS`, with J last for call and recover with photolysis): an unknown, missing, repeated or misplaced name, or J on
the guard, is refused as `std::runtime_error` naming the kernel and the argument.

| Member | Meaning |
| --- | --- |
| `ns()`, `nj()` | Species and photolysis branches |
| `species()`, `elements()` | Names (const references) |
| `default_atol()` | The export's atol, mole fraction |
| `atoms()` | [species, elements] on the device, a copy |
| `count()`, `total()` | The last call's counts (int32) and their sum since construction (int64), copies on the device |

Each tensor accessor (`atoms()`, `count()`, `total()`) makes a device copy and launches a kernel to do it, so keep its result rather
than calling it in a hot loop.

Interface change: these were public data members (`chem.ns`, `chem.nj`, `chem.species`, `chem.elements`, `chem.default_atol`,
`chem.count`, `chem.total`); write them with `()`. The call trusts them and its kernels write to the counts, so a caller can no
longer rebind or resize them.
