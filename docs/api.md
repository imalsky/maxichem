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

## `CudaSolver(net, device, drift=None, photo=False)`

Builds the kernels on the device. `drift` is the element-drift allowance (default `DRIFT`). With `photo=True`, the network's photolysis
branches enter the call, which then takes J. Calls run one at a time on torch's current stream.

| Member | Meaning |
| --- | --- |
| `solver(x, T, P, dt, h, rtol, atol, max_steps, J)` | Returns (x, h, done, accepted steps); raises if it refused a cell |
| `launch(...)` | The same, with a per-cell flag: 0 passed the guard, 1 recomputed and passes, 2 recomputed and still fails |
| `fallback` | The last call's counts, as a dict |
| `check()` | The counts since construction; raises if any cell was below `T_MIN` or above `T_MAX` |
| `export(folder)` | Writes what `cpp/maxichem.hpp` loads |
| `count`, `total` | The counts on the device: the last call's (int32), and those since construction (int64) |
| `info` | The kernel's layout |
| `close()` | Unloads the kernels and drops the tables and scratch; the counts stay readable |

J is [B, len(net.photo)], columns in `net.photo`'s order. It is required with `photo=True` (pass zeros for dark cells) and refused
without it.

A cell is refused when it is bad input (a negative or non-finite mole fraction, |sum(x) - 1| above `X_SUM_TOL`, no mole fraction
above 0, a non-finite T, P <= 0, a negative or non-finite dt, or a negative or non-finite J) or outside [`T_MIN`, `T_MAX`] (counted
as cold or hot). The kernel that steps the cell decides this once (method.md, "Refused cells"); it does not step the cell, and the
guard counts it. After the kernels, `launch` and a call read the counts back, the call's one host sync, and raise ValueError when any
cell was refused, naming the count of each kind, the limits and the first refused cell of each kind with its values. The counts are
added to `total` first, and `fallback` reads them after the raise. `ADSolver`'s calls refuse the same cells; a non-finite log
multiplier is not a refusal there (the cell fails, flag 2). Inputs of the wrong shape, dtype or device raise before any launch.

A non-finite h acts as dt and is counted in `nonfinite_h`. `launch` and a call both add their counts to `total`, which `check()` reads.

## `ADSolver(net, device, cap=CAP, drift=None, photo=False)`

| Member | Meaning |
| --- | --- |
| `jvp(x, T, P, dt, h, dx, dT, dP, ddt, dh, lm, dlm, rc, drc, adapt, rtol, atol, max_steps, J, dJ)` | Returns (x, h, done, accepted steps, dx, dh) |
| `vjp(x, T, P, dt, h, gx, gh, lm, rc, adapt, rtol, atol, max_steps, J)` | Returns (x, h, done, accepted steps, cotangents dict) |
| `fallback`, `check()`, `flag` | Counts as `CudaSolver`, and the last call's per-cell flag |
| `rc0` | The network's rate table [R, W], which a call uses unless given another |
| `close()` | As `CudaSolver` |

A non-finite log multiplier in `lm` takes no try: its cell comes back with done 0, flag 2 and NaN derivatives, counted as unfinished,
recomputed and still failed.

A cell with flag 2 has NaN in its own tangents and in its cotangents x, T, P, dt, h, lm and J. The rate-table cotangent `rc` is a sum over the
batch, so it leaves such a cell out and stays the sum over the other cells: read `fallback["still_failed"]` before using it, since it counts
the cells left out.

`cap` is a positive integer; `drift`, in both solvers, is finite and at least 0.

With a leading axis of at most `DMAX` directions on every tangent, the directions share one call. See
[method.md](method.md#derivatives) for `adapt`.

## Constants in `kincuda`

`ATOL`, `DRIFT`, `REC_STEPS`, `T_MIN`, `T_MAX`, `X_SUM_TOL`, `COUNTS`.

## C++: `maxichem::Solver(folder, device)`

Loads an export from `cpp/maxichem.hpp`. `chem(x, T, P, dt, h, J, rtol, atol, max_steps)` returns the new mole fractions, the next
step sizes, done and the accepted steps. A negative `atol` takes the export's default. It refuses the cells `CudaSolver` refuses,
with the same message, as `std::invalid_argument`; the export's `meta.txt` carries the limits.
