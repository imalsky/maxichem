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

It refuses a reaction whose atoms do not balance, a reaction after `# reverse stops` outside photolysis or condensation, a
photolysis line whose reactant is not its absorber, a repeated absorber and branch, and an `# irreversible` reaction: it holds no
explicit reverse law.

## `CudaSolver(net, device, drift=None, photo=False)`

Builds the kernels on the device. `drift` is the element-drift allowance (default `DRIFT`). With `photo=True`, the network's
on torch's current stream.

| Member | Meaning |
| --- | --- |
| `solver(x, T, P, dt, h, rtol, atol, max_steps, J)` | Returns (x, h, done, accepted steps), with no host sync |
| `launch(...)` | The same, with a per-cell flag: 0 passed the guard, 1 recomputed and passes, 2 recomputed and still fails |
| `fallback` | The last call's counts, as a dict |
| `check()` | The counts since construction; raises if any cell was below `T_MIN` |
| `export(folder)` | Writes what `cpp/maxichem.hpp` loads |
| `count`, `total` | The counts on the device: the last call's (int32), and those since construction (int64) |
| `info` | The kernel's layout |
| `close()` | Unloads the kernels |

J is [B, len(net.photo)], columns in `net.photo`'s order; `None` means 0. A row with a negative or non-finite J or a negative mole
fraction is bad input: not stepped, and counted.

## `ADSolver(net, device, cap=CAP, drift=None, photo=False)`

| Member | Meaning |
| --- | --- |
| `jvp(x, T, P, dt, h, dx, dT, dP, ddt, dh, lm, dlm, rc, drc, adapt, rtol, atol, max_steps, J, dJ)` | Returns (x, h, done, accepted steps, dx, dh) |
| `vjp(x, T, P, dt, h, gx, gh, lm, rc, adapt, rtol, atol, max_steps, J)` | Returns (x, h, done, accepted steps, cotangents dict) |
| `status`, `check()`, `flag` | Counts as `CudaSolver`, and the last call's per-cell flag |
| `close()` | Unloads the kernels |

With a leading axis of at most `DMAX` directions on every tangent, the directions share one call. See
[method.md](method.md#derivatives) for `adapt`.

## Constants in `kincuda`

`ATOL`, `DRIFT`, `K_COLL`, `REC_STEPS`, `T_MIN`, `COUNTS`.

## C++: `maxichem::Solver(folder, device)`

Loads an export from `cpp/maxichem.hpp`. `chem(x, T, P, dt, h, J, rtol, atol, max_steps)` returns the new mole fractions, the next
step sizes, done and the accepted steps. A negative `atol` takes the export's default.
