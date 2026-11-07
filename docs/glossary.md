# Glossary

Names in the code that are shorter than a description. Math symbols follow the equations in [method.md](method.md).

## Physical quantities

| Name | Meaning | Unit |
| --- | --- | --- |
| `x` | mole fractions, one row per cell | |
| `T` | temperature | K |
| `P` | pressure | dyn cm^-2 |
| `dt` | length of a call | s |
| `h` | step size carried from call to call | s |
| `J` | photolysis rates, one column per branch | s^-1 |
| `lm` | log multiplier of a reaction's rate coefficients | |
| `nd` | total number density, P / (k_B T) | cm^-3 |
| `kf`, `kr` | forward and reverse rate coefficients | cm^3 s^-1 per molecule pair, s^-1 |

## Network and batch

| Name | Meaning |
| --- | --- |
| `S` | stoichiometry [species, thermal reactions]: products minus reactants |
| `Sp` | stoichiometry of the photolysis branches [species, branches] |
| `B` | cells in a batch |
| `D` | tangent directions in one derivative call |
| `dT`, `dP`, `dJ` | tangents of `T`, `P` and `J` |
| `NS`, `NR` | species and reactions in the kernels (`NR` holds the photolysis branches too) |

## Integrator

| Name | Meaning |
| --- | --- |
| `N` | the cell's state: molecules per starting molecule |
| `M` | the matrix `M0 = I / (gamma h) - Jn` on its sparse pattern, then its LU factors |
| `K`, `K1` to `K4` | Rodas3 stage vectors |
| `Y` | a stage's state; at the end of a try, the new state |
| `F` | the right-hand side at a stage's state |
| `u` | the constant-pressure term `w`; after stage 1, `M0^-1 w` |

## Kernel layout

| Name | Meaning |
| --- | --- |
| `W`, `solver.w` | warps per cell |
| `NL` | threads per cell, `32 W` |
| `CPB` | cells per block |
| `K0`, `NT` | rows of the sparse leading part and of the dense trailing block |
| `lane`, `cl`, `grp` | a thread's lane in its warp, its index in its cell, its cell in the block |
| `tab` | the small tables in shared memory (`Tab`) |
| `work` | a derivative kernel's per-warp work arrays in shared memory |
| `word` | one packed term of a lane stream |
| `pattern` | the sparsity pattern of the matrix after fill, as (row, column) pairs |

## Step control and derivative kernels

| Name | Meaning |
| --- | --- |
| `ha` | the size of this try: min(h, dt - t) |
| `en`, `pen`, `pha`, `prej` | a try's error; the previous try's error and size, and whether it was rejected |
| `nok` | accepted steps so far |
| `igh`, `ih` | 1 / (gamma h), 1 / h |
| `c0` | the Sherman-Morrison factor |
| `dfe`, `dfh`, `dfp`, `dfq` | the step factor's partial derivatives in en, ha, pen and pha |
| `hab`, `enb`, `hb`, `tb`, `dtb` | adjoints (cotangents) of ha, en, h, t and dt |
| `enc`, `hac` | cotangents passed to the previous attempt: of its error and of its size |
| `lbk`, `lbnd` | cotangents of the ln rate coefficients and of ln nd |
| `cot_k`, `lr` | a reaction's `lbk` entries: ln k common to both directions, ln kr alone |
| `rho` | a stage's cotangent times the signed rate coefficient, for the adjoint's gather |
| `wA`, `wG`, `wE` | the collision limit's weights on ln k, on ln kr - ln k, on ln nd |
| `Nd`, `Yd`, `Kd`, `hd` | tangents of `N`, `Y`, `K` and `h` |
| `Nb`, `Yb`, `Kb` | cotangents of `N`, `Y` and `K` |
| `wib`, `wid` | a warp's index in its block; in the whole grid |
| `myt` | this lane's row in the dense trailing block, or -1 |
| `CP` | attempts stored per cell: `CAP`, or `CAPR` on the recovery's path |
