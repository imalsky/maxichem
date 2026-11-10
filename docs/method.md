# Method

maxichem solves a VULCAN reaction network as a 0-D kinetics problem, one independent cell at a time, in CUDA kernels. This
page gives the equations and the production settings. The named constants are in `maxichem/_constants.py` and at the top of
`maxichem/kernels/common.cuh`. [glossary.md](glossary.md) lists the short names.

## Reactor

Within a call, each cell holds T and P constant. The state is N, molecules per starting molecule:

$$\frac{dN}{dt} = \frac{\sum_i N_i}{n_\mathrm{tot}} S\,v(n), \qquad n = n_\mathrm{tot} \frac{N}{\sum_i N_i}, \qquad n_\mathrm{tot} = \frac{P}{k_B T}.$$

S is the stoichiometry (products minus reactants) and v the reaction rates by mass action. In mole fractions x this is the
constant-pressure form

$$\frac{dx}{dt} = \frac{S v - x \sum_i (S v)_i}{n_\mathrm{tot}}.$$

## Rate laws

The reader takes the laws of VULCAN's network files (Tsai et al. 2017, 2021). Each is evaluated from T and n_tot:

| Section of the file | Law |
| --- | --- |
| two-body | $k = A T^B e^{-C/T}$ |
| 3-body | $k = F\,k_0 / (1 + k_0 n_\mathrm{tot} / k_\infty)$; both directions times $n_\mathrm{tot}$ |
| 3-body, no high-pressure rate | $k = k_0$; both directions times $n_\mathrm{tot}$ |
| special | a law held in the code, by equation |
| photolysis | first order, $k = J$, with J (s^-1) given per cell by the caller; no reverse |

$F = 1$, or the Troe factor of a following `Troe [a, T3, T1, T2]` line (Venot et al. 2012, Appendix C), with
$P_r = k_0 n_\mathrm{tot} / k_\infty$. A following `effi {...}` line multiplies k by the bulk gas's efficiency.

The special law is OH + CH3 + M -> CH3OH + M. Its fits are those of Jasper et al. (2007), in the form of Visscher & Moses
(2011, Eqs. 13 and 14): two k0 terms, and a Troe factor without Troe's shift terms,
$\log_{10} F = \log_{10} F_c / (1 + (\log_{10} P_r / N)^2)$ with $N = 0.75 - 1.27 \log_{10} F_c$.

Reverse rates come from the equilibrium constants of the NASA-9 data (McBride et al. 2002):

$$k_r = \frac{k_f}{K} \left(\frac{k_B T}{P_0}\right)^{\sum_i S_i}, \qquad K = e^{-\Delta G / R T}, \qquad P_0 = 1\ \mathrm{bar}.$$

The low-temperature NASA-9 range applies below `NASA_SPLIT_T`.

### Collision limit

A direction of two or three molecules may not exceed `K_COLL` as a bimolecular-equivalent rate,
$k\,n_\mathrm{tot}^{m - 2}$. Where one does, both directions of the reaction are scaled by `K_COLL / q`, with q the larger
direction's rate. Their ratio, and so the equilibrium, is kept. The limit bounds rate fits used far from their range, such
as the reverses of CH + M -> C + H + M and C2H + M -> C2 + H + M at low T.

## Integrator

Rodas3 (Sandu et al. 1997): four stages, order 3, L-stable, stiffly accurate, $\gamma = 1/2$. The form is

$$W = \frac{I}{\gamma h} - J_N, \qquad K_i = W^{-1}\left(f(Y_i) + \sum_j \frac{C_{ij}}{h} K_j\right), \qquad Y_i = N + \sum_j A_{ij} K_j,$$

with $Y_2 = N$ (stage 2 reuses $f(N)$), $Y_3 = N + 2 K_1$, $Y_4 = N + 2 K_1 + K_3$; $C_{21} = 4$, $C_{3j} = (1, -1)$,
$C_{4j} = (1, -1, -8/3)$. The new state is $N' = N + 2 K_1 + K_3 + K_4$, and $K_4$ is the error estimate.

The stage right-hand sides are written with explicit `fma`, so every kernel that repeats the call's arithmetic gets the same
bits whatever the compiler contracts. `tests/test_order.py` measures the order.

### Linear algebra

The exact Jacobian is $J_N = J_n + w\,\mathbf{1}^T$: $J_n = \partial(S v)/\partial n$, and w is the constant-pressure term.
$M_0 = I / (\gamma h) - J_n$ is factored by LU without row exchanges on its sparse pattern, in a minimum-degree order. The
Sherman-Morrison formula adds the rank-one term.

### Step control

- The first try is $\max(\min(h, dt), h_\mathrm{min})$, with h the step size the cell carried from its last call. A
  non-finite h acts as dt, and is counted. Each try is $\min(h, dt - t)$.
- With $q_i = K_{4,i} / (\mathrm{atol} + \mathrm{rtol}\max(|N_i|, |N'_i|))$, err is the RMS of q. While the call has no
  accepted step, err is $\max(\mathrm{RMS}, \max_i |q_i|)$: the first try uses a step size from the last call, and the gas may
  have changed abruptly, so one fast species' error must not hide in the mean.
- A step is accepted when err is at most 1 and finite, and no species is below -atol. Accepted states are clamped at 0.
- The next step is the try's size times $f = 0.9\,\mathrm{err}^{-1/3}$, within `[FACACC, FACMAX]` after an accepted step and
  `[FACREJ, FACMAX]` after a rejected one, and at most `FAC_AFTER_REJECT` after a rejection. A non-finite err gives
  `FAC_NONFINITE`.
- A rejection that follows a rejection fits err ~ h^p to the two tries, $p = \ln(\mathrm{err}_1 / \mathrm{err}_2) / \ln(h_1 / h_2)$,
  and aims at err 0.5: $f = \max((0.5 / \mathrm{err})^{1 / \min(p, 3)}, \mathrm{SECMIN})$, or SECMIN when p is below
  `SECANT_MIN_ORDER`. After an abrupt change of the gas, the error of a long try falls far more slowly than h^3, so the plain
  shrink would take many tries.
- A try after a rejection starts from the same state, so it reuses that try's f(N), Jacobian and w.
- The call ends at `CALL_END` times dt, or after max_steps tries. It returns the mole fractions, the next step size, done and
  the accepted steps.

Production settings: rtol `DEFAULT_RTOL`, atol `ATOL`, max_steps `DEFAULT_MAX_STEPS`. Trace species well below 1e-20 can grow
by many orders of magnitude, as sulfur species do in lit gas below 800 K, so the error control must follow them; atol also sets
the guard's clamping allowance.

## Guard and recovery

After the call, a guard kernel checks every cell: the new state is finite, the call finished, and each element's share of the
atoms is unchanged, up to rounding (relative change at most `drift`) and what the clamping of accepted small negatives can
add. It counts the failures and lists the failed cells.

A recovery kernel recomputes the listed cells from their inputs, with the same rates, streams, stages and step control, but
with W formed densely and factored by LU with threshold partial pivoting. Rows exchange only where the diagonal is below
`PIVTAU` of the column's largest; plain partial pivoting loses the trace species' accuracy at the scale of atol. It has
`REC_STEPS` times the tries, applies the guard's test to its result, and counts the cell as recovered or still failed. A
still-failed cell returns done = 0.

Each cell's arithmetic depends only on its inputs, so results are bitwise reproducible on one GPU whatever the batch order.

### Refused cells

Each kernel screens its cells before stepping them. Bad input (a negative or non-finite x, $|\sum_i x_i - 1| >$ `X_SUM_TOL`,
no x above 0, a non-finite T, P <= 0, a negative or non-finite dt, a negative or non-finite J) and cells below `T_MIN` are not
stepped; the guard counts them and does not list them for recovery. The kernels' state is N, molecules per starting molecule,
so a sum of 0.9 would raise every density by 11 %: the call refuses it instead of renormalizing. `X_SUM_TOL` is 1e-10: every
stored input the checks feed sums to 1 within 1.3e-15, and the solver's outputs within rounding too. The guard sums x in another
order than the call's warps, so a sum within rounding of the tolerance may be judged differently by the two: a cell the call
refuses and the guard accepts ends recomputed and still failed, and one the call steps and the guard refuses is refused. After the kernels, the call reads the counts back, its one host sync, and raises if it refused a
cell.

### Why the element test works

$E^T W = E^T / (\gamma h)$ for the element matrix E, since $E^T S = 0$. So every stage moves the element budget by
$\gamma h E^T$ times the residual of its solve, and a factor without row exchanges that has lost accuracy shows up there.

### Element-share correction

A cell that passes has the rounding taken out: with $E_i$ the atoms of species i and $e_0$, $e_1$ the element totals of the
input and the result,

$$x_i' = x_i (1 + \lambda \cdot E_i), \qquad (E^T X E) \lambda = e_0 \frac{\sum e_1}{\sum e_0} - e_1,$$

then renormalized. This is the smallest relative change that restores every element's share. An element whose pivot vanishes
is left out, and no species changes by more than `GMAX`, so elements tied together or shares near atol are not fully restored.
The correction does not enter the step control or the derivative kernels' tangents and cotangents: conservation makes its
sensitivity zero, except near atol, where clamping breaks conservation.

### The drift allowance

The rounding drift of the element shares depends on the network's chemistry. A network takes
drift = max(`DRIFT`, 10 r), with r its rounding drift: the largest drift at which the guard still flags a cell over the parity
stress cases (recorded starts and their scrambles to 2 dex, at dt up to 500 s).

## Derivatives

`ADSolver` gives a tangent (JVP) and an adjoint (VJP) of one call. Each also returns the call's result, bitwise that of
`CudaSolver`.

Differentiated inputs: the state x, T, P, dt, the starting step size h, per-cell log rate multipliers lm [B, R], the nine Arrhenius
columns of the rate table rc [R, 9 or 14] (its tangent and cotangent are [R, 9]) and the photolysis rates J [B, branches]. A multiplier scales a reaction's forward and
reverse coefficients together, after the collision limit; a photolysis row has $k_f = J e^{lm}$. Derivatives in J are in J
itself, so they hold at J = 0. The Troe parameters and efficiencies are not differentiated.

The rate derivatives include the Troe factor and the collision limit. With A = ln of a pair's forward coefficient before the
limit and G = ln k_r - A, the limit c adds $d \ln c = -(w_A dA + w_G dG + w_E d\ln n_\mathrm{tot})$ to both directions.
$w_A$ is 1 above the limit and 1/2 at it; $w_G$ is $w_A$ times the reverse direction's share of the larger direction, 1/2 at a
tie.

Two contracts, both on the branch the call takes (accept or reject, clamped species, step-control limits):

- **Frozen** (`adapt=False`): the accepted step sizes are held fixed. No derivative through dt, h or the error estimate.
  Rejected attempts drop out, and h_out has no tangent.
- **Adaptive** (`adapt=True`): the step control is differentiated as well, through the step factor and the secant shrink, with
  rejected attempts included and the carried h in and out. err is the call's, differentiated through the norm it selects.

Ties of min, max, |.| and clamps follow torch's conventions: half to each side of a tie of min or max, a maximum over species
shared evenly among those attaining it, d|a| = 0 at 0, and a clamp passes at its bound.

In a dark cell, the adaptive J tangent at exactly J = 0 holds only for a tiny, state-dependent range of J: a species made only
by J joins the error control once it passes about atol / rtol. For a J derivative there, call at a small positive rate. The
checks use 1e-8 s^-1.

**Tangent.** For each attempt, the stage tangents $W \dot{K}_i = \dot{b}_i + \dot{h} K_i / (\gamma h^2) + \dot{J} K_i$ are solved
with the attempt's factors. $\dot{J} K_i$, the derivative of the Jacobian along the tangent applied to $K_i$, comes reaction by
reaction from hyper-dual products. The rate tangents come from the Arrhenius, falloff, three-body and NASA-9 derivatives.

**Adjoint.** The call runs forward once, storing the state before every attempt; beyond `CAP` attempts it keeps a checkpoint
every ceil(n / CAP) attempts and recomputes each segment. It then reverses the attempts, recomputing each one except the one
whose stages are still in place. Transposed solves are $W^{-T} y = M_0^{-T}(y + c_0 (u \cdot y) \mathbf{1})$ on the same LU.
The rate cotangents reduce to T, P and the per-cell log multipliers in the kernel, and to the Arrhenius table in a second kernel
in a fixed order. No atomics enter the arithmetic, only the cell queue, so the same inputs give the same bits on one GPU.

**Recovered cells.** After each kernel, the call's guard lists the cells `CudaSolver` would recompute. A second kernel redoes
each on the recovery's path and differentiates that path, with the dense pivoted factors and up to `2 CAP` stored attempts. A
result that passes the guard's test has flag 1. One that does not has flag 2, and its tangents and per-cell cotangents are NaN. The rate-table cotangent, a sum over the batch, leaves
such a cell out, and the `still_failed` count says how many were left out: it is then the rate table's cotangent of the other cells alone.

## References

- Jasper, A. W., Klippenstein, S. J., Harding, L. B., & Ruscic, B. 2007, J. Phys. Chem. A, 111, 3932,
  doi:10.1021/jp067585p.
- McBride, B. J., Zehe, M. J., & Gordon, S. 2002, NASA Glenn Coefficients for Calculating Thermodynamic Properties of
  Individual Species, NASA/TP-2002-211556.
- Sandu, A., Verwer, J. G., Blom, J. G., Spee, E. J., Carmichael, G. R., & Potra, F. A. 1997, Atmos. Environ., 31, 3459,
  doi:10.1016/S1352-2310(97)83212-8.
- Tsai, S.-M., Lyons, J. R., Grosheintz, L., Rimmer, P. B., Kitzmann, D., & Heng, K. 2017, ApJS, 228, 20,
  doi:10.3847/1538-4365/228/2/20.
- Tsai, S.-M., Malik, M., Kitzmann, D., Lyons, J. R., Fateev, A., Lee, E., & Heng, K. 2021, ApJ,
  doi:10.3847/1538-4357/ac29bc.
- Venot, O., Hébrard, E., Agúndez, M., et al. 2012, A&A, 546, A43, doi:10.1051/0004-6361/201219310.
- Visscher, C., & Moses, J. I. 2011, ApJ, 738, 72, doi:10.1088/0004-637X/738/1/72.
