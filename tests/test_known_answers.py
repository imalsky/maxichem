"""The solver against answers known without the solver: decays, equilibria, photolysis branches and step control."""

from fractions import Fraction
from pathlib import Path

import helpers
import numpy as np
import pytest
import torch

from maxichem import kincuda, kincuda_ad

pytestmark = pytest.mark.gpu

NASA9_DATA = Path(__file__).parent / "data" / "NASA9"

T_TEST = 1500.0  # K
P_TEST = 1e6  # dyn cm^-2
# dyn cm^-2: 10 bar, off the standard pressure, so the equilibrium of a reaction that changes the moles depends on P
P_EQUILIBRIUM = 1e7
J_TEST = 2.0  # s^-1
DECAY_TOLERANCE = 2e-5  # relative, at rtol 1e-6: the integrator's error; measured 7.9e-6
EQUILIBRIUM_TOLERANCE = 1e-12  # relative, on a mass-action ratio against the NASA-9 closed form; measured 1.3e-15
SHARE_TOLERANCE = 1e-14  # relative: the guard's correction restores each element's share; measured 1.3e-16
SMALL_TANGENT_TOLERANCE = 1e-4  # relative: central differences at dT = 1e-3 K carry about this much
SMALL_ADJOINT_TOLERANCE = 1e-6  # relative: only rounding separates the adjoint from the tangent (DOT_P99 of test_ad.py)


def test_photolysis_decay_at_constant_pressure(read_network, gpu):
    net = read_network("synthetic_decay")
    with kincuda.CudaSolver(net, gpu, photo=True) as solver:
        times = np.array([0.1, 0.5, 1.0, 2.0])
        cells = len(times)
        x0 = helpers.state(net, cells, gpu, N2_1=1.0)
        J = torch.full((cells, len(net.photo)), J_TEST, dtype=torch.float64, device=gpu)
        dt = torch.tensor(times, dtype=torch.float64, device=gpu)
        x, _, done, _ = solver(
            x0,
            helpers.full(cells, T_TEST, gpu),
            helpers.full(cells, P_TEST, gpu),
            dt,
            helpers.full(cells, 1e-6, gpu),
            rtol=1e-6,
            J=J,
        )
    assert bool(done.all())
    a_answer, b_answer = helpers.decay_answer(J_TEST * times)
    assert np.allclose(x[:, net.species.index("N2_1")].cpu().numpy(), a_answer, rtol=DECAY_TOLERANCE, atol=0.0)
    assert np.allclose(x[:, net.species.index("N_1")].cpu().numpy(), b_answer, rtol=DECAY_TOLERANCE, atol=0.0)


def _gibbs(name, T):
    """Return G / RT of a species of tests/data, from its NASA-9 file read here: the range below or above 1000 K."""
    numbers = np.loadtxt(NASA9_DATA / f"{name}.txt").flatten()
    first = 0 if T < NASA_SPLIT else 10  # each range: a1..a7, an unused entry, b1, b2
    return helpers.gibbs_rt(numbers[[first + i for i in (0, 1, 2, 3, 4, 5, 6, 8, 9)]], T)


@pytest.mark.parametrize("stem", ["synthetic_falloff", "synthetic_small"])
@pytest.mark.parametrize("T", [800.0, 1500.0, 2500.0])
def test_long_call_reaches_the_nasa9_equilibrium(read_network, gpu, T, stem):
    net = read_network(stem)
    with kincuda.CudaSolver(net, gpu) as solver:
        cells = 4
        x0 = helpers.state(net, cells, gpu, H=0.1, H2=0.6, OH=0.05, H2O=0.1, O=0.05, He=0.1)
        shares0 = (x0 @ torch.as_tensor(net.atoms, device=gpu)).sum(0)
        x, _, done, _ = solver(
            x0,
            helpers.full(cells, T, gpu),
            helpers.full(cells, P_EQUILIBRIUM, gpu),
            helpers.full(cells, 1e10, gpu),
            helpers.full(cells, 1e-10, gpu),
        )
    assert bool(done.all())
    fractions = {q: float(x[0, net.species.index(q)]) for q in ("H", "H2", "OH", "O", "H2O")}
    g = {q: _gibbs(q, T) for q in fractions}
    # H + OH = H2 + O and H + H2O = OH + H2: no change of moles, so the ratio of mole fractions is K
    k1 = np.exp(-(g["H2"] + g["O"] - g["H"] - g["OH"]))
    k2 = np.exp(-(g["OH"] + g["H2"] - g["H"] - g["H2O"]))
    assert fractions["H2"] * fractions["O"] / (fractions["H"] * fractions["OH"]) == pytest.approx(
        k1, rel=EQUILIBRIUM_TOLERANCE
    )
    assert fractions["OH"] * fractions["H2"] / (fractions["H"] * fractions["H2O"]) == pytest.approx(
        k2, rel=EQUILIBRIUM_TOLERANCE
    )
    # H + H + M = H2 + M: one mole fewer. K_p = (p_H2 / P0) / (p_H / P0)^2 and p_i = x_i P: x_H2 / x_H^2 = K_p P / P0
    k3 = np.exp(-(g["H2"] - 2.0 * g["H"])) * P_EQUILIBRIUM / helpers.STANDARD_PRESSURE
    assert fractions["H2"] / fractions["H"] ** 2 == pytest.approx(k3, rel=EQUILIBRIUM_TOLERANCE)
    shares = (x @ torch.as_tensor(net.atoms, device=gpu)).sum(0)
    assert (
        float(
            (
                (shares / shares.sum() - shares0 / shares0.sum()).abs() / (shares0 / shares0.sum()).clamp_min(1e-300)
            ).max()
        )
        < SHARE_TOLERANCE
    )


def test_small_network_runs_in_every_kernel(read_network, gpu):
    net = read_network("synthetic_small")
    cells = 4
    x0 = helpers.state(net, cells, gpu, H=0.1, H2=0.6, OH=0.05, H2O=0.1, O=0.05, He=0.1)
    T, P = helpers.full(cells, 1500.0, gpu), helpers.full(cells, P_TEST, gpu)
    dt, h = helpers.full(cells, 1e-3, gpu), helpers.full(cells, 5e-4, gpu)
    J = torch.full((cells, len(net.photo)), 1e-4, dtype=torch.float64, device=gpu)
    zero = torch.zeros_like(T)
    with kincuda.CudaSolver(net, gpu, photo=True) as solver:
        forced = solver.launch(x0, T, P, dt, h, J=J, max_steps=10)  # unfinished: the recovery kernel runs
        counts = dict(solver.fallback)
        called = solver.launch(x0, T, P, dt, h, J=J)
    assert counts["recomputed"] == cells
    assert bool(forced[2].all())
    assert torch.allclose(forced[0], called[0], rtol=1e-6)  # dense against sparse factors of the same step
    with kincuda_ad.ADSolver(net, gpu, photo=True) as ad:
        step = 1e-3
        tangent = ad.jvp(x0, T, P, dt, h, torch.zeros_like(x0), torch.ones_like(T), zero, zero, zero, J=J)
        up = ad.jvp(x0, T + step, P, dt, h, torch.zeros_like(x0), zero, zero, zero, zero, J=J)[0]
        down = ad.jvp(x0, T - step, P, dt, h, torch.zeros_like(x0), zero, zero, zero, zero, J=J)[0]
        # a weight per species: the plain sum of x is 1 whatever T is, so its derivative says nothing
        weights = torch.arange(1.0, len(net.species) + 1.0, dtype=torch.float64, device=gpu)
        adjoint = ad.vjp(x0, T, P, dt, h, weights.expand_as(x0), J=J)
    for a, b in zip(called[:4], tangent[:4], strict=True):
        assert torch.equal(a, b)
    for a, b in zip(called[:4], adjoint[:4], strict=True):
        assert torch.equal(a, b)
    difference = (up - down) / (2.0 * step)
    assert torch.allclose(tangent[4], difference, rtol=SMALL_TANGENT_TOLERANCE, atol=1e-9)
    contraction = (weights * tangent[4]).sum(1)
    print(f"small network: adjoint {adjoint[4]['T'].tolist()}, tangent {contraction.tolist()}")
    assert bool((contraction != 0.0).all())
    assert torch.allclose(adjoint[4]["T"], contraction, rtol=SMALL_ADJOINT_TOLERANCE, atol=0.0)


# CO_1's NASA-9 ranges: every term of Cp / R depends on T, with other coefficients below and above the split
NASA_SPLIT = 1000.0  # K: where the ranges meet (McBride et al. 2002)
LOW_RANGE = [5.0e4, -200.0, 3.2, 1.2e-3, -4.0e-7, 6.0e-11, -3.0e-15]
HIGH_RANGE = [-2.0e5, 300.0, 4.5, 2.0e-4, -3.0e-8, 2.0e-12, -1.0e-16]
# K: below the split, above it but below 1200 K, and far above
NASA_POINTS = [900.0, 1100.0, 2500.0]


def _meeting_ranges() -> tuple[list[float], list[float]]:
    """Return LOW_RANGE and HIGH_RANGE as a1..a7, b1, b2, with the high b1 and b2 set so H and S meet at the split."""
    low = [*LOW_RANGE, 0.0, 0.0]
    h_low, s_low = helpers.enthalpy_entropy(low, NASA_SPLIT)
    h_high, s_high = helpers.enthalpy_entropy([*HIGH_RANGE, 0.0, 0.0], NASA_SPLIT)
    return low, [*HIGH_RANGE, NASA_SPLIT * (h_low - h_high), s_low - s_high]


@pytest.mark.parametrize("T", NASA_POINTS)
def test_equilibrium_follows_each_nasa9_range(gpu, tmp_path, T):
    low, high = _meeting_ranges()
    text = "# Invented test network: two isomers\n\n# Two-body Reactions\n1 [ CO_1 + He -> CO_2 + He ] 1e-11 0 0\n"
    thermo = {"CO_1": (low, high), "CO_2": helpers.constant_cp(-500.0), "He": helpers.constant_cp(0.0, 2.5)}
    net = helpers.write_network(tmp_path, text, thermo)
    with kincuda.CudaSolver(net, gpu) as solver:
        x0 = helpers.state(net, 1, gpu, CO_1=0.3, He=0.7)
        one = helpers.full(1, 1.0, gpu)
        x, _, done, _ = solver(x0, T * one, P_TEST * one, 1e10 * one, 1e-10 * one)
    assert bool(done.all())
    a = low if T < NASA_SPLIT else high
    K = np.exp(helpers.gibbs_rt(a, T) - helpers.gibbs_rt(thermo["CO_2"], T))
    ratio = float(x[0, net.species.index("CO_2")] / x[0, net.species.index("CO_1")])
    print(f"CO_2 / CO_1 at {T:g} K: {ratio:.15e}, K {K:.15e}")
    assert ratio == pytest.approx(K, rel=EQUILIBRIUM_TOLERANCE)


PHOTOLYSIS_NETWORK = """# Invented test network: three absorbers, one with two branches; thermal rows of zero rate

# Two-body Reactions
1 [ N2_1 + N_1 -> N_1 + N2_1 ] 0.0 0.0 0.0
3 [ N2_2 + N_2 -> N_2 + N2_2 ] 0.0 0.0 0.0
5 [ N2_3 + N_3 -> N_3 + N2_3 ] 0.0 0.0 0.0
7 [ N2_3 + N_4 -> N_4 + N2_3 ] 0.0 0.0 0.0

# reverse stops
# photo disscoiation (no reversals)
9 [ N2_1 -> N_1 + N_1 ] N2_1 1
11 [ N2_2 -> N_2 + N_2 ] N2_2 1
13 [ N2_3 -> N_3 + N_3 ] N2_3 1
15 [ N2_3 -> N_4 + N_4 ] N2_3 2
"""
# s^-1, by (absorber, branch) and product: no two equal, so a branch given another branch's J changes the answer
BRANCHES = {("N2_1", 1): (0.7, "N_1"), ("N2_2", 1): (2.3, "N_2"), ("N2_3", 1): (0.4, "N_3"), ("N2_3", 2): (1.1, "N_4")}
ABSORBED = {"N2_1": 0.3, "N2_2": 0.3, "N2_3": 0.4}  # the start
PHOTOLYSIS_RTOL = 1e-8
PHOTOLYSIS_TOLERANCE = 1e-6  # relative: the integrator's error at PHOTOLYSIS_RTOL


def _photolysis_answer(t: float) -> dict[str, float]:
    """Return x at t: each absorber decays at the sum of its J, each branch's product grows at 2 J N."""
    n = {}
    for absorber, n0 in ABSORBED.items():
        total = sum(j for (a, _), (j, _) in BRANCHES.items() if a == absorber)
        n[absorber] = n0 * np.exp(-total * t)
        for (a, _), (j, product) in BRANCHES.items():
            if a == absorber:
                n[product] = 2.0 * n0 * j / total * (1.0 - np.exp(-total * t))
    s = sum(n.values())
    return {q: v / s for q, v in n.items()}


def test_each_photolysis_branch_takes_its_own_rate(gpu, tmp_path):
    thermo = {q: helpers.constant_cp(0.0) for q in ("N2_1", "N_1", "N2_2", "N_2", "N2_3", "N_3", "N_4", "He")}
    net = helpers.write_network(tmp_path, PHOTOLYSIS_NETWORK, thermo)
    times = [0.3, 1.0, 3.0]  # s
    cells = len(times)
    J = helpers.tensor([[BRANCHES[(r["absorber"], r["branch"])][0] for r in net.photo]] * cells, gpu)
    with kincuda.CudaSolver(net, gpu, photo=True) as solver:
        x0 = helpers.state(net, cells, gpu, **ABSORBED)
        dt = helpers.tensor(times, gpu)
        x, _, done, _ = solver(
            x0, helpers.full(cells, T_TEST, gpu), helpers.full(cells, P_TEST, gpu), dt, 1e-6 * dt, PHOTOLYSIS_RTOL, J=J
        )
    assert bool(done.all())
    for c, t in enumerate(times):
        for q, want in _photolysis_answer(t).items():
            got = float(x[c, net.species.index(q)])
            print(f"t {t:g} s, {q}: {got:.12e} against {want:.12e}")
            assert got == pytest.approx(want, rel=PHOTOLYSIS_TOLERANCE)


DECAYS = 5.0  # J dt: a first try of h = dt is far too long at rtol 1e-6, so it is rejected
FIRST_TRY_RTOL = 1e-6
MORE_TRIES = 50  # a bound on the rejections searched for


def test_a_rejected_first_try_is_retried(read_network, gpu):
    net = read_network("synthetic_decay")
    J = torch.full((1, len(net.photo)), J_TEST, dtype=torch.float64, device=gpu)
    x0 = helpers.state(net, 1, gpu, N2_1=1.0)
    T, P, dt = helpers.full(1, T_TEST, gpu), helpers.full(1, P_TEST, gpu), helpers.full(1, DECAYS / J_TEST, gpu)
    with kincuda.CudaSolver(net, gpu, photo=True) as solver:
        x, _, done, accepted = solver(x0, T, P, dt, dt.clone(), rtol=FIRST_TRY_RTOL, J=J)
        steps = int(accepted[0])
        # the call's own counts: it finishes within max_steps tries only if no try was rejected
        unfinished = []
        for tries in range(steps, steps + MORE_TRIES):
            solver.launch(x0, T, P, dt, dt.clone(), rtol=FIRST_TRY_RTOL, max_steps=tries, J=J)
            unfinished.append(solver.fallback["unfinished"])
            if not unfinished[-1]:
                break
    print(f"{steps} accepted steps; unfinished at max_steps {steps} and up: {unfinished}")
    assert bool(done.all())
    a_answer, b_answer = helpers.decay_answer(DECAYS)
    assert float(x[0, net.species.index("N2_1")]) == pytest.approx(a_answer, rel=DECAY_TOLERANCE)
    assert float(x[0, net.species.index("N_1")]) == pytest.approx(b_answer, rel=DECAY_TOLERANCE)
    assert unfinished[0] == 1  # at max_steps = the accepted steps: at least one try was rejected
    assert unfinished[-1] == 0


# The test above proves a rejection, but its first trial is negative (N_A = -712/7203 at J dt = 5), so positivity
# alone rejects it. Here the first trial is positive and only its error estimate rejects it.
FIRST_TRY_DECAYS = Fraction(1, 10)  # J dt
ACCEPT_ANY_ERROR = 1e6  # rtol: a positive trial passes


def _first_trial(z: Fraction) -> tuple[list[Fraction], list[Fraction]]:
    """Return N' and K4 of one Rodas3 step (docs/method.md) of A -> 2 B from N = (1, 0) at J h = z, exactly.

    Units of h = 1, so J = z. W = I / (gamma h) - J_N with J_N = [[-J, 0], [2 J, 0]], lower triangular.
    """
    w = (1 / (Fraction(1, 2)) + z, -2 * z, 1 / Fraction(1, 2))  # W_AA, W_BA, W_BB

    def solve(b: list[Fraction]) -> list[Fraction]:
        a = b[0] / w[0]
        return [a, (b[1] - w[1] * a) / w[2]]

    def f(y: list[Fraction]) -> list[Fraction]:
        return [-z * y[0], 2 * z * y[0]]

    n = [Fraction(1), Fraction(0)]
    k1 = solve(f(n))
    k2 = solve([f(n)[i] + 4 * k1[i] for i in range(2)])
    y3 = [n[i] + 2 * k1[i] for i in range(2)]
    k3 = solve([f(y3)[i] + k1[i] - k2[i] for i in range(2)])
    y4 = [y3[i] + k3[i] for i in range(2)]
    k4 = solve([f(y4)[i] + k1[i] - k2[i] - Fraction(8, 3) * k3[i] for i in range(2)])
    return [y4[i] + k4[i] for i in range(2)], k4


@pytest.mark.parametrize("norm", ["root mean square", "largest"])
def test_a_positive_first_try_is_rejected_for_its_error(read_network, gpu, norm):
    # err_i = K4_i / (atol + rtol max(|N_i|, |N'_i|)); a first try takes the larger of the RMS over all species and the
    # largest |err_i| (docs/method.md). rtol puts the RMS above 1, or the largest alone above 1.
    net = read_network("synthetic_decay")
    new, k4 = _first_trial(FIRST_TRY_DECAYS)
    assert min(new) > 0
    scaled = np.array([float(abs(k4[0])), float(abs(k4[1]) / new[1])])  # err_i times rtol, atol negligible
    rms, largest = np.sqrt((scaled**2).sum() / len(net.species)), scaled.max()
    rtol = rms / 10.0 if norm == "root mean square" else float(np.sqrt(rms * largest))
    print(f"{norm}: rtol {rtol:.3e}, err RMS {rms / rtol:.3f}, largest {largest / rtol:.3f}")
    J = torch.full((1, len(net.photo)), J_TEST, dtype=torch.float64, device=gpu)
    x0 = helpers.state(net, 1, gpu, N2_1=1.0)
    T, P = helpers.full(1, T_TEST, gpu), helpers.full(1, P_TEST, gpu)
    dt = helpers.full(1, float(FIRST_TRY_DECAYS) / J_TEST, gpu)
    with kincuda.CudaSolver(net, gpu, photo=True) as solver:
        solver.launch(x0, T, P, dt, dt.clone(), rtol=ACCEPT_ANY_ERROR, max_steps=1, J=J)
        positive = solver.fallback["unfinished"]
        solver.launch(x0, T, P, dt, dt.clone(), rtol=rtol, max_steps=1, J=J)
        rejected = solver.fallback["unfinished"]
    assert positive == 0  # the one try passes when no error is too large: it is not refused for a negative state
    assert rejected == 1


RECOMBINATION_RTOL = 1e-8
RECOMBINATION_TOLERANCE = 1e-6  # relative: the integrator's error at RECOMBINATION_RTOL
REVERSE_NEGLIGIBLE = 1e-12  # the dissociation against the recombination, at the end of each call
RECOMBINATION_T = helpers.RECOMBINATION_T


def test_recombination_at_constant_pressure(gpu, tmp_path):
    net, thermo = helpers.recombination(tmp_path)
    left = np.array([0.8, 0.5, 0.3])  # N_H at the end of each call
    t = helpers.recombination_time(left, P_TEST)
    x_h = 2.0 * left / (1.0 + left)
    dg = helpers.gibbs_rt(thermo["H2"], RECOMBINATION_T) - 2.0 * helpers.gibbs_rt(thermo["H"], RECOMBINATION_T)
    # v_r / v_f = K_p^-1 (P0 / P) x_H2 / x_H^2
    assert (np.exp(dg) * helpers.STANDARD_PRESSURE / P_TEST * (1.0 - x_h) / x_h**2).max() < REVERSE_NEGLIGIBLE
    cells = len(left)
    with kincuda.CudaSolver(net, gpu) as solver:
        dt = helpers.tensor(t, gpu)
        x, _, done, _ = solver(
            helpers.state(net, cells, gpu, H=1.0),
            helpers.full(cells, RECOMBINATION_T, gpu),
            helpers.full(cells, P_TEST, gpu),
            dt,
            1e-3 * dt,
            rtol=RECOMBINATION_RTOL,
        )
    assert bool(done.all())
    got = x[:, net.species.index("H")].cpu().numpy()
    print(f"x_H {got.tolist()} against {x_h.tolist()}")
    assert np.allclose(got, x_h, rtol=RECOMBINATION_TOLERANCE, atol=0.0)
