"""The solver against answers known without the solver: first-order decay at constant pressure, and equilibrium."""

import helpers
import numpy as np
import pytest
import torch

from maxichem import kincuda, kincuda_ad, kinvulcan

pytestmark = pytest.mark.gpu

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


def _gibbs(net, name, T):
    """Return G / RT of a species of the invented data: constant Cp / R in both ranges."""
    cp, h, b2 = net.lo[net.species.index(name)][[2, 7, 8]]
    return cp + h / T - (cp * np.log(T) + b2)


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
    g = {q: _gibbs(net, q, T) for q in fractions}
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
    k3 = np.exp(-(g["H2"] - 2.0 * g["H"])) * P_EQUILIBRIUM / kinvulcan.P0
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
