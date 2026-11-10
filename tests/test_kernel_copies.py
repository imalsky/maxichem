"""The per-cell setup and step-control tail, written four times (call, recover, jvp, vjp), on the edge inputs.

The cells hold a non-finite or too long starting step, dt = 0, refused cells beside good ones and a first try too long
to pass twice running. Each kernel must return the same bits for the same cell: x, h, done, steps, status and flag.
"""

import helpers
import pytest
import torch

from maxichem import _constants, kincuda, kincuda_ad

pytestmark = pytest.mark.gpu

T_GOOD, P_GOOD, DT, J_GOOD = 1500.0, 1e6, 1e-3, 1e-4  # K, dyn cm^-2, s, s^-1
FEW_TRIES = 10  # max_steps that leaves the good cells unfinished, so the recovery's copies run (and finish)
RECOVERED_TOLERANCE = 1e-6  # relative: a recovered cell against the ordinary call, rounding of two factorizations
RECOVERED_FLOOR = 1e-20  # mole fraction: below it the two are not compared
STATUS = {"good": 0, "cold": 1, "bad input": 2, "hot": 3}  # the kernels' screening results (common.cuh)
# Each cell's (T, dt, h) and what it is; the first is the reference for those that differ only in h
CELLS = {
    "h = dt": (T_GOOD, DT, DT),
    "h = NaN": (T_GOOD, DT, float("nan")),
    "h = +inf": (T_GOOD, DT, float("inf")),
    "h = 10 dt": (T_GOOD, DT, 10.0 * DT),
    "small h": (T_GOOD, DT, 1e-6 * DT),
    "dt = 0": (T_GOOD, 0.0, DT),
    "cold": (250.0, DT, DT),
    "beside the cold cell": (T_GOOD, DT, 1e-6 * DT),
    "bad input": (T_GOOD, DT, DT),
    "hot": (3500.0, DT, DT),
    "first try rejected twice": (T_GOOD, 1e3, 1e3),
}
SAME_AS_H_DT = ("h = NaN", "h = +inf", "h = 10 dt")


def _inputs(net, device):
    """Return x, T, P, dt, h and J of the cells, in CELLS' order."""
    names = list(CELLS)
    cells = len(names)
    x = helpers.state(net, cells, device, H=0.1, H2=0.6, OH=0.05, H2O=0.1, O=0.05, He=0.1)
    x[names.index("bad input"), net.species.index("O")] = -1e-3
    T, dt, h = (helpers.tensor([CELLS[q][i] for q in names], device) for i in range(3))
    J = torch.full((cells, len(net.photo)), J_GOOD, dtype=torch.float64, device=device)
    return x, T, helpers.full(cells, P_GOOD, device), dt, h, J


def _kept(solver, monkeypatch):
    """Switch off the solver's raise on refused cells, keeping the status it would have read; return that list."""
    statuses = []
    monkeypatch.setattr(solver, "_refuse", lambda *a: statuses.append(a[-1].clone()))
    return statuses


def _results(read_network, gpu, monkeypatch, max_steps):
    """Return, per kernel, (x, h, done, steps, status, flag) and its counts for the cells."""
    net = read_network("synthetic_small")
    x, T, P, dt, h, J = _inputs(net, gpu)
    out = {}
    with kincuda.CudaSolver(net, gpu, photo=True) as solver:
        statuses = _kept(solver, monkeypatch)
        *primal, flag = solver.launch(x, T, P, dt, h, max_steps=max_steps, J=J)
        out["call"] = ([*primal, statuses[-1], flag], dict(solver.fallback))
    with kincuda_ad.ADSolver(net, gpu, photo=True) as ad:
        statuses = _kept(ad, monkeypatch)
        zero = torch.zeros_like(T)
        tangent = ad.jvp(x, T, P, dt, h, torch.zeros_like(x), zero, zero, zero, zero, max_steps=max_steps, J=J)
        out["jvp"] = ([*tangent[:4], statuses[-1], torch.clone(ad.flag)], dict(ad.fallback))
        adjoint = ad.vjp(x, T, P, dt, h, torch.ones_like(x), max_steps=max_steps, J=J)
        out["vjp"] = ([*adjoint[:4], statuses[-1], torch.clone(ad.flag)], dict(ad.fallback))
    return out


@pytest.mark.parametrize("max_steps", [_constants.DEFAULT_MAX_STEPS, FEW_TRIES])  # FEW_TRIES: the recovery's copies
def test_every_copy_returns_the_same_cells(read_network, gpu, monkeypatch, max_steps):
    out = _results(read_network, gpu, monkeypatch, max_steps)
    names = list(CELLS)
    fields = ("x", "h", "done", "steps", "status", "flag")
    call, counts = out["call"]
    print(f"max_steps {max_steps}: call counts {counts}")
    for kernel in ("jvp", "vjp"):
        values, kernel_counts = out[kernel]
        for name, a, b in zip(fields, call, values, strict=True):
            differ = [names[c] for c in range(len(names)) if not torch.equal(a[c], b[c])]
            assert not differ, f"{kernel}'s {name} differs from the call's in cells {differ}"
        assert kernel_counts == counts
    x, h, done, steps, status, flag = call
    print("flags", dict(zip(names, flag.tolist(), strict=True)), "steps", dict(zip(names, steps.tolist(), strict=True)))
    reference = names.index("h = dt")
    for name in SAME_AS_H_DT:  # a non-finite h acts as dt; the first try is at most dt
        c = names.index(name)
        assert all(torch.equal(q[c], q[reference]) for q in (x, h, done, steps, flag)), name
    assert torch.equal(x[names.index("beside the cold cell")], x[names.index("small h")])
    for name in ("cold", "bad input", "hot"):  # refused: not stepped, not done
        c = names.index(name)
        assert (int(status[c]), bool(done[c]), float(steps[c])) == (STATUS[name], False, 0.0), name
    c = names.index("dt = 0")  # not refused, done without a step
    assert (int(status[c]), bool(done[c]), float(steps[c])) == (STATUS["good"], True, 0.0)
    assert counts["cold"] == counts["hot"] == counts["bad_input"] == 1
    assert counts["nonfinite_h"] == 2
    stepped = [c for c, q in enumerate(names) if q not in ("cold", "bad input", "hot", "dt = 0")]
    assert bool(done[stepped].all())  # every stepped cell finishes, through the recovery at FEW_TRIES
    if max_steps == FEW_TRIES:
        assert counts["recomputed"] == counts["recovered"] == len(stepped)  # each ran out of tries, then recovered
        net = read_network("synthetic_small")
        *inputs, J = _inputs(net, gpu)
        with kincuda.CudaSolver(net, gpu, photo=True) as solver:
            _kept(solver, monkeypatch)
            trusted = solver.launch(*inputs, J=J)[0]  # the ordinary call with its default tries
        # the recovery's dense factors against the call's sparse ones: rounding apart
        assert torch.allclose(x[stepped], trusted[stepped], rtol=RECOVERED_TOLERANCE, atol=RECOVERED_FLOOR)
