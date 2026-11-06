"""The call against the stored reference: cold, scrambled and chained calls, thermal and lit.

The reference is the call computed by batched fp64 Rodas3 in torch, with the kernel's numerics, stored for exactly these
inputs (tests/libref.py refuses others). The limits and their reasons are in tests/helpers.py. On a sulfur network H2S,
SO2, COS and S2 are scored too. Run with ``pytest tests/test_parity.py -s`` to see each case's numbers.
"""

import time

import helpers
import numpy as np
import pytest
import torch

from maxichem import kincuda

pytestmark = pytest.mark.gpu

SHARE_DRIFT_LIMIT = 1e-10  # relative: an element's share over a chain; the guard's correction holds it to 2e-13


def test_cold_calls(parity, verdict):
    cells = 4096
    x0, T, P = parity.starts(cells, 0, 0.0)
    for dt in (1.0, 500.0):
        tally = helpers.Tally(parity.network)
        args = (x0, T, P, parity.full(cells, dt), parity.full(cells, 1e-10))
        tally.add(parity.network.entry(f"call/cold{dt:g}", *args), parity.solver(*args), parity.solver)
        tally.report(verdict, f"yardstick cold calls, dt {dt:g} s")


def test_scrambled_starts(parity, verdict):
    cells = 4096
    for sigma in (0.0, 1.0, 2.0):
        x0, T, P = parity.starts(cells, int(10 + sigma), sigma)
        for dt in (0.01, 1.0, 50.0, 500.0):
            tally = helpers.Tally(parity.network)
            args = (x0, T, P, parity.full(cells, dt), parity.full(cells, 1e-10))
            tally.add(parity.network.entry(f"call/scr{sigma:g}_{dt:g}", *args), parity.solver(*args), parity.solver)
            tally.report(verdict, f"scrambled starts sigma {sigma:g} dex, dt {dt:g} s")


def _run_chain(parity, solver, key, cells, calls, verdict, name, share_name, J=None, timed=False):
    """Chain calls from panel P's starts against the stored chain, and report parity and the element shares."""
    x, h, t_chain, p_chain, dt_chain = parity.chain_inputs(cells, calls)
    inputs = (x, h, t_chain, p_chain, dt_chain) if J is None else (x, h, t_chain, p_chain, dt_chain, J)
    stored = parity.network.entry(key, *inputs)
    tally, seconds = helpers.Tally(parity.network), 0.0
    share0, largest = parity.shares(x), 0.0
    keywords = {} if J is None else {"J": J}
    for k in range(calls):
        if timed:
            torch.cuda.synchronize()
        started = time.perf_counter()
        T, P = helpers.tensor(t_chain[k], parity.device), helpers.tensor(p_chain[k], parity.device)
        out = solver(x, T, P, helpers.tensor(dt_chain[k], parity.device), h, **keywords)
        if timed:
            torch.cuda.synchronize()
        seconds += time.perf_counter() - started
        tally.add({q: v[k] for q, v in stored.items()}, out, solver)
        x, h = out[:2]
        largest = max(largest, float(((parity.shares(x) - share0).abs() / share0.clamp_min(1e-300)).max()))
    timing = f" (the kernel on its own state; ms per call kernel {seconds / calls * 1e3:.2f}, GPU may be shared)"
    tally.report(verdict, name + (timing if timed else ""))
    verdict(share_name, largest < SHARE_DRIFT_LIMIT, f"largest relative change of a share over the chain {largest:.2e}")


def test_chain(parity, verdict):
    calls, cells = 2000, 1024
    name = f"panel P chain, {calls} calls x {cells} cells"
    _run_chain(parity, parity.solver, "chain", cells, calls, verdict, name, "chain: element shares", timed=True)


def test_lit_calls(parity, verdict):
    J = parity.J
    cells = J.shape[0]
    for name, seed, sigma in (("cold", 0, 0.0), ("scr1", 11, 1.0)):
        x0, T, P = parity.starts(cells, seed, sigma)
        for dt in (1.0, 500.0):
            tally = helpers.Tally(parity.network)
            args = (x0, T, P, parity.full(cells, dt), parity.full(cells, 1e-10))
            stored = parity.network.entry(f"photo/{name}{dt:g}", *args, J)
            tally.add(stored, parity.lit(*args, J=J), parity.lit)
            tally.report(verdict, f"photolysis, {name} starts, dt {dt:g} s")


def test_lit_with_no_light_is_the_thermal_call(parity, verdict):
    cells = parity.J.shape[0]
    x0, T, P = parity.starts(cells, 0, 0.0)
    args = (x0, T, P, parity.full(cells, 500.0), parity.full(cells, 1e-10))
    tally = helpers.Tally(parity.network)
    stored = parity.network.entry("call/cold500", *args)
    tally.add(stored, parity.lit(*args, J=torch.zeros_like(parity.J)), parity.lit)
    tally.report(verdict, "photolysis with J = 0 against the thermal reference, cold starts, dt 500 s")


def test_lit_forced_failure_is_recomputed(parity, verdict):
    cells = parity.J.shape[0]
    x0, T, P = parity.starts(cells, 0, 0.0)
    i = torch.tensor([7], device=parity.device)
    long_call, first_step = parity.full(1, 500.0), parity.full(1, 1e-10)
    y, _, d, n, _ = parity.lit.launch(x0[i], T[i], P[i], long_call, first_step, max_steps=10, J=parity.J[i])
    counts = dict(parity.lit.fallback)
    tries = 10 * kincuda.REC_STEPS
    stored = parity.network.entry(
        "photo/forced", x0[i], T[i], P[i], long_call, first_step, parity.J[i], np.array(tries)
    )
    e = float(parity.network.floored(y[:, parity.network.gas_index], stored["x"]).max())
    ok = (
        counts["unfinished"] == 1
        and counts["recomputed"] == 1
        and e <= helpers.TOL
        and bool(torch.equal(n, stored["steps"]))
        and bool(torch.equal(d, stored["done"]))
    )
    verdict(
        "photolysis: forced failure (max_steps 10) recomputed",
        ok,
        f"counts {counts}; {e:.1e} dex from the reference call with {tries} tries, finished {bool(d[0])}",
    )


def test_lit_chain(parity, verdict):
    cells, calls = 1024, 300
    name = f"photolysis: lit chain, {calls} calls x {cells} cells"
    # J: rows i % L of the layers, as the reference
    _run_chain(
        parity, parity.lit, "photo/chain", cells, calls, verdict, name, "lit chain: element shares", parity.J[:cells]
    )


def test_lit_bad_rates_are_counted(parity, verdict):
    cells = parity.J.shape[0]
    x0, T, P = parity.starts(cells, 0, 0.0)
    xs, p_some = x0[:64].clone(), P[:64]
    j_bad = parity.J[:64].clone()
    j_bad[3, 0], j_bad[5, 1], j_bad[9, 2] = -1.0, float("nan"), float("inf")
    t_bad = T[:64].clone()
    t_bad[11] = 250.0
    j_bad[11, 0] = float("nan")  # a cold cell with a bad J: counted cold
    y, _, d, _ = parity.lit(xs, t_bad, p_some, parity.full(64, 1.0), parity.full(64, 1e-10), J=j_bad)
    counts = dict(parity.lit.fallback)
    good = torch.ones(64, dtype=torch.bool, device=parity.device)
    good[[3, 5, 9, 11]] = False
    kept = int(good.sum())
    yr, _, dr, _ = parity.lit(
        xs[good], t_bad[good], p_some[good], parity.full(kept, 1.0), parity.full(kept, 1e-10), J=j_bad[good]
    )
    normalized = xs[[3, 5, 9]] / xs[[3, 5, 9]].sum(1, keepdim=True)
    ok = (
        counts["bad_input"] == 3
        and counts["cold"] == 1
        and not bool(d[[3, 5, 9, 11]].any())
        and bool(torch.allclose(y[[3, 5, 9]], normalized, rtol=1e-14, atol=0))
        and bool(torch.equal(y[good], yr))
        and bool(torch.equal(d[good], dr))
    )
    verdict(
        "photolysis: bad J counted, cold precedence",
        ok,
        f"counts {counts}; rows 3, 5, 9 (J -1, NaN, inf) not finished and returned as the normalized input, "
        f"row 11 (250 K, NaN J) cold; "
        f"the other 60 cells bitwise those of a batch without them",
    )
    with pytest.raises(ValueError, match="below"):  # the cold cell above
        parity.lit.check()
