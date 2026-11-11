"""The guard and the recovery, input checks, and the independence of results from the batch."""

import ctypes
import json
from pathlib import Path

import helpers
import libref
import numpy as np
import pytest
import torch

from maxichem import _cuda, kincuda, kincuda_ad, kinvulcan, paths

pytestmark = pytest.mark.gpu
CLOSED_SOLVER_BYTES = 4096  # the counts stay readable after close: two small tensors, in allocator blocks


def _settled(parity, x0, T, P):
    """Return the reference's settled states (100 s) and step sizes for these cells."""
    network = parity.network
    settled = libref.entry("guard", network.signature, x0, T, P, path=network.reference)
    return helpers.tensor(settled["xs"], parity.device), helpers.tensor(settled["hs"], parity.device)


def test_nan_input_is_refused(parity, verdict, monkeypatch):
    cells = 64
    network = parity.network
    x0, T, P = parity.starts(cells, 1, 0.0)
    xs, hs = _settled(parity, x0, T, P)
    keep = torch.arange(cells, device=parity.device) != 3
    x = xs.clone()
    x[3, 5] = float("nan")  # a NaN sum
    with pytest.raises(ValueError, match=r"1 cells of bad input: .*\(first: cell 3: sum\(x\) - 1 = nan"):
        parity.solver(x, T, P, parity.full(cells, 1.0), hs)
    with monkeypatch.context() as m:  # what the kernels returned: the other cells untouched
        m.setattr(parity.solver, "_refuse", lambda *_: None)
        y, _, d, _ = parity.solver(x, T, P, parity.full(cells, 1.0), hs)
    counts = dict(parity.solver.fallback)
    stored = network.stored_call(libref.arrays("guard/nan", path=network.reference))["x"][keep]
    e = float(network.floored(y[keep][:, network.gas_index], stored).max())
    ok = (
        counts["bad_input"] == 1
        and counts["recomputed"] == 0
        and not bool(torch.isfinite(y[3]).all())
        and not bool(d[3])
        and e <= helpers.TOL
        and bool(d[keep].all())
    )
    verdict(
        "guard: NaN input",
        ok,
        f"counts {counts}; row 3 non-finite and unfinished, the other {cells - 1} cells within {e:.1e} dex of the "
        "reference",
    )


def test_forced_failure_is_recomputed(parity, verdict):
    cells = 64
    network = parity.network
    x0, T, P = parity.starts(cells, 1, 0.0)
    xs, hs = _settled(parity, x0, T, P)
    x, h, dt = xs.clone(), hs.clone(), parity.full(cells, 1.0)
    x[7], h[7], dt[7] = x0[7], 1e-10, 500.0  # cell 7: cold start at 500 s
    y, _, d, n, flag = parity.solver.launch(x, T, P, dt, h, max_steps=10)
    counts = dict(parity.solver.fallback)
    i = torch.tensor([7], device=parity.device)
    tries = 10 * kincuda.REC_STEPS
    stored = network.entry("guard/forced", x[i], T[i], P[i], dt[i], h[i], np.array(tries))
    e = float(network.floored(y[i][:, network.gas_index], stored["x"]).max())
    ok = (
        counts["unfinished"] == 1
        and counts["recomputed"] == 1
        and counts["recovered"] + counts["still_failed"] == 1
        and e <= helpers.TOL
        and bool(torch.equal(n[i], stored["steps"]))
        and bool(torch.equal(d[i], stored["done"]))
        and int(flag[7]) == (1 if bool(stored["done"]) else 2)
        and bool(d[torch.arange(cells, device=parity.device) != 7].all())
    )
    verdict(
        "guard: forced failure (max_steps 10)",
        ok,
        f"counts {counts}; cell 7 recomputed with {tries} tries ({e:.1e} dex from the reference call "
        f"with those tries, {int(stored['steps'])} accepted steps, finished {bool(stored['done'])}, "
        f"flag {int(flag[7])})",
    )


def test_element_drift_is_recomputed(parity, verdict):
    # With allowance 0 every rounding change of an element share is flagged. atol 1e-18: the default's clamping
    # allowance covers rounding drift.
    cells = 4096
    network = parity.network
    x0, T, P = parity.starts(cells, 2, 1.0)
    args = (x0, T, P, parity.full(cells, 500.0), parity.full(cells, 1e-10))
    with kincuda.CudaSolver(network.net, parity.device, drift=0.0) as strict:
        y, _, dz, _, flag = strict.launch(*args, atol=1e-18)
        counts = dict(strict.fallback)
    yk, _, _, _, flag0 = parity.solver.launch(*args, atol=1e-18)  # the same call, nothing flagged
    i, passed = (flag > 0).nonzero().squeeze(1), flag == 0
    stored = network.entry("guard/drift", *args, np.array(4000 * kincuda.REC_STEPS))["x"][i]
    e = float(network.floored(y[i][:, network.gas_index], stored).max())
    ok = (
        counts["drift"] > 0
        and counts["recomputed"] == counts["drift"]
        and counts["recovered"] + counts["still_failed"] == counts["drift"]
        and e <= helpers.TOL
        and int(flag0.sum()) == 0
        and bool(torch.equal(y[passed], yk[passed]))
        and not bool(dz[flag == 2].any())
    )
    verdict(
        "guard: element drift (allowance 0)",
        ok,
        f"counts {counts}; the flagged cells recomputed on the device ({e:.1e} dex from the reference call; still "
        f"failing an allowance of 0 is expected, and those return done = 0), the others bitwise the kernel's result",
    )


def test_temperature_out_of_range_is_refused(parity, verdict, monkeypatch):
    cells = 4096
    x0, T, P = parity.starts(cells, 2, 1.0)
    t_out = T.clone()
    t_out[0], t_out[1] = 250.0, kincuda.T_MAX + 0.5  # refused
    t_out[2], t_out[3] = kincuda.T_MIN, kincuda.T_MAX  # the range's ends: accepted
    args = (x0, t_out, P, parity.full(cells, 500.0), parity.full(cells, 1e-10))
    refused = r"1 cells below 300 K, .*\(first: cell 0, T = 250 K\).*; 1 cells above 3000 K, .*cell 1, T = 3000.5 K\)"
    with pytest.raises(ValueError, match=refused) as raised:
        parity.solver(*args)
    with pytest.raises(ValueError, match="1 cells below 300 K and 1 above 3000 K"):
        parity.solver.check()
    with monkeypatch.context() as m:
        m.setattr(parity.solver, "_refuse", lambda *_: None)
        _, _, d, _ = parity.solver(*args)
    counts = dict(parity.solver.fallback)
    verdict(
        "guard: T < T_MIN and T > T_MAX refused, check() raises; T = T_MIN and T = T_MAX accepted",
        counts["cold"] == 1 and counts["hot"] == 1 and not bool(d[:2].any()) and bool(d[2:4].all()),
        f"counts {counts}; cells 0 and 1 not stepped, cells 2 and 3 done {d[2:4].tolist()}; ValueError: {raised.value}",
    )


def test_bad_inputs_are_refused(parity, verdict, monkeypatch):
    cells = 64
    x0, T, P = parity.starts(cells, 1, 0.0)
    xb, p_bad, dtb, hb = x0.clone(), P.clone(), parity.full(cells, 50.0), parity.full(cells, 1e-10)
    xb[3] = 0.0
    xb[7, 0] = -1e-3
    p_bad[5] = 0.0
    dtb[9] = -1.0
    xb[11] *= 0.9  # sum 0.9
    xb[13] *= 1.0 + 0.9 * kincuda.X_SUM_TOL  # just inside the tolerance on either side: accepted
    xb[15] *= 1.0 - 0.9 * kincuda.X_SUM_TOL
    bad = [3, 5, 7, 9, 11]
    with pytest.raises(ValueError, match=r"5 cells of bad input: .*\(first: cell 3: sum\(x\) - 1 = -1 "):
        parity.solver(xb, T, p_bad, dtb, hb)
    with pytest.raises(ValueError, match=r"1 cells of bad input: .*\(first: cell 0: sum\(x\) - 1 = -0.1 "):
        parity.solver(xb[11:12], T[11:12], p_bad[11:12], dtb[11:12], hb[11:12])
    with monkeypatch.context() as m:
        m.setattr(parity.solver, "_refuse", lambda *_: None)
        y, _, d, _ = parity.solver(xb, T, p_bad, dtb, hb)
    counts = dict(parity.solver.fallback)
    good = torch.ones(cells, dtype=torch.bool, device=parity.device)
    good[bad] = False
    yr, _, dr, _ = parity.solver(xb[good], T[good], p_bad[good], dtb[good], hb[good])
    ok = (
        counts["bad_input"] == len(bad)
        and counts["nonfinite"] == 0
        and counts["recomputed"] == 0
        and not bool(d[bad].any())
        and bool(d[[13, 15]].all())
        and bool(torch.equal(y[good], yr))
        and bool(torch.equal(d[good], dr))
    )
    verdict(
        "guard: all-zero x row, negative x, P = 0, dt < 0, sum(x) 0.9 refused; sum(x) 1 +- 0.9 X_SUM_TOL accepted",
        ok,
        f"counts {counts}; rows 3, 5, 7, 9, 11 counted as bad input and not done, rows 13 and 15 done; the other "
        f"{cells - len(bad)} cells bitwise those of a batch without the refused ones",
    )


def test_just_outside_the_range_is_refused_and_named(parity, verdict, monkeypatch):
    # T one float64 step outside each end, with dt = 0 and dt > 0: every such cell is refused, named and not done.
    x0, _, P = parity.starts(4, 2, 1.0)
    cold, hot = np.nextafter(kincuda.T_MIN, 0.0), np.nextafter(kincuda.T_MAX, np.inf)
    t_out = helpers.tensor([cold, hot, cold, hot], parity.device)
    dt, h = helpers.tensor([0.0, 0.0, 500.0, 500.0], parity.device), parity.full(4, 1e-10)
    refused = r"2 cells below 300 K, .*\(first: cell 0, T = 300 K\).*; 2 cells above 3000 K, .*\(first: cell 1, "
    with pytest.raises(ValueError, match=refused) as raised:
        parity.solver(x0, t_out, P, dt, h)
    with monkeypatch.context() as m:
        m.setattr(parity.solver, "_refuse", lambda *_: None)
        _, _, done, _ = parity.solver(x0, t_out, P, dt, h)
    counts = dict(parity.solver.fallback)
    verdict(
        "guard: T one step outside each end, dt 0 and 500 s: refused, named, not done",
        counts["cold"] == 2 and counts["hot"] == 2 and not bool(done.any()),
        f"counts {counts}; done {done.tolist()}; ValueError: {raised.value}",
    )


def test_sum_on_the_tolerance_is_judged_once(parity, verdict):
    # The call's warp and the guard add x in different orders. A sum just above X_SUM_TOL in the warp's order and just
    # below it in the network's order is refused; the opposite cell is accepted and stepped. Each is judged once.
    cells = 16
    x0, T, P = parity.starts(cells, 3, 0.0)
    perm = [int(q) for q in parity.solver._tables["perm"].tolist()]
    rows = x0.tolist()
    above = helpers.on_the_tolerance(rows, perm, kincuda.X_SUM_TOL, warp_above=True)
    below = helpers.on_the_tolerance(rows, perm, kincuda.X_SUM_TOL, warp_above=False)
    dt, h = parity.full(1, 50.0), parity.full(1, 1e-10)
    with pytest.raises(ValueError, match=r"1 cells of bad input: .*\(first: cell 0: sum\(x\) - 1 = ") as raised:
        parity.solver(helpers.tensor([above], parity.device), T[:1], P[:1], dt, h)
    _, _, done, _ = parity.solver(helpers.tensor([below], parity.device), T[:1], P[:1], dt, h)
    verdict(
        "guard: a sum on the tolerance, judged in the warp's order only",
        bool(done.all()) and parity.solver.fallback["bad_input"] == 0,
        f"warp {helpers.warp_sum(above, perm) - 1:.16g} / network order {sum(above) - 1:.16g}: {raised.value}; warp "
        f"{helpers.warp_sum(below, perm) - 1:.16g} / network order {sum(below) - 1:.16g}: accepted, done",
    )


def test_non_finite_step_acts_as_dt(parity, verdict):
    cells = 64
    x0, T, P = parity.starts(cells, 2, 1.0)
    dt = parity.full(cells, 50.0)
    ref = parity.solver(x0, T, P, dt, dt.clone())
    none_counted = parity.solver.fallback["nonfinite_h"] == 0
    same, counted = [], []
    for q in (np.nan, -np.inf, np.inf):
        out = parity.solver(x0, T, P, dt, parity.full(cells, q))
        same.append(all(torch.equal(a, b) for a, b in zip(out, ref, strict=True)))
        counted.append(parity.solver.fallback["nonfinite_h"] == cells)
    verdict(
        "non-finite starting step",
        all(same) and all(counted) and none_counted,
        f"h = NaN, -inf, +inf bitwise the h = dt call: {same}; each cell counted once: {counted}; finite h counted "
        f"none: {none_counted}",
    )


def test_launch_adds_to_the_total(parity):
    cells = 8
    x, T, P = parity.starts(cells, 2, 1.0)
    before = parity.solver.total.clone()
    parity.solver.launch(x, T, P, parity.full(cells, 1.0), parity.full(cells, 1.0))
    assert torch.equal(parity.solver.total - before, parity.solver.count.to(torch.int64))


def test_call_and_launch_name_their_results(parity):
    cells = 8
    x, T, P = parity.starts(cells, 2, 1.0)
    args = (x, T, P, parity.full(cells, 50.0), parity.full(cells, 1e-10))
    launched = parity.solver.launch(*args, max_steps=2)  # few tries: the guard recomputes cells, so flags are not all 0
    assert isinstance(launched, kincuda.LaunchResult)
    assert parity.solver.flag is launched.flag
    assert bool((launched.flag != 0).any())
    called = parity.solver(*args, max_steps=2)
    assert isinstance(called, kincuda.CallResult)
    assert all(torch.equal(getattr(called, name), getattr(launched, name)) for name in called._fields)
    assert torch.equal(parity.solver.flag, launched.flag)  # the call keeps the flag too


def test_flag_is_none_after_a_call_that_raised(parity):
    cells = 8
    x, T, P = parity.starts(cells, 2, 1.0)
    dt = h = parity.full(cells, 1.0)
    bad = x.clone()
    bad[3, 0] = float("nan")
    parity.solver(x, T, P, dt, h)
    with pytest.raises(ValueError, match="refused cells"):
        parity.solver.launch(bad, T, P, dt, h)
    assert parity.solver.flag is None  # the guard gives a refused cell flag 0, which would read as passed
    parity.solver(x, T, P, dt, h)
    with pytest.raises(ValueError, match=r"^T: "):  # refused on the host, before any launch
        parity.solver(x[:5], T[:5].cpu(), P[:5], dt[:5], h[:5])
    assert parity.solver.flag is None  # not the previous call's 8-cell flag


def test_wrong_inputs_raise_before_any_launch(parity):
    cells = 64
    x, T, P = parity.starts(cells, 2, 1.0)
    dt = h = parity.full(cells, 50.0)
    with pytest.raises(ValueError, match=r"^T: .* on cpu$"):
        parity.solver(x, T.cpu(), P, dt, h)
    with pytest.raises(ValueError, match="required"):
        parity.lit(x, T, P, dt, h)
    with pytest.raises(ValueError, match=r"^max_steps from 1"):
        parity.solver(x, T, P, dt, h, max_steps=0)


def test_closed_solver_refuses_a_call(parity, tmp_path):
    with kincuda.CudaSolver(parity.network.net, parity.device) as solver:
        assert solver.fallback["cold"] == 0
    solver.close()  # closing twice does nothing
    x, T, P = parity.starts(8, 2, 1.0)
    with pytest.raises(RuntimeError, match="closed"):
        solver(x, T, P, parity.full(8, 1.0), parity.full(8, 1.0))
    with pytest.raises(RuntimeError, match="closed"):
        solver.export(tmp_path)


def test_closed_solver_refuses_its_rate_table(read_network, gpu):
    with kincuda_ad.ADSolver(read_network("synthetic_small"), gpu) as ad:
        assert ad.rc0.shape[0] == ad.nr
    with pytest.raises(RuntimeError, match="closed"):
        _ = ad.rc0


@pytest.mark.parametrize("build", [kincuda.CudaSolver, kincuda_ad.ADSolver])
def test_close_releases_the_gpu_memory(read_network, gpu, build):
    before = torch.cuda.memory_allocated(gpu)
    solver = build(read_network("synthetic_small"), gpu)
    held = torch.cuda.memory_allocated(gpu) - before
    solver.close()
    left = torch.cuda.memory_allocated(gpu) - before
    print(f"{build.__name__}: holds {held} B, {left} B after close")
    assert left <= CLOSED_SOLVER_BYTES < held


def test_results_do_not_depend_on_the_batch(parity, verdict):
    cells = 8192
    x0, T, P = parity.starts(cells, 7, 1.0)
    args = (x0, T, P, parity.full(cells, 50.0), parity.full(cells, 1e-10))
    big, sel = parity.solver.launch(*args), torch.arange(0, cells, 64, device=parity.device)
    perm = torch.randperm(cells, generator=torch.Generator().manual_seed(3)).to(parity.device)
    shuffled = parity.solver.launch(*(q[perm] for q in args))
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(cells, device=parity.device)
    alone = all(torch.equal(p[sel], q) for p, q in zip(big, parity.solver.launch(*(q[sel] for q in args)), strict=True))
    one = all(
        torch.equal(p[sel[:1]], q) for p, q in zip(big, parity.solver.launch(*(q[sel[:1]] for q in args)), strict=True)
    )
    order = all(torch.equal(p, q[inverse]) for p, q in zip(big, shuffled, strict=True))
    verdict(
        "batch size and order",
        alone and one and order,
        f"128 cells alone bitwise inside 8,192 {alone}; one cell alone bitwise {one}; 8,192 cells permuted "
        f"bitwise {order}",
    )


ALLOWANCE = 1e-6  # the drift allowance given to the solver: DRIFT of docs/method.md
ATOMS = {"H": {"H": 1}, "H2": {"H": 2}, "OH": {"H": 1, "O": 1}, "H2O": {"H": 2, "O": 1}, "O": {"O": 1}, "He": {"He": 1}}
DRIFTS = (0.5, 0.99, 1.01, 2.0)  # the largest relative change of an element's share, in units of the allowance


def test_drift_beyond_the_allowance_is_flagged(read_network, gpu):
    # The guard alone, on hand-made pairs of a state and a result whose He share moved by r times the allowance
    net = read_network("synthetic_small")
    x0 = np.array([{"H": 0.1, "H2": 0.6, "OH": 0.05, "H2O": 0.1, "O": 0.05, "He": 0.1}[q] for q in net.species])
    atoms = np.array([[ATOMS[q].get(e, 0) for e in ("H", "O", "He")] for q in net.species], dtype=float)
    share = (x0 @ atoms) / (x0 @ atoms).sum()
    he = net.species.index("He")
    f = share[2]
    results = []
    for r in DRIFTS:
        # He's share (1 + d) f / (1 + f d) moves by r ALLOWANCE relative; the others by less
        d = r * ALLOWANCE / (1.0 - f - f * r * ALLOWANCE)
        x = x0.copy()
        x[he] *= 1.0 + d
        moved = (x @ atoms) / (x @ atoms).sum() / share - 1.0
        assert np.abs(moved).max() == pytest.approx(r * ALLOWANCE, rel=1e-6)
        results.append(x)
    cells = len(DRIFTS)
    with kincuda.CudaSolver(net, gpu, drift=ALLOWANCE) as solver:
        values = {
            **solver._tables,
            "x0": helpers.tensor([x0] * cells, gpu),
            "xout": helpers.tensor(results, gpu),
            "doneo": torch.ones(cells, dtype=torch.int32, device=gpu),
            "noko": helpers.full(cells, 1.0, gpu),
            "status": torch.zeros(cells, dtype=torch.int32, device=gpu),
            "redo": torch.zeros(cells, dtype=torch.int32, device=gpu),
            "list": torch.zeros(cells, dtype=torch.int32, device=gpu),
            "count": solver.count,
            "B": ctypes.c_int(cells),
            "atol": ctypes.c_double(kincuda.ATOL),
        }
        solver.count.zero_()
        stream = torch.cuda.current_stream(gpu).cuda_stream
        with torch.cuda.device(gpu):
            _cuda.launch_kernel(solver._guard_kernel, 1, cells, 0, stream, solver._arguments("guard", values))
        counts = dict(solver.fallback)
    print(f"drift {DRIFTS} allowances: flagged {values['redo'].tolist()}; counts {counts}")
    assert values["redo"].tolist() == [int(r > 1.0) for r in DRIFTS]
    assert counts["drift"] == counts["recomputed"] == sum(r > 1.0 for r in DRIFTS)


def test_one_step_call_whose_recovery_drifts_is_retried(gpu):
    # Box 98666's last call (tests/data/recovery_box98666.json): one step of 5.6e7 s from its first try, the guard
    # flags the element drift, and the recovery's own one step drifts again. Its retry from the inputs with a smaller
    # first try passes.
    case = json.loads((Path(__file__).parent / "data" / "recovery_box98666.json").read_text())
    thermo = paths.vulcan_thermo()
    net = kinvulcan.VulcanNetwork(thermo / case["network"], thermo / "NASA9", species=case["species"])
    J = [case["J"][f"{r['absorber']},{r['branch']}"] for r in net.photo]
    with kincuda.CudaSolver(net, gpu, photo=True) as solver:
        y, _, done, _, flag = solver.launch(
            helpers.tensor([case["x"]], gpu),
            helpers.full(1, case["T"], gpu),
            helpers.full(1, case["P"], gpu),
            helpers.full(1, case["dt"], gpu),
            helpers.full(1, case["h"], gpu),
            J=helpers.tensor([J], gpu),
        )
        counts, allowance = dict(solver.fallback), solver.drift
    shares = [(np.asarray(v) @ net.atoms) / (np.asarray(v) @ net.atoms).sum() for v in (case["x"], y[0].cpu().numpy())]
    moved = float(np.abs(shares[1] / shares[0] - 1.0).max())
    print(f"flag {int(flag[0])}, done {bool(done[0])}, shares moved {moved:.2e} of {allowance:.0e}; {counts}")
    assert (int(flag[0]), bool(done[0]), counts["recovered"], counts["still_failed"]) == (1, True, 1, 0)
    assert moved <= allowance
