"""The kernels' tables and their source on the invented networks: what the streams hold, not how they are packed."""

import gc
import re
import sys

import numpy as np
import pytest
import torch

from maxichem import _constants, _kernel_header, _solver, _tables, kincuda, kincuda_ad


def _stoichiometry_from_stream(tab):
    """Decode the S v lane stream back into the stoichiometry per kernel row."""
    w = tab["widths"]
    ns, nr = tab["ns"], tab["nr"]
    pieces = np.zeros((tab["npart"], nr))
    for word in map(int, tab["sst"].ravel()):
        if word >> 31:
            reaction = word & ((1 << w["r"]) - 1)
            sign = -1.0 if (word >> w["r"]) & 1 else 1.0
            size = 2.0 if (word >> (w["r"] + 1)) & 1 else 1.0
            piece = (word >> (w["r"] + 2)) & ((1 << w["part"]) - 1)
            pieces[piece, reaction] += sign * size
    return np.stack([pieces[i] + pieces[ns + tab["xp"][i] : ns + tab["xp"][i + 1]].sum(0) for i in range(ns)])


@pytest.mark.parametrize("stem", ["synthetic", "synthetic_small"])
@pytest.mark.parametrize("photo", [False, True])
def test_streams_hold_the_stoichiometry(read_network, stem, photo):
    net = read_network(stem)
    tab = _tables.tables(net, 2, photo)
    stoichiometry = np.hstack([net.S, net.Sp]) if photo else net.S
    assert sorted(tab["perm"].tolist()) == list(range(tab["ns"]))
    assert tab["nfix"] <= tab["ns"]  # the extra Jacobian pieces fit in Y
    assert np.array_equal(_stoichiometry_from_stream(tab), stoichiometry[tab["perm"]])


def test_hand_built_reverse_law_is_refused(read_network):
    net = read_network()
    net.reac[0]["rev"] = {"kind": 2, "A": 2e-12, "B": 0.0, "C": 0.0}
    with pytest.raises(ValueError, match=f"reaction {re.escape(net.reac[0]['eq'])} has an explicit reverse law"):
        _tables.tables(net, 2)


def test_hand_built_law_whose_m_does_not_match_is_refused(read_network):
    # The tables fold n_tot into both directions by the law alone, and the collision limit counts M on both sides.
    net = read_network()
    j = next(j for j, r in enumerate(net.reac) if r["eq"].replace(" ", "") == "O+O+M->O2+M")
    net.reac[j] |= {"kind": 2, "A": 1e-30, "B": 0.0, "C": 0.0}
    with pytest.raises(ValueError, match=f"reaction {j}: M on reactant and product side, .* on neither side"):
        _tables.tables(net, 2)
    net = read_network()
    j = next(j for j, r in enumerate(net.reac) if r["eq"].replace(" ", "") == "H+OH->H2+O")
    net.reac[j] |= {"kind": 5, "A0": 1e-30, "B0": 0.0, "C0": 0.0}
    with pytest.raises(ValueError, match=f"reaction {j}: M on no side, .* on both sides"):
        _tables.tables(net, 2)


def test_rate_law_table_takes_the_network_forms(read_network):
    assert _tables.tables(read_network("synthetic_falloff"), 2)["law"] == []
    tab = _tables.tables(read_network(), 2)
    assert tab["law"] == ["eff", "troe"]
    assert tab["rc"].shape[1] == 14


def test_header_selects_the_rate_law_forms(read_network):
    for network, troe, eff in ((read_network(), 1, 1), (read_network("synthetic_falloff"), 0, 0)):
        header = _kernel_header.call_header(_tables.tables(network, 2), 1, True, _constants.DRIFT)
        for name, value in (("HAS_TROE", troe), ("HAS_EFF", eff)):
            assert f"#define {name} {value}\n" in header


@pytest.mark.parametrize("kernel", sorted(_constants.KERNEL_ARGUMENTS))
def test_argument_table_matches_the_kernel_signature(kernel):
    sources = _kernel_header.kernel_sources("")
    text = "".join(sources.values())
    signature = re.search(rf"(?:\s|\b){kernel}\(([^)]*)\)\s*{{", text)
    assert signature is not None
    if kernel in _constants.SOLVER_KERNEL_NAMES:  # J last only in the kernels the callers append it to
        assert signature.group(1).endswith(" J_PARAM") == (kernel in _constants.PHOTOLYSIS_KERNEL_NAMES)
    parameters = signature.group(1).replace(" J_PARAM", "")
    for macro in ("JVP_ARGS", "VJP_ARGS"):
        definition = re.search(rf"#define {macro}\s+((?:.*\\\n)*.*)", text)
        assert definition is not None
        parameters = parameters.replace(macro, definition.group(1).replace("\\\n", " "))
    names = [re.findall(r"\w+", parameter)[-1] for parameter in parameters.split(",")]
    assert tuple(names) == _constants.KERNEL_ARGUMENTS[kernel]


def _good_inputs(cells=4, species=3):
    """Return the arguments of a call that the input checks accept, by name, on the CPU."""
    one = torch.ones(cells, dtype=torch.float64)
    return {
        "x": torch.ones(cells, species, dtype=torch.float64),
        "T": one,
        "P": one,
        "dt": one,
        "h": one,
        "rtol": 1e-3,
        "atol": 1e-25,
        "max_steps": 10,
    }


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"h": torch.ones(3, dtype=torch.float64)}, r"^h: float64 \[4\] .* got torch.float64 \[3\]"),
        ({"P": torch.ones(4, dtype=torch.float32)}, r"^P: .* got torch.float32"),
        ({"x": torch.ones(4, 2, dtype=torch.float64)}, r"^x: \[B, 3\]"),
        ({"rtol": 0.0}, "^rtol and atol"),
        ({"atol": float("nan")}, "^rtol and atol"),
        ({"max_steps": 2.5}, "^max_steps: an integer"),
        ({"max_steps": 2**31}, "^max_steps from 1"),
    ],
)
def test_input_checks_name_the_input(change, message):
    device = torch.device("cpu")
    assert _solver.check_inputs(device, 3, **_good_inputs()) == 4
    with pytest.raises(ValueError, match=message):
        _solver.check_inputs(device, 3, **(_good_inputs() | change))


def test_photolysis_check_refuses_each_wrong_j():
    device = torch.device("cpu")
    good = torch.zeros(4, 2, dtype=torch.float64)
    assert _solver.check_photolysis(None, 4, 0, device) is None
    assert torch.equal(_solver.check_photolysis(good, 4, 2, device), good)
    with pytest.raises(ValueError, match="no photolysis"):
        _solver.check_photolysis(good, 4, 0, device)
    with pytest.raises(ValueError, match="required"):
        _solver.check_photolysis(None, 4, 2, device)
    with pytest.raises(ValueError, match=r"^J: float64 \[4, 2\]"):
        _solver.check_photolysis(good[:-1], 4, 2, device)


@pytest.mark.parametrize("drift", [float("nan"), float("inf"), -1e-6])
def test_bad_drift_is_refused_before_any_device_use(read_network, drift):
    net = read_network()
    with pytest.raises(ValueError, match="drift"):
        kincuda.CudaSolver(net, "cuda", drift=drift)
    with pytest.raises(ValueError, match="drift"):
        kincuda_ad.ADSolver(net, "cuda", drift=drift)


@pytest.mark.parametrize("cap", [0, -1, 2.5])
def test_bad_cap_is_refused_before_any_device_use(read_network, cap):
    with pytest.raises(ValueError, match="cap"):
        kincuda_ad.ADSolver(read_network(), "cuda", cap=cap)


@pytest.mark.parametrize("solver", [kincuda.CudaSolver, kincuda_ad.ADSolver])
def test_failed_constructor_leaves_no_error_at_deletion(read_network, monkeypatch, solver):
    net = read_network()
    net.photo = []
    seen: list[object] = []
    monkeypatch.setattr(sys, "unraisablehook", seen.append)
    with pytest.raises(ValueError, match="photo=True"):
        solver(net, "cuda", photo=True)  # refused before it loads anything: nothing to unload
    gc.collect()
    assert not seen
