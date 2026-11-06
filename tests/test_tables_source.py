"""The kernels' tables and their source on the invented networks: what the streams hold, not how they are packed."""

import copy
import re
from pathlib import Path

import numpy as np
import pytest

from maxichem import _constants, _kernel_header, _tables


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


@pytest.mark.parametrize("photo", [False, True])
def test_streams_hold_the_stoichiometry(read_network, photo):
    net = read_network()
    tab = _tables.tables(net, 2, photo)
    stoichiometry = np.hstack([net.S, net.Sp]) if photo else net.S
    assert sorted(tab["perm"].tolist()) == list(range(tab["ns"]))
    assert np.array_equal(_stoichiometry_from_stream(tab), stoichiometry[tab["perm"]])


@pytest.mark.parametrize("photo", [False, True])
def test_small_network_has_tables(read_network, photo):
    net = read_network("synthetic_small")
    tab = _tables.tables(net, 2, photo)
    stoichiometry = np.hstack([net.S, net.Sp]) if photo else net.S
    assert tab["nfix"] <= tab["ns"]  # the extra Jacobian pieces fit in Y
    assert np.array_equal(_stoichiometry_from_stream(tab), stoichiometry[tab["perm"]])


def test_rate_law_table_takes_the_network_forms(read_network):
    assert _tables.tables(read_network("synthetic_falloff"), 2)["law"] == []
    tab = _tables.tables(read_network(), 2)
    assert tab["law"] == ["eff", "troe"]
    assert tab["rc"].shape[1] == 14


def test_header_selects_the_rate_law_forms(read_network):
    net = read_network()
    reverse = copy.deepcopy(net)
    reverse.reac[0]["rev"] = {"kind": 2, "A": 1e-12, "B": 0.0, "C": 500.0}
    for network, troe, eff, rev in (
        (net, 1, 1, 0),
        (read_network("synthetic_falloff"), 0, 0, 0),
        (reverse, 1, 1, 1),
    ):
        header = _kernel_header.call_header(_tables.tables(network, 2), 1)
        for name, value in (("HAS_TROE", troe), ("HAS_EFF", eff), ("HAS_REV", rev)):
            assert f"#define {name} {value}\n" in header


@pytest.mark.parametrize("kernel", sorted(_constants.KERNEL_ARGUMENTS))
def test_argument_table_matches_the_kernel_signature(kernel):
    sources = _kernel_header.kernel_sources("")
    text = "".join(sources.values())
    signature = re.search(rf"(?:\s|\b){kernel}\(([^)]*)\)\s*{{", text)
    assert signature is not None
    names = [re.findall(r"\w+", parameter)[-1] for parameter in signature.group(1).replace(" J_PARAM", "").split(",")]
    assert tuple(names) == _constants.KERNEL_ARGUMENTS[kernel]


def test_cpp_defaults_match_the_python_defaults():
    header = (Path(__file__).parents[1] / "cpp" / "maxichem.hpp").read_text()
    rtol, steps = re.search(r"kDefaultRtol = ([\d.e-]+);", header), re.search(r"kDefaultMaxSteps = (\d+);", header)
    assert rtol is not None
    assert steps is not None
    assert float(rtol.group(1)) == _constants.DEFAULT_RTOL
    assert int(steps.group(1)) == _constants.DEFAULT_MAX_STEPS
