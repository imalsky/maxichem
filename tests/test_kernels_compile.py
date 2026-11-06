"""Every form of the kernels compiles with NVRTC, with no GPU. The real networks never reach some of these forms."""

import copy

import pytest

from maxichem import _ad_tables, _cuda, _kernel_header, _tables

ARCH = "89"  # sm_89: the RTX 6000 Ada, the card the kernels run on


def _with_reverse_law(net):
    """Return a copy of the network with an explicit reverse law, which the reader does not produce."""
    copied = copy.deepcopy(net)
    copied.reac[0]["rev"] = {"kind": 2, "A": 1e-12, "B": 0.0, "C": 500.0}
    return copied


def _compile(tab):
    """Compile the solver's kernels for these tables."""
    sources = _kernel_header.kernel_sources(_kernel_header.call_header(tab, 1))
    return _cuda.compile_cubin(_cuda.SOLVER_KERNELS, sources, ARCH)


@pytest.mark.parametrize("photo", [False, True])
def test_call_kernels_compile_with_every_rate_law(read_network, photo):
    net = read_network()
    assert _compile(_tables.tables(net, 2, photo))


def test_call_kernels_compile_with_plain_falloff(read_network):
    tab = _tables.tables(read_network("synthetic_falloff"), 2)
    assert _compile(tab)


def test_call_kernels_compile_with_a_reverse_law(read_network):
    tab = _tables.tables(_with_reverse_law(read_network()), 2)
    assert "rev" in tab["law"]
    assert _compile(tab)


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["jvp", "vjp"])
def test_derivative_kernels_compile(read_network, kind):
    net = read_network()
    tab = _tables.tables(net, 2, True)
    header = _kernel_header.derivative_header(tab, _ad_tables.ad_tables(net, tab), kind, 2)
    assert _cuda.compile_cubin(f"{kind}.cu", _kernel_header.kernel_sources(header), ARCH)
