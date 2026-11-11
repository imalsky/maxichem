"""The collision limit against closed forms, on an invented network whose laws are far above the limit.

A direction counts when its molecules plus M make two collision partners, so a dissociation A + M counts with k (M
not folded in) as its bimolecular-equivalent rate. Its capped coefficient is then K_COLL n_tot, with M folded in; the
pair keeps its ratio, so each coefficient below is the closed form of one direction and the NASA-9 ratio. The
derivatives are those of the closed forms at fixed P, from the tangent (rates_tan) and the adjoint (rate_cotangents)
arithmetic, through tests/rate_probe.cu.
"""

from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from maxichem import _ad_tables, _constants, _cuda, _kernel_header, _solver, _tables, kinvulcan

if TYPE_CHECKING:
    import ctypes

pytestmark = pytest.mark.gpu

PROBE = Path(__file__).parent / "rate_probe.cu"
K_COLL = _constants.K_COLL
# s^-1: the first-order N_3 -> N_4 of the invented network and the photolysis rate; as k / n_tot at 1e-3 bar they
# would be 400 times the limit
K_FIRST = 1e9
J_TEST = 1e9
T_TEST = 3000.0  # K
# dyn cm^-2: 1e-3 bar, where H2 + M and O2 + M dissociating are the faster directions, and 10 bar, where the
# associations are (kr / kf over the bimolecular-equivalent rates: 0.032 and 320 for H2, 51 and 0.0051 for O2)
P_LOW, P_HIGH = 1e3, 1e7
RATE_TOLERANCE = 1e-13  # relative: exp and log rounding
DERIVATIVE_TOLERANCE = 1e-12  # relative to 1 / T along T, to 1 along ln P


def _probe(net, gpu, cells):
    """Return the probe's outputs per cell: kf, kr [cells, R], dk [cells, 2 R], g (T, ln nd cotangents) [cells, 2]."""
    prop = torch.cuda.get_device_properties(gpu)
    tab = _tables.tables_for_device(net, True, prop.shared_memory_per_block_optin)
    ext = _ad_tables.ad_tables(net, tab)
    blob, _ = _tables.pack_small_tables(tab)
    _, perwa = _ad_tables.layout(tab, ext, "vjp")
    header = _kernel_header.derivative_header(tab, ext, "vjp", 1, _constants.CAP, _constants.DRIFT)
    cubin = _cuda.compile_cubin(
        PROBE.name, _kernel_header.kernel_sources(header) | {PROBE.name: PROBE.read_text()}, f"{prop.major}{prop.minor}"
    )
    shared = _solver.shared_tables(tab, blob, gpu)
    n, nr = len(cells["T"]), tab["nr"]
    sizes = {"kf": nr, "kr": nr, "cw": 2 * nr, "dk": 2 * nr, "glm": nr, "gA": nr, "gJ": tab["nj"], "g": 2}
    out = {k: torch.zeros((n, s), dtype=torch.float64, device=gpu) for k, s in sizes.items()}
    inputs = [torch.tensor(cells[k], dtype=torch.float64, device=gpu) for k in ("T", "P", "dT", "dlnd", "lbk", "J")]
    work = torch.zeros((n, perwa), dtype=torch.float64, device=gpu)
    args = [*inputs, shared["rc"], shared["rf"], shared["nasa"], shared["blob"], work, *out.values()]
    modules: list[ctypes.c_void_p] = []
    with torch.cuda.device(gpu):
        try:
            _, (fn,) = _cuda.load_module(cubin, ["rate_probe"], modules)
            _cuda.launch_kernel(fn, n, 32, len(blob), torch.cuda.current_stream().cuda_stream, args)
            torch.cuda.synchronize()
        finally:
            _cuda.unload_modules(gpu, modules)
    return {k: v.cpu().numpy() for k, v in out.items()}


def _gibbs(net, name):
    """Return G / RT of a species of the invented data (constant Cp / R) at T_TEST and its T derivative."""
    cp, h, b2 = net.lo[net.species.index(name)][[2, 7, 8]]
    return cp + h / T_TEST - (cp * np.log(T_TEST) + b2), -h / T_TEST**2 - cp / T_TEST


def _closed_forms(net, P):
    """Return {equation: (kf, kr, d ln kf / dT, d ln kr / dT, d ln kf / d ln P, d ln kr / d ln P)} at T_TEST, P."""
    t = T_TEST
    nd, lkt = P / (kinvulcan.KB * t), np.log(kinvulcan.KB * t / kinvulcan.P0)
    (g_h, d_h), (g_h2, d_h2), (g_o, d_o), (g_o2, d_o2) = (_gibbs(net, q) for q in ("H", "H2", "O", "O2"))
    eh, deh = np.exp(2 * g_h - g_h2 + lkt), 2 * d_h - d_h2 + 1 / t  # kr / kf of H2 + M -> H + H + M, its d ln / dT
    eo, deo = np.exp(g_o2 - 2 * g_o - lkt), d_o2 - 2 * d_o - 1 / t  # kr / kf of O + O + M -> O2 + M
    # the faster direction is capped to K_COLL as a bimolecular-equivalent rate: K_COLL nd for A + M, K_COLL for A + B
    if P == P_LOW:
        h2 = (K_COLL * nd, K_COLL * nd * eh, -1 / t, deh - 1 / t, 1, 1)
        o2 = (K_COLL * nd / eo, K_COLL * nd, -1 / t - deo, -1 / t, 1, 1)
    else:
        h2 = (K_COLL / eh, K_COLL, -deh, 0, 0, 0)
        o2 = (K_COLL, K_COLL * eo, 0, deo, 0, 0)
    return {
        "H2 + M -> H + H + M": h2,
        "O + O + M -> O2 + M": o2,
        "N_1 + M -> N_2 + M": (K_COLL * nd, K_COLL * nd, -1 / t, -1 / t, 1, 1),  # a tie: the same q both ways
        "N_3 -> N_4": (K_FIRST, K_FIRST, 0, 0, 0, 0),  # one molecule, no M: not counted
        "H2O -> H + OH": (J_TEST, 0.0, 0, 0, 0, 0),  # photolysis: not counted
    }


def _cells(net):
    """Return the probe's cells: each pressure x thermal reaction x cotangent (ln kf, ln kr) x tangent (T, ln P)."""
    nrt, nr = len(net.reac), len(net.reac) + len(net.photo)
    rows = list(product((P_LOW, P_HIGH), range(nrt), (0, 1), (0, 1)))
    along_t = np.array([q[3] == 0 for q in rows])
    lbk = np.zeros((len(rows), 2 * nr))
    for i, (_, r, side, _) in enumerate(rows):
        lbk[i, r] = 1.0  # the part shared by ln kf and ln kr; with lbk[nr + r] the cotangent is on ln kr alone
        lbk[i, nr + r] = float(side)
    cells = {
        "T": np.full(len(rows), T_TEST),
        "P": np.array([q[0] for q in rows]),
        "dT": along_t.astype(float),
        "dlnd": np.where(along_t, -1.0 / T_TEST, 1.0),  # fixed P along T; along ln P, d ln nd = d ln P
        "lbk": lbk,
        "J": np.full((len(rows), len(net.photo)), J_TEST),
    }
    return rows, cells


def test_capped_coefficients_and_their_derivatives_match_closed_forms(read_network, gpu):
    net = read_network("synthetic_collision")
    rows, cells = _cells(net)
    got = _probe(net, gpu, cells)
    names = [r["eq"] for r in net.reac] + [r["eq"] for r in net.photo]
    nrt, nr = len(net.reac), len(names)
    worst = dict.fromkeys(("rate", "tangent", "adjoint"), 0.0)
    for i, (P, r, side, along) in enumerate(rows):
        forms = _closed_forms(net, P)
        scale = 1.0 / T_TEST if along == 0 else 1.0
        for q, eq in enumerate(names):
            if eq not in forms:
                continue  # the zero-rate H + H2O line
            for value, want in zip((got["kf"][i, q], got["kr"][i, q]), forms[eq][:2], strict=True):
                assert value == pytest.approx(want, rel=RATE_TOLERANCE, abs=0.0), (eq, P, value, want)
                worst["rate"] = max(worst["rate"], abs(value - want) / want if want else 0.0)
            if q < nrt:  # the tangent of ln kf and ln kr along this cell's direction
                tangents = (got["dk"][i, q] / got["kf"][i, q], got["dk"][i, nr + q] / got["kr"][i, q])
                for value, want in zip(tangents, forms[eq][2 + 2 * along : 4 + 2 * along], strict=True):
                    assert abs(value - want) <= DERIVATIVE_TOLERANCE * scale, (eq, P, along, value, want)
                    worst["tangent"] = max(worst["tangent"], abs(value - want) / scale)
        if names[r] in forms:  # the adjoint of ln kf (side 0) or ln kr (side 1) of reaction r, at fixed P along T
            g_temperature, g_density = got["g"][i]
            value = g_temperature - g_density / T_TEST if along == 0 else g_density
            want = _closed_forms(net, P)[names[r]][2 + 2 * along + side]
            assert abs(value - want) <= DERIVATIVE_TOLERANCE * scale, (names[r], P, side, along, value, want)
            worst["adjoint"] = max(worst["adjoint"], abs(value - want) / scale)
    print(f"collision limit against closed forms: largest relative errors {worst}")
