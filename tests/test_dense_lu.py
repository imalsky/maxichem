"""The recovery's dense LU with threshold partial pivoting, against the rule of docs/method.md written out in numpy.

Two kinds of matrix: a diagonally dominant one with its rows shuffled, so that rows exchange at most pivots; and a
random one with every other diagonal small, where some diagonals are not their column's largest but within PIVTAU of
it, so that threshold pivoting keeps the row where plain partial pivoting would exchange it. The harness kernel
tests/dense_lu_check.cu runs the recovery's dense_lu and dense_solve on each matrix.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import helpers
import numpy as np
import pytest
import torch

from maxichem import _constants, _cuda, _kernel_header, _tables

if TYPE_CHECKING:
    import ctypes

pytestmark = pytest.mark.gpu

PIVTAU = 0.1  # docs/method.md: rows exchange only where the diagonal is below this share of the column's largest
MATRICES = 8  # of each kind
DOMINANT = 10.0  # the diagonal of the shuffled matrices, against off-diagonal entries of order 0.1
OFF_DIAGONAL = 0.1
SMALL_DIAGONAL = 1e-3  # on every other diagonal of the random matrices
TIE_ROWS = [3, 7]  # rows of lanes 3 and 7 holding column 0's equal largest |entry|
TIE = 10.0  # above every entry of a standard-normal column of 42
FACTOR_TOLERANCE = 1e-13  # |P A - L U| against |L| |U|, per entry: rounding of sums in another order (fma)
SOLVE_TOLERANCE = 1e-10  # relative, against numpy's solve and factors: rounding times these matrices' condition
HARNESS = Path(__file__).parent / "dense_lu_check.cu"


def threshold_lu(a: np.ndarray) -> tuple[list[int], np.ndarray, int, int]:
    """Return the row order, the factors (unit L below the diagonal, U on and above), exchanges and kept pivots.

    At pivot k the candidate is the row of the largest |entry| of column k at or below k, the lower row on a tie; it
    is exchanged with row k only when |a_kk| is below PIVTAU times that entry. A kept pivot is one where the candidate
    is another row and plain partial pivoting would have exchanged.
    """
    a = a.copy()
    n = len(a)
    order, exchanges, kept = list(range(n)), 0, 0
    for k in range(n):
        best = k + int(np.argmax(np.abs(a[k:, k])))  # the first of equal entries
        if best != k and not abs(a[k, k]) >= PIVTAU * abs(a[best, k]):
            a[[k, best]] = a[[best, k]]
            order[k], order[best] = order[best], order[k]
            exchanges += 1
        elif best != k:
            kept += 1
        a[k + 1 :, k] *= 1.0 / a[k, k]
        a[k + 1 :, k + 1 :] -= np.outer(a[k + 1 :, k], a[k, k + 1 :])
    return order, a, exchanges, kept


def _matrices(ns: int, rng: np.random.Generator) -> np.ndarray:
    """Return MATRICES shuffled dominant matrices, MATRICES random ones with small diagonals, and one with a tie.

    The tie: column 0's largest |entry| twice, in rows of two lanes, so the warp's reduction must take the lower row.
    """
    shuffled = OFF_DIAGONAL * rng.standard_normal((MATRICES, ns, ns)) + DOMINANT * np.eye(ns)
    shuffled = np.stack([q[rng.permutation(ns)] for q in shuffled])
    random = rng.standard_normal((MATRICES, ns, ns))
    random[:, range(0, ns, 2), range(0, ns, 2)] *= SMALL_DIAGONAL
    tie = random[0].copy()
    tie[TIE_ROWS, 0] = (TIE, -TIE)
    return np.concatenate([shuffled, random, tie[None]])


def test_dense_lu_pivots_by_the_threshold_rule(read_network, gpu):
    tab = _tables.tables(read_network(), 2)  # 42 species: a lane holds two rows
    ns = tab["ns"]
    header = _kernel_header.call_header(tab, 1, False, _constants.DRIFT)
    sources = _kernel_header.kernel_sources(header) | {HARNESS.name: HARNESS.read_text()}
    prop = torch.cuda.get_device_properties(gpu)
    cubin = _cuda.compile_cubin(HARNESS.name, sources, f"{prop.major}{prop.minor}")
    rng = np.random.default_rng(11)
    a = _matrices(ns, rng)
    count = len(a)
    y = rng.standard_normal((count, ns))
    factors, solved = helpers.tensor(a, gpu), helpers.tensor(y, gpu)
    dinv = torch.empty(count, ns, dtype=torch.float64, device=gpu)
    rows = torch.empty(count, ns, dtype=torch.int32, device=gpu)
    modules: list[ctypes.c_void_p] = []
    with torch.cuda.device(gpu):
        try:
            _, (kernel,) = _cuda.load_module(cubin, ["dense_lu_check"], modules)
            stream = torch.cuda.current_stream(gpu).cuda_stream
            _cuda.launch_kernel(kernel, count, 32, 0, stream, [factors, dinv, rows, solved])
        finally:
            _cuda.unload_modules(gpu, modules)
    exchanged, kept_rows = [], []
    for m in range(count):
        order, expected, exchanges, kept = threshold_lu(a[m])
        exchanged.append(exchanges)
        kept_rows.append(kept)
        got = factors[m].cpu().numpy()
        lower, upper = np.tril(got, -1) + np.eye(ns), np.triu(got)
        residual = np.abs(lower @ upper - a[m][order]) / (np.abs(lower) @ np.abs(upper))
        solution = np.linalg.solve(a[m], y[m])
        error = np.abs(solved[m].cpu().numpy() - solution).max() / np.abs(solution).max()
        print(f"matrix {m}: {exchanges} exchanges, {kept} kept, residual {residual.max():.1e}, solve {error:.1e}")
        assert rows[m].tolist() == order
        assert residual.max() <= FACTOR_TOLERANCE
        assert np.allclose(got, expected, rtol=SOLVE_TOLERANCE, atol=SOLVE_TOLERANCE * np.abs(expected).max())
        assert np.array_equal(dinv[m].cpu().numpy(), 1.0 / np.diag(upper))
        assert error <= SOLVE_TOLERANCE
    # the cases exercise what they are for: exchanges at most pivots, and rows kept by the threshold
    assert min(exchanged[:MATRICES]) >= ns // 2
    assert min(kept_rows[MATRICES : 2 * MATRICES]) >= 1
    assert rows[-1].tolist()[0] == TIE_ROWS[0]  # the tie went to the lower row
