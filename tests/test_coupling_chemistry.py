"""The stop on unfinished chemistry cells names the cause the solver counted, and only the fix that matches it."""

import pytest
import torch

from coupling.chemistry import UnfinishedCellsError, require_finished
from maxichem.kincuda import COUNTS

T = torch.tensor([1500.0, 280.0, 1200.0], dtype=torch.float64)
P = torch.tensor([1.0e5, 3.0e2, 2.0e6], dtype=torch.float64)


def _counts(**rose: int) -> dict[str, int]:
    return {k: rose.get(k, 0) for k in COUNTS}


@pytest.mark.parametrize(
    ("rose", "names", "not_named"),
    [
        ({"cold": 1}, ["below the solver's 300 K domain", "No solver setting helps"], "--dt"),
        ({"bad_input": 1}, ["bad input", "inputs to the call are wrong"], "--dt"),
        ({"unfinished": 1, "recomputed": 1, "still_failed": 1}, ["1 out of steps", "a shorter --dt"], "inputs"),
    ],
)
def test_message_names_the_counted_cause_and_its_fix(rose, names, not_named):
    done = torch.tensor([True, False, True])
    with pytest.raises(UnfinishedCellsError) as err:
        require_finished(done, T, P, _counts(**rose), "at cycle 4")
    text = str(err.value)
    assert "1 chemistry cell(s) not done at cycle 4; the first, cell 1: T 280.0 K, P 300 Pa" in text
    assert f"The call's counts: {rose}" in text
    for name in names:
        assert name in text
    assert not_named not in text
    require_finished(torch.ones(3, dtype=torch.bool), T, P, _counts(cold=0), "at cycle 5")  # all done: no stop


def test_unexplained_stop_and_shapes():
    with pytest.raises(UnfinishedCellsError, match="no count of the solver explains it"):
        require_finished(torch.tensor([False, True, True]), T, P, _counts(), "at cycle 1")
    with pytest.raises(ValueError, match="must all be"):
        require_finished(torch.ones(2, dtype=torch.bool), T, P, _counts(), "at cycle 1")
