"""The stop on unfinished chemistry cells names the cause the solver counted, and only the fix that matches it."""

import pytest
import torch

from maxichem.coupling.chemistry import UnfinishedCellsError, require_finished
from maxichem.kincuda import COUNTS

T = torch.tensor([1500.0, 280.0, 1200.0], dtype=torch.float64)
P = torch.tensor([1.0e5, 3.0e2, 2.0e6], dtype=torch.float64)
# The counts that leave a cell not done: what the message must name, and the fix it gives
CAUSES = {
    "cold": ("below the solver's 300-3000 K domain (refused at 300 K)", "look at the deck"),
    "hot": ("above the solver's 300-3000 K domain (refused at 3000 K)", "look at the deck"),
    "bad_input": ("bad input", "inputs to the call are wrong"),
    "still_failed": ("still failing the guard after the recovery", "a shorter --dt"),
}
# The counts that do not by themselves: a cell non-finite, out of tries or drifting is recomputed, then counted as
# recovered or still failed; a non-finite starting step acts as dt
NOT_CAUSES = {"nonfinite", "unfinished", "drift", "recomputed", "recovered", "nonfinite_h"}


def _counts(**rose: int) -> dict[str, int]:
    return {k: rose.get(k, 0) for k in COUNTS}


def test_every_count_is_judged():
    assert set(COUNTS) == set(CAUSES) | NOT_CAUSES
    assert not set(CAUSES) & NOT_CAUSES


@pytest.mark.parametrize("count", COUNTS)
def test_message_names_the_counted_cause_and_only_its_fix(count):
    done = torch.tensor([True, False, True])
    with pytest.raises(UnfinishedCellsError) as err:
        require_finished(done, T, P, _counts(**{count: 1}), "at cycle 4")
    text = str(err.value)
    assert "1 chemistry cell(s) not done at cycle 4; the first, cell 1: T 280.0 K, P 300 Pa" in text
    assert f"The call's counts: { {count: 1} }" in text
    if count not in CAUSES:
        assert "no count of the solver explains it" in text
        return
    cause, fix = CAUSES[count]
    assert cause in text
    assert fix in text
    for other, (other_cause, other_fix) in CAUSES.items():
        if other != count:
            assert other_cause not in text
            assert other_fix == fix or other_fix not in text


def test_still_failed_names_what_the_recovery_could_not_mend():
    counts = _counts(unfinished=1, nonfinite=2, recomputed=3, still_failed=3)
    with pytest.raises(UnfinishedCellsError, match=r"in the call: 1 out of steps, 2 non-finite, 0 element drift"):
        require_finished(torch.tensor([True, False, True]), T, P, counts, "at cycle 4")


def test_done_cells_and_shapes():
    require_finished(torch.ones(3, dtype=torch.bool), T, P, _counts(cold=1), "at cycle 5")  # all done: no stop
    with pytest.raises(UnfinishedCellsError, match="no count of the solver explains it"):
        require_finished(torch.tensor([False, True, True]), T, P, _counts(), "at cycle 1")
    with pytest.raises(ValueError, match="must all be"):
        require_finished(torch.ones(2, dtype=torch.bool), T, P, _counts(), "at cycle 1")
