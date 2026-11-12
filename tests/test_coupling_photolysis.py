"""The stand-in photolysis lookup: nearest layer in log pressure, dark at night, cells beyond the table counted."""

import pytest
import torch

from maxichem.coupling.photolysis import nearest_layer_rates

TABLE_P = torch.tensor([1.0e1, 1.0e3, 1.0e5], dtype=torch.float64)  # Pa
TABLE_J = torch.tensor([[3.0, 30.0], [2.0, 20.0], [1.0, 10.0]], dtype=torch.float64)  # s^-1


def test_nearest_layer_dark_night_and_counted_edges():
    P = torch.tensor([1.0, 0.5, 15.0, 3.0e2, 1.0e5, 1.0e7, 1.0e8], dtype=torch.float64)
    day = torch.tensor([True, False, True, True, True, True, False])
    out = nearest_layer_rates(P, TABLE_P, TABLE_J, day)
    rows = [0, 0, 0, 1, 2, 2, 2]  # log10 P 2.48 is nearest log10 3
    expected = TABLE_J[rows] * day[:, None]
    assert torch.equal(out.J, expected)
    # day side: 1 Pa above the top row, 1e7 Pa below the bottom; the night cells take no row and are not counted
    assert (out.above, out.below) == (1, 1)
    with pytest.raises(ValueError, match="must both be"):
        nearest_layer_rates(P, TABLE_P, TABLE_J, day[:3])
    with pytest.raises(ValueError, match="disagree"):
        nearest_layer_rates(P, TABLE_P[:2], TABLE_J, day)
    with pytest.raises(ValueError, match=r"pressure must be finite and positive; its least value is 0\.0 Pa"):
        nearest_layer_rates(P * 0.0, TABLE_P, TABLE_J, day)
    nan_table = TABLE_J.clone()
    nan_table[0, 0] = float("nan")  # would reach the night side as NaN * 0
    with pytest.raises(ValueError, match="table_rates must be finite and not negative"):
        nearest_layer_rates(P, TABLE_P, nan_table, day)
    with pytest.raises(ValueError, match="table_rates must be finite and not negative"):
        nearest_layer_rates(P, TABLE_P, -TABLE_J, day)
    with pytest.raises(ValueError, match="day must be a boolean mask"):  # a 2 in an integer mask doubled J
        nearest_layer_rates(P, TABLE_P, TABLE_J, day.long() * 2)
    with pytest.raises(ValueError, match="the table has no layers"):
        nearest_layer_rates(P, TABLE_P[:0], TABLE_J[:0], day)
