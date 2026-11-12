"""Saved flows and start columns load only into the deck they were made for, and old files without a record never."""

import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from maxichem.coupling.flow_file import (
    DryGas,
    FlowFileError,
    load_flow,
    load_start_column,
    save_flow,
    save_start_column,
)

DECK_GAS = DryGas({"H": 1.68, "He": 0.16}, 2.62, 2.3338643e-3)  # the decks' H2 and He gas
H2_GAS = DryGas({"H": 2.0}, 2.5, 2.016e-3)  # the gas of the flows saved before records
GRID = {"type": "cartesian", "bounds": {"x1min": 0.0, "x1max": 6.0e6}, "cells": {"nx1": 4, "nx2": 2, "nx3": 2}}
CPU = torch.device("cpu")


def test_flow_loads_only_into_its_own_deck(tmp_path):
    w = [torch.arange(5 * 2 * 3 * 4, dtype=torch.float64).reshape(5, 2, 3, 4)]
    shapes = [tuple(w[0].shape)]
    path = tmp_path / "flow.pt"
    save_flow(path, w, 700.0, DECK_GAS, GRID)
    flow = load_flow(path, DECK_GAS, GRID, shapes, CPU)
    assert torch.equal(flow.hydro_w[0], w[0])
    assert flow.tsim_s == 700.0
    with pytest.raises(FlowFileError, match=r"composition: file \{'H': 1.68, 'He': 0.16\}, deck \{'H': 2.0\}"):
        load_flow(path, H2_GAS, GRID, shapes, CPU)
    # each field of the gas alone
    for other, field in (
        (DryGas({"H": 1.68, "He": 0.17}, 2.62, 2.3338643e-3), "composition"),
        (DryGas({"H": 1.68, "He": 0.16}, 2.5, 2.3338643e-3), "cv_R"),
        (DryGas({"H": 1.68, "He": 0.16}, 2.62, 2.016e-3), "molar_mass"),
    ):
        with pytest.raises(FlowFileError, match=rf"another deck \({field}: file") as err:
            load_flow(path, other, GRID, shapes, CPU)
        assert str(err.value).count(": file ") == 1
    with pytest.raises(FlowFileError, match="grid: file"):
        load_flow(path, DECK_GAS, {**GRID, "cells": {"nx1": 8, "nx2": 2, "nx3": 2}}, shapes, CPU)
    with pytest.raises(FlowFileError, match="blocks of shapes"):
        load_flow(path, DECK_GAS, GRID, [(5, 2, 3, 5)], CPU)
    old = tmp_path / "old.pt"
    torch.save({"hydro_w": w, "tsim": 700.0}, old)  # the format before records
    with pytest.raises(FlowFileError, match=r"does not record the gas and grid.*Re-spin the flow"):
        load_flow(old, DECK_GAS, GRID, shapes, CPU)


def test_start_column_loads_only_for_its_own_gas(tmp_path):
    grid = {"nz": 4, "depth_m": 6.0e6}
    levels = np.linspace(1800.0, 900.0, 5)
    path = tmp_path / "column.npz"
    save_start_column(path, levels, DECK_GAS, grid)
    assert np.array_equal(load_start_column(path, DECK_GAS, grid), levels)
    with pytest.raises(FlowFileError, match=r"cv_R: file 2\.62, deck 2\.5"):
        load_start_column(path, H2_GAS, grid)
    old = tmp_path / "old.npz"
    np.savez(old, L=levels, T=levels[1:], P=levels[1:])
    with pytest.raises(FlowFileError, match=r"does not record.*re_start\.py"):
        load_start_column(old, DECK_GAS, grid)
    with pytest.raises(ValueError, match="molar_mass must be finite and positive, got nan"):
        DryGas({"H": 2.0}, 2.5, float("nan"))
    with pytest.raises(FlowFileError, match=r"4 layers need 5 level temperatures, got shape \(2,\)"):
        save_start_column(tmp_path / "short.npz", levels[:2], DECK_GAS, grid)
    with pytest.raises(FlowFileError, match="integer nz >= 1"):
        save_start_column(tmp_path / "no_nz.npz", levels, DECK_GAS, {"depth_m": 6.0e6})
    with pytest.raises(FlowFileError, match="finite and positive"):
        save_start_column(tmp_path / "nan.npz", np.full(5, np.nan), DECK_GAS, grid)
    # a file whose record says 4 layers but holds 2 levels, as a hand-made or truncated file would
    short = tmp_path / "short_on_disk.npz"
    save_start_column(short, levels, DECK_GAS, grid)
    with np.load(short) as f:
        record = f["record"]
    np.savez(short, L=levels[:2], record=record)
    with pytest.raises(FlowFileError, match=r"4 layers need 5 level temperatures, got shape \(2,\)"):
        load_start_column(short, DECK_GAS, grid)


def test_start_column_grid_with_a_tuple_loads(tmp_path):
    # the column's record goes through JSON, which reads a tuple back as a list
    grid = {"nz": 4, "faces_m": (0.0, 6.0e6)}
    levels = np.linspace(1800.0, 900.0, 5)
    path = tmp_path / "column.npz"
    save_start_column(path, levels, DECK_GAS, grid)
    assert np.array_equal(load_start_column(path, DECK_GAS, grid), levels)
    with pytest.raises(FlowFileError, match="grid: file"):
        load_start_column(path, DECK_GAS, {"nz": 4, "faces_m": (0.0, 7.0e6)})


@pytest.mark.parametrize(
    "record",
    [{"gas": {"composition": {"H": 2.0}}, "grid": GRID}, {"gas": asdict(DECK_GAS)}, ["gas"]],
    ids=["gas without cv_R", "no grid", "not a mapping"],
)
def test_malformed_record_is_refused(tmp_path, record):
    w = [torch.zeros(2, 3, dtype=torch.float64)]
    flow = tmp_path / "flow.pt"
    torch.save({"hydro_w": w, "tsim": 700.0, "record": record}, flow)
    with pytest.raises(FlowFileError, match="malformed record"):
        load_flow(flow, DECK_GAS, GRID, [(2, 3)], CPU)
    column = tmp_path / "column.npz"
    np.savez(column, L=np.linspace(1800.0, 900.0, 5), record=np.array(json.dumps(record)))
    with pytest.raises(FlowFileError, match="malformed record"):
        load_start_column(column, DECK_GAS, {"nz": 4})


def test_record_with_a_bad_molar_mass_or_not_json_is_refused(tmp_path):
    record = {"gas": {**asdict(DECK_GAS), "molar_mass": "x"}, "grid": GRID}
    flow = tmp_path / "flow.pt"
    torch.save({"hydro_w": [torch.zeros(2, 3, dtype=torch.float64)], "tsim": 700.0, "record": record}, flow)
    with pytest.raises(FlowFileError, match="record's molar_mass is 'x', not a number"):
        load_flow(flow, DECK_GAS, GRID, [(2, 3)], CPU)
    column = tmp_path / "column.npz"
    np.savez(column, L=np.linspace(1800.0, 900.0, 5), record=np.array("{not json"))
    with pytest.raises(FlowFileError, match="record is not JSON"):
        load_start_column(column, DECK_GAS, {"nz": 4})


def test_grid_with_numpy_numbers_saves_and_loads(tmp_path):
    # a deck read with numpy may hold numpy scalars; they record and compare as the numbers they are
    grid = {"nz": np.int64(4), "depth_m": np.float64(6.0e6), "faces": (np.int64(0), 4)}
    levels = np.linspace(1800.0, 900.0, 5)
    column = tmp_path / "column.npz"
    save_start_column(column, levels, DECK_GAS, grid)
    assert np.array_equal(load_start_column(column, DECK_GAS, grid), levels)
    assert np.array_equal(load_start_column(column, DECK_GAS, {"nz": 4, "depth_m": 6.0e6, "faces": [0, 4]}), levels)
    flow_grid = {**GRID, "cells": {"nx1": np.int64(4), "nx2": 2, "nx3": 2}}
    w = [torch.zeros(2, 3, dtype=torch.float64)]
    flow = tmp_path / "flow.pt"
    save_flow(flow, w, 700.0, DECK_GAS, flow_grid)
    assert torch.equal(load_flow(flow, DECK_GAS, flow_grid, [(2, 3)], CPU).hydro_w[0], w[0])
    with pytest.raises(FlowFileError, match="grid: file"):
        load_flow(flow, DECK_GAS, {**GRID, "cells": {"nx1": np.int64(8), "nx2": 2, "nx3": 2}}, [(2, 3)], CPU)
