"""The column heating on a sphere, its budget check, the level temperatures and the radiation's schedule."""

import math

import pytest
import torch

from coupling.radiation import (
    RadiationBudgetError,
    RadiationClock,
    check_budget,
    column_heating,
    column_residual,
    level_temperatures,
)

R_PLANET = 9.4e7  # m, the hot-Jupiter deck's radius
NL = 140  # layers of the deck that crashed with the area-weighted form


def _sphere_column() -> tuple[torch.Tensor, torch.Tensor]:
    """Return face radii [NL + 1] and layer thicknesses [NL] of a 6000 km column, m."""
    r = torch.linspace(R_PLANET, R_PLANET + 6.0e6, NL + 1, dtype=torch.float64)
    return r, r[1:] - r[:-1]


def test_height_constant_flux_gives_no_heating_on_a_sphere():
    r, dz = _sphere_column()
    net_up = torch.full((3, NL + 1), 2.5e5, dtype=torch.float64)
    heat = column_heating(net_up, dz)
    assert float(heat.abs().max()) == 0.0
    # the area-weighted form this replaces gives -2 F / r per volume here
    area = r**2
    volume = (r[1:] ** 3 - r[:-1] ** 3) / 3
    old = -(area[1:] * net_up[:, 1:] - area[:-1] * net_up[:, :-1]) / volume
    assert float(old.abs().max()) > 1.0e-3
    assert float(column_residual(old, dz, net_up).abs().max()) > 1.0e3


def test_heating_sign_and_column_budget():
    _, dz = _sphere_column()
    z = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.float64), dz]), 0)
    # net upward flux growing with height: each layer loses energy, so it cools
    net_up = 1.0e3 + 50.0 * (z / z[-1]) ** 2 + torch.zeros(2, 1, dtype=torch.float64)
    heat = column_heating(net_up, dz)
    assert bool((heat < 0).all())
    budget = column_residual(heat, dz, net_up)
    assert float(budget.abs().max()) < 1.0e-9 * float(net_up.abs().max())


def test_budget_check_refuses_nan_and_a_heating_off_the_divergence():
    r, dz = _sphere_column()
    z = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.float64), dz]), 0)
    net_up = 1.0e3 + 50.0 * (z / z[-1]) ** 2 + torch.zeros(3, 1, dtype=torch.float64)
    heat = column_heating(net_up, dz)
    assert check_budget(heat, dz, net_up) < 1.0e-13
    nan_flux = net_up.clone()
    nan_flux[1, 7] = float("nan")
    with pytest.raises(RadiationBudgetError, match="the first column 1"):
        check_budget(column_heating(nan_flux, dz), dz, nan_flux)
    area = r**2  # the area-weighted form: off the divergence by 2 F / r
    old = -(area[1:] * net_up[:, 1:] - area[:-1] * net_up[:, :-1]) / ((r[1:] ** 3 - r[:-1] ** 3) / 3)
    with pytest.raises(RadiationBudgetError, match="not the flux divergence"):
        check_budget(old, dz, net_up)


def test_level_temperatures_exact_for_a_linear_profile():
    dz = 6.0e4
    z_centre = dz * (torch.arange(NL, dtype=torch.float64) + 0.5)
    z_face = dz * torch.arange(NL + 1, dtype=torch.float64)
    slope = torch.tensor([[-2.0e-4], [3.0e-4]], dtype=torch.float64)  # K/m: cooling and warming upward
    levels = level_temperatures(1500.0 + slope * z_centre)
    assert torch.allclose(levels, 1500.0 + slope * z_face, rtol=1e-14, atol=0)


def test_radiation_clock_keeps_a_fixed_schedule():
    dt, period = 70.0, 300.0
    clock = RadiationClock(period)
    fired = [n * dt for n in range(60) if clock.due(n * dt)]
    # the first step time at or after each multiple of the period
    expected = [dt * math.ceil(period * k / dt) for k in range(15) if period * k <= 59 * dt]
    assert fired == expected
    assert fired[:9] == [0.0, 350.0, 630.0, 910.0, 1260.0, 1540.0, 1820.0, 2100.0, 2450.0]
    long_steps = RadiationClock(period)
    assert all(long_steps.due(n * 700.0) for n in range(5))  # a step longer than the period fires every step


def test_refusals_name_the_value():
    net_up = torch.zeros((2, 4), dtype=torch.float64)
    cases = [
        (lambda: column_heating(net_up[:, :1], torch.ones(0)), r"net_up must be \[ncol, nl \+ 1\]"),
        (lambda: column_heating(net_up, torch.ones(2)), "dz has 2 layers"),
        (lambda: column_heating(net_up, -torch.ones(3)), "dz must be positive"),
        (lambda: level_temperatures(torch.ones((2, 1))), r"nl >= 2, got shape \(2, 1\)"),
        (lambda: RadiationClock(0.0), "finite and positive, got 0.0 s"),
        (lambda: RadiationClock(300.0).due(float("nan")), "must be finite, got nan s"),
    ]
    for call, message in cases:
        with pytest.raises(ValueError, match=message):
            call()
