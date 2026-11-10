"""The column heating on a sphere, its budget check, the level temperatures and the radiation's schedule."""

import math
from fractions import Fraction

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
    # finite heating, fluxes and thicknesses whose budget overflows
    with pytest.raises(RadiationBudgetError, match="column 0 has a non-finite budget residual"):
        check_budget(
            torch.tensor([[1.0e300, 0.0]], dtype=torch.float64),
            torch.tensor([1.0e10, 1.0], dtype=torch.float64),
            torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64),
        )


def test_level_temperatures_exact_for_a_linear_profile():
    dz = 6.0e4
    z_centre = dz * (torch.arange(NL, dtype=torch.float64) + 0.5)
    z_face = dz * torch.arange(NL + 1, dtype=torch.float64)
    slope = torch.tensor([[-2.0e-4], [3.0e-4]], dtype=torch.float64)  # K/m: cooling and warming upward
    levels = level_temperatures(1500.0 + slope * z_centre)
    assert torch.allclose(levels, 1500.0 + slope * z_face, rtol=1e-14, atol=0)


def _fired_steps(step_s: float, period_s: float, steps: int) -> list[int]:
    """Return the steps at which the radiation fires, driven as the drivers drive the clock."""
    clock, fired = RadiationClock(period_s), []
    for n in range(steps):
        if clock.due():
            fired.append(n)
        clock.advance(step_s)
    return fired


@pytest.mark.parametrize(
    ("step_s", "period_s", "steps"),
    [(70.0, 300.0, 200_000), (70.1, 300.0, 200_000), (0.1, 300.0, 500_000), (35.7, 300.1, 200_000)],
)
def test_radiation_clock_fires_at_the_first_step_after_each_multiple(step_s, period_s, steps):
    # the decimal step and period, as written: every multiple k P fires at step ceil(k P / dt)
    dt, period = Fraction(repr(step_s)), Fraction(repr(period_s))
    last = math.floor((steps - 1) * dt / period)
    expected = sorted({math.ceil(k * period / dt) for k in range(last + 1)})
    assert _fired_steps(step_s, period_s, steps) == expected
    if step_s == 70.0:
        assert [70 * n for n in expected[:9]] == [0, 350, 630, 910, 1260, 1540, 1820, 2100, 2450]
    assert _fired_steps(700.0, 300.0, 5) == [0, 1, 2, 3, 4]  # a step longer than the period fires every step


def test_refusals_name_the_value():
    net_up = torch.zeros((2, 4), dtype=torch.float64)
    cases = [
        (lambda: column_heating(net_up[:, :1], torch.ones(0)), r"net_up must be \[ncol, nl \+ 1\]"),
        (lambda: column_heating(net_up, torch.ones(2)), "dz has 2 layers"),
        (lambda: column_heating(net_up, -torch.ones(3)), "dz must be positive"),
        (lambda: level_temperatures(torch.ones((2, 1))), r"nl >= 2, got shape \(2, 1\)"),
        (lambda: RadiationClock(0.0), "finite and positive, got 0.0 s"),
        (lambda: RadiationClock(300.0).advance(float("nan")), "must be finite and not negative, got nan s"),
        (lambda: check_budget(net_up[:, 1:], torch.tensor([float("inf"), 1.0, 1.0]).double(), net_up), "dz must be"),
        (lambda: check_budget(net_up[:, 1:].float(), torch.ones(3).double(), net_up), "bound is for float64"),
        (lambda: check_budget(net_up, torch.ones(3).double(), net_up), r"heat \[ncol, nl\]"),
    ]
    for call, message in cases:
        with pytest.raises(ValueError, match=message):
            call()
