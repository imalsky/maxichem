"""The column heating on a sphere: no false divergence, the right sign, and an energy budget that closes."""

import torch

from coupling.radiation import column_heating, column_residual

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
