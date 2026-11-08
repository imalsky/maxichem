"""Radiative heating of a column from two-stream fluxes, and the check that it conserves the column's energy.

A two-stream solver gives fluxes per unit area of a plane-parallel column, so the heating is their divergence along
the column, -d(F_up - F_down)/dz, also on a sphere: weighting the fluxes by face areas adds a false 2 F / r per
volume, which grows as 1 / rho per mass and dominates near the lid.
"""

import torch

__all__ = ["column_heating", "column_residual"]


def column_heating(net_up: torch.Tensor, dz: torch.Tensor) -> torch.Tensor:
    """Return the radiative heating of each layer from the net upward flux on its faces.

    Parameters
    ----------
    net_up : torch.Tensor [ncol, nl + 1]
        Net upward flux F_up - F_down on the layer faces, bottom face first, W m^-2.
    dz : torch.Tensor [ncol, nl] or [nl]
        Layer thicknesses, m.

    Returns
    -------
    torch.Tensor [ncol, nl]
        Heating per volume, W m^-3.
    """
    if net_up.dim() != 2 or net_up.shape[1] < 2:
        raise ValueError(f"net_up must be [ncol, nl + 1] with nl >= 1, got shape {tuple(net_up.shape)}")
    if dz.shape[-1] != net_up.shape[1] - 1:
        raise ValueError(f"dz has {dz.shape[-1]} layers, net_up has faces for {net_up.shape[1] - 1}")
    if not bool((dz > 0).all()):
        raise ValueError("dz must be positive everywhere")
    return -(net_up[:, 1:] - net_up[:, :-1]) / dz


def column_residual(heat: torch.Tensor, dz: torch.Tensor, net_up: torch.Tensor) -> torch.Tensor:
    """Return each column's energy budget error: the heating summed over the column minus the net flux it takes in.

    With ``column_heating`` it is round-off; a heating that is not the flux divergence shows up here.

    Parameters
    ----------
    heat : torch.Tensor [ncol, nl]
        Heating per volume, W m^-3.
    dz : torch.Tensor [ncol, nl] or [nl]
        Layer thicknesses, m.
    net_up : torch.Tensor [ncol, nl + 1]
        Net upward flux on the layer faces, bottom face first, W m^-2.

    Returns
    -------
    torch.Tensor [ncol]
        sum(heat * dz) - (net_up[bottom] - net_up[top]), W m^-2.
    """
    return (heat * dz).sum(-1) - (net_up[:, 0] - net_up[:, -1])
