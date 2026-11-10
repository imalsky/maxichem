"""Tracer conversions and the start gas: a hand-worked mixture, elements kept through a reaction, the start's ratios."""

import pytest
import torch

from coupling.composition import (
    START_PER_H,
    element_moles,
    mass_to_mole,
    molar_masses,
    mole_to_mass,
    start_mole_fractions,
    tracer_totals,
)

# species and their atoms, columns H, He, C, N, O, S
ATOMS = {
    "H2": (2, 0, 0, 0, 0, 0),
    "He": (0, 1, 0, 0, 0, 0),
    "CO": (0, 0, 1, 0, 1, 0),
    "H2O": (2, 0, 0, 0, 1, 0),
    "O2": (0, 0, 0, 0, 2, 0),
    "CH4": (4, 0, 1, 0, 0, 0),
    "N2": (0, 0, 0, 2, 0, 0),
    "NH3": (3, 0, 0, 1, 0, 0),
    "H2S": (2, 0, 0, 0, 0, 1),
    "SO2": (0, 0, 0, 0, 2, 1),
}
ELEMENTS = ("H", "He", "C", "N", "O", "S")


def _network(species: list[str]) -> tuple[torch.Tensor, list[str]]:
    """Return the atoms table [S, E] of these species over the elements they hold, and those elements."""
    full = torch.tensor([ATOMS[s] for s in species], dtype=torch.float64)
    keep = [j for j in range(len(ELEMENTS)) if bool(full[:, j].any())]
    return full[:, keep], [ELEMENTS[j] for j in keep]


def test_hand_mixture_and_elements_kept_through_a_reaction():
    atoms, elements = _network(["H2", "He", "H2O", "O2"])
    mw = molar_masses(atoms, elements)
    assert torch.allclose(mw, torch.tensor([2.016e-3, 4.002602e-3, 18.015e-3, 31.998e-3], dtype=torch.float64))
    # x = (0.8, 0.15, 0.05, 0): mean molar mass 3.1139403 g/mol, so q_i = x_i mw_i / 3.1139403
    q = torch.tensor([[0.5179290046119381, 0.19280726094845174, 0.2892637344396102, 0.0]], dtype=torch.float64)
    x = mass_to_mole(q, mw)
    assert torch.allclose(x, torch.tensor([[0.8, 0.15, 0.05, 0.0]], dtype=torch.float64), rtol=0, atol=1e-15)

    # cells with tracer mass other than 1 kg/kg, and a reaction 2 H2 + O2 -> 2 H2O between the calls
    gen = torch.Generator().manual_seed(3)
    q = torch.rand((5, 4), generator=gen, dtype=torch.float64) * torch.tensor([1.0, 0.3, 0.1, 0.2])
    mass = q.sum(1)
    x = mass_to_mole(q, mw)
    assert torch.allclose(mole_to_mass(x, mw, mass), q, rtol=1e-14, atol=0)
    extent = 0.5 * torch.minimum(x[:, 0] / 2, x[:, 3])
    n = x.clone()
    n[:, 0] -= 2 * extent
    n[:, 3] -= extent
    n[:, 2] += 2 * extent
    q_out = mole_to_mass(n / n.sum(1, keepdim=True), mw, mass)
    assert torch.allclose(q_out.sum(1), mass, rtol=1e-14, atol=0)
    assert torch.allclose(element_moles(q_out, mw, atoms), element_moles(q, mw, atoms), rtol=1e-13, atol=0)
    volume_mass = torch.full((5,), 2.0, dtype=torch.float64)
    totals = tracer_totals(q_out, volume_mass, mw, atoms)
    assert torch.allclose(totals, tracer_totals(q, volume_mass, mw, atoms), rtol=1e-13, atol=0)
    assert float(totals[0]) == pytest.approx(2.0 * float(mass.sum()), rel=1e-14)


@pytest.mark.parametrize(
    "species",
    [
        ["H2", "He", "CO", "H2O", "CH4", "O2"],
        ["H2", "He", "CO", "H2O", "N2", "NH3", "CH4"],
        ["H2", "He", "CO", "H2O", "N2", "NH3", "H2S", "SO2"],
    ],
)
def test_start_element_ratios_equal_the_inputs(species):
    atoms, elements = _network(species)
    x = start_mole_fractions(species, elements)
    assert float(x.sum()) == pytest.approx(1.0, abs=1e-15)
    per_atom = x @ atoms
    h = per_atom[elements.index("H")]
    for e in elements:
        if e != "H":
            assert float(per_atom[elements.index(e)] / h) == pytest.approx(START_PER_H[e], rel=1e-14)


def test_start_refuses_a_missing_carrier():
    species = ["H2", "He", "CO", "H2O", "NH3"]  # nitrogen, but no N2 to start it in
    _, elements = _network(species)
    with pytest.raises(ValueError, match=r"N2 \(for N\)"):
        start_mole_fractions(species, elements)
    with pytest.raises(ValueError, match="no atomic mass for \\['Ti'\\]"):
        molar_masses(torch.ones((2, 2)), ["H", "Ti"])


def test_conversions_refuse_unusable_input():
    mw = torch.tensor([2.016e-3, 4.002602e-3], dtype=torch.float64)
    with pytest.raises(ValueError, match="negatives"):
        mass_to_mole(torch.tensor([[0.9, -1e-9]], dtype=torch.float64), mw)
    with pytest.raises(ValueError, match="non-finite"):
        mass_to_mole(torch.tensor([[0.9, float("nan")]], dtype=torch.float64), mw)
    with pytest.raises(ValueError, match="cell 1 holds no tracer mass"):
        mass_to_mole(torch.tensor([[0.9, 0.1], [0.0, 0.0]], dtype=torch.float64), mw)
    one = torch.ones(1, dtype=torch.float64)
    with pytest.raises(ValueError, match="non-finite"):
        mole_to_mass(torch.tensor([[float("inf"), 0.1]], dtype=torch.float64), mw, one)
    with pytest.raises(ValueError, match="cell 0 has mole fractions of no mass"):
        mole_to_mass(torch.zeros((1, 2), dtype=torch.float64), mw, one)
    with pytest.raises(ValueError, match=r"mass must be \[1\]"):
        mole_to_mass(torch.ones((1, 2), dtype=torch.float64), mw, torch.ones(2, dtype=torch.float64))
    with pytest.raises(ValueError, match=r"q must be \[cells, 2\]"):
        mass_to_mole(torch.ones((1, 3), dtype=torch.float64), mw)
    with pytest.raises(ValueError, match="atoms has 3 species rows"):
        element_moles(torch.ones((1, 2), dtype=torch.float64), mw, torch.ones((3, 1), dtype=torch.float64))
    with pytest.raises(ValueError, match=r"cell_mass must be \[1\]"):
        tracer_totals(torch.ones((1, 2), dtype=torch.float64), one.repeat(2), mw, torch.ones((2, 1)))
    with pytest.raises(ValueError, match=r"atoms must be \[species, 2\]"):
        molar_masses(torch.ones((2, 3)), ["H", "He"])
    with pytest.raises(ValueError, match=r"species rows \[1\] have no atoms"):
        molar_masses(torch.tensor([[2.0, 0.0], [0.0, 0.0]]), ["H", "He"])
    with pytest.raises(ValueError, match=r"elements \['Ti'\] with no start carrier"):
        start_mole_fractions(["H2", "He", "CO", "H2O", "TiO"], ["H", "He", "C", "O", "Ti"])
