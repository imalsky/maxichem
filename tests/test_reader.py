"""The reader on the invented networks: rate laws, stoichiometry and refusals."""

import numpy as np
import pytest

from maxichem.kinvulcan import VulcanNetwork


def test_reads_every_rate_law(read_network):
    net = read_network()
    by_eq = {r["eq"].replace(" ", ""): r for r in net.reac}
    assert by_eq["H+OH->H2+O"]["kind"] == 2
    assert by_eq["H+CH3+M->CH4+M"]["troe"] == (0.6, 100.0, 2000.0, 5000.0)
    assert by_eq["H+OH+M->H2O+M"]["eff"] == 2.5  # the bulk gas is H2
    assert by_eq["O+O+M->O2+M"]["kind"] == 5
    assert by_eq["OH+CH3+M->CH3OH+M"]["width"] == "vm11"
    assert [r["absorber"] for r in net.photo] == ["H2O", "CH4", "O2"]
    assert net.skipped == 1  # the condensation line
    assert net.species[-1] == "He"  # no reaction names it


@pytest.mark.parametrize("stem", ["synthetic", "synthetic_falloff"])
def test_reactions_conserve_atoms(read_network, stem):
    net = read_network(stem)
    assert not np.any(net.atoms.T @ net.S)
    assert not np.any(net.atoms.T @ net.Sp)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("[ H + OH -> H2 + O ", "[ H + OH -> H2 + H "),  # atoms do not balance
        ("15 [ OH + CH3 + M -> CH3OH + M       ]", "15 [ O + CH3 + M -> CH3O + M       ]"),  # no law for it
        ("17 [ H2O -> H + OH                   ]  H2O  1", "17 [ H2O -> H + OH  ]  CH4  1"),  # another absorber
        ("19 [ CH4 -> CH3 + H                  ]  CH4  1", "19 [ H2O -> H + OH ]  H2O  1"),  # the same branch twice
        ("# photo disscoiation (no reversals)", "1 [ H + OH -> H2 + O ] 1.0 0.0 0.0\n# photo"),  # after reverse stops
        ("# special cases", "# irreversible\n25 [ H2 + O -> H + OH ] 1.0 0.0 0.0 1.0 0.0 0.0\n# special cases"),
    ],
)
def test_refuses_a_bad_network(tmp_path, network_text, nasa_dir, old, new):
    assert old in network_text
    path = tmp_path / "bad.txt"
    path.write_text(network_text.replace(old, new))
    with pytest.raises(ValueError, match=r"\S"):
        VulcanNetwork(path, nasa_dir)
