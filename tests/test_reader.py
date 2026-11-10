"""The reader on the invented networks: rate laws, stoichiometry and refusals."""

import shutil

import numpy as np
import pytest

from maxichem import VulcanNetwork, _tables, kinvulcan, paths

VULCAN_NETWORKS = {
    "CHO_photo_network.txt",
    "CHO_photo_network_lowT.txt",
    "NCHO_full_photo_network.txt",
    "NCHO_photo_network.txt",
    "NCHO_photo_network_lowT_Jupiter.txt",
    "NCHO_thermo_network.txt",
    "SNCHO_DMS_photo_network_Tsai2024.txt",
    "SNCHO_photo_network.txt",
    "SNCHO_photo_network_2025.txt",
    "SNCHO_photo_network_C3.txt",
    "TiSNCHO_photo_network.txt",
}


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
    ("old", "new", "message"),
    [
        ("[ H + OH -> H2 + O ", "[ H + OH -> H2 + H ", "do not balance"),
        ("15 [ OH + CH3 + M -> CH3OH + M       ]", "15 [ O + CH3 + M -> CH3O + M       ]", "unknown special reaction"),
        ("17 [ H2O -> H + OH                   ]  H2O  1", "17 [ H2O -> H + OH  ]  CH4  1", "not its absorber"),
        ("19 [ CH4 -> CH3 + H                  ]  CH4  1", "19 [ H2O -> H + OH ]  H2O  1", "two photolysis lines"),
        (
            "# photo disscoiation (no reversals)",
            "1 [ H + OH -> H2 + O ] 1.0 0.0 0.0\n# photo",
            "after '# reverse stops'",
        ),
        (
            "# special cases",
            "# irreversible\n25 [ H2 + O -> H + OH ] 1.0 0.0 0.0 1.0 0.0 0.0\n# special cases",
            "no reverse law",
        ),
        ("# Two-body Reactions", "1 [ H + OH -> H2 + O ] 1.0 0.0 0.0\n# Two-body Reactions", "no rate law"),
        ("# 3-body and Disscoiation Reactions", "# Three-body Reactions", "unknown heading"),
        ("[ H + OH -> H2 + O ", "[ H + OH -> H2 + O ] -1.0 0.0 0.0 #", "negative Arrhenius"),
        ("[ H + OH -> H2 + O ", "[ H + OH -> H2 + O ] nan 0.0 0.0 #", "finite rate coefficients"),
        ("[ H + OH -> H2 + O ", "[ H + OH -> H2 + O ] inf 0.0 0.0 #", "finite rate coefficients"),
        ("Troe [0.6, 100.0,", "Troe [1e999, 100.0,", "finite rate coefficients"),
        ("effi {'H2': 2.5,", "effi {'H2': 1e999,", "finite rate coefficients"),
        ("Troe [0.6, 100.0, 2000.0, 5000.0]", "Troe [0.6, 100.0]", "holds 4 numbers"),
        ("Troe [0.6, 100.0, 2000.0, 5000.0]", "Troe [0.6, 100.0", "cannot be read"),
        ("effi {'H2': 2.5,", "effi {'H2': 2.5", "cannot be read"),
        ("]  H2O  1", "]  H2O  one", "must be an integer"),
        ("[ O + O + M -> O2 + M ", "[ O + O + M -> O2 ", r"bad.txt:\d+: M on reactant side, .* on both sides"),
        ("[ H + H + M -> H2 + M ", "[ H + H -> H2 + M ", r"bad.txt:\d+: M on product side, .* on both sides"),
        ("[ H + H + M -> H2 + M ", "[ H + H -> H2 ", "M on no side, .* on both sides"),
        ("[ H + OH -> H2 + O ", "[ H + OH + M -> H2 + O + M ", "M on reactant and product side, .* on neither side"),
    ],
)
def test_refuses_a_bad_network(tmp_path, network_text, nasa_dir, old, new, message):
    assert old in network_text
    path = tmp_path / "bad.txt"
    path.write_text(network_text.replace(old, new))
    with pytest.raises(ValueError, match=message):
        _tables.tables(VulcanNetwork(path, nasa_dir), 2)


def test_repeated_species_are_refused(tmp_path, nasa_dir, read_network, network_text):
    network = tmp_path / "network.txt"
    network.write_text(network_text)
    species = read_network().species
    with pytest.raises(ValueError, match=rf"\['{species[0]}'\] more than once"):
        VulcanNetwork(network, nasa_dir, species=[*species, species[0]])


def test_keeps_free_text_above_the_first_heading_and_titles_below_it(tmp_path, network_text, nasa_dir, read_network):
    path = tmp_path / "commented.txt"
    head = "# free text: rates updated\n# Two-body Reactions\n# id  [ reaction ]\n#\n# 1 [ H + O ]\n"
    path.write_text(network_text.replace("# Two-body Reactions\n", head))
    assert len(VulcanNetwork(path, nasa_dir).reac) == len(read_network().reac)


def _nasa_with_jump(tmp_path, nasa_dir, name, jump):
    """Return a NASA-9 folder whose species has its low range's b2 moved, so G / RT jumps by this much at 1000 K."""
    folder = tmp_path / "NASA9"
    shutil.copytree(nasa_dir, folder)
    numbers = np.loadtxt(nasa_dir / f"{name}.txt").flatten()
    numbers[9] -= jump  # G / RT = H / RT - S / R, and b2 enters S / R
    np.savetxt(folder / f"{name}.txt", numbers.reshape(-1, 5))
    return folder


def test_gibbs_jump_is_refused_above_the_limit(tmp_path, nasa_dir, network_text):
    network = tmp_path / "network.txt"
    network.write_text(network_text)
    VulcanNetwork(network, _nasa_with_jump(tmp_path / "under", nasa_dir, "H2O", 0.9 * kinvulcan.MAX_GIBBS_JUMP))
    with pytest.raises(ValueError, match=f"jumps by {1.1 * kinvulcan.MAX_GIBBS_JUMP:.3g}"):
        VulcanNetwork(network, _nasa_with_jump(tmp_path / "over", nasa_dir, "H2O", 1.1 * kinvulcan.MAX_GIBBS_JUMP))


def test_free_comments_past_the_thermal_sections_are_read(tmp_path, nasa_dir, network_text):
    heading = "# photo disscoiation (no reversals)"
    path = tmp_path / "network.txt"
    path.write_text(network_text.replace(heading, f"{heading}\n# wavelength switch\n# branching info start"))
    assert [r["absorber"] for r in VulcanNetwork(path, nasa_dir).photo] == ["H2O", "CH4", "O2"]


def test_nasa_file_with_four_ranges_reads_its_first_two(tmp_path, nasa_dir, read_network, network_text):
    network = tmp_path / "network.txt"
    network.write_text(network_text)
    folder = tmp_path / "NASA9"
    shutil.copytree(nasa_dir, folder)
    numbers = np.loadtxt(nasa_dir / "H2O.txt").flatten()[:20]
    np.savetxt(folder / "H2O.txt", np.r_[numbers, np.full(20, 7.0)].reshape(-1, 5))
    assert np.array_equal(VulcanNetwork(network, folder).lo, read_network().lo)


@pytest.mark.slow  # marked for the data, not the time: CI does not have VULCAN's files
def test_only_the_network_with_the_unsound_tih_fit_is_refused():
    thermo = paths.vulcan_thermo()
    networks = list(thermo.glob("*network*.txt"))
    assert {network.name for network in networks} >= VULCAN_NETWORKS
    for network in networks:
        if network.name == "TiSNCHO_photo_network.txt":
            with pytest.raises(ValueError, match=r"TiH\.txt: G / RT jumps by 0\.143"):
                VulcanNetwork(network, thermo / "NASA9")
        else:
            VulcanNetwork(network, thermo / "NASA9")


def test_nasa_file_of_another_size_is_refused(tmp_path, nasa_dir, network_text):
    network = tmp_path / "network.txt"
    network.write_text(network_text)
    folder = tmp_path / "NASA9"
    shutil.copytree(nasa_dir, folder)
    (folder / "H2O.txt").write_text("1.0 2.0 3.0\n")
    with pytest.raises(ValueError, match="numbers"):
        VulcanNetwork(network, folder)
