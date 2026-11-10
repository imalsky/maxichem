"""Read VULCAN's reaction lists and NASA-9 files as data.

The rate laws are in docs/method.md, the object's attributes in docs/api.md. Each reaction dict holds its law: kind (2
two-body, 3 falloff, 5 low-pressure only, 6 special), the Arrhenius terms, and optional troe, width and eff.
"""

import ast
import re
from pathlib import Path
from typing import Any

import numpy as np

from ._constants import KB, KIND_FALLOFF, KIND_LOW_PRESSURE, KIND_SPECIAL, KIND_TWO_BODY, NASA_SPLIT_T, P0

__all__ = ["KB", "MAX_GIBBS_JUMP", "P0", "SPECIAL", "VulcanNetwork", "formula"]

# Largest |G/RT, low range - G/RT, high range| at NASA_SPLIT_T. Sound fits meet: over the 11 networks the largest jump
# is 1.6e-6 (TiC). TiH, at 0.143, is refused on purpose, and TiSNCHO with it: below 1000 K its fit has Cp / R from -36
# to 819 (notes.md). A file with a wrong coefficient set reaches 17.6 (HNO3 with its below-200 K set read as the low
# range).
MAX_GIBBS_JUMP = 1.0e-5
TROE_FIELDS = 4  # the a, T3, T1, T2 of a "Troe [...]" line
NASA_FILE_SIZES = (20, 30, 40)  # numbers in a NASA-9 file: two to four ranges of 10 (those past the second are unused)


def formula(name: str) -> dict[str, int]:
    """Return the element counts of a species name, summed over repeats (CH3OH: C 1, H 4, O 1)."""
    c: dict[str, int] = {}
    for e, n in re.findall(r"([A-Z][a-z]?)(\d*)", name):
        c[e] = c.get(e, 0) + int(n or 1)
    return c


# laws not in the reaction list (VULCAN's code holds them), by equation without spaces
SPECIAL = {
    "OH+CH3+M->CH3OH+M": {
        "A0": 1.932e3,
        "B0": -9.88,
        "C0": 7544.0,
        "A1": 5.109e-11,
        "B1": -6.25,
        "C1": 1433.0,
        "Ai": 1.031e-10,
        "Bi": -0.018,
        "Ci": -16.74,
        "troe": (0.8145, 155.8, 1675.0, 4531.0),
        "width": "vm11",
    }
}
STOP = ("# reverse stops", "# condensation", "# photo", "# re_end")
# Headings of the thermal sections, in the order they are tested: one heading starts with another.
SECTIONS = (
    ("# two-body", "two"),
    ("# 3-body reactions without high-pressure", "low"),
    ("# 3-body", "three"),
    ("# special", "special"),
    ("# irreversible", "irrev"),
)
KNOWN_HEADINGS = (*STOP, *(head for head, _ in SECTIONS))
COLUMN_TITLES = "# id"  # the comment line above a section's reactions, which names its columns


def _reaction_line(line: str, where: str) -> tuple[str, list[str]]:
    """Return a reaction line's equation and the fields after it: ``id [ A + B -> C ] fields``."""
    try:
        equation, fields = line.split("[", 1)[1].split("]", 1)
    except IndexError:
        raise ValueError(f"{where}: a reaction line needs [ reaction ]: {line}") from None
    if "->" not in equation:
        raise ValueError(f"{where}: a reaction needs ->: {line}")
    return equation.strip(), fields.split()


def _finite(values: list[float], where: str, line: str) -> list[float]:
    """Return rate coefficients that are all finite; a NaN prefactor would pass as a zero rate."""
    if not np.isfinite(values).all():
        raise ValueError(f"{where}: finite rate coefficients expected, got {values}: {line}")
    return values


def _numbers(fields: list[str], count: int, where: str, line: str) -> list[float]:
    """Return the first count fields as numbers."""
    try:
        values = [float(v) for v in fields[:count]]
    except ValueError:
        raise ValueError(f"{where}: {count} rate coefficients expected: {line}") from None
    if len(values) < count:
        raise ValueError(f"{where}: {count} rate coefficients expected, found {len(values)}: {line}")
    return _finite(values, where, line)


def _photolysis_branch(line: str, where: str, photo: list[dict[str, Any]]) -> dict[str, Any]:
    """Return one photolysis branch: ``id [ A -> P1 + P2 ] A branch``."""
    equation, fields = _reaction_line(line, where)
    if len(fields) < 2:
        raise ValueError(f"{where}: a photolysis line needs its absorber and branch index: {line}")
    lhs, rhs = ([q.strip() for q in side.split("+")] for side in equation.split("->"))
    if lhs != [fields[0]]:
        raise ValueError(f"{where}: a photolysis line whose reactant is not its absorber {fields[0]}: {line}")
    try:
        branch = int(fields[1])
    except ValueError:
        raise ValueError(f"{where}: a photolysis branch index must be an integer: {line}") from None
    if (fields[0], branch) in {(r["absorber"], r["branch"]) for r in photo}:
        raise ValueError(f"{where}: two photolysis lines for absorber {fields[0]}, branch {branch}")
    return {"eq": equation, "names": (lhs, rhs), "absorber": fields[0], "branch": branch}


def _thermal_reaction(line: str, where: str, section: str) -> dict[str, Any]:
    """Return one thermal reaction with its rate law, from the law of its section."""
    equation, fields = _reaction_line(line, where)
    lhs, rhs = (side.split("+") for side in equation.split("->"))
    names = ([q.strip() for q in lhs if q.strip() != "M"], [q.strip() for q in rhs if q.strip() != "M"])
    reaction: dict[str, Any] = {"eq": equation, "names": names}
    if section == "two":
        reaction["kind"] = KIND_TWO_BODY
        reaction["A"], reaction["B"], reaction["C"] = _numbers(fields, 3, where, line)
    elif section == "three":
        reaction["kind"] = KIND_FALLOFF
        keys = ("A0", "B0", "C0", "Ai", "Bi", "Ci")
        reaction.update(zip(keys, _numbers(fields, 6, where, line), strict=True))
    elif section == "irrev":
        raise ValueError(f"no reverse law for the irreversible reaction {equation}: the reader has none")
    elif section == "low":
        reaction["kind"] = KIND_LOW_PRESSURE
        reaction["A0"], reaction["B0"], reaction["C0"] = _numbers(fields, 3, where, line)
    else:
        if equation.replace(" ", "") not in SPECIAL:
            raise ValueError(f"unknown special reaction {equation}")
        reaction["kind"] = KIND_SPECIAL
        reaction.update(SPECIAL[equation.replace(" ", "")])
    return reaction


def _extra_line(line: str, where: str, reactions: list[dict[str, Any]], bulk: str) -> None:
    """Add a "Troe [...]" or "effi {...}" line to the reaction above it."""
    if not reactions:
        raise ValueError(f"{where}: this line must follow a reaction: {line}")
    try:
        values = ast.literal_eval(line[4:].strip())
    except (ValueError, SyntaxError):
        raise ValueError(f"{where}: this line's list or dict cannot be read: {line}") from None
    if line.startswith("Troe"):
        if len(values) != TROE_FIELDS:
            raise ValueError(
                f"{where}: a Troe line holds {TROE_FIELDS} numbers (a, T3, T1, T2), got {len(values)}: {line}"
            )
        reactions[-1]["troe"] = tuple(_finite([float(v) for v in values], where, line))
        return
    efficiencies = values
    if bulk not in efficiencies:
        raise ValueError(f"{where}: no efficiency for the bulk gas {bulk}; pass another bulk: {line}")
    reactions[-1]["eff"] = _finite([float(efficiencies[bulk])], where, line)[0]


def _check_comment(low: str, line: str, where: str) -> None:
    """Refuse a comment after the first heading that a section's reactions could be read under by mistake.

    Free text belongs before the first heading. After it a comment is a known heading, a line of column titles, or a
    reaction that is commented out; any other could be a heading the reader does not know, under which the reactions
    below would keep the law of the section above.
    """
    if not (low.startswith((*KNOWN_HEADINGS, COLUMN_TITLES)) or "[" in low or not low.strip("# ")):
        raise ValueError(
            f"{where}: unknown heading: {line}. The headings are {KNOWN_HEADINGS}; free text goes above the first one"
        )


def _read_reactions(network_file: str | Path, bulk: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Read a reaction list: the thermal reactions, the photolysis branches and the condensation lines skipped."""
    section, stopped, skipped = None, None, 0
    reactions: list[dict[str, Any]] = []
    photo: list[dict[str, Any]] = []
    for number, raw in enumerate(Path(network_file).read_text().split("\n"), 1):
        line, where = raw.strip(), f"{network_file}:{number}"
        if line.startswith("#"):
            low = line.lower()
            if section is not None and not stopped:  # past the thermal sections no comment can change a reaction's law
                _check_comment(low, line, where)
            if low.startswith(STOP):
                stopped = stopped or low
                if low.startswith("# photo"):
                    section = "photo"
                elif not low.startswith("# reverse stops"):
                    section = "skip"
            elif not stopped:
                section = next((name for head, name in SECTIONS if low.startswith(head)), section)
        elif stopped:
            if section == "photo" and "[" in line:
                photo.append(_photolysis_branch(line, where, photo))
            elif section == "skip" and "[" in line:
                skipped += 1
            elif "[" in line:
                raise ValueError(f"reaction after '# reverse stops' outside photolysis / condensation: {line}")
        elif line.startswith(("Troe", "effi")):
            _extra_line(line, where, reactions, bulk)
        elif "[" in line:
            if section is None:
                raise ValueError(f"{where}: a reaction above the first heading has no rate law: {line}")
            reactions.append(_thermal_reaction(line, where, section))
    return reactions, photo, skipped


def _stoichiometry(reactions: list[dict[str, Any]], n_species: int) -> np.ndarray:
    """Return products minus reactants: [species, reactions]."""
    stoichiometry = np.zeros((n_species, len(reactions)), np.float64)
    for j, r in enumerate(reactions):
        for i in r.get("re", ()):
            stoichiometry[i, j] -= 1
        for i in r.get("pr", ()):
            stoichiometry[i, j] += 1
    return stoichiometry


def _gibbs_rt(a: np.ndarray, T: float) -> float:
    """Return G / RT of a species at T (K) from one range's coefficients a1..a7, b1, b2."""
    enthalpy = (
        -a[0] / T**2 + a[1] * np.log(T) / T + a[2] + a[3] * T / 2 + a[4] * T**2 / 3 + a[5] * T**3 / 4 + a[6] * T**4 / 5
    ) + a[7] / T
    entropy = (
        -a[0] / T**2 / 2 - a[1] / T + a[2] * np.log(T) + a[3] * T + a[4] * T**2 / 2 + a[5] * T**3 / 3 + a[6] * T**4 / 4
    ) + a[8]
    return float(enthalpy - entropy)


def _balanced(branch: dict[str, Any]) -> bool:
    """Say whether a photolysis branch's atoms balance."""
    balance: dict[str, int] = {}
    for side, sign in ((branch["names"][0], -1), (branch["names"][1], 1)):
        for q in side:
            for element, count in formula(q.split("_")[0]).items():
                balance[element] = balance.get(element, 0) + sign * count
    return not any(balance.values())


class VulcanNetwork:
    """A VULCAN reaction list with its NASA-9 data.

    Parameters
    ----------
    network_file : str or pathlib.Path
        The reaction list.
    nasa_dir : str or pathlib.Path
        Folder of <species>.txt NASA-9 files.
    species : list of str, optional
        The species order. The default is the order of first appearance, with He appended; a species that only
        photolysis lines name is not added.
    bulk : str
        The gas whose efficiency an "effi" line applies (VULCAN's atm_base).
    """

    def __init__(
        self, network_file: str | Path, nasa_dir: str | Path, species: list[str] | None = None, bulk: str = "H2"
    ) -> None:
        reactions, photo, self.skipped = _read_reactions(network_file, bulk)
        self.species = self._species_order(reactions) if species is None else list(species)
        repeated = sorted({name for name in self.species if self.species.count(name) > 1})
        if repeated:
            raise ValueError(f"species names {repeated} more than once: a name is one row of the state; drop repeats")
        self.lo, self.hi = self._read_nasa9(nasa_dir)
        self.reac = reactions
        self._index_species(reactions, network_file)
        self.duplicates = self._duplicates(reactions)
        self.S = _stoichiometry(reactions, len(self.species))
        counts = [formula(s.split("_")[0]) for s in self.species]
        self.elements = sorted({e for c in counts for e in c})
        self.atoms = np.array([[c.get(e, 0) for e in self.elements] for c in counts], np.float64)
        unbalanced = [reactions[j]["eq"] for j in np.flatnonzero(np.abs(self.atoms.T @ self.S).max(0) > 0)]
        if unbalanced:
            raise ValueError(f"reactions that do not balance their atoms: {unbalanced}")
        self.photo_missing = sorted({q for r in photo for q in r["names"][0] + r["names"][1] if q not in self.species})
        for r in photo:
            if not _balanced(r):
                raise ValueError(f"photolysis branch {r['eq']} does not balance its atoms")
        if not self.photo_missing:
            self._index_species(photo, network_file)
        self.photo, self.Sp = photo, _stoichiometry(photo, len(self.species))

    @staticmethod
    def _species_order(reactions: list[dict[str, Any]]) -> list[str]:
        """Return the order of first appearance, with He appended."""
        species: list[str] = []
        for r in reactions:
            for q in r["names"][0] + r["names"][1]:
                if q not in species:
                    species.append(q)
        if "He" not in species:
            species.append("He")
        return species

    def _read_nasa9(self, nasa_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """Return the NASA-9 coefficients a1..a7, b1, b2 of the two ranges: [species, 9] each."""
        lo, hi = [], []
        for name in self.species:
            path = Path(nasa_dir) / f"{name}.txt"
            if not path.exists():
                raise FileNotFoundError(f"no NASA-9 file for {name}: {path}; pass the folder that holds it")
            a = np.loadtxt(path).flatten()
            if a.size not in NASA_FILE_SIZES:
                raise ValueError(f"{path}: {a.size} numbers, a NASA-9 file holds {NASA_FILE_SIZES}; check the file")
            lo.append(np.concatenate([a[0:7], a[8:10]]))  # the 8th entry of each range is unused
            hi.append(np.concatenate([a[10:17], a[18:20]]))
            jump = abs(_gibbs_rt(lo[-1], NASA_SPLIT_T) - _gibbs_rt(hi[-1], NASA_SPLIT_T))
            if not jump <= MAX_GIBBS_JUMP:
                raise ValueError(
                    f"{path}: G / RT jumps by {jump:.3g} at {NASA_SPLIT_T:g} K, above {MAX_GIBBS_JUMP:g}: the two "
                    "ranges read do not meet; check that the file's first two ranges are the ones below and above it"
                )
        return np.array(lo, np.float64), np.array(hi, np.float64)

    def _index_species(self, reactions: list[dict[str, Any]], network_file: str | Path) -> None:
        """Add each reaction's reactant and product indices."""
        position = {name: i for i, name in enumerate(self.species)}
        for r in reactions:
            missing = [q for q in r["names"][0] + r["names"][1] if q not in position]
            if missing:
                raise ValueError(f"{network_file}: reaction {r['eq']} names {missing}, not in the species list")
            r["re"] = [position[q] for q in r["names"][0]]
            r["pr"] = [position[q] for q in r["names"][1]]

    @staticmethod
    def _duplicates(reactions: list[dict[str, Any]]) -> list[tuple[str, str]]:
        """Return pairs of thermal entries with the same species on both sides, in either orientation.

        A three-body entry differs from a two-body one. Their rates add, as in VULCAN.
        """
        seen: dict[Any, str] = {}
        duplicates = []
        for r in reactions:
            m = ("M",) if r["kind"] != KIND_TWO_BODY else ()
            key = tuple(sorted((tuple(sorted(r["names"][0])) + m, tuple(sorted(r["names"][1])) + m)))
            if key in seen:
                duplicates.append((seen[key], r["eq"]))
            seen.setdefault(key, r["eq"])
        return duplicates
