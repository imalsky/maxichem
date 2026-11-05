"""VULCAN's reaction lists (VULCAN-JAX's NCHO_photo_network.txt and the other VULCAN-format networks) and NASA-9 files,
read as data (the thermal gas-phase part): the reaction list (id, reaction, rate coefficients) and one NASA-9 file per
species (two ranges of 10 coefficients, split at 1000 K). The rate laws, by section (evaluated by the kernels of
kincuda.py):
    two-body                k = A T^B exp(-C / T)
    3-body                  k = k0 / (1 + k0 n_tot / k_inf) * F (k0, k_inf Arrhenius), both directions times n_tot
    without high-pressure   k = k0, both directions times n_tot
    special                 the law of SPECIAL for the equation, falloff as 3-body; OH + CH3 + M -> CH3OH + M
                            (Jasper et al. 2007 fits, Visscher & Moses 2011 eqs 13-14):
                            k0 = 1.932e3 T^-9.88 exp(-7544/T) + 5.109e-11 T^-6.25 exp(-1433/T),
                            k_inf = 1.031e-10 T^-0.018 exp(16.74/T), times F = Fc^(1 / (1 + (log10 Pr / N)^2)) with
                            Fc = 0.1855 exp(-T/155.8) + 0.8145 exp(-T/1675) + exp(-4531/T), N = 0.75 - 1.27 log10 Fc
                            (a Troe factor without Troe's shift terms: troe (0.8145, 155.8, 1675, 4531), width "vm11")
    irreversible            3-body falloff; the reverse law is REVERSE's entry for the equation
F = 1, or the Troe factor of a following "Troe [a, T3, T1, T2]" line (Venot et al. 2012, eq. C.2-C.5) with
Pr = k0 n_tot / k_inf; a following "effi {...}" line multiplies k by the efficiency of the bulk gas. The other reverse
rates come from the equilibrium constants of the NASA-9 data: k_r = k_f / K (k_B T / P0)^(sum of S over the reaction),
K = exp(-dG / RT) with P0 = 1 bar. The thermal reactions end at the first "# reverse stops", "# condensation", "# photo"
or "# re_end" heading (condensation lines and the lines after "# re_end" are skipped and counted in skipped; a reaction
between "# reverse stops" and the next of those headings is refused). The photolysis lines ("# photo" section,
"id [ A -> P1 + P2 ] A branch") are read into photo: one irreversible first-order branch each, absorber A its only
reactant, its rate J n_A with J (s^-1) an input of the call (kincuda.CudaSolver with photo=True); Sp [species, branches]
holds their products minus reactants. A branch whose reactant is not its named absorber or whose atoms do not balance is
refused, as are two lines with the same absorber and branch index; photo_missing lists the branches' species absent from
the species list (photo=True then refuses the network). Element counts come from the species name up to its first "_"
(CH2_1 -> CH2, N_2D -> N). Each reaction dict holds its law: kind (2 two-body, 3 falloff, 5 low-pressure only, 6
special), A, B, C (kind 2) or A0, B0, C0, optional A1, B1, C1 (a second k0 term) and Ai, Bi, Ci (kinds 3, 6); optional
troe (a, T3, T1, T2) with width "vm11" for the special law's form, eff and rev (the explicit reverse law, same keys); eq
(the equation), names, re and pr (reactant and product names and indices). A photo dict holds eq, names, absorber,
branch (the file's branch index), re and pr. The network object holds species, lo / hi (NASA-9 coefficients a1..a7, b1,
b2 of the two ranges), reac, S [species, reactions] (products minus reactants; a thermal reaction whose atoms do not
balance is refused), photo, Sp [species, branches], elements, atoms [species, elements], skipped and duplicates: the
pairs of thermal entries with the same species on both sides (either orientation; a three-body entry differs from a
two-body one). Their rates add, as in VULCAN; the shipped NCHO and SNCHO networks have none.
"""

import ast
import re

import numpy as np

KB = 1.380649e-16  # erg K^-1
P0 = 1.0e6  # dyn cm^-2 (1 bar)


def formula(name):
    """Element counts of a species name, summed over repeats (CH3OH: C 1, H 4, O 1)."""
    c = {}
    for e, n in re.findall(r"([A-Z][a-z]?)(\d*)", name):
        c[e] = c.get(e, 0) + int(n or 1)
    return c


# laws not in the reaction list (VULCAN's code holds them), by equation without spaces
SPECIAL = {
    "OH+CH3+M->CH3OH+M": dict(
        A0=1.932e3,
        B0=-9.88,
        C0=7544.0,
        A1=5.109e-11,
        B1=-6.25,
        C1=1433.0,
        Ai=1.031e-10,
        Bi=-0.018,
        Ci=-16.74,
        troe=(0.8145, 155.8, 1675.0, 4531.0),
        width="vm11",
    )
}
REVERSE = {
    "H+CH3+M->M+CH4": dict(
        kind=3, A0=1.40e-06, B0=0.0, C0=45700.0, Ai=2.40e16, Bi=0.0, Ci=52790.0, troe=(0.69, 90.0, 2210.0, 1e18)
    )
}
STOP = ("# reverse stops", "# condensation", "# photo", "# re_end")


class VulcanNetwork:
    """network_file: a VULCAN reaction list; nasa_dir: folder of <species>.txt NASA-9 files; species: optional order
    (default: order of first appearance in the reaction list, He appended; a species that only photolysis lines name is
    not added, pass a species list that holds it to use photolysis); bulk: the gas whose efficiency an effi line applies
    (VULCAN's atm_base)."""

    def __init__(self, network_file, nasa_dir, species=None, bulk="H2"):
        section, reac, photo, self.skipped, stopped = None, [], [], 0, None
        for ln in open(network_file).read().split("\n"):
            s = ln.strip()
            if s.startswith("#"):
                low = s.lower()
                if low.startswith(STOP):
                    stopped = stopped or low
                    if low.startswith("# photo"):
                        section = "photo"
                    elif not low.startswith("# reverse stops"):
                        section = "skip"
                elif stopped:
                    continue
                elif low.startswith("# two-body"):
                    section = "two"
                elif low.startswith("# 3-body reactions without high-pressure"):
                    section = "low"
                elif low.startswith("# 3-body"):
                    section = "three"
                elif low.startswith("# special"):
                    section = "special"
                elif low.startswith("# irreversible"):
                    section = "irrev"
                continue
            if stopped:
                if section == "photo" and "[" in s:
                    body = s.split("[", 1)[1].split("]", 1)
                    eq, rest = body[0].strip(), body[1].split()
                    lhs, rhs = ([q.strip() for q in side.split("+")] for side in eq.split("->"))
                    if lhs != [rest[0]]:
                        raise ValueError(f"photolysis line whose reactant is not its absorber {rest[0]}: {s}")
                    if (rest[0], int(rest[1])) in {(r["absorber"], r["branch"]) for r in photo}:
                        raise ValueError(f"two photolysis lines for absorber {rest[0]}, branch {rest[1]}")
                    photo.append({"eq": eq, "names": (lhs, rhs), "absorber": rest[0], "branch": int(rest[1])})
                elif section == "skip" and "[" in s:
                    self.skipped += 1
                elif "[" in s:
                    raise ValueError(f"reaction after '# reverse stops' outside photolysis / condensation: {s}")
                continue
            if s.startswith("Troe"):
                reac[-1]["troe"] = tuple(float(v) for v in ast.literal_eval(s[4:].strip()))
                continue
            if s.startswith("effi"):
                reac[-1]["eff"] = float(ast.literal_eval(s[4:].strip())[bulk])
                continue
            if not s or "[" not in s or section is None:
                continue
            body = s.split("[", 1)[1].split("]", 1)
            eq, rest = body[0].strip(), body[1].split()
            lhs, rhs = (side.split("+") for side in eq.split("->"))
            lhs = [q.strip() for q in lhs if q.strip() != "M"]
            rhs = [q.strip() for q in rhs if q.strip() != "M"]
            r = {"eq": eq, "names": (lhs, rhs)}
            if section == "two":
                r["kind"] = 2
                r["A"], r["B"], r["C"] = (float(v) for v in rest[:3])
            elif section in ("three", "irrev"):
                r["kind"] = 3
                r["A0"], r["B0"], r["C0"], r["Ai"], r["Bi"], r["Ci"] = (float(v) for v in rest[:6])
                if section == "irrev":
                    if eq.replace(" ", "") not in REVERSE:
                        raise ValueError(f"no reverse law for the irreversible reaction {eq}")
                    r["rev"] = dict(REVERSE[eq.replace(" ", "")])
            elif section == "low":
                r["kind"] = 5
                r["A0"], r["B0"], r["C0"] = (float(v) for v in rest[:3])
            else:
                if eq.replace(" ", "") not in SPECIAL:
                    raise ValueError(f"unknown special reaction {eq}")
                r["kind"] = 6
                r.update(SPECIAL[eq.replace(" ", "")])
            reac.append(r)
        if species is None:
            species = []
            for r in reac:
                for q in r["names"][0] + r["names"][1]:
                    if q not in species:
                        species.append(q)
            if "He" not in species:
                species.append("He")
        self.species = list(species)
        lo, hi = [], []
        for sp in self.species:
            a = np.loadtxt(f"{nasa_dir}/{sp}.txt").flatten()
            lo.append(np.concatenate([a[0:7], a[8:10]]))  # a1..a7, b1, b2 (the 8th entry of each range is unused)
            hi.append(np.concatenate([a[10:17], a[18:20]]))
        self.lo, self.hi = np.array(lo), np.array(hi)
        for r in reac:
            r["re"] = [self.species.index(q) for q in r["names"][0]]
            r["pr"] = [self.species.index(q) for q in r["names"][1]]
        self.reac = reac
        seen, self.duplicates = {}, []  # same species on both sides (either orientation, M kept): VULCAN
        for r in reac:  # adds the rates of such entries, and so does this reader
            m = ("M",) if r["kind"] != 2 else ()
            key = tuple(sorted((tuple(sorted(r["names"][0])) + m, tuple(sorted(r["names"][1])) + m)))
            if key in seen:
                self.duplicates.append((seen[key], r["eq"]))
            seen.setdefault(key, r["eq"])
        S = np.zeros((len(self.species), len(reac)))
        for j, r in enumerate(reac):
            for i in r["re"]:
                S[i, j] -= 1
            for i in r["pr"]:
                S[i, j] += 1
        self.S = S
        counts = [formula(s.split("_")[0]) for s in self.species]
        self.elements = sorted({e for c in counts for e in c})
        self.atoms = np.array([[c.get(e, 0) for e in self.elements] for c in counts], float)
        bad = [reac[j]["eq"] for j in np.flatnonzero(np.abs(self.atoms.T @ S).max(0) > 0)]
        if bad:
            raise ValueError(f"reactions that do not balance their atoms: {bad}")
        self.photo_missing = sorted({q for r in photo for q in r["names"][0] + r["names"][1] if q not in self.species})
        Sp = np.zeros((len(self.species), len(photo)))
        for j, r in enumerate(photo):
            bal = {}
            for side, sg in ((r["names"][0], -1), (r["names"][1], 1)):
                for q in side:
                    for e, k in formula(q.split("_")[0]).items():
                        bal[e] = bal.get(e, 0) + sg * k
            if any(bal.values()):
                raise ValueError(f"photolysis branch {r['eq']} does not balance its atoms")
            if not self.photo_missing:
                r["re"] = [self.species.index(q) for q in r["names"][0]]
                r["pr"] = [self.species.index(q) for q in r["names"][1]]
                for i in r["re"]:
                    Sp[i, j] -= 1
                for i in r["pr"]:
                    Sp[i, j] += 1
        self.photo, self.Sp = photo, Sp
