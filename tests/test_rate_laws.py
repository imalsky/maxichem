"""Forward rate coefficients of each rate law the reader takes, against the laws written out here.

Each reaction turns one isomer of CO into another (CO_1 -> CO_2) beside spectators it does not change, so the first
isomer's mole fraction relaxes to the pair's equilibrium at lambda = (k_f + k_r) times the spectators' densities,
exactly: x(t) = x_eq + (x_0 - x_eq) exp(-lambda t). Each reaction runs in a cell of its own with dt = 1 / lambda, and
the lambda seen there is the solver's rate coefficient. The laws are in docs/method.md, "Rate laws".
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import helpers
import numpy as np
import pytest

from maxichem import kincuda

pytestmark = pytest.mark.gpu

K_COLL = 1e-9  # cm3 s^-1: the collision limit of docs/method.md, typed here
# The Troe factor (Gilbert et al. 1983): c = -0.4 - 0.67 log Fc, N = 0.75 - 1.27 log Fc, d = 0.14
TROE_C = (-0.4, -0.67)
TROE_N = (0.75, -1.27)
TROE_D = 0.14
CONDITIONS = [(600.0, 1e4), (1500.0, 1e6), (2600.0, 1e8)]  # (K, dyn cm^-2): the falloff rows from Pr 1e-3 to 20
RTOL = 1e-9
RATE_TOLERANCE = 1e-6  # relative, on lambda: the integrator's error at RTOL; measured worst 1.0e-9
TRACE = 1e-9  # mole fraction of a reactant traced in a bath, whose own change is then negligible
UNCAPPED_SHARE = 0.1  # a direction of one molecule and M stays below this share of K_COLL: no rule caps it
REVERSE_NEGLIGIBLE = 1e-12  # the reverse of the traced association against its forward, at the end of the call
BOUND = -2.0e5  # K: H / R of CH3OH below that of OH + CH3, so that its dissociation is negligible


def arrhenius(a: float, b: float, c: float, T: float) -> float:
    """Return A T^B exp(-C / T)."""
    return float(a * T**b * np.exp(-c / T))


def troe_factor(T: float, pr: float, troe: tuple[float, ...], vm11: bool) -> float:
    """Return F: Troe's form, or with vm11 the form of Visscher & Moses (2011), with no shift and no d term."""
    a, t3, t1, t2 = troe
    lfc = np.log10((1.0 - a) * np.exp(-T / t3) + a * np.exp(-T / t1) + np.exp(-t2 / T))
    n = TROE_N[0] + TROE_N[1] * lfc
    if vm11:
        f = np.log10(pr) / n
    else:
        shifted = np.log10(pr) + TROE_C[0] + TROE_C[1] * lfc
        f = shifted / (n - TROE_D * shifted)
    return float(10.0 ** (lfc / (1.0 + f * f)))


def falloff(k0: float, k_inf: float, nd: float, T: float, troe: tuple[float, ...] | None, vm11: bool = False) -> float:
    """Return k = F k0 / (1 + Pr) with Pr = k0 n_tot / k_inf (M not folded in)."""
    pr = k0 * nd / k_inf
    return k0 / (1.0 + pr) * (troe_factor(T, pr, troe, vm11) if troe else 1.0)


FALLOFF_K0 = (3.0e-14, -1.0, 0.0)  # cm3 s^-1
FALLOFF_KINF = (3.16, 0.5, 0.0)  # s^-1
TROE = (0.6, 100.0, 2000.0, 5000.0)  # a, T3, T1, T2
LOW = (3.0e-13, -0.5, 800.0)  # cm3 s^-1
EFFICIENCY = 2.5  # the bulk gas's (H2) of the effi line
# OH + CH3 + M -> CH3OH + M: Jasper et al. (2007) in the form of Visscher & Moses (2011), as kinvulcan.SPECIAL holds it
SPECIAL_K0 = ((1.932e3, -9.88, 7544.0), (5.109e-11, -6.25, 1433.0))
SPECIAL_KINF = (1.031e-10, -0.018, -16.74)
SPECIAL_TROE = (0.8145, 155.8, 1675.0, 4531.0)


@dataclass
class Row:
    """One reaction, its law written out, and the cell that sees it."""

    section: str  # the network file's section
    line: str  # its reaction line, and a Troe or effi line below it
    reactant: str  # the species whose decay is seen
    product: str
    rate: Callable[[float, float], float]  # k from (T, n_tot), M not folded in
    partners: tuple[str, ...] = ()  # the other reactants: unchanged spectators, or a bath
    product_enthalpy: float = -1000.0  # K: H / R of the product isomer minus the reactant's, which sets K
    capped: bool = False  # the collision limit acts, at every condition
    bath: dict[str, float] = field(default_factory=lambda: {"He": 0.5, "H2": 0.4})
    start: float = 0.1  # the reactant's mole fraction
    troe: bool = False  # the row compiles the kernels' Troe form


ROWS = {
    "Arrhenius": Row(
        "two",
        "[ CO_1 + He -> CO_2 + He ] 2.0E-12 0.7 1500.0",
        "CO_1",
        "CO_2",
        lambda T, _: arrhenius(2e-12, 0.7, 1500.0, T),
        ("He",),
    ),
    # K < 1: the reverse is the larger direction, and both directions take its limit
    "capped, two molecules": Row(
        "two", "[ CO_11 + He -> CO_12 + He ] 5.0E-09 0.0 0.0", "CO_11", "CO_12", lambda *_: 5e-9, ("He",), 1000.0, True
    ),
    "capped, three molecules": Row(
        "two",
        "[ CO_13 + He + H2 -> CO_14 + He + H2 ] 1.0E-23 0.0 0.0",
        "CO_13",
        "CO_14",
        lambda *_: 1e-23,
        ("He", "H2"),
        capped=True,
    ),
    "Lindemann falloff": Row(
        "three",
        "[ CO_7 + M -> CO_8 + M ] 3.0E-14 -1.0 0.0 3.16 0.5 0.0",
        "CO_7",
        "CO_8",
        lambda T, nd: falloff(arrhenius(*FALLOFF_K0, T), arrhenius(*FALLOFF_KINF, T), nd, T, None),
    ),
    "Troe falloff": Row(
        "three",
        "[ CO_9 + M -> CO_10 + M ] 3.0E-14 -1.0 0.0 3.16 0.5 0.0\nTroe [0.6, 100.0, 2000.0, 5000.0]",
        "CO_9",
        "CO_10",
        lambda T, nd: falloff(arrhenius(*FALLOFF_K0, T), arrhenius(*FALLOFF_KINF, T), nd, T, TROE),
        troe=True,
    ),
    "low-pressure only": Row(
        "low", "[ CO_3 + M -> CO_4 + M ] 3.0E-13 -0.5 800.0", "CO_3", "CO_4", lambda T, _: arrhenius(*LOW, T)
    ),
    # traced in pure H2, where the bulk gas's efficiency is every collider's: k = eff k0 under any reading of effi
    "low-pressure only, efficiency": Row(
        "low",
        "[ CO_5 + M -> CO_6 + M ] 3.0E-13 -0.5 800.0\neffi {'H2': 2.5, 'He': 0.4}",
        "CO_5",
        "CO_6",
        lambda T, _: EFFICIENCY * arrhenius(*LOW, T),
        bath={"H2": 1.0 - TRACE},
        start=TRACE,
    ),
    # OH traced in a CH3 bath: first order at k n_tot n_CH3
    "two k0 terms, Troe of Visscher & Moses": Row(
        "special",
        "[ OH + CH3 + M -> CH3OH + M ]",
        "OH",
        "CH3OH",
        lambda T, nd: falloff(
            sum(arrhenius(*q, T) for q in SPECIAL_K0), arrhenius(*SPECIAL_KINF, T), nd, T, SPECIAL_TROE, vm11=True
        ),
        ("CH3",),
        bath={"CH3": 0.5, "He": 0.5 - TRACE},
        start=TRACE,
        troe=True,
    ),
}
HEADINGS = {
    "two": "# Two-body Reactions",
    "three": "# 3-body and Disscoiation Reactions",
    "low": "# 3-body reactions without high-pressure rates",
    "special": "# special cases",
}


def network(rows: list[Row], folder):
    """Write and read the network of these rows, each product isomer product_enthalpy above its reactant."""
    sections = [
        "\n".join([heading, *(f"{2 * i + 1} {row.line}" for i, row in enumerate(rows) if row.section == section)])
        for section, heading in HEADINGS.items()
        if any(row.section == section for row in rows)
    ]
    text = "# Invented test network: one reaction per rate law\n\n" + "\n\n".join(sections) + "\n\n# reverse stops\n"
    thermo = {"He": helpers.constant_cp(0.0, 2.5), "H2": helpers.constant_cp(0.0)}
    for row in rows:
        thermo[row.reactant] = helpers.constant_cp(0.0)
        thermo[row.product] = helpers.constant_cp(row.product_enthalpy)
        thermo.update({q: helpers.constant_cp(0.0) for q in row.partners})
    if "CH3OH" in thermo:
        thermo["CH3OH"] = helpers.constant_cp(BOUND)
    return helpers.write_network(folder, text, thermo), thermo


def expected(row: Row, T: float, P: float, thermo) -> tuple[float, float]:
    """Return lambda (s^-1) and the reactant's equilibrium mole fraction, from the laws written out above."""
    nd = float(helpers.number_density(T, P))
    k = row.rate(T, nd)
    kf = k * (nd if row.section != "two" else 1.0)
    g = {q: helpers.gibbs_rt(a, T) for q, a in thermo.items()}
    if row.section == "special":  # an association: its reverse is checked negligible instead
        # v_r / v_f = K_p^-1 (P0 / P) x_CH3OH / (x_OH x_CH3), x_CH3OH / x_OH < e after one e-fold
        dg = g["CH3OH"] - g["OH"] - g["CH3"]
        assert np.exp(dg) * helpers.STANDARD_PRESSURE / P * np.e / row.bath["CH3"] < REVERSE_NEGLIGIBLE
        K, kr = np.inf, 0.0
    else:
        K = float(np.exp(g[row.reactant] - g[row.product]))
        kr = kf / K
    molecules = 1 + len(row.partners)
    if molecules >= 2:  # docs/method.md: a direction of m molecules counts at k n_tot^(m - 2)
        c = min(1.0, K_COLL / (max(kf, kr) * nd ** (molecules - 2)))
    else:  # one molecule and M: not limited today; left for the coming change of that rule, so kept far below it
        assert max(k, k / K) < UNCAPPED_SHARE * K_COLL
        c = 1.0
    assert (c < 1.0) == row.capped
    density = float(np.prod([nd * row.bath[q] for q in row.partners]))
    return c * (kf + kr) * density, row.start / (1.0 + K)


@pytest.mark.parametrize("form", ["troe", "plain"])  # plain: no Troe line, so the kernels compile their plain falloff
def test_forward_rate_coefficients(gpu, tmp_path, form):
    rows = {name: row for name, row in ROWS.items() if form == "troe" or not row.troe}
    net, thermo = network(list(rows.values()), tmp_path)
    cases = [(name, row, T, P) for name, row in rows.items() for T, P in CONDITIONS]
    x0 = np.zeros((len(cases), len(net.species)))
    lam, x_eq = np.zeros(len(cases)), np.zeros(len(cases))
    for c, (_, row, T, P) in enumerate(cases):
        x0[c, net.species.index(row.reactant)] = row.start
        for q, f in row.bath.items():
            x0[c, net.species.index(q)] = f
        lam[c], x_eq[c] = expected(row, T, P, thermo)
    T, P = (helpers.tensor([case[i] for case in cases], gpu) for i in (2, 3))
    dt = 1.0 / lam
    with kincuda.CudaSolver(net, gpu) as solver:
        x, _, done, _ = solver(
            helpers.tensor(x0, gpu), T, P, helpers.tensor(dt, gpu), helpers.tensor(1e-3 * dt, gpu), rtol=RTOL
        )
    assert bool(done.all())
    seen = np.array([float(x[c, net.species.index(row.reactant)]) for c, (_, row, _, _) in enumerate(cases)])
    starts = np.array([row.start for _, row, _, _ in cases])
    errors = np.abs(np.log((starts - x_eq) / (seen - x_eq)) / (dt * lam) - 1.0)
    for (name, _, temperature, pressure), e, rate in zip(cases, errors, lam, strict=True):
        print(f"{form}: {name} at {temperature:g} K, {pressure:g} dyn cm^-2: lambda {rate:.6e} s^-1, error {e:.2e}")
    assert errors.max() < RATE_TOLERANCE
