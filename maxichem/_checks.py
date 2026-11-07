"""What the checks and the benches share: the gases they score against VULCAN and the gas they start from."""

GASES = ["H2O", "CO", "CO2", "CH4", "NH3", "HCN", "C2H2", "H", "OH", "O", "H2", "N2"]  # the 12 scored gases
FLOORS = [1e-10] * 7 + [1e-8] * 3 + [1e-6] * 2  # mole fraction: below this a gas is not scored, per GASES
SOLAR_START = {"H2": 0.84, "He": 0.16, "H2O": 9e-4, "CO": 5e-4, "CH4": 1e-5, "N2": 6e-5, "NH3": 1e-6}  # mole fractions
H2S_PER_H2_SOLAR = 1.41e-5  # the sulfur of a solar gas, for starts made here
