"""The solver's settings and constants, by kind, each with its unit and its reason or source.

The kernels get the call and method settings and the physical constants through their generated header
(_kernel_header.py), so each value is written once, here. The method is in docs/method.md; a published formula's
coefficients stay beside the formula in kernels/, and the checks' acceptance bars are in _checks.py. This file holds
the solver's settings; the coupling's few constants are named in its own modules (maxichem/coupling/). Hardware
limits and file formats stay beside the code that meets them (kincuda.py, kinvulcan.py), and cpp/ keeps copies of a
few values here, held equal by tests/test_constants_source.py.
"""

# Call settings: what a call accepts, and its defaults.
T_MIN = 300.0  # K: the solver's domain starts here (colder cells are counted, not stepped, and the call raises)
# K: the domain ends here, as hotter cells do. The NASA-9 fits run to 6000 K (McBride et al. 2002); accuracy against
# VULCAN is checked only up to 3000 K, the edge of the truth sets
T_MAX = 3000.0
# |sum(x) - 1| above this is bad input. Measured |sum(x) - 1| at most 1.3e-15 over every stored input the checks feed
# (truth starts, paths and their mixes, panel P, the yardstick, the derivative boxes) and 1.3e-15 over the
# solver's own outputs (60-call chains, both default networks): far above rounding, far below the densities' effect on
# a call (rtol 1e-3)
X_SUM_TOL = 1e-10
DRIFT = 1e-6  # default drift allowance (NCHO): relative change of an element's share left to rounding
REC_STEPS = 10  # the recovery's tries, in units of the call's max_steps (40,000 at the default 4,000)
DEFAULT_RTOL = 1e-3  # relative tolerance of a call
DEFAULT_MAX_STEPS = 4000  # tries per call before a cell is flagged as unfinished
ATOL = 1e-25  # default absolute tolerance, in mole fraction (docs/method.md)
# attempts stored per cell before the adjoint switches to segment checkpoints; a call may take max_steps up to
# (2 CAP)^2 / REC_STEPS, 0.4 CAP^2, so this CAP's limit is DEFAULT_MAX_STEPS
CAP = 100
DMAX = 16  # tangent directions per jvp call

# Input data the reader accepts.
# Largest |G/RT, low range - G/RT, high range| at NASA_SPLIT_T. Sound fits meet: over the 11 networks the largest jump
# is 1.6e-6 (TiC). TiH, at 0.143, is refused on purpose, and TiSNCHO with it: below 1000 K its fit has Cp / R from -36
# to 819 (notes.md). A file with a wrong coefficient set reaches 17.6 (HNO3 with its below-200 K set read as the low
# range).
MAX_GIBBS_JUMP = 1.0e-5

# Method settings: the step control, the recovery's pivoting and the guard's correction (docs/method.md). Unitless
# unless stated; the kernels' constexprs of the same names take them from the header.
H_MIN = 1e-12  # s: the smallest first try of a call
CALL_END = 1.0 - 1e-14  # a call ends at this share of dt, so rounding does not force one more step
STEP_SAFETY = 0.9  # the safety factor on err^(-1/3)
ERR_FLOOR = 1e-10  # keeps the step factor finite at err = 0
FAC_NONFINITE = 0.1  # the step factor after a non-finite err
FACACC = 0.2  # the smallest step factor after an accepted step
FACREJ = 0.2  # the smallest step factor after a rejected try
FACMAX = 20.0  # the largest step factor
FAC_AFTER_REJECT = 0.5  # the largest step factor after a rejected try
SECMIN = 1e-4  # the smallest secant shrink
SECANT_MIN_ORDER = 0.05  # below this fitted order of err in h, the shrink is SECMIN
# Rodas3's error order (err ~ h^3): the largest order the secant fit may use, and the plain law's cube root, which the
# kernels take as cbrt (a static_assert holds them together)
SECANT_MAX_ORDER = 3.0
SECANT_TARGET = 0.5  # the err a secant shrink aims at
# recovery: exchange rows only below this share of the column's largest; plain partial pivoting (1) gave 12 times the
# solve error at the step control's scale (notes.md, "Do not retry")
PIVTAU = 0.1
ELEMENT_PIVOT_MIN = 1e-10  # the guard leaves out an element whose pivot falls below this share
GMAX = 1e-3  # the element-share correction changes a species by at most this, relative
NTAIL = 32  # rows of the trailing block factored densely in registers, one lane each, so at most WARP_LANES
# Warps per cell: WARPS_PER_CELL, or WARPS_PER_CELL_LARGE for a network whose per-cell arrays fit at most
# LARGE_NETWORK_CELLS times in a block's shared memory. At 4 warps x 32 lanes and 160-190 registers per thread the
# register file already caps such a block at 2-3 cells, so the extra lanes cost no cells
WARPS_PER_CELL = 2
WARPS_PER_CELL_LARGE = 4
LARGE_NETWORK_CELLS = 3

# Physical constants.
KB = 1.380649e-16  # erg K^-1: Boltzmann's constant (exact in the SI since 2019)
P0 = 1.0e6  # dyn cm^-2 (1 bar): the standard pressure of the NASA-9 Gibbs energies (McBride et al. 2002)
NASA_SPLIT_T = 1000.0  # K: where the NASA-9 fits change range (McBride et al. 2002)
K_COLL = 1e-9  # cm3 s^-1: no thermal rate may exceed this as a bimolecular-equivalent rate (docs/method.md)
# The rate rule the checks' truth sets and stored references must be made with (the collision limit counts M as a
# collision partner, docs/method.md); the truth generator and the torch reference generator write the same string
RATE_RULE = "collision-limit-2026-11-10: M counts as a collision partner"

# Layout: sizes and codes shared by the host code, the kernels, the export and cpp/.
WARP_LANES = 32  # threads per warp on every CUDA GPU; the kernels write it as 32 (lane = threadIdx.x & 31)
DOUBLE_BYTES = 8  # bytes per float64, the unit of the kernels' shared memory and scratch
GUARD_BLOCK_THREADS = 128  # threads per block of the guard kernel; any size works, this one packs cells densely
ARRH_BLOCK_THREADS = 256  # threads per block of the arrh kernel, a power of 2: its reduction halves it
# Columns of a reaction's row in the rate table rc: ln A, B, C of k0, of the second k0 term and of k_inf (the Arrhenius
# columns, the derivatives' rc), then Troe's a, T3, T1, T2 and the efficiency
ARRHENIUS_COLUMNS = 9
RATE_ROW_COLUMNS = 14
# A reaction's law, as VULCAN's network files number them: a reader dict's "kind"
KIND_TWO_BODY = 2
KIND_FALLOFF = 3  # three-body with a high-pressure rate
KIND_LOW_PRESSURE = 5  # three-body, low-pressure rate only
KIND_SPECIAL = 6  # a law held in the code (kinvulcan.SPECIAL), with falloff
FALLOFF_KINDS = (KIND_FALLOFF, KIND_SPECIAL)  # the laws with k_inf
THREE_BODY_KINDS = (KIND_FALLOFF, KIND_LOW_PRESSURE, KIND_SPECIAL)  # the laws with M: times n_tot both ways
# nonfinite_h: cells whose starting step was not finite, so it acted as dt
COUNTS = (
    "bad_input",
    "cold",
    "hot",
    "nonfinite",
    "unfinished",
    "drift",
    "recomputed",
    "recovered",
    "still_failed",
    "nonfinite_h",
)
EXPORT_FORMAT = 6  # raised by a change the export's reader depends on; cpp/maxichem.cpp refuses any other
COUNT_SLOTS = ("queue", *COUNTS, "recovery_queue")  # the device's count array: the two cell queues around COUNTS
NO_SLOT = 0xFFFF  # marks a (row, column) with no slot in the kernels' uint16 map

# Bits of a reaction's flag word in the rf table: its rate law, and fields read by shifting.
RATE_LAW_FLAGS = {
    "FL_FALLOFF": 1,
    "FL_SECOND_K0": 1 << 1,
    "FL_THREE_BODY": 1 << 2,
    "FL_TROE": 1 << 6,
    "FL_EFF": 1 << 7,
    "FL_VM11": 1 << 9,
}
RATE_LAW_SHIFTS = {
    "FL_DMU_SHIFT": 3,  # three bits: products minus reactants, plus 4
    "FL_FORWARD_SHIFT": 10,  # two bits: molecules on the forward side
    "FL_REVERSE_SHIFT": 12,  # two bits: molecules on the reverse side
}

# Each kernel's parameters, in the order of its signature in kernels/. The Python and C++ callers build their launches
# from this table; a test compares it with the kernel files. A photolysis network adds PHOTOLYSIS_ARGUMENT at the end of
# PHOTOLYSIS_KERNEL_NAMES' lists; the derivative kernels always take it, null without photolysis.
_JVP_ARGUMENTS = (
    "x0", "Ta", "Pa", "dta", "hin", "lm", "rc", "dx0", "dTa", "dPa", "ddta", "dhin", "dlm", "drc", "D",
    "xout", "hout", "doneo", "noko", "status", "dxout", "dhout", "count", "B", "rtol", "atol", "max_steps", "adapt",
    "perm", "blob", "rf", "nasa", "scratch", "sst", "jt", "lu", "optr", "sround", "jround", "Jin", "dJin",
)  # fmt: skip
_VJP_ARGUMENTS = (
    "x0", "Ta", "Pa", "dta", "hin", "lm", "rc", "gx", "gh", "xout", "hout", "doneo", "noko", "status",
    "gx0", "gT", "gP", "gdt", "ghin", "glm", "count", "B", "rtol", "atol", "max_steps", "adapt",
    "perm", "blob", "rf", "nasa", "scratch", "rec", "sst", "jt", "lu", "optr", "gt", "xg",
    "sround", "jround", "ground", "Jin", "gJ", "gA",
)  # fmt: skip
_RECOVERY_ARGUMENTS = ("redo", "list", "wscr", "didx", "atoms")  # what a derivative kernel's recovery adds
SOLVER_KERNEL_NAMES = ("call", "guard", "recover")  # the kernels of SOLVER_KERNELS' file, and what the export holds
PHOTOLYSIS_KERNEL_NAMES = ("call", "recover")  # the solver kernels that take J last with photolysis; the guard has none
KERNEL_ARGUMENTS = {
    "call": (
        "x0", "Ta", "Pa", "dta", "hio", "xout", "doneo", "noko", "status", "count", "B", "rtol", "atol", "max_steps",
        "perm", "blob", "rc", "rf", "nasa", "scratch", "sst", "jt", "lu", "optr", "sround", "jround",
    ),
    "guard": ("x0", "xout", "doneo", "noko", "status", "atoms", "redo", "list", "count", "B", "atol"),
    "recover": (
        "x0", "Ta", "Pa", "dta", "hin", "xout", "hio", "doneo", "noko", "redo", "list", "count", "rtol", "atol",
        "max_steps", "perm", "blob", "rc", "rf", "nasa", "scratch", "sst", "jt", "didx", "atoms", "sround", "jround",
    ),
    "jvp": _JVP_ARGUMENTS,
    "jvp_rec": _JVP_ARGUMENTS + _RECOVERY_ARGUMENTS,
    "vjp": _VJP_ARGUMENTS,
    "vjp_rec": _VJP_ARGUMENTS + _RECOVERY_ARGUMENTS,
    "arrh": ("gA", "Ta", "Pa", "rc", "rf", "grc", "B"),
}  # fmt: skip
PHOTOLYSIS_ARGUMENT = "Jin"
