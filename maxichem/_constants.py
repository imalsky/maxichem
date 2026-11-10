"""Settings shared by the solver modules, the kernels' header and the export."""

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
K_COLL = 1e-9  # cm3 s^-1: no thermal rate may exceed this as a bimolecular-equivalent rate (docs/method.md)
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
EXPORT_FORMAT = 5  # raised by a change the export's reader depends on; cpp/maxichem.cpp refuses any other
# attempts stored per cell before the adjoint switches to segment checkpoints; a call may take max_steps up to
# (2 CAP)^2 / REC_STEPS, 0.4 CAP^2, so this CAP's limit is DEFAULT_MAX_STEPS
CAP = 100
DMAX = 16  # tangent directions per jvp call
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
# SOLVER_KERNEL_NAMES' lists; the derivative kernels always take it, null without photolysis.
_JVP_ARGUMENTS = (
    "x0", "Ta", "Pa", "dta", "hin", "lm", "rc", "dx0", "dTa", "dPa", "ddta", "dhin", "dlm", "drc", "D",
    "xout", "hout", "doneo", "noko", "dxout", "dhout", "count", "B", "rtol", "atol", "max_steps", "adapt",
    "perm", "blob", "rf", "nasa", "scratch", "sst", "jt", "lu", "optr", "sround", "jround", "Jin", "dJin",
)  # fmt: skip
_VJP_ARGUMENTS = (
    "x0", "Ta", "Pa", "dta", "hin", "lm", "rc", "gx", "gh", "xout", "hout", "doneo", "noko",
    "gx0", "gT", "gP", "gdt", "ghin", "glm", "count", "B", "rtol", "atol", "max_steps", "adapt",
    "perm", "blob", "rf", "nasa", "scratch", "rec", "sst", "jt", "lu", "optr", "gt", "xg",
    "sround", "jround", "ground", "Jin", "gJ", "gA",
)  # fmt: skip
_RECOVERY_ARGUMENTS = ("redo", "list", "wscr", "didx", "atoms")  # what a derivative kernel's recovery adds
SOLVER_KERNEL_NAMES = ("call", "guard", "recover")  # the kernels of SOLVER_KERNELS' file, and what the export holds
KERNEL_ARGUMENTS = {
    "call": (
        "x0", "Ta", "Pa", "dta", "hio", "xout", "doneo", "noko", "count", "B", "rtol", "atol", "max_steps",
        "perm", "blob", "rc", "rf", "nasa", "scratch", "sst", "jt", "lu", "optr", "sround", "jround",
    ),
    "guard": ("x0", "Ta", "Pa", "dta", "xout", "doneo", "noko", "atoms", "redo", "list", "count", "B", "atol"),
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

GUARD_BLOCK_THREADS = 128  # threads per block of the guard kernel; any size works, this one packs cells densely
ARRH_BLOCK_THREADS = 256  # threads per block of the arrh kernel, a power of 2: its reduction halves it
