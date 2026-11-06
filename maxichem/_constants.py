"""Settings shared by the solver modules, the kernels' header and the export."""

T_MIN = 300.0  # K: the solver's domain starts here (colder cells are counted, not stepped)
DRIFT = 1e-6  # default drift allowance (NCHO): relative change of an element's share left to rounding
REC_STEPS = 10  # the recovery's tries, in units of the call's max_steps (40,000 at the default 4,000)
DEFAULT_RTOL = 1e-3  # relative tolerance of a call
DEFAULT_MAX_STEPS = 4000  # tries per call before a cell is flagged as unfinished
ATOL = 1e-25  # default absolute tolerance, in mole fraction (docs/method.md)
K_COLL = 1e-9  # cm3 s^-1: no thermal rate may exceed this as a bimolecular-equivalent rate (docs/method.md)
COUNTS = ("bad_input", "cold", "nonfinite", "unfinished", "drift", "recomputed", "recovered", "still_failed")
EXPORT_FORMAT = 2  # raised when the export's files or the kernels' arguments change; cpp/maxichem.cpp checks it
MAX_SPECIES = 254  # species slots are uint8, with one value left for the padding
NTAIL = 32  # rows of the trailing block factored in registers (at most 32)
CAP = 100  # attempts stored per cell before the adjoint switches to segment checkpoints (calls up to CAP^2 attempts)
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
    "FL_REV": 1 << 8,
    "FL_VM11": 1 << 9,
}
RATE_LAW_SHIFTS = {
    "FL_DMU_SHIFT": 3,  # three bits: products minus reactants, plus 4
    "FL_FORWARD_SHIFT": 10,  # two bits: molecules on the forward side
    "FL_REVERSE_SHIFT": 12,  # two bits: molecules on the reverse side
}

# Each kernel's parameters, in the order of its signature in kernels/. The Python and C++ callers build their launches
# from this table; a test compares it with the kernel files. A photolysis network adds PHOTOLYSIS_ARGUMENT at the end.
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
}  # fmt: skip
PHOTOLYSIS_ARGUMENT = "Jin"

GUARD_BLOCK_THREADS = 128  # threads per block of the guard kernel; any size works, this one packs cells densely
MAX_THREADS_PER_BLOCK = 1024  # CUDA's limit
MAX_NAMED_BARRIERS = 15  # per block: the hardware's named barriers, one per cell (bar.sync in common.cuh)
REGISTER_ALLOCATION_UNIT = 8  # registers per thread are allocated in this unit
