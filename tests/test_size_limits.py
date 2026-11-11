"""tools/check_size_limits.py on small C sources: argument counts and where an exception's reason may sit."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("limits", Path(__file__).parents[1] / "tools" / "check_size_limits.py")
assert _SPEC is not None
assert _SPEC.loader is not None
limits = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(limits)

# A list of 10 defined in one file and an optional parameter defined two ways (with photolysis or without), as the
# kernels' JVP_ARGS and J_PARAM are; the function in another file adds a nested declarator and one more parameter.
MACROS = """
#define CELL_ARGS                                                    \\
  const double *x0, const double *Ta, const double *Pa, double *out, \\
      int B, double rtol, double atol, int max_steps, int adapt, int D
#if NJ
#define J_PARAM , const double *Jin
#else
#define J_PARAM
#endif
"""
AT_THE_LIMIT = """
template <bool DN>
__device__ void step(CELL_ARGS, double (&c)[NT] J_PARAM) {
  for (int i = 0; i < 3; i++) c[i] = f(x0[i], Ta[i]);
}
"""
OVER = AT_THE_LIMIT.replace("(&c)[NT]", "(&c)[NT], int lane")


def _failures(text, macros_text=MACROS):
    return limits.check_c_family(Path("k.cuh"), text, limits.c_macros([macros_text, text]))


def test_arguments_count_macro_lists_nested_declarators_and_the_wider_definition():
    assert _failures(AT_THE_LIMIT) == []  # 10 + c + Jin
    assert _failures(OVER) == ["k.cuh:2: step has 13 arguments, limit 12"]
    assert _failures(OVER, MACROS.replace("#define J_PARAM , const double *Jin", "#define J_PARAM")) == []
    assert _failures("void none() {}\nint unused(void) { return 0; }\n") == []


THIRTEEN = ", ".join(f"int a{i}" for i in range(13))
UNSUPPORTED = "k.cuh:2: unsupported declaration: the scan cannot read its name or count its parameters; write it in a "
UNSUPPORTED += "plain form"


def test_specialisations_operators_and_template_parameters_are_counted():
    assert _failures(f"\ntemplate <>\nvoid f<int>({THIRTEEN}) {{}}\n") == ["k.cuh:2: f has 13 arguments, limit 12"]
    assert _failures(f"\nstd::array<int, 4> S::operator()({THIRTEEN}) {{}}\n") == [
        "k.cuh:2: S::operator() has 13 arguments, limit 12"
    ]
    twelve = ", ".join(f"A<int, int> a{i}" for i in range(12))
    assert _failures(f"\nvoid g({twelve}) {{}}\n") == []


def test_forms_the_scan_cannot_read_fail_instead_of_passing():
    pointer = "\nint (*h(int a, int b))(int) { return 0; }\n"
    macro = "\n#define PAIR(t) t first, t second\nvoid m(PAIR(int)) {}\n"
    assert _failures(pointer) == [UNSUPPORTED]
    assert _failures(macro, "") == ["k.cuh:3: " + UNSUPPORTED.split(": ", 1)[1]]
    # an exception does not let a form through that the scan cannot count
    assert _failures("\n// size-limit-ok: reason\n" + pointer.lstrip()) == [UNSUPPORTED.replace(":2:", ":3:")]


def test_an_exception_sits_above_the_declaration_or_inside_it():
    above = "// what it does\n// size-limit-ok: the reason\n" + OVER.lstrip()
    inside = OVER.replace("template <bool DN>\n", "template <bool DN>\n// size-limit-ok: the reason\n")
    too_far = "// size-limit-ok: the reason\n// what it does\n" + OVER.lstrip()
    assert _failures(above) == []
    assert _failures(inside) == []
    assert _failures(too_far) == ["k.cuh:3: step has 13 arguments, limit 12"]
