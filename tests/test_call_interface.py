"""The call interface: settings after the required inputs are keyword only, and the results are named tuples."""

import inspect
from collections.abc import Callable
from typing import Any

import pytest

from maxichem.kincuda import CallResult, CudaSolver, LaunchResult
from maxichem.kincuda_ad import ADSolver, JvpResult, VjpResult

# each call, its function under the device decorator, and the number of tensors it takes by position
CALLS: dict[str, tuple[type, Callable[..., object], int]] = {
    "call": (CudaSolver, CudaSolver.__call__, 5),
    "launch": (CudaSolver, inspect.unwrap(CudaSolver.launch), 5),
    "jvp": (ADSolver, inspect.unwrap(ADSolver.jvp), 10),
    "vjp": (ADSolver, inspect.unwrap(ADSolver.vjp), 6),
}
CONSTRUCTORS: dict[str, Callable[..., object]] = {"CudaSolver": CudaSolver.__init__, "ADSolver": ADSolver.__init__}


@pytest.mark.parametrize("name", [*CALLS, *CONSTRUCTORS])
def test_every_optional_argument_is_keyword_only(name):
    # CudaSolver(net, device, True) set drift to 1 with photolysis off, and solver(..., 1e-25, 1e-3) swapped the
    # tolerances; checked one parameter at a time, so making any single one positional again fails here
    function = CALLS[name][1] if name in CALLS else CONSTRUCTORS[name]
    parameters = list(inspect.signature(function).parameters.values())[1:]  # past self
    positional = [p for p in parameters if p.kind is not inspect.Parameter.KEYWORD_ONLY]
    assert [p.name for p in positional if p.default is not p.empty] == []
    assert len(positional) == (CALLS[name][2] if name in CALLS else 2)  # the tensors, or net and device


@pytest.mark.parametrize("name", CALLS)
def test_settings_by_position_are_refused(name):
    cls, function, inputs = CALLS[name]
    solver: object = object.__new__(cls)  # binding fails before the body reads any of the solver's state
    with pytest.raises(TypeError, match=r"positional arguments? but \d+ were given"):
        function(solver, *([None] * inputs), 1e-25)


def test_results_are_named_tuples_in_the_old_order():
    fields: dict[Any, tuple[str, ...]] = {  # Any: mypy types a NamedTuple class without its _fields
        CallResult: ("x", "h", "done", "steps"),
        LaunchResult: ("x", "h", "done", "steps", "flag"),
        JvpResult: ("x", "h", "done", "steps", "dx", "dh"),
        VjpResult: ("x", "h", "done", "steps", "cotangents"),
    }
    for result, names in fields.items():
        assert result._fields == names
        assert tuple(result(*names)) == names  # unpacks and indexes as the bare tuple did
