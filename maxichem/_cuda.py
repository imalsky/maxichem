"""The CUDA driver and NVRTC through ctypes: compile, load and launch kernels on torch's stream."""

import contextlib
import ctypes
import functools
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import nvidia.cuda_nvrtc
import torch

SOLVER_KERNELS = "kincuda.cu"  # the kernel file that holds the call, guard and recovery kernels
# CUfunction_attribute values of the driver API
ATTR_LOCAL_BYTES = 3
ATTR_REGISTERS = 4
ATTR_MAX_DYNAMIC_SHARED_BYTES = 8
FP_CONTRACTION = b"--fmad=true"  # the default, stated: the kernels' bits depend on which operations fuse
NVRTC_LIBRARY = "libnvrtc.so.12"  # the library inside the nvidia-cuda-nvrtc wheel
_CUDA: list[ctypes.CDLL] = []  # the driver library once loaded: a cache


def driver() -> ctypes.CDLL:
    """Load the CUDA driver library (ctypes) once and return it."""
    if not _CUDA:
        lib = ctypes.CDLL("libcuda.so.1")
        vp, ui = ctypes.c_void_p, ctypes.c_uint
        lib.cuLaunchKernel.argtypes = [vp, ui, ui, ui, ui, ui, ui, ui, vp, ctypes.POINTER(vp), vp]
        lib.cuModuleLoadData.argtypes = [ctypes.POINTER(vp), ctypes.c_char_p]
        lib.cuModuleGetFunction.argtypes = [ctypes.POINTER(vp), vp, ctypes.c_char_p]
        lib.cuModuleUnload.argtypes = [vp]
        lib.cuFuncSetAttribute.argtypes = [vp, ctypes.c_int, ctypes.c_int]
        lib.cuFuncGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, vp]
        lib.cuOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            vp,
            ctypes.c_int,
            ctypes.c_size_t,
        ]
        _CUDA.append(lib)
    return _CUDA[0]


def check_driver(result: int, what: str) -> None:
    """Raise RuntimeError, with the driver's message, unless a driver call succeeded."""
    if result != 0:
        message = ctypes.c_char_p()
        driver().cuGetErrorString(result, ctypes.byref(message))
        raise RuntimeError(f"CUDA driver, {what}: error {result} ({(message.value or b'').decode()})")


def compile_cubin(name: str, sources: dict[str, str], arch: str) -> bytes:
    """Compile the file name, a key of sources (the others are headers), with the wheel's NVRTC; arch as in sm_89."""
    nv = ctypes.CDLL(str(Path(next(iter(nvidia.cuda_nvrtc.__path__))) / "lib" / NVRTC_LIBRARY))
    nv.nvrtcGetErrorString.restype = ctypes.c_char_p

    def chk(result: int, log: str = "") -> None:
        if result != 0:
            raise RuntimeError(f"NVRTC: {nv.nvrtcGetErrorString(result).decode()}\n{log}")

    prog = ctypes.c_void_p()
    headers = [key for key in sources if key != name]
    texts = (ctypes.c_char_p * len(headers))(*[sources[key].encode() for key in headers])
    names = (ctypes.c_char_p * len(headers))(*[key.encode() for key in headers])
    chk(nv.nvrtcCreateProgram(ctypes.byref(prog), sources[name].encode(), name.encode(), len(headers), texts, names))
    try:
        opts = [f"-arch=sm_{arch}".encode(), b"-std=c++17", FP_CONTRACTION]
        result = nv.nvrtcCompileProgram(prog, len(opts), (ctypes.c_char_p * len(opts))(*opts))
        size = ctypes.c_size_t()
        chk(nv.nvrtcGetProgramLogSize(prog, ctypes.byref(size)))
        log = ctypes.create_string_buffer(size.value)
        chk(nv.nvrtcGetProgramLog(prog, log))
        chk(result, log.value.decode())
        chk(nv.nvrtcGetCUBINSize(prog, ctypes.byref(size)))
        cubin = ctypes.create_string_buffer(size.value)
        chk(nv.nvrtcGetCUBIN(prog, cubin))
    finally:
        nv.nvrtcDestroyProgram(ctypes.byref(prog))
    return cubin.raw


def launch_kernel(fn: ctypes.c_void_p, grid: int, block: int, smem: int, stream: int, args: Iterable[Any]) -> None:
    """Launch a kernel on a stream with args: tensors (passed as pointers) or ctypes scalars."""
    vals = [ctypes.c_void_p(a.data_ptr()) if isinstance(a, torch.Tensor) else a for a in args]
    ptrs = (ctypes.c_void_p * len(vals))(*[ctypes.addressof(v) for v in vals])
    check_driver(driver().cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, smem, stream, ptrs, None), "launch")


def load_module(
    cubin: bytes, names: Iterable[str], loaded: list[ctypes.c_void_p]
) -> tuple[ctypes.c_void_p, list[ctypes.c_void_p]]:
    """Load a cubin and return its module with its kernels, by the names given.

    The module's handle goes into ``loaded`` first, so a caller can unload it if a later step fails.
    """
    cu = driver()
    module = ctypes.c_void_p()
    check_driver(cu.cuModuleLoadData(ctypes.byref(module), cubin), "load")
    loaded.append(module)
    functions = []
    for name in names:
        fn = ctypes.c_void_p()
        check_driver(cu.cuModuleGetFunction(ctypes.byref(fn), module, name.encode()), f"function {name}")
        functions.append(fn)
    return module, functions


def unload_modules(device: torch.device, modules: list[ctypes.c_void_p]) -> None:
    """Unload CUDA modules on their device, after the device's work is done, and empty the list."""
    with torch.cuda.device(device):
        torch.cuda.synchronize(device)
        while modules:
            check_driver(driver().cuModuleUnload(modules.pop()), "unload")


def unload_at_deletion(device: torch.device, modules: list[ctypes.c_void_p]) -> None:
    """Unload what a solver still holds when it is deleted; a destructor must not raise, the process may be ending."""
    with contextlib.suppress(Exception):
        if modules:
            unload_modules(device, modules)


def opt_in_shared_memory(fn: ctypes.c_void_p, smem: int) -> None:
    """Let a kernel use this much shared memory per block (bytes)."""
    check_driver(driver().cuFuncSetAttribute(fn, ATTR_MAX_DYNAMIC_SHARED_BYTES, smem), "shared memory")


def blocks_per_multiprocessor(fn: ctypes.c_void_p, threads: int, smem: int) -> int:
    """Opt a kernel in to this shared memory and return the blocks one multiprocessor holds at once."""
    opt_in_shared_memory(fn, smem)
    blocks = ctypes.c_int()
    occupancy = driver().cuOccupancyMaxActiveBlocksPerMultiprocessor(ctypes.byref(blocks), fn, threads, smem)
    check_driver(occupancy, "occupancy")
    if blocks.value < 1:
        raise RuntimeError(f"a kernel of {threads} threads and {smem} B of shared memory does not fit on this device")
    return blocks.value


def device_of(device: torch.device | str) -> torch.device:
    """Return the CUDA device with its index (a bare "cuda" means the current device)."""
    d = torch.device(device)
    return d if d.index is not None else torch.device(d.type, torch.cuda.current_device())


def on_device(method: Callable[..., Any]) -> Callable[..., Any]:
    """Run a solver's method with its device current.

    The driver API (module loads, kernel launches) acts on the current device, which torch restores after each of its
    own calls.
    """

    @functools.wraps(method)
    def run(self: Any, *args: Any, **kwargs: Any) -> Any:
        with torch.cuda.device(self.dev):
            return method(self, *args, **kwargs)

    return run


def function_attribute(fn: ctypes.c_void_p, attribute: int) -> int:
    """Return a kernel's attribute: ``ATTR_LOCAL_BYTES`` or ``ATTR_REGISTERS``, per thread."""
    value = ctypes.c_int()
    check_driver(driver().cuFuncGetAttribute(ctypes.byref(value), attribute, fn), "attribute")
    return value.value
