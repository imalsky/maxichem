"""Shared test settings: the invented networks of tests/data."""

from pathlib import Path

import pytest

from maxichem.kinvulcan import VulcanNetwork

DATA = Path(__file__).parent / "data"


@pytest.fixture
def nasa_dir():
    """Return the folder of the invented NASA-9 files."""
    return DATA / "NASA9"


@pytest.fixture
def read_network():
    """Return a function that reads an invented network from tests/data by its stem."""

    def read(stem: str = "synthetic") -> VulcanNetwork:
        return VulcanNetwork(DATA / f"{stem}_network.txt", DATA / "NASA9")

    return read


@pytest.fixture
def network_text():
    """Return the text of the main invented network, for tests that change a line of it."""
    return (DATA / "synthetic_network.txt").read_text()


@pytest.fixture
def verdict():
    """Return a function that prints each case's result; the test fails at its end if any case failed."""
    failed = []

    def record(name: str, ok: bool, text: str) -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {text}", flush=True)
        if not ok:
            failed.append(f"{name}: {text}")

    yield record
    if failed:
        pytest.fail("\n".join(failed))


@pytest.fixture(scope="session")
def gpu():
    """Return the CUDA device, or skip the test where there is none."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("this test needs a GPU")
    return torch.device("cuda")


@pytest.fixture(scope="session", params=["NCHO_photo_network.txt", "SNCHO_photo_network_2025.txt"])
def parity(request, gpu):
    """Return a real network with its stored reference and solvers."""
    helpers = pytest.importorskip("helpers")
    case = helpers.Parity(request.param, gpu)
    yield case
    case.close()
