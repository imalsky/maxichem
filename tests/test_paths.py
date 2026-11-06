"""The data locations read their environment variables when used, and missing data give a clear error."""

import pytest

from maxichem import paths


def test_data_root_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("CHEM_DATA", str(tmp_path))
    assert tmp_path == paths.DATA
    assert paths.LIBREF.is_relative_to(tmp_path)


def test_unset_thermo_folder_says_what_to_set(monkeypatch):
    monkeypatch.delenv("VULCAN_THERMO", raising=False)
    with pytest.raises(RuntimeError, match="VULCAN_THERMO"):
        _ = paths.VULCAN_THERMO


def test_need_names_what_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="absent"):
        paths.need(tmp_path, tmp_path / "absent")
