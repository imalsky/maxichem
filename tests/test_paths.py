"""The data locations read their environment variables when used, and missing data give a clear error."""

import json

import libref
import numpy as np
import pytest

from bench import kintest_truth
from maxichem import _constants, paths


def test_data_root_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("CHEM_DATA", str(tmp_path))
    assert tmp_path == paths.data()
    assert paths.libref().is_relative_to(tmp_path)
    assert paths.truth("boxes_ncho") == tmp_path / "truth_v3" / "boxes_ncho"


def test_unset_thermo_folder_says_what_to_set(monkeypatch):
    monkeypatch.delenv("VULCAN_THERMO", raising=False)
    with pytest.raises(RuntimeError, match="VULCAN_THERMO"):
        paths.vulcan_thermo()


def test_need_names_what_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="absent"):
        paths.need(tmp_path, tmp_path / "absent")


@pytest.mark.parametrize("rule", [None, "another rule", _constants.RATE_RULE])
def test_reference_of_another_rate_rule_is_refused(tmp_path, rule):
    path = tmp_path / "reference.npz"
    np.savez(path, atol=1e-25, **({"rate_rule": rule} if rule else {}))
    if rule == _constants.RATE_RULE:
        assert float(libref.load(path)["atol"]) == 1e-25
        return
    with pytest.raises(ValueError, match=f"rate rule '{rule or 'none recorded'}', not '{_constants.RATE_RULE}'"):
        libref.load(path)


@pytest.mark.parametrize("rule", [None, "another rule", _constants.RATE_RULE])
def test_truth_path_of_another_rate_rule_is_refused(tmp_path, rule):
    qual = tmp_path / "qual"
    (qual / "QC").mkdir(parents=True)
    (qual / "manifest.json").write_text(json.dumps({"rate_rule": _constants.RATE_RULE}))
    spec = json.dumps({"job": {}} | ({"rate_rule": rule} if rule else {}))
    np.savez(tmp_path / "src.npz", T=np.ones(3))
    np.savez(qual / "QC" / "p.npz", kind="main", x_call=np.ones((3, 2)), spec=spec)
    jobs = [{"src": str(tmp_path / "src.npz"), "out": str(qual / "QC" / "p.npz"), "n": 3}]
    (qual / "jobs.json").write_text(json.dumps(jobs))
    if rule == _constants.RATE_RULE:
        kintest_truth.check_jobs(qual)
        return
    with pytest.raises(SystemExit, match="1 truth paths made with another rate rule"):
        kintest_truth.check_jobs(qual)


@pytest.mark.parametrize("rule", [None, "another rule", _constants.RATE_RULE])
def test_truth_of_another_rate_rule_is_refused(tmp_path, rule):
    (tmp_path / "manifest.json").write_text(json.dumps({"boxes": 1} | ({"rate_rule": rule} if rule else {})))
    if rule == _constants.RATE_RULE:
        assert kintest_truth.rate_rule_manifest(tmp_path)["boxes"] == 1
        return
    with pytest.raises(SystemExit, match=f"rate rule '{rule or 'none recorded'}', not '{_constants.RATE_RULE}'"):
        kintest_truth.rate_rule_manifest(tmp_path)
