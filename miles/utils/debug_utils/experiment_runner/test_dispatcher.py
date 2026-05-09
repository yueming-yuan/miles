"""Unit tests for dispatcher (mock subprocess)."""

from pathlib import Path
from unittest.mock import patch

import pytest

from miles.utils.debug_utils.experiment_runner.dispatcher import _build_remote_command, dispatch


def test_build_remote_command_quotes_paths_and_module() -> None:
    cmd = _build_remote_command(
        experiment="a1",
        miles_repo_dir="/workspace/miles with space",
        registry_module="my_proj.registry",
    )
    assert "/workspace/miles with space" in cmd or "'/workspace/miles with space'" in cmd
    assert "--exp a1" in cmd
    assert "--registry-module my_proj.registry" in cmd


def test_dispatch_empty_experiments_returns_empty(tmp_path: Path) -> None:
    results = dispatch(
        experiments=[],
        pods=["pod1"],
        registry_module="x.y",
        log_dir=tmp_path,
    )
    assert results == []


def test_dispatch_no_pods_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one pod"):
        dispatch(experiments=["a"], pods=[], registry_module="x.y", log_dir=tmp_path)


def test_dispatch_runs_one_per_pod_until_pool_exhausted(tmp_path: Path) -> None:
    """3 experiments, 2 pods -> 3 results, both pods used at least once."""
    pods_used: list[str] = []

    class _FakeReturn:
        returncode = 0

    def _fake_run(cmd, stdout=None, stderr=None, check=False):
        # cmd[0]=rcli, cmd[1]=exec, cmd[2]=pod_name, cmd[3]=remote_command
        pods_used.append(cmd[2])
        return _FakeReturn()

    with patch("miles.utils.debug_utils.experiment_runner.dispatcher.subprocess.run", side_effect=_fake_run):
        results = dispatch(
            experiments=["a", "b", "c"],
            pods=["pod1", "pod2"],
            registry_module="x.y",
            log_dir=tmp_path,
        )

    assert len(results) == 3
    assert {r.experiment for r in results} == {"a", "b", "c"}
    assert {r.pod for r in results}.issubset({"pod1", "pod2"})
    assert all(r.return_code == 0 for r in results)


def test_dispatch_preserves_nonzero_return_code(tmp_path: Path) -> None:
    class _FakeReturn:
        returncode = 7

    def _fake_run(cmd, stdout=None, stderr=None, check=False):
        return _FakeReturn()

    with patch("miles.utils.debug_utils.experiment_runner.dispatcher.subprocess.run", side_effect=_fake_run):
        results = dispatch(
            experiments=["a"],
            pods=["pod1"],
            registry_module="x.y",
            log_dir=tmp_path,
        )
    assert len(results) == 1
    assert results[0].return_code == 7


def test_dispatch_log_path_per_experiment(tmp_path: Path) -> None:
    class _FakeReturn:
        returncode = 0

    def _fake_run(cmd, stdout=None, stderr=None, check=False):
        return _FakeReturn()

    with patch("miles.utils.debug_utils.experiment_runner.dispatcher.subprocess.run", side_effect=_fake_run):
        results = dispatch(
            experiments=["a", "b"],
            pods=["pod1"],
            registry_module="x.y",
            log_dir=tmp_path,
        )
    log_paths = {r.log_path for r in results}
    assert len(log_paths) == 2
    for p in log_paths:
        assert "pod1" in p
        assert any(name in p for name in ("a", "b"))
