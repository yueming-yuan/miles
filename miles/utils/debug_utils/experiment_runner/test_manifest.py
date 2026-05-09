"""Unit tests for manifest writing."""

import json
import subprocess
from pathlib import Path

from miles.utils.debug_utils.experiment_runner.manifest import GitInfo, collect_git, write_manifest


def test_write_manifest_round_trips(tmp_path: Path) -> None:
    git_info = GitInfo(repo_dir="/fake/miles", head_sha="abc123def", head_short="abc123d", is_dirty=False)
    path = write_manifest(
        output_dir=tmp_path,
        run_config_snapshot={"name": "a1", "tp": 8},
        spec_snapshot={"name": "a1", "kind": "grafter_pair"},
        git_infos={"miles": git_info},
        image="radixark/miles:deepseek-v4@sha256:abc",
        started_at="2026-05-08T10:00:00Z",
        finished_at="2026-05-08T10:30:00Z",
    )
    assert path == tmp_path / "manifest.json"
    payload = json.loads(path.read_text())
    assert payload["spec_snapshot"]["name"] == "a1"
    assert payload["run_config_snapshot"]["tp"] == 8
    assert payload["git"]["miles"]["head_sha"] == "abc123def"
    assert payload["git"]["miles"]["is_dirty"] is False
    assert payload["image"] == "radixark/miles:deepseek-v4@sha256:abc"


def test_collect_git_returns_none_for_non_repo(tmp_path: Path) -> None:
    assert collect_git(tmp_path) is None


def test_collect_git_in_initialised_repo(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@test"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "test"], check=True)
    (tmp_path / "file.txt").write_text("hello")
    subprocess.run(["git", "-C", str(tmp_path), "add", "file.txt"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "init"], check=True)

    info = collect_git(tmp_path)
    assert info is not None
    assert len(info.head_sha) == 40
    assert len(info.head_short) >= 7
    assert info.is_dirty is False


def test_collect_git_detects_dirty(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@test"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "test"], check=True)
    (tmp_path / "file.txt").write_text("hello")
    subprocess.run(["git", "-C", str(tmp_path), "add", "file.txt"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "init"], check=True)
    (tmp_path / "file.txt").write_text("changed")

    info = collect_git(tmp_path)
    assert info is not None
    assert info.is_dirty is True
