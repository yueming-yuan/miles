"""Tests for bootstrap helpers (no real dump dirs needed)."""

import json
from pathlib import Path
from unittest.mock import patch

from tools_debug.experiments.bootstrap import _find_first, _legacy_dirs, bootstrap


def test_legacy_dirs_glob_matches_grafter_a1(tmp_path: Path) -> None:
    """Real dirs end with descriptive suffix (q-lora-only); glob handles it."""
    sg_dir = tmp_path / "v4-iter59-grafter-a1-q-lora-only-sg"
    mg_dir = tmp_path / "v4-iter59-grafter-a1-q-lora-only-mg"
    sg_dir.mkdir()
    mg_dir.mkdir()
    dirs = _legacy_dirs("a1", tmp_path)
    assert dirs == {"sg": sg_dir, "mg": mg_dir}


def test_legacy_dirs_glob_matches_combo(tmp_path: Path) -> None:
    sg_dir = tmp_path / "v4-iter59-grafter-combo-lora-proj-router-sg"
    sg_dir.mkdir()
    dirs = _legacy_dirs("combo-lora-proj-router", tmp_path)
    assert dirs == {"sg": sg_dir}


def test_legacy_dirs_direct_baseline_paths(tmp_path: Path) -> None:
    (tmp_path / "v4-iter59-sg-natural-e8env").mkdir()
    (tmp_path / "v4-iter59-mg-fix0505").mkdir()
    sg = _legacy_dirs("sg-natural-prefill-e8env", tmp_path)
    mg = _legacy_dirs("mg-fix0505", tmp_path)
    assert sg == {"sg": tmp_path / "v4-iter59-sg-natural-e8env"}
    assert mg == {"mg": tmp_path / "v4-iter59-mg-fix0505"}


def test_legacy_dirs_returns_empty_when_no_match(tmp_path: Path) -> None:
    assert _legacy_dirs("nonexistent-spec", tmp_path) == {}


def test_find_first_returns_first_existing(tmp_path: Path) -> None:
    (tmp_path / "second.json").write_text("{}")
    (tmp_path / "third.json").write_text("{}")
    p = _find_first(tmp_path, ["first.json", "second.json", "third.json"])
    assert p == tmp_path / "second.json"


def test_find_first_returns_none_for_missing_dir() -> None:
    assert _find_first(None, ["any.json"]) is None


def test_bootstrap_dry_run_does_not_write(tmp_path: Path) -> None:
    runs = tmp_path / "runs.jsonl"
    n = bootstrap(runs_jsonl=runs, dry_run=True)
    assert not runs.exists()
    assert n == 0  # no real dumps in this test env


def test_bootstrap_writes_row_when_dump_dir_exists(tmp_path: Path) -> None:
    fake_root = tmp_path / "dumper-out"
    fake_root.mkdir()
    sg_dir = fake_root / "v4-iter59-grafter-a1-q-lora-only-sg"
    mg_dir = fake_root / "v4-iter59-grafter-a1-q-lora-only-mg"
    sg_dir.mkdir()
    mg_dir.mkdir()
    (mg_dir / "megatron_logprobs").mkdir()
    (sg_dir / "sgl_response.json").write_text("{}")

    runs = tmp_path / "runs.jsonl"
    with patch("tools_debug.experiments.bootstrap.CANONICAL") as mock_canonical:
        mock_canonical.output_root = str(fake_root)
        mock_canonical.name = "v4_e8env"
        n = bootstrap(runs_jsonl=runs, dry_run=False)

    assert n >= 1
    rows = [json.loads(line) for line in runs.read_text().splitlines() if line.strip()]
    a1 = next(r for r in rows if r["name"] == "a1")
    assert a1["sg_dump_dir"] == str(sg_dir)
    assert a1["mg_dump_dir"] == str(mg_dir)
    assert a1["sg_baseline_json"] == str(sg_dir / "sgl_response.json")
    assert a1["mg_logprob_dir"] == str(mg_dir / "megatron_logprobs")
    assert a1["kind"] == "grafter_pair"
