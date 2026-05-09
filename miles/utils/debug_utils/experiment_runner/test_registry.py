"""Unit tests for registry append/load + markdown rendering."""

import json
from pathlib import Path

from miles.utils.debug_utils.experiment_runner.registry import (
    RunRecord,
    append_run,
    find_latest_run,
    load_runs,
    render_markdown,
)


def _record(name: str, **overrides) -> RunRecord:
    base = dict(
        name=name,
        description=f"description for {name}",
        kind="grafter_pair",
        canonical_name="v4_canonical",
        started_at="2026-05-08T10:00:00Z",
        finished_at="2026-05-08T10:30:00Z",
        sg_dump_dir=f"/storage/dumper-out/{name}-sg",
        sg_baseline_json=None,
        mg_dump_dir=f"/storage/dumper-out/{name}-mg",
        mg_logprob_dir=f"/storage/dumper-out/{name}-mg/megatron_logprobs",
        git={"miles": "abc123", "sglang": "def456"},
        image="radixark/miles:deepseek-v4@sha256:e437...",
        spec_snapshot={"name": name},
        run_config_snapshot={"name": name},
    )
    base.update(overrides)
    return RunRecord(**base)


def test_append_and_load_run(tmp_path: Path) -> None:
    p = tmp_path / "runs.jsonl"
    append_run(_record("a1"), jsonl_path=p)
    append_run(_record("a2"), jsonl_path=p)
    runs = load_runs(p)
    assert [r.name for r in runs] == ["a1", "a2"]
    assert runs[0].git == {"miles": "abc123", "sglang": "def456"}


def test_load_runs_empty_path(tmp_path: Path) -> None:
    assert load_runs(tmp_path / "missing.jsonl") == []


def test_find_latest_run_picks_newest(tmp_path: Path) -> None:
    p = tmp_path / "runs.jsonl"
    append_run(_record("a1", finished_at="2026-05-08T10:00:00Z"), jsonl_path=p)
    append_run(_record("a1", finished_at="2026-05-08T12:00:00Z"), jsonl_path=p)
    append_run(_record("a2", finished_at="2026-05-08T11:00:00Z"), jsonl_path=p)

    runs = load_runs(p)
    latest = find_latest_run(runs, "a1")
    assert latest is not None
    assert latest.finished_at == "2026-05-08T12:00:00Z"


def test_find_latest_run_missing(tmp_path: Path) -> None:
    assert find_latest_run([], "nope") is None


def _write_cmp(jsonl_path: Path, *, target: str, baseline: str, mean: float, max_: float, p99: float) -> None:
    row = {
        "target": target,
        "baseline": baseline,
        "all": {
            "num_positions": 100,
            "mean_abs_diff": mean,
            "median_abs_diff": mean,
            "p95_abs_diff": p99,
            "p99_abs_diff": p99,
            "max_abs_diff": max_,
        },
    }
    with jsonl_path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def test_render_markdown_with_one_baseline(tmp_path: Path) -> None:
    runs_p = tmp_path / "runs.jsonl"
    cmp_p = tmp_path / "cmp.jsonl"
    append_run(_record("a1"), jsonl_path=runs_p)
    append_run(_record("a2"), jsonl_path=runs_p)
    _write_cmp(cmp_p, target="a1", baseline="canonical", mean=1.228, max_=26.72, p99=19.19)
    _write_cmp(cmp_p, target="a2", baseline="canonical", mean=1.336, max_=29.89, p99=21.28)

    md = render_markdown(runs_p, cmp_p)
    assert "| run | kind | description | canonical:mean | canonical:max | canonical:p99 |" in md
    assert "| a1 | grafter_pair | description for a1 | 1.228 | 26.72 | 19.19 |" in md
    assert "| a2 | grafter_pair | description for a2 | 1.336 | 29.89 | 21.28 |" in md


def test_render_markdown_with_multiple_baselines(tmp_path: Path) -> None:
    runs_p = tmp_path / "runs.jsonl"
    cmp_p = tmp_path / "cmp.jsonl"
    append_run(_record("a1"), jsonl_path=runs_p)
    _write_cmp(cmp_p, target="a1", baseline="canonical", mean=1.0, max_=10.0, p99=5.0)
    _write_cmp(cmp_p, target="a1", baseline="sg-noise", mean=0.001, max_=0.5, p99=0.01)

    md = render_markdown(runs_p, cmp_p, baseline_order=["canonical", "sg-noise"])
    header_line = md.splitlines()[0]
    assert header_line.index("canonical:mean") < header_line.index("sg-noise:mean")
    assert "0.001" in md
    assert "1" in md  # canonical mean


def test_render_markdown_handles_missing_comparison(tmp_path: Path) -> None:
    """Run present in runs.jsonl but no comparison row -- empty cells."""
    runs_p = tmp_path / "runs.jsonl"
    cmp_p = tmp_path / "cmp.jsonl"
    append_run(_record("a1"), jsonl_path=runs_p)
    append_run(_record("a2"), jsonl_path=runs_p)
    _write_cmp(cmp_p, target="a1", baseline="canonical", mean=1.0, max_=10.0, p99=5.0)

    md = render_markdown(runs_p, cmp_p)
    a2_line = next(line for line in md.splitlines() if line.startswith("| a2 "))
    assert "|  |  |  |" in a2_line  # three empty cells for the missing baseline


def test_render_markdown_metric_range_response(tmp_path: Path) -> None:
    runs_p = tmp_path / "runs.jsonl"
    cmp_p = tmp_path / "cmp.jsonl"
    append_run(_record("a1"), jsonl_path=runs_p)
    row = {
        "target": "a1",
        "baseline": "canonical",
        "all": {
            "num_positions": 100,
            "mean_abs_diff": 1.0,
            "median_abs_diff": 1.0,
            "p95_abs_diff": 1.0,
            "p99_abs_diff": 1.0,
            "max_abs_diff": 1.0,
        },
        "response": {
            "num_positions": 32,
            "mean_abs_diff": 2.5,
            "median_abs_diff": 2.5,
            "p95_abs_diff": 2.5,
            "p99_abs_diff": 2.5,
            "max_abs_diff": 2.5,
        },
        "prompt_length": 68,
    }
    with cmp_p.open("a") as fh:
        fh.write(json.dumps(row) + "\n")

    md_all = render_markdown(runs_p, cmp_p, metric_range="all")
    md_resp = render_markdown(runs_p, cmp_p, metric_range="response")

    assert "1" in md_all
    assert "2.5" in md_resp


def test_run_record_to_dict_roundtrip() -> None:
    r = _record("a1")
    d = r.to_dict()
    assert RunRecord.from_dict(d) == r
