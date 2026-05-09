"""Unit tests for comparator metrics + jsonl writer."""

import json
from pathlib import Path

from miles.utils.debug_utils.experiment_runner.comparator import compare_logprob_dirs, write_comparison_jsonl


def _write_rank_json(directory: Path, *, entries: list[dict], rank: int = 0) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "rank": rank,
        "tp_size": 1,
        "cp_size": 1,
        "pp_size": 1,
        "logprob_entries": [entries],
    }
    (directory / f"rank_{rank}.json").write_text(json.dumps(payload))


def _entries(values: list[tuple[int, int, float]]) -> list[dict]:
    return [{"global_position": p, "token_id": t, "logprob": lp, "is_valid": True} for p, t, lp in values]


def test_compare_two_dirs_zero_diff(tmp_path: Path) -> None:
    base = tmp_path / "base"
    tgt = tmp_path / "tgt"
    same = _entries([(0, 1, -1.0), (1, 2, -2.0), (2, 3, -3.0)])
    _write_rank_json(base, entries=same)
    _write_rank_json(tgt, entries=same)

    cmp = compare_logprob_dirs(baseline_dir=base, target_dir=tgt)
    assert cmp.all.num_positions == 3
    assert cmp.all.mean_abs_diff == 0.0
    assert cmp.all.max_abs_diff == 0.0


def test_compare_two_dirs_constant_diff(tmp_path: Path) -> None:
    base = tmp_path / "base"
    tgt = tmp_path / "tgt"
    _write_rank_json(base, entries=_entries([(0, 1, -1.0), (1, 2, -2.0)]))
    _write_rank_json(tgt, entries=_entries([(0, 1, -1.5), (1, 2, -2.5)]))

    cmp = compare_logprob_dirs(baseline_dir=base, target_dir=tgt)
    assert cmp.all.num_positions == 2
    assert abs(cmp.all.mean_abs_diff - 0.5) < 1e-9
    assert abs(cmp.all.max_abs_diff - 0.5) < 1e-9


def test_compare_split_by_prompt_length(tmp_path: Path) -> None:
    base = tmp_path / "base"
    tgt = tmp_path / "tgt"
    # positions: 0..3 prompt, 4..5 response (prompt_length=5 -> response starts at p>=4)
    _write_rank_json(
        base,
        entries=_entries([(0, 1, -1.0), (1, 2, -1.0), (2, 3, -1.0), (3, 4, -1.0), (4, 5, -1.0), (5, 6, -1.0)]),
    )
    _write_rank_json(
        tgt,
        entries=_entries([(0, 1, -1.1), (1, 2, -1.1), (2, 3, -1.1), (3, 4, -1.1), (4, 5, -1.5), (5, 6, -1.5)]),
    )

    cmp = compare_logprob_dirs(baseline_dir=base, target_dir=tgt, prompt_length=5)
    assert cmp.prompt is not None and cmp.response is not None
    # prompt positions are p < 4: positions 0, 1, 2, 3 -> 4 entries, all diff 0.1
    assert cmp.prompt.num_positions == 4
    assert abs(cmp.prompt.mean_abs_diff - 0.1) < 1e-9
    # response positions p >= 4: positions 4, 5 -> 2 entries, diff 0.5
    assert cmp.response.num_positions == 2
    assert abs(cmp.response.mean_abs_diff - 0.5) < 1e-9
    assert cmp.prompt_length == 5


def test_compare_handles_empty_overlap(tmp_path: Path) -> None:
    base = tmp_path / "base"
    tgt = tmp_path / "tgt"
    _write_rank_json(base, entries=_entries([(0, 1, -1.0)]))
    _write_rank_json(tgt, entries=_entries([(99, 1, -1.0)]))
    cmp = compare_logprob_dirs(baseline_dir=base, target_dir=tgt)
    assert cmp.all.num_positions == 0


def test_jsonl_appends_row(tmp_path: Path) -> None:
    base = tmp_path / "base"
    tgt = tmp_path / "tgt"
    _write_rank_json(base, entries=_entries([(0, 1, -1.0)]))
    _write_rank_json(tgt, entries=_entries([(0, 1, -1.5)]))

    cmp = compare_logprob_dirs(baseline_dir=base, target_dir=tgt)
    out = tmp_path / "comparisons.jsonl"
    write_comparison_jsonl(cmp, target_run_name="exp_a", baseline_run_name="canonical", jsonl_path=out)
    write_comparison_jsonl(cmp, target_run_name="exp_b", baseline_run_name="canonical", jsonl_path=out)

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["target"] == "exp_a"
    assert rows[1]["target"] == "exp_b"
    assert rows[0]["baseline"] == "canonical"
    assert "all" in rows[0]


def test_to_jsonable_includes_optional_ranges(tmp_path: Path) -> None:
    base = tmp_path / "base"
    tgt = tmp_path / "tgt"
    _write_rank_json(base, entries=_entries([(0, 1, -1.0), (1, 2, -1.0)]))
    _write_rank_json(tgt, entries=_entries([(0, 1, -1.5), (1, 2, -1.0)]))
    cmp_no_split = compare_logprob_dirs(baseline_dir=base, target_dir=tgt)
    cmp_with_split = compare_logprob_dirs(baseline_dir=base, target_dir=tgt, prompt_length=1)

    j1 = cmp_no_split.to_jsonable()
    j2 = cmp_with_split.to_jsonable()
    assert "prompt" not in j1 and "response" not in j1
    assert "prompt" in j2 and "response" in j2
    assert j2["prompt_length"] == 1
