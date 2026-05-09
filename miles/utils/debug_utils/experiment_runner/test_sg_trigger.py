"""Unit tests for sg_trigger conversion helpers (no network)."""

import json
from pathlib import Path

import pytest

from miles.utils.debug_utils.experiment_runner.sg_trigger import (
    baseline_entries_from_meta,
    write_baseline_logprob_json,
)


def test_baseline_entries_skip_index_0() -> None:
    meta = {
        "input_token_logprobs": [
            [None, 100, None],
            [-1.5, 200, None],
            [-2.0, 300, None],
        ]
    }
    entries = baseline_entries_from_meta(meta)
    assert entries == [
        {"global_position": 0, "token_id": 200, "logprob": -1.5, "is_valid": True},
        {"global_position": 1, "token_id": 300, "logprob": -2.0, "is_valid": True},
    ]


def test_baseline_entries_skip_none_logprob_mid_stream() -> None:
    meta = {
        "input_token_logprobs": [
            [None, 100, None],
            [-1.5, 200, None],
            [None, 300, None],
            [-3.0, 400, None],
        ]
    }
    entries = baseline_entries_from_meta(meta)
    assert [e["global_position"] for e in entries] == [0, 2]
    assert [e["token_id"] for e in entries] == [200, 400]


def test_baseline_entries_empty_meta_yields_empty_list() -> None:
    assert baseline_entries_from_meta({}) == []


def test_write_baseline_logprob_json_schema_matches_megatron(tmp_path: Path) -> None:
    meta = {
        "input_token_logprobs": [
            [None, 100, None],
            [-0.5, 200, None],
        ]
    }
    out = tmp_path / "rank_0.json"
    n = write_baseline_logprob_json(meta_info=meta, output_path=out)
    payload = json.loads(out.read_text())

    assert n == 1
    assert payload["rank"] == 0
    assert payload["tp_size"] == 1
    assert payload["cp_size"] == 1
    assert payload["pp_size"] == 1
    assert len(payload["logprob_entries"]) == 1
    assert payload["logprob_entries"][0] == [
        {"global_position": 0, "token_id": 200, "logprob": -0.5, "is_valid": True},
    ]


def test_write_baseline_logprob_json_creates_parent_dirs(tmp_path: Path) -> None:
    meta = {"input_token_logprobs": [[None, 1, None], [-1.0, 2, None]]}
    out = tmp_path / "nested" / "sub" / "rank_0.json"
    write_baseline_logprob_json(meta_info=meta, output_path=out)
    assert out.exists()


def test_baseline_entries_handles_none_top_level_field() -> None:
    """Server may return None instead of empty list when no logprobs requested."""
    meta = {"input_token_logprobs": None}
    assert baseline_entries_from_meta(meta) == []


def test_baseline_entries_skip_none_entry_object() -> None:
    """Defensive against any layer that returns None for an item."""
    meta = {
        "input_token_logprobs": [
            [None, 1, None],
            None,
            [-1.0, 2, None],
        ]
    }
    entries = baseline_entries_from_meta(meta)
    assert len(entries) == 1
    assert entries[0]["global_position"] == 1
    assert entries[0]["token_id"] == 2


@pytest.mark.parametrize(
    "lp_value,token,expected_logprob",
    [
        (-1.5, 999, -1.5),
        (0.0, 1, 0.0),
        (-12.345, 42, -12.345),
    ],
)
def test_baseline_entries_preserves_float_precision(lp_value: float, token: int, expected_logprob: float) -> None:
    meta = {"input_token_logprobs": [[None, 0, None], [lp_value, token, None]]}
    entries = baseline_entries_from_meta(meta)
    assert entries[0]["logprob"] == pytest.approx(expected_logprob)
    assert entries[0]["token_id"] == token
