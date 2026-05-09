"""Unit tests for orchestrator pure helpers (no subprocess / network)."""

from pathlib import Path

import torch

from miles.utils.debug_utils.experiment_runner.orchestrator import (
    _load_first_sample_tokens,
    _pick_baseline_logprob_dir,
    _pick_logprob_dir,
    _run_config_to_dict,
    _RunOutputs,
    _spec_to_dict,
)
from miles.utils.debug_utils.experiment_runner.registry import RunRecord
from miles.utils.debug_utils.experiment_runner.spec import CanonicalConfig, ExperimentSpec, RunKind, compose_run_config


def _record(name: str = "x", **overrides) -> RunRecord:
    base = dict(
        name=name,
        description="",
        kind="grafter_pair",
        canonical_name="c",
        started_at="2026-05-08T10:00:00Z",
        finished_at="2026-05-08T10:30:00Z",
        sg_dump_dir=None,
        sg_baseline_json=None,
        mg_dump_dir=None,
        mg_logprob_dir=None,
        git={},
        image=None,
        spec_snapshot={},
        run_config_snapshot={},
    )
    base.update(overrides)
    return RunRecord(**base)


def test_spec_to_dict_serialises_kind_as_string() -> None:
    spec = ExperimentSpec(name="x", description="", kind=RunKind.SG_ONLY)
    d = _spec_to_dict(spec)
    assert d["kind"] == "sg_only"
    assert d["name"] == "x"


def test_run_config_to_dict_serialises_kind_as_string() -> None:
    spec = ExperimentSpec(name="x", description="", kind=RunKind.MG_ONLY)
    canonical = CanonicalConfig(name="c")
    cfg = compose_run_config(canonical, spec)
    d = _run_config_to_dict(cfg)
    assert d["kind"] == "mg_only"


def test_pick_logprob_dir_prefers_outputs_mg(tmp_path: Path) -> None:
    record = _record(mg_logprob_dir=str(tmp_path / "from_record"))
    outputs = _RunOutputs(mg_logprob_dir=tmp_path / "from_outputs")
    assert _pick_logprob_dir(record, outputs) == tmp_path / "from_outputs"


def test_pick_logprob_dir_falls_back_to_record_mg(tmp_path: Path) -> None:
    record = _record(mg_logprob_dir=str(tmp_path / "rec"))
    outputs = _RunOutputs()
    assert _pick_logprob_dir(record, outputs) == Path(str(tmp_path / "rec"))


def test_pick_logprob_dir_falls_back_to_sg_baseline_parent(tmp_path: Path) -> None:
    record = _record()
    outputs = _RunOutputs(sg_baseline_json=tmp_path / "sg" / "rank_0.json")
    assert _pick_logprob_dir(record, outputs) == tmp_path / "sg"


def test_pick_logprob_dir_returns_none_when_no_source() -> None:
    assert _pick_logprob_dir(_record(), _RunOutputs()) is None


def test_pick_baseline_logprob_dir_prefers_sg_baseline(tmp_path: Path) -> None:
    record = _record(
        sg_baseline_json=str(tmp_path / "sg" / "rank_0.json"),
        mg_logprob_dir=str(tmp_path / "mg"),
    )
    assert _pick_baseline_logprob_dir(record) == tmp_path / "sg"


def test_pick_baseline_logprob_dir_falls_back_to_mg(tmp_path: Path) -> None:
    record = _record(mg_logprob_dir=str(tmp_path / "mg"))
    assert _pick_baseline_logprob_dir(record) == tmp_path / "mg"


def test_pick_baseline_logprob_dir_returns_none_when_no_source() -> None:
    assert _pick_baseline_logprob_dir(_record()) is None


def test_load_first_sample_tokens_supports_list_and_tensor(tmp_path: Path) -> None:
    pt_list = tmp_path / "list.pt"
    torch.save({"samples": [{"tokens": [1, 2, 3], "response_length": 1}]}, pt_list)
    assert _load_first_sample_tokens(pt_list) == [1, 2, 3]

    pt_tensor = tmp_path / "tensor.pt"
    torch.save({"samples": [{"tokens": torch.tensor([4, 5, 6]), "response_length": 1}]}, pt_tensor)
    toks = _load_first_sample_tokens(pt_tensor)
    assert list(toks) == [4, 5, 6]


def test_run_outputs_default_is_all_none() -> None:
    outputs = _RunOutputs()
    assert outputs.sg_dump_dir is None
    assert outputs.sg_baseline_json is None
    assert outputs.mg_dump_dir is None
    assert outputs.mg_logprob_dir is None


def test_subprocess_blocking_raises_on_timeout() -> None:
    """Verify the mg subprocess timeout actually kills a hung process."""
    import pytest as _pytest

    from miles.utils.debug_utils.experiment_runner.orchestrator import _run_subprocess_blocking

    with _pytest.raises(RuntimeError, match="exceeded timeout"):
        _run_subprocess_blocking("sleep 30", env={}, log_label="t", timeout_s=1)


def test_subprocess_blocking_passes_zero_exit() -> None:
    from miles.utils.debug_utils.experiment_runner.orchestrator import _run_subprocess_blocking

    _run_subprocess_blocking("true", env={}, log_label="t", timeout_s=5)


def test_subprocess_blocking_raises_on_nonzero_exit() -> None:
    import pytest as _pytest

    from miles.utils.debug_utils.experiment_runner.orchestrator import _run_subprocess_blocking

    with _pytest.raises(RuntimeError, match="rc=7"):
        _run_subprocess_blocking("exit 7", env={}, log_label="t", timeout_s=5)


def test_runner_options_default_mg_timeout_is_one_hour() -> None:
    from miles.utils.debug_utils.experiment_runner.orchestrator import RunnerOptions

    opts = RunnerOptions(runs_jsonl=Path("/x"), comparisons_jsonl=Path("/y"))
    assert opts.mg_run_timeout_s == 3600
