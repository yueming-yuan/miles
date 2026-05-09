"""Unit tests for ``compose_run_config`` and ``ExperimentSpec`` validation."""

import pytest

from miles.utils.debug_utils.experiment_runner.spec import CanonicalConfig, ExperimentSpec, RunKind, compose_run_config


def _canonical() -> CanonicalConfig:
    return CanonicalConfig(
        name="v4_canonical_test",
        sg_env={"SG_FIX_A": "1", "SG_FIX_B": "1"},
        sg_server_args=("--tp-size", "8", "--port", "30000"),
        mg_env={"MG_FIX_A": "1"},
        mg_run_args=("--tp", "8", "--ep", "8"),
        mg_extra_args="--no-load-optim",
        grafter_env={"DUMPER_GRAFTER_BACKEND": "gloo"},
        sg_model_path="/storage/iter59-hf",
        mg_hf_checkpoint="/storage/v4-flash-fp8",
        mg_ref_load="/storage/iter59",
        rollout_data_path="/storage/synthetic.pt",
        sg_baseline_response_path="/storage/sg_baseline.json",
        output_root="/storage/dumper-out",
    )


def test_minimal_sg_only_spec_inherits_canonical_env_unchanged() -> None:
    spec = ExperimentSpec(
        name="sg_baseline",
        description="canonical sg run",
        kind=RunKind.SG_ONLY,
    )
    cfg = compose_run_config(_canonical(), spec)

    assert cfg.sg_env == {"SG_FIX_A": "1", "SG_FIX_B": "1"}
    assert cfg.sg_server_args == ("--tp-size", "8", "--port", "30000")
    assert cfg.canonical_name == "v4_canonical_test"
    assert cfg.kind == RunKind.SG_ONLY


def test_spec_env_overrides_canonical() -> None:
    spec = ExperimentSpec(
        name="sg_no_fix_a",
        description="ablate FIX_A",
        kind=RunKind.SG_ONLY,
        sg_env={"SG_FIX_A": "0"},
    )
    cfg = compose_run_config(_canonical(), spec)

    assert cfg.sg_env["SG_FIX_A"] == "0"
    assert cfg.sg_env["SG_FIX_B"] == "1"


def test_spec_args_are_appended_after_canonical() -> None:
    spec = ExperimentSpec(
        name="sg_chunk8192",
        description="smaller chunked prefill",
        kind=RunKind.SG_ONLY,
        sg_server_args_extra=("--chunked-prefill-size", "8192"),
    )
    cfg = compose_run_config(_canonical(), spec)

    assert cfg.sg_server_args == (
        "--tp-size",
        "8",
        "--port",
        "30000",
        "--chunked-prefill-size",
        "8192",
    )


def test_mg_extra_args_appends_with_space() -> None:
    spec = ExperimentSpec(
        name="mg_replay",
        description="enable routing replay",
        kind=RunKind.MG_ONLY,
        mg_extra_args_append="--use-routing-replay",
    )
    cfg = compose_run_config(_canonical(), spec)
    assert cfg.mg_extra_args == "--no-load-optim --use-routing-replay"


def test_mg_extra_args_drops_empty() -> None:
    spec = ExperimentSpec(
        name="mg_default",
        description="no extras",
        kind=RunKind.MG_ONLY,
    )
    cfg = compose_run_config(_canonical(), spec)
    assert cfg.mg_extra_args == "--no-load-optim"


def test_grafter_pair_requires_at_least_one_filter() -> None:
    with pytest.raises(ValueError, match="grafter_b2t_filter or grafter_t2b_filter"):
        ExperimentSpec(
            name="bad",
            description="missing filter",
            kind=RunKind.GRAFTER_PAIR,
        )


def test_grafter_pair_filter_propagates_to_run_config() -> None:
    spec = ExperimentSpec(
        name="a1",
        description="q lora only",
        kind=RunKind.GRAFTER_PAIR,
        grafter_b2t_filter='name == "input_layernorm"',
        grafter_t2b_filter='name == "attn_q"',
    )
    cfg = compose_run_config(_canonical(), spec)

    assert cfg.grafter_b2t_filter == 'name == "input_layernorm"'
    assert cfg.grafter_t2b_filter == 'name == "attn_q"'
    assert cfg.grafter_env == {"DUMPER_GRAFTER_BACKEND": "gloo"}


def test_baselines_propagate_to_run_config() -> None:
    spec = ExperimentSpec(
        name="a1",
        description="q lora only",
        kind=RunKind.GRAFTER_PAIR,
        grafter_b2t_filter='name == "x"',
        baselines=("sg-natural-prefill-e8env", "mg-fix0505"),
    )
    cfg = compose_run_config(_canonical(), spec)
    assert cfg.baselines == ("sg-natural-prefill-e8env", "mg-fix0505")


def test_canonical_paths_carry_through() -> None:
    spec = ExperimentSpec(
        name="t",
        description="t",
        kind=RunKind.MG_ONLY,
    )
    cfg = compose_run_config(_canonical(), spec)
    assert cfg.sg_model_path == "/storage/iter59-hf"
    assert cfg.mg_hf_checkpoint == "/storage/v4-flash-fp8"
    assert cfg.mg_ref_load == "/storage/iter59"
    assert cfg.rollout_data_path == "/storage/synthetic.pt"
    assert cfg.sg_baseline_response_path == "/storage/sg_baseline.json"
    assert cfg.output_root == "/storage/dumper-out"
