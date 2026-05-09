"""Unit tests for sg_launcher command composition (no subprocess)."""

import shlex

from miles.utils.debug_utils.experiment_runner.sg_launcher import build_sg_launch_command
from miles.utils.debug_utils.experiment_runner.spec import CanonicalConfig, ExperimentSpec, RunKind, compose_run_config


def _canonical() -> CanonicalConfig:
    return CanonicalConfig(
        name="v4_canonical",
        sg_env={"SGLANG_DSV4_FIX_0506": "1", "SGLANG_DSV4_MODE": "2604"},
        sg_server_args=("--tp-size", "8", "--ep-size", "8", "--mem-fraction-static", "0.7"),
        grafter_env={
            "DUMPER_GRAFTER_BACKEND": "gloo",
            "DUMPER_GRAFTER_BASELINE_WORLD_SIZE": "8",
            "DUMPER_GRAFTER_TARGET_WORLD_SIZE": "8",
        },
        sg_model_path="/storage/iter59-hf",
        output_root="/storage/dumper-out",
    )


def _spec_sg_only(name: str = "sg_baseline") -> ExperimentSpec:
    return ExperimentSpec(name=name, description="", kind=RunKind.SG_ONLY)


def _spec_grafter(name: str = "a1") -> ExperimentSpec:
    return ExperimentSpec(
        name=name,
        description="",
        kind=RunKind.GRAFTER_PAIR,
        grafter_b2t_filter='name == "input_layernorm"',
        grafter_t2b_filter='name == "attn_q"',
    )


def test_command_includes_model_path_host_port_and_canonical_args() -> None:
    cfg = compose_run_config(_canonical(), _spec_sg_only())
    launch = build_sg_launch_command(cfg, server_host="0.0.0.0", server_port=30000)

    assert "--model-path /storage/iter59-hf" in launch.command
    assert "--host 0.0.0.0" in launch.command
    assert "--port 30000" in launch.command
    assert "--tp-size 8" in launch.command
    assert "--ep-size 8" in launch.command
    assert "--mem-fraction-static 0.7" in launch.command
    assert launch.server_url == "http://0.0.0.0:30000"


def test_command_appends_spec_extra_args_after_canonical() -> None:
    spec = ExperimentSpec(
        name="chunk8192",
        description="",
        kind=RunKind.SG_ONLY,
        sg_server_args_extra=("--chunked-prefill-size", "8192"),
    )
    cfg = compose_run_config(_canonical(), spec)
    launch = build_sg_launch_command(cfg)
    canonical_idx = launch.command.find("--mem-fraction-static")
    extra_idx = launch.command.find("--chunked-prefill-size")
    assert 0 <= canonical_idx < extra_idx, "spec extras must follow canonical args"


def test_env_merges_canonical_and_spec_with_dumper_basics() -> None:
    spec = ExperimentSpec(
        name="ablate_f4",
        description="",
        kind=RunKind.SG_ONLY,
        sg_env={"SGLANG_DSV4_FIX_0506": "0"},
    )
    cfg = compose_run_config(_canonical(), spec)
    launch = build_sg_launch_command(cfg)

    assert launch.env["SGLANG_DSV4_FIX_0506"] == "0"
    assert launch.env["SGLANG_DSV4_MODE"] == "2604"
    assert launch.env["DUMPER_ENABLE"] == "1"
    assert launch.env["DUMPER_NON_INTRUSIVE_MODE"] == "off"
    assert launch.env["DUMPER_DIR"].endswith("/ablate_f4-sg")


def test_env_includes_dumper_filter_when_set() -> None:
    spec = ExperimentSpec(
        name="filtered",
        description="",
        kind=RunKind.SG_ONLY,
        dumper_filter="layer_id < 3",
    )
    cfg = compose_run_config(_canonical(), spec)
    launch = build_sg_launch_command(cfg)
    assert launch.env["DUMPER_FILTER"] == "layer_id < 3"


def test_env_omits_dumper_filter_when_unset() -> None:
    cfg = compose_run_config(_canonical(), _spec_sg_only())
    launch = build_sg_launch_command(cfg)
    assert "DUMPER_FILTER" not in launch.env


def test_grafter_pair_wires_grafter_env_and_role_target() -> None:
    cfg = compose_run_config(_canonical(), _spec_grafter())
    launch = build_sg_launch_command(cfg)

    assert launch.env["DUMPER_GRAFTER_ENABLE"] == "1"
    assert launch.env["DUMPER_GRAFTER_ROLE"] == "target"
    assert launch.env["DUMPER_GRAFTER_BACKEND"] == "gloo"
    assert launch.env["DUMPER_GRAFTER_BASELINE_WORLD_SIZE"] == "8"
    assert launch.env["DUMPER_GRAFTER_B2T_FILTER"] == 'name == "input_layernorm"'
    assert launch.env["DUMPER_GRAFTER_T2B_FILTER"] == 'name == "attn_q"'


def test_sg_only_does_not_set_grafter_env_keys() -> None:
    cfg = compose_run_config(_canonical(), _spec_sg_only())
    launch = build_sg_launch_command(cfg)
    assert "DUMPER_GRAFTER_ENABLE" not in launch.env
    assert "DUMPER_GRAFTER_ROLE" not in launch.env


def test_log_path_under_output_root_with_run_name() -> None:
    cfg = compose_run_config(_canonical(), _spec_sg_only("my_exp"))
    launch = build_sg_launch_command(cfg)
    assert str(launch.log_path) == "/storage/dumper-out/my_exp-sg/server.log"


def test_command_includes_pythonpath_with_miles_repo() -> None:
    cfg = compose_run_config(_canonical(), _spec_sg_only())
    launch = build_sg_launch_command(cfg, miles_repo_dir="/workspace/miles")
    assert launch.env["PYTHONPATH"].startswith("/workspace/miles")


def test_command_uses_shlex_quote_for_paths_with_spaces() -> None:
    canonical = CanonicalConfig(
        name="x",
        sg_model_path="/odd path/with spaces",
        output_root="/output root",
    )
    cfg = compose_run_config(canonical, _spec_sg_only("e1"))
    launch = build_sg_launch_command(cfg)
    assert shlex.quote("/odd path/with spaces") in launch.command
    assert shlex.quote("/output root/e1-sg/server.log") in launch.command
