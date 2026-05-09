"""Unit tests for mg_launcher command composition."""

from pathlib import Path

from miles.utils.debug_utils.experiment_runner.mg_launcher import build_mg_launch_command
from miles.utils.debug_utils.experiment_runner.spec import CanonicalConfig, ExperimentSpec, RunKind, compose_run_config


def _canonical(**overrides: object) -> CanonicalConfig:
    base = dict(
        name="v4_canonical",
        mg_env={"MILES_DSV4_FIX_0505": "1"},
        mg_extra_args="--no-load-optim --no-load-rng",
        grafter_env={
            "DUMPER_GRAFTER_BACKEND": "gloo",
            "DUMPER_GRAFTER_BASELINE_WORLD_SIZE": "8",
        },
        mg_hf_checkpoint="/storage/v4-flash-fp8",
        mg_ref_load="/storage/iter59",
        rollout_data_path="/storage/synthetic.pt",
        output_root="/storage/dumper-out",
    )
    base.update(overrides)  # type: ignore[arg-type]
    return CanonicalConfig(**base)  # type: ignore[arg-type]


def _spec(name: str = "exp", **overrides: object) -> ExperimentSpec:
    base = dict(name=name, description="t", kind=RunKind.MG_ONLY)
    base.update(overrides)  # type: ignore[arg-type]
    return ExperimentSpec(**base)  # type: ignore[arg-type]


def test_command_includes_all_canonical_flags() -> None:
    cfg = compose_run_config(_canonical(), _spec("baseline"))
    launch = build_mg_launch_command(cfg)
    assert "--hf-checkpoint /storage/v4-flash-fp8" in launch.command
    assert "--ref-load /storage/iter59" in launch.command
    assert "--rollout-data /storage/synthetic.pt" in launch.command
    assert "--extra-args '--no-load-optim --no-load-rng'" in launch.command
    assert "--tp 8" in launch.command
    assert "--ep 8" in launch.command
    assert "--batch-size 1" in launch.command
    assert "--sp" in launch.command


def test_logprob_output_dir_under_output_dir() -> None:
    cfg = compose_run_config(_canonical(), _spec("my_exp"))
    launch = build_mg_launch_command(cfg)
    assert launch.output_dir == Path("/storage/dumper-out/my_exp-mg")
    assert launch.logprob_output_dir == launch.output_dir / "megatron_logprobs"
    assert "--logprob-output /storage/dumper-out/my_exp-mg/megatron_logprobs" in launch.command


def test_dumper_filter_passed_through() -> None:
    cfg = compose_run_config(
        _canonical(),
        _spec("filtered", dumper_filter="layer_id is None or layer_id < 3"),
    )
    launch = build_mg_launch_command(cfg)
    assert "--dumper-filter 'layer_id is None or layer_id < 3'" in launch.command


def test_grafter_pair_sets_role_baseline_in_env(tmp_path: Path) -> None:
    spec = ExperimentSpec(
        name="a1",
        description="",
        kind=RunKind.GRAFTER_PAIR,
        grafter_b2t_filter='name == "input_layernorm"',
        grafter_t2b_filter='name == "attn_q"',
    )
    cfg = compose_run_config(_canonical(), spec)
    launch = build_mg_launch_command(cfg, patcher_yaml_dir=tmp_path)
    assert launch.env["DUMPER_GRAFTER_ENABLE"] == "1"
    assert launch.env["DUMPER_GRAFTER_ROLE"] == "baseline"
    assert launch.env["DUMPER_GRAFTER_B2T_FILTER"] == 'name == "input_layernorm"'
    assert launch.env["DUMPER_GRAFTER_T2B_FILTER"] == 'name == "attn_q"'


def test_mg_only_does_not_set_grafter_env(tmp_path: Path) -> None:
    cfg = compose_run_config(_canonical(), _spec("plain"))
    launch = build_mg_launch_command(cfg, patcher_yaml_dir=tmp_path)
    assert "DUMPER_GRAFTER_ENABLE" not in launch.env
    assert "DUMPER_GRAFTER_ROLE" not in launch.env


def test_patcher_yaml_materialised_when_provided(tmp_path: Path) -> None:
    yaml_content = "patches:\n  - target: x\n    edits: []\n"
    cfg = compose_run_config(
        _canonical(),
        _spec("with_patches", mg_extra_patches_yaml=yaml_content),
    )
    launch = build_mg_launch_command(cfg, patcher_yaml_dir=tmp_path)

    assert launch.patcher_yaml_path is not None
    assert launch.patcher_yaml_path.exists()
    assert launch.patcher_yaml_path.read_text() == yaml_content
    assert f"--source-patcher-config {launch.patcher_yaml_path}" in launch.command


def test_no_patcher_yaml_when_not_set(tmp_path: Path) -> None:
    cfg = compose_run_config(_canonical(), _spec("plain"))
    launch = build_mg_launch_command(cfg, patcher_yaml_dir=tmp_path)
    assert launch.patcher_yaml_path is None
    assert "--source-patcher-config" not in launch.command


def test_mg_run_args_extra_appended_verbatim(tmp_path: Path) -> None:
    spec = _spec(
        "extra_flags",
        mg_run_args_extra=("--top-k", "5", "--apply-chat-template"),
    )
    cfg = compose_run_config(_canonical(), spec)
    launch = build_mg_launch_command(cfg, patcher_yaml_dir=tmp_path)
    assert "--top-k 5" in launch.command
    assert "--apply-chat-template" in launch.command


def test_canonical_mg_run_args_passed_through(tmp_path: Path) -> None:
    canonical = _canonical(mg_run_args=("--prompt-mode", "math"))
    cfg = compose_run_config(canonical, _spec("p"))
    launch = build_mg_launch_command(cfg, patcher_yaml_dir=tmp_path)
    assert "--prompt-mode math" in launch.command


def test_no_extra_args_flag_when_canonical_has_none(tmp_path: Path) -> None:
    canonical = _canonical(mg_extra_args="")
    cfg = compose_run_config(canonical, _spec("clean"))
    launch = build_mg_launch_command(cfg, patcher_yaml_dir=tmp_path)
    assert "--extra-args" not in launch.command


def test_parallel_kwargs_override_defaults(tmp_path: Path) -> None:
    cfg = compose_run_config(_canonical(), _spec("tp4"))
    launch = build_mg_launch_command(cfg, tp=4, pp=2, cp=1, ep=4, etp=2, batch_size=2, patcher_yaml_dir=tmp_path)
    assert "--tp 4" in launch.command
    assert "--pp 2" in launch.command
    assert "--ep 4" in launch.command
    assert "--etp 2" in launch.command
    assert "--batch-size 2" in launch.command


def test_sp_flag_excluded_when_disabled(tmp_path: Path) -> None:
    cfg = compose_run_config(_canonical(), _spec("nosp"))
    launch = build_mg_launch_command(cfg, sp=False, patcher_yaml_dir=tmp_path)
    assert "--sp" not in launch.command
