"""
Replay audit verification toolkit.

Per-model replay-audit tests live next to their canonical training entry
and drive training via the model's canonical script. They wire replay-audit
extras through the model's `extra_args` / `extra_env_vars` hooks (see
`build_replay_audit_extras`) and then call `verify(dump_dir, kinds=...)`.

CLI:
    python tests/e2e/short/test_replay_audit.py compare \
        --dump-dir /tmp/replay-audit/foo --kinds routing,indexer
"""

import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=2000, suite="stage-c-8-gpu-h100", labels=["short"])

# Ray job submit re-runs the entrypoint through /bin/sh and does not preserve
# argv quoting for dumper filter values, so avoid shell-special parentheses.
_REPLAY_FILTER = 'name[:7]=="replay_"'
_COMPARATOR_FILTER = "name=replay_"

app = typer.Typer()


def build_replay_audit_extras(
    *,
    kinds: list[str],
    dump_dir: Path,
    indexer_num_layers: int | None = None,
    indexer_topk: int | None = None,
    dump_fwd_bwd: bool = False,
) -> tuple[str, dict[str, str]]:
    """Build the replay-audit-specific train flags and env vars.

    Returns (extra_train_args, extra_env_vars). The per-model replay-audit
    test passes these through its model's canonical `extra_args` and
    `extra_env_vars` hooks. `build_replay_audit_extras` owns all the
    replay-kind flags so it remains the single source of truth for which
    kinds are captured; argparse `store_true` flags are idempotent if a
    canonical script also emits the same flag.
    """
    if not kinds:
        raise ValueError("kinds must be non-empty")

    kind_flags: list[str] = []
    if "routing" in kinds:
        kind_flags.append("--use-rollout-routing-replay")
    if "indexer" in kinds:
        if indexer_num_layers is None or indexer_topk is None:
            raise ValueError("indexer kind requires indexer_num_layers and indexer_topk")
        kind_flags.append(
            "--use-rollout-indexer-replay "
            f"--rollout-indexer-replay-num-layers {indexer_num_layers} "
            f"--rollout-indexer-replay-topk {indexer_topk}"
        )

    dumper_filter = f"'filter={_REPLAY_FILTER}'"
    fwd_bwd_spec = f"enable_model_value=0 enable_model_grad=0 {dumper_filter}" if dump_fwd_bwd else "enable=false"
    dumper_args = (
        f"--dumper-enable --dumper-dir {dump_dir} "
        f"--dumper-inference {dumper_filter} "
        f"--dumper-fwd-only enable_model_value=0 enable_model_grad=0 {dumper_filter} "
        f"--dumper-fwd-bwd {fwd_bwd_spec} "
    )

    extra_args = " ".join([*kind_flags, dumper_args])
    extra_env_vars = {
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        "MILES_REPLAY_AUDIT_ENABLE": "1",
        "MILES_REPLAY_AUDIT_KINDS": ",".join(kinds),
        "MILES_DISABLE_REPLAY_RESULT_CHECK": "1",
    }
    return extra_args, extra_env_vars


def _verify_files(dump_dir: Path, kinds: list[str], phase: str) -> None:
    source_dir = dump_dir / "engines"
    target_dir = dump_dir / phase
    assert source_dir.is_dir(), f"Missing SGLang engine dump dir: {source_dir}"
    assert target_dir.is_dir(), f"Missing target dump dir: {target_dir}"
    for kind in kinds:
        source_matches = list(source_dir.rglob(f"*name=replay_{kind}_stream_*.pt"))
        target_matches = list(target_dir.rglob(f"*name=replay_{kind}_stream_*.pt"))
        assert source_matches, f"No SGLang capture dumps for replay kind {kind!r} in {source_dir}"
        assert target_matches, f"No Megatron replay dumps for replay kind {kind!r} in {target_dir}"


def _compare_phase(dump_dir: Path, phase: str) -> None:
    source_dir = dump_dir / "engines"
    target_dir = dump_dir / phase
    cmd = [
        sys.executable,
        "-m",
        "sglang.srt.debug_utils.comparator",
        "--baseline-path",
        str(source_dir),
        "--target-path",
        str(target_dir),
        "--output-format",
        "json",
        "--preset",
        "sglang_megatron",
        "--diff-threshold",
        "0",
        "--filter",
        _COMPARATOR_FILTER,
        "--allow-skipped-pattern",
        "^$",
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.stdout.strip():
        print(f"[comparator stdout]\n{result.stdout}")
    if result.stderr.strip():
        print(f"[comparator stderr]\n{result.stderr}")
    assert result.returncode == 0, f"Replay audit comparator failed for {phase} with rc={result.returncode}"


def verify(dump_dir: Path, *, kinds: list[str], dump_fwd_bwd: bool = False) -> None:
    """Verify replay-audit dump files exist and match across SGLang/Megatron."""
    phases = ["fwd_only"]
    if dump_fwd_bwd:
        phases.append("fwd_bwd")
    for phase in phases:
        _verify_files(dump_dir, kinds, phase)
        _compare_phase(dump_dir, phase)


@app.command()
def compare(
    dump_dir: Annotated[str, typer.Option(help="Existing dump directory for one run")],
    kinds: Annotated[str, typer.Option(help="Comma-separated replay kinds (e.g. routing,indexer)")],
    dump_fwd_bwd: Annotated[bool, typer.Option(help="Also verify fwd_bwd phase")] = False,
) -> None:
    parsed_kinds = [k.strip() for k in kinds.split(",") if k.strip()]
    verify(Path(dump_dir), kinds=parsed_kinds, dump_fwd_bwd=dump_fwd_bwd)


if __name__ == "__main__":
    app()
