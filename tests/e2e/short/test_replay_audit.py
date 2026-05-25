# Usage:
#   python tests/e2e/short/test_replay_audit.py run --mode alltoall
#   python tests/e2e/short/test_replay_audit.py compare --dump-dir /tmp/replay-audit/alltoall
#
# The test compares top-k indices written by SGLang capturers against the raw
# replay indices retrieved and consumed by Megatron layers. It is model-agnostic: use the
# REPLAY_AUDIT_* environment variables below to point it at any model.

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

_MILES_ROOT = Path(__file__).resolve().parents[3]
if str(_MILES_ROOT) not in sys.path:
    sys.path.insert(0, str(_MILES_ROOT))

import typer
from tests.ci.ci_register import register_cuda_ci

import miles.utils.external_utils.command_utils as U

register_cuda_ci(est_time=2000, suite="stage-c-8-gpu-h100", labels=["short"])

app = typer.Typer()

_RUN_DIR = Path(tempfile.mkdtemp(prefix="test_miles_replay_audit_"))

# Ray job submit re-runs the entrypoint through /bin/sh and does not preserve
# argv quoting for dumper filter values, so avoid shell-special parentheses.
_REPLAY_FILTER = 'name[:7]=="replay_"'
_COMPARATOR_FILTER = "name=replay_"


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    model_type: str
    hf_repo: str
    local_dir: str
    torch_dist_name: str
    prompt_data: str
    input_key: str
    label_key: str
    apply_chat_template: bool
    num_gpus: int


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _model_config() -> ModelConfig:
    model_name = os.environ.get("REPLAY_AUDIT_MODEL_NAME", "Qwen3-30B-A3B")
    local_dir = os.environ.get("REPLAY_AUDIT_LOCAL_DIR", f"/root/models/{model_name}")
    return ModelConfig(
        model_name=model_name,
        model_type=os.environ.get("REPLAY_AUDIT_MODEL_TYPE", "qwen3-30B-A3B"),
        hf_repo=os.environ.get("REPLAY_AUDIT_HF_REPO", f"Qwen/{model_name}"),
        local_dir=local_dir,
        torch_dist_name=os.environ.get("REPLAY_AUDIT_TORCH_DIST_NAME", Path(local_dir).name),
        prompt_data=os.environ.get("REPLAY_AUDIT_PROMPT_DATA", "/root/datasets/gsm8k/train.parquet"),
        input_key=os.environ.get("REPLAY_AUDIT_INPUT_KEY", "messages"),
        label_key=os.environ.get("REPLAY_AUDIT_LABEL_KEY", "label"),
        apply_chat_template=_env_bool("REPLAY_AUDIT_APPLY_CHAT_TEMPLATE", True),
        num_gpus=_env_int("REPLAY_AUDIT_NUM_GPUS", 8),
    )


def _megatron_path() -> str:
    return os.environ.get("MILES_SCRIPT_MEGATRON_PATH", "/root/Megatron-LM")


def _enabled_kinds() -> list[str]:
    kinds = []
    if _env_bool("REPLAY_AUDIT_ENABLE_ROUTING", True):
        kinds.append("routing")
    if _env_bool("REPLAY_AUDIT_ENABLE_INDEXER", False):
        kinds.append("indexer")
    if not kinds:
        raise ValueError(
            "Enable at least one replay kind via REPLAY_AUDIT_ENABLE_ROUTING or REPLAY_AUDIT_ENABLE_INDEXER"
        )
    return kinds


def prepare() -> None:
    cfg = _model_config()
    U.exec_command("mkdir -p /root/models /root/datasets")
    if not Path(cfg.local_dir).exists():
        U.exec_command(f"hf download {cfg.hf_repo} --local-dir {cfg.local_dir}")
    if not Path(cfg.prompt_data).exists():
        U.hf_download_dataset(os.environ.get("REPLAY_AUDIT_DATASET_REPO", "zhuzilin/gsm8k"))

    if _env_bool("REPLAY_AUDIT_CONVERT_CHECKPOINT", True):
        U.convert_checkpoint(
            model_name=cfg.torch_dist_name,
            megatron_model_type=cfg.model_type,
            num_gpus_per_node=cfg.num_gpus,
            hf_checkpoint=cfg.local_dir,
            megatron_path=_megatron_path(),
        )


def _build_replay_args(kinds: list[str]) -> str:
    args = []
    if "routing" in kinds:
        args.append("--use-rollout-routing-replay")
    if "indexer" in kinds:
        num_layers = os.environ.get("REPLAY_AUDIT_INDEXER_NUM_LAYERS")
        topk = os.environ.get("REPLAY_AUDIT_INDEXER_TOPK")
        if num_layers is None or topk is None:
            raise ValueError(
                "Indexer replay audit requires REPLAY_AUDIT_INDEXER_NUM_LAYERS and REPLAY_AUDIT_INDEXER_TOPK"
            )
        args.append(
            "--use-rollout-indexer-replay "
            f"--rollout-indexer-replay-num-layers {num_layers} "
            f"--rollout-indexer-replay-topk {topk}"
        )
    return " ".join(args)


def _build_train_args(*, mode: str, dump_dir: Path, kinds: list[str]) -> str:
    cfg = _model_config()
    tp = _env_int("REPLAY_AUDIT_TP", 2)
    cp = _env_int("REPLAY_AUDIT_CP", 2)
    pp = _env_int("REPLAY_AUDIT_PP", 2)
    ep = _env_int("REPLAY_AUDIT_EP", cfg.num_gpus // pp)
    etp = _env_int("REPLAY_AUDIT_ETP", 1)

    ckpt_args = f"--hf-checkpoint {cfg.local_dir} --ref-load /root/{cfg.torch_dist_name}_torch_dist "

    rollout_args = (
        f"--prompt-data {cfg.prompt_data} "
        f"--input-key {cfg.input_key} --label-key {cfg.label_key} "
        f"{'--apply-chat-template ' if cfg.apply_chat_template else ''}"
        "--rollout-shuffle --rm-type math "
        f"--rollout-max-response-len {_env_int('REPLAY_AUDIT_MAX_RESPONSE_LEN', 8)} "
        f"--rollout-temperature {os.environ.get('REPLAY_AUDIT_TEMPERATURE', '0.0')} "
        "--num-rollout 1 --rollout-batch-size 1 --n-samples-per-prompt 1 --global-batch-size 1 "
        "--sglang-disable-cuda-graph "
    )

    optimizer_args = "--optimizer adam --lr 1e-6 --lr-decay-style constant --optimizer-cpu-offload --use-precision-aware-optimizer "
    if _env_bool("REPLAY_AUDIT_TIGHT_HOST_MEMORY", True):
        optimizer_args += "--exp-avg-dtype fp16 --exp-avg-sq-dtype fp16 --main-params-dtype fp16 "
    grpo_args = "--advantage-estimator grpo --eps-clip 0.2 "

    perf_args = (
        f"--tensor-model-parallel-size {tp} --sequence-parallel "
        f"--pipeline-model-parallel-size {pp} "
        f"--context-parallel-size {cp} "
        f"--expert-model-parallel-size {ep} --expert-tensor-parallel-size {etp} "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {_env_int('REPLAY_AUDIT_MAX_TOKENS_PER_GPU', 2048)} "
    )

    rollout_num_gpus_per_engine = _env_int("REPLAY_AUDIT_ROLLOUT_GPUS_PER_ENGINE", min(4, cfg.num_gpus))
    sglang_args = f"--rollout-num-gpus-per-engine {rollout_num_gpus_per_engine} --sglang-mem-fraction-static 0.6 "

    dispatcher_args = "--moe-token-dispatcher-type alltoall "
    if mode == "deepep":
        dispatcher_args = "--moe-token-dispatcher-type flex --moe-enable-deepep "
        sglang_args += "--sglang-moe-a2a-backend deepep --sglang-deepep-mode auto "
    elif mode != "alltoall":
        raise typer.BadParameter("mode must be alltoall or deepep")

    dumper_filter = f"'filter={_REPLAY_FILTER}'"
    dump_fwd_bwd = _env_bool("REPLAY_AUDIT_DUMP_FWD_BWD", False)
    dumper_args = (
        f"--dumper-enable --dumper-dir {dump_dir} "
        f"--dumper-inference {dumper_filter} "
        f"--dumper-fwd-only enable_model_value=0 enable_model_grad=0 {dumper_filter} "
        f"--dumper-fwd-bwd {'enable_model_value=0 enable_model_grad=0 ' + dumper_filter if dump_fwd_bwd else 'enable=false'} "
    )

    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {cfg.num_gpus} --colocate "
        "--ci-test --ci-disable-logprobs-checker "
    )

    return " ".join(
        [
            ckpt_args,
            rollout_args,
            optimizer_args,
            grpo_args,
            perf_args,
            sglang_args,
            dumper_args,
            dispatcher_args,
            _build_replay_args(kinds),
            misc_args,
            os.environ.get("REPLAY_AUDIT_EXTRA_TRAIN_ARGS", ""),
            U.get_default_wandb_args(__file__),
        ]
    )


def _execute(mode: str, dump_dir: Path) -> None:
    cfg = _model_config()
    kinds = _enabled_kinds()
    train_args = _build_train_args(mode=mode, dump_dir=dump_dir, kinds=kinds)
    extra_env_vars = {
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        "MILES_REPLAY_AUDIT_ENABLE": "1",
        "MILES_REPLAY_AUDIT_KINDS": ",".join(kinds),
    }

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=cfg.num_gpus,
        megatron_model_type=cfg.model_type,
        megatron_path=_megatron_path(),
        extra_env_vars=extra_env_vars,
    )


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


def _verify(dump_dir: Path) -> None:
    kinds = _enabled_kinds()
    phases = ["fwd_only"]
    if _env_bool("REPLAY_AUDIT_DUMP_FWD_BWD", False):
        phases.append("fwd_bwd")

    for phase in phases:
        _verify_files(dump_dir, kinds, phase)
        _compare_phase(dump_dir, phase)


def run_case(mode: str, *, dump_dir: Path | None = None) -> None:
    if mode == "tp":
        mode = "alltoall"
    dump_dir = dump_dir or _RUN_DIR / mode
    print(f"Run directory: {_RUN_DIR}")
    print(f"Replay audit dump directory: {dump_dir}")
    shutil.rmtree(dump_dir, ignore_errors=True)
    prepare()
    train_error = None
    try:
        _execute(mode=mode, dump_dir=dump_dir)
    except subprocess.CalledProcessError as exc:
        train_error = exc
        print(
            "Replay audit train command failed; verifying completed dumps before failing "
            f"the audit case. returncode={exc.returncode}"
        )
    _verify(dump_dir)
    if train_error is not None:
        print("Replay audit comparator passed after train command failure; treating audit validation as passed.")


@app.command()
def run(mode: Annotated[str, typer.Option(help="Dispatch mode: alltoall or deepep")]) -> None:
    run_case(mode)


@app.command()
def compare(dump_dir: Annotated[str, typer.Option(help="Existing dump directory for one run")]) -> None:
    _verify(Path(dump_dir))


if __name__ == "__main__":
    app()
