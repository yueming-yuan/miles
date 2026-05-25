import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.conftest_dumper import clear_proxy_env
from tests.e2e.short import test_replay_audit


register_cuda_ci(est_time=2400, suite="stage-c-8-gpu-h200", labels=["megatron"])

_GLM5_4LAYER_REPLAY_AUDIT_DEFAULTS = {
    "REPLAY_AUDIT_MODEL_NAME": "GLM-5_4layer",
    "REPLAY_AUDIT_MODEL_TYPE": "glm5-744B-A40B_4layer",
    "REPLAY_AUDIT_HF_REPO": "Pinaster/GLM-5_4layer",
    "REPLAY_AUDIT_LOCAL_DIR": "/root/models/GLM-5_4layer",
    "REPLAY_AUDIT_TORCH_DIST_NAME": "GLM-5_4layer",
    "REPLAY_AUDIT_TORCH_DIST_DIR": "/root/models",
    "REPLAY_AUDIT_PROMPT_DATA": "/root/datasets/dapo-math-17k/dapo-math-17k.jsonl",
    "REPLAY_AUDIT_DATASET_REPO": "zhuzilin/dapo-math-17k",
    "REPLAY_AUDIT_INPUT_KEY": "prompt",
    "REPLAY_AUDIT_LABEL_KEY": "label",
    "REPLAY_AUDIT_APPLY_CHAT_TEMPLATE": "1",
    "REPLAY_AUDIT_NUM_GPUS": "8",
    "REPLAY_AUDIT_CONVERT_NUM_GPUS": "4",
    "REPLAY_AUDIT_CONVERT_EXTRA_ARGS": (
        "--tensor-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
        "--expert-model-parallel-size 1"
    ),
    "REPLAY_AUDIT_TP": "4",
    "REPLAY_AUDIT_CP": "1",
    "REPLAY_AUDIT_PP": "1",
    "REPLAY_AUDIT_EP": "8",
    "REPLAY_AUDIT_ETP": "1",
    "REPLAY_AUDIT_ROLLOUT_GPUS_PER_ENGINE": "8",
    "REPLAY_AUDIT_SGLANG_MEM_FRACTION_STATIC": "0.70",
    "REPLAY_AUDIT_MAX_RESPONSE_LEN": "8",
    "REPLAY_AUDIT_TEMPERATURE": "1.0",
    "REPLAY_AUDIT_NUM_ROLLOUT": "8",
    "REPLAY_AUDIT_ROLLOUT_BATCH_SIZE": "8",
    "REPLAY_AUDIT_N_SAMPLES_PER_PROMPT": "1",
    "REPLAY_AUDIT_GLOBAL_BATCH_SIZE": "8",
    "REPLAY_AUDIT_RM_TYPE": "deepscaler",
    "REPLAY_AUDIT_MAX_TOKENS_PER_GPU": "2048",
    "REPLAY_AUDIT_ENABLE_ROUTING": "1",
    "REPLAY_AUDIT_ENABLE_INDEXER": "1",
    "REPLAY_AUDIT_INDEXER_NUM_LAYERS": "4",
    "REPLAY_AUDIT_INDEXER_TOPK": "2048",
    "REPLAY_AUDIT_DUMP_FWD_BWD": "0",
    "REPLAY_AUDIT_TIGHT_HOST_MEMORY": "1",
    "REPLAY_AUDIT_EXTRA_ENV_VARS": (
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 "
        "SGLANG_NSA_FORCE_MLA=1 "
        "INDEXER_ROPE_NEOX_STYLE=0 "
        "NVSHMEM_DISABLE_NCCL=1"
    ),
    "REPLAY_AUDIT_EXTRA_TRAIN_ARGS": (
        "--allgather-cp "
        "--use-miles-router "
        "--data-pad-size-multiplier 4096 "
        "--log-probs-chunk-size 1024 "
        "--update-weight-buffer-size 2147483648 "
        "--disable-weights-backuper "
        "--sglang-enable-dp-attention "
        "--sglang-ep-size 8 "
        "--sglang-dp-size 8 "
        "--sglang-moe-dense-tp-size 1 "
        "--sglang-enable-dp-lm-head "
        "--sglang-page-size 64 "
        "--sglang-nsa-decode-backend flashmla_sparse "
        "--sglang-nsa-prefill-backend flashmla_sparse "
        "--sglang-attention-backend nsa "
        "--sglang-cuda-graph-max-bs 256 "
        "--sglang-max-total-tokens 8192 "
        "--sglang-max-running-requests 512 "
        "--sglang-chunked-prefill-size 2048 "
        "--sglang-watchdog-timeout 3600"
    ),
}


@contextmanager
def _env_defaults(defaults: dict[str, str]) -> Iterator[None]:
    old_values = {key: os.environ.get(key) for key in defaults}
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def run_glm5_4layer_replay_audit() -> None:
    clear_proxy_env()
    with _env_defaults(_GLM5_4LAYER_REPLAY_AUDIT_DEFAULTS):
        dump_root = Path(os.environ.get("REPLAY_AUDIT_DUMP_ROOT", "/tmp/replay-audit"))
        dump_dir = dump_root / "glm5_4layer_indexer_expert_replay"
        shutil.rmtree(dump_dir, ignore_errors=True)
        test_replay_audit.run_case(mode="megatron_deepep", dump_dir=dump_dir)


def test_glm5_4layer_indexer_and_expert_replay_audit() -> None:
    run_glm5_4layer_replay_audit()


if __name__ == "__main__":
    run_glm5_4layer_replay_audit()
