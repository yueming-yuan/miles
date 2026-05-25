import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from tests.e2e.conftest_dumper import clear_proxy_env
from tests.e2e.short import test_replay_audit


_QWEN3_30B_R3_DEFAULTS = {
    "REPLAY_AUDIT_MODEL_NAME": "Qwen3-30B-A3B",
    "REPLAY_AUDIT_MODEL_TYPE": "qwen3-30B-A3B",
    "REPLAY_AUDIT_HF_REPO": "Qwen/Qwen3-30B-A3B",
    "REPLAY_AUDIT_LOCAL_DIR": "/root/models/Qwen3-30B-A3B",
    "REPLAY_AUDIT_TORCH_DIST_NAME": "Qwen3-30B-A3B",
    "REPLAY_AUDIT_PROMPT_DATA": "/root/datasets/dapo-math-17k/dapo-math-17k.jsonl",
    "REPLAY_AUDIT_DATASET_REPO": "zhuzilin/dapo-math-17k",
    "REPLAY_AUDIT_INPUT_KEY": "prompt",
    "REPLAY_AUDIT_LABEL_KEY": "label",
    "REPLAY_AUDIT_APPLY_CHAT_TEMPLATE": "1",
    "REPLAY_AUDIT_NUM_GPUS": "4",
    "REPLAY_AUDIT_TP": "2",
    "REPLAY_AUDIT_CP": "2",
    "REPLAY_AUDIT_PP": "1",
    "REPLAY_AUDIT_EP": "4",
    "REPLAY_AUDIT_ETP": "1",
    "REPLAY_AUDIT_ROLLOUT_GPUS_PER_ENGINE": "4",
    "REPLAY_AUDIT_MAX_RESPONSE_LEN": "8",
    "REPLAY_AUDIT_TEMPERATURE": "0.0",
    "REPLAY_AUDIT_MAX_TOKENS_PER_GPU": "2048",
    "REPLAY_AUDIT_ENABLE_ROUTING": "1",
    "REPLAY_AUDIT_ENABLE_INDEXER": "0",
    "REPLAY_AUDIT_DUMP_FWD_BWD": "0",
    "REPLAY_AUDIT_TIGHT_HOST_MEMORY": "1",
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


def run_qwen3_30b_r3_replay_audit(dispatch: Literal["tp", "deepep"]) -> None:
    clear_proxy_env()
    mode = "alltoall" if dispatch == "tp" else "deepep"

    with _env_defaults(_QWEN3_30B_R3_DEFAULTS):
        dump_root = Path(os.environ.get("REPLAY_AUDIT_DUMP_ROOT", "/tmp/replay-audit"))
        dump_dir = dump_root / f"qwen3_30b_a3b_r3_{dispatch}"
        shutil.rmtree(dump_dir, ignore_errors=True)
        test_replay_audit.run_case(mode=mode, dump_dir=dump_dir)
