import json
import os
import shutil
from pathlib import Path

from scripts.run_glm5_744b_a40b import (
    ScriptArgs,
    _execute_train,
    _prepare_download,
    _prepare_megatron_ckpt,
    _validate_glm_checkpoint,
)
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.conftest_dumper import clear_proxy_env
from tests.e2e.short import test_replay_audit

register_cuda_ci(est_time=2400, suite="stage-c-8-gpu-h200", labels=["megatron"])

_KINDS = ["routing", "indexer"]
_DUMP_DIR = Path(os.environ.get("REPLAY_AUDIT_DUMP_ROOT", "/tmp/replay-audit")) / "glm5_4layer_indexer_expert_replay"


def _run() -> None:
    clear_proxy_env()
    shutil.rmtree(_DUMP_DIR, ignore_errors=True)

    replay_extra_args, replay_extra_env_vars = test_replay_audit.build_replay_audit_extras(
        kinds=_KINDS,
        dump_dir=_DUMP_DIR,
        indexer_num_layers=4,
        indexer_topk=2048,
    )
    args = ScriptArgs(
        model_name="GLM-5_4layer",
        num_nodes=1,
        enable_optimizer_offload=True,
        # Shrink rollout work to the minimum needed to capture and compare dumps.
        rollout_max_response_len=8,
        num_rollout=8,
        n_samples_per_prompt=1,
        global_batch_size=8,
        extra_args=(
            f"{replay_extra_args} "
            "--ci-test --ci-disable-logprobs-checker "
            "--use-miles-router --disable-weights-backuper "
            "--sglang-disable-cuda-graph "
        ),
        extra_env_vars=json.dumps(replay_extra_env_vars),
    )
    _prepare_download(args)
    _validate_glm_checkpoint(args)
    _prepare_megatron_ckpt(args)
    _execute_train(args)
    test_replay_audit.verify(_DUMP_DIR, kinds=_KINDS)


def test_glm5_4layer_indexer_and_expert_replay_audit() -> None:
    _run()


if __name__ == "__main__":
    _run()
