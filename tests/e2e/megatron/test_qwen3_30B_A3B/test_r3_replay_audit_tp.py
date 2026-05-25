import os
import shutil
from pathlib import Path

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.conftest_dumper import clear_proxy_env
from tests.e2e.megatron.test_qwen3_30B_A3B._common import CaseConfig, execute, prepare
from tests.e2e.short import test_replay_audit

register_cuda_ci(est_time=1800, suite="stage-c-4-gpu-h200", labels=["megatron"])

_KINDS = ["routing"]
_DUMP_DIR = Path(os.environ.get("REPLAY_AUDIT_DUMP_ROOT", "/tmp/replay-audit")) / "qwen3_30b_a3b_r3_tp"


def _run() -> None:
    clear_proxy_env()
    shutil.rmtree(_DUMP_DIR, ignore_errors=True)
    replay_extra_args, replay_extra_env_vars = test_replay_audit.build_replay_audit_extras(
        kinds=_KINDS,
        dump_dir=_DUMP_DIR,
    )
    case = CaseConfig(
        num_gpus_per_node=4,
        cp_size=2,
        pp_size=1,
        use_deepep=False,
        use_r3=True,
        extra_args=f"{replay_extra_args} --ci-disable-logprobs-checker --sglang-disable-cuda-graph ",
        extra_env_vars=replay_extra_env_vars,
    )
    prepare(case, need_fp8=False, need_int4=False, all_bridge=False)
    execute(case, wandb_file=__file__)
    test_replay_audit.verify(_DUMP_DIR, kinds=_KINDS)


def test_qwen3_30b_a3b_r3_replay_audit_tp() -> None:
    _run()


if __name__ == "__main__":
    _run()
