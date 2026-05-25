from tests.ci.ci_register import register_cuda_ci
from tests.e2e.megatron.test_qwen3_30B_A3B._replay_audit_common import run_qwen3_30b_r3_replay_audit

register_cuda_ci(est_time=1800, suite="stage-c-4-gpu-h200", labels=["megatron"])


def test_qwen3_30b_a3b_r3_replay_audit_deepep() -> None:
    run_qwen3_30b_r3_replay_audit("deepep")


if __name__ == "__main__":
    test_qwen3_30b_a3b_r3_replay_audit_deepep()
