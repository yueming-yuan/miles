from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import torch

from miles.utils.replay_base import BaseReplayManager


class _ReplayManager(BaseReplayManager):
    name = "test"
    enable_check_replay_result = False
    replay_check_threshold = 0.0


def test_replay_forward_preserves_raw_padding_rows():
    manager = _ReplayManager()
    manager.enabled = True
    manager.stage = "replay_forward"

    replay = manager.create_replay()
    replay.source_stream_id = 0
    manager.set_current(replay)
    replay.record(torch.tensor([[-1, -1], [2, 3]], dtype=torch.int64))

    def _unused_topk(scores, topk):
        raise AssertionError("replay stage should not call the original topk")

    scores = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    replay_topk = manager.get_topk_fn(_unused_topk, return_probs=False)(scores, 2)

    torch.testing.assert_close(replay.last_forward_top_indices_raw, torch.tensor([[-1, -1], [2, 3]]))
    assert (replay_topk >= 0).all()
