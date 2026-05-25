from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import torch
import pytest

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


def test_replay_check_uses_rowwise_overlap():
    manager = _ReplayManager()
    manager.enable_check_replay_result = True

    orig = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.int64)
    replay = torch.tensor([[2, 9], [10, 3], [-1, -1], [8, 1]], dtype=torch.int64)

    def _old_topk(_scores, _topk):
        return orig

    manager.check_replay_result(_old_topk, torch.empty(orig.shape[0], 16), 2, replay)


def test_replay_check_rejects_row_without_overlap():
    manager = _ReplayManager()
    manager.enable_check_replay_result = True

    orig = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    replay = torch.tensor([[5, 6], [3, 7]], dtype=torch.int64)

    def _old_topk(_scores, _topk):
        return orig

    with pytest.raises(AssertionError, match="R3 mismatch tokens"):
        manager.check_replay_result(_old_topk, torch.empty(orig.shape[0], 16), 2, replay)
