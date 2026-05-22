from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import torch
import pytest

from miles.backends.megatron_utils.replay_utils import _register_replay_list_sequential


class _Replay:
    def __init__(self):
        self.recorded = []

    def record(self, value):
        self.recorded.append(value)


def test_register_replay_list_sequential_records_matching_streams():
    replay_data = torch.arange(5 * 3 * 2).reshape(5, 3, 2)
    replays = [_Replay(), _Replay(), _Replay()]

    _register_replay_list_sequential(replays, replay_data, _models=None)

    for replay_idx, replay in enumerate(replays):
        assert len(replay.recorded) == 1
        torch.testing.assert_close(replay.recorded[0], replay_data[:, replay_idx])


def test_register_replay_list_sequential_rejects_shape_mismatch():
    replay_data = torch.zeros(5, 3, 2)
    replays = [_Replay(), _Replay()]

    with pytest.raises(AssertionError, match="3 streams"):
        _register_replay_list_sequential(replays, replay_data, _models=None)
