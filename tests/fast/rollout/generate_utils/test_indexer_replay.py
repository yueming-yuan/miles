from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

from types import SimpleNamespace

import numpy as np
import pybase64
import pytest

from miles.rollout.generate_utils.generate_endpoint_utils import (
    get_indexer_replay_shape,
    get_indexer_topk_from_response,
)
from miles.utils.types import Sample


def _encode_int32(values: np.ndarray) -> str:
    return pybase64.b64encode(values.astype(np.int32).tobytes()).decode("ascii")


def test_get_indexer_topk_from_response_decodes_explicit_shape():
    args = SimpleNamespace(rollout_indexer_replay_num_layers=2, rollout_indexer_replay_topk=3)
    sample = Sample(tokens=[1, 2, 3])
    values = np.arange(2 * 2 * 3, dtype=np.int32)
    output = {"meta_info": {"indexer_topk": _encode_int32(values)}}

    decoded = get_indexer_topk_from_response(args, output, sample)

    np.testing.assert_array_equal(decoded, values.reshape(2, 2, 3))


def test_get_indexer_replay_shape_falls_back_to_model_attrs():
    args = SimpleNamespace(
        rollout_indexer_replay_num_layers=None,
        rollout_indexer_replay_topk=None,
        num_indexer_layers=4,
        index_topk=16,
    )

    assert get_indexer_replay_shape(args) == (4, 16)


def test_get_indexer_replay_shape_rejects_missing_shape():
    args = SimpleNamespace(rollout_indexer_replay_num_layers=None, rollout_indexer_replay_topk=None)

    with pytest.raises(AttributeError, match="rollout-indexer-replay-num-layers"):
        get_indexer_replay_shape(args)


def test_get_indexer_replay_shape_rejects_non_positive_shape():
    args = SimpleNamespace(rollout_indexer_replay_num_layers=0, rollout_indexer_replay_topk=16)

    with pytest.raises(ValueError, match="must be positive"):
        get_indexer_replay_shape(args)
