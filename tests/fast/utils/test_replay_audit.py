from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

from types import SimpleNamespace

import torch

from miles.utils.debug_utils import replay_audit


class _FakeDumper:
    def __init__(self) -> None:
        self.calls = []

    def dump(self, name, value, dims=None):
        self.calls.append((name, value.detach().cpu().clone(), dims))


def test_dump_sglang_capture_topk_uses_stream_name(monkeypatch):
    fake = _FakeDumper()
    monkeypatch.setenv(replay_audit.ENABLE_ENV, "1")
    monkeypatch.setattr(replay_audit, "_get_sglang_dumper", lambda: fake)

    replay_audit.dump_sglang_capture_topk(
        kind="indexer",
        stream_id=3,
        top_indices=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        dims="t topk # tp:replicated",
    )

    assert len(fake.calls) == 1
    name, value, dims = fake.calls[0]
    assert name == "replay_indexer_stream_0003"
    torch.testing.assert_close(value, torch.tensor([[1, 2], [3, 4]], dtype=torch.int32))
    assert dims == "t topk # tp:replicated"


def test_dump_sglang_capture_topk_honors_kind_filter(monkeypatch):
    fake = _FakeDumper()
    monkeypatch.setenv(replay_audit.ENABLE_ENV, "1")
    monkeypatch.setenv(replay_audit.KINDS_ENV, "routing")
    monkeypatch.setattr(replay_audit, "_get_sglang_dumper", lambda: fake)

    replay_audit.dump_sglang_capture_topk(
        kind="indexer",
        stream_id=0,
        top_indices=torch.tensor([[1]], dtype=torch.int32),
    )

    assert fake.calls == []


def test_dump_current_replay_topk_drops_padding_rows_without_cp_group(monkeypatch):
    fake = _FakeDumper()
    monkeypatch.setenv(replay_audit.ENABLE_ENV, "1")
    monkeypatch.setattr(replay_audit, "_get_sglang_dumper", lambda: fake)
    monkeypatch.setattr(
        replay_audit,
        "_parallel_size",
        lambda axis: {"tp": 2, "cp": 2, "sp": 2, "ep": 4}.get(axis, 1),
    )

    replay = SimpleNamespace(
        source_stream_id=7,
        last_forward_top_indices_raw=torch.tensor([[4, 5], [-1, -1], [6, 7]], dtype=torch.int32),
    )
    manager = SimpleNamespace(stage="replay_forward", get_current=lambda: replay)

    replay_audit.dump_current_replay_topk(kind="indexer", manager=manager)

    assert len(fake.calls) == 1
    name, value, dims = fake.calls[0]
    assert name == "replay_indexer_stream_0007"
    torch.testing.assert_close(
        value, torch.tensor([[4, 5], [6, 7]], dtype=torch.int32)
    )
    assert dims == "t topk"
    assert replay.last_forward_top_indices_raw is None


def test_normalize_replay_topk_reconstructs_cp_zigzag(monkeypatch):
    rank0 = torch.tensor([[0], [1], [-1], [-1]], dtype=torch.int32)
    rank1 = torch.tensor([[2], [3], [4], [5]], dtype=torch.int32)
    monkeypatch.setattr(replay_audit, "_parallel_size", lambda axis: 2 if axis == "cp" else 1)
    monkeypatch.setattr(replay_audit, "_parallel_rank", lambda axis: 0)
    monkeypatch.setattr(replay_audit, "_all_gather_cp", lambda tensor: [rank0, rank1])

    value = replay_audit._normalize_replay_topk_for_dump(rank0)

    torch.testing.assert_close(value, torch.arange(6, dtype=torch.int32).view(6, 1))


def test_normalize_replay_topk_only_dumps_cp_tp_zero(monkeypatch):
    rank0 = torch.tensor([[0], [1], [-1], [-1]], dtype=torch.int32)
    rank1 = torch.tensor([[2], [3], [4], [5]], dtype=torch.int32)
    monkeypatch.setattr(replay_audit, "_parallel_size", lambda axis: 2 if axis == "cp" else 1)
    monkeypatch.setattr(replay_audit, "_parallel_rank", lambda axis: 1 if axis == "cp" else 0)
    monkeypatch.setattr(replay_audit, "_all_gather_cp", lambda tensor: [rank0, rank1])

    value = replay_audit._normalize_replay_topk_for_dump(rank1)

    assert value.numel() == 0


def test_dump_current_replay_topk_requires_stream_id(monkeypatch):
    fake = _FakeDumper()
    monkeypatch.setenv(replay_audit.ENABLE_ENV, "1")
    monkeypatch.setattr(replay_audit, "_get_sglang_dumper", lambda: fake)

    replay = SimpleNamespace(
        source_stream_id=None,
        last_forward_top_indices_raw=torch.tensor([[1, 2]], dtype=torch.int32),
    )
    manager = SimpleNamespace(stage="replay_forward", get_current=lambda: replay)

    replay_audit.dump_current_replay_topk(
        kind="routing",
        manager=manager,
    )

    assert fake.calls == []


def test_dump_current_replay_topk_requires_replay_forward_stage(monkeypatch):
    fake = _FakeDumper()
    monkeypatch.setenv(replay_audit.ENABLE_ENV, "1")
    monkeypatch.setattr(replay_audit, "_get_sglang_dumper", lambda: fake)

    replay = SimpleNamespace(
        source_stream_id=0,
        last_forward_top_indices_raw=torch.tensor([[1, 2]], dtype=torch.int32),
    )
    manager = SimpleNamespace(stage="record", get_current=lambda: replay)

    replay_audit.dump_current_replay_topk(kind="routing", manager=manager)

    assert fake.calls == []
