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


def test_dump_source_replay_data_writes_dumper_compatible_files(tmp_path, monkeypatch):
    monkeypatch.setenv(replay_audit.ENABLE_ENV, "1")
    monkeypatch.setattr(replay_audit, "_CLEANED_SOURCE_DIRS", set())
    monkeypatch.setattr(replay_audit, "_SOURCE_DUMP_INDEX", 0)

    first = torch.arange(2 * 2 * 2, dtype=torch.int32).reshape(2, 2, 2)
    second = torch.arange(8, 12, dtype=torch.int32).reshape(1, 2, 2)

    replay_audit.dump_source_replay_data(
        kind="indexer",
        replay_data=[first, second],
        step=3,
        source_dir=tmp_path,
    )

    files = sorted(tmp_path.glob("*.pt"))
    assert len(files) == 2

    by_name = {torch.load(path, weights_only=False)["meta"]["name"]: path for path in files}
    item = torch.load(by_name["replay_indexer_stream_0001"], weights_only=False)

    assert item["meta"]["step"] == 3
    assert item["meta"]["rank"] == 0
    assert item["meta"]["dims"] == replay_audit.SOURCE_DIMS
    torch.testing.assert_close(
        item["value"],
        torch.tensor([[2, 3], [6, 7], [10, 11]], dtype=torch.int32),
    )


def test_dump_source_replay_data_noops_without_source_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(replay_audit.ENABLE_ENV, "1")
    monkeypatch.delenv(replay_audit.SOURCE_DIR_ENV, raising=False)
    monkeypatch.chdir(tmp_path)

    replay_audit.dump_source_replay_data(
        kind="routing",
        replay_data=torch.zeros(1, 1, 1, dtype=torch.int32),
        step=0,
    )

    assert list(tmp_path.iterdir()) == []


def test_dump_target_replay_indices_filters_padding_rows(monkeypatch):
    fake = _FakeDumper()
    monkeypatch.setenv(replay_audit.ENABLE_ENV, "1")
    monkeypatch.setattr(replay_audit, "_get_sglang_dumper", lambda: fake)
    monkeypatch.setattr(
        replay_audit,
        "_parallel_size",
        lambda axis: {"tp": 2, "sp": 2, "ep": 4}.get(axis, 1),
    )

    replay = SimpleNamespace(source_stream_id=7)
    top_indices = torch.tensor([[4, 5], [-1, -1], [6, 7]], dtype=torch.int32)

    replay_audit.dump_target_replay_indices(kind="indexer", replay=replay, top_indices=top_indices)

    assert len(fake.calls) == 1
    name, value, dims = fake.calls[0]
    assert name == "replay_indexer_stream_0007"
    torch.testing.assert_close(value, torch.tensor([[4, 5], [6, 7]], dtype=torch.int32))
    assert dims == "t[cp:zigzag] topk # tp:replicated sp:replicated ep:replicated"


def test_dump_target_replay_indices_requires_stream_id(monkeypatch):
    fake = _FakeDumper()
    monkeypatch.setenv(replay_audit.ENABLE_ENV, "1")
    monkeypatch.setattr(replay_audit, "_get_sglang_dumper", lambda: fake)

    replay_audit.dump_target_replay_indices(
        kind="routing",
        replay=SimpleNamespace(source_stream_id=None),
        top_indices=torch.tensor([[1, 2]], dtype=torch.int32),
    )

    assert fake.calls == []
