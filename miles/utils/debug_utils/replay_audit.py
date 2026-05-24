import os
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

ENABLE_ENV = "MILES_REPLAY_AUDIT_ENABLE"
SOURCE_DIR_ENV = "MILES_REPLAY_AUDIT_SOURCE_DIR"
KINDS_ENV = "MILES_REPLAY_AUDIT_KINDS"

SOURCE_DIMS = "t topk"
_CLEANED_SOURCE_DIRS: set[Path] = set()
_SOURCE_DUMP_INDEX = 0


def is_enabled(kind: str | None = None) -> bool:
    if os.environ.get(ENABLE_ENV, "0").lower() not in ("1", "true", "yes", "on"):
        return False
    if kind is None:
        return True
    raw_kinds = os.environ.get(KINDS_ENV)
    if raw_kinds is None:
        return True
    return kind in {item.strip() for item in raw_kinds.split(",") if item.strip()}


def stream_name(kind: str, stream_id: int) -> str:
    return f"replay_{kind}_stream_{stream_id:04d}"


def dump_source_replay_data(
    *,
    kind: str,
    replay_data: list[Any] | torch.Tensor,
    step: int,
    source_dir: str | Path | None = None,
) -> None:
    """Dump rollout replay tensors before Megatron-local parallel slicing."""

    if not is_enabled(kind):
        return
    if _rank() != 0:
        return

    raw_path = source_dir or os.environ.get(SOURCE_DIR_ENV)
    if raw_path is None or str(raw_path) == "":
        return
    path = Path(raw_path)

    tensor = _as_source_tensor(replay_data)
    if tensor.numel() == 0:
        return
    if tensor.ndim != 3:
        raise ValueError(
            f"Expected replay source tensor with shape [tokens, streams, topk], got {tuple(tensor.shape)}"
        )

    _cleanup_source_dir_once(path)
    for stream_id in range(tensor.shape[1]):
        _save_dumper_compatible_source(
            path=path,
            name=stream_name(kind, stream_id),
            value=tensor[:, stream_id],
            step=step,
        )


def dump_target_replay_indices(*, kind: str, replay: Any, top_indices: torch.Tensor) -> None:
    """Dump the replay tensor actually consumed by a Megatron topk replacement."""

    if not is_enabled(kind):
        return

    stream_id = getattr(replay, "source_stream_id", None)
    if stream_id is None:
        return

    value = _drop_padding_rows(top_indices.detach())
    if value.numel() == 0:
        return

    dumper = _get_sglang_dumper()
    if dumper is None:
        return

    dumper.dump(stream_name(kind, stream_id), value, dims=target_dims(kind))


def target_dims(kind: str) -> str:
    replicated_axes: list[str] = []
    if _parallel_size("tp") > 1:
        replicated_axes.append("tp:replicated")
    if kind == "indexer" and _parallel_size("sp") > 1:
        replicated_axes.append("sp:replicated")
    if _parallel_size("ep") > 1:
        replicated_axes.append("ep:replicated")

    suffix = f" # {' '.join(replicated_axes)}" if replicated_axes else ""
    if kind == "routing":
        return f"t[cp:zigzag,sp] topk{suffix}"
    if kind == "indexer":
        return f"t[cp:zigzag] topk{suffix}"
    return f"t topk{suffix}"


def _as_source_tensor(replay_data: list[Any] | torch.Tensor) -> torch.Tensor:
    if isinstance(replay_data, torch.Tensor):
        tensor = replay_data
    else:
        tensor = torch.cat([torch.as_tensor(item) for item in replay_data], dim=0)
    return tensor.detach().to(device="cpu", dtype=torch.int32).contiguous()


def _drop_padding_rows(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor
    flat = tensor.reshape(-1, tensor.shape[-1])
    keep = ~(flat == -1).all(dim=-1)
    return flat[keep].reshape(-1, tensor.shape[-1]).contiguous()


def _save_dumper_compatible_source(*, path: Path, name: str, value: torch.Tensor, step: int) -> None:
    global _SOURCE_DUMP_INDEX
    _SOURCE_DUMP_INDEX += 1

    tags = {
        "step": step,
        "rank": 0,
        "dump_index": _SOURCE_DUMP_INDEX,
        "name": name,
        "recompute_status": "disabled",
    }
    filename = "___".join(f"{key}={value}" for key, value in tags.items()) + ".pt"
    meta = {
        **tags,
        "world_rank": 0,
        "world_size": 1,
        "dims": SOURCE_DIMS,
    }
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"value": value.cpu().contiguous(), "meta": meta}, path / filename)


def _cleanup_source_dir_once(path: Path) -> None:
    resolved = path.resolve()
    if resolved in _CLEANED_SOURCE_DIRS:
        return
    if path.is_dir():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    _CLEANED_SOURCE_DIRS.add(resolved)


def _get_sglang_dumper():
    try:
        from sglang.srt.debug_utils.dumper import dumper
    except ImportError:
        return None
    return dumper


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _parallel_size(axis: str) -> int:
    try:
        from megatron.core import parallel_state

        if axis == "tp":
            return parallel_state.get_tensor_model_parallel_world_size()
        if axis == "cp":
            return parallel_state.get_context_parallel_world_size()
        if axis == "ep":
            return parallel_state.get_expert_model_parallel_world_size()
        if axis == "sp":
            try:
                from megatron.training.global_vars import get_args

                args = get_args()
                if getattr(args, "sequence_parallel", False):
                    return parallel_state.get_tensor_model_parallel_world_size()
                return 1
            except (ImportError, AssertionError, AttributeError):
                return 1
    except (ImportError, AssertionError, AttributeError):
        return 1
    return 1
