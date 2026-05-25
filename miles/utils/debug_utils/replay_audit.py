import os

import torch

ENABLE_ENV = "MILES_REPLAY_AUDIT_ENABLE"
KINDS_ENV = "MILES_REPLAY_AUDIT_KINDS"


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
    return f"replay_{kind}_stream_{int(stream_id):04d}"


def dump_sglang_capture_topk(
    *,
    kind: str,
    stream_id: int,
    top_indices: torch.Tensor,
    dims: str = "t topk # tp:replicated",
) -> None:
    """Dump the tensor written by an SGLang top-k capturer."""
    if not is_enabled(kind):
        return

    value = top_indices.detach()

    dumper = _get_sglang_dumper()
    if dumper is None:
        return

    dumper.dump(stream_name(kind, stream_id), value, dims=dims)


def dump_current_replay_topk(*, kind: str, manager, dims: str | None = None) -> None:
    """Dump the raw replay tensor most recently retrieved by the active Megatron module."""
    if not is_enabled(kind):
        return
    if getattr(manager, "stage", None) != "replay_forward":
        return

    replay = manager.get_current()
    if replay is None:
        return
    stream_id = getattr(replay, "source_stream_id", None)
    if stream_id is None:
        return

    top_indices = getattr(replay, "last_forward_top_indices_raw", None)
    if top_indices is None:
        return
    replay.last_forward_top_indices_raw = None

    value = _normalize_replay_topk_for_dump(top_indices.detach())
    if value.numel() == 0:
        return

    dumper = _get_sglang_dumper()
    if dumper is None:
        return

    dumper.dump(stream_name(kind, stream_id), value, dims=dims or target_dims(kind))


def target_dims(_kind: str) -> str:
    return "t topk"


def _normalize_replay_topk_for_dump(tensor: torch.Tensor) -> torch.Tensor:
    flat = _flatten_topk(tensor)
    if flat.numel() == 0:
        return flat

    cp_size = _parallel_size("cp")
    if cp_size <= 1:
        return _drop_padding_rows(flat)

    gathered = _all_gather_cp(flat)
    if gathered is None:
        return _drop_padding_rows(flat)

    if _parallel_rank("cp") != 0 or _parallel_rank("tp") != 0:
        return flat[:0]

    non_padding_rows = sum(_count_non_padding_rows(item) for item in gathered)
    if non_padding_rows == 0:
        return flat[:0]

    chunk_size = (non_padding_rows + 2 * cp_size - 1) // (2 * cp_size)
    local_len = 2 * chunk_size
    if any(item.shape[0] < local_len for item in gathered):
        return _drop_padding_rows(flat)

    chunks: list[torch.Tensor | None] = [None] * (2 * cp_size)
    for cp_rank, item in enumerate(gathered):
        local = item[:local_len]
        chunks[cp_rank] = local[:chunk_size]
        chunks[2 * cp_size - cp_rank - 1] = local[chunk_size:local_len]

    natural = torch.cat([chunk for chunk in chunks if chunk is not None], dim=0)
    return _drop_padding_rows(natural)


def _flatten_topk(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim <= 1:
        return tensor
    return tensor.reshape(-1, tensor.shape[-1]).contiguous()


def _drop_padding_rows(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor
    flat = _flatten_topk(tensor)
    keep = ~(flat == -1).all(dim=-1)
    return flat[keep].contiguous()


def _count_non_padding_rows(tensor: torch.Tensor) -> int:
    if tensor.ndim == 0:
        return int(tensor.numel() > 0)
    flat = _flatten_topk(tensor)
    return int((~(flat == -1).all(dim=-1)).sum().item())


def _all_gather_cp(tensor: torch.Tensor) -> list[torch.Tensor] | None:
    try:
        import torch.distributed as dist
        from megatron.core import parallel_state

        if not dist.is_available() or not dist.is_initialized():
            return None
        group = parallel_state.get_context_parallel_group()
        output = [torch.empty_like(tensor) for _ in range(_parallel_size("cp"))]
        dist.all_gather(output, tensor, group=group)
        return output
    except (ImportError, AssertionError, AttributeError, RuntimeError):
        return None


def _get_sglang_dumper():
    try:
        from sglang.srt.debug_utils.dumper import dumper
    except ImportError:
        return None
    return dumper


def _parallel_rank(axis: str) -> int:
    try:
        from megatron.core import parallel_state

        if axis == "tp":
            return parallel_state.get_tensor_model_parallel_rank()
        if axis == "cp":
            return parallel_state.get_context_parallel_rank()
    except (ImportError, AssertionError, AttributeError):
        return 0
    return 0


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
