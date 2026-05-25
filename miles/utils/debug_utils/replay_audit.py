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

    # Keep padding rows so CP zigzag chunk boundaries remain recoverable.
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

    value = top_indices.detach()
    if value.numel() == 0:
        return

    dumper = _get_sglang_dumper()
    if dumper is None:
        return

    dumper.dump(stream_name(kind, stream_id), value, dims=dims or target_dims(kind))


def target_dims(_kind: str) -> str:
    if _parallel_size("cp") > 1:
        return "s[cp:zigzag] topk"
    return "t topk"


def _get_sglang_dumper():
    try:
        from sglang.srt.debug_utils.dumper import dumper
    except ImportError:
        return None
    return dumper


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
