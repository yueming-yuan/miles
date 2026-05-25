import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)
_TRUTHY = ("1", "true", "yes", "on")


def _get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


class Replay:
    def __init__(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list: list[torch.Tensor] = []
        self.manager_name: str | None = None
        self.source_stream_id: int | None = None
        self.last_forward_top_indices_raw: torch.Tensor | None = None

    def record(self, top_indices: torch.Tensor):
        buf = torch.empty_like(top_indices, device="cpu", pin_memory=True)
        buf.copy_(top_indices)
        self.top_indices_list.append(buf)

    def pop_forward(self, device=None) -> torch.Tensor:
        if self.forward_index >= len(self.top_indices_list):
            shapes = [
                t.shape if isinstance(t, torch.Tensor) else f"non-tensor({type(t)})" for t in self.top_indices_list
            ]
            raise IndexError(
                f"pop_forward out of range: forward_index={self.forward_index}, "
                f"len(top_indices_list)={len(self.top_indices_list)}, shapes={shapes}"
            )
        top_indices = self.top_indices_list[self.forward_index]
        self.forward_index += 1
        result = top_indices.to(device if device is not None else torch.cuda.current_device())
        self.last_forward_top_indices_raw = result
        return result

    def pop_backward(self, device=None) -> torch.Tensor:
        top_indices = self.top_indices_list[self.backward_index]
        self.backward_index += 1
        return top_indices.to(device if device is not None else torch.cuda.current_device())

    def clear(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []
        self.last_forward_top_indices_raw = None

    def clear_forward(self):
        self.forward_index = 0
        self.last_forward_top_indices_raw = None


class BaseReplayManager:
    name: str = ""
    filename: str = ""

    def __init__(self):
        self.replays: list[Replay] = []
        self.current: Replay | None = None
        self.enabled = False
        self.stage = "fallthrough"

    def create_replay(self) -> Replay:
        replay = Replay()
        replay.manager_name = self.name
        self.replays.append(replay)
        return replay

    def set_current(self, replay: Replay):
        self.current = replay

    def get_current(self) -> Replay | None:
        return self.current

    def clear_all(self):
        for replay in self.replays:
            replay.clear()

    def clear_all_forward(self):
        for replay in self.replays:
            replay.clear_forward()

    def get_topk_fn(self, old_topk_fn, return_probs):
        manager = self

        def _get_replay_result(top_indices, scores, topk, *args, **kwargs):
            assert (
                top_indices.shape[0] == scores.shape[0]
            ), f"rank {_get_rank()}: replay n_tokens {top_indices.shape[0]} does not match scores n_tokens {scores.shape[0]}"

            assert (
                top_indices.shape[1] == topk
            ), f"replay topk does not match expected topk, replay topk {top_indices.shape[1]}, topk {topk}"

            if self.enable_check_replay_result:
                self.check_replay_result(old_topk_fn, scores, topk, top_indices, *args, **kwargs)

            padding_mask = top_indices == -1
            if padding_mask.any():
                top_indices = top_indices.clone()
                top_indices[padding_mask] = (
                    torch.arange(padding_mask.sum(), device=top_indices.device, dtype=top_indices.dtype)
                    % scores.shape[1]
                )

            if return_probs:
                return scores.gather(1, top_indices), top_indices
            else:
                return top_indices

        def new_topk_fn(scores, topk, *args, **kwargs):
            if not manager.enabled:
                return old_topk_fn(scores, topk, *args, **kwargs)

            stage = manager.stage
            replay = manager.get_current()

            if stage == "fallthrough":
                return old_topk_fn(scores, topk, *args, **kwargs)

            elif stage == "record":
                result = old_topk_fn(scores, topk, *args, **kwargs)
                if return_probs:
                    probs, top_indices = result
                else:
                    top_indices = result
                replay.record(top_indices)
                return result

            elif stage == "replay_forward":
                top_indices = replay.pop_forward(device=scores.device)
                return _get_replay_result(top_indices, scores, topk, *args, **kwargs)

            elif stage == "replay_backward":
                return _get_replay_result(replay.pop_backward(device=scores.device), scores, topk, *args, **kwargs)

            else:
                return old_topk_fn(scores, topk, *args, **kwargs)

        return new_topk_fn

    def register_to_module(self, module, attr_name: str):
        if not self.enabled:
            return
        replay = self.create_replay()
        setattr(module, attr_name, replay)
        manager = self

        def pre_forward_hook(*args, **kwargs):
            manager.set_current(replay)

        module.register_forward_pre_hook(pre_forward_hook)

    def check_replay_result(self, old_topk_fn, scores, topk, top_indices, *args, **kwargs):
        """
        CI checker for R3. Only enable when enable_check_replay_result=True.
        Calculate the overlapping between training engine's computed routing result
        and replay routing result.
        If mismatch token count > n_tokens * replay_check_threshold, raise error.
        """
        if os.environ.get("MILES_DISABLE_REPLAY_RESULT_CHECK", "0").lower() in _TRUTHY:
            return

        orig_top_indices = old_topk_fn(scores, topk, *args, **kwargs)
        if isinstance(orig_top_indices, tuple):
            _, orig_top_indices = orig_top_indices

        orig_flat = orig_top_indices.reshape(-1, orig_top_indices.shape[-1])  # [n_tokens, topk]
        replay_flat = top_indices.reshape(-1, top_indices.shape[-1])

        has_overlap, is_padding = self._rowwise_topk_overlap(orig_flat, replay_flat)
        is_mismatch = ~has_overlap & ~is_padding

        mismatch_count = is_mismatch.sum().item()
        if mismatch_count == 0:
            return

        threshold = float(os.environ.get("MILES_TEST_R3_THRESHOLD", self.replay_check_threshold))
        mismatch_threshold = threshold * orig_flat.shape[0]
        mismatch_indices = is_mismatch.nonzero(as_tuple=False).squeeze(1)
        log_limit = int(os.environ.get("MILES_REPLAY_CHECK_LOG_LIMIT", "8"))
        for idx in mismatch_indices[:log_limit]:
            i = idx.item()
            lines = []
            for j in range(max(0, i - 3), min(len(orig_flat), i + 4)):
                marker = " <<<" if j == i else ""
                lines.append(f"  token {j}: orig={orig_flat[j].tolist()}, replay={replay_flat[j].tolist()}{marker}")
            logger.warning(
                f"Replay check (rank {_get_rank()}, stage {self.stage}): "
                f"token {i} zero overlap, topk={topk}\n" + "\n".join(lines)
            )
        if len(mismatch_indices) > log_limit:
            logger.warning(
                f"Replay check (rank {_get_rank()}, stage {self.stage}): "
                f"suppressed {len(mismatch_indices) - log_limit} additional mismatch logs"
            )

        if mismatch_count > mismatch_threshold:
            raise AssertionError(f"R3 mismatch tokens ({mismatch_count}) > threshold ({mismatch_threshold:.0f})")

    def _rowwise_topk_overlap(self, orig_flat: torch.Tensor, replay_flat: torch.Tensor):
        if orig_flat.shape[0] != replay_flat.shape[0]:
            raise AssertionError(
                f"replay n_tokens {replay_flat.shape[0]} does not match orig n_tokens {orig_flat.shape[0]}"
            )

        if replay_flat.shape[1] == 0:
            has_overlap = torch.zeros(orig_flat.shape[0], device=orig_flat.device, dtype=torch.bool)
            is_padding = torch.ones(replay_flat.shape[0], device=replay_flat.device, dtype=torch.bool)
            return has_overlap, is_padding

        chunk_rows = int(os.environ.get("MILES_REPLAY_CHECK_CHUNK_ROWS", "1024"))
        if chunk_rows <= 0:
            chunk_rows = orig_flat.shape[0]

        overlap_chunks = []
        padding_chunks = []
        for start in range(0, orig_flat.shape[0], chunk_rows):
            end = min(start + chunk_rows, orig_flat.shape[0])
            orig_chunk = orig_flat[start:end].contiguous()
            replay_chunk = replay_flat[start:end].contiguous()
            if orig_chunk.dtype != replay_chunk.dtype:
                orig_chunk = orig_chunk.to(replay_chunk.dtype)

            sorted_replay = torch.sort(replay_chunk, dim=1).values.contiguous()
            positions = torch.searchsorted(sorted_replay, orig_chunk.contiguous())
            positions = positions.clamp(max=sorted_replay.shape[1] - 1)
            found = sorted_replay.gather(1, positions) == orig_chunk

            overlap_chunks.append((found & (orig_chunk != -1)).any(dim=1))
            padding_chunks.append((replay_chunk == -1).all(dim=1))

        return torch.cat(overlap_chunks, dim=0), torch.cat(padding_chunks, dim=0)


class RoutingReplayManager(BaseReplayManager):
    name = "routing"
    filename = "routing_replay.pt"
    data_key = "rollout_routed_experts"
    if_sp_region = True
    enable_check_replay_result = False
    replay_check_threshold = 1e-2


class IndexerReplayManager(BaseReplayManager):
    name = "indexer"
    filename = "indexer_replay.pt"
    data_key = "rollout_indexer_topk"
    if_sp_region = False
    enable_check_replay_result = False
    replay_check_threshold = 0.7


routing_replay_manager = RoutingReplayManager()
indexer_replay_manager = IndexerReplayManager()
all_replay_managers = [routing_replay_manager, indexer_replay_manager]
