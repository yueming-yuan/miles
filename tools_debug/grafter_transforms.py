"""Grafter transforms for V4 sg↔mg bidirectional cross-stack tensor exchange.

Used as DUMPER_GRAFTER_TRANSFORM_PATH=tools_debug.grafter_transforms.transform.

IMPORTANT: `g.received_list` (passed to the transform) is ALREADY filtered to
just the SENDER ranks (8 entries, not 16) — the grafter's `_sender_slice`
strips out the receiver-side `None` placeholders before invoking the
transform. So:
  - On the SG (target) side receiving b2t: g.received_list has 8 mg tensors.
  - On the MG (baseline) side receiving t2b: g.received_list has 8 sg tensors.

Direction is determined by the receiver's role (env var DUMPER_GRAFTER_ROLE),
NOT by inspecting received_list (which always has senders in slots 0..7).

Sharding semantics (verified against `c4.cuh` source + cr=128 cross-check):
  - layer_input, compress_input  : mg SP-sharded T (cat dim 0); sg full T on rank 0
  - attn_q, attn_v               : mg TP-sharded heads + SP-sharded T;  sg full on rank 0
  - attn_output                  : mg TP-sharded heads + SP-sharded T;  sg full on rank 0
  - compress_final_out (b2t)     : mg replicated per-block (1, T_compressed, head_dim);
                                   sg per-query (T, head_dim) on rank 0; expand via
                                   sg[t] = mg[t // ratio]
  - compress_final_out (t2b)     : sg per-query (T, head_dim) — c4.cuh writes only at
                                   block-end positions; in-between rows are uninitialized
                                   garbage. mg per-block expects block-end mapping
                                   mg[k] = sg[(k+1)*ratio - 1].
"""

from __future__ import annotations

import os

import torch


_BASELINE_WORLD = int(os.environ.get("DUMPER_GRAFTER_BASELINE_WORLD_SIZE", "8"))
_TARGET_WORLD = int(os.environ.get("DUMPER_GRAFTER_TARGET_WORLD_SIZE", "8"))


def _my_direction():
    """Direction this rank should dispatch to, based on its role.

    baseline rank receives from target → 't2b'.
    target rank receives from baseline → 'b2t'.
    """
    role = os.environ.get("DUMPER_GRAFTER_ROLE", "")
    if role == "baseline":
        return "t2b"
    if role == "target":
        return "b2t"
    raise RuntimeError(f"unknown DUMPER_GRAFTER_ROLE={role!r}")


def _sender_parts(received_list):
    """Return the sender contributions, detached. received_list is already filtered to senders."""
    return [t.detach() if isinstance(t, torch.Tensor) else t for t in received_list]


def _b2t_cat_sp_then_squeeze(g):
    """Common b2t pattern: mg SP-sharded T (each rank has bsz dim) → sg full T."""
    target = g.target
    if target.shape[0] == 0:
        return target
    parts = [t for t in _sender_parts(g.received_list) if t is not None]
    if not parts:
        return target
    full = torch.cat(parts, dim=0)
    if full.dim() >= 2 and full.shape[1] == 1:
        full = full.squeeze(1)
    full = full[: target.shape[0]]
    return full.to(device=target.device, dtype=target.dtype).contiguous()


def transform_layer_input_b2t(g):
    return _b2t_cat_sp_then_squeeze(g)


def transform_compress_input_b2t(g):
    """mg: (bsz=1, T/sp, hidden) per rank → sg: (T_actual, hidden) on rank 0.

    bsz dim is leading on mg (unlike layer_input where T is leading); so we
    squeeze the bsz=1 dim first, then cat on T dim across SP ranks.
    """
    target = g.target
    if target.shape[0] == 0:
        return target
    parts = [t for t in _sender_parts(g.received_list) if t is not None]
    if not parts:
        return target
    parts_squeezed = [p.squeeze(0) if p.dim() == 3 and p.shape[0] == 1 else p for p in parts]
    full = torch.cat(parts_squeezed, dim=0)
    full = full[: target.shape[0]]
    return full.to(device=target.device, dtype=target.dtype).contiguous()


def transform_attn_q_b2t(g):
    """mg per-rank (1, T, n_h_local, head_dim) TP-sharded heads → sg full (T_actual, n_heads, head_dim)."""
    target = g.target
    if target.shape[0] == 0:
        return target
    parts = [t for t in _sender_parts(g.received_list) if t is not None]
    if not parts:
        return target
    full = torch.cat(parts, dim=2)
    if full.dim() >= 2 and full.shape[0] == 1:
        full = full.squeeze(0)
    full = full[: target.shape[0]]
    return full.to(device=target.device, dtype=target.dtype).contiguous()


def transform_attn_v_b2t(g):
    """mg per-rank (1, T, head_dim) replicated → sg (T_actual, head_dim) on rank 0.

    Note: attn_v on mg is the merged KV (kv_vanilla), replicated across all 8 ranks
    (NOT TP-sharded), so any one rank suffices. Different from attn_q which is TP-sharded.
    """
    target = g.target
    if target.shape[0] == 0:
        return target
    parts = [t for t in _sender_parts(g.received_list) if t is not None]
    if not parts:
        return target
    p = parts[0]  # all mg ranks have identical content
    if p.dim() == 3 and p.shape[0] == 1:
        p = p.squeeze(0)
    p = p[: target.shape[0]]
    return p.to(device=target.device, dtype=target.dtype).contiguous()


def transform_mqa_wo_b_out_b2t(g):
    """mg per-rank (1, T_padded, hidden=4096) replicated post-RowParallel-allreduce
    → sg (T_actual, hidden) on active DP rank.

    All 8 mg ranks have identical mqa_wo_b_out (post-allreduce, pre-SP-scatter).
    Take any one rank's data, squeeze bsz=1, slice T_padded → T_actual.
    Same shape pattern as transform_attn_v_b2t but with hidden_size dim instead of head_dim.
    """
    return transform_attn_v_b2t(g)


def transform_attn_output_b2t(g):
    return transform_attn_q_b2t(g)


def transform_attn_output_t2b(g):
    """sg per-query (T_actual, n_heads_total, head_dim) on active rank → mg per-rank
    (1, T_padded, n_heads_local, head_dim) TP-sharded heads.

    mg has TP=8, so n_heads_local = n_heads_total // 8. Each baseline rank R should
    receive heads [R*n_h_local : (R+1)*n_h_local].

    Padding T_actual..T_padded on mg side stays unchanged (target.clone()) — those
    correspond to attention positions past the actual sequence; mg's local mask already
    treats them as ignored.
    """
    target = g.target  # (1, T_padded, n_h_local, head_dim) typically
    sg_full = None
    for r in _sender_parts(g.received_list):
        if r is None:
            continue
        if r.shape[0] == 0:
            continue
        sg_full = r  # (T_actual, n_heads_total, head_dim)
        break
    if sg_full is None:
        return target
    sp_idx = _my_baseline_rank()  # 0..7 (mg's local rank inside baseline group)
    if target.dim() == 4 and target.shape[0] == 1:
        n_h_local = target.shape[2]
    elif target.dim() == 3:
        n_h_local = target.shape[1]
    else:
        return target  # unexpected shape, bail
    head_start = sp_idx * n_h_local
    head_end = head_start + n_h_local
    sg_slice = sg_full[:, head_start:head_end, :]  # (T_actual, n_h_local, head_dim)
    out = target.clone()
    T_actual = sg_slice.shape[0]
    if target.dim() == 4:
        T_target = target.shape[1]
        T_copy = min(T_actual, T_target)
        out[0, :T_copy] = sg_slice[:T_copy].to(device=target.device, dtype=target.dtype)
    else:
        T_copy = min(T_actual, target.shape[0])
        out[:T_copy] = sg_slice[:T_copy].to(device=target.device, dtype=target.dtype)
    return out


def transform_compress_final_out_b2t(g):
    """mg replicated per-block (1, T_compressed, head_dim) → sg per-query (T, head_dim).

    Map each query t to the block containing it: sg[t] = mg[t // ratio].
    """
    target = g.target
    if target.shape[0] == 0:
        return target
    cr = int(g.tags.get("compress_ratio", 0))
    if cr <= 0:
        return target
    mg_full = None
    for r in _sender_parts(g.received_list):
        if r is None:
            continue
        if r.shape[-1] != 512:
            continue
        mg_full = r.squeeze(0) if r.dim() >= 2 and r.shape[0] == 1 else r
        break
    if mg_full is None:
        return target
    out = target.clone()
    n_blocks = mg_full.shape[0]
    for t_idx in range(target.shape[0]):
        b = t_idx // cr
        if b < n_blocks:
            out[t_idx] = mg_full[b].to(device=target.device, dtype=target.dtype)
    return out


def _my_baseline_rank():
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


def transform_compress_final_out_t2b(g):
    """sg per-query (T_actual, head_dim=512) → mg per-block (1, T_compressed, head_dim=512).

    mg's compressor input is full-T (not SP-sharded inside the compressor module),
    so each baseline rank has the FULL T_compressed blocks identically. We fill all
    blocks on every mg rank with the same per-query → per-block extraction:
    mg[k] = sg[(k+1)*ratio - 1]   (block-end mapping, c4.cuh / cr=128 verified).

    Padding blocks past T_actual (e.g. last ~12 blocks for T_actual=2126,
    T_compressed=544) are LEFT UNCHANGED at mg's local value — they correspond to
    padding tokens that mg won't use for valid logprob computation.

    Skip indexer compressor (head_dim=128) and empty (idle) senders.
    """
    target = g.target
    cr = int(g.tags.get("compress_ratio", 0))
    if cr <= 0:
        return target
    sg_full = None
    for r in _sender_parts(g.received_list):
        if r is None:
            continue
        if r.shape[0] == 0:
            continue
        if r.shape[-1] != 512:
            continue
        sg_full = r.squeeze(0) if r.dim() >= 2 and r.shape[0] == 1 else r
        break
    if sg_full is None:
        return target
    T_compressed = target.shape[-2]
    out = target.clone()
    flat = out.view(-1, target.shape[-1]) if out.dim() > 2 else out
    for k in range(T_compressed):
        t_idx = (k + 1) * cr - 1
        if t_idx < sg_full.shape[0]:
            flat[k] = sg_full[t_idx].to(device=target.device, dtype=target.dtype)
    return out


def transform_layer_input_t2b(g):
    """sg per-query (T_actual, hc, h) on active rank → mg per-rank (T_sp, 1, hc, h) SP-sharded.

    Each baseline rank R takes T slice [R*T_sp : (R+1)*T_sp] from sg's full tensor,
    then unsqueeze dim 1 to insert the bsz=1 dim mg expects. Padding past T_actual
    stays at mg's original (target.clone()).
    """
    target = g.target
    sg_full = None
    for r in _sender_parts(g.received_list):
        if r is None:
            continue
        if r.shape[0] == 0:
            continue
        sg_full = r.squeeze(0) if r.dim() >= 2 and r.shape[0] == 1 else r
        break
    if sg_full is None:
        return target
    sp_idx = _my_baseline_rank()
    if target.dim() == 4 and target.shape[1] == 1:
        T_per_rank = target.shape[0]
    elif target.dim() == 3:
        T_per_rank = target.shape[0]
    else:
        return target
    t_start = sp_idx * T_per_rank
    t_end = t_start + T_per_rank
    sg_slice = sg_full[t_start:t_end]
    out = target.clone()
    T_copy = min(sg_slice.shape[0], T_per_rank)
    if target.dim() == 4:
        out[:T_copy, 0] = sg_slice[:T_copy].to(device=target.device, dtype=target.dtype)
    else:
        out[:T_copy] = sg_slice[:T_copy].to(device=target.device, dtype=target.dtype)
    return out


def transform_pre_mlp_layernorm_output_b2t(g):
    """mg per-rank (T_sp, 1, h) SP-sharded T → sg full (T_actual, h) on rank 0.

    Same shape pattern as layer_input: SP-sharded over dim 0 with bsz=1 at dim 1.
    """
    return _b2t_cat_sp_then_squeeze(g)


def transform_mlp_output_t2b(g):
    """sg per-query (T_actual, h) on active rank → mg per-rank (T_sp, 1, h) SP-sharded.

    Each baseline rank R takes T slice [R*T_sp : (R+1)*T_sp] from sg's full tensor.
    Padding (T_actual..T_padded) stays unchanged at mg's local value.
    """
    target = g.target
    sg_full = None
    for r in _sender_parts(g.received_list):
        if r is None:
            continue
        if r.shape[0] == 0:
            continue
        sg_full = r.squeeze(0) if r.dim() >= 2 and r.shape[0] == 1 else r
        break
    if sg_full is None:
        return target
    sp_idx = _my_baseline_rank()  # mg's local rank inside baseline group, 0..7
    if target.dim() == 3 and target.shape[1] == 1:
        T_per_rank = target.shape[0]
    elif target.dim() == 2:
        T_per_rank = target.shape[0]
    else:
        return target
    t_start = sp_idx * T_per_rank
    t_end = t_start + T_per_rank
    sg_slice = sg_full[t_start:t_end]
    out = target.clone()
    if target.dim() == 3:
        T_copy = min(sg_slice.shape[0], T_per_rank)
        out[:T_copy, 0, :] = sg_slice[:T_copy].to(device=target.device, dtype=target.dtype)
    else:
        T_copy = min(sg_slice.shape[0], T_per_rank)
        out[:T_copy] = sg_slice[:T_copy].to(device=target.device, dtype=target.dtype)
    return out


def transform_post_norm_hidden_t2b(g):
    """sg per-query (T_actual, h) on active rank → mg per-rank (T_sp, 1, h) SP-sharded.

    Same shape pattern as mlp_output_t2b: post-final_layernorm hidden states.
    Fires once per forward (outside the layer loop, layer_id is None on both sides).
    """
    return transform_mlp_output_t2b(g)


def transform_moe_routing_map_t2b(g):
    """sg dense (T_actual, num_experts=256) bool → mg per-rank (T_sp, 256) bool SP-sharded T.

    Each baseline rank R takes T slice [R*T_sp : (R+1)*T_sp] from sg's full tensor.
    Same shape pattern as mlp_output_t2b but 2D and bool dtype.
    """
    return transform_mlp_output_t2b(g)


def transform_moe_probs_t2b(g):
    """sg dense (T_actual, num_experts=256) float/bf16 → mg per-rank (T_sp, 256) SP-sharded T."""
    return transform_mlp_output_t2b(g)


def transform_hc_attn_pre_t2b(g):
    """sg post-hc_pre attn-side hidden (T_actual, h) → mg per-rank (T_sp, 1, h) SP-sharded.

    Same shape pattern as mlp_output_t2b: post-hc_pre on attention side, fed to input_layernorm.
    """
    return transform_mlp_output_t2b(g)


def transform_hc_ffn_pre_t2b(g):
    """sg post-hc_pre ffn-side hidden (T_actual, h) → mg per-rank (T_sp, 1, h) SP-sharded."""
    return transform_mlp_output_t2b(g)


def transform_hc_attn_post_t2b(g):
    """sg post-hc_post attn-side recombined (T_actual, hc=4, h) → mg per-rank (T_sp, 1, hc, h)
    SP-sharded. Same shape pattern as layer_input_t2b."""
    return transform_layer_input_t2b(g)


def transform_hc_ffn_post_t2b(g):
    """sg post-hc_post ffn-side recombined (T_actual, hc=4, h) → mg per-rank (T_sp, 1, hc, h)
    SP-sharded. Same shape pattern as layer_input_t2b."""
    return transform_layer_input_t2b(g)


def transform_lm_head_logits_t2b(g):
    """sg gathered logits (T_actual, V_sg=129280) on active rank → mg gathered logits
    (b=1, s=T_padded, V_mg=130048) per rank (replicated post-TP-all-gather).

    Vocab dim differs by 768 (mg pads to TP-divisible). The transform copies sg's
    values into mg's first V_sg columns and overwrites mg's padding columns with a
    large negative value so mg's log_softmax denominator is effectively normalized
    over V_sg, matching sg's behavior. T_padded > T_actual padding rows untouched
    (mg's loss_mask skips them).

    Fires once per forward; layer_id None on both sides.
    """
    target = g.target
    sg_full = None
    for r in _sender_parts(g.received_list):
        if r is None:
            continue
        if r.shape[0] == 0:
            continue
        sg_full = r
        break
    if sg_full is None:
        return target
    if target.dim() == 3 and target.shape[0] == 1:
        T_padded = target.shape[1]
        V_mg = target.shape[2]
    elif target.dim() == 2:
        T_padded = target.shape[0]
        V_mg = target.shape[1]
    else:
        return target
    V_sg = sg_full.shape[-1]
    T_copy = min(sg_full.shape[0], T_padded)
    out = target.clone()
    sg_to_target = sg_full[:T_copy].to(device=target.device, dtype=target.dtype)
    if target.dim() == 3:
        out[0, :T_copy, :V_sg] = sg_to_target
        if V_mg > V_sg:
            out[0, :T_copy, V_sg:] = torch.finfo(target.dtype).min / 2
    else:
        out[:T_copy, :V_sg] = sg_to_target
        if V_mg > V_sg:
            out[:T_copy, V_sg:] = torch.finfo(target.dtype).min / 2
    return out


def transform_input_layernorm_b2t(g):
    """mg per-rank (T_sp, 1, h) SP-sharded → sg full (T_actual, h).

    Same shape pattern as pre_mlp_layernorm_output: SP-sharded post-RMSNorm hidden state,
    dumped by mg's TransformerLayer._forward_attention right after `self.input_layernorm(hidden_states)`.
    """
    return _b2t_cat_sp_then_squeeze(g)


def transform_attn_q_t2b(g):
    """sg full (T_actual, n_heads_total=64, head_dim=512) → mg per-rank
    (1, T_padded, n_h_local=8, head_dim=512) TP-sharded heads.

    Per rank R, slice heads [R*n_h_local : (R+1)*n_h_local] from sg's full tensor.
    Padding rows past T_actual stay at mg's local value (target.clone()).
    Same logic as transform_attn_output_t2b.
    """
    return transform_attn_output_t2b(g)


def transform_attn_v_t2b(g):
    """sg full (T_actual, head_dim=512) → mg per-rank (1, T_padded, head_dim=512) REPLICATED.

    attn_v is the merged KV (kv_vanilla, single-head MLA), replicated across all 8 mg ranks
    (NOT TP-sharded). All ranks get the same sg values at [0, :T_actual, :]. Padding rows
    [T_actual:T_padded] stay at mg's local value.
    """
    target = g.target
    sg_full = None
    for r in _sender_parts(g.received_list):
        if r is None:
            continue
        if r.shape[0] == 0:
            continue
        sg_full = r
        break
    if sg_full is None:
        return target
    if target.dim() == 3 and target.shape[0] == 1:
        T_padded = target.shape[1]
    elif target.dim() == 2:
        T_padded = target.shape[0]
    else:
        return target
    T_copy = min(sg_full.shape[0], T_padded)
    out = target.clone()
    sg_to_target = sg_full[:T_copy].to(device=target.device, dtype=target.dtype)
    if target.dim() == 3:
        out[0, :T_copy] = sg_to_target
    else:
        out[:T_copy] = sg_to_target
    return out


_DISPATCH = {
    ("layer_input", "b2t"): transform_layer_input_b2t,
    ("layer_input", "t2b"): transform_layer_input_t2b,
    ("compress_input", "b2t"): transform_compress_input_b2t,
    ("input_layernorm", "b2t"): transform_input_layernorm_b2t,
    ("attn_q", "b2t"): transform_attn_q_b2t,
    ("attn_q", "t2b"): transform_attn_q_t2b,
    ("attn_v", "b2t"): transform_attn_v_b2t,
    ("attn_v", "t2b"): transform_attn_v_t2b,
    ("mqa_wo_b_out", "b2t"): transform_mqa_wo_b_out_b2t,
    ("mqa_wo_b_out", "t2b"): transform_attn_v_t2b,  # same layout: sg (T,h) → mg (1,T_padded,h) replicated
    ("attn_output", "b2t"): transform_attn_output_b2t,
    ("attn_output", "t2b"): transform_attn_output_t2b,
    ("compress_final_out", "b2t"): transform_compress_final_out_b2t,
    ("compress_final_out", "t2b"): transform_compress_final_out_t2b,
    ("pre_mlp_layernorm_output", "b2t"): transform_pre_mlp_layernorm_output_b2t,
    ("mlp_output", "t2b"): transform_mlp_output_t2b,
    ("post_norm_hidden", "t2b"): transform_post_norm_hidden_t2b,
    ("lm_head_logits", "t2b"): transform_lm_head_logits_t2b,
    ("hc_attn_pre", "t2b"): transform_hc_attn_pre_t2b,
    ("hc_attn_post", "t2b"): transform_hc_attn_post_t2b,
    ("hc_ffn_pre", "t2b"): transform_hc_ffn_pre_t2b,
    ("hc_ffn_post", "t2b"): transform_hc_ffn_post_t2b,
    ("moe_routing_map", "t2b"): transform_moe_routing_map_t2b,
    ("moe_probs", "t2b"): transform_moe_probs_t2b,
}


def transform(g):
    name = g.tags.get("name") if hasattr(g, "tags") else None
    direction = _my_direction()
    fn = _DISPATCH.get((name, direction))
    if fn is None:
        raise RuntimeError(f"grafter_transforms: no transform registered for name={name!r} direction={direction!r}")
    return fn(g)
