"""Unit tests for grafter_transforms.py — exercise transform fns on synthetic
inputs that mimic actual mg/sg dump shapes for both b2t and t2b directions.

`received_list` in `g` is the SENDER contributions only (8 tensors), as the
grafter's `_sender_slice` filters before invoking the transform.

Run on pod:
    cd /workspace/miles && python tools_debug/test_grafter_transforms.py
"""
from __future__ import annotations

import os
import sys
os.environ.setdefault("DUMPER_GRAFTER_BASELINE_WORLD_SIZE", "8")
os.environ.setdefault("DUMPER_GRAFTER_TARGET_WORLD_SIZE", "8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from grafter_transforms import (
    transform_attn_output_b2t,
    transform_attn_output_t2b,
    transform_attn_q_b2t,
    transform_attn_q_t2b,
    transform_attn_v_b2t,
    transform_attn_v_t2b,
    transform_compress_final_out_b2t,
    transform_compress_final_out_t2b,
    transform_compress_input_b2t,
    transform_input_layernorm_b2t,
    transform_layer_input_b2t,
    transform_layer_input_t2b,
    transform_lm_head_logits_t2b,
    transform_mlp_output_t2b,
    transform_post_norm_hidden_t2b,
    transform_pre_mlp_layernorm_output_b2t,
    transform,
)


class FakeGraftInput:
    def __init__(self, *, received_list, target, tags):
        self.received_list = received_list
        self.target = target
        self.tags = tags


# --- b2t transforms (sg side, role=target, received_list = mg sender contribs) ---

def test_attn_output_b2t():
    """8 mg ranks each contribute (1, T_padded=2176, n_heads_local=8, head_dim=512); sg target (T_actual=2126, 64, 512)."""
    T_padded, T_actual, n_heads_local, head_dim = 2176, 2126, 8, 512
    parts = [torch.randn(1, T_padded, n_heads_local, head_dim, dtype=torch.bfloat16) for _ in range(8)]
    target = torch.zeros(T_actual, n_heads_local * 8, head_dim, dtype=torch.bfloat16)
    g = FakeGraftInput(received_list=parts, target=target,
                       tags={"name": "attn_output", "compress_ratio": 4})
    out = transform_attn_output_b2t(g)
    assert out.shape == target.shape
    expected = torch.cat(parts, dim=2).squeeze(0)[:T_actual]
    assert torch.allclose(out.float(), expected.float())
    print("test_attn_output_b2t PASS")


def test_attn_output_b2t_empty_target():
    """sg rank 1-7 with DP-attention: target shape[0] == 0 → no-op."""
    parts = [torch.randn(1, 2176, 8, 512, dtype=torch.bfloat16) for _ in range(8)]
    target = torch.zeros(0, 64, 512, dtype=torch.bfloat16)
    g = FakeGraftInput(received_list=parts, target=target,
                       tags={"name": "attn_output", "compress_ratio": 4})
    out = transform_attn_output_b2t(g)
    assert out.shape == target.shape and out.numel() == 0
    print("test_attn_output_b2t_empty_target PASS")


def test_compress_final_out_b2t():
    """mg replicated per-block (1, T_compressed=544, head_dim=512) → sg per-query (T_actual=2126, 512)."""
    T_compressed, T_actual, head_dim, ratio = 544, 2126, 512, 4
    mg_full = torch.randn(1, T_compressed, head_dim, dtype=torch.bfloat16)
    parts = [mg_full] * 8
    target = torch.zeros(T_actual, head_dim, dtype=torch.bfloat16)
    g = FakeGraftInput(received_list=parts, target=target,
                       tags={"name": "compress_final_out", "compress_ratio": 4})
    out = transform_compress_final_out_b2t(g)
    assert out.shape == target.shape
    for t in [0, 3, 4, 7, 1021, 2125]:
        b = t // ratio
        if b < T_compressed:
            assert torch.allclose(out[t].float(), mg_full[0, b].to(out.dtype).float()), \
                f"row {t} != mg block {b}"
    print("test_compress_final_out_b2t PASS")


def test_compress_final_out_b2t_indexer_skipped():
    """head_dim=128 (indexer compressor) → return target unchanged (filter mismatches in real run)."""
    T_compressed, T_actual, head_dim, ratio = 544, 2126, 128, 4
    mg_indexer = torch.randn(1, T_compressed, head_dim, dtype=torch.bfloat16)
    parts = [mg_indexer] * 8
    target = torch.full((T_actual, head_dim), 7.0, dtype=torch.bfloat16)
    g = FakeGraftInput(received_list=parts, target=target,
                       tags={"name": "compress_final_out", "compress_ratio": 4})
    out = transform_compress_final_out_b2t(g)
    assert torch.equal(out, target)
    print("test_compress_final_out_b2t_indexer_skipped PASS")


def test_layer_input_b2t():
    """mg SP-sharded T: each rank gets (T_padded/8 = 272, 1, hc=4, h=4096); sg full (T_actual=2126, hc, h)."""
    T_per_rank, hc, h, T_actual = 272, 4, 4096, 2126
    parts = [torch.randn(T_per_rank, 1, hc, h, dtype=torch.bfloat16) for _ in range(8)]
    target = torch.zeros(T_actual, hc, h, dtype=torch.bfloat16)
    g = FakeGraftInput(received_list=parts, target=target,
                       tags={"name": "layer_input"})
    out = transform_layer_input_b2t(g)
    assert out.shape == target.shape
    expected = torch.cat(parts, dim=0).squeeze(1)[:T_actual]
    assert torch.allclose(out.float(), expected.float())
    print("test_layer_input_b2t PASS")


def test_compress_input_b2t():
    """mg full-T per rank (1, 2176, 4096) replicated; sg target (2126, 4096)."""
    T_padded, h, T_actual = 2176, 4096, 2126
    parts = [torch.randn(1, T_padded, h, dtype=torch.bfloat16)] * 8  # all ranks identical content
    target = torch.zeros(T_actual, h, dtype=torch.bfloat16)
    g = FakeGraftInput(received_list=parts, target=target,
                       tags={"name": "compress_input", "compress_ratio": 4})
    out = transform_compress_input_b2t(g)
    assert out.shape == target.shape
    print("test_compress_input_b2t PASS")


# --- t2b transforms (mg side, role=baseline, received_list = sg sender contribs) ---

def test_compress_final_out_t2b():
    """sg per-query (T_actual=2126, head_dim=512) — c4.cuh writes only at block-end positions
    (t=3, 7, ...); in-between positions have garbage. mg target (1, T_compressed=544, 512).
    Block-end mapping: mg[k] = sg[(k+1)*ratio - 1].
    """
    T_compressed, T_actual, head_dim, ratio = 544, 2126, 512, 4
    sg_full = torch.zeros(T_actual, head_dim, dtype=torch.bfloat16)
    # block-end positions: t=3, 7, 11, ..., 2127 (capped at < 2126)
    for k in range(T_compressed):
        t_idx = (k + 1) * ratio - 1
        if t_idx < T_actual:
            sg_full[t_idx] = torch.full((head_dim,), float(k), dtype=torch.bfloat16)
    # in-between positions are garbage (we put 1e30 to simulate real c4.cuh behavior)
    for t in range(T_actual):
        if (t + 1) % ratio != 0:
            sg_full[t] = torch.full((head_dim,), 1e30, dtype=torch.bfloat16)
    parts = [sg_full] + [torch.zeros(0, head_dim, dtype=torch.bfloat16)] * 7  # rank 0 active, 1-7 idle
    target = torch.zeros(1, T_compressed, head_dim, dtype=torch.bfloat16)
    g = FakeGraftInput(received_list=parts, target=target,
                       tags={"name": "compress_final_out", "compress_ratio": 4})
    out = transform_compress_final_out_t2b(g)
    assert out.shape == target.shape
    for k in [0, 1, 100, 530]:
        t_idx = (k + 1) * ratio - 1
        if t_idx < T_actual:
            assert torch.allclose(out[0, k].float(), torch.full((head_dim,), float(k), dtype=torch.bfloat16).float()), \
                f"mg block {k} != block-end value (got garbage instead)"
    # block 532..543 should be unchanged (target's original = 0)
    for k in [532, 540, 543]:
        assert torch.equal(out[0, k], torch.zeros(head_dim, dtype=torch.bfloat16))
    print("test_compress_final_out_t2b PASS")


def test_compress_final_out_t2b_skip_empty_idle():
    """Among 8 sg sender contributions, only one is non-empty (DP rank 0 active);
    empty ones must be skipped."""
    T_compressed, T_actual, head_dim, ratio = 544, 2126, 512, 4
    sg_active = torch.randn(T_actual, head_dim, dtype=torch.bfloat16)
    sg_idle = [torch.zeros(0, head_dim, dtype=torch.bfloat16) for _ in range(7)]
    parts = [sg_active] + sg_idle
    target = torch.zeros(1, T_compressed, head_dim, dtype=torch.bfloat16)
    g = FakeGraftInput(received_list=parts, target=target,
                       tags={"name": "compress_final_out", "compress_ratio": 4})
    out = transform_compress_final_out_t2b(g)
    assert torch.allclose(out[0, 0].float(), sg_active[3].to(out.dtype).float())
    print("test_compress_final_out_t2b_skip_empty_idle PASS")


def test_compress_final_out_t2b_indexer_skipped():
    """head_dim=128 (indexer compressor) — sg sender contributes wrong shape (last_dim=128); transform returns target unchanged."""
    T_compressed, T_actual, head_dim_indexer = 544, 2126, 128
    sg_indexer = torch.randn(T_actual, head_dim_indexer, dtype=torch.bfloat16)
    parts = [sg_indexer] * 8
    target = torch.full((1, T_compressed, head_dim_indexer), 7.0, dtype=torch.bfloat16)
    g = FakeGraftInput(received_list=parts, target=target,
                       tags={"name": "compress_final_out", "compress_ratio": 4})
    out = transform_compress_final_out_t2b(g)
    assert torch.equal(out, target)
    print("test_compress_final_out_t2b_indexer_skipped PASS")


# --- dispatcher tests (set role explicitly via env) ---

def test_dispatcher_target_role_routes_compress_final_out_b2t():
    os.environ["DUMPER_GRAFTER_ROLE"] = "target"
    target = torch.zeros(2126, 512, dtype=torch.bfloat16)
    mg_full = torch.randn(1, 544, 512, dtype=torch.bfloat16)
    g = FakeGraftInput(received_list=[mg_full] * 8, target=target,
                       tags={"name": "compress_final_out", "compress_ratio": 4})
    out = transform(g)
    assert out.shape == target.shape
    assert torch.allclose(out[0].float(), mg_full[0, 0].to(out.dtype).float()), "b2t fn not invoked"
    print("test_dispatcher_target_role_routes_compress_final_out_b2t PASS")


def test_dispatcher_baseline_role_routes_compress_final_out_t2b():
    os.environ["DUMPER_GRAFTER_ROLE"] = "baseline"
    target = torch.zeros(1, 544, 512, dtype=torch.bfloat16)
    sg_full = torch.randn(2126, 512, dtype=torch.bfloat16)
    g = FakeGraftInput(received_list=[sg_full] + [torch.zeros(0, 512, dtype=torch.bfloat16)] * 7,
                       target=target,
                       tags={"name": "compress_final_out", "compress_ratio": 4})
    out = transform(g)
    assert out.shape == target.shape
    assert torch.allclose(out[0, 0].float(), sg_full[3].to(out.dtype).float()), "t2b fn not invoked"
    print("test_dispatcher_baseline_role_routes_compress_final_out_t2b PASS")


# --- E3 attention block transforms ---

def test_attn_q_b2t():
    """mg per-rank (1, T_padded, n_h_local=8, head_dim=512) TP-sharded → sg full (T_actual=2126, 64, 512)."""
    T_padded, T_actual, n_h_local, head_dim = 2176, 2126, 8, 512
    parts = [torch.randn(1, T_padded, n_h_local, head_dim, dtype=torch.bfloat16) for _ in range(8)]
    target = torch.zeros(T_actual, n_h_local * 8, head_dim, dtype=torch.bfloat16)
    g = FakeGraftInput(received_list=parts, target=target, tags={"name": "attn_q"})
    out = transform_attn_q_b2t(g)
    assert out.shape == target.shape
    expected = torch.cat(parts, dim=2).squeeze(0)[:T_actual]
    assert torch.allclose(out.float(), expected.float())
    print("test_attn_q_b2t PASS")


def test_attn_v_b2t_replicated():
    """mg per-rank (1, T_padded, head_dim=512) replicated across ranks → sg (T_actual, 512) on rank 0.
    All 8 mg ranks have IDENTICAL content; we take the first non-empty and trim T."""
    T_padded, T_actual, head_dim = 2176, 2126, 512
    shared = torch.randn(1, T_padded, head_dim, dtype=torch.bfloat16)
    parts = [shared.clone() for _ in range(8)]
    target = torch.zeros(T_actual, head_dim, dtype=torch.bfloat16)
    g = FakeGraftInput(received_list=parts, target=target, tags={"name": "attn_v"})
    out = transform_attn_v_b2t(g)
    assert out.shape == target.shape
    expected = shared.squeeze(0)[:T_actual]
    assert torch.allclose(out.float(), expected.float())
    print("test_attn_v_b2t_replicated PASS")


def test_attn_output_t2b_rank0():
    """sg active (T_actual=2126, n_heads_total=64, head_dim=512) → mg rank 0 (1, T_padded=2176, n_h_local=8, head_dim=512).
    Without dist init, _my_baseline_rank returns 0; rank 0 takes heads [0:8]. T_actual..T_padded stays as target's original."""
    T_padded, T_actual, n_h_total, n_h_local, head_dim = 2176, 2126, 64, 8, 512
    sg_full = torch.randn(T_actual, n_h_total, head_dim, dtype=torch.bfloat16)
    sg_idle = [torch.zeros(0, n_h_total, head_dim, dtype=torch.bfloat16) for _ in range(7)]
    parts = [sg_full] + sg_idle
    target = torch.full((1, T_padded, n_h_local, head_dim), 7.0, dtype=torch.bfloat16)
    g = FakeGraftInput(received_list=parts, target=target, tags={"name": "attn_output"})
    out = transform_attn_output_t2b(g)
    assert out.shape == target.shape
    # rank 0 → heads [0:8]
    expected_slice = sg_full[:, 0:n_h_local, :]
    assert torch.allclose(out[0, :T_actual].float(), expected_slice.float())
    # padding region stays at target's original value (7.0)
    assert torch.equal(out[0, T_actual:], torch.full((T_padded - T_actual, n_h_local, head_dim), 7.0, dtype=torch.bfloat16))
    print("test_attn_output_t2b_rank0 PASS")


def test_attn_output_t2b_skip_empty():
    """If only sg active rank has data, transform must skip the 7 idle empties and use the active."""
    T_padded, T_actual, n_h_total, n_h_local, head_dim = 2176, 2126, 64, 8, 512
    sg_active = torch.randn(T_actual, n_h_total, head_dim, dtype=torch.bfloat16)
    parts = [sg_active] + [torch.zeros(0, n_h_total, head_dim, dtype=torch.bfloat16) for _ in range(7)]
    target = torch.zeros(1, T_padded, n_h_local, head_dim, dtype=torch.bfloat16)
    g = FakeGraftInput(received_list=parts, target=target, tags={"name": "attn_output"})
    out = transform_attn_output_t2b(g)
    assert torch.allclose(out[0, 0, 0].float(), sg_active[0, 0].float())
    print("test_attn_output_t2b_skip_empty PASS")


# --- E4 MLP/MoE block transforms ---

def test_pre_mlp_layernorm_output_b2t():
    """mg SP-sharded (T_sp=272, 1, h=4096) per rank → sg full (T_actual=2126, h)."""
    T_per_rank, h, T_actual = 272, 4096, 2126
    parts = [torch.randn(T_per_rank, 1, h, dtype=torch.bfloat16) for _ in range(8)]
    target = torch.zeros(T_actual, h, dtype=torch.bfloat16)
    g = FakeGraftInput(received_list=parts, target=target,
                       tags={"name": "pre_mlp_layernorm_output"})
    out = transform_pre_mlp_layernorm_output_b2t(g)
    assert out.shape == target.shape
    expected = torch.cat(parts, dim=0).squeeze(1)[:T_actual]
    assert torch.allclose(out.float(), expected.float())
    print("test_pre_mlp_layernorm_output_b2t PASS")


def test_mlp_output_t2b_rank0():
    """sg active (T_actual=2126, h=4096) → mg rank 0 (T_sp=272, 1, h).
    Without dist init, _my_baseline_rank returns 0; rank 0 takes T slice [0:272]."""
    T_per_rank, h, T_actual = 272, 4096, 2126
    sg_full = torch.randn(T_actual, h, dtype=torch.bfloat16)
    parts = [sg_full] + [torch.zeros(0, h, dtype=torch.bfloat16) for _ in range(7)]
    target = torch.full((T_per_rank, 1, h), 7.0, dtype=torch.bfloat16)
    g = FakeGraftInput(received_list=parts, target=target,
                       tags={"name": "mlp_output"})
    out = transform_mlp_output_t2b(g)
    assert out.shape == target.shape
    # rank 0 → T slice [0:T_per_rank=272]
    assert torch.allclose(out[:T_per_rank, 0, :].float(), sg_full[:T_per_rank].float())
    print("test_mlp_output_t2b_rank0 PASS")


def test_mlp_output_t2b_padding_stays():
    """sg has T_actual=2126 rows, mg rank 7 expects [7*272 : 7*272+272] = [1904:2176].
    First 222 rows (1904..2125) come from sg; last 50 (2126..2175) stay at target's value."""
    # Simulate by constructing target on rank where slice partially exceeds sg's data.
    # Note: we can't easily set rank=7 in unit test (no dist), so verify the slicing logic
    # via target T_per_rank > sg_full.shape[0] - sp_idx*T_per_rank case.
    # Use rank=0 with custom target/sg_full to mimic the boundary.
    T_per_rank, h = 272, 4096
    # Make sg shorter than expected slice → simulating boundary at rank 7
    sg_full = torch.randn(150, h, dtype=torch.bfloat16)  # only 150 rows; rank 0 wants 272
    parts = [sg_full] + [torch.zeros(0, h, dtype=torch.bfloat16) for _ in range(7)]
    target = torch.full((T_per_rank, 1, h), 7.0, dtype=torch.bfloat16)
    g = FakeGraftInput(received_list=parts, target=target, tags={"name": "mlp_output"})
    out = transform_mlp_output_t2b(g)
    # First 150 rows from sg, rest stay at 7.0
    assert torch.allclose(out[:150, 0, :].float(), sg_full.float())
    assert torch.equal(out[150:, 0, :], torch.full((T_per_rank - 150, h), 7.0, dtype=torch.bfloat16))
    print("test_mlp_output_t2b_padding_stays PASS")


def test_input_layernorm_b2t():
    """mg SP-sharded T: each rank gets (T_sp=272, 1, h=4096); sg target (T_actual=2126, h)."""
    T_per_rank, h, T_actual = 272, 4096, 2126
    parts = [torch.randn(T_per_rank, 1, h, dtype=torch.bfloat16) for _ in range(8)]
    target = torch.zeros(T_actual, h, dtype=torch.bfloat16)
    g = FakeGraftInput(received_list=parts, target=target,
                       tags={"name": "input_layernorm"})
    out = transform_input_layernorm_b2t(g)
    assert out.shape == target.shape
    expected = torch.cat(parts, dim=0).squeeze(1)[:T_actual]
    assert torch.allclose(out.float(), expected.float())
    print("test_input_layernorm_b2t PASS")


def test_attn_v_t2b():
    """sg active (T_actual=2126, head_dim=512) → mg per-rank (1, T_padded=2176, head_dim=512) REPLICATED.
    All ranks place sg values at [0, :T_actual, :]; padding rows [T_actual:T_padded] stay at mg's local value."""
    T_padded, T_actual, head_dim = 2176, 2126, 512
    sg_active = torch.randn(T_actual, head_dim, dtype=torch.bfloat16)
    parts = [sg_active] + [torch.zeros(0, head_dim, dtype=torch.bfloat16) for _ in range(7)]
    target = torch.full((1, T_padded, head_dim), 7.0, dtype=torch.bfloat16)
    g = FakeGraftInput(received_list=parts, target=target, tags={"name": "attn_v"})
    out = transform_attn_v_t2b(g)
    assert out.shape == target.shape
    assert torch.allclose(out[0, :T_actual].float(), sg_active.float())
    assert torch.equal(out[0, T_actual:], torch.full((T_padded - T_actual, head_dim), 7.0, dtype=torch.bfloat16))
    print("test_attn_v_t2b PASS")


def test_attn_q_t2b_rank0():
    """sg active (T_actual=2126, n_heads_total=64, head_dim=512) → mg rank 0 (1, T_padded=2176, n_h_local=8, head_dim=512).
    Without dist init, _my_baseline_rank returns 0; rank 0 takes heads [0:8]."""
    T_padded, T_actual, n_h_total, n_h_local, head_dim = 2176, 2126, 64, 8, 512
    sg_full = torch.randn(T_actual, n_h_total, head_dim, dtype=torch.bfloat16)
    sg_idle = [torch.zeros(0, n_h_total, head_dim, dtype=torch.bfloat16) for _ in range(7)]
    parts = [sg_full] + sg_idle
    target = torch.full((1, T_padded, n_h_local, head_dim), 7.0, dtype=torch.bfloat16)
    g = FakeGraftInput(received_list=parts, target=target, tags={"name": "attn_q"})
    out = transform_attn_q_t2b(g)
    assert out.shape == target.shape
    expected_slice = sg_full[:, 0:n_h_local, :]
    assert torch.allclose(out[0, :T_actual].float(), expected_slice.float())
    assert torch.equal(out[0, T_actual:], torch.full((T_padded - T_actual, n_h_local, head_dim), 7.0, dtype=torch.bfloat16))
    print("test_attn_q_t2b_rank0 PASS")


def test_lm_head_logits_t2b_rank0():
    """sg active (T_actual=2126, V_sg=129280) → mg (b=1, T_padded=2176, V_mg=130048).
    Mg's first 129280 vocab cols replaced with sg's; padding cols [129280:130048]
    masked to large-negative so log_softmax denom matches sg's V_sg-only normalization.
    Rows past T_actual stay at target's original value."""
    T_padded, T_actual, V_sg, V_mg = 2176, 2126, 129280, 130048
    sg_full = torch.randn(T_actual, V_sg, dtype=torch.bfloat16)
    parts = [sg_full] + [torch.zeros(0, V_sg, dtype=torch.bfloat16) for _ in range(7)]
    target = torch.full((1, T_padded, V_mg), 7.0, dtype=torch.bfloat16)
    g = FakeGraftInput(received_list=parts, target=target,
                       tags={"name": "lm_head_logits"})
    out = transform_lm_head_logits_t2b(g)
    assert out.shape == target.shape
    # First V_sg cols on first T_actual rows = sg
    assert torch.allclose(out[0, :T_actual, :V_sg].float(), sg_full.float())
    # Padding cols = large-negative
    pad = out[0, :T_actual, V_sg:].float()
    assert (pad < -1e30).all()
    # Rows past T_actual stay at 7.0
    assert torch.equal(out[0, T_actual:], torch.full((T_padded - T_actual, V_mg), 7.0, dtype=torch.bfloat16))
    print("test_lm_head_logits_t2b_rank0 PASS")


def test_post_norm_hidden_t2b_rank0():
    """sg active (T_actual=2126, h=4096) → mg rank 0 (T_sp=272, 1, h).
    Same shape pattern as mlp_output_t2b: post-final_layernorm hidden states.
    Fires once per forward (outside the layer loop)."""
    T_per_rank, h, T_actual = 272, 4096, 2126
    sg_full = torch.randn(T_actual, h, dtype=torch.bfloat16)
    parts = [sg_full] + [torch.zeros(0, h, dtype=torch.bfloat16) for _ in range(7)]
    target = torch.full((T_per_rank, 1, h), 7.0, dtype=torch.bfloat16)
    g = FakeGraftInput(received_list=parts, target=target,
                       tags={"name": "post_norm_hidden"})
    out = transform_post_norm_hidden_t2b(g)
    assert out.shape == target.shape
    assert torch.allclose(out[:T_per_rank, 0, :].float(), sg_full[:T_per_rank].float())
    print("test_post_norm_hidden_t2b_rank0 PASS")


def test_layer_input_t2b_rank0():
    """sg (T_actual=2126, hc=4, h=4096) → mg rank 0 (T_sp=272, 1, hc, h).
    Without dist init, _my_baseline_rank returns 0; rank 0 takes T slice [0:272]."""
    T_per_rank, hc, h, T_actual = 272, 4, 4096, 2126
    sg_full = torch.randn(T_actual, hc, h, dtype=torch.bfloat16)
    parts = [sg_full] + [torch.zeros(0, hc, h, dtype=torch.bfloat16) for _ in range(7)]
    target = torch.full((T_per_rank, 1, hc, h), 7.0, dtype=torch.bfloat16)
    g = FakeGraftInput(received_list=parts, target=target, tags={"name": "layer_input"})
    out = transform_layer_input_t2b(g)
    assert out.shape == target.shape
    assert torch.allclose(out[:T_per_rank, 0].float(), sg_full[:T_per_rank].float())
    print("test_layer_input_t2b_rank0 PASS")


def test_dispatcher_unknown_name():
    os.environ["DUMPER_GRAFTER_ROLE"] = "target"
    target = torch.zeros(1)
    g = FakeGraftInput(received_list=[torch.zeros(1)] * 8, target=target,
                       tags={"name": "totally_made_up_op"})
    try:
        transform(g)
    except RuntimeError as e:
        assert "no transform registered" in str(e)
        print("test_dispatcher_unknown_name PASS")
        return
    raise AssertionError("expected RuntimeError")


if __name__ == "__main__":
    # b2t (sg side, role=target)
    test_attn_output_b2t()
    test_attn_output_b2t_empty_target()
    test_compress_final_out_b2t()
    test_compress_final_out_b2t_indexer_skipped()
    test_layer_input_b2t()
    test_compress_input_b2t()
    test_attn_q_b2t()
    test_attn_v_b2t_replicated()
    # t2b (mg side, role=baseline)
    test_compress_final_out_t2b()
    test_compress_final_out_t2b_skip_empty_idle()
    test_compress_final_out_t2b_indexer_skipped()
    test_attn_output_t2b_rank0()
    test_attn_output_t2b_skip_empty()
    test_pre_mlp_layernorm_output_b2t()
    test_mlp_output_t2b_rank0()
    test_mlp_output_t2b_padding_stays()
    test_layer_input_t2b_rank0()
    test_post_norm_hidden_t2b_rank0()
    test_lm_head_logits_t2b_rank0()
    test_input_layernorm_b2t()
    test_attn_q_t2b_rank0()
    test_attn_v_t2b()
    # dispatcher
    test_dispatcher_target_role_routes_compress_final_out_b2t()
    test_dispatcher_baseline_role_routes_compress_final_out_t2b()
    test_dispatcher_unknown_name()
    print("\nALL TESTS PASS")
