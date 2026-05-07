"""Compare layer 0 across all canonical tensors between sglang and megatron.

Strategy:
- SGLang: rank 0 has full sequence (DP attention), per chunk. Concat chunks.
- Megatron: 8 SP ranks each have 1/8 of sequence. Concat all.
- Slice to common length for comparison.
"""
from pathlib import Path
from collections import defaultdict

import torch


SG = Path("/storage/yueming/dumper-out/v4-base-worst-sglang/dump_20260504_110000_681864")
MG = Path("/storage/yueming/dumper-out/v4-base-worst-megatron/standalone")


def parse(fn: str) -> dict:
    parts: dict[str, str] = {}
    for seg in fn.split("___"):
        if "=" in seg:
            k, v = seg.split("=", 1)
            v = v[: -len(".pt")] if v.endswith(".pt") else v
            parts[k] = v
    return parts


def load(p: Path) -> torch.Tensor:
    return torch.load(p, weights_only=False, map_location="cpu")["value"]


def get_sg_layer0_full(name: str, compress_ratio: str = "_none_") -> torch.Tensor | None:
    """Sglang: pick rank 0's layer 0 (= smallest dump_index per step), concat across steps."""
    by_step: dict[int, tuple[int, Path]] = {}
    for f in SG.iterdir():
        p = parse(f.name)
        if p.get("name") != name:
            continue
        if int(p.get("rank", "0")) != 0:
            continue
        if p.get("compress_ratio", "_none_") != compress_ratio:
            continue
        di = int(p["dump_index"])
        step = int(p["step"])
        if step not in by_step or di < by_step[step][0]:
            by_step[step] = (di, f)
    if not by_step:
        return None
    chunks = [load(f) for _, f in sorted(by_step.values())]
    nonempty = [c for c in chunks if c.shape[0] > 0]
    if not nonempty:
        return None
    return torch.cat(nonempty, dim=0)


def get_mg_layer0_full(name: str, compress_ratio: str = "_none_") -> torch.Tensor | None:
    """Megatron: layer_id=0 dumps across 8 ranks, concat along seq (dim 0)."""
    by_rank: dict[int, torch.Tensor] = {}
    for f in MG.iterdir():
        p = parse(f.name)
        if p.get("name") != name:
            continue
        if p.get("layer_id", "_none_") != "0":
            continue
        if p.get("compress_ratio", "_none_") != compress_ratio:
            continue
        rank = int(p["rank"])
        by_rank[rank] = load(f)
    if not by_rank:
        return None
    parts = [squeeze_size1(by_rank[r]) for r in sorted(by_rank.keys())]
    if all(p.shape == parts[0].shape for p in parts) and parts[0].dim() == 0:
        return torch.stack(parts)
    try:
        return torch.cat(parts, dim=0)
    except Exception:
        # try stacking on rank dim if shapes differ on dim 0
        return None


def squeeze_size1(t: torch.Tensor) -> torch.Tensor:
    while True:
        sz1 = [i for i, s in enumerate(t.shape) if s == 1]
        if not sz1:
            return t
        t = t.squeeze(sz1[0])


def compare(name: str, compress_ratio: str = "_none_"):
    try:
        sg = get_sg_layer0_full(name, compress_ratio)
    except Exception as e:
        sg = None
    try:
        mg = get_mg_layer0_full(name, compress_ratio)
    except Exception as e:
        mg = None
    suffix = f" (cr={compress_ratio})" if compress_ratio != "_none_" else ""
    if sg is None and mg is None:
        return f"{name}{suffix}: not found in either"
    if sg is None:
        return f"{name}{suffix}: only in megatron, mg shape={tuple(mg.shape)}"
    if mg is None:
        return f"{name}{suffix}: only in sglang, sg shape={tuple(sg.shape)}"

    sg = squeeze_size1(sg.float())
    mg = squeeze_size1(mg.float())

    if sg.dim() != mg.dim():
        return f"{name}{suffix}: dim mismatch sg={sg.dim()}D{tuple(sg.shape)} mg={mg.dim()}D{tuple(mg.shape)}"

    n_compare = min(sg.shape[0], mg.shape[0])

    # Both should agree on remaining dims
    if sg.shape[1:] != mg.shape[1:]:
        return f"{name}{suffix}: tail-shape mismatch sg={tuple(sg.shape)} mg={tuple(mg.shape)}"

    sg_s = sg[:n_compare]
    mg_s = mg[:n_compare]
    diff = (sg_s - mg_s).abs()
    rel = diff / (mg_s.abs() + 1e-9)
    info = (
        f"sg{tuple(sg.shape)} mg{tuple(mg.shape)} cmp[:{n_compare}] "
        f"abs_max={diff.max().item():.3e} abs_mean={diff.mean().item():.3e} "
        f"rel_max={rel.max().item():.3e} rel_mean={rel.mean().item():.3e}"
    )
    return f"{name}{suffix}: {info}"


def diagnose_alignment():
    """Identify which mg ranks correspond to which sglang positions."""
    name = "layer_input"
    sg = squeeze_size1(get_sg_layer0_full(name).float())
    print(f"sglang layer_input shape: {tuple(sg.shape)}")

    by_rank = {}
    for f in MG.iterdir():
        p = parse(f.name)
        if p.get("name") != name or p.get("layer_id", "_none_") != "0":
            continue
        rank = int(p["rank"])
        by_rank[rank] = squeeze_size1(load(f).float())

    # For each mg rank, search for matching sglang positions
    for rank in sorted(by_rank.keys()):
        mg_t = by_rank[rank]
        n = mg_t.shape[0]
        # Check if mg_t matches sg[rank*n : (rank+1)*n] (contiguous SP)
        s_lo = rank * n
        s_hi = s_lo + n
        if s_hi <= sg.shape[0]:
            sg_chunk = sg[s_lo:s_hi]
            diff_contig = (sg_chunk - mg_t).abs().max().item()
        else:
            diff_contig = float("nan")

        # Try reversed (zigzag pair: 0+last, 1+second-last, etc.)
        # Zigzag: rank r has positions [r*chunk, (r+1)*chunk) AND [(2*sp-1-r)*chunk, (2*sp-r)*chunk)
        # For SP=8, chunk=144. rank 0 has [0:144] + [16*144 - 144 : 16*144] = [0:144] + [2160:2304]
        # mg_t is shape (288, ...) which is 2 chunks of 144. So check
        chunk = n // 2
        first_lo = rank * chunk
        first_hi = first_lo + chunk
        second_lo = (2 * 8 - 1 - rank) * chunk
        second_hi = second_lo + chunk
        if first_hi <= sg.shape[0] and second_hi <= sg.shape[0]:
            sg_first = sg[first_lo:first_hi]
            sg_second = sg[second_lo:second_hi]
            mg_first = mg_t[:chunk]
            mg_second = mg_t[chunk:]
            diff_zigzag1 = (sg_first - mg_first).abs().max().item()
            diff_zigzag2 = (sg_second - mg_second).abs().max().item()
        else:
            diff_zigzag1 = diff_zigzag2 = float("nan")

        print(
            f"  mg rank {rank} (n={n}): contig_diff_max={diff_contig:.3e}  "
            f"zigzag1_diff_max={diff_zigzag1:.3e} zigzag2_diff_max={diff_zigzag2:.3e}"
        )


def main():
    diagnose_alignment()
    print()
    names_canonical = [
        "layer_input",
        "attn_output",
        "pre_mlp_residual",
        "pre_mlp_layernorm_output",
        "mlp_output",
        "moe_router_logits",
        "moe_topk_ids",
    ]
    names_v4_no_cr = [
        "wq_a_out",
        "q_lora_after_norm",
        "wq_b_out",
        "q_heads_after_norm",
        "attn_q",
        "wkv_out",
        "kv_after_norm",
        "attn_v",
        "mqa_wo_a_out",
        "mqa_wo_b_out",
    ]
    names_v4_with_cr = [
        ("compressor_out", "0"),
        ("compressor_out", "4"),
        ("compressor_out", "128"),
        ("compressor_kv_score", "4"),
        ("compressor_kv_score", "128"),
        ("compress_forward_out", "4"),
        ("compress_forward_out", "128"),
        ("compress_fused_norm_rope_out", "4"),
        ("compress_fused_norm_rope_out", "128"),
        ("compress_final_out", "4"),
        ("compress_final_out", "128"),
        ("attn_q_in", "4"),
        ("attn_q_in", "128"),
        ("attn_kv_in", "4"),
        ("attn_kv_in", "128"),
        ("attn_topk_idxs", "4"),
        ("attn_topk_idxs", "128"),
        ("indexer_compute_q_out", "4"),
        ("indexer_q_fp8", "4"),
        ("indexer_kv_cache_view", "4"),
        ("indexer_compute_weights_raw", "4"),
        ("indexer_weights", "4"),
        ("indexer_logits", "4"),
    ]
    print("=== canonical names ===")
    for n in names_canonical:
        print(compare(n))
    print("\n=== V4 names (no compress_ratio) ===")
    for n in names_v4_no_cr:
        print(compare(n))
    print("\n=== V4 names (with compress_ratio) ===")
    for n, cr in names_v4_with_cr:
        print(compare(n, cr))


if __name__ == "__main__":
    main()
