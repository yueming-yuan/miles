"""V2 visualization on V4-Flash sglang vs megatron dumps.

Builds per-token diff tensors directly from dumps (bypasses comparator's summary).

Outputs:
1. heatmap_43layers.png  — heatmap, x=token_id (0..T), y=layer_id (0..42), color=|diff| at layer_input
2. layer_pair_op_chain_L{a}_vs_L{b}.png — line chart, x=operator (in exec order), y=|diff| at the
   single token with max layer_input diff (chosen ONCE across plots)

Caches the aligned diff tensor as .npz for re-runs.
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


# Operator order in V4 forward (per layer, sequential)
OP_ORDER = [
    "layer_input",
    "wq_a_out",
    "q_lora_after_norm",
    "wq_b_out",
    "q_heads_after_norm",
    "attn_q",
    "wkv_out",
    "kv_after_norm",
    "attn_v",
    "compressor_kv_score",
    "mqa_wo_a_out",
    "mqa_wo_b_out",
    "attn_output",
    "pre_mlp_layernorm_output",
    "mlp_output",
]


def parse_meta_from_filename(fn: str) -> dict:
    parts: dict[str, str] = {}
    for seg in fn.split("___"):
        if "=" in seg:
            k, v = seg.split("=", 1)
            v = v[: -3] if v.endswith(".pt") else v
            parts[k] = v
    return parts


def gather_sg_per_layer(sg_dir: Path, name: str) -> dict[int, list[Path]]:
    """SG: layer_id absent from meta. Use ordering: i-th attn_q within (rank=0, step=s)
    corresponds to layer i. Concat across steps gives layer i's full tensor."""
    by_step: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    for f in sg_dir.iterdir():
        m = parse_meta_from_filename(f.name)
        if m.get("name") != name or int(m.get("rank", -1)) != 0:
            continue
        by_step[int(m["step"])].append((int(m["dump_index"]), f))
    by_layer_step: dict[int, dict[int, Path]] = defaultdict(dict)
    n_layers = 0
    for step, lst in by_step.items():
        lst.sort()
        n_layers = max(n_layers, len(lst))
        for layer_idx, (_, f) in enumerate(lst):
            by_layer_step[layer_idx][step] = f
    return {layer_id: [by_layer_step[layer_id][s] for s in sorted(by_layer_step[layer_id])]
            for layer_id in range(n_layers)}


def gather_mg_per_layer(mg_dir: Path, name: str) -> dict[int, list[Path]]:
    """MG: layer_id is in filename. Per layer, gather all 8 ranks' files."""
    by_layer: dict[int, list[Path]] = defaultdict(list)
    for f in mg_dir.iterdir():
        m = parse_meta_from_filename(f.name)
        if m.get("name") != name:
            continue
        if "layer_id" not in m:
            continue
        if "compress_ratio" in m:
            # canonical name only — skip cr-specific variants in this pass
            continue
        by_layer[int(m["layer_id"])].append(f)
    return by_layer


def load_tensor(p: Path) -> torch.Tensor:
    t = torch.load(p, weights_only=False, map_location="cpu")["value"]
    return t.detach() if t.requires_grad else t


def align_sg_steps(files: list[Path]) -> torch.Tensor:
    """Concat sg's per-step chunks along dim 0 (token)."""
    chunks = [load_tensor(f).float() for f in files]
    return torch.cat([c for c in chunks if c.shape[0] > 0], dim=0)


def align_mg(files: list[Path], expected_t: int) -> torch.Tensor:
    """Detect mg layout per shape:
    - SP-sharded T (each rank has T/sp tokens): cat all 8 ranks along dim 0
    - Replicated (each rank has full T): take rank 0
    - TP-sharded heads (per-rank heads_per_rank, full T): cat ranks along heads dim
      (use rank 0 + truncate sg to match per-rank heads since we don't know which dim is heads)
    """
    by_rank = {}
    for f in files:
        m = parse_meta_from_filename(f.name)
        by_rank[int(m["rank"])] = load_tensor(f).float()
    parts = [by_rank[r] for r in sorted(by_rank.keys())]
    n_ranks = len(parts)
    p0 = parts[0]

    # Case 1: SP-sharded T → first dim is short, cat → ~expected_t
    if p0.shape[0] < expected_t and p0.shape[0] * n_ranks >= expected_t:
        return torch.cat(parts, dim=0)

    # Case 2: Replicated or TP-sharded — take rank 0
    t = p0
    # Squeeze leading singletons that aren't the token dim
    while t.dim() >= 2 and t.shape[0] == 1 and t.shape[1] >= 8:
        t = t.squeeze(0)
    return t


def compute_per_token_diff(sg: torch.Tensor, mg: torch.Tensor) -> np.ndarray | None:
    """|sg - mg| reduced over all non-token dims → vector of length T.
    Truncates each dim to the minimum of (sg, mg) so TP-sharded mg can still compare
    against the corresponding partial-heads slice of sg.
    """
    sg = sg.squeeze()
    mg = mg.squeeze()
    if sg.dim() == 0 or mg.dim() == 0:
        return None
    if sg.dim() != mg.dim():
        return None
    # Truncate each dim to common min
    slc = tuple(slice(0, min(sg.shape[d], mg.shape[d])) for d in range(sg.dim()))
    sg = sg[slc]
    mg = mg[slc]
    if sg.shape != mg.shape:
        return None
    if sg.numel() == 0:
        return None
    diff = (sg - mg).abs()
    if diff.dim() > 1:
        diff = diff.flatten(start_dim=1).max(dim=1).values
    return diff.detach().cpu().numpy()


def build_per_token_matrix(sg_dir: Path, mg_dir: Path, name: str, n_layers: int = 43) -> np.ndarray:
    """Returns array of shape (n_layers, T) with per-token |diff| at each layer.
    NaN where alignment failed."""
    sg_by_layer = gather_sg_per_layer(sg_dir, name)
    mg_by_layer = gather_mg_per_layer(mg_dir, name)
    rows: list[np.ndarray | None] = []
    max_T = 0
    for li in range(n_layers):
        if li not in sg_by_layer or li not in mg_by_layer:
            rows.append(None)
            continue
        try:
            sg_t = align_sg_steps(sg_by_layer[li])
            mg_t = align_mg(mg_by_layer[li], sg_t.shape[0])
            diff = compute_per_token_diff(sg_t, mg_t)
        except Exception as e:
            print(f"  layer {li} {name}: align error: {e}")
            diff = None
        rows.append(diff)
        if diff is not None:
            max_T = max(max_T, diff.shape[0])
    matrix = np.full((n_layers, max_T), np.nan, dtype=np.float32)
    for li, d in enumerate(rows):
        if d is None: continue
        matrix[li, :d.shape[0]] = d
    return matrix


def viz_heatmap(matrix: np.ndarray, out_path: str, title: str = "layer_input |sg-mg|"):
    fig, ax = plt.subplots(figsize=(20, 9))
    masked = np.ma.masked_invalid(matrix)
    # Use log-scale color but allow zero (clip lower)
    eps = 1e-6
    im = ax.imshow(np.log10(masked + eps), aspect="auto", cmap="viridis", origin="lower",
                   interpolation="nearest")
    ax.set_xlabel("token_id", fontsize=11)
    ax.set_ylabel("layer_id", fontsize=11)
    ax.set_title(f"Per-token, per-layer log10(|sg-mg|) — {title}", fontsize=12)
    cbar = plt.colorbar(im, ax=ax, label="log10(|diff|)")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved {out_path}")


def viz_layer_pair_at_token(per_op_diffs: dict[str, np.ndarray], token_idx: int,
                             la: int, lb: int, out_path: str):
    """Line chart, x=op (in OP_ORDER), y=|diff| at chosen token, two lines: layer la, lb."""
    ops = [op for op in OP_ORDER if op in per_op_diffs and per_op_diffs[op] is not None]
    a_vals = []
    b_vals = []
    for op in ops:
        m = per_op_diffs[op]
        if la < m.shape[0] and token_idx < m.shape[1]:
            a_vals.append(m[la, token_idx])
        else:
            a_vals.append(np.nan)
        if lb < m.shape[0] and token_idx < m.shape[1]:
            b_vals.append(m[lb, token_idx])
        else:
            b_vals.append(np.nan)

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(ops))
    ax.plot(x, a_vals, marker="o", linewidth=2, label=f"layer {la}", color="steelblue")
    ax.plot(x, b_vals, marker="s", linewidth=2, label=f"layer {lb}", color="darkorange")
    for i, (a, b) in enumerate(zip(a_vals, b_vals)):
        if np.isnan(a) or np.isnan(b) or a == 0:
            continue
        ratio = b / a
        sign = "↑" if ratio > 1 else "↓"
        ax.annotate(f"{sign}{ratio:.1f}x",
                    xy=(i, max(a, b)),
                    xytext=(0, 5),
                    textcoords="offset points",
                    fontsize=8, ha="center", color="darkred")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(ops, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("|sg-mg| at chosen token (log)", fontsize=11)
    ax.set_title(f"Operator chain at token #{token_idx} — layer {la} vs layer {lb}", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, which="both")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sg-dir", required=True)
    ap.add_argument("--mg-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()

    sg = Path(args.sg_dir)
    mg = Path(args.mg_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cache_path = Path(args.cache) if args.cache else out / "per_token_cache.npz"

    if cache_path.exists():
        print(f"loading cache {cache_path}")
        npz = np.load(cache_path, allow_pickle=True)
        per_op = {k: npz[k] for k in npz.files}
    else:
        per_op = {}
        for name in OP_ORDER:
            print(f"processing {name}...")
            m = build_per_token_matrix(sg, mg, name)
            per_op[name] = m
            print(f"  {name}: shape={m.shape}, finite_count={np.isfinite(m).sum()}")
        np.savez(cache_path, **per_op)
        print(f"saved cache {cache_path}")

    # Heatmap on layer_input (most reliable across all 43 layers)
    li_matrix = per_op.get("layer_input")
    if li_matrix is not None and np.isfinite(li_matrix).sum() > 0:
        viz_heatmap(li_matrix, str(out / "heatmap_43layers.png"),
                    title="layer_input")

    # Pick max-diff tokens: (a) global max, (b) interior max (exclude token 0 + last 2 tokens)
    if li_matrix is not None:
        per_token_max = np.nanmax(li_matrix, axis=0)
        global_max = int(np.nanargmax(per_token_max))
        interior = per_token_max.copy()
        interior[0] = np.nan
        interior[-2:] = np.nan
        interior_max = int(np.nanargmax(interior))
        print(f"global max-diff token: {global_max} (cumulative max = {per_token_max[global_max]:.3e})")
        print(f"interior max-diff token: {interior_max} (cumulative max = {per_token_max[interior_max]:.3e})")

        for chosen_tag, chosen in [("global", global_max), ("interior", interior_max)]:
            for la, lb in [(0, 1), (1, 2), (2, 3)]:
                viz_layer_pair_at_token(per_op, chosen, la, lb,
                                        str(out / f"layer_pair_op_chain_L{la}_vs_L{lb}_{chosen_tag}_token{chosen}.png"))


if __name__ == "__main__":
    main()
