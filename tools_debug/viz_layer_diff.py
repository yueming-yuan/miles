"""Two visualizations on cmp jsonl:
   1. Per-layer overview (43 layers x N tensor names) — line chart
   2. Operator chain comparison for layer-pairs (L0vsL1, L1vsL2, L2vsL3) — grouped bars

Reads a comparator JSONL produced by `python -m sglang.srt.debug_utils.comparator`.
Extracts per (name, layer_id) mean_abs_diff / max_abs_diff from comparison_tensor records.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Approximate execution order within a V4 transformer layer
OP_ORDER = [
    "layer_input",
    "q_lora_after_norm",
    "attn_q",
    "kv_after_norm",
    "attn_v",
    "compressor_kv_score",
    "mqa_wo_b_out",
    "pre_mlp_layernorm_output",
]


def load_diffs(path: str) -> dict[str, dict[int, dict]]:
    """Return {name: {layer_id: {mean, max, rel, passed}}}."""
    by_name_layer: dict[str, dict[int, dict]] = defaultdict(dict)
    for line in open(path):
        e = json.loads(line)
        if e.get("type") != "comparison_tensor":
            continue
        rbi = e.get("raw_bundle_info", {})
        yf = rbi.get("y", {}).get("files", [])
        if not yf:
            continue
        fn = yf[0].get("filename", "")
        m = re.search(r"layer_id=(\d+)", fn)
        if not m:
            continue
        layer_id = int(m.group(1))
        diff = e.get("diff")
        if diff is None:
            continue
        # avoid duplicates: skip if same (name, layer_id) seen with cr variants
        # (compressor_kv_score has cr=4 and cr=128 entries — keep only one per layer for now)
        cr_m = re.search(r"compress_ratio=(\d+)", fn)
        # store per-name per-layer; if dupe, keep the first
        nkey = e.get("name")
        if cr_m and nkey == "compressor_kv_score":
            nkey = f"{nkey}_cr{cr_m.group(1)}"
        if layer_id in by_name_layer[nkey]:
            continue
        by_name_layer[nkey][layer_id] = {
            "mean": diff.get("mean_abs_diff"),
            "max": diff.get("max_abs_diff"),
            "rel": diff.get("rel_diff"),
            "passed": diff.get("passed"),
        }
    return by_name_layer


def viz_per_layer(diffs: dict, out_path: str, n_layers: int = 43) -> None:
    """Line chart: x = layer_id 0..42, y = mean_abs_diff (log scale). One line per tensor name."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    names = sorted(diffs.keys())
    cmap = plt.get_cmap("tab10")

    for plot_idx, metric in enumerate(["mean", "max"]):
        ax = axes[plot_idx]
        for i, name in enumerate(names):
            xs = sorted(diffs[name].keys())
            ys = [diffs[name][x][metric] for x in xs]
            ax.plot(xs, ys, marker="o", markersize=4, label=name, color=cmap(i % 10), alpha=0.85)
        ax.set_yscale("log")
        ax.set_ylabel(f"{metric}_abs_diff (log)", fontsize=11)
        ax.grid(True, axis="y", alpha=0.3, which="both")
        if plot_idx == 0:
            ax.set_title(
                f"Per-layer mean/max abs diff (sglang vs megatron, V4-Flash iter-59)",
                fontsize=12,
            )
        if plot_idx == 1:
            ax.set_xlabel("layer_id", fontsize=11)
        ax.legend(fontsize=8, loc="upper right", ncol=3)
        ax.set_xticks(range(0, n_layers, 2))

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved {out_path}")


def viz_layer_pair(diffs: dict, layer_a: int, layer_b: int, out_path: str) -> None:
    """Grouped bar chart for one layer-pair.
    x-axis: operator name in execution order
    y-axis: mean_abs_diff (log scale)
    Two bars per op: layer_a, layer_b
    """
    ops = [op for op in OP_ORDER if op in diffs]
    # also append cr-specific compressor variants if available
    for k in sorted(diffs):
        if k.startswith("compressor_kv_score_cr") and k not in ops:
            ops.append(k)

    n_ops = len(ops)
    a_means = [diffs[op].get(layer_a, {}).get("mean") if layer_a in diffs[op] else None for op in ops]
    b_means = [diffs[op].get(layer_b, {}).get("mean") if layer_b in diffs[op] else None for op in ops]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(n_ops)
    width = 0.35
    a_vals = [v if v is not None else 0 for v in a_means]
    b_vals = [v if v is not None else 0 for v in b_means]
    bars_a = ax.bar(x - width / 2, a_vals, width, label=f"layer {layer_a}", color="steelblue")
    bars_b = ax.bar(x + width / 2, b_vals, width, label=f"layer {layer_b}", color="darkorange")

    # Annotate jump factor when both present
    for i, (a_v, b_v) in enumerate(zip(a_means, b_means)):
        if a_v is None or b_v is None:
            continue
        if a_v == 0:
            continue
        ratio = b_v / a_v
        sign = "↑" if ratio > 1 else "↓"
        ax.annotate(f"{sign}{ratio:.1f}x",
                    xy=(i, max(a_v, b_v)),
                    xytext=(0, 3),
                    textcoords="offset points",
                    fontsize=9, color="darkred", ha="center")
        # Also write the actual values in small text
        ax.annotate(f"{a_v:.2e}\n{b_v:.2e}",
                    xy=(i, max(a_v, b_v)),
                    xytext=(0, 18),
                    textcoords="offset points",
                    fontsize=7, ha="center", color="gray")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(ops, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("mean_abs_diff (log scale)", fontsize=11)
    ax.set_title(f"Operator chain mean abs diff — layer {layer_a} vs layer {layer_b}", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, which="both")

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmp-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    diffs = load_diffs(args.cmp_jsonl)
    print(f"loaded {len(diffs)} names: {sorted(diffs.keys())}")
    for n in diffs:
        print(f"  {n}: {len(diffs[n])} layers")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Per-layer overview
    viz_per_layer(diffs, str(out / "per_layer_43.png"))

    # 2. Three layer-pair plots
    for la, lb in [(0, 1), (1, 2), (2, 3)]:
        viz_layer_pair(diffs, la, lb, str(out / f"layer_pair_L{la}_vs_L{lb}.png"))


if __name__ == "__main__":
    main()
