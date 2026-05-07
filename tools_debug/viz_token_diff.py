"""Visualize per-token logprob diff (sglang vs megatron) for one chosen sample.

Layout:
  Top subplot: x-axis = response tokens (pure text), y-axis = abs(sg_logprob - mg_logprob)
  Bottom subplot: placeholder for per-layer diff (TODO)

Inputs:
  --sg JSON          sglang per-token logprobs (rank_0.json from Phase 1)
  --mg JSON          megatron per-token logprobs (same path)
  --sample-idx INT   which sample in the array (default 12)
  --rollout-pt PATH  the .pt with samples (for tokenizer + sample.tokens)
  --tokenizer PATH   HF tokenizer dir (default: same as iter-59 HF)
  --out PNG          output path

Usage:
  python tools_debug/viz_token_diff.py \\
    --sg /storage/.../sglang_logprobs/rank_0.json \\
    --mg /storage/.../megatron_logprobs/rank_0.json \\
    --rollout-pt /storage/.../iter59_le4096.pt \\
    --sample-idx 12 \\
    --out /storage/.../viz_sample12.png
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer


def load_logprobs(path: str, sample_idx: int) -> dict[int, dict]:
    """Return {global_position -> entry} for the chosen sample."""
    data = json.load(open(path))["logprob_entries"]
    return {e["global_position"]: e for e in data[sample_idx]}


def build_per_token_diff(
    sg_by_pos: dict[int, dict],
    mg_by_pos: dict[int, dict],
) -> tuple[list[int], list[int], list[float], list[float], list[float]]:
    """Aligned response positions: positions, token_ids, sg_lp, mg_lp, abs_diff."""
    positions = sorted(sg_by_pos.keys())
    tids, sgs, mgs, diffs = [], [], [], []
    for gp in positions:
        if gp not in mg_by_pos:
            continue
        se = sg_by_pos[gp]
        me = mg_by_pos[gp]
        if me["token_id"] != se["token_id"]:
            continue
        tids.append(se["token_id"])
        sgs.append(se["logprob"])
        mgs.append(me["logprob"])
        diffs.append(abs(se["logprob"] - me["logprob"]))
    return positions[: len(tids)], tids, sgs, mgs, diffs


def render_token_label(text: str, max_chars: int = 8) -> str:
    """Compact per-token label: replace whitespace, truncate."""
    s = text.replace("\n", "\\n").replace("\t", "\\t").replace(" ", "·")
    if len(s) > max_chars:
        s = s[: max_chars - 1] + "…"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sg", required=True)
    ap.add_argument("--mg", required=True)
    ap.add_argument("--sample-idx", type=int, default=12)
    ap.add_argument("--rollout-pt", required=True)
    ap.add_argument(
        "--tokenizer",
        default="/storage/shidong/personal_data/job_workspaces/shidong-dsv4-flash-eval/dsv4-flash-iter59-hf",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=2200, help="Truncate x-axis if response very long")
    ap.add_argument("--window", type=int, default=None,
                    help="If set, show only this many tokens, centered on --center (default: max-diff token)")
    ap.add_argument("--center", type=int, default=None,
                    help="Global position to center the window on (default: position of max abs_diff)")
    args = ap.parse_args()

    sg_by_pos = load_logprobs(args.sg, args.sample_idx)
    mg_by_pos = load_logprobs(args.mg, args.sample_idx)
    positions, tids, sgs, mgs, diffs = build_per_token_diff(sg_by_pos, mg_by_pos)
    n = len(diffs)
    print(f"sample {args.sample_idx}: {n} aligned tokens, mean_diff={sum(diffs)/n:.4f}, max_diff={max(diffs):.4f}")
    # Apply window if specified
    if args.window is not None:
        if args.center is not None:
            # Find index in positions list closest to --center
            center_idx = min(range(len(positions)), key=lambda i: abs(positions[i] - args.center))
        else:
            # Default: center on max-diff token
            center_idx = max(range(len(diffs)), key=lambda i: diffs[i])
        half = args.window // 2
        lo = max(0, center_idx - half)
        hi = min(n, lo + args.window)
        lo = max(0, hi - args.window)
        positions, tids, sgs, mgs, diffs = (
            positions[lo:hi], tids[lo:hi], sgs[lo:hi], mgs[lo:hi], diffs[lo:hi],
        )
        n = len(diffs)
        print(f"window: showing {n} tokens at gp [{positions[0]}..{positions[-1]}], centered on idx {center_idx} (gp={positions[center_idx-lo]})")
    elif n > args.max_tokens:
        positions, tids, sgs, mgs, diffs = (
            positions[: args.max_tokens],
            tids[: args.max_tokens],
            sgs[: args.max_tokens],
            mgs[: args.max_tokens],
            diffs[: args.max_tokens],
        )
        n = len(diffs)

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    token_texts = [tok.decode([tid]) for tid in tids]
    labels = [render_token_label(t) for t in token_texts]

    sample = torch.load(args.rollout_pt, weights_only=False)["samples"][args.sample_idx]
    response_text = sample.get("response", "")
    sample_index = sample.get("index", "?")

    fig = plt.figure(figsize=(max(20, n / 30), 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.4)

    # Top: per-token logprob diff
    ax_top = fig.add_subplot(gs[0])
    ax_top.bar(range(n), diffs, color="steelblue", width=1.0)
    ax_top.set_ylabel("|sglang_logprob − megatron_logprob|", fontsize=11)
    ax_top.set_title(
        f"Per-token logprob diff — sample {args.sample_idx} (global_index={sample_index}, n={n})\n"
        f"mean={sum(diffs)/n:.3f}  max={max(diffs):.2f}  p99={sorted(diffs)[max(0,int(n*0.99)-1)]:.2f}",
        fontsize=12,
    )
    ax_top.set_xlim(-0.5, n - 0.5)
    ax_top.grid(True, axis="y", alpha=0.3)
    # Always label every token if window is small enough; else stride-sample
    if n <= 80:
        # secondary ticks above showing global position every 8th token
        ax_top.set_xticks(range(n))
        ax_top.set_xticklabels([f"gp{positions[i]}\n{labels[i]}" for i in range(n)],
                               rotation=90, fontsize=8, family="monospace")
    elif n <= 300:
        ax_top.set_xticks(range(n))
        ax_top.set_xticklabels(labels, rotation=90, fontsize=6, family="monospace")
    else:
        stride = max(1, n // 100)
        idx = list(range(0, n, stride))
        ax_top.set_xticks(idx)
        ax_top.set_xticklabels([labels[i] for i in idx], rotation=90, fontsize=5, family="monospace")
    # Mark top-5 worst tokens
    top5 = sorted(range(n), key=lambda i: -diffs[i])[:5]
    for i in top5:
        ax_top.annotate(
            f"{tids[i]}:{labels[i]}",
            xy=(i, diffs[i]),
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=6,
            color="darkred",
            ha="center",
        )

    # Bottom: placeholder for per-layer diff
    ax_bot = fig.add_subplot(gs[1])
    ax_bot.text(
        0.5, 0.5,
        "[per-layer diff — TODO]\nnext iteration: heatmap of layer × token, value = scaled diff",
        ha="center", va="center",
        fontsize=12, color="gray",
    )
    ax_bot.set_xticks([])
    ax_bot.set_yticks([])

    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"saved {args.out}")

    # Also dump a side-channel json with the resolved data
    side = Path(args.out).with_suffix(".json")
    side.write_text(json.dumps({
        "sample_idx": args.sample_idx,
        "sample_index": sample_index,
        "n_tokens": n,
        "positions": positions,
        "token_ids": tids,
        "token_texts": token_texts,
        "sg_logprobs": sgs,
        "mg_logprobs": mgs,
        "abs_diffs": diffs,
        "response_text_preview": response_text[:200] if isinstance(response_text, str) else None,
    }))
    print(f"saved {side}")


if __name__ == "__main__":
    main()
