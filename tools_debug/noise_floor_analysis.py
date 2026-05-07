"""Stage 1c — produce the noise-floor table.

Inputs:
  --sg1: sg1 sgl_response.json (output_token_logprobs at decode positions)
  --sg2: sg2 sgl_response.json (input_token_logprobs at all prefill positions)
  --mg1-dir: mg1 megatron_logprobs/ (8 rank_*.json)
  --mg2-dir: mg2 megatron_logprobs/ (8 rank_*.json)
  --prompt-len: int, sg1's prompt length (= 2094 here)

Outputs:
  Table: per-position-bucket (prefill / decode), per-comparison (sg-sg / mg-mg / sg-mg-cross)
  metrics: count, p50, p99, max, mean.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def load_sg1_decode_logprobs(path: str) -> dict[int, float]:
    """sg1: output_token_logprobs at decode positions [prompt_len..prompt_len+resp_len)."""
    d = json.load(open(path))
    otp = d["meta_info"]["output_token_logprobs"]
    # otp entries are [logprob, token_id, top_logprobs]
    n = len(otp)
    # caller knows prompt_len; we return offsets 0..n-1 keyed for caller to add prompt_len
    return {i: float(e[0]) for i, e in enumerate(otp)}


def load_sg2_prefill_logprobs(path: str) -> dict[int, float]:
    """sg2: input_token_logprobs at all prefill positions, indexed by global pos.
    sglang's input_token_logprobs[i] is logprob of input_ids[i+1] given input_ids[:i+1].
    So the entry at index i represents the logprob of TOKEN at GLOBAL position (i+1).
    """
    d = json.load(open(path))
    itp = d["meta_info"]["input_token_logprobs"]
    out = {}
    for i, e in enumerate(itp):
        if e[0] is None:
            continue
        out[i + 1] = float(e[0])
    return out


def load_mg_logprobs(dir_path: str) -> dict[int, float]:
    """Concatenate all 8 ranks' logprob_entries by global_position."""
    out = {}
    for f in sorted(Path(dir_path).glob("rank_*.json")):
        d = json.load(open(f))
        for batch in d["logprob_entries"]:
            for e in batch:
                if not e.get("is_valid", True):
                    continue
                p = int(e["global_position"])
                out[p] = float(e["logprob"])
    return out


def stats(arr: np.ndarray, label: str) -> str:
    if len(arr) == 0:
        return f"{label}: n=0"
    return (
        f"{label}: n={len(arr):4d} "
        f"mean={arr.mean():.4f} p50={np.median(arr):.4f} p90={np.quantile(arr, 0.9):.4f} "
        f"p99={np.quantile(arr, 0.99):.4f} max={arr.max():.4f}"
    )


def diff_at_common(a: dict[int, float], b: dict[int, float], positions=None) -> tuple[np.ndarray, np.ndarray]:
    keys = sorted(set(a.keys()) & set(b.keys()))
    if positions is not None:
        keys = [k for k in keys if k in positions]
    diffs = np.array([abs(a[k] - b[k]) for k in keys])
    return np.array(keys), diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sg1", required=True, help="sg1 sgl_response.json (decode logprobs)")
    ap.add_argument("--sg2", required=True, help="sg2 sgl_response.json (prefill logprobs)")
    ap.add_argument("--sg3", default=None, help="sg3 sgl_response.json (prefill logprobs, fresh run)")
    ap.add_argument("--mg1-dir", required=True)
    ap.add_argument("--mg2-dir", default=None)
    ap.add_argument("--prompt-len", type=int, required=True)
    ap.add_argument("--total-len", type=int, required=True, help="prompt_len + decode_len")
    args = ap.parse_args()

    prompt_len = args.prompt_len
    total_len = args.total_len
    decode_positions = set(range(prompt_len, total_len))
    prefill_positions = set(range(1, prompt_len))  # pos 0 has no logprob (no preceding context)
    all_positions = decode_positions | prefill_positions

    sg1_decode_off = load_sg1_decode_logprobs(args.sg1)
    sg1 = {prompt_len + k: v for k, v in sg1_decode_off.items()}
    sg2 = load_sg2_prefill_logprobs(args.sg2)
    sg3 = load_sg2_prefill_logprobs(args.sg3) if args.sg3 else {}
    mg1 = load_mg_logprobs(args.mg1_dir)
    mg2 = load_mg_logprobs(args.mg2_dir) if args.mg2_dir else {}

    def cov(d): return f"{len(d)} positions, range {min(d)}..{max(d)}" if d else "EMPTY"
    print(f"sg1 (decode):    {cov(sg1)}")
    print(f"sg2 (prefill):   {cov(sg2)}")
    print(f"sg3 (prefill):   {cov(sg3)}") if sg3 else None
    print(f"mg1:             {cov(mg1)}")
    print(f"mg2:             {cov(mg2)}") if mg2 else None
    print()

    def report(label, a, b, positions):
        pos, diffs = diff_at_common(a, b, positions)
        print(stats(diffs, label))
        if len(diffs):
            idx = np.argsort(-diffs)[:5]
            print(f"  worst 5: {[(int(pos[i]), round(float(diffs[i]), 3)) for i in idx]}")

    print("=== C1: sg1 vs sg2 (decode pos) — sg prefill vs decode kernel + sg-noise ===")
    report("C1", sg1, sg2, decode_positions)
    print()

    if sg3:
        print("=== C2: sg2 vs sg3 (all pos) — pure sg prefill kernel noise ===")
        report("C2", sg2, sg3, all_positions)
        print()

    if mg2:
        print("=== C3: mg1 vs mg2 (all pos) — pure mg forward kernel noise ===")
        report("C3", mg1, mg2, all_positions)
        print()

    print("=== C4: sg1 vs mg1 (decode pos) — cross-stack at sg-decode kernel ===")
    report("C4", sg1, mg1, decode_positions)
    print()

    print("=== C5: sg2 vs mg1 (all pos) — cross-stack at sg-prefill kernel ===")
    report("C5", sg2, mg1, all_positions)
    print()

    if sg3 and mg2:
        print("=== Derived: cross-stack signal above noise (C5 - C2 - C3) ===")
        # For each position, signal_p = max(0, |sg2-mg1|_p - sqrt(|sg2-sg3|_p^2 + |mg1-mg2|_p^2))
        positions_list = sorted(set(sg2) & set(sg3) & set(mg1) & set(mg2) & all_positions)
        sig = []
        for p in positions_list:
            cross = abs(sg2[p] - mg1[p])
            ng_sg = abs(sg2[p] - sg3[p])
            ng_mg = abs(mg1[p] - mg2[p])
            noise = (ng_sg ** 2 + ng_mg ** 2) ** 0.5
            sig.append((p, max(0.0, cross - noise), cross, noise))
        sig.sort(key=lambda x: -x[1])
        print(f"  positions where signal > 5 nats: {sum(1 for s in sig if s[1] > 5)}")
        print(f"  positions where signal > 1 nat:  {sum(1 for s in sig if s[1] > 1)}")
        print(f"  top-10 cleanest above-noise positions:")
        for p, s, c, n in sig[:10]:
            print(f"    pos {p:5d}  signal={s:6.2f}  cross={c:6.2f}  noise={n:5.2f}")


if __name__ == "__main__":
    main()
