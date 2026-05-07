"""Build a subset of shidong's iter-59 rollout for fast worst-sample triage."""

import argparse
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--n", type=int, default=32)
    args = ap.parse_args()

    data = torch.load(args.src, weights_only=False)
    samples = data["samples"]
    sorted_by_len = sorted(samples, key=lambda s: len(s["tokens"]))
    step = max(1, len(sorted_by_len) // args.n)
    subset = sorted_by_len[::step][: args.n]

    print(f"selected {len(subset)} of {len(samples)} samples")
    for s in subset:
        print(f"  index={s['index']:7d}  total_len={len(s['tokens']):5d}  response_length={s['response_length']:5d}  reward={s['reward']}")

    out = {"rollout_id": data["rollout_id"], "samples": subset}
    torch.save(out, args.dst)
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
