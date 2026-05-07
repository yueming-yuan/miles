"""Compare per-token logprobs between sglang and megatron lm_head_logits dumps.

Self-contained: takes two dumper output directories (one from sg, one from mg),
finds each side's `lm_head_logits` tensor, applies log_softmax locally, and reports
per-token diff statistics over a chosen token range.

No dependence on historical rollout logprobs or the sglang HTTP curl response —
the bytes-of-truth are the .pt dump files.

Usage:
    python tools_debug/compare_logprobs_from_dumps.py \\
        --sg-dump-dir /storage/yueming/dumper-out/<run>-sg \\
        --mg-dump-dir /storage/yueming/dumper-out/<run>-mg \\
        --rollout-data /storage/yueming/rollout-subsets/iter59_synthetic_2094_plus32.pt

Optional:
    --token-range {all,response,prompt}   default all
    --align-vocab                         slice mg logits to sg's V before log_softmax
                                          (matches sg's softmax denominator; eliminates
                                          vocab-padding effect)
    --top-n N                             print top N worst positions (default 10)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


def _find_lm_head_logits_dump(dump_dir: Path, *, prefer_phase: str = "EXTEND") -> Path:
    """Find an active-rank lm_head_logits dump file under dump_dir (recursive).

    Active = shape[0] > 0 (sg has empty tensors on idle DP ranks; mg replicates across
    TP so all ranks have the same data). We prefer phase=EXTEND for prefill.
    """
    pattern = str(dump_dir / "**" / "*name=lm_head_logits*.pt")
    candidates = sorted(glob.glob(pattern, recursive=True))
    if not candidates:
        raise FileNotFoundError(f"no lm_head_logits dump under {dump_dir}")

    chosen: tuple[int, str, Path] | None = None  # (priority, path) where lower = better
    for path in candidates:
        meta = torch.load(path, weights_only=False)
        value = meta["value"] if isinstance(meta, dict) else meta
        if value.numel() == 0 or value.shape[0] == 0:
            continue
        is_extend = f"phase={prefer_phase}" in path
        is_rank0 = "rank=0" in path
        priority = (0 if is_extend else 1) * 10 + (0 if is_rank0 else 1)
        if chosen is None or priority < chosen[0]:
            chosen = (priority, path, Path(path))

    if chosen is None:
        raise FileNotFoundError(
            f"all lm_head_logits dumps under {dump_dir} are empty (idle ranks)"
        )
    return chosen[2]


def _load_logits(path: Path) -> torch.Tensor:
    obj = torch.load(path, weights_only=False)
    return obj["value"] if isinstance(obj, dict) else obj


def _normalize_to_T_V(t: torch.Tensor) -> torch.Tensor:
    """Squeeze a leading bsz=1 dim if present. Return shape [T, V]."""
    if t.dim() == 3 and t.shape[0] == 1:
        return t.squeeze(0)
    if t.dim() == 2:
        return t
    raise ValueError(f"unexpected logits shape {tuple(t.shape)}; expected [T,V] or [1,T,V]")


def _load_tokens(rollout_path: Path, sample_index: int) -> tuple[list[int], int]:
    data = torch.load(rollout_path, weights_only=False)
    s = data["samples"][sample_index]
    tokens = list(s["tokens"]) if not isinstance(s["tokens"], list) else s["tokens"]
    response_length = int(s["response_length"])
    return tokens, response_length


@dataclass
class CompareReport:
    range_label: str
    n_positions: int
    n_skipped_oob: int
    max_abs_diff: float
    mean_abs_diff: float
    median_abs_diff: float
    p95_abs_diff: float
    p99_abs_diff: float
    sg_mean_logprob: float
    mg_mean_logprob: float
    worst_pos: int
    worst_token: int
    worst_sg_lp: float
    worst_mg_lp: float
    top_n_worst: list[tuple[int, int, float, float, float]]  # (pos, token, sg_lp, mg_lp, diff)


def compare(
    *,
    sg_dump_dir: Path,
    mg_dump_dir: Path,
    rollout_data: Path,
    sample_index: int = 0,
    token_range: str = "all",
    align_vocab: bool = False,
    top_n: int = 10,
) -> CompareReport:
    sg_path = _find_lm_head_logits_dump(sg_dump_dir, prefer_phase="EXTEND")
    mg_path = _find_lm_head_logits_dump(mg_dump_dir, prefer_phase="EXTEND")
    print(f"[compare] sg dump: {sg_path}", flush=True)
    print(f"[compare] mg dump: {mg_path}", flush=True)

    sg_logits = _normalize_to_T_V(_load_logits(sg_path))  # [T_sg, V_sg]
    mg_logits = _normalize_to_T_V(_load_logits(mg_path))  # [T_mg_padded, V_mg]
    print(
        f"[compare] sg shape={tuple(sg_logits.shape)} dtype={sg_logits.dtype}; "
        f"mg shape={tuple(mg_logits.shape)} dtype={mg_logits.dtype}",
        flush=True,
    )

    tokens, response_length = _load_tokens(rollout_data, sample_index)
    n_tokens = len(tokens)
    prompt_length = n_tokens - response_length
    print(
        f"[compare] tokens={n_tokens} prompt_length={prompt_length} response_length={response_length}",
        flush=True,
    )

    T_sg = sg_logits.shape[0]
    T_mg = mg_logits.shape[0]
    V_sg = sg_logits.shape[1]
    V_mg = mg_logits.shape[1]

    if align_vocab and V_mg > V_sg:
        mg_logits = mg_logits[:, :V_sg]

    # log_softmax in FP32 on each side's natural domain (or aligned vocab)
    sg_lp = torch.log_softmax(sg_logits.float(), dim=-1)  # [T_sg, V_sg]
    mg_lp = torch.log_softmax(mg_logits.float(), dim=-1)  # [T_mg_padded, V_mg or V_sg]

    # Build the position list per --token-range
    # Position p predicts tokens[p+1]; valid p ∈ [0, n_tokens-2]
    if token_range == "all":
        positions = list(range(0, n_tokens - 1))
        range_label = f"ALL (positions 0..{n_tokens-2}, predicting tokens 1..{n_tokens-1})"
    elif token_range == "response":
        positions = list(range(prompt_length - 1, n_tokens - 1))
        range_label = (
            f"RESPONSE (positions {prompt_length-1}..{n_tokens-2}, predicting tokens "
            f"{prompt_length}..{n_tokens-1}, last {response_length} tokens)"
        )
    elif token_range == "prompt":
        positions = list(range(0, prompt_length - 1))
        range_label = (
            f"PROMPT (positions 0..{prompt_length-2}, predicting tokens 1..{prompt_length-1})"
        )
    else:
        raise ValueError(f"unknown token_range={token_range!r}")

    # Filter to positions where both sides have valid data
    skipped_oob = 0
    diffs: list[tuple[int, int, float, float, float]] = []
    for p in positions:
        if p >= T_sg or p >= T_mg:
            skipped_oob += 1
            continue
        label = int(tokens[p + 1])
        if label < 0 or label >= V_sg:  # don't index out of sg's vocab
            skipped_oob += 1
            continue
        sg_v = sg_lp[p, label].item()
        mg_v = mg_lp[p, label].item()
        diffs.append((p, label, sg_v, mg_v, abs(sg_v - mg_v)))

    if not diffs:
        raise RuntimeError(
            f"no valid positions in range={token_range!r}; T_sg={T_sg}, T_mg={T_mg}, n_tokens={n_tokens}"
        )

    abs_diffs = sorted([d[4] for d in diffs])
    n = len(abs_diffs)
    sg_mean = sum(d[2] for d in diffs) / n
    mg_mean = sum(d[3] for d in diffs) / n
    worst = max(diffs, key=lambda d: d[4])
    top_n_sorted = sorted(diffs, key=lambda d: -d[4])[:top_n]

    def _pct(arr: list[float], q: float) -> float:
        idx = max(0, min(len(arr) - 1, int(round((len(arr) - 1) * q))))
        return arr[idx]

    return CompareReport(
        range_label=range_label,
        n_positions=n,
        n_skipped_oob=skipped_oob,
        max_abs_diff=abs_diffs[-1],
        mean_abs_diff=sum(abs_diffs) / n,
        median_abs_diff=_pct(abs_diffs, 0.5),
        p95_abs_diff=_pct(abs_diffs, 0.95),
        p99_abs_diff=_pct(abs_diffs, 0.99),
        sg_mean_logprob=sg_mean,
        mg_mean_logprob=mg_mean,
        worst_pos=worst[0],
        worst_token=worst[1],
        worst_sg_lp=worst[2],
        worst_mg_lp=worst[3],
        top_n_worst=top_n_sorted,
    )


def _print_report(r: CompareReport, *, align_vocab: bool) -> None:
    print()
    print("=" * 70)
    print(f"Logprob comparison (sg dump vs mg dump, log_softmax computed locally)")
    print("=" * 70)
    print(f"  Token range        : {r.range_label}")
    print(f"  Vocab alignment    : {'mg sliced to sg V before log_softmax' if align_vocab else 'each side natural V'}")
    print(f"  Positions compared : {r.n_positions}")
    print(f"  Skipped OOB        : {r.n_skipped_oob}")
    print()
    print(f"  Max  abs diff      : {r.max_abs_diff:.6e}")
    print(f"  Mean abs diff      : {r.mean_abs_diff:.6e}")
    print(f"  Median abs diff    : {r.median_abs_diff:.6e}")
    print(f"  P95 abs diff       : {r.p95_abs_diff:.6e}")
    print(f"  P99 abs diff       : {r.p99_abs_diff:.6e}")
    print()
    print(f"  sg mean logprob    : {r.sg_mean_logprob:.6f}")
    print(f"  mg mean logprob    : {r.mg_mean_logprob:.6f}")
    print()
    print(f"  Worst position     : {r.worst_pos}")
    print(f"    token_id         : {r.worst_token}")
    print(f"    sg logprob       : {r.worst_sg_lp:.6f}")
    print(f"    mg logprob       : {r.worst_mg_lp:.6f}")
    print()
    print(f"  Top {len(r.top_n_worst)} worst positions:")
    print(f"    {'pos':>6}  {'token':>6}  {'sg_lp':>12}  {'mg_lp':>12}  {'diff':>12}")
    for pos, tok, sg_v, mg_v, diff in r.top_n_worst:
        print(f"    {pos:>6}  {tok:>6}  {sg_v:>12.6f}  {mg_v:>12.6f}  {diff:>12.6f}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sg-dump-dir", required=True, type=Path,
                        help="Directory containing sglang's dumper output (recursively searched for lm_head_logits)")
    parser.add_argument("--mg-dump-dir", required=True, type=Path,
                        help="Directory containing megatron's dumper output (recursively searched for lm_head_logits)")
    parser.add_argument("--rollout-data", required=True, type=Path,
                        help="Path to a .pt file with samples[0].tokens for label lookup")
    parser.add_argument("--sample-index", default=0, type=int)
    parser.add_argument("--token-range", default="all", choices=["all", "response", "prompt"],
                        help="Which token positions to compare (default: all)")
    parser.add_argument("--align-vocab", action="store_true",
                        help="Slice mg logits to sg's V before log_softmax (eliminates mg vocab-padding effect)")
    parser.add_argument("--top-n", default=10, type=int)
    args = parser.parse_args()

    if not args.sg_dump_dir.exists():
        sys.exit(f"sg dump dir not found: {args.sg_dump_dir}")
    if not args.mg_dump_dir.exists():
        sys.exit(f"mg dump dir not found: {args.mg_dump_dir}")
    if not args.rollout_data.exists():
        sys.exit(f"rollout-data not found: {args.rollout_data}")

    report = compare(
        sg_dump_dir=args.sg_dump_dir,
        mg_dump_dir=args.mg_dump_dir,
        rollout_data=args.rollout_data,
        sample_index=args.sample_index,
        token_range=args.token_range,
        align_vocab=args.align_vocab,
        top_n=args.top_n,
    )
    _print_report(report, align_vocab=args.align_vocab)


if __name__ == "__main__":
    main()
