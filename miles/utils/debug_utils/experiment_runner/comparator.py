"""Compute per-token logprob diff metrics between two ``rank_*.json`` directories.

Reuses ``logprob_comparator._load_and_merge`` to deduplicate per-rank entries, then
computes all metrics (mean/median/p95/p99/max + per-token-range slices) -- the
runner persists these to ``comparisons.jsonl`` and never relies on a binary
pass/fail threshold (unlike the upstream ``compare_logprobs``).

Token-range slicing splits the merged position set into ``prompt`` (positions
``[0, prompt_length-1)``), ``response`` (positions ``[prompt_length-1, total-1)``),
and ``all`` (the union). The split is parameterised because different baselines
encode the prompt/response boundary differently (sg HTTP returns prefill positions
0..total-2; mg returns labels-shifted prefill+response).
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import statistics
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class RangeMetrics:
    """Per-token-range diff statistics. Fields are nan when ``num_positions==0``."""

    num_positions: int
    mean_abs_diff: float
    median_abs_diff: float
    p95_abs_diff: float
    p99_abs_diff: float
    max_abs_diff: float


@dataclasses.dataclass(frozen=True)
class LogprobComparison:
    """Result of comparing two logprob dirs across token ranges."""

    baseline_dir: str
    target_dir: str
    all: RangeMetrics
    prompt: RangeMetrics | None = None
    response: RangeMetrics | None = None
    prompt_length: int | None = None

    def to_jsonable(self) -> dict:
        d: dict = {
            "baseline_dir": self.baseline_dir,
            "target_dir": self.target_dir,
            "all": dataclasses.asdict(self.all),
        }
        if self.prompt is not None:
            d["prompt"] = dataclasses.asdict(self.prompt)
        if self.response is not None:
            d["response"] = dataclasses.asdict(self.response)
        if self.prompt_length is not None:
            d["prompt_length"] = self.prompt_length
        return d


def compare_logprob_dirs(
    *,
    baseline_dir: Path,
    target_dir: Path,
    prompt_length: int | None = None,
) -> LogprobComparison:
    """Compare two ``rank_*.json`` directories; return all-range metrics.

    If ``prompt_length`` is provided, additionally compute ``prompt``-only and
    ``response``-only metrics (response = positions ``>= prompt_length - 1``,
    matching the labels-shifted convention).
    """
    lc = importlib.import_module("miles.utils.debug_utils.run_megatron.logprob_comparator")
    baseline = lc._load_and_merge(baseline_dir)
    target = lc._load_and_merge(target_dir)

    keys = sorted(set(baseline.keys()) & set(target.keys()))
    pairs = [(k, abs(baseline[k].logprob - target[k].logprob)) for k in keys]

    return LogprobComparison(
        baseline_dir=str(baseline_dir),
        target_dir=str(target_dir),
        all=_metrics([d for _, d in pairs]),
        prompt=_metrics_for(pairs, lambda k: k[1] < (prompt_length - 1)) if prompt_length is not None else None,
        response=_metrics_for(pairs, lambda k: k[1] >= (prompt_length - 1)) if prompt_length is not None else None,
        prompt_length=prompt_length,
    )


def write_comparison_jsonl(
    comparison: LogprobComparison,
    *,
    target_run_name: str,
    baseline_run_name: str,
    jsonl_path: Path,
) -> None:
    """Append one comparison row to ``jsonl_path``.

    Row schema::

        {"target": <name>, "baseline": <name>, "metrics": {...}}

    The runner / dispatcher writes one row per (target, baseline) pair. The
    markdown renderer reads ``jsonl_path`` and emits one column per baseline.
    """
    row = {
        "target": target_run_name,
        "baseline": baseline_run_name,
        **comparison.to_jsonable(),
    }
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def _metrics(diffs: list[float]) -> RangeMetrics:
    if not diffs:
        nan = float("nan")
        return RangeMetrics(0, nan, nan, nan, nan, nan)
    sorted_diffs = sorted(diffs)
    n = len(sorted_diffs)
    return RangeMetrics(
        num_positions=n,
        mean_abs_diff=statistics.fmean(sorted_diffs),
        median_abs_diff=sorted_diffs[n // 2],
        p95_abs_diff=sorted_diffs[min(n - 1, int(n * 0.95))],
        p99_abs_diff=sorted_diffs[min(n - 1, int(n * 0.99))],
        max_abs_diff=sorted_diffs[-1],
    )


def _metrics_for(pairs, predicate) -> RangeMetrics:  # type: ignore[no-untyped-def]
    return _metrics([d for k, d in pairs if predicate(k)])
