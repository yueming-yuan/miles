"""JSONL-backed registry of runs and comparisons + markdown renderer.

Two append-only stores:

- ``runs.jsonl``: one row per completed run -- spec snapshot, paths, git SHAs,
  start/end times. The runner appends here when a run finishes.
- ``comparisons.jsonl``: one row per (target_run, baseline_run, metrics) tuple.
  Both the runner (auto, for the spec's declared baselines) and a manual
  ``compare`` subcommand append here.

Both files are append-only; deletes / edits go through a separate compaction
script (not implemented yet -- not needed in steady state since runs are immutable
and rerunning a spec just adds another row with a fresh timestamp).

The renderer reads both files and emits a markdown table with one row per target
run, columns for spec metadata, and one ``<baseline>: mean / max / p99``
column-group per distinct baseline that appears in ``comparisons.jsonl``. Stale /
older comparisons for the same (target, baseline) pair are deduplicated to the
latest by timestamp.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any


@dataclasses.dataclass(frozen=True)
class RunRecord:
    """One ``runs.jsonl`` row.

    All paths are strings (jsonable). ``spec_snapshot`` holds the dict of the
    ExperimentSpec at run time; ``run_config_snapshot`` holds the resolved
    RunConfig. Both are stored verbatim so a later reader can fully reconstruct
    the exact launch context without referencing live spec code.
    """

    name: str
    description: str
    kind: str
    canonical_name: str
    started_at: str
    finished_at: str
    sg_dump_dir: str | None
    sg_baseline_json: str | None
    mg_dump_dir: str | None
    mg_logprob_dir: str | None
    git: dict[str, str]
    image: str | None
    spec_snapshot: dict
    run_config_snapshot: dict

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> RunRecord:
        return cls(**d)


def append_run(record: RunRecord, *, jsonl_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a") as fh:
        fh.write(json.dumps(record.to_dict()) + "\n")


def load_runs(jsonl_path: Path) -> list[RunRecord]:
    if not jsonl_path.exists():
        return []
    return [RunRecord.from_dict(json.loads(line)) for line in jsonl_path.read_text().splitlines() if line.strip()]


def find_latest_run(runs: list[RunRecord], name: str) -> RunRecord | None:
    """Return the most-recently-finished run with the given ``name``.

    Multiple rows for the same name happen when a spec is rerun. Latest by
    ``finished_at`` (ISO 8601 strings sort lexicographically).
    """
    matches = [r for r in runs if r.name == name]
    if not matches:
        return None
    return max(matches, key=lambda r: r.finished_at)


def load_comparisons(jsonl_path: Path) -> list[dict]:
    if not jsonl_path.exists():
        return []
    return [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]


def render_markdown(
    runs_jsonl: Path,
    comparisons_jsonl: Path,
    *,
    metric_range: str = "all",
    baseline_order: list[str] | None = None,
) -> str:
    """Render the run registry as markdown.

    Layout: one row per distinct run name (latest revision). Columns:
    ``run | kind | description | git.<repo>``, then per-baseline column groups
    ``<baseline>:mean | <baseline>:max | <baseline>:p99``. The ``metric_range``
    selects which sub-range (``all`` / ``prompt`` / ``response``) to read from
    each comparison row.

    ``baseline_order`` controls column order; baselines not in the list are
    appended alphabetically. When omitted, all baselines are listed alphabetically.
    """
    runs = load_runs(runs_jsonl)
    comparisons = load_comparisons(comparisons_jsonl)

    latest_run_by_name = {r.name: r for r in sorted(runs, key=lambda r: r.finished_at)}
    target_names = sorted(latest_run_by_name.keys())

    latest_cmp: dict[tuple[str, str], dict] = {}
    for row in comparisons:
        key = (row["target"], row["baseline"])
        prev = latest_cmp.get(key)
        if prev is None or row.get("finished_at", row.get("baseline_dir", "")) > prev.get(
            "finished_at", prev.get("baseline_dir", "")
        ):
            latest_cmp[key] = row

    distinct_baselines = sorted({b for _, b in latest_cmp.keys()})
    if baseline_order:
        ordered = [b for b in baseline_order if b in distinct_baselines]
        ordered += [b for b in distinct_baselines if b not in baseline_order]
        distinct_baselines = ordered

    headers: list[str] = ["run", "kind", "description"]
    for b in distinct_baselines:
        headers += [f"{b}:mean", f"{b}:max", f"{b}:p99"]

    lines: list[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for tname in target_names:
        run = latest_run_by_name[tname]
        cells: list[str] = [tname, run.kind, _md_escape(run.description)]
        for b in distinct_baselines:
            row = latest_cmp.get((tname, b))
            if row is None:
                cells += ["", "", ""]
                continue
            metrics = row.get(metric_range)
            if metrics is None:
                cells += ["", "", ""]
                continue
            cells += [_fmt(metrics["mean_abs_diff"]), _fmt(metrics["max_abs_diff"]), _fmt(metrics["p99_abs_diff"])]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f):
        return ""
    return f"{f:.4g}"


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")
