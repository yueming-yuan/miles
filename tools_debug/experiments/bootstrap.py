"""Register pre-existing dump dirs into runs.jsonl so the framework can reference them.

Legacy ``tools_debug/launch_*.sh`` scripts wrote dumps under predictable names
(``v4-iter59-grafter-<tag>-{sg,mg}``, ``v4-iter59-mg-fix0505``, etc.) that the new
runner can attribute to specs in ``tools_debug.experiments.registry`` by name.
This module walks the output root, matches each dir against a spec, and emits a
synthetic ``runs.jsonl`` row pointing at the dump -- without re-running anything.

Once bootstrapped, ``python -m miles.utils.debug_utils.experiment_runner compare
--target a1 --baselines sg-natural-prefill-e8env`` works against legacy dumps
through the same code path as fresh runs.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from tools_debug.experiments import CANONICAL, EXPERIMENTS

from miles.utils.debug_utils.experiment_runner.registry import RunRecord, append_run

_LEGACY_DUMP_PREFIX = "v4-iter59-"


def _legacy_dirs(spec_name: str, output_root: Path) -> dict[str, Path]:
    """Map a spec's expected legacy dump-dir paths under ``output_root``.

    Legacy script naming -- mirrors what the original launch_*.sh OUTPUT vars used.
    Returns a dict with optional ``sg`` and ``mg`` keys; missing keys mean that
    side was never produced (e.g. sg_only specs have no mg dir).
    """
    sg_legacy_name = f"{_LEGACY_DUMP_PREFIX}{_legacy_sg_tag(spec_name)}"
    mg_legacy_name = f"{_LEGACY_DUMP_PREFIX}{_legacy_mg_tag(spec_name)}"
    out: dict[str, Path] = {}
    sg_path = output_root / sg_legacy_name
    mg_path = output_root / mg_legacy_name
    if sg_path.is_dir():
        out["sg"] = sg_path
    if mg_path.is_dir():
        out["mg"] = mg_path
    return out


def _legacy_sg_tag(spec_name: str) -> str:
    if spec_name == "sg-natural-prefill-e8env":
        return "sg-natural-e8env"
    if spec_name.startswith("sg-prefill-"):
        return spec_name
    if spec_name == "mg-fix0505":
        return "mg-fix0505"
    if spec_name.startswith("mg-replay-"):
        return spec_name
    return f"grafter-{spec_name}-sg"


def _legacy_mg_tag(spec_name: str) -> str:
    if spec_name == "mg-fix0505":
        return "mg-fix0505"
    if spec_name.startswith("mg-replay-"):
        return spec_name
    if spec_name in ("sg-natural-prefill-e8env",) or spec_name.startswith("sg-prefill-"):
        return ""
    return f"grafter-{spec_name}-mg"


def bootstrap(*, runs_jsonl: Path, dry_run: bool = False) -> int:
    """Walk EXPERIMENTS, emit one runs.jsonl row per spec whose dump dir exists.

    Returns the number of rows written. ``dry_run=True`` lists what would be
    written without modifying the registry.
    """
    output_root = Path(CANONICAL.output_root)
    written = 0
    for spec_name, spec in EXPERIMENTS.items():
        dirs = _legacy_dirs(spec_name, output_root)
        if not dirs:
            continue
        sg_dump_dir = dirs.get("sg")
        mg_dump_dir = dirs.get("mg")
        sg_baseline_json = _find_first(sg_dump_dir, ["sgl_response_1.json", "sgl_response.json", "sg_baseline.json"])
        mg_logprob_dir = (mg_dump_dir / "megatron_logprobs") if mg_dump_dir is not None else None
        if mg_logprob_dir is not None and not mg_logprob_dir.is_dir():
            mg_logprob_dir = None

        if sg_dump_dir is None and mg_dump_dir is None:
            continue

        synth_finished_at = _legacy_synth_timestamp(spec_name)
        record = RunRecord(
            name=spec_name,
            description=spec.description,
            kind=spec.kind.value,
            canonical_name=CANONICAL.name,
            started_at=synth_finished_at,
            finished_at=synth_finished_at,
            sg_dump_dir=str(sg_dump_dir) if sg_dump_dir else None,
            sg_baseline_json=str(sg_baseline_json) if sg_baseline_json else None,
            mg_dump_dir=str(mg_dump_dir) if mg_dump_dir else None,
            mg_logprob_dir=str(mg_logprob_dir) if mg_logprob_dir else None,
            git={},
            image=None,
            spec_snapshot=_spec_snapshot(spec),
            run_config_snapshot={},
        )
        if dry_run:
            print(f"[bootstrap] would register {spec_name}: sg={sg_dump_dir} mg={mg_dump_dir}", flush=True)
        else:
            append_run(record, jsonl_path=runs_jsonl)
            print(f"[bootstrap] registered {spec_name}: sg={sg_dump_dir} mg={mg_dump_dir}", flush=True)
        written += 1
    return written


def _find_first(directory: Path | None, candidates: list[str]) -> Path | None:
    if directory is None:
        return None
    for c in candidates:
        p = directory / c
        if p.is_file():
            return p
    return None


def _legacy_synth_timestamp(spec_name: str) -> str:
    """Synthetic finished_at for legacy dumps. Set to a date before the framework
    existed so legacy rows sort before fresh runs of the same spec."""
    return f"2026-05-07T00:00:00+00:00#legacy-{spec_name}"


def _spec_snapshot(spec) -> dict:
    d = dataclasses.asdict(spec)
    d["kind"] = spec.kind.value
    return d


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap legacy dump dirs into runs.jsonl")
    parser.add_argument("--runs-jsonl", default="/storage/yueming/experiment_runner/runs.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="list registrations without writing")
    args = parser.parse_args(argv)

    runs_jsonl = Path(args.runs_jsonl)
    n = bootstrap(runs_jsonl=runs_jsonl, dry_run=args.dry_run)
    print(f"[bootstrap] {n} rows {'would be ' if args.dry_run else ''}written to {runs_jsonl}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
