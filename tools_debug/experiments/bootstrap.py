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

_LEGACY_DIRECT_DIRS: dict[str, dict[str, str]] = {
    "sg-natural-prefill-e8env": {"sg": "sg-natural-e8env"},
    "mg-fix0505": {"mg": "mg-fix0505"},
    "sg-prefill-chunk8192": {"sg": "sg-prefill-chunk8192"},
    "sg-prefill-no-f4": {"sg": "sg-prefill-no-f4"},
    "mg-replay-moe": {"mg": "mg-replay-moe"},
    "mg-replay-both": {"mg": "mg-replay-both"},
}


def _legacy_dirs(spec_name: str, output_root: Path) -> dict[str, Path]:
    """Locate a spec's pre-existing dump dirs under ``output_root``.

    Two patterns:
    - direct: ``v4-iter59-<tag>``, used by sg_only / mg_only baselines and ablations
      (entries in ``_LEGACY_DIRECT_DIRS``).
    - grafter glob: ``v4-iter59-grafter-<spec>-*-{sg,mg}`` for grafter pairs. Glob
      handles the descriptive suffix the legacy launchers added (e.g.
      ``q-lora-only`` for a1, ``router-only`` for m1) without requiring a hardcoded
      table per spec.

    Returns a dict with optional ``sg`` and ``mg`` keys; missing keys mean that
    side was never produced (e.g. sg_only specs have no mg dir).
    """
    out: dict[str, Path] = {}
    direct = _LEGACY_DIRECT_DIRS.get(spec_name)
    if direct is not None:
        for side, tag in direct.items():
            p = output_root / f"{_LEGACY_DUMP_PREFIX}{tag}"
            if p.is_dir():
                out[side] = p
        return out

    for side in ("sg", "mg"):
        # match both <spec>-<suffix>-{side} (e.g. a1-q-lora-only-sg) and bare
        # <spec>-{side} (e.g. combo-lora-proj-router-sg)
        with_suffix = sorted(output_root.glob(f"{_LEGACY_DUMP_PREFIX}grafter-{spec_name}-*-{side}"))
        bare = output_root / f"{_LEGACY_DUMP_PREFIX}grafter-{spec_name}-{side}"
        if with_suffix:
            out[side] = with_suffix[0]
        elif bare.is_dir():
            out[side] = bare
    return out


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
