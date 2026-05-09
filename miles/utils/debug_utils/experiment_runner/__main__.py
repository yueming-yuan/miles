"""CLI entry point for ``python -m miles.utils.debug_utils.experiment_runner``.

Subcommands:

- ``run --exp <name> --registry-module <pkg.mod>``: execute one experiment in
  the current pod (subprocess-based; assumes sg/mg are launchable here).
- ``compare --target <name> --baselines <a,b,c>``: compute and append comparison
  rows for an existing target run against named baselines.
- ``render --output <md_path>``: re-emit the markdown registry from
  runs.jsonl + comparisons.jsonl.
- ``list --registry-module <pkg.mod>``: print all experiment names registered
  in the module.
- ``dispatch --experiments a,b,c --pods p1,p2 --registry-module <pkg.mod>``:
  run on a pool of rcli-managed pods.

The ``--registry-module`` argument names a Python module that must export
``CANONICAL`` (a CanonicalConfig) and ``EXPERIMENTS`` (a dict[str, ExperimentSpec]).
This is the plugin seam between the framework and project-specific specs.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

from miles.utils.debug_utils.experiment_runner.comparator import compare_logprob_dirs, write_comparison_jsonl
from miles.utils.debug_utils.experiment_runner.dispatcher import dispatch
from miles.utils.debug_utils.experiment_runner.orchestrator import RunnerOptions, run_experiment
from miles.utils.debug_utils.experiment_runner.registry import find_latest_run, load_runs, render_markdown


def _load_registry(module_name: str) -> tuple[Any, dict]:
    """Import a registry module; expect ``CANONICAL`` and ``EXPERIMENTS``."""
    mod = importlib.import_module(module_name)
    if not hasattr(mod, "CANONICAL") or not hasattr(mod, "EXPERIMENTS"):
        raise ValueError(
            f"registry module {module_name!r} must export CANONICAL (CanonicalConfig) "
            "and EXPERIMENTS (dict[str, ExperimentSpec])"
        )
    return mod.CANONICAL, dict(mod.EXPERIMENTS)


def _runner_options_from_args(args: argparse.Namespace) -> RunnerOptions:
    return RunnerOptions(
        runs_jsonl=Path(args.runs_jsonl),
        comparisons_jsonl=Path(args.comparisons_jsonl),
        sg_repo_dir=args.sg_repo_dir,
        mg_repo_dir=args.mg_repo_dir,
        miles_repo_dir=args.miles_repo_dir,
        server_host=args.server_host,
        server_port=args.server_port,
        sg_ready_timeout_s=args.sg_ready_timeout_s,
        sg_request_timeout_s=args.sg_request_timeout_s,
        image=args.image,
        tp=args.tp,
        pp=args.pp,
        cp=args.cp,
        ep=args.ep,
        etp=args.etp,
        batch_size=args.batch_size,
        sp=not args.no_sp,
        model_type=args.model_type,
    )


def _cmd_run(args: argparse.Namespace) -> int:
    canonical, specs = _load_registry(args.registry_module)
    if args.exp not in specs:
        print(f"unknown experiment: {args.exp!r}; known: {sorted(specs)}", file=sys.stderr)
        return 1
    options = _runner_options_from_args(args)
    record = run_experiment(specs[args.exp], canonical, options)
    print(f"[runner] completed: {record.name} (kind={record.kind})", flush=True)
    print(f"[runner] mg_logprob_dir={record.mg_logprob_dir}", flush=True)
    print(f"[runner] sg_baseline_json={record.sg_baseline_json}", flush=True)
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    runs = load_runs(Path(args.runs_jsonl))
    target = find_latest_run(runs, args.target)
    if target is None:
        print(f"target {args.target!r} not in registry", file=sys.stderr)
        return 1

    target_dir = _resolve_logprob_dir(target, prefer_mg=True)
    if target_dir is None:
        print(f"target {target.name} has no logprob dir on either side", file=sys.stderr)
        return 1

    for baseline_name in [b.strip() for b in args.baselines.split(",") if b.strip()]:
        baseline = find_latest_run(runs, baseline_name)
        if baseline is None:
            print(f"baseline {baseline_name!r} not in registry; skipping", file=sys.stderr)
            continue
        baseline_dir = _resolve_logprob_dir(baseline, prefer_mg=False)
        if baseline_dir is None:
            print(f"baseline {baseline_name} has no logprob dir; skipping", file=sys.stderr)
            continue
        cmp = compare_logprob_dirs(
            baseline_dir=baseline_dir,
            target_dir=target_dir,
            prompt_length=args.prompt_length if args.prompt_length > 0 else None,
        )
        write_comparison_jsonl(
            cmp,
            target_run_name=target.name,
            baseline_run_name=baseline_name,
            jsonl_path=Path(args.comparisons_jsonl),
        )
        print(
            f"[compare] {target.name} vs {baseline_name}: "
            f"mean={cmp.all.mean_abs_diff:.4g} max={cmp.all.max_abs_diff:.4g} "
            f"p99={cmp.all.p99_abs_diff:.4g} (n={cmp.all.num_positions})",
            flush=True,
        )
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    md = render_markdown(
        Path(args.runs_jsonl),
        Path(args.comparisons_jsonl),
        metric_range=args.metric_range,
        baseline_order=([b.strip() for b in args.baseline_order.split(",")] if args.baseline_order else None),
    )
    Path(args.output).write_text(md)
    print(f"[render] wrote {args.output}", flush=True)
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    _, specs = _load_registry(args.registry_module)
    for name in sorted(specs):
        spec = specs[name]
        print(f"{name:32s} {spec.kind.value:14s} {spec.description}")
    return 0


def _cmd_dispatch(args: argparse.Namespace) -> int:
    experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
    pods = [p.strip() for p in args.pods.split(",") if p.strip()]
    log_dir = Path(args.dispatch_log_dir) if args.dispatch_log_dir else None
    results = dispatch(
        experiments=experiments,
        pods=pods,
        registry_module=args.registry_module,
        miles_repo_dir=args.miles_repo_dir,
        log_dir=log_dir,
        rcli_binary=args.rcli_binary,
    )
    failures = [r for r in results if r.return_code != 0]
    print(f"[dispatch] {len(results)} runs; {len(failures)} failures", flush=True)
    return 0 if not failures else 1


def _resolve_logprob_dir(record, *, prefer_mg: bool) -> Path | None:
    primary = record.mg_logprob_dir if prefer_mg else record.sg_baseline_json
    fallback = record.sg_baseline_json if prefer_mg else record.mg_logprob_dir
    if primary:
        return Path(primary).parent if primary.endswith(".json") else Path(primary)
    if fallback:
        return Path(fallback).parent if fallback.endswith(".json") else Path(fallback)
    return None


def _add_runner_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runs-jsonl", default="/storage/yueming/experiment_runner/runs.jsonl")
    parser.add_argument("--comparisons-jsonl", default="/storage/yueming/experiment_runner/comparisons.jsonl")
    parser.add_argument("--sg-repo-dir", default="/workspace/sglang")
    parser.add_argument("--mg-repo-dir", default="/workspace/Megatron-LM")
    parser.add_argument("--miles-repo-dir", default="/workspace/miles")
    parser.add_argument("--server-host", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=30000)
    parser.add_argument("--sg-ready-timeout-s", type=int, default=1800)
    parser.add_argument("--sg-request-timeout-s", type=int, default=1800)
    parser.add_argument("--image", default=None)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=8)
    parser.add_argument("--etp", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-sp", action="store_true")
    parser.add_argument("--model-type", default="deepseek-v4-flash")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="experiment_runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="execute one experiment locally")
    p_run.add_argument("--exp", required=True)
    p_run.add_argument("--registry-module", required=True)
    _add_runner_options(p_run)
    p_run.set_defaults(func=_cmd_run)

    p_cmp = sub.add_parser("compare", help="compare an existing run against named baselines")
    p_cmp.add_argument("--target", required=True)
    p_cmp.add_argument("--baselines", required=True, help="comma-separated baseline run names")
    p_cmp.add_argument("--runs-jsonl", default="/storage/yueming/experiment_runner/runs.jsonl")
    p_cmp.add_argument("--comparisons-jsonl", default="/storage/yueming/experiment_runner/comparisons.jsonl")
    p_cmp.add_argument("--prompt-length", type=int, default=0, help="0 disables prompt/response split")
    p_cmp.set_defaults(func=_cmd_compare)

    p_render = sub.add_parser("render", help="re-render the markdown registry")
    p_render.add_argument("--runs-jsonl", default="/storage/yueming/experiment_runner/runs.jsonl")
    p_render.add_argument("--comparisons-jsonl", default="/storage/yueming/experiment_runner/comparisons.jsonl")
    p_render.add_argument("--metric-range", default="all", choices=["all", "prompt", "response"])
    p_render.add_argument("--baseline-order", default=None, help="comma-separated baseline names for column order")
    p_render.add_argument("--output", required=True)
    p_render.set_defaults(func=_cmd_render)

    p_list = sub.add_parser("list", help="list registered experiments")
    p_list.add_argument("--registry-module", required=True)
    p_list.set_defaults(func=_cmd_list)

    p_disp = sub.add_parser("dispatch", help="run experiments concurrently across rcli-managed pods")
    p_disp.add_argument("--experiments", required=True, help="comma-separated experiment names")
    p_disp.add_argument("--pods", required=True, help="comma-separated rcli pod names")
    p_disp.add_argument("--registry-module", required=True)
    p_disp.add_argument("--miles-repo-dir", default="/workspace/miles")
    p_disp.add_argument("--dispatch-log-dir", default=None)
    p_disp.add_argument("--rcli-binary", default="rcli")
    p_disp.set_defaults(func=_cmd_dispatch)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
