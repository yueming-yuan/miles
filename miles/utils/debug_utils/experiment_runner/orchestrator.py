"""High-level orchestrator: spec name -> dump dirs -> registry rows.

Composes spec + canonical, builds sg/mg launch commands, executes the right
subset for each ``RunKind``, runs comparisons against declared baselines, and
appends rows to ``runs.jsonl`` + ``comparisons.jsonl``.

Designed to run *inside* the target pod -- subprocess-based, no rcli. The
dispatcher handles cross-pod scheduling separately by invoking
``python -m miles.utils.debug_utils.experiment_runner run --exp <name>`` on each
pod over rcli.

For ``RunKind.GRAFTER_PAIR``, sg server + HTTP trigger + mg run must be active
concurrently so the gloo rendezvous matches per-layer dumps. Implemented as:
1. Start sg server (background), wait for /port-bind ready.
2. Start the HTTP trigger (background thread, blocks on prefill response).
3. Start the mg run (foreground subprocess, blocks until forward completes).
4. Join the trigger thread, save sg's per-token logprobs.
5. Run comparisons.
"""

from __future__ import annotations

import dataclasses
import datetime
import os
import subprocess
import threading
from pathlib import Path
from typing import Any

import torch

from miles.utils.debug_utils.experiment_runner.comparator import (
    LogprobComparison,
    compare_logprob_dirs,
    write_comparison_jsonl,
)
from miles.utils.debug_utils.experiment_runner.manifest import GitInfo, write_manifest
from miles.utils.debug_utils.experiment_runner.mg_launcher import build_mg_launch_command
from miles.utils.debug_utils.experiment_runner.registry import RunRecord, append_run, find_latest_run, load_runs
from miles.utils.debug_utils.experiment_runner.sg_launcher import build_sg_launch_command, probe_sg_ready
from miles.utils.debug_utils.experiment_runner.sg_trigger import (
    trigger_sg_generate,
    wait_for_sg_ready,
    write_baseline_logprob_json,
)
from miles.utils.debug_utils.experiment_runner.spec import (
    CanonicalConfig,
    ExperimentSpec,
    RunConfig,
    RunKind,
    compose_run_config,
)


@dataclasses.dataclass(frozen=True)
class RunnerOptions:
    """Runtime options that vary by call-site, not by spec."""

    runs_jsonl: Path
    comparisons_jsonl: Path
    sg_repo_dir: str = "/workspace/sglang"
    mg_repo_dir: str = "/workspace/Megatron-LM"
    miles_repo_dir: str = "/workspace/miles"
    server_host: str = "0.0.0.0"
    server_port: int = 30000
    sg_ready_timeout_s: int = 300
    """Cap on sglang server startup. ~3-4min normal cold-load on V4-Flash; if
    it exceeds 5min the server is hung (image, weight load, or kernel JIT)."""
    sg_request_timeout_s: int = 900
    mg_run_timeout_s: int = 1800
    """Hard wall-clock cap for the mg subprocess. 30min covers ~15min forward +
    grafter rendezvous (DUMPER_GRAFTER_TIMEOUT=600s gives gloo 10min per stuck
    op before raising). If the cap is hit the subprocess group is SIGKILLed
    so torchrun children do not leak GPU memory."""
    image: str | None = None
    tp: int = 8
    pp: int = 1
    cp: int = 1
    ep: int = 8
    etp: int = 1
    batch_size: int = 1
    sp: bool = True
    model_type: str = "deepseek-v4-flash"


@dataclasses.dataclass
class _RunOutputs:
    sg_dump_dir: Path | None = None
    sg_baseline_json: Path | None = None
    mg_dump_dir: Path | None = None
    mg_logprob_dir: Path | None = None


def run_experiment(
    spec: ExperimentSpec,
    canonical: CanonicalConfig,
    options: RunnerOptions,
) -> RunRecord:
    """Execute one experiment end-to-end. Returns the registry row that was appended."""
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    run_config = compose_run_config(canonical, spec)
    outputs = _execute(run_config, options)

    finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    git_infos = _collect_git_infos(options)
    primary_output_dir = _primary_output_dir(run_config, outputs)
    write_manifest(
        output_dir=primary_output_dir,
        run_config_snapshot=_run_config_to_dict(run_config),
        spec_snapshot=_spec_to_dict(spec),
        git_infos=git_infos,
        image=options.image,
        started_at=started_at,
        finished_at=finished_at,
    )

    record = RunRecord(
        name=run_config.name,
        description=run_config.description,
        kind=run_config.kind.value,
        canonical_name=run_config.canonical_name,
        started_at=started_at,
        finished_at=finished_at,
        sg_dump_dir=str(outputs.sg_dump_dir) if outputs.sg_dump_dir else None,
        sg_baseline_json=str(outputs.sg_baseline_json) if outputs.sg_baseline_json else None,
        mg_dump_dir=str(outputs.mg_dump_dir) if outputs.mg_dump_dir else None,
        mg_logprob_dir=str(outputs.mg_logprob_dir) if outputs.mg_logprob_dir else None,
        git={k: g.head_sha for k, g in git_infos.items()},
        image=options.image,
        spec_snapshot=_spec_to_dict(spec),
        run_config_snapshot=_run_config_to_dict(run_config),
    )
    append_run(record, jsonl_path=options.runs_jsonl)

    _run_baseline_comparisons(record, run_config, outputs, options)
    return record


def _execute(run_config: RunConfig, options: RunnerOptions) -> _RunOutputs:
    if run_config.kind == RunKind.SG_ONLY:
        return _execute_sg_only(run_config, options)
    if run_config.kind == RunKind.MG_ONLY:
        return _execute_mg_only(run_config, options)
    if run_config.kind == RunKind.GRAFTER_PAIR:
        return _execute_grafter_pair(run_config, options)
    raise ValueError(f"unknown kind: {run_config.kind}")


def _execute_sg_only(run_config: RunConfig, options: RunnerOptions) -> _RunOutputs:
    sg_launch = build_sg_launch_command(
        run_config,
        sg_repo_dir=options.sg_repo_dir,
        miles_repo_dir=options.miles_repo_dir,
        server_host=options.server_host,
        server_port=options.server_port,
    )
    sg_proc = _start_sg_or_attach(sg_launch, options)
    try:
        meta = _trigger_sg(run_config, options)
        baseline_json = sg_launch.log_path.parent / "sg_baseline.json"
        write_baseline_logprob_json(meta_info=meta, output_path=baseline_json)
    finally:
        if sg_proc is not None:
            pass  # leave running for pod-level reuse

    return _RunOutputs(
        sg_dump_dir=sg_launch.log_path.parent,
        sg_baseline_json=baseline_json,
    )


def _execute_mg_only(run_config: RunConfig, options: RunnerOptions) -> _RunOutputs:
    mg_launch = build_mg_launch_command(
        run_config,
        miles_repo_dir=options.miles_repo_dir,
        tp=options.tp,
        pp=options.pp,
        cp=options.cp,
        ep=options.ep,
        etp=options.etp,
        batch_size=options.batch_size,
        sp=options.sp,
        model_type=options.model_type,
    )
    _run_subprocess_blocking(mg_launch.command, mg_launch.env, log_label="mg", timeout_s=options.mg_run_timeout_s)

    return _RunOutputs(
        mg_dump_dir=mg_launch.output_dir,
        mg_logprob_dir=mg_launch.logprob_output_dir,
    )


def _execute_grafter_pair(run_config: RunConfig, options: RunnerOptions) -> _RunOutputs:
    sg_launch = build_sg_launch_command(
        run_config,
        sg_repo_dir=options.sg_repo_dir,
        miles_repo_dir=options.miles_repo_dir,
        server_host=options.server_host,
        server_port=options.server_port,
    )
    mg_launch = build_mg_launch_command(
        run_config,
        miles_repo_dir=options.miles_repo_dir,
        tp=options.tp,
        pp=options.pp,
        cp=options.cp,
        ep=options.ep,
        etp=options.etp,
        batch_size=options.batch_size,
        sp=options.sp,
        model_type=options.model_type,
    )
    _start_sg_or_attach(sg_launch, options)

    trigger_meta_holder: dict[str, Any] = {}

    def _trigger_thread() -> None:
        trigger_meta_holder["meta"] = _trigger_sg(run_config, options)

    t = threading.Thread(target=_trigger_thread, daemon=True)
    t.start()

    _run_subprocess_blocking(mg_launch.command, mg_launch.env, log_label="mg", timeout_s=options.mg_run_timeout_s)
    t.join(timeout=options.sg_request_timeout_s)
    if "meta" not in trigger_meta_holder:
        raise RuntimeError("sg trigger thread did not return; check sg server log")

    baseline_json = sg_launch.log_path.parent / "sg_baseline.json"
    write_baseline_logprob_json(meta_info=trigger_meta_holder["meta"], output_path=baseline_json)

    return _RunOutputs(
        sg_dump_dir=sg_launch.log_path.parent,
        sg_baseline_json=baseline_json,
        mg_dump_dir=mg_launch.output_dir,
        mg_logprob_dir=mg_launch.logprob_output_dir,
    )


def _start_sg_or_attach(sg_launch, options: RunnerOptions) -> subprocess.Popen | None:
    if probe_sg_ready(host=options.server_host, port=options.server_port):
        return None
    sg_launch.log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["bash", "-c", sg_launch.command],
        env=sg_launch.env,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    wait_for_sg_ready(host=options.server_host, port=options.server_port, timeout_s=options.sg_ready_timeout_s)
    return proc


def _trigger_sg(run_config: RunConfig, options: RunnerOptions) -> dict[str, Any]:
    if run_config.rollout_data_path is None:
        raise ValueError(f"spec {run_config.name}: rollout_data_path required for sg trigger")
    sample = _load_first_sample_tokens(Path(run_config.rollout_data_path))
    server_url = f"http://{options.server_host}:{options.server_port}"
    result = trigger_sg_generate(
        input_token_ids=sample,
        server_url=server_url,
        timeout_s=options.sg_request_timeout_s,
    )
    return result.meta_info


def _load_first_sample_tokens(rollout_path: Path) -> list[int]:
    """Load sample 0's token list from a synthetic rollout pt.

    The synthetic rollout produced by ``tools_debug/build_synthetic_rollout.py``
    has shape ``{samples: [{tokens: [...], response_length: int, ...}, ...]}``.
    """
    data = torch.load(rollout_path, weights_only=False)
    s0 = data["samples"][0]
    toks = s0["tokens"]
    return list(toks) if not isinstance(toks, list) else toks


def _run_subprocess_blocking(
    command: str,
    env: dict[str, str],
    *,
    log_label: str,
    timeout_s: int | None = None,
) -> None:
    """Run a foreground subprocess; raise if it exits non-zero or runs past the cap.

    On timeout the process group is killed (start_new_session=True so the bash
    wrapper + python + torchrun children all receive SIGKILL) so the run fails
    fast rather than leaving zombies that eat GPU memory.
    """
    proc = subprocess.Popen(["bash", "-c", command], env=env, start_new_session=True)
    try:
        rc = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, 9)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        raise RuntimeError(f"{log_label} subprocess exceeded timeout of {timeout_s}s; killed")
    if rc != 0:
        raise RuntimeError(f"{log_label} subprocess exited with rc={rc}")


def _collect_git_infos(options: RunnerOptions) -> dict[str, GitInfo]:
    from miles.utils.debug_utils.experiment_runner.manifest import collect_git

    out: dict[str, GitInfo] = {}
    for label, repo in (
        ("sglang", Path(options.sg_repo_dir)),
        ("miles", Path(options.miles_repo_dir)),
        ("megatron", Path(options.mg_repo_dir)),
    ):
        info = collect_git(repo)
        if info is not None:
            out[label] = info
    return out


def _primary_output_dir(run_config: RunConfig, outputs: _RunOutputs) -> Path:
    if outputs.mg_dump_dir is not None:
        return outputs.mg_dump_dir
    if outputs.sg_dump_dir is not None:
        return outputs.sg_dump_dir
    return Path(run_config.output_root) / run_config.name


def _spec_to_dict(spec: ExperimentSpec) -> dict:
    d = dataclasses.asdict(spec)
    d["kind"] = spec.kind.value
    return d


def _run_config_to_dict(run_config: RunConfig) -> dict:
    d = dataclasses.asdict(run_config)
    d["kind"] = run_config.kind.value
    return d


def _run_baseline_comparisons(
    target: RunRecord,
    run_config: RunConfig,
    outputs: _RunOutputs,
    options: RunnerOptions,
) -> None:
    if not run_config.baselines:
        return
    runs = load_runs(options.runs_jsonl)
    for baseline_name in run_config.baselines:
        baseline = find_latest_run(runs, baseline_name)
        if baseline is None:
            print(f"[runner] WARN: baseline {baseline_name!r} not in registry; skipping comparison", flush=True)
            continue
        comparison = _compare_pair(target=target, baseline=baseline, outputs=outputs)
        if comparison is None:
            print(
                f"[runner] WARN: cannot compare target={target.name} vs baseline={baseline_name} "
                f"(missing logprob dir on one side); skipping",
                flush=True,
            )
            continue
        write_comparison_jsonl(
            comparison,
            target_run_name=target.name,
            baseline_run_name=baseline_name,
            jsonl_path=options.comparisons_jsonl,
        )


def _compare_pair(
    *,
    target: RunRecord,
    baseline: RunRecord,
    outputs: _RunOutputs,
) -> LogprobComparison | None:
    """Pick logprob dirs from each side and run the comparator.

    Logprob source priority: mg_logprob_dir > sg_baseline_json's parent. Both
    runs must contribute one logprob source for a comparison to make sense --
    the target's mg side is preferred (more positions); the baseline's sg side
    is preferred when the baseline is an sg_only run.
    """
    target_dir = _pick_logprob_dir(target, outputs)
    baseline_dir = _pick_baseline_logprob_dir(baseline)
    if target_dir is None or baseline_dir is None:
        return None
    return compare_logprob_dirs(baseline_dir=baseline_dir, target_dir=target_dir)


def _pick_logprob_dir(target: RunRecord, outputs: _RunOutputs) -> Path | None:
    if outputs.mg_logprob_dir is not None:
        return outputs.mg_logprob_dir
    if target.mg_logprob_dir is not None:
        return Path(target.mg_logprob_dir)
    if outputs.sg_baseline_json is not None:
        return outputs.sg_baseline_json.parent
    return None


def _pick_baseline_logprob_dir(baseline: RunRecord) -> Path | None:
    if baseline.sg_baseline_json is not None:
        return Path(baseline.sg_baseline_json).parent
    if baseline.mg_logprob_dir is not None:
        return Path(baseline.mg_logprob_dir)
    return None
