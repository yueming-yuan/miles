"""Compose ``run_megatron run`` invocation from a ``RunConfig``.

The megatron side is already a proper CLI app (``miles.utils.debug_utils.run_megatron``).
This module just builds the right command-line invocation -- it does not duplicate any
of that CLI's logic. Spec deltas (env vars, extra flags, source-patcher YAML, dumper
filter, grafter b2t/t2b filters) are translated into the appropriate flags / env vars.

The patcher YAML is materialised to a temporary file at compose time so the same path
appears in the manifest.json for reproducibility.
"""

from __future__ import annotations

import dataclasses
import os
import shlex
import tempfile
from pathlib import Path

from miles.utils.debug_utils.experiment_runner.spec import RunConfig


@dataclasses.dataclass(frozen=True)
class MgLaunchCommand:
    """Composed ``run_megatron run`` command + the env it should be run with."""

    command: str
    """Full shell command. ``cd``s into miles repo so the editable install resolves,
    invokes ``python -m miles.utils.debug_utils.run_megatron run`` with all flags,
    tees stdout+stderr to ``log_path``."""

    env: dict[str, str]
    """Full env (os.environ + run_config.mg_env + grafter env if grafter_pair)."""

    log_path: Path
    """run.log path; ``tee``'d to in the command."""

    output_dir: Path
    """``--output-dir`` value: ``<output_root>/<run_name>-mg``."""

    logprob_output_dir: Path
    """``--logprob-output`` value: ``<output_dir>/megatron_logprobs``. Comparator reads from here."""

    patcher_yaml_path: Path | None
    """If a source-patcher YAML was materialised, this points to the temp file. None otherwise."""


def build_mg_launch_command(
    run_config: RunConfig,
    *,
    miles_repo_dir: str = "/workspace/miles",
    tp: int = 8,
    pp: int = 1,
    cp: int = 1,
    ep: int = 8,
    etp: int = 1,
    batch_size: int = 1,
    sp: bool = True,
    model_type: str = "deepseek-v4-flash",
    output_subdir: str = "mg",
    patcher_yaml_dir: Path | None = None,
) -> MgLaunchCommand:
    """Compose the full ``run_megatron run`` command for ``run_config``.

    Routes ``run_config``'s fields to the CLI flags:
    - ``mg_hf_checkpoint`` -> ``--hf-checkpoint``
    - ``mg_ref_load`` -> ``--ref-load``
    - ``rollout_data_path`` -> ``--rollout-data`` (mg_only and grafter_pair both consume the synthetic rollout)
    - ``mg_extra_patches_yaml`` -> materialised to a temp file -> ``--source-patcher-config``
    - ``dumper_filter`` -> ``--dumper-filter``
    - ``mg_extra_args`` -> ``--extra-args`` (megatron-native flags)
    - ``mg_run_args`` -> appended verbatim (escape hatch for spec-specific flags)

    Parallel sizes (tp/pp/cp/ep/etp/sp/batch_size/model_type) are taken from kwargs --
    these are call-site decisions, not per-spec deltas. Override per call when porting
    older single-rank or different-parallel runs.
    """
    output_dir = Path(run_config.output_root) / f"{run_config.name}-{output_subdir}"
    log_path = output_dir / "run.log"
    logprob_output_dir = output_dir / "megatron_logprobs"

    env: dict[str, str] = {**os.environ, **run_config.mg_env}
    if run_config.kind.value == "grafter_pair":
        env.update(run_config.grafter_env)
        env["DUMPER_GRAFTER_ENABLE"] = "1"
        env["DUMPER_GRAFTER_ROLE"] = "baseline"
        if run_config.grafter_b2t_filter:
            env["DUMPER_GRAFTER_B2T_FILTER"] = run_config.grafter_b2t_filter
        if run_config.grafter_t2b_filter:
            env["DUMPER_GRAFTER_T2B_FILTER"] = run_config.grafter_t2b_filter

    patcher_yaml_path: Path | None = None
    if run_config.mg_extra_patches_yaml is not None:
        if patcher_yaml_dir is None:
            patcher_yaml_dir = Path(tempfile.gettempdir())
        patcher_yaml_dir.mkdir(parents=True, exist_ok=True)
        patcher_yaml_path = patcher_yaml_dir / f"{run_config.name}_mg_patcher.yaml"
        patcher_yaml_path.write_text(run_config.mg_extra_patches_yaml)

    flags: list[str] = [
        "--model-type",
        model_type,
        "--tp",
        str(tp),
        "--pp",
        str(pp),
        "--cp",
        str(cp),
        "--ep",
        str(ep),
        "--etp",
        str(etp),
        "--batch-size",
        str(batch_size),
        "--output-dir",
        str(output_dir),
        "--logprob-output",
        str(logprob_output_dir),
    ]
    if sp:
        flags.append("--sp")
    if run_config.mg_hf_checkpoint:
        flags += ["--hf-checkpoint", run_config.mg_hf_checkpoint]
    if run_config.mg_ref_load:
        flags += ["--ref-load", run_config.mg_ref_load]
    if run_config.rollout_data_path:
        flags += ["--rollout-data", run_config.rollout_data_path]
    if patcher_yaml_path is not None:
        flags += ["--source-patcher-config", str(patcher_yaml_path)]
    if run_config.dumper_filter:
        flags += ["--dumper-filter", run_config.dumper_filter]
    if run_config.mg_extra_args:
        flags += ["--extra-args", run_config.mg_extra_args]
    flags += list(run_config.mg_run_args)

    flags_quoted = " ".join(shlex.quote(f) for f in flags)
    cmd = (
        f"mkdir -p {shlex.quote(str(output_dir))} && "
        f"cd {shlex.quote(miles_repo_dir)} && "
        f"python -m miles.utils.debug_utils.run_megatron run {flags_quoted} 2>&1 | "
        f"tee {shlex.quote(str(log_path))}"
    )

    return MgLaunchCommand(
        command=cmd,
        env=env,
        log_path=log_path,
        output_dir=output_dir,
        logprob_output_dir=logprob_output_dir,
        patcher_yaml_path=patcher_yaml_path,
    )
