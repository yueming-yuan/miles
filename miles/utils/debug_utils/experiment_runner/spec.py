"""ExperimentSpec, CanonicalConfig and the pure composition function.

These dataclasses are the framework's plugin interface. A project (e.g. V4 RL divergence
debug) defines one ``CanonicalConfig`` instance describing the baseline launch (env vars,
sglang server args, megatron worker args, ...) and a registry of ``ExperimentSpec``
instances each declaring only the deltas to apply for that experiment.

``compose_run_config`` deterministically merges a canonical config with a spec into a
``RunConfig``. The merge is a pure function -- no side effects -- so unit tests can
verify exact env/args composition without touching infra.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Mapping


class RunKind(str, enum.Enum):
    """Topology of an experiment run.

    - ``sg_only``: launch sglang HTTP server and trigger one forward; collect
      per-token logprobs from the HTTP response. No megatron side.
    - ``mg_only``: launch megatron standalone via ``run_megatron run --rollout-data``;
      collect per-token logprobs from ``rank_*.json``. No sglang side.
    - ``grafter_pair``: launch both sides; bidirectional dumper-grafter rendezvous
      injects activations across stacks per the spec's b2t/t2b filters.
    """

    SG_ONLY = "sg_only"
    MG_ONLY = "mg_only"
    GRAFTER_PAIR = "grafter_pair"


@dataclasses.dataclass(frozen=True)
class CanonicalConfig:
    """Project-defined baseline configuration. Single source of truth.

    Every experiment inherits from this; ``ExperimentSpec`` declares only deltas. To
    define a project-specific canonical config, instantiate this dataclass with the
    project's required env / args / paths.

    Fields are intentionally minimal and infrastructure-agnostic. Project-specific
    paths (model, rollout, output root) live here so ablation specs do not have to
    repeat them.
    """

    name: str
    """Identifier used in run-registry rows that reference this canonical config."""

    sg_env: Mapping[str, str] = dataclasses.field(default_factory=dict)
    """Environment variables exported before launching the sglang server."""

    sg_server_args: tuple[str, ...] = ()
    """Arguments passed to ``python -m sglang.launch_server``."""

    mg_env: Mapping[str, str] = dataclasses.field(default_factory=dict)
    """Environment variables exported before invoking ``run_megatron``."""

    mg_run_args: tuple[str, ...] = ()
    """Arguments passed to ``python -m miles.utils.debug_utils.run_megatron run``."""

    mg_extra_args: str = ""
    """Value of ``--extra-args`` (passed through to the worker as megatron flags)."""

    grafter_env: Mapping[str, str] = dataclasses.field(default_factory=dict)
    """Common DUMPER_GRAFTER_* env vars (master addr/port, world sizes, transform path)."""

    sg_model_path: str | None = None
    """``--model-path`` for sglang. Spec can override via ``sg_server_args``."""

    mg_hf_checkpoint: str | None = None
    """``--hf-checkpoint`` for run_megatron. Spec can override via ``mg_run_args``."""

    mg_ref_load: str | None = None
    """``--ref-load`` for run_megatron."""

    rollout_data_path: str | None = None
    """``--rollout-data`` synthetic rollout pt; consumed by mg_only and grafter_pair runs."""

    sg_baseline_response_path: str | None = None
    """JSON path with sg's per-token logprobs serving as the cross-stack baseline."""

    output_root: str = "/tmp/experiment_runner"
    """Each run's dump dirs live under ``<output_root>/<exp_name>-{sg,mg}/``."""


@dataclasses.dataclass(frozen=True)
class ExperimentSpec:
    """Declarative description of one experiment as a delta vs a canonical config.

    All ``*_env`` and ``*_args_extra`` fields are *additive* on top of canonical:
    env dicts merge (spec keys override canonical keys), arg tuples concatenate.

    To remove or disable a fix that the canonical sets, override the env var to
    ``"0"`` (the codebase reads fix toggles via ``os.environ.get(VAR, "0") == "1"``,
    so ``"0"`` and unset are equivalent).
    """

    name: str
    """Stable identifier; appears in dump dir names and registry rows."""

    description: str
    """Short human-readable summary of what this experiment changes."""

    kind: RunKind
    """Topology -- see ``RunKind``."""

    sg_env: Mapping[str, str] = dataclasses.field(default_factory=dict)
    """Delta sglang env vars merged onto canonical."""

    sg_server_args_extra: tuple[str, ...] = ()
    """Extra sglang server args appended after canonical args."""

    mg_env: Mapping[str, str] = dataclasses.field(default_factory=dict)
    """Delta megatron env vars."""

    mg_run_args_extra: tuple[str, ...] = ()
    """Extra ``run_megatron run`` flags appended after canonical args."""

    mg_extra_args_append: str = ""
    """Tokens appended (space-joined) to canonical ``mg_extra_args``."""

    sg_extra_patches_yaml: str | None = None
    """Source-patcher YAML (sglang side) merged with canonical patches at run time."""

    mg_extra_patches_yaml: str | None = None
    """Source-patcher YAML (megatron side) merged with canonical patches."""

    grafter_b2t_filter: str | None = None
    """Filter expression for baseline->target graft. Required when kind=grafter_pair."""

    grafter_t2b_filter: str | None = None
    """Filter expression for target->baseline graft. Required when kind=grafter_pair."""

    dumper_filter: str | None = None
    """Optional ``DUMPER_FILTER`` expression scoping which dump points are recorded."""

    baselines: tuple[str, ...] = ()
    """Names of prior runs to compare against after this run completes.

    The runner looks each name up in the registry, uses the recorded sglang or megatron
    logprob dir as comparator baseline, and emits one ``comparisons.jsonl`` row per pair.
    Adding more baselines does not require re-running this experiment -- a separate
    ``compare`` subcommand can compute new pairs on demand.
    """

    def __post_init__(self) -> None:
        if self.kind == RunKind.GRAFTER_PAIR:
            if self.grafter_b2t_filter is None and self.grafter_t2b_filter is None:
                raise ValueError(
                    f"spec {self.name!r}: kind=grafter_pair requires at least one of "
                    "grafter_b2t_filter or grafter_t2b_filter"
                )


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """Final, fully-resolved configuration produced by ``compose_run_config``.

    Carries everything a launcher needs; no further reference to the spec or canonical
    is required. Serialised verbatim into each run's ``manifest.json`` so reruns are
    bit-for-bit reproducible (modulo intrinsic kernel non-determinism).
    """

    name: str
    description: str
    kind: RunKind
    canonical_name: str

    sg_env: Mapping[str, str]
    sg_server_args: tuple[str, ...]
    sg_model_path: str | None

    mg_env: Mapping[str, str]
    mg_run_args: tuple[str, ...]
    mg_extra_args: str
    mg_hf_checkpoint: str | None
    mg_ref_load: str | None

    grafter_env: Mapping[str, str]
    grafter_b2t_filter: str | None
    grafter_t2b_filter: str | None

    sg_extra_patches_yaml: str | None
    mg_extra_patches_yaml: str | None

    dumper_filter: str | None
    rollout_data_path: str | None
    sg_baseline_response_path: str | None

    output_root: str
    baselines: tuple[str, ...]


def compose_run_config(canonical: CanonicalConfig, spec: ExperimentSpec) -> RunConfig:
    """Merge a canonical config with an experiment spec into a ``RunConfig``.

    Pure function. Does not read environment, files, or network.

    Merge rules:
    - env dicts: ``{**canonical, **spec}`` -- spec keys override canonical keys.
    - arg tuples: concatenate ``canonical + spec`` (canonical first; spec extras appended).
    - ``mg_extra_args``: space-join ``canonical.mg_extra_args`` and
      ``spec.mg_extra_args_append`` (drops empty strings).
    """
    return RunConfig(
        name=spec.name,
        description=spec.description,
        kind=spec.kind,
        canonical_name=canonical.name,
        sg_env=_merge_env(canonical.sg_env, spec.sg_env),
        sg_server_args=tuple(canonical.sg_server_args) + tuple(spec.sg_server_args_extra),
        sg_model_path=canonical.sg_model_path,
        mg_env=_merge_env(canonical.mg_env, spec.mg_env),
        mg_run_args=tuple(canonical.mg_run_args) + tuple(spec.mg_run_args_extra),
        mg_extra_args=_join_nonempty(canonical.mg_extra_args, spec.mg_extra_args_append),
        mg_hf_checkpoint=canonical.mg_hf_checkpoint,
        mg_ref_load=canonical.mg_ref_load,
        grafter_env=dict(canonical.grafter_env),
        grafter_b2t_filter=spec.grafter_b2t_filter,
        grafter_t2b_filter=spec.grafter_t2b_filter,
        sg_extra_patches_yaml=spec.sg_extra_patches_yaml,
        mg_extra_patches_yaml=spec.mg_extra_patches_yaml,
        dumper_filter=spec.dumper_filter,
        rollout_data_path=canonical.rollout_data_path,
        sg_baseline_response_path=canonical.sg_baseline_response_path,
        output_root=canonical.output_root,
        baselines=tuple(spec.baselines),
    )


def _merge_env(base: Mapping[str, str], override: Mapping[str, str]) -> dict[str, str]:
    return {**base, **override}


def _join_nonempty(*parts: str) -> str:
    return " ".join(p for p in parts if p)
