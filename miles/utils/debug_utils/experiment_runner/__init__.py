"""Experiment runner: a unified, plugin-based framework for cross-stack debug experiments.

Each experiment is declared as an ``ExperimentSpec`` (delta vs a ``CanonicalConfig``).
A ``Runner`` composes the canonical config with the spec's deltas, launches the required
sides (sglang HTTP server, megatron standalone run, both with grafter rendezvous), runs
a comparator against named baselines, and records results in a JSONL registry.

Public API:

- ``ExperimentSpec``        -- per-experiment delta declaration
- ``CanonicalConfig``       -- abstract base for project-specific canonical configs
- ``RunKind``               -- enum: ``sg_only`` / ``mg_only`` / ``grafter_pair``
- ``compose_run_config``    -- pure function: spec + canonical -> RunConfig
"""

from miles.utils.debug_utils.experiment_runner.spec import (
    CanonicalConfig,
    ExperimentSpec,
    RunConfig,
    RunKind,
    compose_run_config,
)

__all__ = [
    "CanonicalConfig",
    "ExperimentSpec",
    "RunConfig",
    "RunKind",
    "compose_run_config",
]
