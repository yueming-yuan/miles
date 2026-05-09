"""V4 RL divergence debug experiments.

Defines:
- ``CANONICAL``: the V4 baseline launch configuration (env vars, sglang server
  args, megatron worker args, paths) all experiments inherit from. Single source
  of truth.
- ``EXPERIMENTS``: dict of named ExperimentSpec entries, each declaring only the
  delta vs ``CANONICAL``.

Use via:
    cd /workspace/miles
    python -m miles.utils.debug_utils.experiment_runner run \\
        --exp <name> --registry-module tools_debug.experiments
"""

from tools_debug.experiments.canonical import CANONICAL
from tools_debug.experiments.registry import EXPERIMENTS

__all__ = ["CANONICAL", "EXPERIMENTS"]
