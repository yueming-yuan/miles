# experiment_runner

Plugin-based framework for cross-stack debug experiments (sglang inference vs.
megatron training). Each experiment is a single declarative `ExperimentSpec`
defining only the deltas vs. a project's `CanonicalConfig`. The runner composes
both into a `RunConfig`, executes the right subset (`sg_only` / `mg_only` /
`grafter_pair`), runs comparisons against named baselines, and persists results
to an append-only JSONL registry.

## Workflow

1. **Define a project's canonical config + experiments.** A project ships a
   Python module exporting `CANONICAL` (a `CanonicalConfig`) and `EXPERIMENTS`
   (a `dict[str, ExperimentSpec]`). For the V4 RL divergence work see
   `tools_debug/experiments/`.

2. **Run one experiment in a pod.** From inside the pod:

   ```
   cd /workspace/miles
   python -m miles.utils.debug_utils.experiment_runner run \
       --exp a1 \
       --registry-module tools_debug.experiments \
       --runs-jsonl /storage/yueming/experiment_runner/runs.jsonl \
       --comparisons-jsonl /storage/yueming/experiment_runner/comparisons.jsonl
   ```

   The runner starts an sglang server (or attaches to one already on
   `0.0.0.0:30000`), launches `run_megatron run`, drives the HTTP trigger,
   writes `manifest.json`, appends one row to `runs.jsonl`, then computes
   comparisons against every name in the spec's `baselines`.

3. **Compare an existing run vs. another baseline.** No re-running required:

   ```
   python -m miles.utils.debug_utils.experiment_runner compare \
       --target a1 --baselines sg-natural-prefill-e8env,mg-fix0505 \
       --runs-jsonl ... --comparisons-jsonl ...
   ```

4. **Re-render the markdown registry.**

   ```
   python -m miles.utils.debug_utils.experiment_runner render \
       --output /Users/.../docs/v4_divergence/auto-registry.md \
       --runs-jsonl ... --comparisons-jsonl ... \
       --baseline-order sg-natural-prefill-e8env,mg-fix0505
   ```

5. **Dispatch many experiments across multiple pods.** From the laptop, after
   `rcli job new --name yueming-v4-debug-{1,2,3}` is up:

   ```
   python -m miles.utils.debug_utils.experiment_runner dispatch \
       --experiments a1,a2,a3,a4,a5b,m1,combo-lora-proj-router \
       --pods yueming-v4-debug-1,yueming-v4-debug-2,yueming-v4-debug-3 \
       --registry-module tools_debug.experiments
   ```

## Plugin contract

A registry module must export:

```python
CANONICAL: CanonicalConfig    # one instance, the project's baseline
EXPERIMENTS: dict[str, ExperimentSpec]
```

Adding a new experiment is one dict entry:

```python
EXPERIMENTS["my_ablation"] = ExperimentSpec(
    name="my_ablation",
    description="brief one-line description",
    kind=RunKind.SG_ONLY,
    sg_env={"SGLANG_DSV4_FIX_0506": "0"},  # delta only; canonical fills the rest
    baselines=("sg-natural-prefill-e8env",),
)
```

To remove a fix the canonical sets, override the env var to `"0"` -- the
codebase reads fix toggles via `os.environ.get(VAR, "0") == "1"`, so `"0"` and
unset are equivalent. There is no `UNSET` sentinel.

## Reproducibility

Every run writes `<output_dir>/manifest.json` capturing:

- spec snapshot (the full ExperimentSpec dict at run time)
- run_config snapshot (the resolved RunConfig)
- git HEAD SHAs of the synced repos (sglang, miles, megatron)
- container image SHA
- start/end timestamps

A future rerun can `git checkout <sha>` each repo, pull the same image SHA, and
expect identical dump bytes (modulo intrinsic kernel non-determinism, which the
sglang run-vs-run mean is ~1.78e-3 nats per the V4 measurements).

The registry is append-only: rerunning a spec adds a fresh row. The renderer
uses the latest revision (newest `finished_at`).

## Bootstrap of legacy dumps

For projects migrating from per-experiment shell-scripts: a project-specific
``bootstrap`` module can register existing dump dirs into ``runs.jsonl``
without re-running. See ``tools_debug/experiments/bootstrap.py``.

## Module layout

| Module | Role |
|---|---|
| `spec.py` | `ExperimentSpec`, `CanonicalConfig`, `RunKind`, `compose_run_config` |
| `sg_trigger.py` | Programmatic sglang HTTP trigger + JSON conversion |
| `sg_launcher.py` | sglang server command composition + context-manager-managed lifecycle |
| `mg_launcher.py` | `run_megatron run` invocation composition |
| `comparator.py` | Multi-range logprob diff metrics + `comparisons.jsonl` writer |
| `registry.py` | `runs.jsonl` reader/writer + markdown renderer |
| `manifest.py` | Per-run `manifest.json` (git SHAs, image SHA) |
| `orchestrator.py` | `run_experiment(spec, canonical, options)` -- top-level entry |
| `dispatcher.py` | Multi-pod parallel scheduling via `rcli exec` |
| `__main__.py` | CLI: `run` / `compare` / `render` / `list` / `dispatch` |
