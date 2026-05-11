"""V4 canonical configuration -- baseline launch for all V4 divergence experiments.

Reflects the ``sg-natural-prefill-e8env`` reference run. Mirrors the env vars and
server / worker args the original tools_debug/launch_*.sh scripts set verbatim.

To disable a fix this canonical sets, an ablation spec overrides the env var to
``"0"`` (the codebase reads fix toggles via ``os.environ.get(VAR, "0") == "1"``).
"""

from miles.utils.debug_utils.experiment_runner import CanonicalConfig

CANONICAL = CanonicalConfig(
    name="v4_e8env",
    sg_env={
        "SGLANG_SKIP_CHECKPOINT_LOAD_CHECK": "1",
        "SGL_JIT_DEEPGEMM_PRECOMPILE": "false",
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
        "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
        "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false",
        "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
        "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
        "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
        "SGLANG_DSV4_MODE": "2604",
        "SGLANG_FIX_MTP_HC_HIDDEN": "1",
        "SGLANG_OPT_DEEPGEMM_SCALE_CONVERT_AT_INIT": "1",
        "SGLANG_DSV4_FP4_EXPERTS": "0",
        "SGLANG_DSV4_FIX_0506": "1",
    },
    sg_server_args=(
        "--trust-remote-code",
        "--random-seed",
        "0",
        "--nnodes",
        "1",
        "--node-rank",
        "0",
        "--tp-size",
        "8",
        "--dp-size",
        "8",
        "--pp-size",
        "1",
        "--ep-size",
        "8",
        "--skip-server-warmup",
        "--enable-draft-weights-cpu-backup",
        "--enable-dp-attention",
        "--attention-backend",
        "compressed",
        "--page-size",
        "256",
        "--max-running-requests",
        "8",
        "--chunked-prefill-size",
        "32768",
        "--weight-loader-drop-cache-after-load",
        "--mem-fraction-static",
        "0.35",
        "--disable-cuda-graph",
        "--watchdog-timeout",
        "1800",
    ),
    mg_env={
        "MILES_DSV4_FIX_0505": "1",
    },
    mg_extra_args="--no-load-optim --no-load-rng",
    grafter_env={
        "DUMPER_GRAFTER_BASELINE_WORLD_SIZE": "8",
        "DUMPER_GRAFTER_TARGET_WORLD_SIZE": "8",
        "DUMPER_GRAFTER_TRANSFORM_PATH": "tools_debug.grafter_transforms.transform",
        "DUMPER_GRAFTER_BACKEND": "gloo",
        "DUMPER_GRAFTER_TIMEOUT": "600",
        "DUMPER_GRAFTER_MASTER_ADDRESS": "127.0.0.1",
        "DUMPER_GRAFTER_MASTER_PORT": "29501",
    },
    sg_model_path="/storage/yueming/iter59-hf-fp8",
    mg_hf_checkpoint="/storage/models/sgl-project/DeepSeek-V4-Flash-FP8",
    mg_ref_load="/storage/yueming/saves/training-jobs/260501-gsm8k-seqlen8192/checkpoints",
    rollout_data_path="/storage/yueming/rollout-subsets/iter59_synthetic_2094_plus32.pt",
    sg_baseline_response_path="/storage/yueming/dumper-out/v4-iter59-sg-natural-e8env/sg_baseline.json",
    output_root="/storage/yueming/dumper-out",
)


CANONICAL_DUMPER_FILTER = "layer_id is None or layer_id < 3 or layer_id == 24"
"""Default DUMPER_FILTER for grafter experiments: scope to layers 0-2 (representative
of cr=0, cr=4, cr=128 mix) plus layer 24 (a deeper layer for sanity check). Reduces
dump volume by ~85% vs unfiltered while keeping the diagnostically critical layers."""
