#!/bin/bash
# E4 — MLP/MoE BLOCK swap.
# b2t: mg sends pre_mlp_layernorm_output (post-RMSNorm pre-MLP) to sg.
# t2b: sg sends mlp_output (post-MLP pre-residual-add) back to mg.
# Net effect on mg: pure-mg-everywhere except the MLP/MoE block (router +
# experts) is replaced by sg's implementation on identical post-RMSNorm input.
#
# YAML strategy: V4 plugin doesn't dump pre_mlp_layernorm_output / mlp_output
# natively (it lives in canonical Megatron's `TransformerLayer._forward_mlp`).
# We use a STRIPPED-DOWN source-patcher YAML that ONLY patches the MLP target,
# NOT the attention paths (those would conflict with V4's MQALayer dumps — see
# E2/E3 launch scripts and grafter SKILL #15).
set -ex

OUTPUT=/storage/yueming/dumper-out/v4-iter59-grafter-e4-mlp-block-mg
mkdir -p "$OUTPUT"

# Write the MLP-only patcher YAML inline so the experiment is self-contained.
PATCHER_YAML=/tmp/megatron_mlp_only_patcher.yaml
cat > "$PATCHER_YAML" << 'YAML_EOF'
patches:
  - target: megatron.core.transformer.transformer_layer.TransformerLayer._forward_mlp
    edits:
      - match: "residual = hidden_states"
        append: "dumper.dump('pre_mlp_residual', residual, dims='t[cp:zigzag,sp] 1 h # tp:replicated ep:replicated')"
      - match: "pre_mlp_layernorm_output = self._forward_pre_mlp_layernorm(hidden_states)"
        append: "dumper.dump('pre_mlp_layernorm_output', pre_mlp_layernorm_output, dims='t[cp:zigzag,sp] 1 h # tp:replicated ep:replicated')"
      - match: "return self._forward_post_mlp("
        prepend: "dumper.dump('mlp_output', mlp_output_with_bias[0], dims='t[cp:zigzag,sp] 1 h # tp:replicated ep:replicated')"
YAML_EOF

export MILES_DSV4_FIX_0505=1

export DUMPER_GRAFTER_ENABLE=1
export DUMPER_GRAFTER_ROLE=baseline
export DUMPER_GRAFTER_B2T_FILTER="${DUMPER_GRAFTER_B2T_FILTER:-name == \"pre_mlp_layernorm_output\" and graft_phase == \"prefill\"}"
export DUMPER_GRAFTER_T2B_FILTER="${DUMPER_GRAFTER_T2B_FILTER:-name == \"mlp_output\" and graft_phase == \"prefill\"}"
export DUMPER_GRAFTER_MASTER_ADDRESS="${DUMPER_GRAFTER_MASTER_ADDRESS:-172.16.174.250}"
export DUMPER_GRAFTER_MASTER_PORT="${DUMPER_GRAFTER_MASTER_PORT:-29500}"
export DUMPER_GRAFTER_BASELINE_WORLD_SIZE=8
export DUMPER_GRAFTER_TARGET_WORLD_SIZE=8
export DUMPER_GRAFTER_TRANSFORM_PATH=tools_debug.grafter_transforms.transform
export DUMPER_GRAFTER_BACKEND="${DUMPER_GRAFTER_BACKEND:-gloo}"
export DUMPER_GRAFTER_TIMEOUT="${DUMPER_GRAFTER_TIMEOUT:-600}"

cd /workspace/miles
python -m miles.utils.debug_utils.run_megatron run \
  --model-type deepseek-v4-flash \
  --hf-checkpoint /storage/models/sgl-project/DeepSeek-V4-Flash-FP8 \
  --ref-load /storage/yueming/saves/training-jobs/260501-gsm8k-seqlen8192/checkpoints \
  --sp \
  --rollout-data /storage/yueming/rollout-subsets/iter59_synthetic_2094_plus32.pt \
  --tp 8 --pp 1 --cp 1 --ep 8 --etp 1 \
  --batch-size 1 \
  --output-dir "$OUTPUT" \
  --source-patcher-config "$PATCHER_YAML" \
  --logprob-output "$OUTPUT/megatron_logprobs" \
  --extra-args "--no-load-optim --no-load-rng" \
  2>&1 | tee "$OUTPUT/run.log"
