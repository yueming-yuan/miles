#!/bin/bash
# E8 — E7 plus lm_head logits graft.
#
# E7 floor at max=2.33 = mg.lm_head matmul + mg's vocab-padding-induced log_softmax
# denominator diff (mg includes 768 garbage padding rows that sg slices out).
# E8 grafts sg's gathered logits (T, V_sg=129280) into mg's slot (1, T_padded, V_mg=130048),
# fills mg's padding cols with large-negative so log_softmax denom matches sg's.
#
# Residual after E8 = mg.log_softmax(grafted_logits) − sg.log_softmax(sg_logits) on
# bit-equal input. Both sides do log_softmax(.float()) via torch — should be ~0.
# If E8 ≈ 0: lm_head matmul + vocab-padding fully explain E7's 2.33 max.
# If E8 > 0: log_softmax kernel itself or downstream indexing has noise.
#
# Patches: E7's two patches PLUS the worker's `lm_head_logits` dump (added directly in
# `worker/main.py:242`, no source-patcher needed for that).
set -ex

OUTPUT=/storage/yueming/dumper-out/v4-iter59-grafter-e8-lm-head-baseline-mg
mkdir -p "$OUTPUT"

PATCHER_YAML=/tmp/megatron_e8_layer_input_plus_post_norm_patcher.yaml
cat > "$PATCHER_YAML" << 'YAML_EOF'
patches:
  - target: megatron.core.transformer.transformer_layer.TransformerLayer._forward_attention
    edits:
      - match: |
          inference_context = deprecate_inference_params(inference_context, inference_params)
        append: "dumper.dump('layer_input', hidden_states, dims='t[cp:zigzag,sp] 1 h # tp:replicated ep:replicated')"
  - target: megatron.core.transformer.transformer_block.TransformerBlock.forward
    edits:
      - match: |
                      hidden_states = make_viewless_tensor(
                          inp=hidden_states, requires_grad=True, keep_graph=True
                      )
        append: "dumper.dump('post_norm_hidden', hidden_states, dims='t[cp:zigzag,sp] 1 h # tp:replicated ep:replicated')"
YAML_EOF

export MILES_DSV4_FIX_0505=1

export DUMPER_GRAFTER_ENABLE=1
export DUMPER_GRAFTER_ROLE=baseline
export DUMPER_GRAFTER_B2T_FILTER="${DUMPER_GRAFTER_B2T_FILTER:-False}"
export DUMPER_GRAFTER_T2B_FILTER="${DUMPER_GRAFTER_T2B_FILTER:-name in (\"layer_input\", \"post_norm_hidden\", \"lm_head_logits\") and graft_phase == \"prefill\"}"
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
