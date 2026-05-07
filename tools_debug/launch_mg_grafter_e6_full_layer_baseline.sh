#!/bin/bash
# E6 — "good baseline": swap mg's hidden state with sg's at every layer boundary.
#
# t2b graft on `layer_input` (name fires at the START of every TransformerLayer,
# which is also the END of the previous layer): sg→mg, mg's layer_input is replaced
# with sg's. Mg computes the layer (output discarded by next layer's graft).
#
# Net on mg's forward: mg's layer 1..42 hidden state ALWAYS comes from sg's compute.
# mg's local kernel work is computed but discarded (except layer 42's output, which
# goes through mg's final_norm + lm_head).
#
# Residual cross-stack diff (sg-pure vs mg+E6) = mg's layer-42-kernel diff +
# mg.final_norm + mg.lm_head kernel diff. This is the FLOOR for binary search:
# any sub-experiment that lands close to E6's floor explains "everything except
# last layer kernel + output projection". Sub-experiments that land far have
# uncovered contributions.
#
# YAML: include ONLY the layer_input source-patcher edit (skip attn_output post-
# projection, attn_q/v in Attention.forward, MoE patches — those would conflict
# with V4's MQALayer dumps per F7 / SKILL #15).
set -ex

OUTPUT=/storage/yueming/dumper-out/v4-iter59-grafter-e6-full-layer-baseline-mg
mkdir -p "$OUTPUT"

PATCHER_YAML=/tmp/megatron_layer_input_only_patcher.yaml
cat > "$PATCHER_YAML" << 'YAML_EOF'
patches:
  - target: megatron.core.transformer.transformer_layer.TransformerLayer._forward_attention
    edits:
      - match: |
          inference_context = deprecate_inference_params(inference_context, inference_params)
        append: "dumper.dump('layer_input', hidden_states, dims='t[cp:zigzag,sp] 1 h # tp:replicated ep:replicated')"
YAML_EOF

export MILES_DSV4_FIX_0505=1

export DUMPER_GRAFTER_ENABLE=1
export DUMPER_GRAFTER_ROLE=baseline
# b2t empty (no mg→sg graft).
# t2b: sg sends layer_input to mg. mg's layer_input is replaced.
export DUMPER_GRAFTER_B2T_FILTER="${DUMPER_GRAFTER_B2T_FILTER:-False}"
export DUMPER_GRAFTER_T2B_FILTER="${DUMPER_GRAFTER_T2B_FILTER:-name == \"layer_input\" and graft_phase == \"prefill\"}"
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
