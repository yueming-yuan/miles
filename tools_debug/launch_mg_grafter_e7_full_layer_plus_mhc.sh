#!/bin/bash
# E7 — E6 plus MHC head + final_layernorm graft.
#
# E6 floor at max=10.70 = layer-42 kernel diff + mg.MHC_head + mg.final_norm + mg.lm_head.
# E7 adds a t2b graft on `post_norm_hidden` (after self.norm in sg / after final_layernorm
# + make_viewless_tensor in mg). This eliminates: layer-42 kernel diff, mg.MHC_head,
# mg.final_norm. Remaining residual = mg.lm_head kernel diff (and any sg/mg pre-lm_head
# tensor format differences).
#
# Patches: ONLY the layer_input edit on TransformerLayer._forward_attention (E6's) and
# the post-final-layernorm edit on TransformerBlock.forward (new for E7). No attn or
# MLP patches — those would conflict with V4's MQALayer dumps per F7 / SKILL #15.
set -ex

OUTPUT=/storage/yueming/dumper-out/v4-iter59-grafter-e7-full-layer-plus-mhc-mg
mkdir -p "$OUTPUT"

PATCHER_YAML=/tmp/megatron_e7_layer_input_plus_post_norm_patcher.yaml
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
export DUMPER_GRAFTER_T2B_FILTER="${DUMPER_GRAFTER_T2B_FILTER:-name in (\"layer_input\", \"post_norm_hidden\") and graft_phase == \"prefill\"}"
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
