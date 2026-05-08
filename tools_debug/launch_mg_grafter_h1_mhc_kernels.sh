#!/bin/bash
# H1 — Replace ALL per-layer MHC kernels via grafter (NOT via PR #1's tile kernels).
#
# Replaced kernels (mg uses sg's compute downstream, every layer 0..42):
#   sg.{hc_pre_raw, hc_post_raw} on attn-side AND ffn-side = 4 MHC kernels per layer
#   = 4 × 43 = 172 MHC kernel calls replaced
#
# NOT replaced (mg-natural):
#   input_layernorm; mg.attention block (LoRA + compressor + indexer + sparse-MLA + output proj);
#   pre_mlp_layernorm; mg.MoE/MLP; MHC head (block_head); final_norm; lm_head
#
# Graft tensors per layer:
#   b2t (mg→sg): layer_input, attn_output, mlp_output
#                  - sg.hc_pre_attn computes on mg's layer_input
#                  - sg.hc_post_attn uses mg's attn_output as the "out" arg
#                  - sg.hc_post_ffn uses mg's mlp_output as the "out" arg
#                  - sg-internal post + comb mix matrices flow naturally inside sg
#   t2b (sg→mg): hc_attn_pre, hc_attn_post, hc_ffn_pre, hc_ffn_post
#                  - mg's per-layer recombination uses sg.hc_pre/post outputs
#
# Mg-side new dumps (via source-patcher YAML in this script):
#   1. hc_attn_pre after `hidden_states, hc_attn_post, hc_attn_comb = hc_util.layer_pre(...)`
#   2. hc_attn_post after `hidden_states = hc_util.layer_post(...)`
#   3. hc_ffn_pre after the analogous ffn-side layer_pre
#   4. hc_ffn_post after the analogous ffn-side layer_post
#   plus existing layer_input + mlp_output edits.
#
# Boundary semantic:
#   - hc_attn_pre/ffn_pre: sg (T,h) BF16, mg (T_sp,1,h) BF16. mlp_output_t2b shape pattern.
#   - hc_attn_post/ffn_post: sg (T,hc=4,h) BF16, mg (T_sp,1,hc,h) BF16. layer_input_t2b shape pattern.
#   - mg's input_layernorm runs on sg.hc_pre output (KERNEL is mg.input_layernorm but applied
#     to sg's hc_pre's hidden — which is what production "replace mg.hc_pre with sg.hc_pre"
#     would actually look like).
set -ex

OUTPUT=/storage/yueming/dumper-out/v4-iter59-grafter-h1-mhc-kernels-mg
mkdir -p "$OUTPUT"

PATCHER_YAML=/tmp/megatron_h1_mhc_patcher.yaml
cat > "$PATCHER_YAML" << 'YAML_EOF'
patches:
  - target: megatron.core.transformer.transformer_layer.TransformerLayer._forward_attention
    edits:
      - match: |
          inference_context = deprecate_inference_params(inference_context, inference_params)
        append: "dumper.dump('layer_input', hidden_states, dims='t[cp:zigzag,sp] 1 hc h # tp:replicated ep:replicated')"
      - match: |
              hidden_states, hc_attn_post, hc_attn_comb = hc_util.layer_pre(
                  hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
              )
        append: "dumper.dump('hc_attn_pre', hidden_states, dims='t[cp:zigzag,sp] 1 h # tp:replicated ep:replicated')"
      - match: |
              hidden_states = hc_util.layer_post(
                  attention_output_with_bias, residual, hc_attn_post, hc_attn_comb
              )
        append: "dumper.dump('hc_attn_post', hidden_states, dims='t[cp:zigzag,sp] 1 hc h # tp:replicated ep:replicated')"
  - target: megatron.core.transformer.transformer_layer.TransformerLayer._forward_mlp
    edits:
      - match: |
              hidden_states, hc_ffn_post, hc_ffn_comb = hc_util.layer_pre(
                  hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
              )
        append: "dumper.dump('hc_ffn_pre', hidden_states, dims='t[cp:zigzag,sp] 1 h # tp:replicated ep:replicated')"
      - match: "return self._forward_post_mlp("
        prepend: "dumper.dump('mlp_output', mlp_output_with_bias[0], dims='t[cp:zigzag,sp] 1 h # tp:replicated ep:replicated')"
  - target: megatron.core.transformer.transformer_layer.TransformerLayer._forward_post_mlp
    edits:
      - match: |
                  hidden_states = hc_util.layer_post(
                      mlp_output_with_bias, residual, hc_ffn_post, hc_ffn_comb
                  )
        append: "dumper.dump('hc_ffn_post', hidden_states, dims='t[cp:zigzag,sp] 1 hc h # tp:replicated ep:replicated')"
YAML_EOF

# Source-patcher YAML must include attn_output dump on mg side too (mg's V4 plugin already
# dumps attn_output, but at the LAYER level we need it source-patched alongside layer_input).
# Actually mg V4 plugin dumps attn_output natively (deepseek_v4.py:319). No source-patch needed.

export MILES_DSV4_FIX_0505=1

export DUMPER_GRAFTER_ENABLE=1
export DUMPER_GRAFTER_ROLE=baseline
export DUMPER_GRAFTER_B2T_FILTER="${DUMPER_GRAFTER_B2T_FILTER:-name in (\"layer_input\", \"attn_output\", \"mlp_output\") and graft_phase == \"prefill\"}"
export DUMPER_GRAFTER_T2B_FILTER="${DUMPER_GRAFTER_T2B_FILTER:-name in (\"hc_attn_pre\", \"hc_attn_post\", \"hc_ffn_pre\", \"hc_ffn_post\") and graft_phase == \"prefill\"}"
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
