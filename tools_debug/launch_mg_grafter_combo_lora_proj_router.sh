#!/bin/bash
# Combo: A1 + A2 + A5' + M1 — replace LoRA paths + output projection + MoE router.
#
# **Replaced kernels** (mg uses sg's compute downstream):
#   sg.{wq_a, q_lora_norm, wq_b, q_heads_norm, q_rope}        (A1: Q-LoRA path)
#   sg.{wkv, kv_norm, kv_rope}                                (A2: KV-LoRA path)
#   sg.{wo_a, wo_b}                                           (A5': output projection)
#   sg.MoE.router (sqrtsoftplus + HashTopK + weights)         (M1: routing decisions)
#
# **NOT replaced** (mg-natural compute flows):
#   input_layernorm; compressor; indexer; sparse-MLA / SWA / FlashMLA; routed_experts
#   grouped GEMM; shared expert; output combine; hc_pre/hc_post (MHC); pre_mlp_layernorm;
#   MHC head (block_head); final RMSNorm; lm_head.
#
# Graft tensors per layer (5 b2t + 5 t2b):
#   b2t (mg→sg):
#     - input_layernorm           (A1+A2: sg.{wq_a, wkv, compressor} consume mg's value)
#     - attn_output               (A5': sg.{wo_a, wo_b} consume mg's value)
#     - pre_mlp_layernorm_output  (M1: sg.MoE.router consumes mg's value)
#   t2b (sg→mg):
#     - attn_q                    (A1: mg's downstream sparse-MLA consumes sg's value)
#     - attn_v                    (A2: mg's downstream sparse-MLA consumes sg's value)
#     - mqa_wo_b_out              (A5': mg.hc_post consumes sg's value as 'out' arg)
#     - moe_routing_map           (M1: mg's expert dispatcher consumes sg's value)
#     - moe_probs                 (M1: mg's expert dispatcher consumes sg's value)
#
# Critical correctness checks:
#   1. NO t2b on residual stream (no hc_attn_post / hc_ffn_post / layer_input).
#      All t2b tensors are intermediate values consumed by a downstream kernel,
#      not residual states overriding the layer's hidden state. Verified against
#      H1/H1' failure mode (those grafted hc_*_post → equivalent to E6 layer-hidden
#      swap; gives mean ~0 because residual stream gets fully synced).
#   2. Each b2t has a real downstream consumer on sg side that we want to replace.
#   3. All required dump points exist:
#        mg natively dumps attn_output (deepseek_v4.py:319), mqa_wo_b_out (line 328),
#                          attn_q (in MQALayer), attn_v (in KVLayer)
#        sg natively dumps the same plus pre_mlp_layernorm_output / input_layernorm
#                         (the latter via source-patcher injected here)
#        M1 source-patcher injects moe_routing_map / moe_probs in TopKRouter.forward
#        SGLANG_DSV4_DUMP_ROUTING=1 enables sg's matching dumps in HashTopK
#   4. grafter_transforms.py has all required transforms for these 8 names:
#        input_layernorm_b2t, attn_output_b2t, pre_mlp_layernorm_output_b2t,
#        attn_q_t2b, attn_v_t2b, mqa_wo_b_out_t2b (= attn_v_t2b alias),
#        moe_routing_map_t2b, moe_probs_t2b (newly restored, mlp_output_t2b alias)
set -ex

OUTPUT=/storage/yueming/dumper-out/v4-iter59-grafter-combo-lora-proj-router-mg
mkdir -p "$OUTPUT"

PATCHER_YAML=/tmp/megatron_combo_lora_proj_router_patcher.yaml
cat > "$PATCHER_YAML" << 'YAML_EOF'
patches:
  - target: megatron.core.transformer.transformer_layer.TransformerLayer._forward_attention
    edits:
      - match: |
          input_layernorm_output = self.input_layernorm(hidden_states)
        append: "dumper.dump('input_layernorm', input_layernorm_output, dims='t[cp:zigzag,sp] 1 h # tp:replicated ep:replicated')"
  - target: megatron.core.transformer.transformer_layer.TransformerLayer._forward_mlp
    edits:
      - match: "pre_mlp_layernorm_output = self._forward_pre_mlp_layernorm(hidden_states)"
        append: "dumper.dump('pre_mlp_layernorm_output', pre_mlp_layernorm_output, dims='t[cp:zigzag,sp] 1 h # tp:replicated ep:replicated')"
  - target: megatron.core.transformer.moe.router.TopKRouter.forward
    edits:
      - match: "probs, routing_map = self.routing(logits, padding_mask=padding_mask, input_ids=input_ids)"
        append: "dumper.dump('moe_routing_map', routing_map, dims='t e # tp:replicated ep:replicated'); dumper.dump('moe_probs', probs, dims='t e # tp:replicated ep:replicated')"
YAML_EOF

export MILES_DSV4_FIX_0505=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export DUMPER_GRAFTER_ENABLE=1
export DUMPER_GRAFTER_ROLE=baseline
export DUMPER_GRAFTER_B2T_FILTER="${DUMPER_GRAFTER_B2T_FILTER:-name in (\"input_layernorm\", \"attn_output\", \"pre_mlp_layernorm_output\") and graft_phase == \"prefill\"}"
export DUMPER_GRAFTER_T2B_FILTER="${DUMPER_GRAFTER_T2B_FILTER:-name in (\"attn_q\", \"attn_v\", \"mqa_wo_b_out\", \"moe_routing_map\", \"moe_probs\") and graft_phase == \"prefill\"}"
export DUMPER_GRAFTER_MASTER_ADDRESS="${DUMPER_GRAFTER_MASTER_ADDRESS:-172.16.190.152}"
export DUMPER_GRAFTER_MASTER_PORT="${DUMPER_GRAFTER_MASTER_PORT:-29501}"
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
