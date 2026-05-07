#!/bin/bash
# A4 — Replace ALL of sg.attention from input_layernorm through attn_output.
#
# **Replaced kernels** (mg uses sg's compute downstream):
#   sg.{Q-LoRA path, KV-LoRA path, compressor, indexer, sparse-MLA}
#
# **NOT replaced** (mg-natural compute flows):
#   input_layernorm, KV-LoRA path, compressor, indexer, sparse-MLA, output projection,
#   hyper-connection, MoE, MHC head, final RMSNorm, lm_head.
#
# Graft tensors:
#   b2t (mg→sg): `input_layernorm` (mg's post-RMSNorm hidden, sg replaces its own input
#                to the LoRA stage with mg's value)
#   t2b (sg→mg): `attn_q` (sg's post-RoPE Q, computed on mg's input_layernorm output via
#                sg.Q-LoRA path; mg's own attn_q gets overridden)
#
# Boundary semantic check:
#   - input_layernorm output: shape (T, hidden=4096) BF16 on both sides; same RMSNorm
#     output semantics. mg dump SP-sharded (T_sp,1,4096) per rank → cat across SP for
#     full T, squeeze bsz. (`transform_input_layernorm_b2t`)
#   - attn_q: shape (T, n_heads, head_dim) post-RoPE on both sides; mg has TP-sharded heads
#     per rank (1, T_padded, n_h_local=8, 512); sg has full (T_actual, n_heads=64, 512) on
#     active DP rank. (`transform_attn_q_t2b`)
#
# Note: sg's KV-LoRA / compressor / indexer also consume the b2t-grafted input_layernorm
# value, but only sg.Q-LoRA's output (attn_q) is sent back to mg. Mg's KV-LoRA / compressor
# etc. remain mg-natural.
set -ex

OUTPUT=/storage/yueming/dumper-out/v4-iter59-grafter-a4-attn-block-from-input-layernorm-mg
mkdir -p "$OUTPUT"

PATCHER_YAML=/tmp/megatron_a1_input_layernorm_patcher.yaml
cat > "$PATCHER_YAML" << 'YAML_EOF'
patches:
  - target: megatron.core.transformer.transformer_layer.TransformerLayer._forward_attention
    edits:
      - match: |
          input_layernorm_output = self.input_layernorm(hidden_states)
        append: "dumper.dump('input_layernorm', input_layernorm_output, dims='t[cp:zigzag,sp] 1 h # tp:replicated ep:replicated')"
YAML_EOF

export MILES_DSV4_FIX_0505=1

export DUMPER_GRAFTER_ENABLE=1
export DUMPER_GRAFTER_ROLE=baseline
export DUMPER_GRAFTER_B2T_FILTER="${DUMPER_GRAFTER_B2T_FILTER:-name == \"input_layernorm\" and graft_phase == \"prefill\"}"
export DUMPER_GRAFTER_T2B_FILTER="${DUMPER_GRAFTER_T2B_FILTER:-name == \"attn_output\" and graft_phase == \"prefill\"}"
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
