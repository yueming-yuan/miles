#!/bin/bash
# Combo sg-side. sg's V4 plugin natively dumps:
#   attn_q, attn_v, attn_output, mqa_wo_b_out, input_layernorm, pre_mlp_layernorm_output
#   plus moe_routing_map + moe_probs (gated by SGLANG_DSV4_DUMP_ROUTING=1).
# All needed graft tensors already exist on sg side — no source-patcher needed.
set -ex

OUTPUT=/storage/yueming/dumper-out/v4-iter59-grafter-combo-lora-proj-router-sg
mkdir -p "$OUTPUT"

export DUMPER_ENABLE=1
export DUMPER_NON_INTRUSIVE_MODE=off
export DUMPER_DIR=$OUTPUT
export DUMPER_FILTER='layer_id is None or layer_id < 3 or layer_id == 24'

export SGLANG_SKIP_CHECKPOINT_LOAD_CHECK=1
export SGL_JIT_DEEPGEMM_PRECOMPILE=false
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=false
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=true
export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=true
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false
export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT=true
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=false
export SGLANG_DSV4_MODE=2604
export SGLANG_FIX_MTP_HC_HIDDEN=1
export SGLANG_OPT_DEEPGEMM_SCALE_CONVERT_AT_INIT=1
export SGLANG_DSV4_FP4_EXPERTS=0

export SGLANG_DSV4_FIX_0506=1
export SGLANG_DSV4_DUMP_ROUTING=1
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0
export SGLANG_OPT_USE_TILELANG_MHC_POST=0

export DUMPER_GRAFTER_ENABLE=1
export DUMPER_GRAFTER_ROLE=target
export DUMPER_GRAFTER_B2T_FILTER="${DUMPER_GRAFTER_B2T_FILTER:-name in (\"input_layernorm\", \"attn_output\", \"pre_mlp_layernorm_output\") and graft_phase == \"prefill\"}"
export DUMPER_GRAFTER_T2B_FILTER="${DUMPER_GRAFTER_T2B_FILTER:-name in (\"attn_q\", \"attn_v\", \"mqa_wo_b_out\", \"moe_routing_map\", \"moe_probs\") and graft_phase == \"prefill\"}"
export DUMPER_GRAFTER_MASTER_ADDRESS="${DUMPER_GRAFTER_MASTER_ADDRESS:-172.16.190.152}"
export DUMPER_GRAFTER_MASTER_PORT="${DUMPER_GRAFTER_MASTER_PORT:-29501}"
export DUMPER_GRAFTER_BASELINE_WORLD_SIZE=8
export DUMPER_GRAFTER_TARGET_WORLD_SIZE=8
export DUMPER_GRAFTER_TRANSFORM_PATH=tools_debug.grafter_transforms.transform
export DUMPER_GRAFTER_BACKEND="${DUMPER_GRAFTER_BACKEND:-gloo}"
export DUMPER_GRAFTER_TIMEOUT="${DUMPER_GRAFTER_TIMEOUT:-600}"

cd /workspace/sglang
export PYTHONPATH=/workspace/miles:${PYTHONPATH:-}

python -m sglang.launch_server \
  --model-path /storage/yueming/iter59-hf-fp8 \
  --trust-remote-code --random-seed 0 \
  --host 0.0.0.0 --port 30000 \
  --nnodes 1 --node-rank 0 \
  --tp-size 8 --dp-size 8 --pp-size 1 --ep-size 8 \
  --skip-server-warmup \
  --enable-draft-weights-cpu-backup \
  --enable-dp-attention \
  --attention-backend compressed \
  --page-size 256 --max-running-requests 64 --chunked-prefill-size 32768 \
  --weight-loader-drop-cache-after-load \
  --mem-fraction-static 0.7 \
  --disable-cuda-graph \
  --watchdog-timeout 1800 \
  2>&1 | tee "$OUTPUT/server.log"
