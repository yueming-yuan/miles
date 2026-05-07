#!/bin/bash
set -ex
export DUMPER_ENABLE=1
export DUMPER_NON_INTRUSIVE_MODE=off
export DUMPER_DIR=/storage/yueming/dumper-out/v4-iter59-sg3-noise
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

cd /workspace/sglang

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
  --page-size 256 --max-running-requests 64 --chunked-prefill-size 8192 \
  --weight-loader-drop-cache-after-load \
  --mem-fraction-static 0.7 \
  --disable-cuda-graph \
  2>&1 | tee /storage/yueming/dumper-out/v4-iter59-sg3-noise/server.log
