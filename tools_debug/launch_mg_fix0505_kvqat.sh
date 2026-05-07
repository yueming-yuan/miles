#!/bin/bash
set -ex

OUTPUT=/storage/yueming/dumper-out/v4-iter59-mg-fix0505-kvqat
mkdir -p "$OUTPUT"

export MILES_DSV4_FIX_0505=1
export MEGATRON_USE_KV_QAT=1

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
  --source-patcher-config /tmp/megatron_patcher.yaml \
  --logprob-output "$OUTPUT/megatron_logprobs" \
  --extra-args "--no-load-optim --no-load-rng" \
  2>&1 | tee "$OUTPUT/run.log"
