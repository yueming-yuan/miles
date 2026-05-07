#!/bin/bash
# E3 (实验 A) — entire attention BLOCK swap.
# b2t: mg sends attn_q + attn_v (per-layer attention kernel inputs) to sg.
# t2b: sg sends attn_output back to mg (post-attention, pre-wo_a/wo_b projection).
# Net effect on mg: pure-mg-everywhere except the attention block (indexer +
# compressor + sparse-MLA / FlashMLA / SWA) is replaced by sg's implementation
# computing on identical Q/V inputs.
set -ex

OUTPUT=/storage/yueming/dumper-out/v4-iter59-grafter-e3-attn-block-mg
mkdir -p "$OUTPUT"

export MILES_DSV4_FIX_0505=1

export DUMPER_GRAFTER_ENABLE=1
export DUMPER_GRAFTER_ROLE=baseline
export DUMPER_GRAFTER_B2T_FILTER="${DUMPER_GRAFTER_B2T_FILTER:-name in (\"attn_q\", \"attn_v\") and graft_phase == \"prefill\"}"
export DUMPER_GRAFTER_T2B_FILTER="${DUMPER_GRAFTER_T2B_FILTER:-name == \"attn_output\" and graft_phase == \"prefill\"}"
export DUMPER_GRAFTER_MASTER_ADDRESS="${DUMPER_GRAFTER_MASTER_ADDRESS:-172.16.174.250}"
export DUMPER_GRAFTER_MASTER_PORT="${DUMPER_GRAFTER_MASTER_PORT:-29500}"
export DUMPER_GRAFTER_BASELINE_WORLD_SIZE=8
export DUMPER_GRAFTER_TARGET_WORLD_SIZE=8
export DUMPER_GRAFTER_TRANSFORM_PATH=tools_debug.grafter_transforms.transform
export DUMPER_GRAFTER_BACKEND="${DUMPER_GRAFTER_BACKEND:-gloo}"
export DUMPER_GRAFTER_TIMEOUT="${DUMPER_GRAFTER_TIMEOUT:-600}"

cd /workspace/miles
# NOTE: --source-patcher-config is OMITTED for V4. The conftest_dumper YAML
# patches the canonical Megatron `TransformerLayer._forward_attention` hook
# (intended for vanilla Megatron models like Qwen) and dumps `attn_q`,
# `attn_v`, `attn_output` (post-o_proj). V4 has a CUSTOM attention module
# (`MQALayer` in miles_plugins/models/deepseek_v4/deepseek_v4.py) with
# V4-specific dumps under the same names but DIFFERENT SEMANTICS — most
# notably V4 plugin's `attn_output` is PRE-projection (matches sg side),
# while source-patcher's is POST-projection. Applying both produces
# duplicate dumps with colliding names, breaking grafter (E3 hangs / crashes
# with None in sender_contribs) and confusing downstream comparators.
python -m miles.utils.debug_utils.run_megatron run \
  --model-type deepseek-v4-flash \
  --hf-checkpoint /storage/models/sgl-project/DeepSeek-V4-Flash-FP8 \
  --ref-load /storage/yueming/saves/training-jobs/260501-gsm8k-seqlen8192/checkpoints \
  --sp \
  --rollout-data /storage/yueming/rollout-subsets/iter59_synthetic_2094_plus32.pt \
  --tp 8 --pp 1 --cp 1 --ep 8 --etp 1 \
  --batch-size 1 \
  --output-dir "$OUTPUT" \
  --logprob-output "$OUTPUT/megatron_logprobs" \
  --extra-args "--no-load-optim --no-load-rng" \
  2>&1 | tee "$OUTPUT/run.log"
