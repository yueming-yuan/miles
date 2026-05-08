#!/bin/bash
# A5' — Replace ONLY sg.{wo_a, wo_b} (V4 output projection LoRA), every layer.
#
# Replaced kernels (mg uses sg's compute downstream):
#   sg.{wo_a, wo_b} = the V4 dual-LoRA output projection at the end of attention block
#
# NOT replaced (mg-natural):
#   input_layernorm; full Q-LoRA + KV-LoRA paths; compressor; indexer; sparse-MLA;
#   hyper-connection; pre_mlp_layernorm; MoE; MHC head; final_norm; lm_head
#
# Graft tensors:
#   b2t (mg→sg): `attn_output` (mg's pre-output-projection attn output → sg).
#                sg.wo_a + sg.wo_b run on mg's attn_output.
#   t2b (sg→mg): `mqa_wo_b_out` (sg's post-output-projection → mg).
#                mg uses sg.wo_b(sg.wo_a(mg.attn_output)) at the post-projection slot.
#
# Distinct from A5 (which used input_layernorm as b2t — too wide, hung due to DP-idle bug).
# This A5' is narrower: just the output projection step.
#
# DP-idle fix: sg side now fires empty mqa_wo_a_out + mqa_wo_b_out dumps in idle path
# (deepseek_v4.py:864 area added), so all 8 sg ranks participate in collectives.
set -ex

OUTPUT=/storage/yueming/dumper-out/v4-iter59-grafter-a5b-output-proj-only-mg
mkdir -p "$OUTPUT"

# No source-patcher needed: attn_output and mqa_wo_b_out both already dumped natively
# by V4 plugin in mg (deepseek_v4.py:319 attn_output, line 328 mqa_wo_b_out).

export MILES_DSV4_FIX_0505=1

export DUMPER_GRAFTER_ENABLE=1
export DUMPER_GRAFTER_ROLE=baseline
export DUMPER_GRAFTER_B2T_FILTER="${DUMPER_GRAFTER_B2T_FILTER:-name == \"attn_output\" and graft_phase == \"prefill\"}"
export DUMPER_GRAFTER_T2B_FILTER="${DUMPER_GRAFTER_T2B_FILTER:-name == \"mqa_wo_b_out\" and graft_phase == \"prefill\"}"
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
  --logprob-output "$OUTPUT/megatron_logprobs" \
  --extra-args "--no-load-optim --no-load-rng" \
  2>&1 | tee "$OUTPUT/run.log"
