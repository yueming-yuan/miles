#!/bin/bash
# E5 — sparse-MLA-kernel approximation: E3 (attn block swap) + compress_final_out
# b2t graft so sg's attn pipeline ALSO uses mg's compressed KV (via KV cache write).
#
# What's actually swapped on mg's forward:
#   pure-mg up to attention → mg sends (attn_q, attn_v, compress_final_out) → sg →
#   sg.{indexer + sparse-MLA-kernel} runs on mg's q + mg's vanilla KV + mg's compressed KV
#   (sg's compressor is RE-RUN on sg's own input but its OUTPUT is overwritten by mg's
#   via the compress_final_out b2t graft, so sg's KV cache contains mg's compressed
#   values) → sg.attn_output → t2b graft → mg downstream uses sg's output.
#
# Note: sg's indexer kernel still runs on mg's q (not its output is grafted). Indexer
# divergence per F3 was ~2% so the approximation is close to "sparse-MLA-kernel only".
# A truly isolated sparse-MLA kernel swap requires sg-side code mods because sg's
# kernel API (paged-cache + per-page indices) differs from mg's (concat-KV + topk_idxs).
set -ex

OUTPUT=/storage/yueming/dumper-out/v4-iter59-grafter-e5-sparse-mla-mg
mkdir -p "$OUTPUT"

export MILES_DSV4_FIX_0505=1

export DUMPER_GRAFTER_ENABLE=1
export DUMPER_GRAFTER_ROLE=baseline
# b2t = mg sends to sg: q, v (kernel inputs) + compress_final_out (post-compressor KV)
export DUMPER_GRAFTER_B2T_FILTER="${DUMPER_GRAFTER_B2T_FILTER:-(name in (\"attn_q\", \"attn_v\") and graft_phase == \"prefill\") or (name == \"compress_final_out\" and graft_phase == \"prefill\" and head_dim == 512)}"
# t2b = sg sends to mg: kernel output
export DUMPER_GRAFTER_T2B_FILTER="${DUMPER_GRAFTER_T2B_FILTER:-name == \"attn_output\" and graft_phase == \"prefill\"}"
export DUMPER_GRAFTER_MASTER_ADDRESS="${DUMPER_GRAFTER_MASTER_ADDRESS:-172.16.174.250}"
export DUMPER_GRAFTER_MASTER_PORT="${DUMPER_GRAFTER_MASTER_PORT:-29500}"
export DUMPER_GRAFTER_BASELINE_WORLD_SIZE=8
export DUMPER_GRAFTER_TARGET_WORLD_SIZE=8
export DUMPER_GRAFTER_TRANSFORM_PATH=tools_debug.grafter_transforms.transform
export DUMPER_GRAFTER_BACKEND="${DUMPER_GRAFTER_BACKEND:-gloo}"
export DUMPER_GRAFTER_TIMEOUT="${DUMPER_GRAFTER_TIMEOUT:-600}"

cd /workspace/miles
# No --source-patcher-config (V4 uses MQALayer, conftest YAML targets vanilla path).
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
