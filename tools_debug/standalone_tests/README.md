# Standalone numerical tests

Self-contained verification tests for V4 cross-stack divergence root-cause analysis.
Each test is a single .py file, runs in seconds on a single GPU, no model load needed.

## Naming convention

`test_{NNN}_{topic_hyphen_separated}.py` — incrementing index lets us reference tests
("test_001 says X") without ambiguity. Topic is a short discriminative descriptor.

## Tests

| # | name | what it verifies |
|---|---|---|
| 001 | rmsnorm-per-head-sg-vs-mg | Whether MG's BF16-path q_heads_after_norm RMSNorm is numerically inferior to SG's FP32-internal CUDA kernel; isolates the L0 q_heads_after_norm divergence injection point. |
| 002 | compressor-wkv-gate-sg-vs-mg | Whether SG's BF16 wkv_gate weight (vs MG's FP32 wkv+wgate) injects the L2/L3 `compressor_kv_score` cross-stack divergence. Measured 600× larger error than FP32. |

## Usage

```bash
cd /workspace/miles
python tools_debug/standalone_tests/test_001_rmsnorm-per-head-sg-vs-mg.py \
    --wq-b-pt /storage/yueming/dumper-out/v4-iter59-sg2-noise/dump_*/step=0___rank=0___dump_index=*___name=wq_b_out*.pt
```
