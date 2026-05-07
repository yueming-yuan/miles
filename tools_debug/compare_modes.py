import json, sys
for line in open(sys.argv[1]):
    e = json.loads(line)
    if e.get("type") == "comparison_tensor":
        n = e.get("name")
        if n in ("layer_input", "attn_q", "kv_after_norm", "mqa_wo_b_out"):
            tp = e.get("traced_plan", {})
            print(f"{n}: mode={tp.get('token_aligner_mode')} plan={tp.get('token_aligner_plan')} unified={e.get('unified_shape')}")
