"""Build a synthetic rollout .pt for megatron --rollout-data:
   prompt (= truncated worst sample's tokens) + sglang's generated tokens.
   This way megatron's forward sees the EXACT tokens sglang produced,
   making per-position dumps comparable.
"""
import json
import sys
import torch

src_pt = sys.argv[1]                # truncated subset (sample.tokens = prompt only)
sgl_response_json = sys.argv[2]    # sglang's /generate response
out_pt = sys.argv[3]

d = torch.load(src_pt, weights_only=False)
sample = d["samples"][0]
prompt_tokens = list(sample["tokens"])  # already truncated (response_length=0)

resp = json.load(open(sgl_response_json))
out_ids = resp["output_ids"]            # list[int]
out_lp = resp["meta_info"]["output_token_logprobs"]  # list[[lp, tok_id, _]]
assert len(out_ids) == len(out_lp), f"output_ids vs output_token_logprobs length mismatch: {len(out_ids)} vs {len(out_lp)}"
sg_logprobs = [float(e[0]) for e in out_lp]

new_tokens = prompt_tokens + list(out_ids)
new_resp_len = len(out_ids)

new_sample = dict(sample)
new_sample["tokens"] = new_tokens
new_sample["response_length"] = new_resp_len
new_sample["rollout_log_probs"] = sg_logprobs
new_sample["loss_mask"] = None
new_sample["rollout_routed_experts"] = None
new_sample["rollout_indexer_topk"] = None

out = {"rollout_id": d["rollout_id"], "samples": [new_sample]}
torch.save(out, out_pt)
print(f"saved {out_pt}: total_len={len(new_tokens)} prompt_len={len(prompt_tokens)} resp_len={new_resp_len}")
print(f"sg logprobs preview: {sg_logprobs[:5]} ... {sg_logprobs[-3:]}")
