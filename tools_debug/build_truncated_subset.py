"""Truncate iter59_worst sample to a chosen global position so sglang generates from there."""
import sys
import torch

src = sys.argv[1]
dst = sys.argv[2]
truncate_at = int(sys.argv[3])

d = torch.load(src, weights_only=False)
sample = d["samples"][0]
tokens = sample["tokens"]
prompt_len_orig = len(tokens) - sample["response_length"]
print(f"orig: total_len={len(tokens)} prompt_len={prompt_len_orig} resp_len={sample['response_length']}")
new_tokens = tokens[:truncate_at]
new_sample = dict(sample)
new_sample["tokens"] = new_tokens
new_sample["response_length"] = 0
new_sample["rollout_log_probs"] = []
new_sample["loss_mask"] = None
new_sample["rollout_routed_experts"] = None
new_sample["rollout_indexer_topk"] = None
out = {"rollout_id": d["rollout_id"], "samples": [new_sample]}
torch.save(out, dst)
print(f"new prompt_len={len(new_tokens)} saved to {dst}")
