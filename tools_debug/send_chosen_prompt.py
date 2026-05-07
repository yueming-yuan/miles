"""Send the chosen sample's prompt to SGLang and capture logprobs."""
import json
import time
import argparse
import urllib.request
import urllib.error

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--rollout-pt", required=True)
ap.add_argument("--server", default="http://127.0.0.1:30000")
ap.add_argument("--out", required=True)
ap.add_argument("--temperature", type=float, default=0.0)
args = ap.parse_args()

d = torch.load(args.rollout_pt, weights_only=False)
sample = d["samples"][0]
# Plan-aligned: send prompt only, let sglang generate response. Both sides see same kernel paths
# (prefill on prompt + N decode steps). Megatron then processes prompt+sglang_tokens in one forward.
prompt_len_orig = len(sample["tokens"]) - sample["response_length"]
prompt_token_ids = sample["tokens"][:prompt_len_orig]
import os
resp_len = int(os.environ.get("MAX_NEW_TOKENS", "256"))
print(f"prompt_len={len(prompt_token_ids)} max_new_tokens={resp_len} (sample resp_len={sample['response_length']}) index={sample['index']}", flush=True)

print("waiting for /health...", flush=True)
for i in range(180):
    try:
        with urllib.request.urlopen(args.server + "/health") as r:
            if r.status == 200:
                print(f"server up after {i*5}s", flush=True)
                break
    except urllib.error.URLError:
        pass
    time.sleep(5)
else:
    print("server did not come up within 15 minutes", flush=True)
    raise SystemExit(1)

payload = {
    "input_ids": prompt_token_ids,
    "sampling_params": {
        "temperature": args.temperature,
        "max_new_tokens": resp_len,
        "ignore_eos": False,
        "stop": [],
    },
    "return_logprob": True,
    "return_text_in_logprobs": False,
    "stream": False,
}
print("posting to /generate...", flush=True)
data = json.dumps(payload).encode()
req = urllib.request.Request(args.server + "/generate", data=data, headers={"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=2400) as r:
    resp = json.loads(r.read())
elapsed = time.time() - t0
print(f"generate took {elapsed:.1f}s", flush=True)

with open(args.out, "w") as f:
    json.dump(resp, f, indent=2)
print(f"saved {args.out}", flush=True)

meta = resp.get("meta_info", {})
n_out = meta.get("completion_tokens", -1)
print(f"output_ids count: {n_out}", flush=True)
otp = meta.get("output_token_logprobs", [])
print(f"output_token_logprobs len: {len(otp)} first3: {otp[:3]}", flush=True)
