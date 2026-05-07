"""Send full token IDs in prefill-only mode and capture per-position logprobs.

Used for noise-floor measurement: same input → emit input_token_logprobs at every position.
"""
import argparse
import json
import time
import urllib.request
import urllib.error

ap = argparse.ArgumentParser()
ap.add_argument("--ids-json", required=True, help="json with key full_ids: [int,...]")
ap.add_argument("--server", default="http://127.0.0.1:30000")
ap.add_argument("--out", required=True)
args = ap.parse_args()

ids = json.load(open(args.ids_json))["full_ids"]
print(f"prefill input_ids len={len(ids)}", flush=True)

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
    raise SystemExit("server did not come up within 15 minutes")

payload = {
    "input_ids": ids,
    "sampling_params": {"temperature": 0.0, "max_new_tokens": 1, "ignore_eos": False, "stop": []},
    "return_logprob": True,
    "logprob_start_len": 0,
    "return_text_in_logprobs": False,
    "stream": False,
}
print("posting to /generate (prefill-only)...", flush=True)
data = json.dumps(payload).encode()
req = urllib.request.Request(args.server + "/generate", data=data, headers={"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=2400) as r:
    resp = json.loads(r.read())
elapsed = time.time() - t0
print(f"generate took {elapsed:.1f}s", flush=True)

with open(args.out, "w") as f:
    json.dump(resp, f, indent=2)
meta = resp.get("meta_info", {})
itp = meta.get("input_token_logprobs", [])
otp = meta.get("output_token_logprobs", [])
print(f"input_token_logprobs len: {len(itp)} first3: {itp[:3]}", flush=True)
print(f"output_token_logprobs len: {len(otp)}: {otp}", flush=True)
print(f"saved {args.out}", flush=True)
