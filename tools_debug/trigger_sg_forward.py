"""POST sample 0 of synthetic_2094_plus32 to sg :30000 with return_logprob.

Usage: run from the worker pod (where sg server is listening on 0.0.0.0:30000)
    cd /workspace/miles && python tools_debug/trigger_sg_forward.py
    cd /workspace/miles && python tools_debug/trigger_sg_forward.py \\
        --baseline-output /storage/yueming/sg_prefill_baseline/rank_0.json

Expects /storage/yueming/rollout-subsets/iter59_synthetic_2094_plus32.pt to exist.
Sends the full 2126 tokens; max_new_tokens=1 forces single prefill chunk + 1 decode.

The optional --baseline-output writes sg's per-position prefill input_token_logprobs
in the format expected by miles' logprob_comparator. global_position p has the logprob
for predicting tokens[p+1] (Megatron's labels-shifted convention). Coverage = positions
0..len(tokens)-2, all token kinds (prompt + response).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

import torch


ROLLOUT_PATH = "/storage/yueming/rollout-subsets/iter59_synthetic_2094_plus32.pt"
SG_URL = "http://0.0.0.0:30000/generate"
TIMEOUT = 1800


def _convert_input_token_logprobs_to_baseline_entries(
    input_token_logprobs: list,
) -> list[dict]:
    """sg's input_token_logprobs[i] = log P(tokens[i] | tokens[0..i-1]).
    Each entry is [logprob, token_id, top_logprobs_or_None]; index 0 is
    [None, first_token, None] (no logprob for first token).

    In Megatron's labels-shifted convention, this corresponds to
    global_position = i-1, token_id = tokens[i].
    """
    entries: list[dict] = []
    for i, item in enumerate(input_token_logprobs):
        if i == 0:
            continue
        lp, tok_id = item[0], item[1]
        if lp is None:
            continue
        entries.append({
            "global_position": i - 1,
            "token_id": int(tok_id),
            "logprob": float(lp),
            "is_valid": True,
        })
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-output", default=None,
                        help="If set, write sg-prefill baseline JSON in mg-comparator format to this path.")
    cli_args = parser.parse_args()

    data = torch.load(ROLLOUT_PATH, weights_only=False)
    s0 = data["samples"][0]
    toks = list(s0["tokens"]) if not isinstance(s0["tokens"], list) else s0["tokens"]
    print(f"loaded sample0: total_tokens={len(toks)}  response_length={s0['response_length']}", flush=True)

    payload = {
        "input_ids": toks,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0.0, "top_k": 1},
        "return_logprob": True,
        "logprob_start_len": 0,
        "top_logprobs_num": 0,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(SG_URL, data=body, headers={"Content-Type": "application/json"})

    t0 = time.time()
    print(f"POSTing to {SG_URL} (max_new_tokens=1, return_logprob=True)...", flush=True)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        body = resp.read()
    dt = time.time() - t0
    print(f"sg responded after {dt:.1f}s, body size={len(body)} bytes", flush=True)

    out = json.loads(body)
    if isinstance(out, list):
        out = out[0]
    meta = out.get("meta_info", {})
    print(f"prompt_tokens={meta.get('prompt_tokens')} completion_tokens={meta.get('completion_tokens')}",
          flush=True)
    if "input_token_logprobs" in meta and meta["input_token_logprobs"]:
        n = len(meta["input_token_logprobs"])
        print(f"input_token_logprobs n={n}  first3={meta['input_token_logprobs'][:3]}", flush=True)

    if cli_args.baseline_output:
        from pathlib import Path
        baseline_path = Path(cli_args.baseline_output)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        entries = _convert_input_token_logprobs_to_baseline_entries(meta["input_token_logprobs"])
        payload_json = {
            "rank": 0,
            "tp_size": 1,
            "cp_size": 1,
            "pp_size": 1,
            "logprob_entries": [entries],
        }
        baseline_path.write_text(json.dumps(payload_json, indent=2))
        print(f"baseline saved: {len(entries)} positions -> {baseline_path}", flush=True)


if __name__ == "__main__":
    main()
