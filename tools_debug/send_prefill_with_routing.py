"""Send full token IDs in prefill mode and capture per-position logprobs + routing dumps.

For Stage 3 routing-replay experiment: SGLang returns routed_experts (MoE topk per layer)
and indexer_topk (sparse-MLA topk positions per cr=4 layer) for the entire sequence.

Output: synthetic rollout pt with samples[0] containing tokens, rollout_log_probs,
rollout_routed_experts, rollout_indexer_topk — directly consumable by run_megatron CLI.
"""
import argparse
import base64
import json
import time
import urllib.request
import urllib.error

import numpy as np
import torch


def post_generate(server: str, ids: list[int], temperature: float, max_new_tokens: int) -> dict:
    payload = {
        "input_ids": ids,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": False,
            "stop": [],
        },
        "return_logprob": True,
        "logprob_start_len": 0,
        "return_text_in_logprobs": False,
        "return_routed_experts": True,
        "return_indexer_topk": True,
        "stream": False,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(server + "/generate", data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=2400) as r:
        resp = json.loads(r.read())
    print(f"generate took {time.time() - t0:.1f}s", flush=True)
    return resp


def wait_health(server: str, timeout_s: int = 900) -> None:
    print("waiting for /health...", flush=True)
    for i in range(timeout_s // 5):
        try:
            with urllib.request.urlopen(server + "/health") as r:
                if r.status == 200:
                    print(f"server up after {i * 5}s", flush=True)
                    return
        except urllib.error.URLError:
            pass
        time.sleep(5)
    raise SystemExit("server did not come up")


def decode_routing(meta: dict, num_tokens_minus_1: int, num_layers: int, moe_topk: int,
                   num_c4_layers: int, indexer_topk: int) -> tuple[np.ndarray, np.ndarray]:
    routed_experts = np.frombuffer(
        base64.b64decode(meta["routed_experts"].encode("ascii")), dtype=np.int32,
    ).reshape(num_tokens_minus_1, num_layers, moe_topk)
    indexer = np.frombuffer(
        base64.b64decode(meta["indexer_topk"].encode("ascii")), dtype=np.int32,
    ).reshape(num_tokens_minus_1, num_c4_layers, indexer_topk)
    return routed_experts, indexer


def build_synthetic_rollout(
    full_ids: list[int],
    response_length: int,
    log_probs: list[float],
    routed_experts: np.ndarray,
    indexer_topk: np.ndarray,
) -> dict:
    """Match the schema expected by run_megatron CLI's _load_rollout_data."""
    sample = {
        "tokens": full_ids,
        "response_length": response_length,
        "rollout_log_probs": log_probs,
        "rollout_routed_experts": routed_experts,
        "rollout_indexer_topk": indexer_topk,
        # Optional fields with sentinel values to satisfy Sample schema if accessed
        "index": 0,
        "prompt": "",
        "response": "",
        "loss_mask": [1] * len(full_ids),
        "reward": 0.0,
        "remove_sample": False,
        "weight_versions": [],
        "label": "",
    }
    return {"rollout_id": 59, "samples": [sample]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-json", required=True, help="json with key full_ids: [int,...]")
    ap.add_argument("--server", default="http://127.0.0.1:30000")
    ap.add_argument("--out-response", required=True, help="raw sgl_response.json")
    ap.add_argument("--out-rollout", required=True, help="synthetic rollout .pt for run_megatron")
    ap.add_argument("--num-layers", type=int, default=43)
    ap.add_argument("--moe-topk", type=int, default=6)
    ap.add_argument("--num-c4-layers", type=int, required=True, help="# of layers with compress_ratio=4")
    ap.add_argument("--indexer-topk", type=int, default=512)
    ap.add_argument("--response-length", type=int, default=32, help="logical decode length for synthetic sample")
    args = ap.parse_args()

    ids = json.load(open(args.ids_json))["full_ids"]
    print(f"prefill input_ids len={len(ids)}", flush=True)

    wait_health(args.server)
    resp = post_generate(args.server, ids, temperature=0.0, max_new_tokens=1)

    meta = resp["meta_info"]
    itp = meta["input_token_logprobs"]
    print(f"input_token_logprobs len: {len(itp)} first3: {itp[:3]}", flush=True)

    # SGLang's routing buffer includes the generated tokens (max_new_tokens=1) too,
    # so the token-dim is len(ids) + 1 - 1 = len(ids). For prefill-only with max_new=1.
    routing_token_dim = len(ids)
    routed_experts, indexer = decode_routing(
        meta,
        num_tokens_minus_1=routing_token_dim,
        num_layers=args.num_layers,
        moe_topk=args.moe_topk,
        num_c4_layers=args.num_c4_layers,
        indexer_topk=args.indexer_topk,
    )
    print(f"routed_experts shape: {routed_experts.shape}, dtype={routed_experts.dtype}, "
          f"unique experts (first sample, layer 0): {np.unique(routed_experts[0, 0]).tolist()}", flush=True)
    print(f"indexer_topk shape: {indexer.shape}, dtype={indexer.dtype}, "
          f"unique positions (first sample, layer 0): {np.unique(indexer[0, 0])[:10].tolist()}...", flush=True)

    with open(args.out_response, "w") as f:
        # Drop the heavy base64 strings to keep response file small
        meta_lite = {k: v for k, v in meta.items() if k not in ("routed_experts", "indexer_topk")}
        json.dump({**resp, "meta_info": meta_lite}, f, indent=2)
    print(f"saved response (without base64 routing) → {args.out_response}", flush=True)

    log_probs = [e[0] for e in itp if e[0] is not None]
    rollout = build_synthetic_rollout(
        full_ids=ids,
        response_length=args.response_length,
        log_probs=log_probs,
        routed_experts=routed_experts,
        indexer_topk=indexer,
    )
    torch.save(rollout, args.out_rollout)
    print(f"saved synthetic rollout → {args.out_rollout}", flush=True)


if __name__ == "__main__":
    main()
