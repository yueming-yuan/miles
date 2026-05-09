"""Programmatic sglang HTTP trigger and response-to-comparator-format conversion.

Replaces ad-hoc curl / shell triggers. The runner imports ``trigger_sg_generate``
to send one prompt to a running sglang server and receive its per-token logprobs;
``write_baseline_logprob_json`` converts the response into the same
``rank_*.json`` schema the megatron worker emits, so a single comparator function
handles both sides.

This module is the single entry point for sglang HTTP interaction; nothing else in
the runner should construct sglang request payloads by hand.
"""

from __future__ import annotations

import dataclasses
import json
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


@dataclasses.dataclass(frozen=True)
class SgTriggerResult:
    """Outcome of one sglang trigger call.

    ``meta_info`` is the raw ``meta_info`` dict returned by sglang. Callers
    typically need ``input_token_logprobs`` (per-position prefill logprobs).
    ``elapsed_s`` is end-to-end wall time for the HTTP request, useful for
    rough sanity-checking that a request actually executed (vs. cached).
    """

    meta_info: dict[str, Any]
    output_token_ids: list[int]
    elapsed_s: float


def trigger_sg_generate(
    *,
    input_token_ids: list[int],
    server_url: str,
    max_new_tokens: int = 1,
    temperature: float = 0.0,
    top_k: int = 1,
    return_logprob: bool = True,
    logprob_start_len: int = 0,
    timeout_s: int = 1800,
) -> SgTriggerResult:
    """POST one ``/generate`` request to a running sglang server.

    Defaults reproduce the canonical V4 debug trigger: temperature 0, top-k 1, one
    new token (forces a single prefill chunk + 1 decode), full prefill logprobs
    returned. Caller is responsible for ensuring the server is ready --
    use ``wait_for_sg_ready`` first.
    """
    payload = {
        "input_ids": input_token_ids,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
        },
        "return_logprob": return_logprob,
        "logprob_start_len": logprob_start_len,
        "top_logprobs_num": 0,
    }
    req = urllib.request.Request(
        f"{server_url.rstrip('/')}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read()
    elapsed = time.time() - t0

    out = json.loads(body)
    if isinstance(out, list):
        out = out[0]
    return SgTriggerResult(
        meta_info=out.get("meta_info", {}),
        output_token_ids=list(out.get("output_ids") or out.get("token_ids") or []),
        elapsed_s=elapsed,
    )


def wait_for_sg_ready(
    *,
    host: str,
    port: int,
    timeout_s: int = 1800,
    poll_interval_s: float = 5.0,
) -> None:
    """Block until ``host:port`` accepts TCP connections, or raise on timeout.

    A bound port is the earliest evidence the sglang server has finished startup;
    polling ``/health`` is unreliable because we run with
    ``SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false`` (avoids triggering
    extraneous forward passes during debug).
    """
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=poll_interval_s):
                return
        except OSError as e:
            last_err = e
            time.sleep(poll_interval_s)
    raise TimeoutError(f"sglang server at {host}:{port} not ready after {timeout_s}s; last error: {last_err}")


def baseline_entries_from_meta(meta_info: dict[str, Any]) -> list[dict]:
    """Convert sglang's ``meta_info.input_token_logprobs`` to comparator entries.

    sglang format: ``input_token_logprobs[i] = [logprob, token_id, top_logprobs]``,
    where ``logprob`` is ``log P(tokens[i] | tokens[0..i-1])``. Index 0 carries
    ``[None, first_token, None]`` (no prefix logprob defined).

    Megatron's logprob_comparator uses labels-shifted convention: at
    ``global_position = p`` the logprob predicts ``tokens[p+1]``. So sglang index
    ``i`` corresponds to ``global_position = i - 1``, ``token_id = tokens[i]``.
    """
    entries: list[dict] = []
    raw = meta_info.get("input_token_logprobs") or []
    for i, item in enumerate(raw):
        if i == 0:
            continue
        if item is None:
            continue
        lp, tok_id = item[0], item[1]
        if lp is None:
            continue
        entries.append(
            {
                "global_position": i - 1,
                "token_id": int(tok_id),
                "logprob": float(lp),
                "is_valid": True,
            }
        )
    return entries


def write_baseline_logprob_json(
    *,
    meta_info: dict[str, Any],
    output_path: Path,
) -> int:
    """Write a sglang response's per-token logprobs in run_megatron's JSON format.

    The output file matches the schema produced by
    ``miles/utils/debug_utils/run_megatron/worker/main.py`` so
    ``logprob_comparator.compare_logprobs`` can consume sglang and megatron sides
    with the same code path. Returns the number of valid positions written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    entries = baseline_entries_from_meta(meta_info)
    payload = {
        "rank": 0,
        "tp_size": 1,
        "cp_size": 1,
        "pp_size": 1,
        "logprob_entries": [entries],
    }
    output_path.write_text(json.dumps(payload, indent=2))
    return len(entries)
