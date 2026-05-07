"""Stage 2 — per-token operator-chain ratio analysis (with verified per-op alignment).

For chosen positions, computes per (operator, layer):
  cross_diff = |sg2 - mg1|       (cross-stack)
  intra_diff = |mg1 - mg2|       (intra-mg noise floor)
  ratio = cross_diff / max(intra_diff, eps)

Alignment per op (verified by audit_alignment.py):

| op                        | mg sharding (per rank shape)         | strategy                   |
|---------------------------|--------------------------------------|----------------------------|
| layer_input               | (T_sp, 1, hc, h) SP-sharded          | cat dim 0                  |
| wq_a_out                  | (1, T, q_lora) replicated            | take rank 0                |
| q_lora_after_norm         | (1, T, q_lora) replicated            | take rank 0                |
| wq_b_out                  | (1, T, n_h_local*head_dim) TP-sharded| cat dim -1                 |
| q_heads_after_norm        | (1, T, n_h_local, head_dim) TP-sharded| cat dim 2                 |
| attn_q                    | (1, T, n_h_local, head_dim) TP-sharded| cat dim 2                 |
| wkv_out                   | (1, T, head_dim) replicated          | take rank 0                |
| kv_after_norm             | (1, T, head_dim) replicated          | take rank 0                |
| attn_v                    | (1, T, head_dim) replicated          | take rank 0                |
| compressor_kv_score_attn  | (1, T, 2*coff*512) replicated, cr-tag| take rank 0; SG/MG by shape|
| compressor_kv_score_indexer| (1, T, 2*coff*128) replicated, cr-tag| take rank 0; SG/MG by shape|
| mqa_wo_a_out              | (1, T, n_g_local, o_lora) TP-sharded | cat dim 2                  |
| mqa_wo_b_out              | (1, T, h) replicated                 | take rank 0                |
| attn_output               | (1, T, n_h_local, head_dim) TP-sharded, cr-tag| filter cr; cat dim 2|
| pre_mlp_layernorm_output  | (T_sp, 1, h) SP-sharded               | cat dim 0                  |
| mlp_output                | (T_sp, 1, h) SP-sharded               | cat dim 0                  |

SG side: positional by layer for most ops. wq_a_out/wkv_out are sliced from sg's
fused wqkv_a_out. compressor_kv_score is split by shape into _attn / _indexer
(cr=4 layers have both; cr=128 layers have only _attn).
"""
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


OP_ORDER = [
    "layer_input",
    "wq_a_out",
    "q_lora_after_norm",
    "wq_b_out",
    "q_heads_after_norm",
    "attn_q",
    "wkv_out",
    "kv_after_norm",
    "attn_v",
    "compressor_kv_score_indexer",
    "compressor_kv_score_attn",
    "compress_forward_out_attn",
    "compress_fused_norm_rope_out_attn",
    "compress_final_out_attn",
    "attn_q_in",
    "attn_output",
    "mqa_wo_a_out",
    "mqa_wo_b_out",
    "pre_mlp_layernorm_output",
    "mlp_output",
]


_MG_OP_CONFIG = {
    "layer_input":              dict(name="layer_input",              cr=None,          sharding="sp_t"),
    "wq_a_out":                 dict(name="wq_a_out",                 cr=None,          sharding="replicated"),
    "q_lora_after_norm":        dict(name="q_lora_after_norm",        cr=None,          sharding="replicated"),
    "wq_b_out":                 dict(name="wq_b_out",                 cr=None,          sharding="tp_dim_last"),
    "q_heads_after_norm":       dict(name="q_heads_after_norm",       cr=None,          sharding="tp_dim2"),
    "attn_q":                   dict(name="attn_q",                   cr=None,          sharding="tp_dim2"),
    "wkv_out":                  dict(name="wkv_out",                  cr=None,          sharding="replicated"),
    "kv_after_norm":            dict(name="kv_after_norm",            cr=None,          sharding="replicated"),
    "attn_v":                   dict(name="attn_v",                   cr=None,          sharding="replicated"),
    "mqa_wo_a_out":             dict(name="mqa_wo_a_out",             cr=None,          sharding="tp_dim2"),
    "mqa_wo_b_out":             dict(name="mqa_wo_b_out",             cr=None,          sharding="replicated"),
    "attn_output":              dict(name="attn_output",              cr="match_layer", sharding="tp_dim2"),
    "pre_mlp_layernorm_output": dict(name="pre_mlp_layernorm_output", cr=None,          sharding="sp_t"),
    "mlp_output":               dict(name="mlp_output",               cr=None,          sharding="sp_t"),
    "compressor_kv_score_attn":            dict(name="compressor_kv_score",          cr="match_layer", sharding="replicated", shape_filter="attn"),
    "compressor_kv_score_indexer":         dict(name="compressor_kv_score",          cr="match_layer", sharding="replicated", shape_filter="indexer"),
    "compress_forward_out_attn":           dict(name="compress_forward_out",         cr="match_layer", sharding="replicated", shape_filter="attn_compressor_out"),
    "compress_fused_norm_rope_out_attn":   dict(name="compress_fused_norm_rope_out", cr="match_layer", sharding="replicated", shape_filter="attn_compressor_out"),
    "compress_final_out_attn":             dict(name="compress_final_out",           cr="match_layer", sharding="replicated", shape_filter="attn_compressor_out"),
    "attn_q_in":                           dict(name="attn_q_in",                    cr="match_layer", sharding="tp_dim2"),
    "attn_kv_in":                          dict(name="attn_kv_in",                   cr="match_layer", sharding="replicated"),
}


# ---------- Op-chain patch registry ---------------------------------------
#
# Each named patch list is applied to (OP_ORDER, _MG_OP_CONFIG) to produce a
# new (op_order, op_config) for compute_chain. Patches let us:
#   - skip ops whose values are computed but later discarded (e.g. mg's
#     compress_forward_out / compress_fused_norm_rope_out when grafter
#     overwrites compress_final_out)
#   - insert new ops anchored before/after an existing op (e.g. mark the
#     graft-in boundary explicitly)
#
# Patch schema:
#   {"action": "skip",   "op": <existing op label>}
#   {"action": "insert", "op": <new label>, "anchor": <existing label>,
#    "position": "before" | "after",  "config": <_MG_OP_CONFIG entry>}
#
# To add a new experiment-specific patch, define a new key in OP_PATCHES.
# Select via --patches CLI arg.
OP_PATCHES = {
    "default": [],
    "grafter_e4_mlp_block": [
        # E4 (entire MLP/MoE block swap): mg sends pre_mlp_layernorm_output to sg,
        # sg sends mlp_output back. The ROUTER + EXPERTS + any MoE compute on mg
        # downstream of the layernorm but before the residual add is overwritten
        # to sg's output. mg's MLP-internal dumps (moe_router_logits, moe_topk_ids,
        # etc.) are computed but discarded.
        {"action": "skip", "op": "moe_router_logits"},
        {"action": "skip", "op": "moe_topk_ids"},
        {"action": "insert", "op": "mlp_output_grafted_from_sg",
         "anchor": "mlp_output", "position": "after",
         "config": dict(name="mlp_output", cr=None, sharding="sp_t")},
    ],
    "grafter_e3_attn_block": [
        # E3 (entire attention block swap): mg sends attn_q + attn_v to sg, sg sends
        # attn_output back. Inside the attention block (between attn_q/v and attn_output)
        # mg's intermediate dumps (compressor outputs, indexer outputs, attn_q_in /
        # attn_kv_in / attn_topk_idxs) are computed but DISCARDED — grafter overwrites
        # mg's attn_output with sg's. Skip those dumps in the chain plot.
        {"action": "skip", "op": "compressor_kv_score_indexer"},
        {"action": "skip", "op": "compressor_kv_score_attn"},
        {"action": "skip", "op": "compress_forward_out_attn"},
        {"action": "skip", "op": "compress_fused_norm_rope_out_attn"},
        {"action": "skip", "op": "compress_final_out_attn"},
        {"action": "skip", "op": "attn_q_in"},
        # Insert a marker showing the sg→mg attn_output graft entry point.
        {"action": "insert", "op": "attn_output_grafted_from_sg",
         "anchor": "attn_output", "position": "after",
         "config": dict(name="attn_output", cr="match_layer", sharding="tp_dim2")},
    ],
    "grafter_e2_compressor": [
        # E2 (c4-compressor kernel swap): grafter overwrites mg's compress_final_out
        # with sg's value. Mg's compress_forward_out and compress_fused_norm_rope_out
        # are computed by mg but DISCARDED — they don't influence downstream. Skip
        # them in the chain figure to avoid visual clutter.
        {"action": "skip", "op": "compress_forward_out_attn"},
        {"action": "skip", "op": "compress_fused_norm_rope_out_attn"},
        # Make the graft-in boundary explicit: insert a labeled chain point loading
        # mg-grafter's compress_final_out (which IS sg's grafted-in value, byte-equal
        # to sg's compress_final_out modulo BF16 quantization). Anchored AFTER the
        # original compress_final_out_attn for visual clarity.
        {"action": "insert", "op": "compress_final_out_grafted_from_sg",
         "anchor": "compress_final_out_attn", "position": "after",
         "config": dict(name="compress_final_out", cr="match_layer",
                        sharding="replicated", shape_filter="attn_compressor_out")},
    ],
}


def apply_op_patches(op_order: list[str],
                     op_config: dict[str, dict],
                     patches: list[dict]) -> tuple[list[str], dict[str, dict]]:
    """Apply a list of patches to (op_order, op_config) and return the patched copies."""
    new_order = list(op_order)
    new_config = dict(op_config)
    for p in patches:
        action = p["action"]
        if action == "skip":
            label = p["op"]
            if label in new_order:
                new_order.remove(label)
            new_config.pop(label, None)
        elif action == "insert":
            label = p["op"]
            anchor = p["anchor"]
            position = p.get("position", "after")
            if anchor not in new_order:
                raise ValueError(f"insert anchor {anchor!r} not in op_order")
            if label in new_order:
                raise ValueError(f"insert label {label!r} already in op_order")
            idx = new_order.index(anchor) + (1 if position == "after" else 0)
            new_order.insert(idx, label)
            new_config[label] = p["config"]
        else:
            raise ValueError(f"unknown patch action {action!r}")
    return new_order, new_config


def parse_meta(fn: str) -> dict:
    out = {}
    for seg in fn.split("___"):
        if "=" in seg:
            k, v = seg.split("=", 1)
            v = v[:-3] if v.endswith(".pt") else v
            out[k] = v
    return out


def load_value(p: Path) -> torch.Tensor:
    t = torch.load(p, weights_only=False, map_location="cpu")["value"]
    return (t.detach() if t.requires_grad else t).float()


def build_layer_cr_map(mg_dir: Path) -> dict[int, int]:
    out: dict[int, int] = {}
    for f in mg_dir.iterdir():
        if not f.name.endswith(".pt"):
            continue
        m = parse_meta(f.name)
        if m.get("name") != "attn_output" or "compress_ratio" not in m or "layer_id" not in m:
            continue
        out[int(m["layer_id"])] = int(m["compress_ratio"])
    return out


def expected_compressor_size(cr: int, kind: str) -> int:
    coff = 1 + (cr == 4)
    head_dim = 512 if kind == "attn" else 128
    if kind == "attn_compressor_out":
        return 512
    return 2 * coff * head_dim


def gather_mg_replicated(mg_dir: Path, name: str, layer: int, cr_target: int | None,
                          shape_filter: str | None) -> torch.Tensor | None:
    """Take rank=0 file. Optional cr filter and shape filter (for compressor split)."""
    candidates = []
    for f in mg_dir.iterdir():
        if not f.name.endswith(".pt"):
            continue
        m = parse_meta(f.name)
        if m.get("name") != name or m.get("layer_id") != str(layer) or m.get("rank") != "0":
            continue
        if cr_target is not None and m.get("compress_ratio") != str(cr_target):
            continue
        if cr_target is None and "compress_ratio" in m:
            continue
        candidates.append((int(m["dump_index"]), f, m))
    if not candidates:
        return None
    if shape_filter is not None:
        # Compressor: cr=4 has 2 dumps (indexer smaller, attn larger); cr=128 has only attn
        cr_val = int(cr_target) if cr_target else 0
        target_size = expected_compressor_size(cr_val, shape_filter)
        for di, f, m in candidates:
            t = load_value(f)
            if t.shape[-1] == target_size:
                if t.dim() >= 2 and t.shape[0] == 1:
                    t = t.squeeze(0)
                return t
        return None
    candidates.sort()
    t = load_value(candidates[0][1])
    if t.dim() >= 2 and t.shape[0] == 1:
        t = t.squeeze(0)
    return t


def gather_mg_cat(mg_dir: Path, name: str, layer: int, cr_target: int | None,
                   cat_dim: int) -> torch.Tensor | None:
    """Cat all 8 ranks along cat_dim. Optional cr filter."""
    by_rank: dict[int, torch.Tensor] = {}
    for f in mg_dir.iterdir():
        if not f.name.endswith(".pt"):
            continue
        m = parse_meta(f.name)
        if m.get("name") != name or m.get("layer_id") != str(layer):
            continue
        if cr_target is not None and m.get("compress_ratio") != str(cr_target):
            continue
        if cr_target is None and "compress_ratio" in m:
            continue
        rank = int(m["rank"])
        by_rank[rank] = load_value(f)
    if not by_rank:
        return None
    parts = [by_rank[r] for r in sorted(by_rank.keys())]
    full = torch.cat(parts, dim=cat_dim)
    if full.dim() >= 2 and full.shape[0] == 1:
        full = full.squeeze(0)
    return full


def gather_mg(mg_dir: Path, op: str, layer: int, layer_cr_map: dict[int, int],
              op_config: dict[str, dict] | None = None) -> torch.Tensor | None:
    cfg = (op_config or _MG_OP_CONFIG).get(op)
    if cfg is None:
        return None
    cr_target = layer_cr_map.get(layer) if cfg["cr"] == "match_layer" else None
    if cfg["cr"] == "match_layer" and cr_target is None:
        return None
    if op.startswith("compressor_kv_score") and cr_target == 0:
        return None
    if op == "compressor_kv_score_indexer" and cr_target == 128:
        return None
    sharding = cfg["sharding"]
    shape_filter = cfg.get("shape_filter")
    if sharding == "replicated":
        return gather_mg_replicated(mg_dir, cfg["name"], layer, cr_target, shape_filter)
    if sharding == "sp_t":
        return gather_mg_cat(mg_dir, cfg["name"], layer, cr_target, cat_dim=0)
    if sharding == "tp_dim2":
        return gather_mg_cat(mg_dir, cfg["name"], layer, cr_target, cat_dim=2)
    if sharding == "tp_dim_last":
        return gather_mg_cat(mg_dir, cfg["name"], layer, cr_target, cat_dim=-1)
    return None


def load_sg_per_layer_step_concat(sg_dir: Path, name: str, layer_idx_in_step: int) -> torch.Tensor | None:
    """sg side: layer_id NOT in filename. Within each step (rank=0), the
    `layer_idx_in_step`-th file by dump_index corresponds to the layer.
    Concat across steps to recover full T."""
    by_step: dict[int, list[tuple[int, Path]]] = {}
    for f in sg_dir.iterdir():
        if not f.name.endswith(".pt"):
            continue
        m = parse_meta(f.name)
        if m.get("name") != name or m.get("rank") != "0":
            continue
        by_step.setdefault(int(m["step"]), []).append((int(m["dump_index"]), f))
    if not by_step:
        return None
    chunks = []
    for step in sorted(by_step):
        files = sorted(by_step[step])
        if layer_idx_in_step >= len(files):
            continue
        chunks.append(load_value(files[layer_idx_in_step][1]))
    chunks = [c for c in chunks if c.shape[0] > 1]
    if not chunks:
        return None
    return torch.cat(chunks, dim=0)


def build_sg_compressor_layer_index(sg_dir: Path, layer_cr_map: dict[int, int]) -> dict[tuple[int, str], int]:
    """Map (layer_id, kind) → positional index in step 0 for sg compressor_kv_score.
    Walk dump_idx order; cr=4 layers contribute 2 entries (indexer + attn), cr=128 layers
    contribute 1 (attn only)."""
    cr_layers = sorted([L for L, cr in layer_cr_map.items() if cr > 0])
    files = []
    for f in sg_dir.iterdir():
        if not f.name.endswith(".pt"):
            continue
        m = parse_meta(f.name)
        if m.get("name") != "compressor_kv_score" or m.get("rank") != "0" or m.get("step") != "0":
            continue
        files.append((int(m["dump_index"]), f, m))
    files.sort()
    index: dict[tuple[int, str], int] = {}
    pos = 0
    for L in cr_layers:
        cr = layer_cr_map[L]
        n = 2 if cr == 4 else 1
        layer_files = files[pos:pos + n]
        for i, (di, f, m) in enumerate(layer_files):
            t = load_value(f)
            attn_size = expected_compressor_size(cr, "attn")
            indexer_size = expected_compressor_size(cr, "indexer")
            if t.shape[-1] == attn_size:
                index[(L, "attn")] = pos + i
            elif t.shape[-1] == indexer_size:
                index[(L, "indexer")] = pos + i
        pos += n
    return index


def load_sg(sg_dir: Path, op: str, layer: int,
             layer_cr_map: dict[int, int],
             sg_compressor_idx: dict[tuple[int, str], int],
             op_config: dict[str, dict] | None = None) -> torch.Tensor | None:
    """Load sg-side tensor for `op` at `layer`. Dispatch is driven by op_config[op] when
    available (so inserted/aliased ops route to the same sg loader as their underlying name)."""
    cfg = (op_config or _MG_OP_CONFIG).get(op, {})
    base_name = cfg.get("name", op)
    shape_filter = cfg.get("shape_filter")

    # SG-only fused-QKV slice: mg has separate wq_a_out / wkv_out, sg has one wqkv_a_out.
    if base_name == "wq_a_out":
        wqkv = load_sg_per_layer_step_concat(sg_dir, "wqkv_a_out", layer)
        return wqkv[..., :1024].contiguous() if wqkv is not None else None
    if base_name == "wkv_out":
        wqkv = load_sg_per_layer_step_concat(sg_dir, "wqkv_a_out", layer)
        return wqkv[..., 1024:].contiguous() if wqkv is not None else None

    # compressor_kv_score: cr=4 layers fire it twice (indexer + attn) under the same name;
    # use the precomputed positional index built by build_sg_compressor_layer_index.
    if base_name == "compressor_kv_score":
        kind = "indexer" if shape_filter == "indexer" else "attn"
        idx = sg_compressor_idx.get((layer, kind))
        return load_sg_per_layer_step_concat(sg_dir, "compressor_kv_score", idx) if idx is not None else None

    # compress_{forward_out, fused_norm_rope_out, final_out}: per-query layout on sg side;
    # take block-end positions to align with mg's per-block layout.
    if base_name in ("compress_final_out", "compress_forward_out", "compress_fused_norm_rope_out"):
        idx = sg_compressor_idx.get((layer, "attn"))
        if idx is None:
            return None
        sg_per_query = load_sg_per_layer_step_concat(sg_dir, base_name, idx)
        if sg_per_query is None:
            return None
        cr = layer_cr_map.get(layer, 0)
        if cr <= 0:
            return None
        return sg_per_query[cr - 1::cr].contiguous()

    return load_sg_per_layer_step_concat(sg_dir, base_name, layer)


def _squeeze_singletons(t: torch.Tensor) -> torch.Tensor:
    while True:
        for d in range(t.dim()):
            if t.shape[d] == 1:
                t = t.squeeze(d)
                break
        else:
            return t


def per_position_diff(a: torch.Tensor, b: torch.Tensor, position: int,
                       metric: str = "cosine") -> tuple[float | None, tuple, tuple]:
    """Returns per-position diff using one of:
      - max: max abs diff over feature dim at position
      - mean: mean abs diff over feature dim at position
      - cosine: 1 - 2*<x,y>/(||x||² + ||y||²) — matches sglang comparator's calc_rel_diff (utils.py:106)
    """
    if a is None or b is None:
        return None, (), ()
    a_orig_shape = tuple(a.shape)
    b_orig_shape = tuple(b.shape)
    a = _squeeze_singletons(a)
    b = _squeeze_singletons(b)
    if a.dim() != b.dim():
        return None, a_orig_shape, b_orig_shape
    slc = tuple(slice(0, min(a.shape[d], b.shape[d])) for d in range(a.dim()))
    a = a[slc]
    b = b[slc]
    if position >= a.shape[0]:
        return None, a_orig_shape, b_orig_shape
    if a[position].numel() == 0:
        return None, a_orig_shape, b_orig_shape
    if metric == "max":
        diff = (a[position] - b[position]).abs()
        return float(diff.max().item()), a_orig_shape, b_orig_shape
    if metric == "mean":
        diff = (a[position] - b[position]).abs()
        return float(diff.mean().item()), a_orig_shape, b_orig_shape
    x = a[position].double()
    y = b[position].double()
    denominator = (x * x + y * y).sum()
    if denominator.item() == 0.0:
        return 0.0, a_orig_shape, b_orig_shape
    sim = 2 * (x * y).sum() / denominator
    return float((1 - sim).item()), a_orig_shape, b_orig_shape


def compute_chain(sg_dir: Path, mg1_dir: Path, mg2_dir: Path, position: int,
                   layers: list[int], metric: str = "cosine",
                   op_order: list[str] | None = None,
                   op_config: dict[str, dict] | None = None) -> dict[str, dict]:
    op_order = op_order or OP_ORDER
    op_config = op_config or _MG_OP_CONFIG
    layer_cr_map = build_layer_cr_map(mg1_dir)
    sg_compressor_idx = build_sg_compressor_layer_index(sg_dir, layer_cr_map)
    out = {}
    for L in layers:
        for op in op_order:
            label = f"L{L}:{op}"
            sg_t = load_sg(sg_dir, op, L, layer_cr_map, sg_compressor_idx, op_config=op_config)
            mg1_t = gather_mg(mg1_dir, op, L, layer_cr_map, op_config=op_config)
            mg2_t = gather_mg(mg2_dir, op, L, layer_cr_map, op_config=op_config)
            pos_lookup = position
            base_name = op_config.get(op, {}).get("name", op)
            if base_name in ("compress_final_out", "compress_forward_out", "compress_fused_norm_rope_out"):
                cr = layer_cr_map.get(L, 0)
                if cr > 0:
                    pos_lookup = position // cr
            cross, sg_shape, mg_shape = per_position_diff(sg_t, mg1_t, pos_lookup, metric=metric)
            intra, _, _ = per_position_diff(mg1_t, mg2_t, pos_lookup, metric=metric)
            out[label] = {"cross": cross, "intra": intra, "sg_shape": sg_shape, "mg_shape": mg_shape}
    return out


def plot_chain(chain: dict, position: int, out_path: str, metric: str = "cosine", eps: float = 0.01):
    labels = list(chain.keys())
    cross = [chain[l]["cross"] if chain[l]["cross"] is not None else np.nan for l in labels]
    intra = [chain[l]["intra"] if chain[l]["intra"] is not None else np.nan for l in labels]
    ratio = [(c / max(i, eps)) if (c is not None and i is not None) else np.nan for c, i in zip(cross, intra)]

    fig_w = max(20.0, 0.20 * len(labels))
    fig, axL = plt.subplots(figsize=(fig_w, 7))
    x = np.arange(len(labels))
    axL.plot(x, cross, marker="o", color="crimson", linewidth=1.6, label=f"cross_diff (sg2 vs mg1)")
    axL.plot(x, intra, marker="s", color="steelblue", linewidth=1.6, label=f"intra_diff (mg1 vs mg2)")
    axL.set_yscale("log")
    axL.set_ylabel(f"{metric}_diff (log)", fontsize=11)
    axL.grid(True, axis="y", alpha=0.3, which="both")
    axR = axL.twinx()
    axR.plot(x, ratio, marker="^", color="darkorange", linewidth=1.2, alpha=0.7, label="ratio = cross / intra")
    axR.set_yscale("log")
    axR.set_ylabel("ratio (log)", fontsize=11, color="darkorange")
    axR.tick_params(axis="y", labelcolor="darkorange")
    axL.set_xticks(x)
    axL.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    axL.set_title(f"Operator chain at token #{position} — metric={metric} — cross, intra, ratio", fontsize=12)
    h1, l1 = axL.get_legend_handles_labels()
    h2, l2 = axR.get_legend_handles_labels()
    axL.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sg-dir", required=True)
    ap.add_argument("--mg1-dir", required=True)
    ap.add_argument("--mg2-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--positions", type=int, nargs="+", default=[2125, 1021])
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--metric", choices=["mean", "max", "cosine"], default="cosine")
    ap.add_argument("--patches", choices=sorted(OP_PATCHES.keys()), default="default",
                    help="Named patch set applied to OP_ORDER and _MG_OP_CONFIG (see OP_PATCHES).")
    args = ap.parse_args()

    sg = Path(args.sg_dir)
    mg1 = Path(args.mg1_dir)
    mg2 = Path(args.mg2_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    op_order, op_config = apply_op_patches(OP_ORDER, _MG_OP_CONFIG, OP_PATCHES[args.patches])
    if args.patches != "default":
        print(f"applied patches={args.patches!r}: op_order={op_order}")

    for p in args.positions:
        print(f"=== position {p} (metric={args.metric}) ===")
        chain = compute_chain(sg, mg1, mg2, p, args.layers, metric=args.metric,
                              op_order=op_order, op_config=op_config)
        for label, d in chain.items():
            c = d["cross"]
            i = d["intra"]
            cstr = f"{c:.4f}" if c is not None else "  -  "
            istr = f"{i:.4f}" if i is not None else "  -  "
            r = (c / max(i, 0.01)) if (c is not None and i is not None) else None
            rstr = f"{r:.1f}x" if r is not None else "  -  "
            sg_sh = d["sg_shape"]
            mg_sh = d["mg_shape"]
            print(f"  {label:38s} cross={cstr}  intra={istr}  ratio={rstr}   sg={sg_sh} mg={mg_sh}")
        suffix = f"_{args.patches}" if args.patches != "default" else ""
        plot_chain(chain, p, str(out / f"stage2_chain_token{p}_{args.metric}{suffix}.png"), metric=args.metric)


if __name__ == "__main__":
    main()
