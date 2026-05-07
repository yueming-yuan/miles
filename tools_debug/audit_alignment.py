"""Audit every (op, side) — list actual dump shapes per rank/step/dump_idx.

For each op in OP_ORDER, print:
  - sg dumps for this op at the given layer-positional index (across all steps, ranks)
  - mg dumps for this op at this layer (across all ranks, all dump_indices, all cr-tagged variants)

Use this to verify each op's alignment is unambiguous before recomputing stage2 ratios.
"""
import argparse
from pathlib import Path
from collections import defaultdict

import torch


OPS = [
    "layer_input",
    "wqkv_a_out",
    "wq_a_out",
    "q_lora_after_norm",
    "wq_b_out",
    "q_heads_after_norm",
    "attn_q",
    "wkv_out",
    "kv_after_norm",
    "attn_v",
    "compressor_kv_score",
    "mqa_wo_a_out",
    "mqa_wo_b_out",
    "attn_output",
    "pre_mlp_layernorm_output",
    "mlp_output",
]


def parse_meta(fn):
    out = {}
    for seg in fn.split("___"):
        if "=" in seg:
            k, v = seg.split("=", 1)
            v = v[:-3] if v.endswith(".pt") else v
            out[k] = v
    return out


def shape(p):
    t = torch.load(p, weights_only=False)["value"]
    return tuple(t.shape), str(t.dtype)


def audit_sg(sg_dir: Path, op: str, layer: int):
    by_step_per_rank = defaultdict(list)
    for f in sg_dir.iterdir():
        if not f.name.endswith(".pt"): continue
        m = parse_meta(f.name)
        if m.get("name") != op or m.get("rank") != "0": continue
        by_step_per_rank[int(m["step"])].append((int(m["dump_index"]), f, m))
    if not by_step_per_rank:
        print(f"  SG {op}: no rank=0 dumps")
        return
    files_at_step0 = sorted(by_step_per_rank[0])
    n_per_step = len(files_at_step0)
    if layer >= n_per_step:
        print(f"  SG {op}: only {n_per_step} files in step 0; layer {layer} out of range")
        return
    idx, f, m = files_at_step0[layer]
    s, dt = shape(f)
    cr_str = f", cr={m.get('compress_ratio')}" if 'compress_ratio' in m else ""
    print(f"  SG {op} L{layer} (step0 pos {layer}): shape={s}{cr_str}")


def audit_mg(mg_dir: Path, op: str, layer: int):
    matches = []
    for f in mg_dir.iterdir():
        if not f.name.endswith(".pt"): continue
        m = parse_meta(f.name)
        if m.get("name") != op or m.get("layer_id") != str(layer): continue
        matches.append((int(m["dump_index"]), int(m["rank"]), m, f))
    if not matches:
        print(f"  MG {op} L{layer}: no dumps")
        return
    matches.sort()
    by_dumpidx = defaultdict(list)
    for di, r, m, f in matches:
        by_dumpidx[di].append((r, m, f))
    for di in sorted(by_dumpidx):
        ranks = sorted(by_dumpidx[di])
        r, m, f = ranks[0]
        s, dt = shape(f)
        cr_str = f", cr={m.get('compress_ratio')}" if 'compress_ratio' in m else ""
        print(f"  MG {op} L{layer} dump_idx={di}: rank={r} shape={s}{cr_str}  (n_ranks={len(ranks)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sg-dir", required=True)
    ap.add_argument("--mg-dir", required=True)
    ap.add_argument("--layer", type=int, default=2)
    args = ap.parse_args()

    sg = Path(args.sg_dir)
    mg = Path(args.mg_dir)

    for op in OPS:
        print(f"=== {op} ===")
        audit_sg(sg, op, args.layer)
        audit_mg(mg, op, args.layer)
        print()


if __name__ == "__main__":
    main()
