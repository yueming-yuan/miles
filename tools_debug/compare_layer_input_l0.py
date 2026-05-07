"""Manual compare of `layer_input` at layer 0 between sglang chunks and megatron full-seq.

This is a focused diagnostic that bypasses comparator dim machinery.
"""
import re
from pathlib import Path
from collections import defaultdict

import torch


SG = Path("/storage/yueming/dumper-out/v4-base-worst-sglang/dump_20260504_110000_681864")
MG = Path("/storage/yueming/dumper-out/v4-base-worst-megatron/standalone")


def parse(fn: str) -> dict:
    parts: dict[str, str] = {}
    for seg in fn.split("___"):
        if "=" in seg:
            k, v = seg.split("=", 1)
            v = v[: -len(".pt")] if v.endswith(".pt") else v
            parts[k] = v
    return parts


def load(p: Path) -> torch.Tensor:
    return torch.load(p, weights_only=False, map_location="cpu")["value"]


def get_sglang_layer0_full(name: str) -> dict[int, torch.Tensor]:
    """For sglang, gather all step=0..N chunks for layer 0's `name` per rank, concat."""
    files_by_rank_step: dict[tuple, Path] = {}
    for f in SG.iterdir():
        p = parse(f.name)
        if p.get("name") != name:
            continue
        rank = int(p["rank"])
        step = int(p["step"])
        di = int(p["dump_index"])
        # Layer 0 is the FIRST occurrence of `name` per (rank, step)
        # We pick the smallest dump_index per (rank, step)
        key = (rank, step)
        if key not in files_by_rank_step or di < files_by_rank_step[key][0]:
            files_by_rank_step[key] = (di, f)

    # For each rank, concat chunks across steps in step-order
    result: dict[int, torch.Tensor] = {}
    by_rank: dict[int, list] = defaultdict(list)
    for (rank, step), (di, f) in sorted(files_by_rank_step.items()):
        by_rank[rank].append((step, f))
    for rank, lst in by_rank.items():
        lst.sort(key=lambda x: x[0])
        chunks = [load(f) for _, f in lst]
        result[rank] = chunks
    return result


def get_megatron_layer0(name: str) -> dict[int, torch.Tensor]:
    """For megatron, find layer_id=0 dumps for `name` per rank."""
    by_rank: dict[int, torch.Tensor] = {}
    for f in MG.iterdir():
        p = parse(f.name)
        if p.get("name") != name:
            continue
        if p.get("layer_id", "_none_") != "0":
            continue
        rank = int(p["rank"])
        by_rank[rank] = load(f)
    return by_rank


def main():
    name = "layer_input"
    print(f"=== {name} ===")
    sg = get_sglang_layer0_full(name)
    mg = get_megatron_layer0(name)
    print(f"sglang ranks: {list(sg.keys())}")
    print(f"megatron ranks: {list(mg.keys())}")

    for r in sorted(sg.keys()):
        chunks = sg[r]
        print(f"  sg rank {r}: {len(chunks)} chunks, shapes={[tuple(c.shape) for c in chunks]}, dtype={chunks[0].dtype}")
    for r in sorted(mg.keys()):
        t = mg[r]
        print(f"  mg rank {r}: shape={tuple(t.shape)} dtype={t.dtype}")

    # Compare rank 0
    if 0 in sg and 0 in mg:
        sg_full = torch.cat([c.float() for c in sg[0]], dim=0)
        mg0 = mg[0].float().squeeze()
        print(f"\nsg_full shape: {tuple(sg_full.shape)}, mg0 shape: {tuple(mg0.shape)}")

        # Take the first N positions where N = min seq lengths
        if sg_full.dim() == 3 and mg0.dim() == 3:
            # sg shape (T_sg, ?, H), mg shape (T_mg, ?, H)
            n_compare = min(sg_full.shape[0], mg0.shape[0])
            sg_slice = sg_full[:n_compare]
            mg_slice = mg0[:n_compare]
        elif sg_full.dim() == 2 and mg0.dim() == 2:
            n_compare = min(sg_full.shape[0], mg0.shape[0])
            sg_slice = sg_full[:n_compare]
            mg_slice = mg0[:n_compare]
        else:
            print(f"dim mismatch: sg {sg_full.dim()}D vs mg {mg0.dim()}D")
            return

        if sg_slice.shape != mg_slice.shape:
            print(f"sliced shapes differ: {tuple(sg_slice.shape)} vs {tuple(mg_slice.shape)}")
            return

        diff = (sg_slice - mg_slice).abs()
        rel = diff / (mg_slice.abs() + 1e-9)
        print(f"\nover {n_compare} positions:")
        print(f"  diff stats: mean={diff.mean().item():.4e} max={diff.max().item():.4e} p50={diff.median().item():.4e}")
        print(f"  rel stats:  mean={rel.mean().item():.4e} max={rel.max().item():.4e} p50={rel.median().item():.4e}")

        # Per-position max diff to find divergent token
        if diff.dim() >= 2:
            per_pos = diff.flatten(start_dim=1).max(dim=1).values
            print(f"\nper-position max diff (top 5 by max):")
            top = per_pos.topk(min(5, per_pos.numel()))
            for v, i in zip(top.values, top.indices):
                print(f"  pos {i.item()}: max={v.item():.4e}")


if __name__ == "__main__":
    main()
