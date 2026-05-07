"""Compare canonical-name dumps across all layers.

Identifies the FIRST layer where divergence becomes significant.
"""
from pathlib import Path
from collections import defaultdict

import torch


import os
SG = Path(os.environ.get("CMP_SG", "/storage/yueming/dumper-out/v4-iter59-worst-sglang2/dump_20260504_170520_032864"))
MG = Path(os.environ.get("CMP_MG", "/storage/yueming/dumper-out/v4-iter59-worst-megatron/standalone"))


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


def get_sg_per_layer_concat(name: str) -> dict[int, torch.Tensor]:
    """Sg rank-0 chunks per layer.
    SG dumps don't have layer_id. We use dump_index ordering across steps.
    For each step, we have N occurrences of `name` per rank — these are layers 0..N-1.
    """
    rank0_files: dict[int, list[tuple[int, Path]]] = defaultdict(list)  # step -> [(dump_index, file)]
    for f in SG.iterdir():
        p = parse(f.name)
        if p.get("name") != name:
            continue
        if int(p.get("rank", "0")) != 0:
            continue
        if p.get("compress_ratio", "_none_") != "_none_":
            continue
        di = int(p["dump_index"])
        step = int(p["step"])
        rank0_files[step].append((di, f))

    # For each step, sort by dump_index — i-th becomes layer i
    by_step_layer: dict[int, dict[int, Path]] = {}  # step -> layer_idx -> file
    for step, lst in rank0_files.items():
        lst.sort()
        by_step_layer[step] = {i: f for i, (di, f) in enumerate(lst)}

    # For each layer, concat across steps
    out: dict[int, torch.Tensor] = {}
    if not by_step_layer:
        return out
    n_layers = max(len(v) for v in by_step_layer.values())
    for layer_idx in range(n_layers):
        chunks = []
        for step in sorted(by_step_layer.keys()):
            f = by_step_layer[step].get(layer_idx)
            if f is None:
                continue
            t = load(f)
            if t.shape[0] > 0:
                chunks.append(t)
        if chunks:
            out[layer_idx] = torch.cat(chunks, dim=0).float()
    return out


def get_mg_per_layer_concat(name: str) -> dict[int, torch.Tensor]:
    """Mg has layer_id explicit. Concat ranks 0..7 per layer."""
    by_layer_rank: dict[int, dict[int, torch.Tensor]] = defaultdict(dict)
    for f in MG.iterdir():
        p = parse(f.name)
        if p.get("name") != name:
            continue
        layer_id = p.get("layer_id", "_none_")
        if layer_id == "_none_":
            continue
        if p.get("compress_ratio", "_none_") != "_none_":
            continue
        rank = int(p["rank"])
        layer = int(layer_id)
        by_layer_rank[layer][rank] = load(f)

    out: dict[int, torch.Tensor] = {}
    for layer, by_rank in by_layer_rank.items():
        parts = []
        for r in sorted(by_rank.keys()):
            t = by_rank[r]
            # squeeze any size-1 dim except first
            for dim in range(t.dim() - 1, 0, -1):
                if t.shape[dim] == 1:
                    t = t.squeeze(dim)
            parts.append(t)
        if parts:
            try:
                out[layer] = torch.cat(parts, dim=0).float()
            except Exception:
                pass
    return out


def main():
    for name in ["pre_mlp_layernorm_output", "mlp_output"]:
        sg = get_sg_per_layer_concat(name)
        mg = get_mg_per_layer_concat(name)
        common = sorted(set(sg.keys()) & set(mg.keys()))
        print(f"\n=== {name} (common layers: {len(common)}) ===")
        print("layer  abs_mean       abs_max        abs_max_excl_2249  max_pos")
        for layer in common:
            s = sg[layer]
            m = mg[layer]
            n = min(s.shape[0], m.shape[0])
            s, m = s[:n], m[:n]
            if s.shape != m.shape:
                continue
            diff = (s - m).abs()
            per_pos = diff.flatten(start_dim=1).max(dim=1).values  # (T,)
            max_pos = per_pos.argmax().item()
            # Exclude position 2249 (sg decode boundary)
            mask = torch.ones_like(per_pos, dtype=torch.bool)
            if per_pos.shape[0] > 2249:
                mask[2249] = False
            per_pos_excl = per_pos[mask]
            print(
                f"  {layer:>3}  {diff.mean().item():.3e}  {diff.max().item():.3e}  "
                f"{per_pos_excl.max().item():.3e}     pos={max_pos}"
            )


if __name__ == "__main__":
    main()
