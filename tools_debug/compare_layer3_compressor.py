"""Compare layer 3 (first cr=128 compressor) between sglang and megatron.

Layer 3 is where divergence first emerges in mlp_output.
Drill into compressor + post-compressor names.
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


def get_sg_per_layer(name: str, compress_ratio: str) -> dict[int, torch.Tensor]:
    """Sg rank-0 chunks per layer (positional layer assignment)."""
    rank0_files: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    for f in SG.iterdir():
        p = parse(f.name)
        if p.get("name") != name:
            continue
        if int(p.get("rank", "0")) != 0:
            continue
        if p.get("compress_ratio", "_none_") != compress_ratio:
            continue
        rank0_files[int(p["step"])].append((int(p["dump_index"]), f))
    by_step_layer: dict[int, dict[int, Path]] = {}
    for step, lst in rank0_files.items():
        lst.sort()
        by_step_layer[step] = {i: f for i, (di, f) in enumerate(lst)}
    if not by_step_layer:
        return {}
    n_layers = max(len(v) for v in by_step_layer.values())
    out: dict[int, torch.Tensor] = {}
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


def get_mg_per_layer(name: str, compress_ratio: str) -> dict[int, torch.Tensor]:
    """Mg dumps per layer. Some are tp-replicated (rank 0 has full seq), some are sp-sharded.
    Take rank 0's tensor and squeeze leading 1-dims."""
    by_layer: dict[int, torch.Tensor] = {}
    for f in MG.iterdir():
        p = parse(f.name)
        if p.get("name") != name:
            continue
        layer_id = p.get("layer_id", "_none_")
        if layer_id == "_none_":
            continue
        if p.get("compress_ratio", "_none_") != compress_ratio:
            continue
        if int(p["rank"]) != 0:
            continue
        t = load(f).float()
        # squeeze size-1 dims from front (batch dim)
        while t.dim() > 1 and t.shape[0] == 1:
            t = t.squeeze(0)
        by_layer[int(layer_id)] = t
    return by_layer


def compare_at_layer(name: str, compress_ratio: str, target_layer: int):
    sg = get_sg_per_layer(name, compress_ratio)
    mg = get_mg_per_layer(name, compress_ratio)

    print(f"\n=== {name} (cr={compress_ratio}) ===")
    print(f"  sg layers found: {sorted(sg.keys())}")
    print(f"  mg layers found: {sorted(mg.keys())}")

    if target_layer not in sg or target_layer not in mg:
        # for sglang we mapped positionally — i-th occurrence of name → layer i
        # but sglang's layer indexing within `name` filter is over ALL layers having that compress_ratio
        # for mg, layer_id is the original layer_id
        # sgl layers are [0, 1, ..., k-1] where k = num layers with this compress_ratio
        # mg layers are the actual layer_ids with this compress_ratio
        # So we map positional: sgl[i] corresponds to mg's i-th layer with this compress_ratio
        mg_layers_sorted = sorted(mg.keys())
        if not sg or len(sg) > len(mg_layers_sorted):
            print(f"  cannot align (sgl={len(sg)} positions, mg={len(mg_layers_sorted)} layers)")
            return
        # mg's actual layer_ids with this compress_ratio
        # find the index of target_layer in mg_layers_sorted
        if target_layer not in mg_layers_sorted:
            print(f"  target_layer {target_layer} not in mg keys")
            return
        target_idx = mg_layers_sorted.index(target_layer)
        if target_idx >= len(sg):
            print(f"  sgl doesn't have {target_idx}-th occurrence")
            return
        sg_t = sg[target_idx]
        mg_t = mg[target_layer]
    else:
        sg_t = sg[target_layer]
        mg_t = mg[target_layer]

    print(f"  sg shape={tuple(sg_t.shape)}, mg shape={tuple(mg_t.shape)}")
    if sg_t.shape != mg_t.shape:
        # try to align
        n = min(sg_t.shape[0], mg_t.shape[0])
        sg_t = sg_t[:n]
        mg_t = mg_t[:n]
        if sg_t.shape != mg_t.shape:
            print(f"  cannot align tail dims; comparing aborted")
            return
    diff = (sg_t - mg_t).abs()
    per_pos = diff.flatten(start_dim=1).max(dim=1).values if diff.dim() > 1 else diff
    # exclude position 2249 (boundary) for honest signal
    n_pos = per_pos.shape[0]
    mask = torch.ones(n_pos, dtype=torch.bool)
    if n_pos > 2249:
        mask[2249] = False
    per_pos_excl = per_pos[mask]
    print(
        f"  abs_max={diff.max().item():.3e}  abs_max_excl_2249={per_pos_excl.max().item():.3e}  "
        f"abs_mean={diff.mean().item():.3e}"
    )


def main():
    print("=== Compressor inspection at first cr=4 (layer 2) and first cr=128 (layer 3) layers ===")

    for cr, target_layer in [("4", 2), ("128", 3)]:
        print(f"\n## compress_ratio={cr} layer={target_layer}")
        for name in [
            "compressor_kv_score",
            "compress_forward_out",
            "compress_fused_norm_rope_out",
            "compress_final_out",
        ]:
            compare_at_layer(name, cr, target_layer)


if __name__ == "__main__":
    main()
