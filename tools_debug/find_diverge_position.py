"""Find the position with the largest divergence at layer 0.

Helps identify which token has the catastrophic kernel divergence.
"""
from pathlib import Path
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


def get_sg_layer0_concat(name: str) -> torch.Tensor:
    """Concat sg rank-0 chunks across steps."""
    by_step: dict[int, tuple[int, Path]] = {}
    for f in SG.iterdir():
        p = parse(f.name)
        if p.get("name") != name or int(p.get("rank", "0")) != 0:
            continue
        if p.get("compress_ratio", "_none_") != "_none_":
            continue
        di = int(p["dump_index"])
        step = int(p["step"])
        if step not in by_step or di < by_step[step][0]:
            by_step[step] = (di, f)
    chunks = [load(f) for _, f in sorted(by_step.values())]
    nonempty = [c for c in chunks if c.shape[0] > 0]
    return torch.cat(nonempty, dim=0).float()


def get_mg_layer0_concat(name: str) -> torch.Tensor:
    """Concat mg ranks 0..7 in order."""
    by_rank: dict[int, torch.Tensor] = {}
    for f in MG.iterdir():
        p = parse(f.name)
        if p.get("name") != name or p.get("layer_id", "_none_") != "0":
            continue
        if p.get("compress_ratio", "_none_") != "_none_":
            continue
        rank = int(p["rank"])
        by_rank[rank] = load(f).squeeze(1) if load(f).dim() == 4 else load(f)
    return torch.cat([by_rank[r].float() for r in sorted(by_rank.keys())], dim=0)


def main():
    for name in ["pre_mlp_layernorm_output", "mlp_output"]:
        sg = get_sg_layer0_concat(name)
        mg = get_mg_layer0_concat(name)
        # both should be (T, hidden)
        if sg.dim() == 3:
            sg = sg.squeeze(1)
        if mg.dim() == 3:
            mg = mg.squeeze(1)
        n = min(sg.shape[0], mg.shape[0])
        sg, mg = sg[:n], mg[:n]
        diff = (sg - mg).abs()
        # per-position max (over hidden dim)
        per_pos = diff.max(dim=-1).values  # (T,)
        topk = per_pos.topk(20)
        print(f"\n=== {name} ({sg.shape}) ===")
        print(f"abs_max overall = {diff.max().item():.3e}, abs_mean = {diff.mean().item():.3e}")
        print(f"top 20 divergent positions (out of {n}):")
        for v, i in zip(topk.values, topk.indices):
            pos = i.item()
            print(f"  pos {pos:5d}: max_abs_diff={v.item():.3e}")

        # Bucket positions by range
        positions = topk.indices.tolist()
        print(f"  position range distribution: min={min(positions)}, max={max(positions)}, all<2249? {all(p < 2249 for p in positions)}")


if __name__ == "__main__":
    main()
