"""Direct dump comparison: per-layer diff between sglang and megatron.

For each common (name, layer_id) pair, gather rank-shards, concat along seq, compare.
Skips tensors with shape mismatches that need TP-aware reshape.
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import torch


_NAME_RE = re.compile(
    r"step=(\d+)___rank=(\d+)___dump_index=(\d+)___name=([^_]+(?:_[^_=]+)*?)(___compress_ratio=([^_]+))?(___recompute_status=[^_]+)?(___layer_id=(\d+))?\.pt$"
)


def parse_filename(fn: str) -> dict | None:
    base = fn.split("/")[-1]
    parts: dict[str, str] = {}
    for seg in base.split("___"):
        if "=" in seg:
            k, v = seg.split("=", 1)
            v = v[: -len(".pt")] if v.endswith(".pt") else v
            parts[k] = v
    return parts


def gather_index(root: Path) -> dict:
    """Per (name, compress_ratio, step) → list of (sorted_dump_index_within_name, rank, layer_id, filepath).

    For sglang: layer_id is _none_, dumps are ordered by dump_index per rank.
    For megatron: layer_id is per-file. We treat layer_id as a sort key.
    Strategy: per (name, compress_ratio, step), group by rank, sort by (layer_id_int_or_dump_index)
    then assign sequential layer_idx 0..N per rank. Both sides should yield identical layer_idx orderings if forward order is consistent.
    """
    by_name: dict[tuple, list[tuple[int, int, str, Path]]] = defaultdict(list)
    for f in root.rglob("*.pt"):
        p = parse_filename(f.name)
        if not p or "name" not in p:
            continue
        name = p["name"]
        compress_ratio = p.get("compress_ratio", "_none_")
        step = int(p.get("step", "0"))
        rank = int(p.get("rank", "0"))
        layer_id = p.get("layer_id", "_none_")
        dump_index = int(p.get("dump_index", "0"))
        by_name[(name, compress_ratio, step)].append((dump_index, rank, layer_id, f))

    # Convert to (name, compress_ratio, step, layer_idx, rank) → filepath
    index: dict[tuple, dict[int, Path]] = defaultdict(dict)
    for (name, cr, step), items in by_name.items():
        # group by rank, sort each rank's items by layer_id (int) if available, else dump_index
        by_rank: dict[int, list[tuple[int, str, Path]]] = defaultdict(list)
        for (di, r, lid, f) in items:
            by_rank[r].append((di, lid, f))
        for r, lst in by_rank.items():
            lst.sort(key=lambda t: (int(t[1]) if t[1] != "_none_" else t[0]))
            for layer_idx, (di, lid, f) in enumerate(lst):
                index[(name, cr, step, layer_idx)][r] = f
    return index


def load_value(p: Path) -> torch.Tensor:
    d = torch.load(p, weights_only=False, map_location="cpu")
    v = d["value"]
    return v.detach().cpu() if hasattr(v, "device") else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", default="/tmp/compare_simple.txt")
    ap.add_argument("--diff-threshold", type=float, default=1e-4)
    args = ap.parse_args()

    bl_idx = gather_index(Path(args.baseline))
    tg_idx = gather_index(Path(args.target))

    bl_keys = set(bl_idx.keys())
    tg_keys = set(tg_idx.keys())
    common = bl_keys & tg_keys
    only_bl = bl_keys - tg_keys
    only_tg = tg_keys - bl_keys

    print(f"baseline keys: {len(bl_keys)}")
    print(f"target keys:   {len(tg_keys)}")
    print(f"common keys:   {len(common)}")

    out = open(args.out, "w")
    out.write(f"baseline_root={args.baseline}\n")
    out.write(f"target_root={args.target}\n")
    out.write(f"baseline_n={len(bl_keys)} target_n={len(tg_keys)} common={len(common)}\n\n")

    rows = []
    for key in sorted(common):
        name, compress_ratio, step, layer_id = key
        bl_files = bl_idx[key]
        tg_files = tg_idx[key]

        bl_tensors = []
        for r in sorted(bl_files):
            try:
                bl_tensors.append(load_value(bl_files[r]))
            except Exception as e:
                rows.append((name, layer_id, step, "load_fail_baseline", str(e)[:60]))
                bl_tensors = None
                break
        if bl_tensors is None:
            continue
        tg_tensors = []
        for r in sorted(tg_files):
            try:
                tg_tensors.append(load_value(tg_files[r]))
            except Exception as e:
                rows.append((name, layer_id, step, "load_fail_target", str(e)[:60]))
                tg_tensors = None
                break
        if tg_tensors is None:
            continue

        # Concatenate sglang sp-shards along dim 0 (seq) — sglang is THD with t-shard
        # Concatenate megatron sp-shards along dim 0 (seq) — megatron is also THD with t-shard
        if len(bl_tensors) == 0 or len(tg_tensors) == 0:
            continue

        try:
            if len(bl_tensors) == 1:
                bl_full = bl_tensors[0]
            else:
                bl_full = torch.cat(bl_tensors, dim=0)
            if len(tg_tensors) == 1:
                tg_full = tg_tensors[0]
            else:
                tg_full = torch.cat(tg_tensors, dim=0)
        except Exception as e:
            rows.append((name, layer_id, step, "cat_fail", str(e)[:60]))
            continue

        # squeeze size-1 dims
        bl_full = bl_full.squeeze()
        tg_full = tg_full.squeeze()

        if bl_full.shape != tg_full.shape:
            rows.append(
                (name, layer_id, step, f"shape_mismatch", f"bl={tuple(bl_full.shape)} tg={tuple(tg_full.shape)}")
            )
            continue

        # Compute abs diff
        bl_f = bl_full.float().cpu()
        tg_f = tg_full.float().cpu()
        diff = (bl_f - tg_f).abs()
        rel = diff / (tg_f.abs() + 1e-9)
        rows.append(
            (
                name,
                layer_id,
                step,
                "ok",
                f"shape={tuple(bl_full.shape)} mean_abs={diff.mean().item():.4e} max_abs={diff.max().item():.4e} mean_rel={rel.mean().item():.4e} max_rel={rel.max().item():.4e}",
            )
        )

    rows.sort(key=lambda r: (r[1] if r[1] != "_none_" else "z", r[0], r[2]))
    out.write("layer_id name step status info\n")
    for name, layer_id, step, status, info in rows:
        out.write(f"{layer_id:>3} {name:35s} step={step:>2} {status:20s} {info}\n")

    out.write(f"\nbaseline-only ({len(only_bl)}):\n")
    for k in sorted(only_bl)[:50]:
        out.write(f"  {k}\n")
    out.write(f"\ntarget-only ({len(only_tg)}):\n")
    for k in sorted(only_tg)[:50]:
        out.write(f"  {k}\n")
    out.close()
    print(f"wrote {args.out}, {len(rows)} rows")

    # Show top divergent ops
    ok_rows = [r for r in rows if r[3] == "ok"]
    print(f"\n{len(ok_rows)} successful comparisons")
    print("Sorted by max_abs (top 20):")
    def parse_max_abs(info):
        m = re.search(r"max_abs=([\d.e+-]+)", info)
        return float(m.group(1)) if m else 0.0
    ok_rows.sort(key=lambda r: -parse_max_abs(r[4]))
    for name, layer_id, step, status, info in ok_rows[:20]:
        print(f"  layer={layer_id} {name} step={step} {info}")


if __name__ == "__main__":
    main()
