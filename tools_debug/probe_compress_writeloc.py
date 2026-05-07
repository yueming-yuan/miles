"""Probe: do sg's per-query compress_final_out rows correspond to mg's per-block rows
via the write_loc mapping?

Test: for sg row t, is sg_compress_final_out[t] ≈ mg_compress_final_out[write_loc[t]]?

Loads:
  - sg compress_final_out at L2 (cr=4 attention compressor, head_dim=512)
  - sg compress_write_loc at L2 (matched cr=4 attention)
  - mg compress_final_out at L2 (cr=4 attention)
"""
import argparse
from pathlib import Path

import torch


def parse_meta(fn):
    out = {}
    for seg in fn.split("___"):
        if "=" in seg:
            k, v = seg.split("=", 1)
            v = v[:-3] if v.endswith(".pt") else v
            out[k] = v
    return out


def find_sg_attn_compressor_first(sg_dir, name, cr_str):
    """First (smallest dump_idx) attention-compressor file at step 0 rank 0 for given cr.
    For compress_final_out + cr=4: filter by last_dim == 512 (attention compressor head_dim)."""
    matches = []
    for f in sg_dir.iterdir():
        m = parse_meta(f.name)
        if m.get("name") != name or m.get("rank") != "0" or m.get("step") != "0":
            continue
        if m.get("compress_ratio") != cr_str:
            continue
        matches.append((int(m["dump_index"]), f))
    matches.sort()
    return matches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sg-dir", default="/storage/yueming/dumper-out/v4-iter59-sg-fix0506-writeloc")
    ap.add_argument("--mg-dir", default="/storage/yueming/dumper-out/v4-iter59-mg-fix0505/standalone")
    ap.add_argument("--cr", type=int, default=4)
    ap.add_argument("--mg-layer", type=int, default=2)
    args = ap.parse_args()

    cr_str = str(args.cr)
    ratio = args.cr

    # Find sg dump_dir
    sg_dump_dirs = sorted([d for d in Path(args.sg_dir).iterdir() if d.is_dir() and d.name.startswith("dump_")])
    sg_dump = sg_dump_dirs[0]
    print(f"sg dump dir: {sg_dump}")

    # 1) Inspect compress_write_loc
    print("\n=== compress_write_loc inspect (cr=4, all step 0 rank 0) ===")
    wl_files = find_sg_attn_compressor_first(sg_dump, "compress_write_loc", cr_str)
    print(f"  count: {len(wl_files)}")
    for di, f in wl_files[:6]:
        t = torch.load(f, weights_only=False)["value"]
        print(f"  dump_idx={di}: shape={tuple(t.shape)} dtype={t.dtype} min={t.min().item()} max={t.max().item()}")
    print()
    # First write_loc dump: print values
    wl0 = torch.load(wl_files[0][1], weights_only=False)["value"]
    print(f"  first write_loc[:32]: {wl0[:32].tolist()}")
    print(f"  unique values count: {torch.unique(wl0).numel()}")
    print(f"  last 32: {wl0[-32:].tolist()}")
    print()

    # 2) Find sg compress_final_out (cr=4 attention compressor — last_dim==512)
    print("=== compress_final_out (sg L2 attn compressor) ===")
    cf_files = find_sg_attn_compressor_first(sg_dump, "compress_final_out", cr_str)
    sg_cf = None
    for di, f in cf_files:
        t = torch.load(f, weights_only=False)["value"]
        if t.shape[-1] == 512:
            sg_cf = t.float()
            sg_di = di
            print(f"  picked dump_idx={di}: shape={tuple(t.shape)} (head_dim=512 = attn compressor)")
            break

    # 3) Find sg compress_write_loc that pairs with that compress_final_out
    # write_loc dump_idx should be just before compress_final_out's dump_idx in execution order
    # Pick the wl_files entry with largest dump_idx < sg_di
    sg_wl = None
    for di, f in reversed(wl_files):
        if di < sg_di:
            sg_wl = torch.load(f, weights_only=False)["value"]
            print(f"  paired write_loc dump_idx={di}: shape={tuple(sg_wl.shape)} (just before compress_final_out di={sg_di})")
            break
    if sg_wl is None:
        # fallback: take first
        sg_wl = torch.load(wl_files[0][1], weights_only=False)["value"]

    # 4) Load mg compress_final_out at chosen layer cr attn
    mg_dir = Path(args.mg_dir)
    mg_cf = None
    for f in mg_dir.iterdir():
        m = parse_meta(f.name)
        if m.get("name") == "compress_final_out" and m.get("layer_id") == str(args.mg_layer) and m.get("rank") == "0" and m.get("compress_ratio") == cr_str:
            t = torch.load(f, weights_only=False)["value"]
            if t.shape[-1] == 512:
                mg_cf = t.squeeze(0).float()
                print(f"\n=== mg L{args.mg_layer} cr={cr_str} attn compress_final_out: shape={tuple(t.shape)} squeezed={tuple(mg_cf.shape)}")
                break
    if mg_cf is None:
        print(f"!! mg layer={args.mg_layer} cr={cr_str} attn compressor not found")
        return

    # 5) Test mapping: for sg row t, look up mg row at write_loc[t]
    print("\n=== Mapping test: sg[t] vs mg[write_loc[t]] ===")
    print(f"sg_cf shape: {tuple(sg_cf.shape)}, sg_wl shape: {tuple(sg_wl.shape)}")
    n_test = min(20, sg_cf.shape[0])
    for t in range(n_test):
        wl_t = sg_wl[t].item() if t < sg_wl.numel() else None
        if wl_t is None or wl_t < 0 or wl_t >= mg_cf.shape[0]:
            print(f"  t={t}: write_loc={wl_t} (out of mg range {mg_cf.shape[0]})")
            continue
        diff = (sg_cf[t] - mg_cf[wl_t]).abs().max().item()
        print(f"  t={t}: write_loc={wl_t}, max_abs_diff={diff:.4e}")
    print()
    print(f"=== Hypothesis: sg[(k+1)*{ratio}-1] ≈ mg[k] (block-end mapping) ===")
    diffs = []
    for k in range(min(60, mg_cf.shape[0])):
        t = (k + 1) * ratio - 1
        if t >= sg_cf.shape[0]:
            break
        diff = (sg_cf[t] - mg_cf[k]).abs().max().item()
        diffs.append(diff)
    print(f"  block-end pairs tested: {len(diffs)}")
    print(f"  max over all: {max(diffs):.4e}")
    print(f"  mean: {sum(diffs)/len(diffs):.4e}")
    print(f"  first 10: {[f'{d:.3e}' for d in diffs[:10]]}")
    print()
    print("=== sanity: also try sg[k*ratio] (block-start) ===")
    diffs_start = []
    for k in range(min(20, mg_cf.shape[0])):
        t = k * ratio
        if t >= sg_cf.shape[0]:
            break
        diff = (sg_cf[t] - mg_cf[k]).abs().max().item()
        diffs_start.append(diff)
    print(f"  block-start pairs first 10: {[f'{d:.3e}' for d in diffs_start[:10]]}")
    print()
    print(f"=== values: sg[{ratio-1}] vs mg[0] elementwise (first 10 cols) ===")
    bend = ratio - 1
    print(f"  sg[{bend}][:10]:  {sg_cf[bend][:10].tolist()}")
    print(f"  mg[0][:10]:  {mg_cf[0][:10].tolist()}")
    print(f"  diff [:10]:  {(sg_cf[bend]-mg_cf[0])[:10].tolist()}")


if __name__ == "__main__":
    main()
