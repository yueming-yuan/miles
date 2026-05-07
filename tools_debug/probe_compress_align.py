"""Quick probe: how do sg and mg compress_final_out align at the row level?"""
import torch
from pathlib import Path


def parse_meta(fn):
    out = {}
    for seg in fn.split("___"):
        if "=" in seg:
            k, v = seg.split("=", 1)
            v = v[:-3] if v.endswith(".pt") else v
            out[k] = v
    return out


sg_dir = Path("/storage/yueming/dumper-out/v4-iter59-sg-fix0506/dump_20260506_024058_992864")
mg_dir = Path("/storage/yueming/dumper-out/v4-iter59-mg-fix0505/standalone")

sg_files = sorted([
    (int(parse_meta(f.name)["dump_index"]), f)
    for f in sg_dir.iterdir()
    if parse_meta(f.name).get("name") == "compress_final_out"
    and parse_meta(f.name).get("rank") == "0"
    and parse_meta(f.name).get("step") == "0"
    and parse_meta(f.name).get("compress_ratio") == "4"
])
sg_t = None
for di, f in sg_files:
    t = torch.load(f, weights_only=False)["value"]
    if t.shape[-1] == 512:
        sg_t = t.float()
        print(f"sg L2 attn compress_final_out: dump_idx={di} shape={tuple(t.shape)}")
        break

mg_t = None
for f in mg_dir.iterdir():
    m = parse_meta(f.name)
    if m.get("name") == "compress_final_out" and m.get("layer_id") == "2" and m.get("rank") == "0" and m.get("compress_ratio") == "4":
        t = torch.load(f, weights_only=False)["value"]
        if t.shape[-1] == 512:
            mg_t = t.squeeze(0).float()
            print(f"mg L2 attn compress_final_out: shape={tuple(t.shape)} squeezed={tuple(mg_t.shape)}")
            break

print()
print("sg first 3 rows, first 5 cols:")
print(sg_t[:3, :5])
print()
print("mg first 3 rows, first 5 cols:")
print(mg_t[:3, :5])
print()
for stride in (1, 2, 3, 4, 8):
    print(f"check sg[i*{stride}] vs mg[i] for i=0..3:")
    for i in range(4):
        if i * stride < sg_t.shape[0]:
            diff = (sg_t[i * stride] - mg_t[i]).abs().max().item()
            print(f"  i={i}: sg[{i*stride}] vs mg[{i}] max_diff={diff:.4e}")
