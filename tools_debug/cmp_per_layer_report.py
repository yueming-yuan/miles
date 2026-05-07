"""Build per-layer table of comparator successful results."""
import json
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "/storage/yueming/dumper-out/v4-iter59-cmp7.jsonl"

ok = []
for line in open(path):
    e = json.loads(line)
    if e.get("type") == "comparison_tensor":
        ok.append(e)

print(f"total comparison_tensor: {len(ok)}")

# Each entry has raw_bundle_info.y.files[*].filename which encodes layer_id, compress_ratio
# Group by (name, layer_id, compress_ratio)
groups = defaultdict(list)
for e in ok:
    name = e.get("name")
    raw = e.get("raw_bundle_info", {})
    target_files = raw.get("y", {}).get("files", [])
    if not target_files:
        continue
    fn = target_files[0].get("filename", "")
    # parse layer_id and compress_ratio from filename
    import re
    layer_m = re.search(r"layer_id=(\d+)", fn)
    cr_m = re.search(r"compress_ratio=(\d+)", fn)
    layer_id = int(layer_m.group(1)) if layer_m else None
    cr = int(cr_m.group(1)) if cr_m else None
    groups[(name, cr)].append((layer_id, e.get("diff", {})))

for (name, cr), entries in sorted(groups.items()):
    cr_label = f" (cr={cr})" if cr is not None else ""
    print(f"\n=== {name}{cr_label} ({len(entries)} layers) ===")
    print("layer_id  mean_abs_diff  max_abs_diff  rel_diff  passed")
    entries.sort(key=lambda x: (x[0] if x[0] is not None else 999))
    for layer_id, d in entries:
        if d is None:
            print(f"  {str(layer_id):>3}  (no diff)")
            continue
        passed = d.get("passed")
        ma = d.get("mean_abs_diff", float("nan"))
        mx = d.get("max_abs_diff", float("nan"))
        rd = d.get("rel_diff", float("nan"))
        print(f"  {str(layer_id):>3}  {ma:.3e}  {mx:.3e}  {rd:.3e}  {passed}")
