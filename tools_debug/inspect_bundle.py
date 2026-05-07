"""Inspect a comparator bundle's file list per step."""
import json
import re
import sys
from collections import defaultdict

path = sys.argv[1]
target_name = sys.argv[2]

for line in open(path):
    e = json.loads(line)
    if e.get("type") != "comparison_tensor":
        continue
    if e.get("name") != target_name:
        continue
    rbi = e.get("raw_bundle_info", {})
    yf = rbi.get("y", {}).get("files", [])
    xf = rbi.get("x", {}).get("files", [])
    yfn = yf[0].get("filename", "") if yf else ""
    layer_m = re.search(r"layer_id=(\d+)", yfn)
    layer = layer_m.group(1) if layer_m else "?"
    print(f"\n=== {target_name} target_layer={layer} ===")
    print(f"y_files: {len(yf)} (one per rank)")
    print(f"x_files: {len(xf)}")

    # group x files by step
    by_step = defaultdict(list)
    for f in xf:
        fn = f.get("filename", "")
        sm = re.search(r"step=(\d+)", fn)
        step = int(sm.group(1)) if sm else -1
        by_step[step].append((tuple(f.get("shape", ())), f.get("rank"), fn[-90:]))
    for s in sorted(by_step):
        items = by_step[s]
        print(f"  step={s}: {len(items)} files, sample shape={items[0][0]} rank={items[0][1]}")
        for shape, rank, fn in items[:3]:
            print(f"    {fn[-100:]}")
    break
