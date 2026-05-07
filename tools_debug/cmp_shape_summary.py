"""Print per-name shape summary from comparator jsonl."""
import json
import sys
path = sys.argv[1]
seen = set()
for line in open(path):
    e = json.loads(line)
    if e.get("type") != "comparison_tensor":
        continue
    n = e.get("name")
    if n in seen:
        continue
    seen.add(n)
    bl = e.get("baseline", {}) or {}
    tg = e.get("target", {}) or {}
    diff = e.get("diff")
    print(
        f"{n}: bl_shape={bl.get('shape')} tg_shape={tg.get('shape')} "
        f"unified={e.get('unified_shape')} mismatch={e.get('shape_mismatch')} "
        f"has_diff={diff is not None}"
    )
