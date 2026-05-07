"""Summarize comparator jsonl: errors + successes by name + per-layer trend."""
import json
import sys
from collections import defaultdict


path = sys.argv[1] if len(sys.argv) > 1 else "/storage/yueming/dumper-out/v4-iter59-cmp6.jsonl"

errors = []
ok = []
skip = defaultdict(int)
for line in open(path):
    e = json.loads(line)
    if e.get("type") == "comparison_error":
        errors.append((e.get("name"), e.get("exception_message", "")[:240]))
    elif e.get("type") == "comparison_tensor":
        ok.append(e)
    elif e.get("type") == "comparison_skip":
        skip[e.get("name")] += 1

print(f"=== Errors ({len(errors)}) ===")
for n, m in errors:
    print(f"  {n}: {m}")
print()

print(f"=== Successful comparisons ({len(ok)}) ===")
by_name = defaultdict(list)
for e in ok:
    by_name[e.get("name")].append(e.get("diff", {}))
for name in sorted(by_name):
    diffs = sorted(by_name[name], key=lambda d: d.get("mean_abs_diff", 0))
    n = len(diffs)
    lo = diffs[0]
    hi = diffs[-1]
    fmt_lo = f"mean={lo.get('mean_abs_diff', 0):.3e} max={lo.get('max_abs_diff', 0):.3e}"
    fmt_hi = f"mean={hi.get('mean_abs_diff', 0):.3e} max={hi.get('max_abs_diff', 0):.3e}"
    print(f"  {name}: {n} bundles | best [{fmt_lo}] | worst [{fmt_hi}]")

print()
print(f"=== Skipped names (top 15) ===")
for n, c in sorted(skip.items(), key=lambda x: -x[1])[:15]:
    print(f"  {n}: {c}")
