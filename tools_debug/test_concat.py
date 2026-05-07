"""Test concat across steps for attn_q vs layer_input."""
import sys, glob, torch
D = sys.argv[1]
for name in ["attn_q", "layer_input"]:
    print(f"\n=== {name} ===")
    by_step = {}
    for s in range(35):
        files = sorted(glob.glob(f"{D}/step={s}___rank=0___dump_index=*___name={name}___*.pt"))
        if not files: continue
        d = torch.load(files[0], weights_only=False)
        by_step[s] = d["value"]
        print(f"  step={s}: shape={tuple(d['value'].shape)}")
    print(f"  total steps: {len(by_step)}")
    try:
        cat = torch.cat([by_step[s] for s in sorted(by_step)], dim=0)
        print(f"  concat shape: {tuple(cat.shape)}")
    except Exception as e:
        print(f"  concat error: {e}")
