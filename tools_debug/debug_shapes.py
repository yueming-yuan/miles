"""Inspect per-step shapes of attn_q vs layer_input on sglang side."""
import sys, glob, torch
D = sys.argv[1]
for s in [0, 1, 2, 30, 32]:
    files = sorted(glob.glob(f"{D}/step={s}___rank=0___dump_index=*___name=attn_q___*.pt"))
    if files:
        d = torch.load(files[0], weights_only=False)
        print(f"attn_q step={s}: shape={tuple(d['value'].shape)} dump_index_count={len(files)}")
    li = sorted(glob.glob(f"{D}/step={s}___rank=0___dump_index=*___name=layer_input___*.pt"))
    if li:
        d = torch.load(li[0], weights_only=False)
        print(f"layer_input step={s}: shape={tuple(d['value'].shape)} dump_index_count={len(li)}")
