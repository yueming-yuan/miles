import torch, glob
for r in range(8):
    files = sorted(glob.glob(f"/storage/yueming/dumper-out/v4-iter59-worst-sglang2/dump_*/step=0___rank={r}___name=mlp_output*.pt"))
    if files:
        d = torch.load(files[0], weights_only=False)
        v = d["value"]
        print(f"rank {r}: shape={tuple(v.shape)}, nonzero_count={int((v != 0).sum().item())}, abs_sum={float(v.abs().sum().item()):.4e}")
