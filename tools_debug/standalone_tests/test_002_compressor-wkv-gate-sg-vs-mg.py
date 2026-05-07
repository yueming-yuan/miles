"""test_002_compressor-wkv-gate-sg-vs-mg.py

Question: how much numerical divergence does SG's BF16 wkv_gate weight introduce
vs. MG's FP32 wkv/wgate weight, on the V4 compressor's first GEMM?

Setup:
  Load actual compressor.wkv + compressor.wgate weights (FP32) from HF iter59 ckpt.
  Concatenate row-wise (matches sg's fused wkv_gate). Use a real `input_layernorm`
  BF16 tensor from sg2 dump as input (T, hidden).

  Three GEMMs:
    1. ref_fp64  : input.double() @ weight.double().T → FP64 → cast to FP32  (truth)
    2. mg_path   : input.float()  @ weight (FP32) .T  → FP32                  (matches MG + reference)
    3. sg_path   : input(BF16) @ weight.to(bf16).T → FP32                     (matches SG cublas BF16×BF16→FP32)

  Compare each path against ref_fp64.
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def load_fused_wkv_gate(ckpt_dir: Path, layer: int) -> torch.Tensor:
    """Load wkv + wgate from HF ckpt and cat row-wise (matches sg's fused wkv_gate)."""
    idx_path = ckpt_dir / "model.safetensors.index.json"
    weight_map = json.load(open(idx_path))["weight_map"]
    keys = (f"model.layers.{layer}.self_attn.compressor.wkv.weight",
            f"model.layers.{layer}.self_attn.compressor.wgate.weight")
    parts = []
    for key in keys:
        sf = ckpt_dir / weight_map[key]
        with safe_open(str(sf), framework="pt") as f:
            parts.append(f.get_tensor(key))
    assert all(p.dtype == torch.float32 for p in parts), [p.dtype for p in parts]
    return torch.cat(parts, dim=0)  # (2*coff*head_dim, hidden)


def stat(diff: torch.Tensor, label: str) -> None:
    a = diff.float().abs().flatten()
    n = a.numel()
    p99 = a.kthvalue(max(1, int(0.99 * n))).values.item()
    p50 = a.kthvalue(max(1, n // 2)).values.item()
    print(f"  {label:30s} max={a.max().item():.4e}  mean={a.mean().item():.4e}  p50={p50:.4e}  p99={p99:.4e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="/storage/yueming/iter59-hf-fp8")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--input-pt", required=True, help="real input dump (.pt with key 'value', BF16, ending in (T, hidden))")
    args = ap.parse_args()

    ckpt = Path(args.ckpt_dir)
    weight_fp32 = load_fused_wkv_gate(ckpt, args.layer)
    print(f"Weight: {tuple(weight_fp32.shape)} dtype={weight_fp32.dtype}  (loaded as FP32 from ckpt)")
    out_dim, hidden = weight_fp32.shape

    raw = torch.load(args.input_pt, weights_only=False)["value"].detach()
    print(f"Input raw: {tuple(raw.shape)}  dtype={raw.dtype}")
    if raw.dim() > 2:
        raw = raw.reshape(-1, raw.shape[-1])
    if raw.shape[-1] != hidden:
        # Take the slice matching the weight's input dim (sg's layer_input has hc_mult; flatten it)
        raw = raw[..., :hidden] if raw.shape[-1] >= hidden else raw
    x_bf16 = raw.contiguous().to(torch.bfloat16).cuda()
    weight_fp32 = weight_fp32.cuda()
    print(f"Input shaped: {tuple(x_bf16.shape)}  dtype={x_bf16.dtype}")
    print(f"Input stats: max={x_bf16.float().abs().max().item():.4f}  mean_abs={x_bf16.float().abs().mean().item():.4e}")
    print()

    print("Computing three paths...")

    # ref: FP64 truth
    out_ref = (x_bf16.double() @ weight_fp32.double().T).to(torch.float32)
    print(f"  out_ref:  shape={tuple(out_ref.shape)} dtype={out_ref.dtype}")

    # mg path: FP32 input × FP32 weight → FP32
    out_mg = (x_bf16.float() @ weight_fp32.T).to(torch.float32)
    print(f"  out_mg:   shape={tuple(out_mg.shape)} dtype={out_mg.dtype}")

    # sg path: BF16 input × BF16 weight → FP32 (cublas)
    weight_bf16 = weight_fp32.to(torch.bfloat16)
    out_sg = torch.mm(x_bf16, weight_bf16.T, out_dtype=torch.float32)
    print(f"  out_sg:   shape={tuple(out_sg.shape)} dtype={out_sg.dtype}")
    print()

    print("=== |X - ref_fp64| ===")
    stat(out_mg - out_ref, "mg_path - ref")
    stat(out_sg - out_ref, "sg_path - ref")
    print()

    print("=== mg vs sg directly ===")
    stat(out_mg - out_sg, "mg_path - sg_path")
    print()

    def per_token_max(d):
        return d.float().abs().reshape(d.shape[0], -1).max(dim=1).values

    pt_mg = per_token_max(out_mg - out_ref)
    pt_sg = per_token_max(out_sg - out_ref)
    print("=== Per-token max-feature |diff vs ref| ===")
    print(f"  mg vs ref:  max={pt_mg.max().item():.4e}  mean={pt_mg.mean().item():.4e}")
    print(f"  sg vs ref:  max={pt_sg.max().item():.4e}  mean={pt_sg.mean().item():.4e}")


if __name__ == "__main__":
    main()
