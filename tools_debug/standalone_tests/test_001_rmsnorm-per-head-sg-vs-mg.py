"""test_001_rmsnorm-per-head-sg-vs-mg.py

Question: is MG's BF16-throughout per-head RMSNorm the numerical injection point
for the L0 q_heads_after_norm cross-stack divergence (~0.18 max abs diff observed)?

Setup:
  Take a real wq_b_out BF16 tensor (post Q-LoRA-B linear, shape (T, n_heads*head_dim))
  from a sg2 dump. Reshape to (T, n_heads, head_dim). Apply three implementations
  of weightless per-head RMSNorm:
    1. ref_fp32:  cast input to FP32, do everything in FP32, cast result to BF16
    2. mg_path:   exact MG formula on BF16 inputs throughout
                  q * torch.rsqrt(q.square().mean(-1, keepdim=True) + eps)
    3. sg_path:   SGLang's rmsnorm_self CUDA kernel (FP32 internals, BF16 store)

Conclusion criteria:
  Compare |out - ref_fp32| for mg and sg.
  - If sg_path ≈ ref_fp32 (within ~ULP) AND mg_path differs by ~observed cross_diff,
    MG's BF16 RMSNorm is the divergence injection point. Fix = promote to FP32 in MG.
  - If both diverge similarly, the issue is elsewhere (eps mismatch, input pre-scale, …)
"""
import argparse
from pathlib import Path

import torch


def mg_path(q_bf16: torch.Tensor, eps: float) -> torch.Tensor:
    """Megatron formula: pure BF16."""
    return q_bf16 * torch.rsqrt(q_bf16.square().mean(-1, keepdim=True) + eps)


def sg_triton_path(q_bf16: torch.Tensor, eps: float) -> torch.Tensor:
    """SGLang's Triton `rms_normalize_triton` (FP32 internals, in-place).
    This is the path V4 uses for head_dim=128 (the JIT CUDA `rmsnorm_self`
    requires head_dim ≥ kVecSize*kWarpThreads = 8*32 = 256 for BF16, so V4 falls
    through to the Triton kernel)."""
    from sglang.srt.models.deepseek_v4 import rms_normalize_triton
    return rms_normalize_triton(q_bf16.clone(), eps)


def ref_fp32(q_bf16: torch.Tensor, eps: float) -> torch.Tensor:
    """FP32 reference: full precision inside, BF16 cast at the very end."""
    q_fp32 = q_bf16.float()
    out = q_fp32 * torch.rsqrt(q_fp32.square().mean(-1, keepdim=True) + eps)
    return out.to(torch.bfloat16)


def stat(diff: torch.Tensor, label: str) -> None:
    a = diff.float().abs().flatten()
    # quantile() fails for >16M elements; use kthvalue instead
    n = a.numel()
    p99_k = max(1, int(0.99 * n))
    p99 = a.kthvalue(p99_k).values.item() if n > 0 else 0.0
    p50 = a.kthvalue(max(1, n // 2)).values.item() if n > 0 else 0.0
    print(
        f"  {label:30s} "
        f"max={a.max().item():.4e}  "
        f"mean={a.mean().item():.4e}  "
        f"p50={p50:.4e}  "
        f"p99={p99:.4e}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wq-b-pt", required=True, help="Real wq_b_out dump file (.pt with key 'value')")
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--eps", type=float, default=1e-6)
    args = ap.parse_args()

    raw = torch.load(args.wq_b_pt, weights_only=False)["value"].detach()
    print(f"Input file:  {args.wq_b_pt}")
    print(f"Raw shape:   {tuple(raw.shape)}  dtype: {raw.dtype}")

    last_dim = raw.shape[-1]
    assert last_dim % args.head_dim == 0, f"last dim {last_dim} % head_dim {args.head_dim} != 0"
    n_heads = last_dim // args.head_dim

    # Squeeze any leading batch=1 if present, then flatten T dims
    if raw.dim() > 2:
        raw = raw.reshape(-1, last_dim)
    q = raw.reshape(-1, n_heads, args.head_dim).contiguous().cuda().to(torch.bfloat16)
    print(f"Reshape:     {tuple(q.shape)}  (T={q.shape[0]}, n_heads={n_heads}, head_dim={args.head_dim})")
    q_abs = q.float().abs()
    print(f"Q stats:     max={q_abs.max().item():.4f}  mean_abs={q_abs.mean().item():.4f}")
    print(f"Eps:         {args.eps}")
    print()

    print("Computing three implementations...")
    out_ref = ref_fp32(q.clone(), args.eps)
    out_mg = mg_path(q.clone(), args.eps)
    out_sg = sg_triton_path(q.clone(), args.eps)
    print(f"  out_ref:  shape={tuple(out_ref.shape)} dtype={out_ref.dtype}")
    print(f"  out_mg:   shape={tuple(out_mg.shape)} dtype={out_mg.dtype}")
    print(f"  out_sg:   shape={tuple(out_sg.shape)} dtype={out_sg.dtype}")
    print()

    print("=== |X - ref_fp32| (BF16 cast as final ground truth) ===")
    stat(out_mg - out_ref, "mg_path - ref_fp32")
    stat(out_sg - out_ref, "sg_triton - ref_fp32")
    print()

    print("=== mg vs sg directly ===")
    stat(out_mg - out_sg, "mg_path - sg_triton")
    print()

    # Per-token max-feature diff (matches our viz_v2 metric)
    def per_token_max(d):
        return d.float().abs().reshape(d.shape[0], -1).max(dim=1).values

    pt_mg = per_token_max(out_mg - out_ref)
    pt_sg = per_token_max(out_sg - out_ref)
    pt_mgsg = per_token_max(out_mg - out_sg)
    print("=== Per-token max-feature |diff| (matches viz_v2 metric) ===")
    print(f"  mg vs ref: max={pt_mg.max().item():.4e}  mean={pt_mg.mean().item():.4e}  worst-3 tok: {pt_mg.argsort(descending=True)[:3].tolist()}")
    print(f"  sg vs ref: max={pt_sg.max().item():.4e}  mean={pt_sg.mean().item():.4e}  worst-3 tok: {pt_sg.argsort(descending=True)[:3].tolist()}")
    print(f"  mg vs sg:  max={pt_mgsg.max().item():.4e}  mean={pt_mgsg.mean().item():.4e}  worst-3 tok: {pt_mgsg.argsort(descending=True)[:3].tolist()}")


if __name__ == "__main__":
    main()
