"""Inspect (sg, mg) shapes for V4 operator dumps. Both sides at layer 0, step 0, rank 0."""
import sys, glob, torch
SG_DIR = sys.argv[1]
MG_DIR = sys.argv[2]
NAMES = [
    "wq_a_out", "q_lora_after_norm", "wq_b_out", "q_heads_after_norm", "attn_q",
    "wkv_out", "kv_after_norm", "attn_v",
    "mqa_wo_a_out", "mqa_wo_b_out",
    "pre_mlp_residual", "pre_mlp_layernorm_output", "mlp_output",
    "attn_output",
    "compress_forward_out", "compress_fused_norm_rope_out", "compress_final_out", "compressor_out",
    "attn_q_in", "attn_kv_in", "attn_topk_idxs",
    "moe_router_logits", "moe_topk_ids",
]

def first(globber):
    files = sorted(glob.glob(globber))
    if not files: return None
    return torch.load(files[0], weights_only=False)

print(f"{'name':<35} {'sg shape':<25} {'mg layer_id=0':<25} {'mg has cr':<8}")
for name in NAMES:
    sg = first(f"{SG_DIR}/step=0___rank=0___dump_index=*___name={name}___*.pt")
    mg_l0 = first(f"{MG_DIR}/step=0___rank=0___dump_index=*___name={name}___*layer_id=0.pt")
    mg_l0_cr4 = first(f"{MG_DIR}/step=0___rank=0___dump_index=*___name={name}___*compress_ratio=4*.pt")
    sg_s = tuple(sg["value"].shape) if sg is not None else "-"
    mg_s = tuple(mg_l0["value"].shape) if mg_l0 is not None else "-"
    mg_cr_s = tuple(mg_l0_cr4["value"].shape) if mg_l0_cr4 is not None else "-"
    print(f"{name:<35} {str(sg_s):<25} {str(mg_s):<25} {str(mg_cr_s):<8}")
