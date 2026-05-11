"""Registry of V4 RL divergence debug experiments.

Each entry is one ExperimentSpec declaring only its delta from CANONICAL. To
add a new experiment, append an entry; nothing else changes.

Run names match the legacy tools_debug/launch_{sg,mg}_grafter_*.sh script tags
so existing dump dirs (``v4-iter59-grafter-<tag>-{sg,mg}``) remain referencable
by the same identifier.
"""

from tools_debug.experiments.canonical import CANONICAL_DUMPER_FILTER

from miles.utils.debug_utils.experiment_runner import ExperimentSpec, RunKind


def _grafter(
    *,
    name: str,
    description: str,
    b2t: str,
    t2b: str,
    baselines: tuple[str, ...] = ("sg-natural-prefill-e8env",),
    mg_extra_patches_yaml: str | None = None,
    sg_env: dict[str, str] | None = None,
) -> ExperimentSpec:
    return ExperimentSpec(
        name=name,
        description=description,
        kind=RunKind.GRAFTER_PAIR,
        grafter_b2t_filter=b2t,
        grafter_t2b_filter=t2b,
        dumper_filter=CANONICAL_DUMPER_FILTER,
        baselines=baselines,
        mg_extra_patches_yaml=mg_extra_patches_yaml,
        sg_env=sg_env or {},
    )


_PREFILL = ' and graft_phase == "prefill"'


# Megatron-side source-patcher YAML fragments. Each spec composes whichever
# patches it needs into ``mg_extra_patches_yaml``. The targets injected here are
# names mg's V4 plugin does NOT dump natively, so the grafter would otherwise see
# a collective-op-sequence mismatch (mg skips a name sg fires, gathered slots
# misalign, downstream slice access hits None).
_PATCH_INPUT_LAYERNORM = """\
  - target: megatron.core.transformer.transformer_layer.TransformerLayer._forward_attention
    edits:
      - match: |
          input_layernorm_output = self.input_layernorm(hidden_states)
        append: "dumper.dump('input_layernorm', input_layernorm_output, dims='t[cp:zigzag,sp] 1 h # tp:replicated ep:replicated')"
"""

_PATCH_PRE_MLP_LAYERNORM = """\
  - target: megatron.core.transformer.transformer_layer.TransformerLayer._forward_mlp
    edits:
      - match: "pre_mlp_layernorm_output = self._forward_pre_mlp_layernorm(hidden_states)"
        append: "dumper.dump('pre_mlp_layernorm_output', pre_mlp_layernorm_output, dims='t[cp:zigzag,sp] 1 h # tp:replicated ep:replicated')"
"""

_PATCH_MOE_ROUTING = """\
  - target: megatron.core.transformer.moe.router.TopKRouter.forward
    edits:
      - match: "probs, routing_map = self.routing(logits, padding_mask=padding_mask, input_ids=input_ids)"
        append: "dumper.dump('moe_routing_map', routing_map, dims='t e # tp:replicated ep:replicated'); dumper.dump('moe_probs', probs, dims='t e # tp:replicated ep:replicated')"
"""


def _patches(*fragments: str) -> str:
    """Assemble a source-patcher YAML from one or more patch fragments."""
    return "patches:\n" + "".join(fragments)

EXPERIMENTS: dict[str, ExperimentSpec] = {
    # ------------------------ baselines ------------------------
    "sg-natural-prefill-e8env": ExperimentSpec(
        name="sg-natural-prefill-e8env",
        description="Canonical sg HTTP run -- single source of truth baseline.",
        kind=RunKind.SG_ONLY,
    ),
    "mg-fix0505": ExperimentSpec(
        name="mg-fix0505",
        description="Canonical mg standalone forward (F1+F2 fix on).",
        kind=RunKind.MG_ONLY,
        baselines=("sg-natural-prefill-e8env",),
        dumper_filter=CANONICAL_DUMPER_FILTER,
    ),
    # ------------------------ sg ablations ------------------------
    "sg-prefill-chunk8192": ExperimentSpec(
        name="sg-prefill-chunk8192",
        description="sg natural prefill with --chunked-prefill-size 8192 (vs canonical 32768).",
        kind=RunKind.SG_ONLY,
        sg_server_args_extra=("--chunked-prefill-size", "8192"),
        baselines=("sg-natural-prefill-e8env",),
    ),
    "sg-prefill-no-f4": ExperimentSpec(
        name="sg-prefill-no-f4",
        description="sg natural prefill with F4 fix (SGLANG_DSV4_FIX_0506) disabled.",
        kind=RunKind.SG_ONLY,
        sg_env={"SGLANG_DSV4_FIX_0506": "0"},
        baselines=("sg-natural-prefill-e8env",),
    ),
    # ------------------------ mg ablations ------------------------
    "mg-replay-moe": ExperimentSpec(
        name="mg-replay-moe",
        description="mg with MoE routing replay loaded from rollout data (use_routing_replay).",
        kind=RunKind.MG_ONLY,
        mg_run_args_extra=("--routing-replay-load-path", "/storage/yueming/replay-data/iter59_routing.pt"),
        baselines=("sg-natural-prefill-e8env", "mg-fix0505"),
        dumper_filter=CANONICAL_DUMPER_FILTER,
    ),
    "mg-replay-both": ExperimentSpec(
        name="mg-replay-both",
        description="mg with both MoE routing and indexer topk replays loaded from rollout.",
        kind=RunKind.MG_ONLY,
        mg_run_args_extra=(
            "--routing-replay-load-path",
            "/storage/yueming/replay-data/iter59_routing.pt",
            "--indexer-replay-load-path",
            "/storage/yueming/replay-data/iter59_indexer.pt",
        ),
        baselines=("sg-natural-prefill-e8env", "mg-fix0505"),
        dumper_filter=CANONICAL_DUMPER_FILTER,
    ),
    # ------------------------ A series: attn LoRA / output proj ------------------------
    "a1": _grafter(
        name="a1",
        description="Replace sg.Q-LoRA path only. b2t=input_layernorm, t2b=attn_q.",
        b2t=f'name == "input_layernorm"{_PREFILL}',
        t2b=f'name == "attn_q"{_PREFILL}',
    ),
    "a2": _grafter(
        name="a2",
        description="Replace sg.KV-LoRA path only. b2t=input_layernorm, t2b=attn_v.",
        b2t=f'name == "input_layernorm"{_PREFILL}',
        t2b=f'name == "attn_v"{_PREFILL}',
    ),
    "a3": _grafter(
        name="a3",
        description="Replace sg.Q-LoRA + KV-LoRA paths. t2b=attn_q,attn_v.",
        b2t=f'name == "input_layernorm"{_PREFILL}',
        t2b=f'name in ("attn_q", "attn_v"){_PREFILL}',
    ),
    "a4": _grafter(
        name="a4",
        description="Replace sg.attn block from input_layernorm. t2b=attn_output.",
        b2t=f'name == "input_layernorm"{_PREFILL}',
        t2b=f'name == "attn_output"{_PREFILL}',
    ),
    "a5b": _grafter(
        name="a5b",
        description="Replace sg.O-projection only (no attn compute). b2t=attn_output, t2b=mqa_wo_b_out.",
        b2t=f'name == "attn_output"{_PREFILL}',
        t2b=f'name == "mqa_wo_b_out"{_PREFILL}',
    ),
    # ------------------------ M series: MoE router ------------------------
    "m1": _grafter(
        name="m1",
        description="Replace sg.MoE router only. t2b=moe_routing_map+moe_probs.",
        b2t=f'name == "pre_mlp_layernorm_output"{_PREFILL}',
        t2b=f'name in ("moe_routing_map", "moe_probs"){_PREFILL}',
    ),
    # ------------------------ Combo ------------------------
    "combo-lora-proj-router": _grafter(
        name="combo-lora-proj-router",
        description="A1 + A2 + A5' + M1 (LoRA paths + output proj + MoE router).",
        b2t=f'name in ("input_layernorm", "attn_output", "pre_mlp_layernorm_output"){_PREFILL}',
        t2b=f'name in ("attn_q", "attn_v", "mqa_wo_b_out", "moe_routing_map", "moe_probs"){_PREFILL}',
        # mg dumps attn_output / mqa_wo_b_out / attn_q / attn_v natively in V4 plugin;
        # input_layernorm / pre_mlp_layernorm_output / moe_routing_map / moe_probs need patches.
        mg_extra_patches_yaml=_patches(_PATCH_INPUT_LAYERNORM, _PATCH_PRE_MLP_LAYERNORM, _PATCH_MOE_ROUTING),
        # sg's HashTopK only dumps moe_routing_map / moe_probs when SGLANG_DSV4_DUMP_ROUTING=1.
        sg_env={"SGLANG_DSV4_DUMP_ROUTING": "1"},
    ),
    # ------------------------ E series: per-block isolation ------------------------
    "e2-compressor": _grafter(
        name="e2-compressor",
        description="Replace sg.compressor (cr=128 only). b2t=compress_input, t2b=compress_final_out (head_dim==512).",
        b2t=f'name == "compress_input"{_PREFILL} and head_dim == 512',
        t2b=f'name == "compress_final_out"{_PREFILL} and head_dim == 512',
    ),
    "e3-attn-block": _grafter(
        name="e3-attn-block",
        description="Replace sg.attn block from attn_q/attn_v. b2t=attn_q+attn_v, t2b=attn_output.",
        b2t=f'name in ("attn_q", "attn_v"){_PREFILL}',
        t2b=f'name == "attn_output"{_PREFILL}',
    ),
    "e4-mlp-block": _grafter(
        name="e4-mlp-block",
        description="Replace sg.MLP/MoE block. b2t=pre_mlp_layernorm_output, t2b=mlp_output.",
        b2t=f'name == "pre_mlp_layernorm_output"{_PREFILL}',
        t2b=f'name == "mlp_output"{_PREFILL}',
    ),
    "e5-sparse-mla": _grafter(
        name="e5-sparse-mla",
        description="Sparse-MLA-kernel approximation. b2t=attn_q+attn_v+compress_final_out(head_dim==512), t2b=attn_output. (invalid: compress_final_out boundary semantics differ)",
        b2t=(
            f'(name in ("attn_q", "attn_v"){_PREFILL}) or '
            f'(name == "compress_final_out"{_PREFILL} and head_dim == 512)'
        ),
        t2b=f'name == "attn_output"{_PREFILL}',
    ),
    "e6-full-layer-baseline": _grafter(
        name="e6-full-layer-baseline",
        description="Full per-layer override: t2b=layer_input only. Probes whole-layer divergence baseline.",
        b2t="False",
        t2b=f'name == "layer_input"{_PREFILL}',
    ),
    "e7-full-layer-plus-mhc": _grafter(
        name="e7-full-layer-plus-mhc",
        description="E6 + post_norm_hidden override (post-final-RMSNorm).",
        b2t="False",
        t2b=f'name in ("layer_input", "post_norm_hidden"){_PREFILL}',
    ),
    "e8-lm-head": _grafter(
        name="e8-lm-head",
        description="E7 + lm_head_logits override. Confirms mg.collected logprobs == log_softmax(sg.lm_head_logits).",
        b2t="False",
        t2b=f'name in ("layer_input", "post_norm_hidden", "lm_head_logits"){_PREFILL}',
    ),
    # ------------------------ H series: hyper-connection (invalid) ------------------------
    "h1": _grafter(
        name="h1",
        description="MHC kernel swap via hc_attn_*/hc_ffn_* tensors. (invalid: hc_*_post overrides residual stream, equivalent to E6.)",
        b2t=f'name in ("layer_input", "attn_output", "mlp_output"){_PREFILL}',
        t2b=f'name in ("hc_attn_pre", "hc_attn_post", "hc_ffn_pre", "hc_ffn_post"){_PREFILL}',
    ),
    "h1prime": _grafter(
        name="h1prime",
        description="H1 with mqa_wo_b_out instead of attn_output as b2t input. (still invalid for the same reason as H1.)",
        b2t=f'name in ("layer_input", "mqa_wo_b_out", "mlp_output"){_PREFILL}',
        t2b=f'name in ("hc_attn_pre", "hc_attn_post", "hc_ffn_pre", "hc_ffn_post"){_PREFILL}',
    ),
}
