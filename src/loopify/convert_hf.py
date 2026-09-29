"""Convert a HuggingFace checkpoint into a loopify/nanotron checkpoint.

Supported architectures: ``olmo2`` (reordered norm + whole-vector QK-norm),
``qwen3`` (per-head QK-norm), ``qwen2`` and ``llama`` (neither).

The looped and dense models share a checkpoint layout — looping only changes how
many times a block is executed — so one conversion serves both arms of an
experiment:

    torchrun --nproc_per_node=1 -m loopify.convert_hf \\
        --hf allenai/OLMo-2-0425-1B --out ckpts/olmo2-1b-nt
"""

import dataclasses
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict

import nanotron
import torch
from nanotron.trainer import mark_tied_parameters
from transformers import AutoConfig, AutoModelForCausalLM

from loopify.config import LoopifyConfig
from loopify.model import LoopifyForTraining

# model_type -> (qk_norm, norm_placement)
ARCH_DEFAULTS = {
    "olmo2": ("whole", "post"),
    "qwen3": ("per_head", "pre"),
    "qwen2": (None, "pre"),
    "llama": (None, "pre"),
}


def hf_config_to_kwargs(hf_config) -> dict:
    """The LoopifyConfig fields that describe a Hugging Face model's architecture,
    as a plain dict -- what a training config's ``model_config`` holds."""
    model_type = hf_config.model_type
    assert model_type in ARCH_DEFAULTS, f"unsupported architecture {model_type}"
    qk_norm, norm_placement = ARCH_DEFAULTS[model_type]
    # model.py implements plain RoPE only; a scaled variant (e.g. Llama 3.1's) would convert silently wrong.
    rope_scaling = getattr(hf_config, "rope_scaling", None) or {}
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
    assert rope_type == "default", f"scaled RoPE ({rope_type}) is not supported"

    # Qwen3-4B's heads (128) are wider than hidden_size / num_heads (80); carry the
    # real width through rather than letting nanotron's default apply.
    head_dim = getattr(hf_config, "head_dim", None)
    if head_dim == hf_config.hidden_size // hf_config.num_attention_heads:
        head_dim = None

    kwargs = dict(
        bos_token_id=getattr(hf_config, "bos_token_id", None) or 0,
        eos_token_id=hf_config.eos_token_id,
        hidden_act=hf_config.hidden_act,
        hidden_size=hf_config.hidden_size,
        initializer_range=hf_config.initializer_range,
        intermediate_size=hf_config.intermediate_size,
        max_position_embeddings=hf_config.max_position_embeddings,
        num_attention_heads=hf_config.num_attention_heads,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_key_value_heads=hf_config.num_key_value_heads,
        pad_token_id=getattr(hf_config, "pad_token_id", None),
        rms_norm_eps=hf_config.rms_norm_eps,
        rope_theta=float(hf_config.rope_theta),
        rope_interleaved=False,  # matches HF's rotate_half convention
        tie_word_embeddings=hf_config.tie_word_embeddings,
        vocab_size=hf_config.vocab_size,
        # HF Qwen2 always has Q/K/V biases; its config need not expose this flag.
        attention_bias=True if model_type == "qwen2" else getattr(hf_config, "attention_bias", False),
        qk_norm=qk_norm,
        norm_placement=norm_placement,
        head_dim=head_dim,
    )
    return kwargs


def hf_config_to_loopify(hf_config, **overrides) -> LoopifyConfig:
    kwargs = hf_config_to_kwargs(hf_config)
    kwargs.update(overrides)
    return LoopifyConfig(**kwargs)


def hf_to_nanotron_weights(hf_sd: Dict[str, torch.Tensor], config: LoopifyConfig) -> Dict[str, torch.Tensor]:
    """Build the nanotron parameter dict from a HF state dict.

    nanotron packs q/k/v into one ``qkv_proj`` and gate/up into one ``gate_up_proj``;
    everything else is a rename.
    """
    embed = hf_sd["model.embed_tokens.weight"]
    out = {
        "model.token_position_embeddings.pp_block.token_embedding.weight": embed,
        "model.final_layer_norm.pp_block.weight": hf_sd["model.norm.weight"],
        # HF omits lm_head.weight when the embeddings are tied.
        "model.lm_head.pp_block.weight": hf_sd.get("lm_head.weight", embed),
    }

    for i in range(config.num_hidden_layers):
        hf = f"model.layers.{i}"
        nt = f"model.decoder.{i}.pp_block"
        out[f"{nt}.attn.qkv_proj.weight"] = torch.cat(
            [
                hf_sd[f"{hf}.self_attn.q_proj.weight"],
                hf_sd[f"{hf}.self_attn.k_proj.weight"],
                hf_sd[f"{hf}.self_attn.v_proj.weight"],
            ]
        )
        if config.attention_bias:
            out[f"{nt}.attn.qkv_proj.bias"] = torch.cat(
                [
                    hf_sd[f"{hf}.self_attn.q_proj.bias"],
                    hf_sd[f"{hf}.self_attn.k_proj.bias"],
                    hf_sd[f"{hf}.self_attn.v_proj.bias"],
                ]
            )
        out[f"{nt}.attn.o_proj.weight"] = hf_sd[f"{hf}.self_attn.o_proj.weight"]
        if config.qk_norm is not None:
            out[f"{nt}.attn.q_norm.weight"] = hf_sd[f"{hf}.self_attn.q_norm.weight"]
            out[f"{nt}.attn.k_norm.weight"] = hf_sd[f"{hf}.self_attn.k_norm.weight"]

        out[f"{nt}.mlp.gate_up_proj.weight"] = torch.cat(
            [hf_sd[f"{hf}.mlp.gate_proj.weight"], hf_sd[f"{hf}.mlp.up_proj.weight"]]
        )
        out[f"{nt}.mlp.down_proj.weight"] = hf_sd[f"{hf}.mlp.down_proj.weight"]

        if config.norm_placement == "pre":
            out[f"{nt}.input_layernorm.weight"] = hf_sd[f"{hf}.input_layernorm.weight"]
            out[f"{nt}.post_attention_layernorm.weight"] = hf_sd[f"{hf}.post_attention_layernorm.weight"]
        else:  # OLMo-2 names its two norms after the branches they follow
            out[f"{nt}.post_attention_layernorm.weight"] = hf_sd[f"{hf}.post_attention_layernorm.weight"]
            out[f"{nt}.post_feedforward_layernorm.weight"] = hf_sd[f"{hf}.post_feedforward_layernorm.weight"]

    return out


_PARALLEL_CONTEXT = None


def _single_process_context():
    """One process group per process — building a second one would fail."""
    global _PARALLEL_CONTEXT
    if _PARALLEL_CONTEXT is None:
        _PARALLEL_CONTEXT = nanotron.parallel.ParallelContext(
            data_parallel_size=1, pipeline_parallel_size=1, tensor_parallel_size=1
        )
    return _PARALLEL_CONTEXT


def build_nanotron_model(config: LoopifyConfig, dtype=torch.bfloat16, device=torch.device("cuda")):
    from nanotron.config import (
        OneForwardOneBackwardPipelineEngine,
        ParallelismArgs,
        TensorParallelLinearMode,
    )

    parallel_config = ParallelismArgs(
        dp=1,
        pp=1,
        tp=1,
        pp_engine=OneForwardOneBackwardPipelineEngine(),
        tp_mode=TensorParallelLinearMode.ALL_REDUCE,
        tp_linear_async_communication=False,
    )
    parallel_context = _single_process_context()
    model = nanotron.models.build_model(
        model_builder=lambda: LoopifyForTraining(
            config=config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
            random_states=None,
        ),
        parallel_context=parallel_context,
        dtype=dtype,
        device=device,
    )
    mark_tied_parameters(model=model, parallel_context=parallel_context)
    return model, parallel_context


def convert(hf_path: str, out_path: Path, dtype=torch.bfloat16, **config_overrides):
    hf_config = AutoConfig.from_pretrained(hf_path)
    config = hf_config_to_loopify(hf_config, **config_overrides)
    print(f"loopify config: {config}")

    hf_model = AutoModelForCausalLM.from_pretrained(hf_path, dtype=torch.float32)
    weights = hf_to_nanotron_weights(hf_model.state_dict(), config)
    del hf_model

    model, parallel_context = build_nanotron_model(config, dtype=dtype)

    copied, missing = set(), []
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in weights:
                missing.append(name)
                continue
            param.copy_(weights[name].to(param.dtype))
            copied.add(name)
    assert not missing, f"no HF source for: {missing}"
    unused = sorted(set(weights) - copied)
    # With tied embeddings nanotron stores one tensor, so lm_head is expected here.
    print(f"copied {len(copied)} tensors; unused mapping entries: {unused}")

    out_path.mkdir(parents=True, exist_ok=True)
    nanotron.serialize.save_weights(model=model, parallel_context=parallel_context, root_folder=out_path)
    with open(out_path / "model_config.json", "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=2)
    print(f"saved nanotron checkpoint to {out_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert a HF checkpoint to loopify/nanotron format")
    parser.add_argument("--hf", type=str, required=True, help="HF model id or local path")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--loop-start", type=int, default=0)
    parser.add_argument("--loop-end", type=int, default=0)
    parser.add_argument("--loop-k", type=int, default=1)
    args = parser.parse_args()

    convert(
        args.hf,
        args.out,
        dtype=getattr(torch, args.dtype),
        loop_start=args.loop_start,
        loop_end=args.loop_end,
        loop_k=args.loop_k,
    )
