"""Export a loopify/nanotron checkpoint to Hugging Face format.

With --unroll the loop is written out as a plain deeper model (e.g. 54 layers for
Qwen3-4B looped [13, 22) x3) that transformers and vLLM load unmodified. Without it
the weights keep the base model's layout and the loop has to be re-applied around
them; loop_config.json records the span and count either way.

    torchrun --nproc_per_node=1 -m loopify.export_hf --unroll \\
        --ckpt runs/<run>/checkpoints/<step> \\
        --hf-ref <original HF model> --out hf/<run>
"""

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict

import nanotron
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from loopify.config import LoopifyConfig
from loopify.convert_hf import build_nanotron_model


def nanotron_to_hf_weights(nt_sd: Dict[str, torch.Tensor], config: LoopifyConfig) -> Dict[str, torch.Tensor]:
    head_dim = config.head_dim or config.hidden_size // config.num_attention_heads
    q_size = config.num_attention_heads * head_dim
    kv_size = config.num_key_value_heads * head_dim

    out = {
        "model.embed_tokens.weight": nt_sd["model.token_position_embeddings.pp_block.token_embedding.weight"],
        "model.norm.weight": nt_sd["model.final_layer_norm.pp_block.weight"],
    }
    if not config.tie_word_embeddings:
        out["lm_head.weight"] = nt_sd["model.lm_head.pp_block.weight"]

    for i in range(config.num_hidden_layers):
        hf = f"model.layers.{i}"
        nt = f"model.decoder.{i}.pp_block"
        q, k, v = nt_sd[f"{nt}.attn.qkv_proj.weight"].split([q_size, kv_size, kv_size], dim=0)
        out[f"{hf}.self_attn.q_proj.weight"] = q
        out[f"{hf}.self_attn.k_proj.weight"] = k
        out[f"{hf}.self_attn.v_proj.weight"] = v
        if config.attention_bias:
            qb, kb, vb = nt_sd[f"{nt}.attn.qkv_proj.bias"].split([q_size, kv_size, kv_size], dim=0)
            out[f"{hf}.self_attn.q_proj.bias"] = qb
            out[f"{hf}.self_attn.k_proj.bias"] = kb
            out[f"{hf}.self_attn.v_proj.bias"] = vb
        out[f"{hf}.self_attn.o_proj.weight"] = nt_sd[f"{nt}.attn.o_proj.weight"]
        if config.qk_norm is not None:
            out[f"{hf}.self_attn.q_norm.weight"] = nt_sd[f"{nt}.attn.q_norm.weight"]
            out[f"{hf}.self_attn.k_norm.weight"] = nt_sd[f"{nt}.attn.k_norm.weight"]

        gate, up = nt_sd[f"{nt}.mlp.gate_up_proj.weight"].chunk(2, dim=0)
        out[f"{hf}.mlp.gate_proj.weight"] = gate
        out[f"{hf}.mlp.up_proj.weight"] = up
        out[f"{hf}.mlp.down_proj.weight"] = nt_sd[f"{nt}.mlp.down_proj.weight"]

        if config.norm_placement == "pre":
            out[f"{hf}.input_layernorm.weight"] = nt_sd[f"{nt}.input_layernorm.weight"]
            out[f"{hf}.post_attention_layernorm.weight"] = nt_sd[f"{nt}.post_attention_layernorm.weight"]
        else:
            out[f"{hf}.post_attention_layernorm.weight"] = nt_sd[f"{nt}.post_attention_layernorm.weight"]
            out[f"{hf}.post_feedforward_layernorm.weight"] = nt_sd[f"{nt}.post_feedforward_layernorm.weight"]

    return out


def unroll(hf_sd: Dict[str, torch.Tensor], config: LoopifyConfig) -> Dict[str, torch.Tensor]:
    """Rewrite a looped model as a plain stack of its executed blocks.

    Running decoder[7:11] three times is the same computation as a deeper model
    whose extra layers are copies of layers 7-10: a Qwen3/OLMo block depends on its
    weights and its input, not on its index. So the unrolled checkpoint is an
    ordinary architecture that vLLM, HF, llama.cpp and every other runtime already
    load, with no custom modeling code — at the price of storing the repeated
    layers once per pass.
    """
    out = {k: v for k, v in hf_sd.items() if not k.startswith("model.layers.")}
    for target, source in enumerate(config.layer_order):
        prefix = f"model.layers.{source}."
        for k, v in hf_sd.items():
            if k.startswith(prefix):
                out[f"model.layers.{target}." + k[len(prefix):]] = v.clone()
    return out


def main():
    parser = ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True, help="nanotron checkpoint dir (contains model/)")
    parser.add_argument("--hf-ref", type=str, required=True, help="original HF model, for config + tokenizer")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--unroll",
        action="store_true",
        help="write the executed blocks as a plain deeper model (loads in vLLM etc. unmodified)",
    )
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    config_path = args.ckpt / "model_config.json"
    if not config_path.exists():  # a training checkpoint keeps it next to the weights
        config_path = args.ckpt.parent / "model_config.json"
    config = LoopifyConfig(**json.load(open(config_path)))

    model, parallel_context = build_nanotron_model(config, dtype=dtype)
    nanotron.serialize.load_weights(model=model, parallel_context=parallel_context, root_folder=args.ckpt)
    nt_sd = {name: param.detach() for name, param in model.named_parameters()}

    hf_config = AutoConfig.from_pretrained(args.hf_ref)
    weights = nanotron_to_hf_weights(nt_sd, config)
    if args.unroll:
        weights = unroll(weights, config)
        depth = config.effective_num_layers
        hf_config.num_hidden_layers = depth
        # Per-layer settings have to cover the new depth too.
        if getattr(hf_config, "max_window_layers", None) is not None:
            hf_config.max_window_layers = depth
        if getattr(hf_config, "layer_types", None):
            hf_config.layer_types = [hf_config.layer_types[s] for s in config.layer_order]
    hf_model = AutoModelForCausalLM.from_config(hf_config, dtype=dtype)
    missing, unexpected = hf_model.load_state_dict(weights, strict=False)
    # Tied lm_head is the one weight HF fills in itself.
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert all("lm_head" in m for m in missing), f"missing keys: {missing}"
    hf_model.tie_weights()

    # A base model's generation config stops only at end-of-text; a chat-trained one must also stop
    # at the end of its turn. Sample the way the models are evaluated.
    tok = AutoTokenizer.from_pretrained(args.hf_ref)
    eos = hf_config.eos_token_id
    eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end != tok.unk_token_id and im_end not in eos:
        eos = [im_end] + eos
    hf_model.generation_config = GenerationConfig(
        bos_token_id=hf_config.bos_token_id, eos_token_id=eos, do_sample=True, temperature=0.6, top_p=0.95
    )

    (args.out / "full_model").mkdir(parents=True, exist_ok=True)
    hf_model.save_pretrained(args.out / "full_model")
    tok.save_pretrained(args.out / "full_model")

    assert config.loop_inject_scale == 0.0, (
        "input injection is not implemented in the evaluation-side loop wrapper yet; "
        "exporting it would silently evaluate a different model than was trained"
    )
    loop_config = {
        "base_model": args.hf_ref,
        "loop_start": config.loop_start,
        "loop_end": config.loop_end,
        "loop_k": config.loop_k,
        "unrolled": args.unroll,
        "num_hidden_layers": hf_config.num_hidden_layers,
    }
    json.dump(loop_config, open(args.out / "loop_config.json", "w"), indent=2)
    print(f"exported to {args.out}: {loop_config}")


if __name__ == "__main__":
    main()
