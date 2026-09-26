"""Check that an unrolled export computes exactly what the looped model computes.

The release artefact is the unrolled checkpoint — a plain, deeper model that any
runtime loads without custom code. That is only honest if running it gives the
same outputs as running the looped model it came from, so compare the two
directly on the same weights and the same tokens.

    torchrun --nproc_per_node=1 tools/check_unroll.py --hf <Qwen3-4B-Base path> \\
        --loop-start 13 --loop-end 22 --loop-k 3
"""

from argparse import ArgumentParser

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from loopify.convert_hf import build_nanotron_model, hf_config_to_loopify, hf_to_nanotron_weights
from loopify.export_hf import nanotron_to_hf_weights, unroll


def main():
    parser = ArgumentParser()
    parser.add_argument("--hf", required=True)
    parser.add_argument("--loop-start", type=int, required=True)
    parser.add_argument("--loop-end", type=int, required=True)
    parser.add_argument("--loop-k", type=int, required=True)
    parser.add_argument("--seq-len", type=int, default=256)
    args = parser.parse_args()
    device, dtype = torch.device("cuda"), torch.bfloat16

    hf_config = AutoConfig.from_pretrained(args.hf)
    base = AutoModelForCausalLM.from_pretrained(args.hf, dtype=dtype)
    config = hf_config_to_loopify(
        hf_config, loop_start=args.loop_start, loop_end=args.loop_end, loop_k=args.loop_k
    )
    weights = hf_to_nanotron_weights(base.state_dict(), config)
    del base

    # The looped model, as trained.
    looped, _ = build_nanotron_model(config, dtype=dtype, device=device)
    with torch.no_grad():
        for name, param in looped.named_parameters():
            param.copy_(weights[name].to(param.dtype))
    looped.eval()

    torch.manual_seed(0)
    ids = torch.randint(0, config.vocab_size, (1, args.seq_len), device=device)
    positions = torch.arange(args.seq_len, device=device).unsqueeze(0)
    with torch.no_grad():
        ref = looped.model(input_ids=ids, position_ids=positions).reshape(1, args.seq_len, -1).float()
    nt_sd = {name: p.detach() for name, p in looped.named_parameters()}
    del looped
    torch.cuda.empty_cache()

    # The same weights, unrolled into a plain model of the executed depth.
    hf_config.num_hidden_layers = config.effective_num_layers
    if getattr(hf_config, "max_window_layers", None) is not None:
        hf_config.max_window_layers = config.effective_num_layers
    if getattr(hf_config, "layer_types", None):
        hf_config.layer_types = [hf_config.layer_types[s] for s in config.layer_order]
    unrolled = unroll(nanotron_to_hf_weights(nt_sd, config), config)

    def run_flat(precision):
        model = AutoModelForCausalLM.from_config(hf_config, dtype=precision).to(device).eval()
        missing, unexpected = model.load_state_dict(unrolled, strict=False)
        assert not unexpected and all("lm_head" in m for m in missing), (missing, unexpected)
        model.tie_weights()
        with torch.no_grad():
            out = model(ids).logits.float()
        del model
        torch.cuda.empty_cache()
        return out

    # fp32 unrolled is what the unrolled model *is*; bf16 unrolled shows how far
    # precision alone moves it. The looped model must be no further than that.
    truth = run_flat(torch.float32)
    flat_bf16 = run_flat(dtype)

    def report(name, got):
        rel = (got - truth).norm().item() / truth.norm().item()
        agree = (got.argmax(-1) == truth.argmax(-1)).float().mean().item()
        print(f"[{name} vs unrolled fp32] rel_l2={rel:.4e} argmax_agreement={agree:.4%}")
        return rel, agree

    floor, floor_agree = report(f"unrolled {config.effective_num_layers}-layer HF bf16", flat_bf16)
    rel, agree = report("looped loopify bf16", ref)
    ok = rel <= 1.5 * floor and agree >= floor_agree - 0.03
    print(f"UNROLL EQUIVALENCE: {'PASS' if ok else 'FAIL'} (looped rel {rel:.3e} vs bf16 floor {floor:.3e})")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
