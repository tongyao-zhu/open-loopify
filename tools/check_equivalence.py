"""Check that loopify at k=1 reproduces the HuggingFace model it was converted from.

This is the guard rail for every retrofit: if the k=1 loopify model and the HF
model disagree, any difference we later measure between a dense and a looped run
is an implementation artefact, not an effect of looping.

    torchrun --nproc_per_node=1 tools/check_equivalence.py --hf allenai/OLMo-2-0425-1B
"""

from argparse import ArgumentParser

import torch
from transformers import AutoModelForCausalLM

from loopify.convert_hf import build_nanotron_model, hf_config_to_loopify, hf_to_nanotron_weights


def nanotron_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
    with torch.no_grad():
        logits = model.model(input_ids=input_ids, position_ids=position_ids)
    return logits.reshape(input_ids.shape[0], seq_len, -1).float()


def main():
    parser = ArgumentParser()
    parser.add_argument("--hf", type=str, required=True)
    parser.add_argument("--seq-len", type=int, default=256)
    # flash-attn only takes fp16/bf16, so the comparison runs at the training dtype.
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--loop-k", type=int, default=3, help="k used for the 'looping changes the output' check")
    parser.add_argument("--loop-start", type=int, default=None)
    parser.add_argument("--loop-end", type=int, default=None)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda")

    hf_model = AutoModelForCausalLM.from_pretrained(args.hf, dtype=torch.float32).to(device).eval()
    config = hf_config_to_loopify(hf_model.config)
    weights = hf_to_nanotron_weights(hf_model.state_dict(), config)

    torch.manual_seed(0)
    input_ids = torch.randint(0, config.vocab_size, (1, args.seq_len), device=device)
    with torch.no_grad():
        # fp32 HF is the reference; bf16 HF tells us how much of any gap is just
        # rounding, since loopify has to run in bf16 (flash-attn takes no fp32).
        ref = hf_model(input_ids).logits.float()
        hf_lowp = hf_model.to(dtype)(input_ids).logits.float()

    def report(name, got):
        rel = (got - ref).norm().item() / ref.norm().item()
        agree = (got.argmax(-1) == ref.argmax(-1)).float().mean().item()
        print(f"[{name} vs HF fp32] max|diff|={(got - ref).abs().max().item():.4e} "
              f"rel_l2={rel:.4e} argmax_agreement={agree:.4%}")
        return rel, agree

    base_rel, base_agree = report(f"HF {args.dtype}", hf_lowp)

    def build(loop_start, loop_end, loop_k):
        cfg = hf_config_to_loopify(
            hf_model.config, loop_start=loop_start, loop_end=loop_end, loop_k=loop_k
        )
        model, _ = build_nanotron_model(cfg, dtype=dtype, device=device)
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(weights[name].to(param.dtype))
        model.eval()
        return model, cfg

    # 1. k=1 must match HF, to within what bf16 alone already costs.
    model, _ = build(0, 0, 1)
    got = nanotron_logits(model, input_ids)
    rel, agree = report("loopify k=1", got)

    # 2. Looping must actually change the computation.
    s = args.loop_start if args.loop_start is not None else config.num_hidden_layers // 4
    e = args.loop_end if args.loop_end is not None else 3 * config.num_hidden_layers // 4
    looped, cfg = build(s, e, args.loop_k)
    got_looped = nanotron_logits(looped, input_ids)
    print(
        f"[k={args.loop_k}, span [{s},{e})] executed_blocks={cfg.effective_num_layers} "
        f"max|diff vs k=1|={(got_looped - got).abs().max().item():.4e}"
    )

    # loopify is equivalent if it is no further from the fp32 reference than the
    # same model in the same low precision already is.
    ok = rel <= 2 * base_rel and agree >= base_agree - 0.02
    print(
        f"EQUIVALENCE: {'PASS' if ok else 'FAIL'} "
        f"(loopify rel {rel:.3e} vs {args.dtype} floor {base_rel:.3e})"
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
