# How it works, and what changed in nanotron

## The loop

`src/loopify/` sits next to an otherwise-stock nanotron:

- **`config.py`** — `LoopifyConfig`: nanotron's `Qwen2Config` plus `loop_start`, `loop_end` and
  `loop_k`, and the knobs that let one model class cover several families: `qk_norm` (`per_head`
  for Qwen3, `whole` for OLMo-2), `norm_placement` (`pre` / `post`), and a `head_dim` decoupled
  from `hidden_size / num_heads` (Qwen3-4B's heads are 128 wide, not 80).
- **`model.py`** — prelude, middle span repeated `loop_k` times, coda. `loop_style` picks the
  order: `model` repeats the span as a unit (7,8,9,10,7,8,9,10 — ETD, Ouro, Huginn); `layer`
  repeats each layer in place (7,7,7,8,8,8 — Loopie). Optional input injection
  (`loop_inject_scale`) and random depth (`loop_k_sample`).
- **`data.py`** — `ShuffledWindows`. nanotron reads each data source front to back, so with
  several sources every step saw long runs of neighbouring windows, and every arm of an
  experiment logged the same loss curve. Windows are now shuffled within each source, seeded by
  the source's name so every arm sees the same order.
- **`convert_hf.py` / `export_hf.py`** — Hugging Face ↔ nanotron. `export_hf --unroll` writes the
  looped model as a plain checkpoint of its executed depth, copying each repeated layer into its
  slot. This is exact because these blocks do not depend on their layer index; per-layer
  `layer_types` are remapped with them.

Two checks guard the conversions. `tools/check_equivalence.py`: with no loop, the converted model
must be no further from the fp32 original than bf16 rounding alone puts it (OLMo-2-1B: relative
error 1.61e-2 against a bf16 floor of 1.53e-2, same argmax agreement). `tools/check_unroll.py`
holds the unrolled export to the same standard against the looped model. The weight mapping
round-trips bit-exactly — all 179 OLMo-2-1B tensors, to nanotron and back.

Supported: OLMo-2, Qwen3, Qwen2, Llama. Tested end to end on OLMo-2-0425-1B, Qwen3-1.7B-Base and
Qwen3-4B-Base.

## Changes to nanotron

open-loopify is a fork of [nanotron](https://github.com/huggingface/nanotron) at `2411b022`. Each
change is marked with a `loopify:` comment:

- `models/llama.py` — `Parametrizator` gets the full `Config`, not `config.model` (upstream raises
  `AttributeError` on any random-init run at this commit).
- `nn/moe.py` — `grouped_gemm` is an optional import; dense models no longer need it.
- `config/config.py` — `ModelArgs` builds a `LoopifyConfig` when the config has loop keys.
- `data/tokenized_bytes.py` — follows upstream datatrove's argument names; the old S3-only extras
  are refused rather than silently ignored.
- `data/nemo_dataset/blendable_dataset.py` — consumption statistics work for local paths.
- `run_train.py` — accepts a padded vocabulary (model vocabulary ≥ tokenizer's, as in OLMo-2).
- `optim/gradient_accumulator.py` — ZeRO-1's gradient reduce-scatter sat behind a
  `NotImplementedError`, and the code under it dropped the collective's handle and referenced a
  `dp_pg` that does not exist. Fixed and checked step by step against stage 0 on a dense and a
  looped model: identical losses and throughput. Without it a 1B model does not fit on 40GB.

## Things to know

- **Reentrant checkpointing breaks looping under DDP.** nanotron's default raises "mark a variable
  ready only once", because a repeated block's parameters become ready once per repetition.
  loopify checkpoints non-reentrantly, which random depth needs anyway.
- **`log_attn_probs` is on by default** and returns the full attention matrix from every block
  call. At 16k context that alone runs out of memory; generated configs switch it off.
- **Several data sources** are read front to back unless shuffled — see `ShuffledWindows` above.
