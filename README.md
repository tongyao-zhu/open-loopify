<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo/open-loopify-logo-dark.svg">
    <img alt="open-loopify: looped language models, in the open" src="docs/logo/open-loopify-logo.svg" width="600">
  </picture>
</p>

<p align="center"><b>Make a pretrained model think deeper by running its own layers more than once — no new parameters.</b></p>

open-loopify loops a span of a released model's layers — for Qwen3-4B, layers 13–21 run three
times, 54 blocks of compute per token instead of 36 — mid-trains it on open reasoning data, and
exports an ordinary checkpoint that `transformers` and vLLM load with no custom code.

**Weights:** [tyzhu/open-loopify-Qwen3-4B](https://huggingface.co/tyzhu/open-loopify-Qwen3-4B) on the Hugging Face Hub.

## ✨ Highlights

- 🔁 **Loop a pretrained model** — pick the layers and how many times they run. The recipe below
  is for Qwen3.
- 🏋️ **Full training pipeline** on [nanotron](https://github.com/huggingface/nanotron), with a
  dense control generated from the same config.
- 📦 **Ships as an ordinary checkpoint** — the loop is unrolled on export.
- 📊 **Reasoning evaluation** — AIME 2024/25, HMMT Feb 2025, AMC 2023, MATH-500, GPQA-Diamond.
  Interrupted runs resume.
- 🧹 **Clean reasoning data** — traces that were cut off by their generator's length limit are
  filtered out before training.

## 📊 Results

Qwen3-4B-Base, layers [13, 22) looped 3×, mid-trained on 1.57B tokens of open reasoning traces
(one 8×A100-80GB node, 23 hours):

| model | AIME 2024 | AIME 2025 | HMMT Feb 2025 | AMC 2023 | MATH-500 | GPQA-Diamond |
|---|---|---|---|---|---|---|
| Qwen3-4B-Base | 9.6 | 5.0 | 0.4 | 42.5 | 68.6 | 33.8 |
| **+ open-loopify** | **42.5** | **36.7** | **22.5** | **80.3** | **90.4** | **37.9** |
| Qwen3-4B (official, with large-scale RL) | 62.1 | 47.1 | 32.9 | 88.4 | 94.0 | 53.5 |

Given 28k tokens to think instead of 16k (`--max-tokens 28672` in [Evaluate](#-evaluate)) it reaches
50.0 / 41.2 / 24.6 on AIME 2024 / AIME 2025 / HMMT.

**Looped vs dense on the same data.** Both start from Qwen3-4B-Base and see the same tokens in
the same order; the only difference is the loop.

![Looped vs dense accuracy against training tokens on AIME 2024, AIME 2025, MATH-500 and GPQA-Diamond](docs/loop_vs_dense_tokens.png)

At equal training tokens the looped model is ahead almost everywhere (15 of 16 comparisons). At
equal compute — the dense model trained 1.5× longer — the looped model still leads on competition
math (AIME 2024/25 and HMMT Feb 2025: +3.6 points, 95% CI [+0.6, +6.8]; +6.5 [+2.5, +10.7] with
28k tokens to think), ties on MATH-500 and trails on GPQA-Diamond.

All models are scored the same way; see [Evaluate](#-evaluate).

## ⚙️ Install

Needs Linux, an NVIDIA driver for CUDA 12 (≥ 525), and `g++`/`make` (the data loader compiles a
small C++ helper on first use). Training the 4B recipe takes 8 GPUs with 80 GB each; using or
evaluating a model takes one.

```bash
git clone https://github.com/tongyao-zhu/open-loopify && cd open-loopify
conda create -n loopify python=3.10 -y && conda activate loopify
pip install -e . https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

Evaluation runs on vLLM, which pins its own torch — give it a separate environment:

```bash
conda create -n loopify-eval python=3.10 -y && conda activate loopify-eval
pip install vllm==0.13.0 math-verify==0.9.0 datasets==4.2.0
```

## 🚀 Use a looped model

An exported model is an ordinary checkpoint:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "tyzhu/open-loopify-Qwen3-4B"  # or your own export, e.g. hf/qwen3-4b-loop/full_model
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda()

question = "What is the sum of all positive divisors of 36?"
prompt = f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
inputs = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                 return_dict=True, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=16384, do_sample=True, temperature=0.6, top_p=0.95)
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Or serve it: `vllm serve tyzhu/open-loopify-Qwen3-4B`. Give it room to think — answers often run
5k–15k tokens.

## 🔁 Loop your own model

The recipe above, in five steps:

```bash
# 1. convert the checkpoint to nanotron
torchrun --nproc_per_node=1 -m loopify.convert_hf --hf Qwen/Qwen3-4B-Base --out ckpts/qwen3-4b-base

# 2. download, filter and tokenize the data
python data/prepare_tokens.py --model Qwen/Qwen3-4B-Base --out data/reason \
    --sources openthoughts3=3200 --shards 8          # drops traces >= 15k tokens
python data/prepare_tokens.py --model Qwen/Qwen3-4B-Base --out data/reason \
    --sources openr1math=600 --max-doc-tokens 0      # already clean

# 3. write the configs: the looped model and a dense control on the same data
python configs/make_config.py --hf Qwen/Qwen3-4B-Base --ckpt ckpts/qwen3-4b-base \
    --loop 13:22:3 --data "data/reason/openthoughts3_p*:175" data/reason/openr1math:700 \
    --train-steps 3000 --seq-len 16384 --recompute --zero1

# 4. train (8 GPUs)
torchrun --nproc_per_node=8 run_loopify.py \
    --config-file configs/qwen3_4b_base_loop_s17_recompute_zero1.yaml

# 5. export as a plain 54-layer checkpoint
torchrun --nproc_per_node=1 -m loopify.export_hf --unroll \
    --ckpt runs/qwen3_4b_base_loop_s17/checkpoints/3000 \
    --hf-ref Qwen/Qwen3-4B-Base --out hf/qwen3-4b-loop
```

`--loop 13:22:3` means layers 13 to 21 run 3 times. To match the dense control's compute as
well as its data, train it longer by the ratio of blocks per token that `make_config.py` prints —
here 54 / 36 = 1.5, so `--arms dense --train-steps 4500 --tag compute` (the tag gives it its own
run directory, so it does not resume the 3000-step dense run).

Two checks worth running once per model:

```bash
# with no loop, the converted model must reproduce the original
torchrun --nproc_per_node=1 tools/check_equivalence.py --hf Qwen/Qwen3-4B-Base
# the unrolled export must reproduce the looped model
torchrun --nproc_per_node=1 tools/check_unroll.py --hf Qwen/Qwen3-4B-Base \
    --loop-start 13 --loop-end 22 --loop-k 3
```

## 🧪 Evaluate

```bash
conda activate loopify-eval
python tools/eval_reason.py generate --model hf/qwen3-4b-loop/full_model --out results/qwen3-4b-loop \
    --benchmarks aime24,aime25,hmmt25,amc23,math500,gpqa
python tools/eval_reason.py grade --out results/qwen3-4b-loop
```

Zero-shot with the chat template, temperature 0.6, top-p 0.95, up to 16k generated tokens; AIME,
HMMT and AMC are averaged over 8 samples per problem. GPQA-Diamond is gated on the Hub: accept its
terms and run `hf auth login` first, or leave `gpqa` out. A run that is interrupted picks up where it
stopped when started again with the same `--out`.

## 📁 Files

```
src/loopify/
  config.py        LoopifyConfig: loop span and count, plus Qwen3 / OLMo-2 architecture knobs
  model.py         the loop: prelude, repeated span, coda
  data.py          per-source shuffling of training windows
  convert_hf.py    Hugging Face -> nanotron
  export_hf.py     nanotron -> Hugging Face, with --unroll for a plain checkpoint
configs/make_config.py   looped + dense configs from one source
data/prepare_tokens.py   download, filter, tokenize and pack training data
run_loopify.py           training entry point
tools/                   equivalence checks and evaluation
docs/nanotron-changes.md what this fork changes in nanotron
docs/logo/               the logo, drawn by make_logo.py
```

## 📚 Related work

Closest to this project: McLeish et al., [*Teaching Pretrained Language Models to Think Deeper
with Retrofitted Recurrence*](https://arxiv.org/abs/2511.07384). Also: Huginn (depth recurrence
from scratch), Ouro, ETD, Relaxed Recursive Transformers, Loopie.

## License

Apache-2.0, like nanotron. Training data (OpenThoughts3-1.2M, OpenR1-Math-220k) is Apache-2.0.
