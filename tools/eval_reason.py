"""Evaluate a reasoning model the way open reasoning releases report it.

Zero-shot, chat template, long sampled generations, answers graded from the last
\\boxed{} with math_verify. AIME has only 30 problems per year, so it is sampled
several times and reported as the mean accuracy over samples (avg@n).

Two stages, because vLLM and the grader live in different environments here:

    # 1. generate (an environment with vLLM)
    python tools/eval_reason.py generate --model <hf dir> --out <dir>
    # 2. grade (an environment with math_verify)
    python tools/eval_reason.py grade --out <dir>

Generation keeps every finished sample in <out>/generations.partial.jsonl, so an
interrupted run resumes when started again with the same --out.

An unrolled loopify export is a plain checkpoint, so it needs nothing special here.
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def load_benchmarks(names, aime_samples):
    from datasets import load_dataset

    known = {"aime24", "aime25", "hmmt25", "amc23", "math500", "gsm8k", "gpqa"}
    unknown = set(names) - known
    if unknown:
        raise SystemExit(f"unknown benchmark(s) {sorted(unknown)}; choose from {sorted(known)}")
    items = []
    if "aime24" in names:
        for r in load_dataset("HuggingFaceH4/aime_2024", split="train"):
            items.append(dict(bench="aime24", id=f"aime24/{r['id']}", question=r["problem"],
                              answer=str(r["answer"]), n=aime_samples))
    if "aime25" in names:
        for r in load_dataset("MathArena/aime_2025", split="train"):
            items.append(dict(bench="aime25", id=f"aime25/{r['problem_idx']}", question=r["problem"],
                              answer=str(r["answer"]), n=aime_samples))
    if "hmmt25" in names:
        for r in load_dataset("MathArena/hmmt_feb_2025", split="train"):
            items.append(dict(bench="hmmt25", id=f"hmmt25/{r['problem_idx']}", question=r["problem"],
                              answer=str(r["answer"]), n=aime_samples))
    if "amc23" in names:
        for r in load_dataset("math-ai/amc23", split="test"):
            items.append(dict(bench="amc23", id=f"amc23/{r['id']}", question=r["question"],
                              answer=str(r["answer"]), n=aime_samples))
    if "math500" in names:
        for i, r in enumerate(load_dataset("HuggingFaceH4/MATH-500", split="test")):
            items.append(dict(bench="math500", id=f"math500/{i}", question=r["problem"], answer=r["answer"], n=1))
    if "gsm8k" in names:
        for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="test")):
            gold = r["answer"].split("####")[-1].strip().replace(",", "")
            items.append(dict(bench="gsm8k", id=f"gsm8k/{i}", question=r["question"], answer=gold, n=1))
    if "gpqa" in names:
        rng = random.Random(0)  # fixed option order, so every model sees the same questions
        for i, r in enumerate(load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")):
            options = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            options = [o.strip() for o in options]
            order = list(range(4))
            rng.shuffle(order)
            letters = "ABCD"
            body = "\n".join(f"{letters[j]}. {options[k]}" for j, k in enumerate(order))
            gold = letters[order.index(0)]
            q = f"{r['Question'].strip()}\n\n{body}\n\nAnswer with the letter of the correct option."
            items.append(dict(bench="gpqa", id=f"gpqa/{i}", question=q, answer=gold, n=1))
    return items


# Settings that change what a sample is; a resumed run must match them exactly.
SAMPLING_KEYS = ("model", "benchmarks", "aime_samples", "max_tokens", "temperature", "top_p")


def generate(args):
    """Generate every sample, appending each to generations.partial.jsonl the moment it
    finishes. A run killed partway -- a pod recreated, a node reclaimed -- picks up where
    it stopped when started again with the same --out; generations.jsonl is written only
    once every sample exists."""
    from transformers import AutoTokenizer
    from tqdm import tqdm
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    settings = {k: getattr(args, k) for k in SAMPLING_KEYS}
    args_file = out / "generate_args.json"
    if args_file.exists():
        before = {k: json.load(open(args_file)).get(k) for k in SAMPLING_KEYS}
        if before != settings:
            raise SystemExit(f"{out} holds samples generated with {before}; this run asks for {settings}. "
                             "Use a new --out rather than mixing the two.")
    json.dump(vars(args), open(args_file, "w"), indent=2)

    items = load_benchmarks(args.benchmarks.split(","), args.aime_samples)
    # One request per sample, keyed by problem and sample number so a restart can tell
    # which are done, and each with its own deterministic seed: with a shared seed the
    # k samples of a prompt start identically and avg@k collapses toward avg@1.
    requests = {f"{it['id']}#{j}": (idx, j) for idx, it in enumerate(items) for j in range(it["n"])}

    partial = out / "generations.partial.jsonl"
    done = {}
    if partial.exists():
        for line in open(partial):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:  # the line being written when the process died
                continue
            done[row["key"]] = row
    pending = [key for key in requests if key not in done]
    print(f"{len(done)} of {len(requests)} samples already generated, {len(pending)} to go", flush=True)

    if pending:
        tok = AutoTokenizer.from_pretrained(args.model)
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tp,
            max_model_len=args.max_tokens + 4096,
            gpu_memory_utilization=args.gpu_memory_utilization,
            seed=0,
        )
        engine = llm.llm_engine
        prompts = {}
        for key in pending:
            idx, j = requests[key]
            if idx not in prompts:
                prompts[idx] = tok.apply_chat_template(
                    [{"role": "user", "content": f"{items[idx]['question']}\n{INSTRUCTION}"}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            engine.add_request(key, prompts[idx], SamplingParams(
                temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
                seed=idx * 1000 + j, output_kind=RequestOutputKind.FINAL_ONLY))

        with open(partial, "a") as f, tqdm(total=len(requests), initial=len(done), desc="Processed prompts") as bar:
            while engine.has_unfinished_requests():
                for res in engine.step():
                    if not res.finished:
                        continue
                    o = res.outputs[0]
                    idx, _ = requests[res.request_id]
                    row = dict(items[idx], key=res.request_id, text=o.text, tokens=len(o.token_ids),
                               truncated=o.finish_reason == "length")
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                    done[res.request_id] = row
                    bar.update(1)

    missing = [key for key in requests if key not in done]
    assert not missing, f"{len(missing)} samples never finished, e.g. {missing[:3]}"
    tmp = out / "generations.jsonl.tmp"
    with open(tmp, "w") as f:
        for key in requests:
            f.write(json.dumps(done[key]) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out / "generations.jsonl")
    print(f"wrote {len(requests)} generations to {out}")


def last_boxed(text):
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1 : k]
    return None


def grade(args):
    from math_verify import parse, verify

    out = Path(args.out)
    rows = [json.loads(line) for line in open(out / "generations.jsonl")]
    by_bench = {}
    for r in rows:
        # Grade only the final answer, after any <think> block.
        answer_part = r["text"].split("</think>")[-1]
        pred = last_boxed(answer_part) or last_boxed(r["text"])
        if r["bench"] == "gpqa":
            m = re.search(r"[ABCD]", pred or "")
            ok = bool(m) and m.group(0) == r["answer"]
        else:
            try:
                ok = pred is not None and verify(parse(f"${r['answer']}$"), parse(f"${pred}$"))
            except Exception:
                ok = False
        r["pred"], r["correct"] = pred, bool(ok)
        by_bench.setdefault(r["bench"], []).append(r)

    summary = {}
    for bench, rs in sorted(by_bench.items()):
        summary[bench] = dict(
            accuracy=round(100 * sum(r["correct"] for r in rs) / len(rs), 2),
            samples=len(rs),
            mean_tokens=round(sum(r["tokens"] for r in rs) / len(rs)),
            truncated=round(100 * sum(r["truncated"] for r in rs) / len(rs), 1),
        )
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    with open(out / "graded.jsonl", "w") as f:
        for rs in by_bench.values():
            for r in rs:
                f.write(json.dumps(r) + "\n")
    for bench, s in summary.items():
        print(f"{bench:8s} {s['accuracy']:6.2f}%  ({s['samples']} samples, "
              f"{s['mean_tokens']} tokens avg, {s['truncated']}% truncated)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="stage", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--benchmarks", default="aime24,aime25,math500,gpqa,gsm8k")
    g.add_argument("--aime-samples", type=int, default=8)
    # The default can come from the environment, so a queue can ask for a longer budget
    # without changing the shell wrapper that calls this.
    g.add_argument("--max-tokens", type=int, default=int(os.environ.get("EVAL_MAX_TOKENS", 16384)))
    g.add_argument("--temperature", type=float, default=0.6)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--tp", type=int, default=1)
    g.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    r = sub.add_parser("grade")
    r.add_argument("--out", required=True)
    a = p.parse_args()
    generate(a) if a.stage == "generate" else grade(a)
