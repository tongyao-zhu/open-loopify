"""Download, filter, tokenize and pack training data into the token files nanotron reads.

Each source becomes ``<out>/<source>/<source>.ds`` -- a flat uint32 token stream with
documents separated by EOS (datatrove's ``.ds`` layout) -- and an ``info.json`` recording
its token count. ``--shards N`` splits a source across N worker processes and folders.

Reasoning traces are rendered with the tokenizer's chat template. Traces of
``--max-doc-tokens`` tokens or more (default 15000) are dropped: generated datasets such
as OpenThoughts3 were produced under a 16k-token cap, and a trace that hit it stops
mid-sentence -- a model trained on such traces learns never to finish.

    python data/prepare_tokens.py --model Qwen/Qwen3-4B-Base --out data/reason \\
        --mixture reasoning-v1
"""

import argparse
import json
import os
import time
from multiprocessing import Process
from pathlib import Path

import numpy as np

# Sources yielded as chat messages: reasoning traces, subject to the length gate.
CHAT_SOURCES = {"openthoughts3", "openr1math"}

MIXTURE_DIR = Path(__file__).with_name("mixtures")
DEFAULT_SOURCES = ("tinyGSM-MIND=2000,mathcoder2-synthmath=2000,dolmino_math_synth=1000,"
                   "metamath-owmfilter=400,tulu_math=200")


def job_name(job):
    return job["source"] if job["shard"] is None else f'{job["source"]}_p{job["shard"][0]}'


def preparation_plan(args):
    """Resolve a named recipe or the existing custom-source CLI into worker jobs."""
    recipe = None
    if args.mixture:
        if args.sources is not None or args.shards is not None or args.max_doc_tokens is not None:
            raise ValueError("--mixture sets sources, shards and length filters; omit their overrides")
        if args.mixture not in {p.stem for p in MIXTURE_DIR.glob("*.json")}:
            raise ValueError(f"Unknown mixture: {args.mixture}")
        recipe = json.loads((MIXTURE_DIR / f"{args.mixture}.json").read_text())
        sources = recipe["sources"]
    else:
        sources = [dict(source=k, budget_tokens=int(float(v) * 1e6),
                        shards=args.shards if args.shards is not None else 1,
                        max_doc_tokens=args.max_doc_tokens if args.max_doc_tokens is not None else 15000)
                   for k, v in (x.split("=") for x in (args.sources or DEFAULT_SOURCES).split(","))]
    jobs = []
    for source in sources:
        count = source["shards"]
        if count < 1 or source["budget_tokens"] < count or source["max_doc_tokens"] < 0:
            raise ValueError("Token budgets and shard counts must be positive; length filters must be nonnegative")
        for i in range(count):
            job = dict(source=source["source"], budget=source["budget_tokens"] // count,
                       shard=None if count == 1 else (i, count), max_doc_tokens=source["max_doc_tokens"])
            if recipe:
                job["sampling_weight"] = source["sampling_weight"] / count
            jobs.append(job)
    return jobs, recipe


def check_prepared_job(job, out, model, allow_missing=False):
    """Do not silently reuse shards tokenized with a different recipe or tokenizer."""
    name = job_name(job)
    folder = Path(out) / name
    sidecar = folder / "info.json"
    if allow_missing and not sidecar.exists():
        return
    if not sidecar.exists():
        raise ValueError(f"Incomplete mixture: missing {sidecar}")
    info = json.loads(sidecar.read_text())
    expected = dict(source=name, tokenizer=model, budget=job["budget"], max_doc_tokens=job["max_doc_tokens"])
    if any(info.get(k) != v for k, v in expected.items()):
        raise ValueError(f"{sidecar} does not match this mixture/tokenizer; use a fresh --out directory")
    tokens = info.get("tokens", 0)
    data = folder / f"{name}.ds"
    if tokens <= 0 or not data.is_file() or data.stat().st_size != tokens * 4:
        raise ValueError(f"Incomplete or empty token data in {folder}")


def write_manifest(jobs, args, recipe):
    if recipe:
        for job in jobs:
            check_prepared_job(job, args.out, args.model)
    sources = {job_name(job): json.loads((Path(args.out) / job_name(job) / "info.json").read_text())
               for job in jobs}
    manifest = dict(sources=sources, total_tokens=sum(v["tokens"] for v in sources.values()))
    if recipe:
        manifest.update(mixture=recipe["name"], recipe=recipe, tokenizer=args.model,
                        sampling_weights={job_name(job): job["sampling_weight"] for job in jobs})
    path = Path(args.out) / "manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)
    return manifest

DOLMINO = "allenai/dolmino-mix-1124"
# Subsets of Dolmino's math split. Its gsm8k subset -- GSM8K's own train split -- is
# left out, so it cannot leak into a GSM8K evaluation.
DOLMINO_MATH = ["tinyGSM-MIND", "mathcoder2-synthmath", "dolmino_math_synth", "metamath-owmfilter", "tulu_math"]


def docs(source: str, shard=None):
    """Yield documents: a plain string, or a list of chat messages to be rendered
    with the tokenizer's chat template (reasoning traces, so the trained model
    speaks the format it will be used in)."""
    from datasets import load_dataset

    def sharded(ds):
        """One worker's slice of a streaming dataset, split by file so no worker
        downloads what another one will."""
        return ds if shard is None else ds.shard(num_shards=shard[1], index=shard[0])

    if source == "openthoughts3":
        # QwQ-32B reasoning traces over math, code and science; Apache-2.0.
        role = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant"}
        for r in sharded(load_dataset("open-thoughts/OpenThoughts3-1.2M", split="train", streaming=True)):
            yield [{"role": role[m["from"]], "content": m["value"]} for m in r["conversations"]]
        return
    if source == "openr1math":
        # DeepSeek-R1 traces on NuminaMath problems, kept only when verified; Apache-2.0.
        for r in sharded(load_dataset("open-r1/OpenR1-Math-220k", "default", split="train", streaming=True)):
            if r.get("messages"):
                yield r["messages"]
        return

    if source in DOLMINO_MATH:
        from huggingface_hub import HfApi

        files = sorted(
            f
            for f in HfApi().list_repo_files(DOLMINO, repo_type="dataset")
            # Dolmino mixes plain .jsonl with .gz and .zst within the same split.
            if f.startswith(f"data/math/{source}/")
            and f.endswith((".jsonl", ".json", ".jsonl.gz", ".json.gz", ".jsonl.zst", ".json.zst"))
        )
        assert files, f"no files found for {source}"
        ds = load_dataset(DOLMINO, data_files=files, split="train", streaming=True)
        for r in ds:
            yield r["text"]
    elif source == "megamath":
        for r in load_dataset("LLM360/MegaMath", data_dir="megamath-web-pro", split="train", streaming=True):
            yield r["text"]
    elif source == "finemath":
        for r in load_dataset("HuggingFaceTB/finemath", "finemath-4plus", split="train", streaming=True):
            yield r["text"]
    else:
        raise ValueError(source)


def run(source: str, budget: int, out: str, model: str, max_doc_tokens: int = 0, shard=None):
    from transformers import AutoTokenizer

    # Each shard is its own folder, which is also how the trainer sees it: a data
    # stage lists all of them with equal weight, so shards need no special support.
    name = source if shard is None else f"{source}_p{shard[0]}"
    folder = os.path.join(out, name)
    os.makedirs(folder, exist_ok=True)
    sidecar = os.path.join(folder, "info.json")
    if os.path.exists(sidecar):
        print(f"{name}: already done, skipping", flush=True)
        return

    tok = AutoTokenizer.from_pretrained(model)
    eos = tok.eos_token_id
    path = os.path.join(folder, name + ".ds")
    buf = np.memmap(path, dtype=np.uint32, mode="w+", shape=(budget,))
    n = ndocs = nlong = 0
    t0 = time.time()
    batch = []

    gate = max_doc_tokens if source in CHAT_SOURCES else 0

    def flush(batch):
        nonlocal n, ndocs
        nonlocal nlong
        for ids in tok(batch, add_special_tokens=False).input_ids:
            # A trace that reached its generator's cap has no ending to learn from.
            if gate and len(ids) >= gate:
                nlong += 1
                continue
            ids = list(ids) + [eos]
            take = min(len(ids), budget - n)
            if take <= 0:
                return False
            buf[n : n + take] = np.asarray(ids[:take], dtype=np.uint32)
            n += take
            ndocs += 1
            if n >= budget:
                return False
        return True

    for d in docs(source, shard):
        if isinstance(d, list):  # chat messages: render them the way the model will be prompted
            d = tok.apply_chat_template(d, tokenize=False)
        if not d or len(d) < 50:
            continue
        batch.append(d)
        if len(batch) >= 256:
            if not flush(batch):
                break
            batch = []
            if ndocs % 51200 == 0:
                print(f"{name} {n/1e6:.0f}M tokens {ndocs} docs "
                      f"({nlong} dropped at the {gate}-token gate) {round(time.time()-t0)}s", flush=True)
    if batch and n < budget:
        flush(batch)
    buf.flush()
    del buf

    # A short source leaves the tail of the memmap as zeros; truncate to what we wrote.
    if n < budget:
        with open(path, "r+b") as f:
            f.truncate(n * 4)
    Path(sidecar).write_text(json.dumps(
        {"source": name, "tokens": n, "docs": ndocs, "dropped_too_long": nlong,
         "max_doc_tokens": gate, "budget": budget, "tokenizer": model, "seconds": time.time() - t0}
    ))
    print(f"{name} DONE {n} tokens, {ndocs} docs, {nlong} dropped too long", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="model whose tokenizer to use: Hugging Face id or path")
    p.add_argument("--out", required=True)
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--mixture", choices=sorted(p.stem for p in MIXTURE_DIR.glob("*.json")),
                           help="built-in data mixture: sets sources, budgets, filters and sampling weights")
    selection.add_argument(
        "--sources",
        default=None,
        help="comma-separated source=budget_in_millions_of_tokens",
    )
    p.add_argument(
        "--shards",
        type=int,
        default=None,
        help="split each source across this many worker processes (and folders); "
        "the token budget is split with them",
    )
    p.add_argument(
        "--max-doc-tokens",
        type=int,
        default=None,
        help="drop chat documents this long or longer -- they are traces that hit "
        "their generator's token cap and stop mid-sentence (0 disables)",
    )
    a = p.parse_args()
    try:
        jobs, recipe = preparation_plan(a)
        previous = Path(a.out) / "manifest.json"
        if previous.exists():
            old = json.loads(previous.read_text())
            if old.get("mixture") and (old.get("recipe") != recipe or old.get("tokenizer") != a.model):
                raise ValueError("This output contains a different mixture/tokenizer; use a fresh --out directory")
        if recipe:
            for job in jobs:
                check_prepared_job(job, a.out, a.model, allow_missing=True)
    except ValueError as exc:
        p.error(str(exc))
    os.makedirs(a.out, exist_ok=True)
    procs = [Process(target=run, args=(job["source"], job["budget"], a.out, a.model,
                                     job["max_doc_tokens"], job["shard"])) for job in jobs]
    for q in procs:
        q.start()
    for q in procs:
        q.join()
    failed = [f"{q.name} (exit {q.exitcode})" for q in procs if q.exitcode != 0]
    if failed:
        raise SystemExit("data preparation workers failed: " + ", ".join(failed))

    manifest = write_manifest(jobs, a, recipe)
    print(json.dumps(manifest, indent=1))
