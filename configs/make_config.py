"""Write training configs for a looped model and its dense control.

Both come from one source, so the loop is the only difference between them: same model,
same data in the same order, same seed, same schedule.

    python configs/make_config.py --hf Qwen/Qwen3-4B-Base --ckpt ckpts/qwen3-4b-base \\
        --loop 13:22:3 --data "data/reason/openthoughts3_p*:175" data/reason/openr1math:700 \\
        --train-steps 3000 --seq-len 16384 --recompute --zero1

writes configs/qwen3_4b_base_loop_s17_recompute_zero1.yaml and the matching _dense_ config.
"""

import argparse
import glob
import json
import os
import re
from pathlib import Path

import yaml

# "dense" is the control: the same model and data, with no layer repeated.
ARMS = {"loop": True, "dense": False}

TOKENS_PER_STEP = 524_288


def parse_data(specs):
    """--data DIR[:WEIGHT] ... -> (folders, weights).

    A DIR may be a glob (quote it), so the shards written by prepare_tokens.py --shards
    come in one argument. Without a weight, a folder is sampled in proportion to the
    tokens it holds, as recorded in its info.json.
    """
    folders, weights = [], []
    for spec in specs:
        path, weight = spec, None
        head, _, tail = spec.rpartition(":")
        if head and re.fullmatch(r"\d+(\.\d+)?", tail):
            path, weight = head, float(tail)
        for folder in sorted(glob.glob(path)) or [path]:
            folder = os.path.abspath(folder)
            if weight is None:
                info = os.path.join(folder, "info.json")
                w = json.load(open(info))["tokens"] if os.path.exists(info) else 1
            else:
                w = weight
            folders.append(folder)
            weights.append(int(w) if float(w).is_integer() else w)
    return folders, weights


def model_name(hf):
    """A short run-name prefix: Qwen/Qwen3-4B-Base -> qwen3_4b_base."""
    parts = Path(hf.rstrip("/")).parts
    # A path into the Hugging Face cache ends in .../models--Org--Name/snapshots/<hash>.
    name = parts[parts.index("snapshots") - 1].split("--")[-1] if "snapshots" in parts else parts[-1]
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def build_config(args, arm, architecture, data_folders, data_weights):
    start, end, k = (int(x) for x in args.loop.split(":"))
    looped = ARMS[arm]
    dp = args.gpus // args.tp
    warmup = min(200, args.train_steps // 10)  # 200 for the real runs; short test runs still decay
    assert TOKENS_PER_STEP % (dp * args.seq_len) == 0, "tokens per step must divide evenly across ranks"
    run = f"{args.name}_{arm}_s{args.seed}" + (f"_{args.tag}" if args.tag else "")
    checkpoints = os.path.join(os.path.abspath(args.run_root), run, "checkpoints")

    model_config = dict(
        architecture,
        loop_start=start if looped else 0,
        loop_end=end if looped else 0,
        loop_k=k if looped else 1,
        loop_inject_scale=0.0,
        loop_k_sample=None,
        loop_k_seed=args.seed,
        loop_style="model",
        # nanotron defaults this to on, returning the full attention matrix from every
        # block call -- at long context that alone runs out of memory.
        log_attn_probs=False,
    )
    return {
        "general": {"project": "open-loopify", "run": run, "seed": args.seed, "ignore_sanity_checks": True},
        "checkpoints": {
            "checkpoints_path": checkpoints,
            # Also the resume path: a relaunch picks up from the newest checkpoint.
            "resume_checkpoint_path": checkpoints,
            "checkpoint_interval": args.ckpt_every,
            "save_initial_state": False,
            "save_final_state": True,
            "checkpoints_path_is_shared_file_system": False,
        },
        "model": {
            "init_method": {"path": os.path.abspath(args.ckpt)},
            "dtype": "bfloat16",
            "make_vocab_size_divisible_by": 1,
            "ddp_bucket_cap_mb": 25,
            "model_config": model_config,
        },
        "optimizer": {
            "zero_stage": 1 if args.zero1 else 0,
            "weight_decay": 0.0,
            "clip_grad": 1.0,
            "accumulate_grad_in_fp32": True,
            "learning_rate_scheduler": {
                "learning_rate": args.lr,
                "lr_warmup_steps": warmup,
                "lr_warmup_style": "linear",
                "lr_decay_style": "cosine",
                "lr_decay_steps": args.train_steps - warmup,
                "min_decay_lr": float(f"{args.lr / 10:g}"),
            },
            "optimizer_factory": {
                "name": "adamW",
                "adam_beta1": 0.9,
                "adam_beta2": 0.95,
                "adam_eps": 1.0e-8,
                "torch_adam_is_fused": True,
            },
        },
        "parallelism": {
            "dp": dp,
            "pp": 1,
            "tp": args.tp,
            "expert_parallel_size": 1,
            "pp_engine": "1f1b",
            "tp_mode": "REDUCE_SCATTER",
            "tp_linear_async_communication": True,
            "recompute_layer": args.recompute,
        },
        "tokens": {
            "sequence_length": args.seq_len,
            "micro_batch_size": 1,
            "batch_accumulation_per_replica": TOKENS_PER_STEP // (dp * args.seq_len),
            "train_steps": args.train_steps,
            "val_check_interval": -1,
            "limit_val_batches": 0,
        },
        "tokenizer": {"tokenizer_name_or_path": args.hf},
        "data_stages": [
            {
                "name": "mid-training",
                "start_training_step": 1,
                "data": {
                    "dataset": {
                        "dataset_folder": data_folders,
                        "dataset_weights": data_weights,
                        "tokenizer_name": args.hf,
                        "vocab_size": model_config["vocab_size"],
                        "token_size_in_bytes": 4,
                        "return_positions": True,
                        "shuffle_files": True,
                    },
                    "num_loading_workers": 2,
                    "seed": args.seed,
                },
            }
        ],
        "logging": {"log_level": "info", "log_level_replica": "info", "iteration_step_info_interval": 1},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf", required=True, help="Hugging Face model id or path")
    parser.add_argument("--ckpt", required=True, help="the nanotron checkpoint loopify.convert_hf wrote for --hf")
    parser.add_argument("--loop", required=True, help="START:END:K -- decoder layers [START, END) run K times")
    parser.add_argument("--data", required=True, nargs="+", metavar="DIR[:WEIGHT]",
                        help="token folders from data/prepare_tokens.py; a DIR may be a quoted glob; "
                        "unweighted folders are sampled in proportion to their tokens")
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--train-steps", type=int, required=True, help=f"steps of {TOKENS_PER_STEP:,} tokens")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-5, help="peak learning rate; cosine decay to a tenth of it")
    parser.add_argument("--gpus", type=int, default=8, help="GPUs per run")
    parser.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    parser.add_argument("--zero1", action="store_true", help="shard optimizer state across data-parallel ranks (the 4B recipe uses it)")
    parser.add_argument("--recompute", action="store_true", help="recompute activations (for long context)")
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--name", help="run-name prefix (default: from the model name)")
    parser.add_argument("--tag", default="", help="appended to the run name")
    parser.add_argument("--run-root", default="runs", help="where checkpoints go")
    parser.add_argument("--out-dir", type=Path, default=Path("configs"))
    args = parser.parse_args()

    from transformers import AutoConfig

    from loopify.convert_hf import hf_config_to_kwargs

    args.name = args.name or model_name(args.hf)
    architecture = hf_config_to_kwargs(AutoConfig.from_pretrained(args.hf))
    folders, weights = parse_data(args.data)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ("_recompute" if args.recompute else "") + ("_zero1" if args.zero1 else "") + (
        f"_tp{args.tp}" if args.tp > 1 else ""
    )
    for arm in args.arms:
        cfg = build_config(args, arm, architecture, folders, weights)
        path = args.out_dir / f"{cfg['general']['run']}{suffix}.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        mc = cfg["model"]["model_config"]
        blocks = mc["num_hidden_layers"] + (mc["loop_end"] - mc["loop_start"]) * (mc["loop_k"] - 1)
        print(f"wrote {path}  ({blocks} blocks per token)")
