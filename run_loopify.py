"""Train a looped model.

Same entrypoint as nanotron's ``run_train.py`` — including its dataloaders — with
loopify's model and config classes registered:

    torchrun --nproc_per_node=8 run_loopify.py \\
        --config-file configs/qwen3_4b_base_loop_s17_recompute_zero1.yaml
"""

import argparse

from nanotron.trainer import DistributedTrainer
from run_train import get_dataloader

from loopify.config import LoopifyConfig
from loopify.model import LoopifyForTraining


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
    parser.add_argument(
        "--sanity-check-dataloader-interval",
        type=int,
        default=None,
        help="Optional interval to print dataloader samples",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # The config's model_config is already parsed into a LoopifyConfig by
    # ModelArgs (it dispatches on the loop keys), so only the model class needs
    # registering here.
    assert LoopifyConfig is not None
    trainer = DistributedTrainer(args.config_file, model_class=LoopifyForTraining)
    dataloader = get_dataloader(trainer, args.sanity_check_dataloader_interval)
    trainer.train(dataloader)
