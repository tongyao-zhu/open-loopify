"""Visit a packed corpus in a fixed random order instead of front to back.

datatrove reads a ``.ds`` file sequentially and says so in its own docstring: it is
"heavily optimized for sequential reads, and we actually pre-shuffled the data".
Corpora packed without that pre-shuffle therefore reach the model in corpus order,
and since Megatron's blending consumes each source strictly in sequence, one step's
batch is a handful of *adjacent* windows from each source. Any local structure —
a run of same-template synthetic problems, one long document spanning many windows
— then becomes a correlated easy or hard stretch of training, visible as bands in
the loss curve that are identical across every arm of an experiment.

Permuting the window index fixes that without rewriting the data. The cost is
random rather than sequential reads: 128 reads of ~16KB per step, which is nothing
on a local NVMe (datatrove's warning is aimed at S3-backed files).
"""

import numpy as np
from torch.utils.data import Dataset


class ShuffledWindows(Dataset):
    """A packed dataset whose windows are visited in a fixed random order.

    The permutation depends only on the seed and the length, so every arm of an
    experiment — and every resume — sees the same order.
    """

    def __init__(self, dataset: Dataset, seed: int, name: str = ""):
        self.dataset = dataset
        # Mix the source's name in, or two corpora of equal length would be walked
        # in exactly the same order.
        rng = np.random.default_rng([seed, *(ord(c) for c in name[-16:])])
        self.order = rng.permutation(len(dataset))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item):
        # The blend can ask for more windows than a source holds when that source is
        # over-subscribed (Qwen3's mix needs ~525k of megamath's 488k windows over
        # 15000 steps). Wrap into a second pass over the same permutation, as the
        # unwrapped dataset does, rather than indexing off the end.
        return self.dataset[int(self.order[item % len(self.order)])]

    def __getattr__(self, name):
        # Everything else (folder_path, lens, files, subset_log, ...) belongs to
        # the dataset we wrap.
        return getattr(self.__dict__["dataset"], name)
