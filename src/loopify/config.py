"""Configuration for looped (weight-shared recurrent-depth) transformers."""

from dataclasses import dataclass
from typing import List, Optional

from nanotron.config.models_config import Qwen2Config


@dataclass
class LoopifyConfig(Qwen2Config):
    """nanotron's Qwen2Config plus the knobs loopify needs.

    Three additions on top of Qwen2:

    * ``qk_norm`` — the query/key RMSNorm used by Qwen3 (``"per_head"``: one norm
      over ``head_dim``) and by OLMo-2 (``"whole"``: a single norm over the
      concatenated heads, i.e. over ``num_heads * head_dim``).
    * ``norm_placement="post"`` — OLMo-2's reordered norm, where the RMSNorm sits
      on the *output* of each residual branch instead of on its input.
    * ``loop_start`` / ``loop_end`` / ``loop_k`` — run ``decoder[loop_start:loop_end]``
      ``loop_k`` times with shared weights. This is ETD's encode-think-decode:
      a prelude, a middle block repeated k times, and a coda. No new parameters,
      so a looped model has exactly the same checkpoint layout as the dense model
      it was initialised from.

    ``loop_k=1`` reduces to the plain dense model, which is what the equivalence
    test in ``tools/check_equivalence.py`` relies on.
    """

    is_loopify_config: bool = True
    # Width of one attention head. nanotron assumes hidden_size / num_heads, which
    # holds for Qwen3-1.7B and -8B but not -4B (2560 / 32 = 80, while its heads are
    # 128 wide). None keeps nanotron's assumption.
    head_dim: Optional[int] = None
    qk_norm: Optional[str] = None  # None | "per_head" (Qwen3) | "whole" (OLMo-2)
    norm_placement: str = "pre"  # "pre" (Llama/Qwen) | "post" (OLMo-2)
    loop_start: int = 0
    loop_end: int = 0
    loop_k: int = 1
    # How the span is revisited. "model" repeats the span as a unit
    # (7,8,9,10,7,8,9,10,...), which is what ETD, Ouro and Huginn do. "layer"
    # repeats each layer in place before moving on (7,7,7,8,8,8,...), which is
    # Loopie's layer-loop. Same parameters and same FLOPs either way — only the
    # order differs — so the two are directly comparable.
    loop_style: str = "model"
    # Input injection: add `scale` x the prelude's output back into the hidden
    # state before every repetition after the first. Recurrent-depth models
    # (Huginn, McLeish et al.) report this is what keeps a deep loop from
    # drifting away from the state it is supposed to be refining. 0 disables it,
    # and adds no parameters either way.
    loop_inject_scale: float = 0.0
    # Train with a loop count drawn per step instead of a fixed one. A model
    # trained this way can be run at several depths at inference (test-time
    # scaling) rather than only the depth it was trained at.
    loop_k_sample: Optional[List[int]] = None
    loop_k_seed: int = 1234

    def __post_init__(self):
        super().__post_init__()
        assert self.qk_norm in (None, "per_head", "whole"), f"unknown qk_norm {self.qk_norm}"
        assert self.norm_placement in ("pre", "post"), f"unknown norm_placement {self.norm_placement}"
        assert self.loop_style in ("model", "layer"), f"unknown loop_style {self.loop_style}"
        assert not (self.loop_style == "layer" and self.loop_inject_scale), (
            "input injection is defined against the prelude's output, which only has a "
            "meaning for a span repeated as a unit (loop_style='model')"
        )
        assert self.loop_k >= 1
        assert 0 <= self.loop_start <= self.loop_end <= self.num_hidden_layers, (
            f"bad loop span [{self.loop_start}, {self.loop_end}) for {self.num_hidden_layers} layers"
        )
        if self.loop_k_sample:
            assert min(self.loop_k_sample) >= 1, "loop counts must be >= 1"
            assert self.loop_end > self.loop_start, "loop_k_sample needs a non-empty loop span"

    @property
    def layer_order(self) -> List[int]:
        """Indices into ``decoder``, in execution order (the middle span repeated)."""
        s, e, k = self.loop_start, self.loop_end, self.loop_k
        if self.loop_style == "model":
            middle = list(range(s, e)) * k
        else:  # layer-loop: each layer refines its own output before moving on
            middle = [i for i in range(s, e) for _ in range(k)]
        return list(range(s)) + middle + list(range(e, self.num_hidden_layers))

    @property
    def effective_num_layers(self) -> int:
        """Blocks actually executed per token — what compute scales with."""
        return len(self.layer_order)
