"""loopify — turn a pretrained autoregressive transformer into a looped one.

A looped model reuses a span of decoder blocks several times per token instead of
adding new ones: more compute per token, identical parameter count, identical
checkpoint layout. loopify implements this on top of nanotron so that retrofitting
an existing checkpoint is a mid-training run, not a from-scratch one.
"""

from loopify.config import LoopifyConfig
from loopify.model import LoopifyForTraining, LoopifyModel

__all__ = ["LoopifyConfig", "LoopifyForTraining", "LoopifyModel"]
