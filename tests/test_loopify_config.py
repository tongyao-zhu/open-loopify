"""Unit tests for the loop bookkeeping — no GPU, no checkpoint.

The execution order and the block count are what every compute and cost number in
an experiment is derived from, so they are worth pinning down cheaply.

    python -m pytest tests/test_config.py
"""

import pytest

from loopify.config import LoopifyConfig


def cfg(**kw):
    base = dict(num_hidden_layers=16, hidden_size=64, intermediate_size=128, num_attention_heads=4)
    base.update(kw)
    return LoopifyConfig(**base)


def test_dense_is_the_plain_stack():
    c = cfg()
    assert c.layer_order == list(range(16))
    assert c.effective_num_layers == 16


def test_etd_span_repeats_in_place():
    c = cfg(loop_start=7, loop_end=11, loop_k=3)
    assert c.layer_order == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 7, 8, 9, 10, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    assert c.effective_num_layers == 24


def test_layer_loop_repeats_each_layer_in_place():
    c = cfg(loop_start=7, loop_end=11, loop_k=3, loop_style="layer")
    assert c.layer_order == [0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 12, 13, 14, 15]
    # Same blocks executed as model-loop, only in a different order.
    assert c.effective_num_layers == cfg(loop_start=7, loop_end=11, loop_k=3).effective_num_layers


def test_injection_is_rejected_for_layer_loop():
    with pytest.raises(AssertionError):
        cfg(loop_start=7, loop_end=11, loop_k=3, loop_style="layer", loop_inject_scale=1.0)


def test_k_of_one_is_dense_order():
    c = cfg(loop_start=7, loop_end=11, loop_k=1)
    assert c.layer_order == list(range(16))


def test_empty_span_is_dense_at_any_k():
    c = cfg(loop_start=5, loop_end=5, loop_k=4)
    assert c.layer_order == list(range(16))


def test_span_must_be_inside_the_stack():
    with pytest.raises(AssertionError):
        cfg(loop_start=7, loop_end=17, loop_k=2)
    with pytest.raises(AssertionError):
        cfg(loop_start=11, loop_end=7, loop_k=2)


def test_random_depth_needs_a_span():
    with pytest.raises(AssertionError):
        cfg(loop_k_sample=[1, 2, 3])
    cfg(loop_start=7, loop_end=11, loop_k_sample=[1, 2, 3])  # fine with one


def test_unknown_architecture_knobs_are_rejected():
    with pytest.raises(AssertionError):
        cfg(qk_norm="per_channel")
    with pytest.raises(AssertionError):
        cfg(norm_placement="sandwich")
