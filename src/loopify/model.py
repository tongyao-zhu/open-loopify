"""Looped transformer on top of nanotron's Qwen2 stack.

The structural change is confined to :meth:`LoopifyModel.forward`: decoder blocks
in ``[loop_start, loop_end)`` run ``loop_k`` times with shared weights. Everything
else (attention kernels, TP/PP plumbing, optimizer, checkpointing) is nanotron's.

Two architecture options are added so the same code covers the checkpoints we
retrofit: query/key RMSNorm (Qwen3, OLMo-2) and OLMo-2's reordered norm.
"""

from typing import Dict, List, Optional, Union

import torch
import torch.utils.checkpoint
from torch import nn

from nanotron import logging
from nanotron.config import ParallelismArgs
from nanotron.logging import log_rank
from nanotron.models import NanotronModel
from nanotron.models.qwen import (
    Loss,
    LossWithZLoss,
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2ForTraining,
    Qwen2MLP,
    Qwen2Model,
    get_flops,
)
from nanotron.nn.layer_norm import LlamaRMSNorm as RMSNorm
from nanotron.nn.layer_norm import TritonRMSNorm
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.block import PipelineBlock, TensorPointer
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelColumnLinear,
    TensorParallelLinearMode,
    TensorParallelRowLinear,
)
from nanotron.random import RandomStates

from loopify.config import LoopifyConfig

logger = logging.get_logger(__name__)


class LoopifyAttention(Qwen2Attention):
    """Qwen2 attention with optional query/key normalisation.

    ``qk_norm="per_head"`` matches Qwen3 (RMSNorm over ``head_dim``);
    ``qk_norm="whole"`` matches OLMo-2 (one RMSNorm over all heads at once).
    """

    def __init__(self, config: LoopifyConfig, parallel_config, tp_pg, cp_pg, layer_idx: int):
        super().__init__(
            config=config, parallel_config=parallel_config, tp_pg=tp_pg, cp_pg=cp_pg, layer_idx=layer_idx
        )
        if config.head_dim is not None and config.head_dim != self.head_dim:
            self._rebuild_for_head_dim(config, parallel_config, tp_pg)
        self.qk_norm = config.qk_norm
        norm_class = TritonRMSNorm if config._fused_rms_norm else RMSNorm
        if self.qk_norm == "per_head":
            self.q_norm = norm_class(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = norm_class(self.head_dim, eps=config.rms_norm_eps)
        elif self.qk_norm == "whole":
            # OLMo-2 normalises across the concatenated heads, so the heads must
            # all live on the same rank.
            assert tp_pg.size() == 1, "qk_norm='whole' normalises across heads; it is only correct with tp=1"
            self.q_norm = norm_class(self.q_size, eps=config.rms_norm_eps)
            self.k_norm = norm_class(self.kv_size, eps=config.rms_norm_eps)

    def _rebuild_for_head_dim(self, config: LoopifyConfig, parallel_config, tp_pg):
        """Size the projections for heads that are not hidden_size / num_heads wide.

        Qwen3-4B projects its 2560-wide residual stream up to 32 heads of 128
        (4096) and back down; nanotron's attention would build 32 heads of 80.
        """
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.local_q_size = self.local_num_heads * self.head_dim
        self.local_kv_size = self.local_num_kv_heads * self.head_dim

        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        async_comm = parallel_config.tp_linear_async_communication if parallel_config is not None else False
        self.qkv_proj = TensorParallelColumnLinear(
            self.hidden_size,
            self.q_size + 2 * self.kv_size,
            pg=tp_pg,
            mode=tp_mode,
            bias=config.attention_bias,
            async_communication=async_comm,
            contiguous_chunks=(self.q_size, self.kv_size, self.kv_size),
            tp_recompute_allgather=parallel_config.tp_recompute_allgather,
        )
        self.o_proj = TensorParallelRowLinear(
            self.q_size,
            self.hidden_size,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=async_comm,
        )
        from nanotron.nn.rotary import FlashRotaryEmbedding

        self.rotary_emb = FlashRotaryEmbedding(
            dim=self.head_dim,
            base=config.rope_theta,
            interleaved=config.rope_interleaved,
            seq_len_interpolation_factor=config.rope_seq_len_interpolation_factor,
        )

    def _forward_packed(self, qkv, seq_length, position_ids, cu_seqlens):
        if self.qk_norm is not None:
            q, k, v = qkv.split([self.local_q_size, self.local_kv_size, self.local_kv_size], dim=-1)
            if self.qk_norm == "per_head":
                q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(-1, self.local_q_size)
                k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(-1, self.local_kv_size)
            else:
                q = self.q_norm(q.contiguous())
                k = self.k_norm(k.contiguous())
            qkv = torch.cat([q, k, v], dim=-1)
        return super()._forward_packed(qkv, seq_length, position_ids, cu_seqlens)


class LoopifyDecoderLayer(Qwen2DecoderLayer):
    """A decoder block that can be pre-norm (Llama/Qwen) or post-norm (OLMo-2).

    NOTE: this deliberately does not call ``Qwen2DecoderLayer.__init__``, which
    hardcodes pre-norm and plain Qwen2 attention. We build the submodules here and
    inherit only ``forward``/``_checkpointed_forward`` from the parent.
    """

    def __init__(self, config: LoopifyConfig, parallel_config, tp_pg, cp_pg, layer_idx: int) -> None:
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.norm_placement = config.norm_placement
        norm_class = TritonRMSNorm if config._fused_rms_norm else RMSNorm

        if self.norm_placement == "pre":
            self.input_layernorm = norm_class(config.hidden_size, eps=config.rms_norm_eps)

        self.attn = LoopifyAttention(
            config=config,
            parallel_config=parallel_config,
            tp_pg=tp_pg,
            cp_pg=cp_pg,
            layer_idx=layer_idx,
        )

        # In pre-norm this normalises the MLP *input*; in OLMo-2's reordered norm
        # it normalises the attention *output*. Both match the HF parameter name.
        self.post_attention_layernorm = norm_class(config.hidden_size, eps=config.rms_norm_eps)
        if self.norm_placement == "post":
            self.post_feedforward_layernorm = norm_class(config.hidden_size, eps=config.rms_norm_eps)

        if config.moe_config and layer_idx in config.moe_config.layers:
            from nanotron.nn.moe import Qwen2MoELayer

            self.mlp = Qwen2MoELayer(
                config=config, parallel_config=parallel_config, tp_pg=tp_pg, layer_idx=layer_idx
            )
        else:
            self.mlp = Qwen2MLP(
                config=config,
                parallel_config=parallel_config,
                tp_pg=tp_pg,
                intermediate_size=config.intermediate_size,
            )

        self.recompute_layer = parallel_config.recompute_layer

    def _checkpointed_forward(self, hidden_states, position_ids, cu_seqlens):
        """Recompute this block, the DDP-compatible way.

        nanotron uses reentrant checkpointing, which cannot handle a block that
        runs several times in one forward pass: DDP sees the same parameter marked
        ready once per backward and raises. Non-reentrant checkpointing has no such
        restriction, and also tolerates the loop count changing between steps.
        """
        return torch.utils.checkpoint.checkpoint(
            self._core_forward, hidden_states, position_ids, cu_seqlens, use_reentrant=False
        )

    def _core_forward(
        self,
        hidden_states: Union[torch.Tensor, TensorPointer],
        position_ids: Union[torch.Tensor, TensorPointer],
        cu_seqlens: Union[torch.Tensor, TensorPointer],
    ) -> List[Union[torch.Tensor, TensorPointer]]:
        if self.norm_placement == "pre":
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.attn(
                hidden_states=hidden_states, position_ids=position_ids, cu_seqlens=cu_seqlens
            )["hidden_states"]
            hidden_states = hidden_states + residual

            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states=hidden_states)["hidden_states"]
            hidden_states = hidden_states + residual
        else:  # OLMo-2: normalise each branch's output, no input norm
            residual = hidden_states
            hidden_states = self.attn(
                hidden_states=hidden_states, position_ids=position_ids, cu_seqlens=cu_seqlens
            )["hidden_states"]
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = hidden_states + residual

            residual = hidden_states
            hidden_states = self.mlp(hidden_states=hidden_states)["hidden_states"]
            hidden_states = self.post_feedforward_layernorm(hidden_states)
            hidden_states = hidden_states + residual

        return hidden_states, position_ids, cu_seqlens


class LoopifyModel(Qwen2Model):
    """Qwen2Model whose middle decoder blocks are executed ``loop_k`` times."""

    def __init__(
        self,
        config: LoopifyConfig,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
    ):
        super().__init__(config=config, parallel_context=parallel_context, parallel_config=parallel_config)

        # Rebuild the decoder with loopify's block. PipelineBlock is lazy — the
        # wrapped modules are only materialised by nanotron's build_model — so
        # replacing the list here costs nothing.
        self.decoder = nn.ModuleList(
            [
                PipelineBlock(
                    p2p=self.p2p,
                    module_builder=LoopifyDecoderLayer,
                    module_kwargs={
                        "config": config,
                        "parallel_config": parallel_config,
                        "tp_pg": parallel_context.tp_pg,
                        "cp_pg": parallel_context.cp_pg,
                        "layer_idx": layer_idx,
                    },
                    module_input_keys={"hidden_states", "position_ids", "cu_seqlens"},
                    module_output_keys={"hidden_states", "position_ids", "cu_seqlens"},
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.layer_order = config.layer_order
        # Drawn on CPU from a fixed seed so every data-parallel rank picks the same
        # loop count for the same micro-batch without any extra communication.
        self._k_generator = torch.Generator().manual_seed(config.loop_k_seed)
        log_rank(
            f"loopify: {config.num_hidden_layers} blocks, "
            f"{config.loop_style}-loop over [{config.loop_start}, {config.loop_end}) "
            f"x{config.loop_k_sample or config.loop_k} "
            f"-> {config.effective_num_layers} executed blocks "
            f"(inject={config.loop_inject_scale}, qk_norm={config.qk_norm}, "
            f"norm_placement={config.norm_placement})",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
        position_ids: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length], -1 is padding
    ):
        assert (
            self.config._attn_implementation != "llama3_ring_attention"
        ), "loopify does not support ring attention"

        output = self.token_position_embeddings(input_ids=input_ids, position_ids=position_ids)

        cu_seqlens = None
        if position_ids.numel() > 0:
            start_indices = torch.where(position_ids.view(-1) == 0)[0]
            cu_seqlens = torch.cat(
                [start_indices, torch.tensor([position_ids.numel()], dtype=torch.int32, device=start_indices.device)]
            ).to(torch.int32)

        decoder_states = {
            "hidden_states": output["input_embeds"],
            "position_ids": output["position_ids"],
            "cu_seqlens": cu_seqlens,
        }

        # The only difference from a dense stack: the middle span is revisited.
        s, e = self.config.loop_start, self.config.loop_end
        k = self.sample_loop_k()
        for layer_idx in range(s):  # prelude: encode
            decoder_states = self.decoder[layer_idx](**decoder_states)

        if self.config.loop_style == "model":
            # Repeat the span as a unit: 7,8,9,10, 7,8,9,10, ...
            encoded = decoder_states["hidden_states"]
            inject = self.config.loop_inject_scale
            for step in range(k):  # think
                if step > 0 and inject:
                    decoder_states["hidden_states"] = decoder_states["hidden_states"] + inject * encoded
                for layer_idx in range(s, e):
                    decoder_states = self.decoder[layer_idx](**decoder_states)
        else:
            # Loopie's layer-loop: each layer refines its own output k times
            # before the computation moves on: 7,7,7, 8,8,8, ...
            for layer_idx in range(s, e):
                for _ in range(k):
                    decoder_states = self.decoder[layer_idx](**decoder_states)

        for layer_idx in range(e, self.config.num_hidden_layers):  # coda: decode
            decoder_states = self.decoder[layer_idx](**decoder_states)

        hidden_states = self.final_layer_norm(input=decoder_states["hidden_states"])["hidden_states"]
        return self.lm_head(x=hidden_states)["logits"]

    def sample_loop_k(self) -> int:
        """Loop count for this forward pass.

        Fixed at ``loop_k`` unless the run trains over a range of depths, in which
        case every rank draws the same value from the shared CPU generator.
        """
        choices = self.config.loop_k_sample
        if not choices or not self.training:
            return self.config.loop_k
        idx = torch.randint(len(choices), (1,), generator=self._k_generator).item()
        return choices[idx]

    def get_flops_per_sec(self, iteration_time_in_sec, sequence_length, global_batch_size):
        """Same as Qwen2Model's, but counts the blocks that actually ran."""
        world_size = self.parallel_context.world_pg.size()
        model_flops, hardware_flops = get_flops(
            num_layers=self.config.effective_num_layers,
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            num_key_value_heads=self.config.num_key_value_heads,
            vocab_size=self.config.vocab_size,
            ffn_hidden_size=self.config.intermediate_size,
            seq_len=sequence_length,
            batch_size=global_batch_size,
        )
        return (
            model_flops / (iteration_time_in_sec * world_size * 1e12),
            hardware_flops / (iteration_time_in_sec * world_size * 1e12),
        )


class LoopifyForTraining(Qwen2ForTraining):
    def __init__(
        self,
        config: LoopifyConfig,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
        random_states: Optional[RandomStates] = None,
    ):
        # Same as Qwen2ForTraining.__init__ but with loopify's model, so we run
        # NanotronModel's initialiser directly instead of the parent's.
        NanotronModel.__init__(self)
        self.model = LoopifyModel(
            config=config, parallel_context=parallel_context, parallel_config=parallel_config
        )

        loss_kwargs = {"tp_pg": parallel_context.tp_pg}
        if config.z_loss_enabled:
            loss_kwargs["z_loss_coefficient"] = config.z_loss_coefficient
        self.loss = PipelineBlock(
            p2p=self.model.p2p,
            module_builder=LossWithZLoss if config.z_loss_enabled else Loss,
            module_kwargs=loss_kwargs,
            module_input_keys={"sharded_logits", "label_ids", "label_mask"},
            module_output_keys={"loss", "z_loss"} if config.z_loss_enabled else {"loss"},
        )
        self.parallel_context = parallel_context
        self.config = config
        self.parallel_config = parallel_config
