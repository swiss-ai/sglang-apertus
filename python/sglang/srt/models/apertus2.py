# SPDX-License-Identifier: Apache-2.0
# Adapted from the Swiss AI Initiative Apertus 2 reference implementation.

import re
from collections.abc import Iterable
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.apertus2 import Apertus2Config
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.activation import SSSGLUAndMul
from sglang.srt.layers.layer_boundary import declare_attn, declare_ffn, make_stages
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layer_boundary.residual.post_norm import PostNormAdd
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import get_moe_runner_backend
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from sglang.srt.models.kimi_linear import KimiDeltaAttention
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import make_pp_layers
from sglang.srt.utils.common import BumpAllocator, set_weight_attrs


def resolve_quantile_balancing_method(config) -> str:
    raw = getattr(config, "moe_router_quantile_balancing_method", "sigmoid")
    method = {
        "average": "sigmoid",
        "histogram": "sigmoid",
        "legacy_average": "legacy",
    }.get(raw, raw)
    if method not in {"sigmoid", "legacy"}:
        raise ValueError(f"Unsupported Apertus2 quantile-balancing method: {raw!r}.")
    return method


def quantile_balancing_routing_native(
    logits: torch.Tensor,
    qb_beta: torch.Tensor,
    top_k: int,
    renormalize: bool,
    sigmoid_selection: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = logits.float()
    scores = logits.sigmoid()
    selection_scores = (scores if sigmoid_selection else logits) - qb_beta.float()
    ids = selection_scores.topk(top_k, dim=-1, sorted=False).indices
    weights = scores.gather(1, ids)
    if renormalize:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return weights.float(), ids.to(torch.int32)


class Apertus2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "sssglu",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        reduce_results: bool = False,
    ):
        super().__init__()
        if hidden_act != "sssglu":
            raise ValueError("Apertus2 currently supports hidden_act='sssglu' only.")
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SSSGLUAndMul()

    def forward(self, hidden_states):
        projected, _ = self.gate_up_proj(hidden_states)
        return self.down_proj(self.act_fn(projected))[0]


class Apertus2Gate(ReplicatedLinear):
    """Preserve FP32 router logits without rounding through BF16 output."""

    def forward(self, hidden_states):
        if hidden_states.is_cuda and hidden_states.dtype == torch.bfloat16:
            return torch.mm(hidden_states, self.weight.T, out_dtype=torch.float32), None
        return F.linear(hidden_states.float(), self.weight.float()), None


class Apertus2MoE(nn.Module):
    def __init__(self, config, layer_idx, quant_config=None, prefix=""):
        super().__init__()
        if config.hidden_act != "sssglu":
            raise ValueError("Apertus2 currently supports hidden_act='sssglu' only.")
        if not config.use_quantile_balancing or (config.n_group, config.topk_group) != (
            1,
            1,
        ):
            raise ValueError("Apertus2 MoE requires ungrouped quantile-beta routing.")
        if not get_moe_runner_backend().is_triton():
            raise ValueError(
                "Apertus2 SSSGLU experts require --moe-runner-backend triton."
            )
        self.routed_scaling_factor = config.routed_scaling_factor
        self.qb_sigmoid_selection = (
            resolve_quantile_balancing_method(config) == "sigmoid"
        )
        self.gate = Apertus2Gate(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        self.gate.qb_beta = nn.Parameter(
            torch.zeros(config.n_routed_experts, dtype=torch.float32),
            requires_grad=False,
        )
        latent_size = config.moe_latent_size
        moe_hidden_size = latent_size or config.hidden_size
        self.latent_down_proj = self.latent_up_proj = None
        if latent_size:
            self.latent_down_proj = ReplicatedLinear(
                config.hidden_size,
                latent_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.latent_down_proj",
            )
            self.latent_up_proj = ReplicatedLinear(
                latent_size,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.latent_up_proj",
            )
        self.experts = FusedMoE(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=moe_hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=layer_idx,
            quant_config=quant_config,
            activation="sssglu",
            inplace=False,
            reduce_results=False,
            prefix=f"{prefix}.experts",
        )
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            layer_id=layer_idx,
            renormalize=config.norm_topk_prob,
            custom_routing_function=self._qb_routing,
            output_format=TopKOutputFormat.STANDARD,
        )
        self.shared_experts = (
            Apertus2MLP(
                config.hidden_size,
                config.moe_intermediate_size * config.n_shared_experts,
                config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_experts",
            )
            if config.n_shared_experts
            else None
        )

    def _qb_routing(self, hidden_states, gating_output, topk, renormalize):
        return quantile_balancing_routing_native(
            gating_output,
            self.gate.qb_beta,
            topk,
            renormalize,
            self.qb_sigmoid_selection,
        )

    def forward(self, hidden_states):
        router_logits, _ = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        routed_input = (
            self.latent_down_proj(hidden_states)[0]
            if self.latent_down_proj is not None
            else hidden_states
        )
        routed = self.experts(routed_input, topk_output)
        # TP1 makes the expert output complete before the replicated up projection.
        # Stage boundaries own output sums; scale before the BF16 projection.
        routed = routed * self.routed_scaling_factor
        if self.latent_up_proj is not None:
            routed = self.latent_up_proj(routed)[0]
        if self.shared_experts is not None:
            routed = routed + self.shared_experts(hidden_states)
        return routed


class Apertus2Attention(nn.Module):
    def __init__(self, config, layer_idx, quant_config=None, prefix=""):
        super().__init__()
        tp_size = get_parallel().tp_size
        self.num_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = max(1, config.num_key_value_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.use_rope = config.no_rope_layers[layer_idx] == 1
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            prefix=f"{prefix}.o_proj",
        )
        self.attention_output_gate = config.attention_output_gate
        if self.attention_output_gate:
            self.g_proj = ColumnParallelLinear(
                config.hidden_size,
                config.num_attention_heads * self.head_dim,
                bias=config.attention_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )
        self.rotary_emb = (
            get_rope(
                self.head_dim,
                rotary_dim=int(
                    config.rope_parameters.get("partial_rotary_factor", 1.0)
                    * self.head_dim
                ),
                max_position=config.max_position_embeddings,
                base=config.rope_parameters["rope_theta"],
                rope_scaling=config.rope_parameters,
                is_neox_style=True,
            )
            if self.use_rope
            else None
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_idx,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, positions, forward_batch, zero_allocator=None):
        if hidden_states.shape[0] == 0:
            return hidden_states
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.reshape(-1, self.head_dim)).reshape_as(q)
        k = self.k_norm(k.reshape(-1, self.head_dim)).reshape_as(k)
        if self.use_rope:
            q, k = self.rotary_emb(positions, q, k)
        output = self.attn(q, k, v, forward_batch)
        if self.attention_output_gate:
            gate, _ = self.g_proj(hidden_states)
            output = (output * gate.float().sigmoid()).to(output.dtype)
        return self.o_proj(output)[0]


class Apertus2KDAAttention(KimiDeltaAttention):
    def __init__(self, config, layer_idx, quant_config=None, prefix=""):
        kimi_config = SimpleNamespace(
            linear_attn_config=config.linear_attn_config,
            dtype=torch.get_default_dtype(),
        )
        super().__init__(
            layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            config=kimi_config,
            quant_config=quant_config,
            rms_norm_eps=config.rms_norm_eps,
            prefix=prefix,
            safe_gate=config.gate_lower_bound is not None,
            lower_bound=config.gate_lower_bound,
            reduce_results=False,
        )
        self.output_gate_bias = None
        if config.linear_attn_output_gate_bias:
            self.output_gate_bias = nn.Parameter(
                torch.empty(self.num_heads * self.head_dim // self.shard_tp_size)
            )
            set_weight_attrs(
                self.output_gate_bias, {"weight_loader": sharded_weight_loader(0)}
            )

    def forward_qkvbfg_fused(self, hidden_states):
        if self.output_gate_bias is None:
            return super().forward_qkvbfg_fused(hidden_states)
        projected = self.fused_qkvbfg_a_proj(hidden_states)
        qkv, beta, fg_a = projected.split(self.split_sizes, dim=-1)
        # Add the trained bias in the GEMM epilogue before storing BF16 output.
        # A separate addition to the BF16 matmul result adds a rounding step.
        bias = torch.stack(
            (torch.zeros_like(self.output_gate_bias), self.output_gate_bias)
        ).unsqueeze(1)
        forget_gate, output_gate = torch.baddbmm(
            bias,
            fg_a.view(-1, 2, self.head_dim).transpose(0, 1),
            self.fused_fg_b_proj.weight.transpose(-1, -2),
        )
        return qkv, beta, forget_gate, output_gate

    def forward(self, hidden_states, positions, forward_batch, zero_allocator):
        if forward_batch.forward_mode != ForwardMode.EXTEND:
            return super().forward(
                hidden_states, positions, forward_batch, zero_allocator
            )
        project = (
            self.forward_qkvbfg_fused if self.do_fuse_qkvbfg else self.forward_qkvbfg
        )
        mixed_qkv, beta, forget_gate, output_gate = project(hidden_states)
        # Flat raw gates select the backend's raw-beta activation contract.
        core = self.attn(
            forward_batch,
            mixed_qkv=mixed_qkv,
            a=forget_gate.unsqueeze(0),
            b=beta.unsqueeze(0),
        )
        norm_gate = output_gate.unflatten(-1, (-1, self.head_dim))
        core = self.o_norm(core, norm_gate).squeeze(0).flatten(-2)
        return self.o_proj(core)[0]


class ScaledPostNormAdd(PostNormAdd):
    def __init__(self, norm, multiplier, *, applied_at_exit=False):
        super().__init__(norm, applied_at_exit=applied_at_exit)
        self.multiplier = multiplier

    def update(self, hidden_states, residual):
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        return residual + self.multiplier * hidden_states


class Apertus2DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx, quant_config=None, prefix=""):
        super().__init__()
        attention_class = Apertus2Attention
        if config.is_kda_layer(layer_idx):
            attention_class = Apertus2KDAAttention
        self.self_attn = attention_class(
            config, layer_idx, quant_config=quant_config, prefix=f"{prefix}.self_attn"
        )
        sparse = config.is_moe_layer(layer_idx)
        self.mlp = (
            Apertus2MoE(config, layer_idx, quant_config, f"{prefix}.mlp")
            if sparse
            else Apertus2MLP(
                config.hidden_size,
                config.intermediate_size,
                config.hidden_act,
                quant_config,
                f"{prefix}.mlp",
            )
        )
        self.attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.sandwich_norm
            else None
        )
        self.post_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.sandwich_norm
            else None
        )
        attn_update = ScaledPostNormAdd(
            self.post_attention_layernorm, config.residual_multiplier
        )
        ffn_update = ScaledPostNormAdd(
            self.post_feedforward_layernorm,
            config.residual_multiplier,
            applied_at_exit=True,
        )
        next_sparse = layer_idx + 1 < config.num_hidden_layers and config.is_moe_layer(
            layer_idx + 1
        )
        self.attn_boundary, self.ffn_boundary = make_stages(
            (declare_attn(update=attn_update), self.attention_layernorm),
            (
                declare_ffn(
                    sparse=sparse,
                    next_layer_sparse=next_sparse,
                    update=ffn_update,
                ),
                self.feedforward_layernorm,
            ),
            previous=declare_ffn(
                sparse=config.is_moe_layer(layer_idx - 1),
                next_layer_sparse=sparse,
                update=ffn_update,
            )
            if layer_idx
            else None,
            terminal=layer_idx == config.num_hidden_layers - 1,
        )

    def forward(self, positions, hidden_states, forward_batch, zero_allocator):
        hidden_states = self.attn_boundary.prepare(hidden_states, forward_batch)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
        )
        hidden_states = self.attn_boundary.finish(hidden_states, forward_batch)
        hidden_states = self.ffn_boundary.prepare(hidden_states, forward_batch)
        return self.ffn_boundary.finish(self.mlp(hidden_states), forward_batch)


class Apertus2Model(nn.Module):
    def __init__(self, config: Apertus2Config, quant_config=None, prefix=""):
        super().__init__()
        self.pp_group = get_parallel().pp_group
        self.embedding_multiplier = config.embedding_multiplier
        self.embed_tokens = (
            VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
            if self.pp_group.is_first_rank
            else PPMissingLayer()
        )
        self.layers, self.start_layer, self.end_layer = make_pp_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Apertus2DecoderLayer(config, idx, quant_config, prefix),
            prefix=f"{prefix}.layers",
        )
        self.norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if self.pp_group.is_last_rank
            else PPMissingLayer()
        )

    def forward(
        self,
        input_ids,
        positions,
        forward_batch,
        inputs_embeds=None,
        pp_proxy_tensors=None,
    ):
        if self.pp_group.is_first_rank:
            hidden_states = (
                self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
            )
            hidden_states = hidden_states * self.embedding_multiplier
            residual_batch.start(forward_batch)
        else:
            assert pp_proxy_tensors is not None
            hidden_states = self.layers[self.start_layer].attn_boundary.from_pp(
                pp_proxy_tensors, forward_batch
            )
        zero_allocator = BumpAllocator(
            buffer_size=(self.end_layer - self.start_layer) * 2,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        for layer_idx in range(self.start_layer, self.end_layer):
            recorder = get_global_expert_distribution_recorder()
            ctx = recorder.with_current_layer(layer_idx)
            with ctx:
                hidden_states = self.layers[layer_idx](
                    positions, hidden_states, forward_batch, zero_allocator
                )
        if not self.pp_group.is_last_rank:
            return residual_batch.to_pp(hidden_states, forward_batch)
        return residual_batch.final_norm(
            hidden_states, forward_batch, self.norm, skip_empty=True
        )


_EXPERT_WEIGHT = re.compile(
    r"^(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)


class Apertus2ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkvbfg_a_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
            "b_proj",
            "f_a_proj",
            "g_a_proj",
        ],
        "fused_fg_b_proj": ["f_b_proj", "g_b_proj"],
    }

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        if quant_config is not None or torch.get_default_dtype() != torch.bfloat16:
            raise ValueError(
                "Apertus2 currently supports unquantized BF16 inference only."
            )
        parallel = get_parallel()
        if (
            parallel.tp_size != 1
            or parallel.attn_dp_size > 1
            or parallel.moe_ep_size > 1
        ):
            raise ValueError(
                "Apertus2 currently requires --tp-size 1 without DP/EP. "
                "Use --pp-size to distribute layers across GPUs."
            )
        if config.linear_layer_ids and not isinstance(self, Apertus2KDAForCausalLM):
            raise ValueError("KDA layers require architecture Apertus2KDAForCausalLM.")
        if isinstance(self, Apertus2KDAForCausalLM) and not config.linear_layer_ids:
            raise ValueError("Apertus2KDAForCausalLM requires linear_attention layers.")
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_parallel().pp_group
        self.model = Apertus2Model(config, quant_config, f"{prefix}.model".lstrip("."))
        self.start_layer = self.model.start_layer
        self.end_layer = self.model.end_layer
        self.lm_head = (
            ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.lm_head".lstrip("."),
            )
            if self.pp_group.is_last_rank
            else PPMissingLayer()
        )
        self.logits_processor = LogitsProcessor(config)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
        )
        if not self.pp_group.is_last_rank:
            return hidden_states
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def _is_non_local_pp_weight(self, name):
        if self.pp_group.world_size == 1:
            return False
        layer_idx = get_layer_id(name)
        if layer_idx is not None:
            return not self.model.start_layer <= layer_idx < self.model.end_layer
        if name.startswith("model.embed_tokens."):
            return not self.pp_group.is_first_rank
        if name.startswith(("model.norm.", "lm_head.")):
            return not self.pp_group.is_last_rank
        return False

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        params = dict(self.named_parameters())
        loaded_params = set()
        for name, weight in weights:
            if self._is_non_local_pp_weight(name) or "rotary_emb." in name:
                continue
            expert_match = _EXPERT_WEIGHT.match(name)
            if expert_match is not None:
                parent, expert_id, projection = expert_match.groups()
                shard_id = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}[
                    projection
                ]
                target = f"{parent}.{'w2' if shard_id == 'w2' else 'w13'}_weight"
                param = params[target]
                param.weight_loader(
                    param, weight, target, expert_id=int(expert_id), shard_id=shard_id
                )
                loaded_params.add(target)
                continue
            layer_idx = get_layer_id(name)
            is_kda = layer_idx is not None and self.config.is_kda_layer(layer_idx)
            if is_kda and name.endswith(".self_attn.A_log"):
                weight = weight.reshape(1, 1, -1, 1)
            if is_kda and name.endswith(".self_attn.g_b_proj.bias"):
                name = name.replace(".g_b_proj.bias", ".output_gate_bias")
            mappings = [
                (".gate_proj.", ".gate_up_proj.", 0),
                (".up_proj.", ".gate_up_proj.", 1),
            ]
            if is_kda:
                mappings += [
                    (f".{leaf}.", ".fused_qkvbfg_a_proj.", shard)
                    for shard, leaf in enumerate(
                        (
                            "q_proj",
                            "k_proj",
                            "v_proj",
                            "b_proj",
                            "f_a_proj",
                            "g_a_proj",
                        )
                    )
                ]
                mappings += [
                    (".f_b_proj.", ".fused_fg_b_proj.", 0),
                    (".g_b_proj.", ".fused_fg_b_proj.", 1),
                    (".q_conv1d.", ".qkv_conv1d.", 0),
                    (".k_conv1d.", ".qkv_conv1d.", 1),
                    (".v_conv1d.", ".qkv_conv1d.", 2),
                ]
            else:
                mappings += [
                    (".q_proj.", ".qkv_proj.", "q"),
                    (".k_proj.", ".qkv_proj.", "k"),
                    (".v_proj.", ".qkv_proj.", "v"),
                ]
            for leaf, packed, shard_id in mappings:
                if leaf in name:
                    name = name.replace(leaf, packed)
                    param = params[name]
                    param.weight_loader(param, weight, shard_id)
                    break
            else:
                param = params[name]
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
            loaded_params.add(name)
        return loaded_params


class Apertus2KDAForCausalLM(Apertus2ForCausalLM):
    pass


EntryClass = [Apertus2ForCausalLM, Apertus2KDAForCausalLM]
