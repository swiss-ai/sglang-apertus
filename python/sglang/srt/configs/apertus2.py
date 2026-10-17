# SPDX-License-Identifier: Apache-2.0
"""Native configuration for Apertus 2 full-attention and KDA models."""

from transformers.configuration_utils import PretrainedConfig

from sglang.srt.configs.linear_attn_model_registry import (
    LinearAttnModelSpec,
    register_linear_attn_model,
)
from sglang.srt.configs.mamba_utils import (
    KimiLinearCacheParams,
    KimiLinearStateShape,
    mamba2_state_dtype,
)
from sglang.srt.runtime_context import get_parallel


class Apertus2Config(PretrainedConfig):
    model_type = "apertus2"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_local_experts": "n_routed_experts"}
    default_theta = 500000.0

    def __init__(
        self,
        vocab_size=200064,
        hidden_size=768,
        intermediate_size=1920,
        num_hidden_layers=10,
        num_attention_heads=6,
        num_key_value_heads=3,
        head_dim=128,
        hidden_act="sssglu",
        max_position_embeddings=8192,
        rms_norm_eps=1e-5,
        attention_bias=False,
        attention_output_gate=False,
        rope_parameters=None,
        layer_types=None,
        no_rope_layers=None,
        sliding_window=None,
        linear_num_key_heads=None,
        linear_num_value_heads=None,
        linear_key_head_dim=None,
        linear_value_head_dim=None,
        linear_conv_kernel_dim=None,
        gate_lower_bound=None,
        linear_attn_output_gate_bias=None,
        linear_attn_a_log_per_channel=False,
        moe_intermediate_size=448,
        moe_latent_size=None,
        n_routed_experts=128,
        n_shared_experts=1,
        num_experts_per_tok=4,
        first_k_dense_replace=1,
        moe_layer_freq=None,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        use_quantile_balancing=False,
        moe_router_quantile_balancing_method="sigmoid",
        n_group=1,
        topk_group=1,
        sandwich_norm=False,
        embedding_multiplier=27.712812921102035,
        residual_multiplier=0.22360679774997896,
        tie_word_embeddings=False,
        num_nextn_predict_layers=0,
        pad_token_id=3,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        if kwargs.pop("keel", False) or kwargs.pop("keel_alpha", None) is not None:
            raise ValueError("Apertus2 does not support KEEL residuals.")
        if tie_word_embeddings:
            raise ValueError("Apertus2 requires untied embedding and output weights.")
        if num_nextn_predict_layers or kwargs.get("mtp_num_hidden_layers"):
            raise ValueError("Apertus2 MTP layers are not supported.")
        if kwargs.get("mlp_bias", False):
            raise ValueError(
                "Apertus2 does not support mlp_bias=True; use biasless MLP projections."
            )
        if num_experts_per_tok == 1 and norm_topk_prob:
            raise ValueError(
                "Apertus2 requires norm_topk_prob=False when num_experts_per_tok=1."
            )
        if num_experts_per_tok > 1 and not norm_topk_prob:
            raise ValueError(
                "Apertus2 requires norm_topk_prob=True when num_experts_per_tok>1."
            )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_output_gate = attention_output_gate
        self.rope_parameters = (
            dict(rope_parameters)
            if rope_parameters is not None
            else {"rope_theta": self.default_theta, "rope_type": "default"}
        )
        self.rope_parameters.setdefault("partial_rotary_factor", 1.0)
        self.layer_types = (
            list(layer_types)
            if layer_types is not None
            else ["full_attention"] * num_hidden_layers
        )
        if len(self.layer_types) != num_hidden_layers or any(
            value not in {"full_attention", "linear_attention"}
            for value in self.layer_types
        ):
            raise ValueError(
                "Apertus2 layer_types must contain one full_attention or "
                "linear_attention entry per layer. Sliding attention is unsupported."
            )
        self.no_rope_layers = (
            list(no_rope_layers)
            if no_rope_layers is not None
            else [1] * num_hidden_layers
        )
        if len(self.no_rope_layers) != num_hidden_layers or any(
            value not in (0, 1) for value in self.no_rope_layers
        ):
            raise ValueError(
                "Apertus2 no_rope_layers must contain one 0 or 1 per layer."
            )
        self.sliding_window = sliding_window
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.gate_lower_bound = gate_lower_bound
        self.linear_attn_output_gate_bias = (
            bool(self.linear_layer_ids)
            if linear_attn_output_gate_bias is None
            else linear_attn_output_gate_bias
        )
        if linear_attn_a_log_per_channel:
            raise ValueError("Apertus2 currently supports headwise A_log only.")
        self.linear_attn_a_log_per_channel = linear_attn_a_log_per_channel
        self.linear_attn_config = None
        if self.linear_layer_ids:
            geometry = (
                linear_num_key_heads,
                linear_num_value_heads,
                linear_key_head_dim,
                linear_value_head_dim,
                linear_conv_kernel_dim,
            )
            if any(value is None or value <= 0 for value in geometry):
                raise ValueError("Apertus2 KDA requires positive linear_* geometry.")
            if linear_num_key_heads != linear_num_value_heads or (
                linear_key_head_dim != linear_value_head_dim
            ):
                raise ValueError("Apertus2 KDA requires equal key/value head geometry.")
            if gate_lower_bound is not None and not -5.0 <= gate_lower_bound < 0.0:
                raise ValueError("Apertus2 gate_lower_bound must be in [-5, 0).")
            self.linear_attn_config = {
                "num_heads": linear_num_value_heads,
                "head_dim": linear_value_head_dim,
                "short_conv_kernel_size": linear_conv_kernel_dim,
                "kda_layers": [i + 1 for i in self.linear_layer_ids],
                "full_attn_layers": [i + 1 for i in self.full_attention_layer_ids],
            }
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_latent_size = moe_latent_size
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        if not 0 <= first_k_dense_replace <= num_hidden_layers:
            raise ValueError(
                "Apertus2 first_k_dense_replace is outside the layer stack."
            )
        self.moe_layer_freq = (
            list(moe_layer_freq)
            if moe_layer_freq is not None
            else [int(i >= first_k_dense_replace) for i in range(num_hidden_layers)]
        )
        if len(self.moe_layer_freq) != num_hidden_layers or any(
            value not in (0, 1) for value in self.moe_layer_freq
        ):
            raise ValueError(
                "Apertus2 moe_layer_freq must contain one 0 or 1 per layer."
            )
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.use_quantile_balancing = use_quantile_balancing
        self.moe_router_quantile_balancing_method = moe_router_quantile_balancing_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.sandwich_norm = sandwich_norm
        self.embedding_multiplier = embedding_multiplier
        self.residual_multiplier = residual_multiplier
        self.num_nextn_predict_layers = 0
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

    def is_moe_layer(self, layer_idx):
        return bool(self.n_routed_experts and self.moe_layer_freq[layer_idx])

    def is_kda_layer(self, layer_idx):
        return self.layer_types[layer_idx] == "linear_attention"

    @property
    def linear_layer_ids(self):
        return [
            i for i, kind in enumerate(self.layer_types) if kind == "linear_attention"
        ]

    @property
    def full_attention_layer_ids(self):
        return [
            i for i, kind in enumerate(self.layer_types) if kind == "full_attention"
        ]

    @property
    def mamba2_cache_params(self):
        return KimiLinearCacheParams(
            shape=KimiLinearStateShape.create(
                tp_world_size=get_parallel().attn_tp_size,
                num_heads=self.linear_num_value_heads,
                head_dim=self.linear_value_head_dim,
                conv_kernel_size=self.linear_conv_kernel_dim,
            ),
            layers=self.linear_layer_ids,
            dtype=mamba2_state_dtype(self),
        )


register_linear_attn_model(
    LinearAttnModelSpec(
        config_class=Apertus2Config,
        backend_class_name=(
            "sglang.srt.layers.attention.linear.kda_backend.KDAAttnBackend"
        ),
        arch_names=["Apertus2KDAForCausalLM"],
        config_predicate=lambda config: bool(config.linear_layer_ids),
        uses_mamba_radix_cache=True,
        support_mamba_cache=True,
        support_mamba_cache_extra_buffer=True,
    )
)
