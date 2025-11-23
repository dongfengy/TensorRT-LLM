import os
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn.parameter import Parameter
from tqdm import tqdm
from transformers import GptOssConfig

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType
from tensorrt_llm.models.modeling_utils import QuantAlgo

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import (PositionalEmbeddingParams,
                                           PredefinedAttentionMask, RopeParams)
from ..distributed import (AllReduce, AllReduceFusionOp, AllReduceParams,
                           allgather)
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding

# isort and yapf will fight against each other here, so we disable isort
# isort: off
from ..modules.fused_moe import (MoE, MoEWeightLoadingMode,
                                 RenormalizeMoeRoutingMethod, TritonFusedMoE,
                                 create_moe)
from ..modules.fused_moe.routing import (get_cached_perfect_router_logits,
                                         precompute_common_perfect_router_logits
                                         )
# isort: on
from typing import Dict, List, Optional

from tensorrt_llm._torch.modules.fused_moe import (BaseMoeRoutingMethod,
                                                   RenormalizeMoeRoutingMethod,
                                                   TritonFusedMoE, create_moe)
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP

from ..modules.linear import Linear, TensorParallelMode
from ..modules.rms_norm import RMSNorm
from ..speculative import SpecMetadata
from ..utils import Fp4QuantizedTensor
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import (DecoderModel, duplicate_kv_weight, filter_weights,
                             register_auto_model)

# Use TinyGEMM when the number of tokens is not larger than this threshold
MIN_LATENCY_TINYGEMM_NUM_TOKENS = 128


class AttentionBlock(Attention):

    def __init__(
        self,
        config: ModelConfig[GptOssConfig],
        layer_idx: int = 0,
        use_custom_cublas_mm: bool = False,
    ):
        pretrained_config = config.pretrained_config

        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.yarn,
            rope=RopeParams(
                dim=pretrained_config.head_dim,
                theta=pretrained_config.rope_theta,
                scale_type=RotaryScalingType.yarn,
                scale=pretrained_config.rope_scaling['factor'],
                max_positions=pretrained_config.max_position_embeddings,
                original_max_positions=pretrained_config.
                rope_scaling['original_max_position_embeddings'],
                beta_fast=pretrained_config.rope_scaling['beta_fast'],
                beta_slow=pretrained_config.rope_scaling['beta_slow'],
                duplicate_data=False),
            is_neox=False,
        )

        super().__init__(
            hidden_size=pretrained_config.hidden_size,
            num_attention_heads=pretrained_config.num_attention_heads,
            num_key_value_heads=pretrained_config.num_key_value_heads,
            max_position_embeddings=pretrained_config.max_position_embeddings,
            bias=True,
            pos_embd_params=pos_embd_params,
            layer_idx=layer_idx,
            dtype=pretrained_config.torch_dtype,
            dense_bias=True,
            config=config,
            q_scaling=1.0,
            attention_chunk_size=None,
            use_custom_cublas_mm=use_custom_cublas_mm,
        )

        # Only apply sliding window to every other layer
        self.sliding_window = pretrained_config.sliding_window if layer_idx % 2 == 0 else None

        self.sinks = Parameter(
            torch.empty(pretrained_config.num_attention_heads // self.tp_size,
                        dtype=torch.float32))
        self.norm = RMSNorm(hidden_size=pretrained_config.hidden_size,
                            eps=pretrained_config.rms_norm_eps,
                            dtype=pretrained_config.torch_dtype)

    def forward(
        self,
        position_ids: Optional[torch.LongTensor],
        hidden_states: torch.Tensor | Fp4QuantizedTensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.
        CAUSAL,
        all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        attention_window_size = self.sliding_window
        attn_output = super().forward(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            attention_mask=attention_mask,
            all_reduce_params=all_reduce_params,
            lora_params=lora_params,
            attention_window_size=attention_window_size,
            attention_sinks=self.sinks.data,
            **kwargs)
        return attn_output, residual

    def load_weights(self, weights: Dict):
        sinks = weights[0]['sinks'][self.num_heads *
                                    self.tp_rank:self.num_heads *
                                    (self.tp_rank + 1)]
        self.sinks.data = sinks.to(torch.float32).to("cuda")


def _is_ref_moe_module(module: nn.Module) -> bool:
    """Returns True when module is the reference MoE wrapper itself or one of its children."""
    if isinstance(module, RefGatedMLPFusedMoE):
        return True
    parent = getattr(module, "_ref_moe_parent", None)
    return parent is not None


class RefGatedMLPFusedMoE(nn.Module):

    def __init__(
        self,
        num_experts: int,
        routing_method: BaseMoeRoutingMethod,
        hidden_size: int,
        intermediate_size: int,
        dtype: Optional[torch.dtype] = None,
        model_config: ModelConfig = ModelConfig(),
        use_cute_dsl_blockscaling_mm: bool = False,
        bias=False,
        swiglu_alpha: Optional[float] = None,
        swiglu_beta: Optional[float] = None,
        swiglu_limit: Optional[float] = None,
        weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.VANILLA
    ):
        super().__init__()
        self.num_experts = num_experts
        self.routing_method = routing_method
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.bias = bias

        self.dtype = dtype
        self.quant_config = model_config.quant_config
        self.weight_loading_mode = weight_loading_mode

        self.swiglu_alpha = swiglu_alpha
        self.swiglu_beta = swiglu_beta
        self.swiglu_limit = swiglu_limit

        def custom_swiglu(x, expert_idx: int = 0):
            gate, value = x.chunk(2, dim=-1)
            if swiglu_limit[expert_idx] is not None and swiglu_limit[
                    expert_idx] != float("inf"):
                gate = gate.clamp(max=swiglu_limit[expert_idx])
                value = value.clamp(min=-swiglu_limit[expert_idx],
                                    max=swiglu_limit[expert_idx])

            alpha = swiglu_alpha[expert_idx] if swiglu_alpha[
                expert_idx] is not None else 1.0
            gate_act = gate * torch.sigmoid(gate * alpha)

            beta = swiglu_beta[expert_idx] if swiglu_beta[
                expert_idx] is not None else 0.0

            return gate_act * (value + beta)

        self.experts = nn.ModuleList([
            GatedMLP(
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                bias=bias,
                dtype=self.dtype,
                config=model_config,
                use_cute_dsl_blockscaling_mm=use_cute_dsl_blockscaling_mm,
                activation=lambda x: custom_swiglu(x, expert_idx=expert_idx),
            ) for expert_idx in range(self.num_experts)
        ])

    def _get_expert_entry(self, container, expert_idx: int):
        if container is None:
            return None
        if isinstance(container, dict):
            return container.get(expert_idx)
        if isinstance(container, (list, tuple)):
            return container[expert_idx]
        if torch.is_tensor(container):
            if container.dim() == 0:
                return container
            return container[expert_idx]
        raise TypeError(f"Unsupported container type for expert weights: "
                        f"{type(container)}")

    def _expand_fused_gate_up_weights(self, weights: Dict) -> Dict:
        if self.weight_loading_mode != MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
            assert self.weight_loading_mode == MoEWeightLoadingMode.VANILLA
            return weights

        required = ["gate_up_proj", "down_proj"]
        for key in required:
            if key not in weights:
                raise KeyError(
                    f"Missing required key '{key}' for fused gate-up weights.")

        expanded = dict(weights)
        gate_up_scales = weights.get("gate_up_proj_weight_scale")
        down_scales = weights.get("down_proj_weight_scale")

        for expert in range(self.num_experts):
            fused_gate = self._get_expert_entry(weights["gate_up_proj"], expert)
            if fused_gate is None:
                raise KeyError(
                    f"Missing fused gate_up_proj entry for expert {expert}")
            fused_gate = fused_gate.transpose(0, 1).contiguous()
            w1_weight, w3_weight = fused_gate.chunk(2, dim=0)
            expanded[f"{expert}.w1.weight"] = w1_weight
            expanded[f"{expert}.w3.weight"] = w3_weight

            fused_bias = self._get_expert_entry(
                weights.get("gate_up_proj.bias"), expert)
            if fused_bias is not None:
                w1_bias, w3_bias = fused_bias.chunk(2, dim=0)
                expanded[f"{expert}.w1.bias"] = w1_bias
                expanded[f"{expert}.w3.bias"] = w3_bias

            fused_scale = (self._get_expert_entry(gate_up_scales, expert)
                           if gate_up_scales is not None else None)
            if fused_scale is not None:
                fused_scale = fused_scale.transpose(0, 1).contiguous()
                w1_scale, w3_scale = fused_scale.chunk(2, dim=0)
                expanded[f"{expert}.w1.weight_scale"] = w1_scale
                expanded[f"{expert}.w3.weight_scale"] = w3_scale

            down_weight = self._get_expert_entry(weights["down_proj"], expert)
            down_weight = down_weight.transpose(0, 1).contiguous()
            expanded[f"{expert}.w2.weight"] = down_weight

            down_bias = self._get_expert_entry(weights.get("down_proj.bias"),
                                               expert)
            if down_bias is not None:
                expanded[f"{expert}.w2.bias"] = down_bias

            down_scale = (self._get_expert_entry(down_scales, expert)
                          if down_scales is not None else None)
            if down_scale is not None:
                expanded[f"{expert}.w2.weight_scale"] = down_scale.transpose(
                    0, 1).contiguous()

            if "gate_up_proj_weight_scale_2" in weights:
                expanded[f"{expert}.w1.weight_scale_2"] = weights[
                    "gate_up_proj_weight_scale_2"]
                expanded[f"{expert}.w3.weight_scale_2"] = weights[
                    "gate_up_proj_weight_scale_2"]
            if "down_proj_weight_scale_2" in weights:
                expanded[f"{expert}.w2.weight_scale_2"] = weights[
                    "down_proj_weight_scale_2"]

            if "gate_up_proj_input_scale" in weights:
                expanded[f"{expert}.w1.input_scale"] = weights[
                    "gate_up_proj_input_scale"]
                expanded[f"{expert}.w3.input_scale"] = weights[
                    "gate_up_proj_input_scale"]
            if "down_proj_input_scale" in weights:
                expanded[f"{expert}.w2.input_scale"] = weights[
                    "down_proj_input_scale"]

        return expanded

    def forward(self, hidden_states: torch.Tensor,
                router_logits: torch.Tensor) -> torch.Tensor:
        assert hidden_states.shape[-1] == self.hidden_size
        hidden_states = hidden_states.view(-1, self.hidden_size)

        selected_experts, routing_weights = self.routing_method.apply(
            router_logits)

        final_hidden_states = torch.zeros(hidden_states.shape,
                                          dtype=hidden_states.dtype,
                                          device=hidden_states.device)

        for expert_id in range(self.num_experts):
            if not torch.any(selected_experts == expert_id):
                continue
            batch_idx, nth_expert = torch.where(selected_experts == expert_id)
            expert_inputs = hidden_states[batch_idx]

            output = self.experts[expert_id](expert_inputs)
            final_hidden_states[batch_idx] += routing_weights[
                batch_idx, nth_expert, None] * output.float()

        final_hidden_states = final_hidden_states.reshape(hidden_states.shape)
        return final_hidden_states

    def load_weights(self, weights: List[Dict]):
        assert len(weights) == 1
        weights = self._expand_fused_gate_up_weights(weights[0])
        print("hello alpha:", self.swiglu_alpha)
        print("hello beta:", self.swiglu_beta)
        print("hello limit:", self.swiglu_limit)

        for expert in range(self.num_experts):
            gate_up_proj_weights = [{}, {}]
            down_proj_weights = [{}]

            gate_up_proj_weights[0]['weight'] = weights[f"{expert}.w1.weight"]
            gate_up_proj_weights[1]['weight'] = weights[f"{expert}.w3.weight"]
            down_proj_weights[0]['weight'] = weights[f"{expert}.w2.weight"]
            print("hello w1 weight:", gate_up_proj_weights[0]['weight'])
            print("hello w3 weight:", gate_up_proj_weights[1]['weight'])
            print("hello w2 weight:", down_proj_weights[0]['weight'])
            if self.bias:
                gate_up_proj_weights[0]['bias'] = weights[f"{expert}.w1.bias"]
                gate_up_proj_weights[1]['bias'] = weights[f"{expert}.w3.bias"]
                down_proj_weights[0]['bias'] = weights[f"{expert}.w2.bias"]
                print("hello w1 bias:", gate_up_proj_weights[0]['bias'])
                print("hello w3 bias:", gate_up_proj_weights[1]['bias'])
                print("hello w2 bias:", down_proj_weights[0]['bias'])

            if self.quant_config and self.quant_config.quant_algo == QuantAlgo.FP8:
                gate_up_proj_weights[0]['weight_scale'] = weights[
                    f"{expert}.w1.weight_scale"]
                gate_up_proj_weights[1]['weight_scale'] = weights[
                    f"{expert}.w3.weight_scale"]
                down_proj_weights[0]['weight_scale'] = weights[
                    f"{expert}.w2.weight_scale"]
                gate_up_proj_weights[0]['input_scale'] = weights[
                    f"{expert}.w1.input_scale"]
                gate_up_proj_weights[1]['input_scale'] = weights[
                    f"{expert}.w3.input_scale"]
                down_proj_weights[0]['input_scale'] = weights[
                    f"{expert}.w2.input_scale"]
            elif self.quant_config and self.quant_config.quant_algo in (
                    QuantAlgo.NVFP4, QuantAlgo.W4A8_NVFP4_FP8):
                gate_up_proj_weights[0]['weight_scale'] = weights[
                    f"{expert}.w1.weight_scale"]
                gate_up_proj_weights[1]['weight_scale'] = weights[
                    f"{expert}.w3.weight_scale"]
                down_proj_weights[0]['weight_scale'] = weights[
                    f"{expert}.w2.weight_scale"]
                gate_up_proj_weights[0]['input_scale'] = weights[
                    f"{expert}.w1.input_scale"]
                gate_up_proj_weights[1]['input_scale'] = weights[
                    f"{expert}.w3.input_scale"]
                down_proj_weights[0]['input_scale'] = weights[
                    f"{expert}.w2.input_scale"]
                gate_up_proj_weights[0]['weight_scale_2'] = weights[
                    f"{expert}.w1.weight_scale_2"]
                gate_up_proj_weights[1]['weight_scale_2'] = weights[
                    f"{expert}.w3.weight_scale_2"]
                down_proj_weights[0]['weight_scale_2'] = weights[
                    f"{expert}.w2.weight_scale_2"]
                print("hello w1 weight_scale_2:",
                      gate_up_proj_weights[0]['weight_scale_2'])
                print("hello w3 weight_scale_2:",
                      gate_up_proj_weights[1]['weight_scale_2'])
                print("hello w2 weight_scale_2:",
                      down_proj_weights[0]['weight_scale_2'])
            elif (self.quant_config and self.quant_config.quant_algo
                  == QuantAlgo.FP8_BLOCK_SCALES):
                gate_up_proj_weights[0]["weight_scale"] = weights[
                    f"{expert}.w1.weight_scale"]
                gate_up_proj_weights[1]["weight_scale"] = weights[
                    f"{expert}.w3.weight_scale"]
                down_proj_weights[0]["weight_scale"] = weights[
                    f"{expert}.w2.weight_scale"]
            elif self.quant_config and self.quant_config.quant_algo == QuantAlgo.W4A8_MXFP4_MXFP8:
                gate_up_proj_weights[0]['weight_scale'] = weights[
                    f"{expert}.w1.weight_scale"]
                gate_up_proj_weights[1]['weight_scale'] = weights[
                    f"{expert}.w3.weight_scale"]
                down_proj_weights[0]['weight_scale'] = weights[
                    f"{expert}.w2.weight_scale"]

            self.experts[expert].gate_up_proj.load_weights(gate_up_proj_weights)
            self.experts[expert].down_proj.load_weights(down_proj_weights)


class MLPBlock(torch.nn.Module):

    def __init__(
        self,
        config: ModelConfig[GptOssConfig],
        layer_idx: int,
        reduce_results: bool = True,
        use_custom_cublas_mm: bool = False,
    ):
        super().__init__()

        self.config = config  # Store config as instance variable
        pretrained_config = config.pretrained_config
        self.num_experts = pretrained_config.num_local_experts
        moe_load_balancer_config = config.moe_load_balancer
        self.num_slots = moe_load_balancer_config.num_slots if moe_load_balancer_config and moe_load_balancer_config.num_slots else self.num_experts

        self.layer_idx = layer_idx
        self.enable_attention_dp = config.mapping.enable_attention_dp
        self.mapping = config.mapping

        self.norm = RMSNorm(hidden_size=pretrained_config.hidden_size,
                            eps=pretrained_config.rms_norm_eps,
                            dtype=pretrained_config.torch_dtype)

        self.gate = Linear(
            in_features=pretrained_config.hidden_size,
            out_features=pretrained_config.num_local_experts,
            bias=True,
            dtype=pretrained_config.torch_dtype,
            use_custom_cublas_mm=use_custom_cublas_mm,
        )

        self.routing_method = RenormalizeMoeRoutingMethod(
            top_k=pretrained_config.num_experts_per_tok,
            output_dtype=torch.bfloat16
            if config.moe_backend.upper() == "TRTLLM" else torch.float32)

        self.swiglu_alpha = torch.tensor(
            [1.702] * (self.num_slots // config.mapping.moe_ep_size),
            dtype=torch.float32).cuda()
        self.swiglu_beta = torch.tensor(
            [1.0] * (self.num_slots // config.mapping.moe_ep_size),
            dtype=torch.float32).cuda()
        self.swiglu_limit = torch.tensor(
            [7.0] * (self.num_slots // config.mapping.moe_ep_size),
            dtype=torch.float32).cuda()
        # Prepare MoE creation parameters
        moe_params = {
            'routing_method': self.routing_method,
            'num_experts': pretrained_config.num_local_experts,
            'hidden_size': pretrained_config.hidden_size,
            'intermediate_size': pretrained_config.intermediate_size,
            'dtype': pretrained_config.torch_dtype,
            'reduce_results': not self.enable_attention_dp and reduce_results,
            'model_config': config,
            'weight_loading_mode': MoEWeightLoadingMode.FUSED_GATE_UP_PROJ,
            'bias': True,
            'swiglu_alpha': self.swiglu_alpha,
            'swiglu_beta': self.swiglu_beta,
            'swiglu_limit': self.swiglu_limit
        }

        use_ref_fused_moe = os.environ.get('USE_REF_FUSED_MOE', '0') == '1'

        self.experts = create_moe(**moe_params)
        if not use_ref_fused_moe:
            print("Using original moe")
        else:
            print("Using reference moe")

            moe_params_ref = dict(moe_params)
            reduce_results_flag = moe_params_ref.pop('reduce_results')
            assert reduce_results_flag, (
                "RefGatedMLPFusedMoE only supports reduce_results=True. "
                "Set reduce_results to True before enabling USE_REF_FUSED_MOE.")

            ref_moe = RefGatedMLPFusedMoE(**moe_params_ref)
            # mark children so weight loader can skip descending into them
            setattr(ref_moe, "_ref_moe_parent", 1)
            for child in ref_moe.modules():
                setattr(child, "_ref_moe_parent", 1)
            self.experts_ref = ref_moe
            setattr(self.experts, "my_ref", ref_moe)

        # Perfect router caching - precompute common logits if enabled
        if os.environ.get('ENABLE_PERFECT_ROUTER', '0') == '1':
            precompute_common_perfect_router_logits(
                num_experts=pretrained_config.num_local_experts,
                experts_per_token=pretrained_config.num_experts_per_tok,
                moe_ep_size=config.mapping.moe_ep_size,
                dtype=pretrained_config.torch_dtype)

    @staticmethod
    def swiglu(x, alpha: float = 1.702):
        """
        This function is not really used in self.forward(), it's kept here for reference.
        It's implemented as part of the MoE kernels.
        """
        # Note we add an extra bias of 1 to the linear layer
        x_glu, x_linear = torch.chunk(x, 2, dim=-1)
        out_glu = x_glu * torch.sigmoid(alpha * x_glu)
        return out_glu * (x_linear + 1)

    def _create_ideal_expert_load_balanced_logits(
            self, num_tokens: int, num_experts: int,
            device: torch.device) -> torch.Tensor:
        """
        Create ideal logits that produce GPU-aware load balanced expert assignment.
         This method now uses the global cache to access precomputed logits to optimize performance.
        """
        pretrained_config = self.config.pretrained_config

        # Use global cached logits
        return get_cached_perfect_router_logits(
            num_tokens=num_tokens,
            num_experts=num_experts,
            experts_per_token=pretrained_config.experts_per_token,
            moe_ep_size=self.config.mapping.moe_ep_size,
            device=device,
            dtype=pretrained_config.torch_dtype)

    def compute_gate_output(self, x: torch.Tensor) -> torch.Tensor:
        if get_sm_version() in [
                90, 100, 103
        ] and x.shape[0] <= MIN_LATENCY_TINYGEMM_NUM_TOKENS:
            weight = self.gate.weight
            bias = self.gate.bias
            g = torch.ops.trtllm.tinygemm2(x, weight, bias)
        else:
            g = self.gate(x)
        return g

    def forward_normal(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        hidden_dim = orig_shape[-1]
        x = x.view(-1, hidden_dim)

        # t = self.norm(x) was done in the parent block
        t = x

        g = self.compute_gate_output(t)
        # Use ideal load balanced logits if enabled, otherwise use gate output
        if os.environ.get('ENABLE_PERFECT_ROUTER', '0') == '1':
            # WARNING: This discards the learned gate output and uses ideal logits for perfect load balancing
            # Only use this for testing load balancing strategies, not for actual inference
            num_tokens, num_experts = g.shape
            g = self._create_ideal_expert_load_balanced_logits(
                num_tokens=num_tokens, num_experts=num_experts, device=x.device)

        # When attention_dp is not enabled, don't pass those parameters
        if os.environ.get('USE_REF_FUSED_MOE', '0') == '1':
            #print("Using reference MoE forward")
            expert_output = self.experts_ref(hidden_states=t, router_logits=g)
            expert_output_gen = None  # self.experts(x=t, router_logits=g)
            #print("ref", expert_output)
            #print("gen", expert_output_gen)
        else:
            #print("Using original MoE forward")
            expert_output = self.experts(x=t, router_logits=g)
            #print("gen", expert_output)

        expert_output = expert_output.view(orig_shape)
        return expert_output, residual

    def forward_attn_dp(
        self,
        x: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        hidden_dim = orig_shape[-1]
        x = x.view(-1, hidden_dim)

        # t = self.norm(x) was done in the parent block
        t = x

        # Get attention_dp parameters
        all_rank_num_tokens = attn_metadata.all_rank_num_tokens

        if self.mapping.tp_size > 1 and all_rank_num_tokens is not None:
            if (isinstance(self.experts, (TritonFusedMoE))):
                t = allgather(t, self.mapping, dim=0, sizes=all_rank_num_tokens)

        g = self.compute_gate_output(t)
        # Use ideal load balanced logits if enabled, otherwise use gate output
        if os.environ.get('ENABLE_PERFECT_ROUTER', '0') == '1':
            # WARNING: This discards the learned gate output and uses ideal logits for perfect load balancing
            # Only use this for testing load balancing strategies, not for actual inference
            # The gate is still computed to maintain realistic performance measurement
            num_tokens, num_experts = g.shape
            g = self._create_ideal_expert_load_balanced_logits(
                num_tokens=num_tokens, num_experts=num_experts, device=x.device)

        # Let CutlassFusedMoE and TRTLLMGenFusedMoE handle allgather internally
        # Pass the normalized tensor (t) as input to experts, not x
        expert_output = self.experts(x=t,
                                     router_logits=g,
                                     all_rank_num_tokens=all_rank_num_tokens,
                                     use_dp_padding=False)

        expert_output = expert_output.view(orig_shape)
        return expert_output, residual

    def forward(
        self,
        x: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.enable_attention_dp:
            return self.forward_attn_dp(x, attn_metadata, residual)
        else:
            return self.forward_normal(x, residual)


class TransformerBlock(DecoderLayer):

    def __init__(
        self,
        config: ModelConfig[GptOssConfig],
        layer_idx: int,
        use_custom_cublas_mm: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        mapping = config.mapping
        self.enable_attn_dp = mapping.enable_attention_dp
        self.is_tp = mapping.has_tp() and not self.enable_attn_dp

        pretrained_config = config.pretrained_config
        self.input_layernorm = RMSNorm(
            hidden_size=pretrained_config.hidden_size,
            eps=pretrained_config.rms_norm_eps,
            dtype=pretrained_config.torch_dtype)

        self.attn = AttentionBlock(config, layer_idx, use_custom_cublas_mm)

        self.post_attention_layernorm = RMSNorm(
            hidden_size=pretrained_config.hidden_size,
            eps=pretrained_config.rms_norm_eps,
            dtype=pretrained_config.torch_dtype)

        self.mlp = MLPBlock(config,
                            layer_idx,
                            reduce_results=not self.is_tp,
                            use_custom_cublas_mm=use_custom_cublas_mm)

        self.mapping = config.mapping

        self.next_layer_layernorm = RMSNorm(
            hidden_size=pretrained_config.hidden_size,
            eps=pretrained_config.rms_norm_eps,
            dtype=pretrained_config.torch_dtype)

        # setup for tp
        self.allreduce = AllReduce(mapping=config.mapping,
                                   strategy=config.allreduce_strategy,
                                   dtype=config.pretrained_config.torch_dtype)

    def forward_normal(
        self,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = ...,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        x, residual = self.attn(position_ids,
                                hidden_states,
                                attn_metadata,
                                residual=residual,
                                **kwargs)
        x, residual = self.post_attention_layernorm(x, residual)

        x, residual = self.mlp(x, attn_metadata, residual)

        if spec_metadata is not None:
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, x,
                                                      residual)

        x, residual = self.next_layer_layernorm(x, residual)
        return x, residual

    def forward_tp(
        self,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = ...,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        x, residual = self.attn(
            position_ids,
            hidden_states,
            attn_metadata,
            residual=residual,
            all_reduce_params=AllReduceParams(enable_allreduce=False),
            **kwargs)

        x, residual = self.allreduce(
            x,
            all_reduce_params=AllReduceParams(
                fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                residual=residual,
                norm_weight=self.post_attention_layernorm.weight,
                eps=self.post_attention_layernorm.variance_epsilon,
                trigger_completion_at_end=False,
            ))

        x, residual = self.mlp(x, attn_metadata, residual)

        if spec_metadata is not None and spec_metadata.is_layer_capture(
                self.layer_idx):
            # In eagle3 mode, we capture the value in the boundary of decoder layer.
            # If fusing rms in the next layer, the value is not correct. Thus, if
            # this layer will be captured, we should not fuse the rms in the next
            # layer.
            x = self.allreduce(x,
                               all_reduce_params=AllReduceParams(
                                   fusion_op=AllReduceFusionOp.NONE,
                                   trigger_completion_at_end=False,
                               ))
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, x,
                                                      residual)
            x, residual = self.next_layer_layernorm(x, residual)
        else:
            x, residual = self.allreduce(
                x,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=self.next_layer_layernorm.weight,
                    eps=self.next_layer_layernorm.variance_epsilon,
                    trigger_completion_at_end=False,
                ))

        return x, residual

    def forward(
        self,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = ...,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.is_tp:
            _forward = self.forward_tp
        else:
            _forward = self.forward_normal
        return _forward(position_ids,
                        hidden_states,
                        attn_metadata,
                        residual,
                        spec_metadata=spec_metadata,
                        **kwargs)


class Transformer(DecoderModel):

    def __init__(self, model_config: ModelConfig[GptOssConfig]):
        super().__init__(model_config)
        config = self.model_config

        # Triton MoE kernels require installing Triton main branch,
        # which may be incompatible with torch.compile due to version mismatch.
        enable_torch_compile_for_embedding = model_config.moe_backend != "TRITON"

        # Use custom cublas since we need LUT to tune the perf.
        prop = torch.cuda.get_device_properties(0)
        sm_version = prop.major * 10 + prop.minor
        self.use_custom_cublas_mm = sm_version == 121

        if model_config.mapping.enable_attention_dp:
            # When attention_dp is enabled, we cannot do all_reduce since
            # the problem size of different ranks are different.
            # So, we don't do parallelism here.
            self.embedding = Embedding(
                config.pretrained_config.vocab_size,
                config.pretrained_config.hidden_size,
                dtype=config.pretrained_config.torch_dtype,
                enable_torch_compile_for_embedding=
                enable_torch_compile_for_embedding)
        else:
            self.embedding = Embedding(
                config.pretrained_config.vocab_size,
                config.pretrained_config.hidden_size,
                dtype=config.pretrained_config.torch_dtype,
                mapping=config.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                enable_torch_compile_for_embedding=
                enable_torch_compile_for_embedding,
                use_custom_cublas_mm=self.use_custom_cublas_mm,
            )
        # For modeling_speculative, different name expected
        self.embed_tokens = self.embedding
        self.block = nn.ModuleList([
            TransformerBlock(
                model_config,
                layer_idx,
                use_custom_cublas_mm=self.use_custom_cublas_mm,
            ) for layer_idx in range(config.pretrained_config.num_hidden_layers)
        ])
        self.norm = RMSNorm(
            hidden_size=config.pretrained_config.hidden_size,
            eps=config.pretrained_config.rms_norm_eps,
            dtype=config.pretrained_config.torch_dtype,
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        hidden_states = inputs_embeds or self.embedding(input_ids)

        residual = None
        for block in self.block:
            hidden_states, residual = block(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
            )

        return hidden_states


@register_auto_model("GptOssForCausalLM")
class GptOssForCausalLM(SpecDecOneEngineForCausalLM[Transformer, GptOssConfig]):

    params_map = {
        # TRTLLM module name : GptOss module name
        "qkv_proj": "qkv",
        "o_proj": "out",
        "lm_head": "unembedding",
    }

    hf_params_map = {
        # TRTLLM module name : HuggingFace module name
        "embedding": "embed_tokens",
        # Order matters for attn.norm and attn.
        'attn.norm': 'input_layernorm',
        'attn': 'self_attn',
        'mlp.norm': 'post_attention_layernorm',
        'block': 'layers',
        'gate': 'router',
    }

    def __init__(
        self,
        model_config: ModelConfig[GptOssConfig],
    ):
        # Map config to HF format.
        if hasattr(model_config.pretrained_config, 'num_experts'):
            model_config.pretrained_config.num_local_experts = model_config.pretrained_config.num_experts
            model_config.pretrained_config.num_experts_per_tok = model_config.pretrained_config.experts_per_token
            model_config.pretrained_config.rope_scaling = {
                'factor':
                model_config.pretrained_config.rope_scaling_factor,
                'beta_fast':
                model_config.pretrained_config.rope_ntk_beta,
                'beta_slow':
                model_config.pretrained_config.rope_ntk_alpha,
                'original_max_position_embeddings':
                model_config.pretrained_config.initial_context_length,
            }
        if model_config.pretrained_config.torch_dtype is None:
            model_config.pretrained_config.torch_dtype = torch.bfloat16

        assert model_config.mapping.pp_size == 1, "Pipeline parallelism is not supported."

        super().__init__(
            Transformer(model_config),
            model_config=model_config,
        )

    def __post_init__(self):
        # Do not call super().__post_init__()
        params_map_reverse = {v: k for k, v in self.params_map.items()}

        quant_config = self.model_config.quant_config
        if quant_config.exclude_modules:
            for i, module in enumerate(quant_config.exclude_modules):
                names = module.split(".")
                if names[-1] in params_map_reverse:
                    names[-1] = params_map_reverse[names[-1]]
                prefix = [] if names[0] == "model" else ["model"]
                quant_config.exclude_modules[i] = '.'.join(prefix + names)

        super().apply_quant_config_exclude_modules()

        for _, module in self.named_modules():
            if callable(getattr(module, "create_weights", None)):
                module.create_weights()

    def load_weights(self, weights: Dict):
        is_ori_model = True
        for k, v in weights.items():
            if 'q_proj' in k:
                is_ori_model = False

        is_nvfp4 = self.model_config.quant_config.quant_mode.has_nvfp4()

        if is_ori_model:
            self.load_ori_weights(weights)
        elif is_nvfp4:
            self.load_nvfp4_weights(weights)
        else:
            self.load_hf_weights(weights)

    def post_load_weights(self):
        for idx, layer in enumerate(
                self.model.block[:self.config.num_hidden_layers]):
            if idx == 0:
                layer.input_layernorm = layer.attn.norm

            layer.post_attention_layernorm = layer.mlp.norm

            if idx == self.config.num_hidden_layers - 1:
                layer.next_layer_layernorm = self.model.norm
            else:
                layer.next_layer_layernorm = self.model.block[idx + 1].attn.norm

    def load_hf_weights(self, weights: Dict):
        num_expert = self.config.num_local_experts

        for name, module in tqdm(list(self.named_modules()),
                                 desc="Loading weights"):
            if len(module._parameters) <= 0 or name.startswith("draft_model"):
                continue

            module_weights = {}
            for k, v in self.hf_params_map.items():
                name = name.replace(k, v)
            module_weights = filter_weights(name, weights)

            if isinstance(module, MoE):
                try:
                    # For BF16 ckpt.
                    # Deinterleave for gate and up.
                    gate_up_weight = module_weights['gate_up_proj']
                    gate, up = gate_up_weight[:, :, ::2], gate_up_weight[:, :,
                                                                         1::2]
                    gate_up_weight = torch.cat([gate, up], dim=-1)
                    gate_up_bias = module_weights['gate_up_proj_bias']
                    gate, up = gate_up_bias[:, ::2], gate_up_bias[:, 1::2]
                    gate_up_bias = torch.cat([gate, up], dim=-1)
                    moe_weights = {
                        'gate_up_proj': [
                            gate_up_weight.to(self.model.dtype)[i, :, :]
                            for i in range(num_expert)
                        ],
                        'down_proj': [
                            module_weights['down_proj'][i, :, :].to(
                                self.model.dtype) for i in range(num_expert)
                        ],
                        'gate_up_proj.bias':
                        [gate_up_bias[i, :] for i in range(num_expert)],
                        'down_proj.bias': [
                            module_weights['down_proj_bias'][i, :]
                            for i in range(num_expert)
                        ]
                    }
                except:
                    # For MXFP4 ckpt.
                    # Deinterleave for gate and up.
                    gate_up_weight = module_weights[
                        'gate_up_proj_blocks'].flatten(-2, -1)
                    gate_weight, up_weight = gate_up_weight[:, ::
                                                            2, :], gate_up_weight[:,
                                                                                  1::
                                                                                  2, :]
                    gate_up_weight = torch.cat([gate_weight, up_weight], dim=-2)
                    gate_up_bias = module_weights['gate_up_proj_bias']
                    gate_bias, up_bias = gate_up_bias[:, ::
                                                      2], gate_up_bias[:, 1::2]
                    gate_up_bias = torch.cat([gate_bias, up_bias], dim=-1)
                    gate_up_weight_scale = module_weights['gate_up_proj_scales']
                    gate_weight_scale, up_weight_scale = gate_up_weight_scale[:, ::
                                                                              2, :], gate_up_weight_scale[:,
                                                                                                          1::
                                                                                                          2, :]
                    gate_up_weight_scale = torch.cat(
                        [gate_weight_scale, up_weight_scale], dim=-2)
                    moe_weights = {
                        'gate_up_proj': [
                            gate_up_weight[i, :, :].transpose(0, 1)
                            for i in range(num_expert)
                        ],
                        'down_proj': [
                            module_weights['down_proj_blocks'].flatten(
                                -2, -1)[i, :, :].transpose(0, 1)
                            for i in range(num_expert)
                        ],
                        'gate_up_proj.bias':
                        [gate_up_bias[i, :] for i in range(num_expert)],
                        'down_proj.bias': [
                            module_weights['down_proj_bias'][i, :]
                            for i in range(num_expert)
                        ],
                        'gate_up_proj_weight_scale': [
                            gate_up_weight_scale[i, :, :].transpose(0, 1)
                            for i in range(num_expert)
                        ],
                        'down_proj_weight_scale': [
                            module_weights['down_proj_scales']
                            [i, :, :].transpose(0, 1) for i in range(num_expert)
                        ]
                    }

                    if self.model_config.quant_config.quant_algo == 'W4A16_MXFP4':
                        for i in range(num_expert):
                            moe_weights[
                                f"{i}.w1.weight_scale_inv"] = gate_weight_scale[
                                    i, :, :]
                            moe_weights[
                                f"{i}.w3.weight_scale_inv"] = up_weight_scale[
                                    i, :, :]
                            moe_weights[
                                f"{i}.w2.weight_scale_inv"] = module_weights[
                                    'down_proj_scales'][i, :, :]

                module.load_weights(weights=[moe_weights])
            elif hasattr(module, "load_weights"):
                if 'qkv' in name:
                    # For qkv_proj
                    q_weight_bias = filter_weights(
                        name.replace('qkv_proj', 'q_proj'), weights)
                    k_weight_bias = filter_weights(
                        name.replace('qkv_proj', 'k_proj'), weights)
                    v_weight_bias = filter_weights(
                        name.replace('qkv_proj', 'v_proj'), weights)
                    module.load_weights(
                        weights=[q_weight_bias, k_weight_bias, v_weight_bias])
                else:
                    # For o_proj, sinks.
                    module.load_weights(weights=[module_weights])
            else:
                # Load four LN weights (attn.norm, mlp.norm, input_layernorm, post_attention_layernorm).
                if 'next_layer_layernorm' in name:
                    continue

                for n, p in module._parameters.items():
                    if p is not None:
                        p.data.copy_(module_weights[n][:])

    def load_nvfp4_weights(self, weights: Dict):
        num_expert = self.config.num_local_experts

        for name, module in tqdm(list(self.named_modules()),
                                 desc="Loading weights"):
            if len(module._parameters) <= 0 or name.startswith("draft_model"):
                continue

            module_weights = {}
            for k, v in self.hf_params_map.items():
                name = name.replace(k, v)
            module_weights = filter_weights(name, weights)

            if isinstance(module, MoE):
                print(f"Loading NVFP4 MoE weights for module: {name}")
                assert getattr(module, "quant_config", None) is not None and \
                   module.quant_config.quant_mode.has_nvfp4()
                gate_up = module_weights.get('gate_up_proj', None)
                down = module_weights.get('down_proj', None)
                gate_up_bias = module_weights.get('gate_up_proj_bias', None)
                down_bias = module_weights.get('down_proj_bias', None)

                # Only fp32 bias is supported for NVFP4 MoE.
                if gate_up_bias.dtype != torch.float32:
                    gate_up_bias = gate_up_bias.to(torch.float32)
                if down_bias.dtype != torch.float32:
                    down_bias = down_bias.to(torch.float32)

                moe_weights = {}
                if gate_up is not None:
                    moe_weights['gate_up_proj'] = [
                        gate_up[i, :, :] for i in range(num_expert)
                    ]
                if down is not None:
                    moe_weights['down_proj'] = [
                        down[i, :, :] for i in range(num_expert)
                    ]
                if gate_up_bias is not None:
                    moe_weights['gate_up_proj.bias'] = [
                        gate_up_bias[i, :] for i in range(num_expert)
                    ]
                if down_bias is not None:
                    moe_weights['down_proj.bias'] = [
                        down_bias[i, :] for i in range(num_expert)
                    ]

                # Per-expert block scales (transpose to expected layout)
                if 'gate_up_proj_weight_scale' in module_weights:
                    gu_ws = module_weights['gate_up_proj_weight_scale']
                    moe_weights['gate_up_proj_weight_scale'] = [
                        gu_ws[i, :, :] for i in range(num_expert)
                    ]
                if 'down_proj_weight_scale' in module_weights:
                    dp_ws = module_weights['down_proj_weight_scale']
                    moe_weights['down_proj_weight_scale'] = [
                        dp_ws[i, :, :] for i in range(num_expert)
                    ]

                # Module-level globals for NVFP4 loaders
                for src_key in [
                        'gate_up_proj_weight_scale_2',
                        'down_proj_weight_scale_2',
                        'gate_up_proj_input_scale',
                        'down_proj_input_scale',
                ]:
                    if src_key in module_weights:
                        moe_weights[src_key] = module_weights[src_key]

                print(
                    "also give the weight to ref moe if exists but don't give to ref moe directly"
                )
                if os.environ.get('USE_REF_FUSED_MOE', '0') == '1':
                    module.my_ref.load_weights(weights=[moe_weights])
                else:
                    module.load_weights(weights=[moe_weights])
            elif _is_ref_moe_module(module):
                continue
            elif hasattr(module, "load_weights"):
                if 'qkv' in name:
                    # For qkv_proj
                    q_weight_bias = filter_weights(
                        name.replace('qkv_proj', 'q_proj'), weights)
                    k_weight_bias = filter_weights(
                        name.replace('qkv_proj', 'k_proj'), weights)
                    v_weight_bias = filter_weights(
                        name.replace('qkv_proj', 'v_proj'), weights)
                    module.load_weights(
                        weights=[q_weight_bias, k_weight_bias, v_weight_bias])
                else:
                    # For o_proj, sinks.
                    module.load_weights(weights=[module_weights])
            else:
                # Load four LN weights (attn.norm, mlp.norm, input_layernorm, post_attention_layernorm).
                if 'next_layer_layernorm' in name:
                    continue

                for n, p in module._parameters.items():
                    if p is not None:
                        p.data.copy_(module_weights[n][:])

    def load_ori_weights(self, weights: Dict):
        head_dim = self.config.head_dim
        num_q_head = self.config.num_attention_heads
        num_kv_head = self.config.num_key_value_heads
        num_expert = self.config.num_local_experts
        enable_attention_dp = self.model_config.mapping.enable_attention_dp
        tp_size = self.model_config.mapping.tp_size

        for name, module in tqdm(list(self.named_modules()),
                                 desc="Loading weights"):
            if len(module._parameters) <= 0 or name.startswith("draft_model"):
                continue
            names = name.split(".")
            module_weights = {}
            if names[-1] in self.params_map:
                names[-1] = self.params_map[names[-1]]

            # Drop the first "model" prefix
            if names[0] == 'model':
                name = '.'.join(names[1:])
            else:
                name = '.'.join(names)
            module_weights = filter_weights(name, weights)
            if isinstance(module, MoE):
                # [num_experts, intermediate_size * 2, hidden_size]
                gate_up_proj = filter_weights(name.replace("experts", "mlp1"),
                                              weights)
                # [num_experts, intermediate_size, hidden_size]
                down_proj = filter_weights(name.replace("experts", "mlp2"),
                                           weights)
                try:
                    # Official MXFP4 ckpt.
                    gate_up_weight = gate_up_proj['weight.blocks'].flatten(
                        -2, -1)
                    gate, up = gate_up_weight[:, ::2, :], gate_up_weight[:, 1::
                                                                         2, :]
                    gate_up_weight = torch.cat([gate, up], dim=-2)
                    gate_up_bias = gate_up_proj['bias']
                    gate, up = gate_up_bias[:, ::2], gate_up_bias[:, 1::2]
                    gate_up_bias = torch.cat([gate, up], dim=-1)
                    moe_weights = {
                        'gate_up_proj': [
                            gate_up_weight[i, :, :].transpose(0, 1)
                            for i in range(num_expert)
                        ],
                        'down_proj': [
                            down_proj['weight.blocks'].flatten(
                                -2, -1)[i, :, :].transpose(0, 1)
                            for i in range(num_expert)
                        ],
                        'gate_up_proj.bias':
                        [gate_up_bias[i, :] for i in range(num_expert)],
                        'down_proj.bias':
                        [down_proj['bias'][i, :] for i in range(num_expert)]
                    }
                except:
                    # For BF16 ckpt.
                    moe_weights = {
                        'gate_up_proj': [
                            gate_up_proj['weight'][i, :, :].transpose(0, 1).to(
                                self.model.dtype) for i in range(num_expert)
                        ],
                        'down_proj': [
                            down_proj['weight'][i, :, :].transpose(0, 1).to(
                                self.model.dtype) for i in range(num_expert)
                        ],
                        'gate_up_proj.bias':
                        [gate_up_proj['bias'][i, :] for i in range(num_expert)],
                        'down_proj.bias':
                        [down_proj['bias'][i, :] for i in range(num_expert)]
                    }
                # Only for Official MXFP4 ckpt.
                if 'weight.scales' in gate_up_proj:
                    gate_up_weight_scale = gate_up_proj['weight.scales']
                    gate, up = gate_up_weight_scale[:, ::
                                                    2, :], gate_up_weight_scale[:,
                                                                                1::
                                                                                2, :]
                    gate_up_weight_scale = torch.cat([gate, up], dim=-2)
                    moe_weights['gate_up_proj_weight_scale'] = [
                        gate_up_weight_scale[i, :, :].transpose(0, 1)
                        for i in range(num_expert)
                    ]

                    if self.model_config.quant_config.quant_algo == 'W4A16_MXFP4':
                        for i in range(num_expert):
                            moe_weights[f"{i}.w1.weight_scale_inv"] = gate[
                                i, :, :]
                            moe_weights[f"{i}.w3.weight_scale_inv"] = up[
                                i, :, :]

                if 'weight.scales' in down_proj:
                    moe_weights['down_proj_weight_scale'] = [
                        down_proj['weight.scales'][i, :, :].transpose(0, 1)
                        for i in range(num_expert)
                    ]

                    if self.model_config.quant_config.quant_algo == 'W4A16_MXFP4':
                        for i in range(num_expert):
                            moe_weights[f"{i}.w2.weight_scale_inv"] = down_proj[
                                'weight.scales'][i, :, :]

                module.load_weights(weights=[moe_weights])
            elif hasattr(module, "load_weights"):
                # Load Attention module weights.
                if 'qkv' in name:
                    q_weight = module_weights['weight'][:head_dim *
                                                        num_q_head, :]
                    k_weight = module_weights['weight'][head_dim *
                                                        num_q_head:head_dim *
                                                        (num_q_head +
                                                         num_kv_head), :]
                    v_weight = module_weights['weight'][-head_dim *
                                                        num_kv_head:, :]
                    q_bias = module_weights['bias'][:head_dim * num_q_head]
                    k_bias = module_weights['bias'][head_dim *
                                                    num_q_head:head_dim *
                                                    (num_q_head + num_kv_head)]
                    v_bias = module_weights['bias'][-head_dim * num_kv_head:]

                    # Handle KV weight duplication for GQA
                    tensors_need_duplication = ['weight', 'bias']
                    if module.quant_config.quant_mode.has_mxfp4():
                        tensors_need_duplication.append('weight_scale')

                    # Duplicate KV weights if needed
                    tensor_parallel_size = tp_size if not enable_attention_dp else 1

                    k_weight_dict = {'weight': k_weight, 'bias': k_bias}
                    v_weight_dict = {'weight': v_weight, 'bias': v_bias}

                    if 'weight_scale' in module_weights:
                        k_weight_dict['weight_scale'] = module_weights[
                            'weight_scale'][head_dim * num_q_head:head_dim *
                                            (num_q_head + num_kv_head), :]
                        v_weight_dict['weight_scale'] = module_weights[
                            'weight_scale'][-head_dim * num_kv_head:, :]

                    k_weight_dict = {
                        k: (duplicate_kv_weight(
                            weight=v,
                            num_kv_heads=num_kv_head,
                            tensor_parallel_size=tensor_parallel_size)
                            if k in tensors_need_duplication else v)
                        for k, v in k_weight_dict.items()
                    }

                    v_weight_dict = {
                        k: (duplicate_kv_weight(
                            weight=v,
                            num_kv_heads=num_kv_head,
                            tensor_parallel_size=tensor_parallel_size)
                            if k in tensors_need_duplication else v)
                        for k, v in v_weight_dict.items()
                    }

                    qkv_weights = [{
                        'weight': q_weight,
                        'bias': q_bias
                    }, k_weight_dict, v_weight_dict]
                    module.load_weights(weights=qkv_weights)
                else:
                    # Dense & gate & sinks
                    module.load_weights(weights=[module_weights])
            else:
                # Load LN weights.
                if names[-1].endswith("layernorm") and names[-3] == "block":
                    # skip loading weights for the fused norms
                    continue
                for n, p in module._parameters.items():
                    if p is not None:
                        p.data.copy_(module_weights[n.replace(
                            "weight", "scale")][:])
