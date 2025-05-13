# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

##############################################################################
import json
import time
from pathlib import Path

import triton.profiler as proton

class ProfilerTriton:

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        log_dir = Path("/tmp/bench")
        log_dir.mkdir(parents=True, exist_ok=True)
        self.hatchet = log_dir / "bench.hatchet"
        proton.start(str(self.hatchet.with_suffix('')), hook="triton")
        self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_value, traceback):
        print("=" * 60)
        proton.finalize()
        with open(self.hatchet) as f:
            data = json.load(f)
        root = next((d for d in data if 'children' in d), None)
        if root is None:
            print(f"[{self.name}] No profiling data found.")
            return
        timings = []

        def collect(nodes):
            for node in nodes:
                frame = node['frame']['name']
                time_ns = node['metrics'].get('time (ns)', 0)
                if time_ns > 0:
                    timings.append((frame, time_ns))
                if node.get('children'):
                    collect(node['children'])

        collect(root['children'])
        timings.sort(key=lambda x: x[1], reverse=True)
        total_ns = sum(t for _, t in timings)
        print(f"[{self.name}] CUDA kernel timings (largest first):")
        for name, t_ns in timings:
            print(f"  {t_ns / 1e6:.3f} ms : {name}")
        print(f"[{self.name}] Total CUDA time: {total_ns / 1e6:.3f} ms")
        end_time = time.perf_counter()
        elapsed = end_time - self.start_time
        print(f"[{self.name}] Python native wall time: {elapsed * 1e3:.3f} ms")
        print("=" * 60)


def printTorchTensorInfo(tensor, name):
    print(f"{name}: {tensor.shape} {tensor.dtype} {tensor.device}")

##############################################################################
import torch
import torch.nn.functional as F
import tensorrt_llm
_ = tensorrt_llm # noqa: F401


class moe_args:

    def __init__(self, num_tokens, num_experts, hidden_size, intermediate_size,
                 top_k, padding, hidden_states, hidden_states_scale,
                 hidden_states_scale_global, expert_logits, gemm1_weights,
                 gemm1_scales, gemm1_scales_global, gemm2_weights, gemm2_scales,
                 gemm2_scales_global, permute_info,
                 use_routing_scales_on_input):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        self.padding = padding
        self.hidden_states = hidden_states
        self.hidden_states_scale = hidden_states_scale
        self.hidden_states_scale_global = hidden_states_scale_global
        self.expert_logits = expert_logits
        self.gemm1_weights = gemm1_weights
        self.gemm1_scales = gemm1_scales
        self.gemm1_scales_global = gemm1_scales_global
        self.gemm2_weights = gemm2_weights
        self.gemm2_scales = gemm2_scales
        self.gemm2_scales_global = gemm2_scales_global
        self.permute_info = permute_info
        self.use_routing_scales_on_input = use_routing_scales_on_input


class moe_args_dequant:

    def __init__(self, num_tokens, num_experts, hidden_size, intermediate_size,
                 top_k, padding, hidden_states, expert_logits, gemm1_weights,
                 gemm2_weights, permute_info, use_routing_scales_on_input):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        self.padding = padding
        self.hidden_states = hidden_states
        self.expert_logits = expert_logits
        self.gemm1_weights = gemm1_weights
        self.gemm2_weights = gemm2_weights
        self.permute_info = permute_info
        self.use_routing_scales_on_input = use_routing_scales_on_input


def routing_reference(expertLogits, topK, padding):
    originalDevice = expertLogits.device
    expertLogits = expertLogits.cpu()
    numTokens, numExperts = expertLogits.shape
    assert topK <= numExperts

    numTokensPerExpert = torch.zeros(numExperts, dtype=torch.int64)
    expandedTokenIdxToExpert = -torch.ones(numTokens * topK, dtype=torch.int64)
    expandedTokenIdxToIdxInExpert = -torch.ones(numTokens * topK,
                                                dtype=torch.int64)

    topKLogits, topKIndices = torch.topk(expertLogits, topK, dim=1)
    for tokenIdx in range(numTokens):
        for k in range(topK):
            expandedIdx = tokenIdx * topK + k
            expertIndex = topKIndices[tokenIdx, k]
            expandedTokenIdxToExpert[expandedIdx] = expertIndex
            expandedTokenIdxToIdxInExpert[expandedIdx] = numTokensPerExpert[
                expertIndex]
            numTokensPerExpert[expertIndex] += 1

    paddedTokensPerExpertPrefixSum = torch.zeros(numExperts + 1,
                                                 dtype=torch.int64)
    for ii in range(numExperts):

        def divUpMul(a, b):
            return (a + b - 1) // b * b

        paddedTokensPerExpertPrefixSum[
            ii + 1] = paddedTokensPerExpertPrefixSum[ii] + divUpMul(
                numTokensPerExpert[ii], padding)
    permutedBufferSize = paddedTokensPerExpertPrefixSum[numExperts]

    expandedTokenIdxToPermutedIdx = -torch.ones(numTokens * topK,
                                                dtype=torch.int64)
    permutedIdxToExpandedIdx = -torch.ones(permutedBufferSize,
                                           dtype=torch.int64)
    permutedIdxToTokenIdx = -torch.ones(permutedBufferSize, dtype=torch.int64)
    for tokenIdx in range(numTokens):
        for k in range(topK):
            expandedIdx = tokenIdx * topK + k
            expert = expandedTokenIdxToExpert[expandedIdx]
            offsetWithinExpert = expandedTokenIdxToIdxInExpert[expandedIdx]
            offsetForExpert = paddedTokensPerExpertPrefixSum[expert]
            permutedIdx = offsetForExpert + offsetWithinExpert

            expandedTokenIdxToPermutedIdx[expandedIdx] = permutedIdx
            permutedIdxToExpandedIdx[permutedIdx] = expandedIdx
            permutedIdxToTokenIdx[permutedIdx] = tokenIdx
    return {
        "paddedTokensPerExpertPrefixSum":
        paddedTokensPerExpertPrefixSum.to(originalDevice),
        "permutedBufferSize":
        permutedBufferSize.item(),
        "expandedTokenIdxToPermutedIdx":
        expandedTokenIdxToPermutedIdx.to(originalDevice),
        "permutedIdxToExpandedIdx":
        permutedIdxToExpandedIdx.to(originalDevice),
        "numTokensPerExpert":
        numTokensPerExpert.to(originalDevice),
        "expandedTokenIdxToExpert":
        expandedTokenIdxToExpert.to(originalDevice),
        "topKLogits":
        topKLogits.to(originalDevice),
        "permutedIdxToTokenIdx":
        permutedIdxToTokenIdx.to(originalDevice),
        "topKIndices":
        topKIndices.to(originalDevice)
    }


def noaux_tc_ref(logits, bias, n_group, topk_group, top_k,
                 routed_scaling_factor):
    scores = F.sigmoid(logits)
    scores_with_bias = scores + bias
    scores_shape = list(scores_with_bias.shape)
    group_scores = torch.sum(torch.topk(
        scores_with_bias.view(scores_shape[:-1] +
                              [n_group, scores_shape[-1] // n_group]),
        k=2,
        dim=-1,
        largest=True,
        sorted=True)[0],
                             dim=-1)
    _, group_idx = torch.topk(group_scores,
                              k=topk_group,
                              dim=-1,
                              largest=True,
                              sorted=True)
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(-1, group_idx, 1)
    score_mask = group_mask.unsqueeze(
        -1).expand(scores_shape[:-1] +
                   [n_group, scores_shape[-1] // n_group]).reshape(scores_shape)
    scores_with_bias = scores_with_bias * score_mask
    _, topk_idx = torch.topk(scores_with_bias,
                             k=top_k,
                             dim=-1,
                             largest=True,
                             sorted=True)
    new_mask = torch.zeros_like(scores)
    new_mask.scatter_(-1, topk_idx, 1)
    scores = scores * new_mask
    score_sum = torch.sum(scores, dim=-1, keepdim=True) + 1e-20
    scores = scores / score_sum * routed_scaling_factor
    return scores


def routing_reference_no_aux(expert_logits,
                             routing_bias,
                             top_k,
                             n_groups,
                             top_k_groups,
                             routed_scaling,
                             padding,
                             use_routing_scales_on_input=False):
    routing_logits = expert_logits.to(dtype=torch.float, device='cuda')
    if use_routing_scales_on_input:
        # if using routing scales on input, topK == 1 and the score is a plain sigmoid
        scores = F.sigmoid(routing_logits)
    else:
        scores = noaux_tc_ref(routing_logits, routing_bias, n_groups,
                              top_k_groups, top_k, routed_scaling)
    permute_info = routing_reference(scores, top_k, padding)
    # print("permute_info: ", permute_info)
    return permute_info, scores


def dequant_reference_dsfp8(input, scale, transpose_scale, block_m, block_n):
    input = input.to(torch.float)
    scale = scale.to(torch.float)
    if transpose_scale:
        scale = scale.t()
    output = torch.zeros_like(input)
    m, n = input.shape
    m_tile = 128 if block_m else 1
    n_tile = 128 if block_n else 1
    assert m % m_tile == 0
    assert n % n_tile == 0
    assert scale.shape == (m // m_tile, n // n_tile)
    if m_tile == 1:
        for j in range(0, n, n_tile):
            output[:, j:j +
                   n_tile] = input[:, j:j + n_tile] * scale[:, j //
                                                            n_tile][:, None]
    elif n_tile == 1:
        for i in range(0, m, m_tile):
            output[i:i + m_tile] = input[i:i + m_tile] * scale[i // m_tile]
    else:
        for i in range(0, m, m_tile):
            for j in range(0, n, n_tile):
                output[i:i + m_tile, j:j +
                       n_tile] = input[i:i + m_tile, j:j +
                                       n_tile] * scale[i // m_tile, j // n_tile]
    return output


def run_moe_dequant(args, quant_mode=["fp4", "dsFp8", "perTensorFp8"]):
    # Permute
    total_num_padded_tokens = args.permute_info["permutedBufferSize"]
    expanded_idx_to_permuted_idx = args.permute_info[
        "expandedTokenIdxToPermutedIdx"].cpu()
    num_tokens_per_expert = args.permute_info["numTokensPerExpert"].cpu()
    permute_output = torch.full((total_num_padded_tokens, args.hidden_size),
                                float('nan'),
                                device='cuda').to(torch.float)
    for i in range(args.num_tokens):
        for j in range(args.top_k):
            permuted_idx = expanded_idx_to_permuted_idx[i * args.top_k + j]
            permute_output[permuted_idx] = args.hidden_states[i]
    # Gemm1
    gemm1_output = torch.full(
        (total_num_padded_tokens, 2 * args.intermediate_size),
        float('nan'),
        device='cuda').to(torch.float)
    i = 0
    for expert_idx in range(args.num_experts):
        my_num_tokens = num_tokens_per_expert[expert_idx]
        if my_num_tokens == 0:
            continue
        my_a = permute_output[i:i + my_num_tokens]
        my_b = args.gemm1_weights[expert_idx]
        my_c = my_a @ my_b.t()
        gemm1_output[i:i + my_num_tokens] = my_c
        i += my_num_tokens
        i = (i + args.padding - 1) // args.padding * args.padding

    if args.use_routing_scales_on_input:
        assert args.top_k == 1
        # For each token and its top_k experts
        for token_idx in range(args.num_tokens):
            for k in range(args.top_k):
                # Get the permuted index for this token's k-th expert
                expanded_idx = token_idx * args.top_k + k
                permuted_idx = expanded_idx_to_permuted_idx[expanded_idx]
                expert_weight = args.permute_info["topKLogits"].to(torch.float)
                # Get the expert weight for this token and expert
                weight = expert_weight[token_idx, k]
                # Scale the corresponding row in gemm1_output
                gemm1_output[permuted_idx] *= weight

    # Activation
    activation_output = torch.full(
        (total_num_padded_tokens, args.intermediate_size),
        float('nan'),
        device='cuda').to(torch.float)

    i = 0
    for expert_idx in range(args.num_experts):
        my_num_tokens = num_tokens_per_expert[expert_idx]
        if my_num_tokens == 0:
            continue
        my_a = gemm1_output[i:i + my_num_tokens]
        my_x1 = my_a[:, :args.intermediate_size]
        my_x2 = my_a[:, args.intermediate_size:]
        activation_output[i:i + my_num_tokens] = F.silu(my_x2) * my_x1
        i += my_num_tokens
        i = (i + args.padding - 1) // args.padding * args.padding

    if quant_mode == "fp4":
        activation_output, c_global_sf = quant_dequant_fp4(
            activation_output.to(torch.bfloat16), False, True)
        activation_output = activation_output.to(torch.float)
        args.c_global_sf = c_global_sf
    elif quant_mode == "perTensorFp8":
        activation_output, c_global_sf = quant_dequant_per_tensor_fp8(
            activation_output.to(torch.bfloat16))
        activation_output = activation_output.to(torch.float)
        args.c_global_sf = c_global_sf

    # Gemm2
    gemm2_output = torch.full((total_num_padded_tokens, args.hidden_size),
                              float('nan'),
                              device='cuda').to(torch.float)
    i = 0
    for expert_idx in range(args.num_experts):
        my_num_tokens = num_tokens_per_expert[expert_idx]
        if my_num_tokens == 0:
            continue
        my_a = activation_output[i:i + my_num_tokens]
        my_b = args.gemm2_weights[expert_idx]
        my_c = my_a @ my_b.t()
        gemm2_output[i:i + my_num_tokens] = my_c
        i += my_num_tokens
        i = (i + args.padding - 1) // args.padding * args.padding
    # Finalize
    expert_weight = args.permute_info["topKLogits"].to(torch.float)
    finalize_output = torch.full((args.num_tokens, args.hidden_size),
                                 float('nan'),
                                 device='cuda').to(torch.float)
    for i in range(args.num_tokens):
        acc = torch.zeros(args.hidden_size, dtype=torch.float, device='cuda')
        for top_k_idx in range(args.top_k):
            expanded_idx = i * args.top_k + top_k_idx
            permuted_idx = expanded_idx_to_permuted_idx[expanded_idx]
            original_vector = gemm2_output[permuted_idx]
            weight = expert_weight[
                i, top_k_idx] if not args.use_routing_scales_on_input else 1.0
            acc += original_vector * weight
        finalize_output[i] = acc
    return finalize_output


def run_moe_reference_dsfp8(args):
    hidden_states_dequant = dequant_reference_dsfp8(args.hidden_states,
                                                    args.hidden_states_scale,
                                                    True, False, True)

    gemm1_weights_dequant = {}
    for i in range(args.num_experts):
        gemm1_weights_dequant[i] = dequant_reference_dsfp8(
            args.gemm1_weights[i], args.gemm1_scales[i], False, True, True)

    gemm2_weights_dequant = {}
    for i in range(args.num_experts):
        gemm2_weights_dequant[i] = dequant_reference_dsfp8(
            args.gemm2_weights[i], args.gemm2_scales[i], False, True, True)

    args_dequant = moe_args_dequant(
        args.num_tokens, args.num_experts, args.hidden_size,
        args.intermediate_size, args.top_k, args.padding, hidden_states_dequant,
        args.expert_logits, gemm1_weights_dequant, gemm2_weights_dequant,
        args.permute_info, args.use_routing_scales_on_input)

    return run_moe_dequant(args_dequant, "dsFp8"), args_dequant


def test_moe_fp8(num_tokens, num_experts, hidden_size, intermediate_size, n_runs):
    torch.random.manual_seed(0)

    #
    # Data Generation
    #
    top_k = 8
    padding = 8
    n_groups = 8
    top_k_groups = 4
    routed_scaling = 2.5

    assert top_k <= num_experts
    assert top_k == 8
    assert top_k_groups == 4
    assert num_experts > n_groups
    assert num_experts % n_groups == 0
    assert top_k < (top_k_groups * num_experts / n_groups)
    assert hidden_size % 128 == 0
    assert intermediate_size % 128 == 0

    expert_logits = torch.randn((num_tokens, num_experts),
                                device='cuda').to(torch.float)
    routing_bias = torch.randn(num_experts, device='cuda', dtype=torch.bfloat16)

    hidden_states = torch.randn((num_tokens, hidden_size),
                                device='cuda').to(torch.float8_e4m3fn)
    hidden_states_scale = 2 * torch.rand(
        (hidden_size // 128, num_tokens), device='cuda').to(torch.float)

    gemm1_weights = torch.randn(
        (num_experts, 2 * intermediate_size, hidden_size),
        device='cuda').to(torch.float8_e4m3fn)
    gemm1_scales = 2 * torch.rand(
        (num_experts, 2 * intermediate_size // 128, hidden_size // 128),
        device='cuda').to(torch.float)
    gemm2_weights = torch.randn((num_experts, hidden_size, intermediate_size),
                                device='cuda').to(torch.float8_e4m3fn)
    gemm2_scales = 2 * torch.rand(
        (num_experts, hidden_size // 128, intermediate_size // 128),
        device='cuda').to(torch.float)

    permute_info, scores = routing_reference_no_aux(expert_logits, routing_bias,
                                                    top_k, n_groups,
                                                    top_k_groups,
                                                    routed_scaling, padding)

    args = moe_args(num_tokens, num_experts, hidden_size, intermediate_size,
                    top_k, padding, hidden_states, hidden_states_scale, None,
                    scores, gemm1_weights, gemm1_scales, None, gemm2_weights,
                    gemm2_scales, None, permute_info, False)

    printTorchTensorInfo(hidden_states, "tokens")
    printTorchTensorInfo(expert_logits, "logits")
    printTorchTensorInfo(gemm1_weights, "gemm1")
    printTorchTensorInfo(gemm2_weights, "gemm2")
    print(f"{top_k} of {num_experts} experts active")
    print("num runs", n_runs)

    with ProfilerTriton("Trtllm") as p:
        for i in range(n_runs):
            output = torch.ops.trtllm.fp8_block_scale_moe_runner(
                expert_logits, routing_bias, hidden_states, hidden_states_scale,
                gemm1_weights, gemm1_scales, gemm2_weights, gemm2_scales,
                num_experts, top_k, n_groups, top_k_groups, intermediate_size,
                0, num_experts, routed_scaling)

    output_dequant_actual = output.to(torch.float)
    #
    # Run the reference implementations
    #
    output_dequant_reference, _ = run_moe_reference_dsfp8(args)

    #
    # Check the results
    #
    def check_accuracy(a, b, atol, rtol, percent):
        if torch.any(torch.isnan(a)):
            raise Exception("NaN in a")
        if torch.any(torch.isnan(b)):
            raise Exception("NaN in b")
        assert a.shape == b.shape
        left = torch.abs(a - b)
        right = atol + rtol * torch.abs(b)
        count = torch.sum(left > right)
        mismatch_percent = count / a.numel()
        print("Mismatch percentage: %f" % mismatch_percent)
        if mismatch_percent > 1 - percent:
            raise Exception("Mismatch percentage is %f for rtol %f" %
                            (mismatch_percent, rtol))

    check_accuracy(output_dequant_reference,
                   output_dequant_actual,
                   atol=0.1,
                   rtol=0.85,
                   percent=0.925)


def computeTrtllm(num_tokens,n_runs):
    #num_tokens = 1024
    num_experts = 128
    hidden_size = 5120
    intermediate_size = 4096
    test_moe_fp8(num_tokens, num_experts, hidden_size, intermediate_size,n_runs)

##############################################################################
from pathlib import Path
import matplotlib.pyplot as plt
import json
import triton.profiler as proton
import torch
import triton_bench.swiglu
from triton_bench.numerics_details.mxfp import downcast_to_mxfp
from triton_bench.matmul_ogs import MicroscalingCtx, matmul_ogs, PrecisionConfig, FlexCtx
from triton_bench.numerics import InFlexData
from triton_bench.routing import routing
from triton_bench.target_info import is_hip, get_cdna_version
from dataclasses import dataclass

if torch.cuda.is_available() and not is_hip():
    from triton._C.libtriton import nvidia
    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    cublas = None


def _query_gpu_specs():
    import subprocess
    if is_hip():
        cmd = ["rocm-smi", "--showproductname", "-d=0", "--csv"]
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        model = output.splitlines()[1].split(",")[2]
        if model in ["0x74a9", "0x74a1"]:
            name = "AMD Instinct MI300X"
        elif model == "0x74a5":
            name = "AMD Instinct MI325X"
        else:
            name = "AMD"
    else:
        cmd = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i=0"]
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        name = output.splitlines()[0]

    gpu_specs = {
        "NVIDIA H100 80GB HBM3": {"MAX_TFLOPS8": 1979, "MAX_TFLOPS16": 989, "MAX_TBPS": 3.35},
        "HGX GB200": {"MAX_TFLOPS8": 4500, "MAX_TFLOPS16": 2250, "MAX_TBPS": 8.0},
        "AMD Instinct MI300X": {"MAX_TFLOPS8": 2615, "MAX_TFLOPS16": 1307, "MAX_TBPS": 5.3},
        "AMD Instinct MI325X": {"MAX_TFLOPS8": 2615, "MAX_TFLOPS16": 1307, "MAX_TBPS": 6.0},
    }
    return gpu_specs.get(name)


SPECS = _query_gpu_specs()


def quantize(w, dtype, dev, **opt):
    if dtype == "bf16":
        wq = w.to(torch.bfloat16).transpose(-1, -2).contiguous().transpose(-1, -2)
        return wq, InFlexData(), MicroscalingCtx()
    elif dtype == "fp8":
        fp8e4_dtype = torch.float8_e4m3fn if get_cdna_version() != 3 \
            else torch.float8_e4m3fnuz
        wq = w.to(fp8e4_dtype).transpose(-1, -2).contiguous().transpose(-1, -2)
        return wq, InFlexData(dtype=wq.dtype, scale=w.abs().max().unsqueeze(0)), \
                   MicroscalingCtx()
    else:
        assert dtype == "mx4", f"{dtype=}"
        swizzle_mx_scale = opt["swizzle_mx_scale"]
        swizzle_axis = 2 if swizzle_mx_scale else None
        w = w.to(torch.bfloat16)
        w, mx_scales, weight_scale_shape = downcast_to_mxfp(w, torch.uint8, axis=1, swizzle_axis=swizzle_axis)
        return w, InFlexData(), MicroscalingCtx(weight_scale=mx_scales, swizzle_mx=swizzle_mx_scale,
                                                actual_weight_scale_shape=weight_scale_shape)


@dataclass
class PerfData:
    time: float
    flops: float
    bytes: float

    @property
    def tflops(self):
        return self.flops / self.time * 1e-3

    @property
    def tbps(self):
        return self.bytes / self.time * 1e-3

    @property
    def opint(self):
        # operational intensity
        assert self.bytes > 0
        return self.flops / self.bytes

    @property
    def util(self) -> float:
        if SPECS is None:
            return 0.0

        peak_flops = max(SPECS["MAX_TFLOPS8"], SPECS.get("MAX_TFLOPS16", 0))
        min_t_flop = self.flops / peak_flops * 1e-3  # ns → µs
        min_t_bw = self.bytes / SPECS["MAX_TBPS"] * 1e-3
        return max(min_t_flop, min_t_bw) / self.time


def bench_mlp(batch, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP, EP, name, n_runs):
    assert n_expts_tot % EP == 0
    assert dim2 % TP == 0
    dev = "cuda"

    # input
    # weights
    wg = torch.randn((dim1, n_expts_tot), device=dev)
    w1 = torch.randn((n_expts_tot // EP, dim1, dim2 // TP), device=dev)
    w2 = torch.randn((n_expts_tot // EP, dim2 // TP // 2, dim1), device=dev)
    # biases
    bg = torch.randn((n_expts_tot, ), device=dev)
    b1 = torch.randn((dim2 // TP, ), device=dev)
    b2 = torch.randn((dim1, ), device=dev)

    # -- numerics --
    optg = dict()
    opt1 = {"swizzle_mx_scale": True} if w_dtype == "mx4" else dict()
    opt2 = {"swizzle_mx_scale": True} if w_dtype == "mx4" else dict()
    wg, wg_flex, wg_mx = quantize(wg, "bf16", dev, **optg)
    w1, w1_flex, w1_mx = quantize(w1, w_dtype, dev, **opt1)
    w2, w2_flex, w2_mx = quantize(w2, w_dtype, dev, **opt2)
    pcg = PrecisionConfig(mx_ctx=wg_mx, flex_ctx=FlexCtx(rhs_data=wg_flex))
    pcs = triton_bench.swiglu.PrecisionConfig(limit=1.0)
    pc1 = PrecisionConfig(mx_ctx=w1_mx, flex_ctx=FlexCtx(rhs_data=w1_flex))
    pc2 = PrecisionConfig(mx_ctx=w2_mx, flex_ctx=FlexCtx(rhs_data=w2_flex))

    x_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}[x_dtype]
    # special treatment of fp8_e4m3 on AMD CDNA3 because it uses fp8_e4m3fnuz
    if x_dtype == torch.float8_e4m3fn and get_cdna_version() == 3:
        x_dtype = torch.float8_e4m3fnuz

    x = torch.randn((batch, dim1), device=dev)
    xg = x.to(wg.dtype if n_expts_tot > 1 else x_dtype)
    x = x.to(x_dtype)
    # run layer
    assert n_expts_tot > 1
    logits = matmul_ogs(xg, wg, bg, precision_config=pcg)

    printTorchTensorInfo(x, "tokens")
    printTorchTensorInfo(logits, "logits")
    printTorchTensorInfo(w1, "gemm1")
    printTorchTensorInfo(w2, "gemm2")
    print(f"{n_expts_act} of {n_expts_tot} experts active")
    print("num runs", n_runs)

    old_x = x

    with ProfilerTriton("Trtllm") as p: 
        for i in range(n_runs):
            rdata, gather_indx, scatter_indx = routing(logits, n_expts_act, simulated_ep=EP)
            x = matmul_ogs(old_x, w1, None, rdata, gather_indx=gather_indx, precision_config=pc1)
            x = triton_bench.swiglu.swiglu(x, 1.0, pcs, routing_data=rdata)
            x = matmul_ogs(x, w2, None, rdata, scatter_indx=scatter_indx, precision_config=pc2)


def computeTritonFp8(num_tokens,n_runs):
    bench_mlp(num_tokens, 5120, 8192, 128, 8, "fp8", "fp8", TP=1, EP=1, name="llama4-maverick", n_runs=n_runs)


def computeTritonMx4(num_tokens,n_runs):
    bench_mlp(num_tokens, 5120, 8192, 128, 8, "fp8", "mx4", TP=1, EP=1, name="llama4-maverick", n_runs=n_runs)


import sys
torch.manual_seed(int(sys.argv[1]))
print("Running with seed", sys.argv[1])

for num_tokens in [1, 128, 256, 512, 1024]:
    computeTrtllm(num_tokens,50)
    computeTritonFp8(num_tokens,50)
    computeTritonMx4(num_tokens,50)