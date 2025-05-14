from utils import ProfilerTriton, printTorchTensorInfo

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
    if n_group > 1:
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
        score_mask = group_mask.unsqueeze(-1).expand(
            scores_shape[:-1] +
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
    assert top_k <= 8
    assert top_k_groups <= 4
    assert num_experts > n_groups
    assert num_experts % n_groups == 0
    assert num_experts % 4 == 0
    assert top_k < (top_k_groups * num_experts / n_groups)
    assert hidden_size % 128 == 0
    assert intermediate_size % 128 == 0

    expert_logits = torch.randn((num_tokens, num_experts),
                                device='cuda').to(torch.float)
    routing_bias = torch.zeros(num_experts, device='cuda', dtype=torch.bfloat16)

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

    with ProfilerTriton("TrtllmFp8", num_tokens) as p:
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


def computeTrtllmFp8(num_tokens,n_runs):
    #num_tokens = 1024
    num_experts = 128
    hidden_size = 5120
    intermediate_size = 4096
    test_moe_fp8(num_tokens, num_experts, hidden_size, intermediate_size,n_runs)