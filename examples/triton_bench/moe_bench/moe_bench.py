import sys
from computeTriton import computeTritonFp8, computeTritonMx4
from computeTrtllm import computeTrtllmFp8, computeTrtllmFp4
from utils import table_all, table_gemm
import torch

num_tokens_all=[1, 2, 4, 8, 16, 32, 64,128, 256, 512, 1024]
current_tokens = []

for num_tokens in num_tokens_all:
    expert_logits = torch.randn((num_tokens, 128),
                                device='cuda').to(torch.float)
    computeTrtllmFp8(num_tokens,50,expert_logits)
    computeTrtllmFp4(num_tokens,50,expert_logits)
    computeTritonFp8(num_tokens,50,expert_logits)
    computeTritonMx4(num_tokens,50,expert_logits)

    current_tokens.append(num_tokens)

    names = ["TrtllmFp8", "TrtllmFp4", "TritonFp8", "TritonMx4"]
    print("AllKernels",end="\t")
    for name in names:
        print(f"{name}",end="\t")
    print()
    for num_tokens in current_tokens:
        print(f"NumTokens={num_tokens}",end="\t")
        for name in names:
            try:
                print(f"{table_all[num_tokens][name]:.3f}ms",end="\t")
            except KeyError:
                print(f"NaN",end="\t")
        print()
    print("OnlyGemms",end="\t")
    for name in names:
        print(f"{name}",end="\t")
    print()
    for num_tokens in current_tokens:
        print(f"NumTokens={num_tokens}",end="\t")
        for name in names:
            try:
                print(f"{table_gemm[num_tokens][name]:.3f}ms",end="\t")
            except KeyError:
                print(f"NaN",end="\t")
        print()
    
    print(flush=True)