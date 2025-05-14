import sys
from computeTriton import computeTritonFp8, computeTritonMx4
from computeTrtllm import computeTrtllmFp8
from utils import table_all, table_gemm
import torch

num_tokens_all = [1, 128, 256, 512, 1024]

for num_tokens in num_tokens_all:
    expert_logits = torch.randn((num_tokens, 128),
                                device='cuda').to(torch.float)
    computeTrtllmFp8(num_tokens,50,expert_logits)
    computeTritonFp8(num_tokens,50,expert_logits)
    computeTritonMx4(num_tokens,50,expert_logits)

names = ["TrtllmFp8", "TritonFp8","TritonMx4"]
print("AllKernels",end="\t")
for name in names:
    print(f"{name}",end="\t")
print()
for num_tokens in num_tokens_all:
    print(f"NumTokens={num_tokens}",end="\t")
    for name in names:
        print(f"{table_all[num_tokens][name]:.3f} ms",end="\t")
    print()
print("OnlyGemms",end="\t")
for name in names:
    print(f"{name}",end="\t")
print()
for num_tokens in num_tokens_all:
    print(f"NumTokens={num_tokens}",end="\t")
    for name in names:
        print(f"{table_gemm[num_tokens][name]:.3f} ms",end="\t")
    print()