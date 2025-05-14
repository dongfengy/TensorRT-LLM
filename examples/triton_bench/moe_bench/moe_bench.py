import sys
from computeTriton import computeTritonFp8, computeTritonMx4
from computeTrtllm import computeTrtllmFp8

torch.manual_seed(int(sys.argv[1]))
print("Running with seed", sys.argv[1])

for num_tokens in [1]:
    computeTrtllmFp8(num_tokens,50)
    computeTritonFp8(num_tokens,50)
    computeTritonMx4(num_tokens,50)