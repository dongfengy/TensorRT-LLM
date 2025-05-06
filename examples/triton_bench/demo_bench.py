# A simple example of how to use the Triton profiler to benchmark a matrix multiplication operation

import json
from pathlib import Path

import torch
import triton.profiler as proton
from triton_bench.matmul_ogs import (FlexCtx, MicroscalingCtx, PrecisionConfig,
                                     matmul_ogs)
from triton_bench.numerics import InFlexData


def quantize(weight: torch.Tensor, dtype: str):
    assert dtype == "fp8", "Only fp8 quantization is supported in this example"
    wq = weight.to(torch.float8_e4m3fn).transpose(-1,
                                                  -2).contiguous().transpose(
                                                      -1, -2)
    scale = weight.abs().max().unsqueeze(0)
    flex = InFlexData(dtype=wq.dtype, scale=scale)
    return wq, flex, MicroscalingCtx()


# ---- Configuration ----
batch = 32768
dim1, dim2 = 8192, 8192
device = torch.device("cuda")

print("=" * 60)

# ---- Generate data ----
w1 = torch.randn((dim1, dim2), device=device)
w1_q, w1_flex, w1_ctx = quantize(w1, "fp8")
pc1 = PrecisionConfig(mx_ctx=w1_ctx, flex_ctx=FlexCtx(rhs_data=w1_flex))
b1 = torch.randn((dim2, ), device=device)

x = torch.randn((batch, dim1), device=device).to(torch.float8_e4m3fn)

# ---- Profiling setup ----
log_dir = Path("/tmp/demo_bench")
log_dir.mkdir(parents=True, exist_ok=True)
hatchet = log_dir / "demo_bench.hatchet"

# ---- Run profile ----
proton.start(str(hatchet.with_suffix('')), hook="triton")
for _ in range(100):
    x = matmul_ogs(x, w1_q, b1, precision_config=pc1)
proton.finalize()

# ---- Parse metrics ----
with open(hatchet) as f:
    data = json.load(f)
mats = [c for c in data[0]["children"] if "_matmul" in c["frame"]["name"]]
total_bytes = sum(c["metrics"]["bytes"] for c in mats)
total_flops = sum(c["metrics"].get("flops8", 0) +
                  c["metrics"].get("flops16", 0) for c in mats)
total_time = sum(c["metrics"].get("time (ns)", 0) for c in data[0]["children"])

tflops = total_flops / total_time * 1e-3
tbps = total_bytes / total_time * 1e-3

print(f"TFLOPS={tflops:.2f}, TBPS={tbps:.2f}")
print("=" * 60)
