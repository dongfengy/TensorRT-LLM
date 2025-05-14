from utils import ProfilerTriton, printTorchTensorInfo

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


def bench_mlp(batch, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP, EP, name, n_runs, name2,loo):
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
    logits=loo.to(logits.dtype)

    printTorchTensorInfo(x, "tokens")
    printTorchTensorInfo(logits, "logits")
    print("First two values of logits", logits[0, :2])
    printTorchTensorInfo(w1, "gemm1")
    printTorchTensorInfo(w2, "gemm2")
    print(f"{n_expts_act} of {n_expts_tot} experts active")
    print("num runs", n_runs)

    old_x = x

    with ProfilerTriton(name2,batch) as p: 
        for i in range(n_runs):
            rdata, gather_indx, scatter_indx = routing(logits, n_expts_act, simulated_ep=EP)
            x = matmul_ogs(old_x, w1, None, rdata, gather_indx=gather_indx, precision_config=pc1)
            x = triton_bench.swiglu.swiglu(x, 1.0, pcs, routing_data=rdata)
            x = matmul_ogs(x, w2, None, rdata, scatter_indx=scatter_indx, precision_config=pc2)


def computeTritonFp8(num_tokens,n_runs,expert_logits):
    bench_mlp(num_tokens, 5120, 8192, 128, 8, "fp8", "fp8", TP=1, EP=1, name="llama4-maverick", n_runs=n_runs,name2="TritonFp8",loo=expert_logits)


def computeTritonMx4(num_tokens,n_runs,expert_logits):
    bench_mlp(num_tokens, 5120, 8192, 128, 8, "fp8", "mx4", TP=1, EP=1, name="llama4-maverick", n_runs=n_runs,name2="TritonMx4",loo=expert_logits)
