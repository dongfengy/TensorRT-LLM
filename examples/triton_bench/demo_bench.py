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

# A simple example of how to use the Triton profiler to benchmark a matrix multiplication operation


def computeTriton():
    import torch
    from triton_bench.matmul_ogs import (FlexCtx, MicroscalingCtx,
                                         PrecisionConfig, matmul_ogs)
    from triton_bench.numerics import InFlexData
    batch = 32768
    dim1, dim2 = 8192, 8192
    device = torch.device("cuda")
    w1 = torch.randn((dim1, dim2), device=device)

    def quantize(weight: torch.Tensor, dtype: str):
        assert dtype == "fp8", "Only fp8 quantization is supported in this example"
        wq = weight.to(torch.float8_e4m3fn).transpose(
            -1, -2).contiguous().transpose(-1, -2)
        scale = weight.abs().max().unsqueeze(0)
        flex = InFlexData(dtype=wq.dtype, scale=scale)
        return wq, flex, MicroscalingCtx()

    w1_q, w1_flex, w1_ctx = quantize(w1, "fp8")
    pc1 = PrecisionConfig(mx_ctx=w1_ctx, flex_ctx=FlexCtx(rhs_data=w1_flex))
    b1 = torch.randn((dim2, ), device=device)
    x = torch.randn((batch, dim1), device=device).to(torch.float8_e4m3fn)
    for _ in range(100):
        x = matmul_ogs(x, w1_q, b1, precision_config=pc1)


def computeTorch():
    import torch
    batch = 32768
    dim1, dim2 = 8192, 8192
    device = torch.device("cuda")
    w1 = torch.randn((dim1, dim2), device=device)
    b1 = torch.randn((dim2, ), device=device)
    x = torch.randn((batch, dim1), device=device)
    for _ in range(100):
        x = torch.nn.functional.linear(x, w1, b1)


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


with ProfilerTriton("Triton") as p:
    computeTriton()

with ProfilerTriton("Torch") as p:
    computeTorch()
