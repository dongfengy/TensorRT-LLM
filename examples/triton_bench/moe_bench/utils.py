import json
import time
from pathlib import Path

import triton.profiler as proton

table_all={}
table_gemm={}

class ProfilerTriton:

    def __init__(self, name, num_tokens):
        self.name = name
        self.num_tokens = num_tokens

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
        gemm_ns = 0
        gemm_cnt=0
        for name, t_ns in timings:
            print(f"  {t_ns / 1e6:.3f} ms : {name}")
            if name.startswith("MoE_Proj") or name.startswith("_matmul_og"):
                gemm_ns += t_ns
                gemm_cnt+=1
        assert gemm_cnt==2
        print(f"[{self.name}] Total CUDA time: {total_ns / 1e6:.3f} ms")
        print(f"[{self.name}] Total GEMM time: {gemm_ns / 1e6:.3f} ms")
        end_time = time.perf_counter()
        elapsed = end_time - self.start_time
        print(f"[{self.name}] Python native wall time: {elapsed * 1e3:.3f} ms")
        print("=" * 60)
        if self.num_tokens not in table_all:
            table_all[self.num_tokens] = {}
        if self.num_tokens not in table_gemm:
            table_gemm[self.num_tokens] = {}
        table_all[self.num_tokens][self.name] = total_ns / 1e6
        table_gemm[self.num_tokens][self.name] = gemm_ns / 1e6


def printTorchTensorInfo(tensor, name):
    print(f"{name}: {tensor.shape} {tensor.dtype} {tensor.device}")
