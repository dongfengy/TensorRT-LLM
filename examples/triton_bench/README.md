# OpenAI Triton Benchmark Tools And Kernels

OpenAI Triton is integrated as a submodule for TRT-LLM users to access the benchmark tools and kernels.

## Install OpenAI Triton

We assume user is running Triton and TRT-LLM in TRT-LLM's own container. See the `docker` folder in this repo.

1. Remove the older version of Triton provided by base container nvcr.io/nvidia/pytorch. We will need to delete the folder directly since it's not indexed by pip.
```
rm -rf /usr/local/lib/python3.12/dist-packages/triton
```
2. Install the latest Triton
```
pip install triton==x.x.x
```
TODO: Provide the version that matches our Triton submodule commit after Triton has new releases that support the latest bench code. For now a workaround is to `pip install /home/scratch.dongfengy_sw_1/triton_moe/triton-3.3.0+gite32c3b13-cp312-cp312-linux_x86_64.whl`

## Install Triton Bench

Triton Bench is organized into a separate python package. To install:
```
pushd 3rdparty/triton/bench
pip install -e .
popd
```

## Run Benchmarks
### Run Demo Bench
`python bench/bench_mlp.py`
### Run Triton Example Bench
`python 3rdparty/triton/bench/bench/bench_mlp.py`
