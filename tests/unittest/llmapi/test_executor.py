import asyncio
import datetime
import json
import os
import signal
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from queue import Empty

import pytest
import torch
import zmq

import tensorrt_llm.executor.postproc_worker as postproc_worker_module
import tensorrt_llm.executor.worker as worker_module
from tensorrt_llm._utils import mpi_world_size
from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.executor import (DetokenizedGenerationResultBase,
                                   GenerationRequest, GenerationResult,
                                   GenerationResultBase, PostprocWorker)
from tensorrt_llm.executor.ipc import FusedIpcQueue, ZeroMqQueue
from tensorrt_llm.llmapi.tokenizer import TransformersTokenizer
from tensorrt_llm.llmapi.utils import AsyncQueue
from tensorrt_llm.sampling_params import SamplingParams

# isort: off
from utils.llm_data import llm_models_root
# isort: on

WORLD_SIZE = mpi_world_size()


@pytest.fixture(scope="module")
def engine_path():
    return Path(tempfile.tempdir) / "llm_engine"


def test_invalid_sampling_params():
    with pytest.raises(ValueError):
        # n > 1 does not allow greedy decoding, which is deterministic.
        SamplingParams(max_tokens=4, n=4, top_k=1, top_p=0.0)
    with pytest.raises(ValueError):
        # n > beam_width is not possible because n exceeds the number of beam
        # search results
        SamplingParams(max_tokens=4, n=4, best_of=3, use_beam_search=True)


def test_FusedIpcQueue():
    producer_queue = FusedIpcQueue(is_server=True, fuse_message=False)
    consumer_queue = FusedIpcQueue(is_server=False,
                                   address=producer_queue.address,
                                   fuse_message=False)

    def producer(queue: FusedIpcQueue, n: int):
        for i in range(n):
            queue.put(i)
        queue.put(None)

    def consumer(queue: FusedIpcQueue):
        to_continue = True
        while to_continue:
            item = queue.get()
            item = [item] if not isinstance(item, list) else item

            for i in item:
                if i is None:
                    to_continue = False
                    break
                print(f"consumer got {i}")

    producer_thread = threading.Thread(target=producer,
                                       args=(producer_queue, 10))
    consumer_thread = threading.Thread(target=consumer, args=(consumer_queue, ))

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()


def create_rsp(id, finished: bool = False):
    result = tllm.Result()
    result.output_token_ids = [[id]]
    result.context_logits = None
    result.generation_logits = None
    result.log_probs = None
    result.cum_log_probs = None
    if finished:
        result.finish_reasons = [tllm.FinishReason.END_ID]
    result.is_final = finished
    result.sequence_index = 0
    return tllm.Response(request_id=0, result=result, client_id=0)


def test_GenerationResultBase():
    sampling_params = SamplingParams(max_tokens=4)
    result = GenerationResultBase(
        id=2,
        sampling_params=sampling_params,
    )
    result._handle_response(create_rsp(2, finished=False))
    result._handle_response(create_rsp(3, finished=False))
    result._handle_response(create_rsp(4, finished=True))
    print(result.outputs[0])
    assert len(result.outputs[0].token_ids) == 3
    assert result._done


def test_GenerationResult():
    request = GenerationRequest(prompt_token_ids=[12, 23, 34],
                                sampling_params=SamplingParams(max_tokens=4))
    result = GenerationResult(request)

    for i in range(11):
        result._handle_response(create_rsp(i + 33, finished=False))
    result._handle_response(create_rsp(44, finished=True))
    assert len(result.outputs[0].token_ids) == 12
    assert result._done


def test_result_timeout_raises():
    request = GenerationRequest(prompt_token_ids=[12, 23, 34],
                                sampling_params=SamplingParams(max_tokens=4))
    result = GenerationResult(request)

    # Queue stays empty (no worker pushing responses) -> must time out fast, not block indefinitely.
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        result.result(timeout=0.1)
    elapsed = time.monotonic() - start
    assert elapsed < 2.0, f"result() did not honor timeout (took {elapsed:.2f}s)"
    assert not result._done


def test_result_timeout_budget_across_steps():
    request = GenerationRequest(prompt_token_ids=[12, 23, 34],
                                sampling_params=SamplingParams(max_tokens=4))
    result = GenerationResult(request)

    # A single non-final response is available, then the queue goes empty and the request never
    # completes.
    result.queue.put(create_rsp(33, finished=False))

    start = time.monotonic()
    with pytest.raises(TimeoutError):
        result.result(timeout=0.1)
    elapsed = time.monotonic() - start
    assert elapsed < 2.0, f"result() did not honor timeout (took {elapsed:.2f}s)"
    assert not result._done


def test_result_zero_timeout_completes_with_queued_responses():
    request = GenerationRequest(prompt_token_ids=[12, 23, 34],
                                sampling_params=SamplingParams(max_tokens=4))
    result = GenerationResult(request)

    result.queue.put(create_rsp(33, finished=False))
    result.queue.put(create_rsp(44, finished=True))

    assert result.result(timeout=0) is result
    assert result._done
    assert len(result.outputs[0].token_ids) == 2


def test_sync_queue_zero_timeout_checks_for_queued_item():
    queue = AsyncQueue()
    queue.put("ready")

    with pytest.warns(UserWarning):
        assert queue.sync_q.get(timeout=0) == "ready"
    with pytest.warns(UserWarning), pytest.raises(Empty):
        queue.sync_q.get(timeout=0)


def test_result_completes_within_timeout():
    request = GenerationRequest(prompt_token_ids=[12, 23, 34],
                                sampling_params=SamplingParams(max_tokens=4))
    result = GenerationResult(request)

    result.queue.put(create_rsp(33, finished=False))
    result.queue.put(create_rsp(44, finished=True))

    ret = result.result(timeout=30.0)
    assert ret is result
    assert result._done
    assert len(result.outputs[0].token_ids) == 2


def test_DetokenizedGenerationResultBase():
    sampling_params = SamplingParams(max_tokens=4)
    model_path = llm_models_root() / "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = TransformersTokenizer.from_pretrained(model_path)
    result = DetokenizedGenerationResultBase(
        id=2,
        sampling_params=sampling_params,
        tokenizer=tokenizer,
    )
    result._handle_response(create_rsp(20, finished=False))
    result._handle_response(create_rsp(30, finished=False))
    result._handle_response(create_rsp(40, finished=True))
    print(result.outputs[0])
    assert len(result.outputs[0].token_ids) == 3
    assert result._done


def test_abort_on_GenerationResultBase():
    """abort() and aborted() are available on GenerationResultBase."""
    sampling_params = SamplingParams(max_tokens=4)
    result = GenerationResultBase(id=1, sampling_params=sampling_params)
    assert not result.aborted()
    result.abort()
    assert result.aborted()


def test_abort_on_DetokenizedGenerationResultBase():
    """DetokenizedGenerationResultBase inherits abort() so postprocess workers can call it without AttributeError (NVBug 5955173)."""
    sampling_params = SamplingParams(max_tokens=4)
    result = DetokenizedGenerationResultBase(id=1,
                                             sampling_params=sampling_params)
    assert not result.aborted()
    result._handle_response(create_rsp(10, finished=False))
    assert not result._done

    result.abort()
    assert result.aborted()


def test_PostprocWorker_Output_should_abort():
    """PostprocWorker.Output carries should_abort flag for worker-to-main-thread abort signal propagation."""
    out_default = PostprocWorker.Output(client_id=0, res=None, is_final=False)
    assert out_default.should_abort is False

    out_abort = PostprocWorker.Output(client_id=0,
                                      res=None,
                                      is_final=False,
                                      should_abort=True)
    assert out_abort.should_abort is True


def test_handle_response_propagates_should_abort():
    """When a PostprocWorker.Output has should_abort=True, _handle_response on the main-thread GenerationResult calls abort() (NVBug 5955173)."""
    sampling_params = SamplingParams(max_tokens=4)
    result = GenerationResultBase(id=1, sampling_params=sampling_params)
    assert not result.aborted()

    output = PostprocWorker.Output(client_id=1,
                                   res="mock_sse_data",
                                   is_final=False,
                                   should_abort=True)
    result._handle_response(output)
    assert result.aborted()
    assert result._outputs[0]._postprocess_result == "mock_sse_data"


def test_PostprocWorker_Output_tracing_fields():
    """PostprocWorker.Output carries finish_reason and num_generated_tokens for
    tracing on the num_postprocess_workers > 0 path. Both fields are optional
    and default to None when not populated by the worker."""
    out = PostprocWorker.Output(client_id=0, res=None, is_final=False)
    assert out.finish_reason is None
    assert out.num_generated_tokens is None


def test_handle_response_postproc_nonstreaming_propagates_metadata():
    """On the non-streaming path (res is not CompletionOutput), finish_reason and
    token_ids are set on _outputs[0] from the PostprocWorker.Output fields."""
    sampling_params = SamplingParams(max_tokens=10)
    result = GenerationResultBase(id=1, sampling_params=sampling_params)

    output = PostprocWorker.Output(
        client_id=1,
        res="mock_sse_data",  # non-streaming: res is not a CompletionOutput
        is_final=True,
        finish_reason="stop",
        num_generated_tokens=5,
    )
    result._handle_response(output)

    assert result._done
    assert result._outputs[0].finish_reason == "stop"
    assert len(result._outputs[0].token_ids) == 5


class _PostprocProcessStub:

    def __init__(self, pid=123, returncode=None, wait_results=()):
        self.pid = pid
        self._returncode = returncode
        self._wait_results = list(wait_results)
        self.signals = []

    @property
    def returncode(self):
        return self.poll()

    def poll(self):
        return self._returncode

    def wait(self, timeout=None):
        del timeout
        result = self._wait_results.pop(0)
        if result == "timeout":
            raise TimeoutError
        self._returncode = result
        return result

    def signal_process_group(self, sig):
        self.signals.append(sig)


def test_postproc_worker_spec_round_trips_hmac_keys():
    payload = postproc_worker_module.encode_postproc_worker_spec(
        ("ipc://input", b"\x00\xff"),
        [("ipc://output-0", None), ("ipc://output-1", b"\x01\x02")],
        "/tokenizer",
        "hooks.make",
    )
    spec = json.loads(payload)

    assert spec["feedin_ipc_addr"]["hmac_key"] == "00ff"
    assert spec["feedout_ipc_addrs"][0]["hmac_key"] is None
    assert postproc_worker_module._decode_ipc_addr(
        spec["feedin_ipc_addr"]) == ("ipc://input", b"\x00\xff")
    assert postproc_worker_module._decode_ipc_addr(
        spec["feedout_ipc_addrs"][0]) == ("ipc://output-0", None)
    assert postproc_worker_module._decode_ipc_addr(
        spec["feedout_ipc_addrs"][1]) == ("ipc://output-1", b"\x01\x02")


def test_spawn_postproc_worker_uses_clean_exec_contract(monkeypatch):
    opened_fds = []
    spawn_call = {}
    writes = []

    def open_pipe():
        pipe_fds = os.pipe()
        opened_fds.extend(pipe_fds)
        return pipe_fds

    def posix_spawn(executable, argv, env, **kwargs):
        spawn_call.update(executable=executable, argv=argv, env=env, **kwargs)
        return 43210

    monkeypatch.setattr(worker_module, "_open_cloexec_pipe", open_pipe)
    monkeypatch.setattr(
        worker_module, "split_mpi_env", lambda: ({
            "PATH": "/clean",
            "SAFE": "1"
        }, {
            "OMPI_COMM_WORLD_RANK": "63"
        }))
    monkeypatch.setattr(worker_module.os, "posix_spawn", posix_spawn)
    monkeypatch.setattr(worker_module, "_write_all",
                        lambda fd, payload: writes.append((fd, payload)))

    payload = b'{"hmac_key":"private-0102"}'
    processes = []
    ready_fd = worker_module._spawn_postproc_worker(payload, processes)
    spec_read_fd, spec_write_fd, _, ready_write_fd = opened_fds
    child_ready_fd = int(
        spawn_call["env"][postproc_worker_module.POSTPROC_WORKER_READY_FD_ENV])
    try:
        executable = os.path.abspath(worker_module.sys.executable)
        assert spawn_call["executable"] == executable
        assert spawn_call["argv"][:5] == [
            executable,
            "-I",
            "-S",
            "-c",
            worker_module.POSTPROC_WORKER_SUBPROCESS_COMMAND,
        ]
        assert all(
            os.path.isabs(path) for path in json.loads(spawn_call["argv"][5]))
        assert spawn_call["env"] == {
            "PATH":
            "/clean",
            "SAFE":
            "1",
            "TLLM_DISABLE_MPI":
            "1",
            postproc_worker_module.POSTPROC_WORKER_READY_FD_ENV:
            str(child_ready_fd),
        }
        assert "private-0102" not in json.dumps(spawn_call["argv"])
        assert "private-0102" not in json.dumps(spawn_call["env"])
        assert spawn_call["file_actions"] == [
            (os.POSIX_SPAWN_DUP2, spec_read_fd, 0),
            (os.POSIX_SPAWN_DUP2, ready_write_fd, child_ready_fd),
            (os.POSIX_SPAWN_CLOSE, spec_read_fd),
            (os.POSIX_SPAWN_CLOSE, spec_write_fd),
            (os.POSIX_SPAWN_CLOSE, ready_fd),
            (os.POSIX_SPAWN_CLOSE, ready_write_fd),
        ]
        assert spawn_call["setpgroup"] == 0
        assert spawn_call["setsigmask"] == ()
        assert spawn_call["setsigdef"] == worker_module._SPAWN_DEFAULT_SIGNALS
        assert writes == [(spec_write_fd, payload)]
        assert [process.pid for process in processes] == [43210]
        os.fstat(ready_fd)
        for fd in (spec_read_fd, spec_write_fd, ready_write_fd, child_ready_fd):
            with pytest.raises(OSError):
                os.fstat(fd)
    finally:
        worker_module._close_fd(ready_fd)


@pytest.mark.parametrize(
    ("ready_payload", "returncode", "timeout", "error"),
    [
        (b"R", None, None, None),
        (b"", None, None, "before signaling READY"),
        (b"R", 17, None, "code 17 immediately after READY"),
        (None, None, "0.01", "not ready within"),
    ],
)
def test_wait_postproc_workers_ready_states(monkeypatch, ready_payload,
                                            returncode, timeout, error):
    ready_read_fd, ready_write_fd = os.pipe()
    process = _PostprocProcessStub(returncode=returncode)
    try:
        if ready_payload is not None:
            if ready_payload:
                os.write(ready_write_fd, ready_payload)
            os.close(ready_write_fd)
            ready_write_fd = -1
        monkeypatch.setenv("TLLM_POSTPROC_WORKER_READY_TIMEOUT", timeout or "1")
        if error is None:
            worker_module._wait_postproc_workers_ready([process],
                                                       [ready_read_fd])
        else:
            with pytest.raises(RuntimeError, match=error):
                worker_module._wait_postproc_workers_ready([process],
                                                           [ready_read_fd])
    finally:
        worker_module._close_fd(ready_read_fd)
        worker_module._close_fd(ready_write_fd)


def test_start_postproc_workers_preserves_partial_process_for_cleanup(
        monkeypatch):
    ready_read_fd, ready_write_fd = os.pipe()
    worker_module._close_fd(ready_write_fd)
    first_process = _PostprocProcessStub(pid=321)

    def spawn(spec, processes):
        del spec
        if not processes:
            processes.append(first_process)
            return ready_read_fd
        raise RuntimeError("second worker failed")

    monkeypatch.setattr(worker_module, "_spawn_postproc_worker", spawn)
    config = postproc_worker_module.PostprocWorkerConfig(
        num_postprocess_workers=2, postprocess_tokenizer_dir="/tokenizer")
    result_queues = [
        type("Queue", (), {"address": ("ipc://input-0", None)})(),
        type("Queue", (), {"address": ("ipc://input-1", None)})(),
    ]
    processes = []

    with pytest.raises(RuntimeError, match="second worker failed"):
        worker_module._start_postproc_workers(
            result_queues,
            [("ipc://output", None)],
            config,
            processes,
        )

    assert processes == [first_process]
    with pytest.raises(OSError):
        os.fstat(ready_read_fd)


def test_reap_postproc_workers_uses_graceful_then_bounded_escalation():
    graceful = _PostprocProcessStub(pid=1, wait_results=[0])
    stubborn = _PostprocProcessStub(
        pid=2, wait_results=["timeout", "timeout", -signal.SIGKILL])

    worker_module._reap_postproc_workers([graceful, stubborn])

    assert graceful.returncode == 0
    assert graceful.signals == []
    assert stubborn.returncode == -signal.SIGKILL
    assert stubborn.signals == [signal.SIGTERM, signal.SIGKILL]


def _ZeroMqQueue_sync_sync_task(addr: str):
    print(f"Setup receiver: {addr}")
    pull_pipe = ZeroMqQueue(address=addr, is_server=False, is_async=True)
    print(f"after setup receiver")

    total = 0

    async def task():
        print(f"running task")
        for i in range(10):
            print(f"waiting for msg")
            msg = await pull_pipe.get_async()
            print(f"received: {msg}")
            nonlocal total
            total += msg

    print(f"to run task")
    asyncio.run(task())

    return total


def test_ZeroMqQueue_sync_async():
    # sync send, async recv
    push_pipe = ZeroMqQueue(is_async=False, is_server=True)

    pool = ProcessPoolExecutor(max_workers=1)
    res = pool.submit(_ZeroMqQueue_sync_sync_task, push_pipe.address)

    for i in range(10):
        print(f"put: {i}")
        push_pipe.put(i)

    assert res.result() == 45
    pool.shutdown()
    push_pipe.close()


def _ZeroMqQueue_serialization_complicated_dataclass(addr: str,
                                                     iterations: int):
    pull_pipe = ZeroMqQueue(address=addr, is_server=False, is_async=True)

    total = 0

    async def task():
        print(f"running task")
        for i in range(iterations):
            print(f"waiting for msg")
            msg = await pull_pipe.get_async()
            # print(f"received: {msg}")
            nonlocal total
            try:
                total += msg.prompt_token_ids[0]
            except Exception as e:
                print(f"error: {e}")

    print(f"to run task")
    asyncio.run(task())

    return total


def test_ZeroMqQueue_serialization_complicated_dataclass():
    # sync send message, async recv message
    push_pipe = ZeroMqQueue(is_async=False, is_server=True)
    iterations = 2

    pool = ProcessPoolExecutor(max_workers=1)
    res = pool.submit(_ZeroMqQueue_serialization_complicated_dataclass,
                      push_pipe.address, iterations)

    TokenRangeRetentionConfig = tllm.KvCacheRetentionConfig.TokenRangeRetentionConfig
    kvcache_config = tllm.KvCacheRetentionConfig(
        [TokenRangeRetentionConfig(0, 2, 30, datetime.timedelta(seconds=30))],
        80, None, tllm.KvCacheTransferMode.DRAM, "test_dir")

    sampling_params = SamplingParams(max_tokens=4,
                                     embedding_bias=torch.randn(2, 2))

    for i in range(iterations):
        request = GenerationRequest(prompt_token_ids=[i],
                                    sampling_params=sampling_params,
                                    kv_cache_retention_config=kvcache_config)
        # print(f"put with msg: {request}")
        push_pipe.put(request)

    print(res.result())
    assert res.result() == iterations * (iterations - 1) / 2
    pool.shutdown()
    push_pipe.close()


Input = PostprocWorker.Input
Output = PostprocWorker.Output


def ResponsePostprocessWorker_record_creator(input: Input, tokenizer):
    assert input.sampling_params is not None
    return DetokenizedGenerationResultBase(
        id=input.rsp.client_id,
        sampling_params=input.sampling_params,
        tokenizer=tokenizer)


def ResponsePostprocessWorker_worker_task(pull_pipe_addr, push_pipe_addr,
                                          tokenizer_dir):
    worker = PostprocWorker(
        pull_pipe_addr=pull_pipe_addr,
        push_pipe_addrs=[push_pipe_addr],
        tokenizer_dir=tokenizer_dir,
        record_creator=ResponsePostprocessWorker_record_creator)
    worker.start()


def test_ResponsePostprocessWorker():

    input_pipe = ZeroMqQueue(is_server=True)
    out_pipe = ZeroMqQueue(is_server=True, socket_type=zmq.PULL)

    pool = ProcessPoolExecutor(max_workers=1)
    print("submit task")
    fut = pool.submit(
        ResponsePostprocessWorker_worker_task, input_pipe.address,
        out_pipe.address,
        str(llm_models_root() / "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"))

    inputs = [
        Input(rsp=create_rsp(123),
              sampling_params=SamplingParams(max_tokens=4),
              streaming=False) for i in range(11)
    ]
    inputs.append(
        Input(rsp=create_rsp(123, finished=True),
              sampling_params=SamplingParams(max_tokens=4),
              streaming=True))

    def unbatch():

        for inp in inputs:
            print("put rsp")
            input_pipe.put(inp)

        for i in range(len(inputs)):
            out = out_pipe.get()
            print("output", out)

    def batch():

        input_pipe.put(inputs)
        outs = out_pipe.get()
        print(f"outputs: {outs}")

    unbatch()
    batch()

    input_pipe.put(None)  # tell worker to shutdown
    fut.result()

    pool.shutdown()
    input_pipe.close()
    out_pipe.close()


def test_get_params_for_first_rsp_returns_disaggregated_params_once():
    """Verify _get_params_for_first_rsp extracts disaggregated_params from the GenerationResult on the first call and returns None after.

    Regression test for https://nvbugs/5991957.
    """
    from types import SimpleNamespace

    from tensorrt_llm.executor.base_worker import _get_params_for_first_rsp

    disagg_params = DisaggregatedParams(
        request_type="generation_only",
        first_gen_tokens=[7],
        ctx_request_id=12345,
    )
    request = GenerationRequest(
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=4),
        disaggregated_params=disagg_params,
    )
    request.set_id(42)
    result = GenerationResult(request, disaggregated_params=disagg_params)

    worker = SimpleNamespace(_results={42: result})

    # First call: should return all three params
    sp, pp, dp = _get_params_for_first_rsp(worker, 42)
    assert sp is not None
    assert pp is None  # no postproc_params on this request
    assert dp is not None
    assert dp.request_type == "generation_only"
    assert dp.first_gen_tokens == [7]
    assert dp.ctx_request_id == 12345
    assert result._params_transmitted is True

    # Second call: _params_transmitted is True, all should be None
    sp2, pp2, dp2 = _get_params_for_first_rsp(worker, 42)
    assert sp2 is None
    assert pp2 is None
    assert dp2 is None


def test_PostprocWorker_disaggregated_params():
    """GEN-side: disaggregated_params seeded on the record persists across
    multiple responses (streaming pattern).

    Regression test for https://nvbugs/5991957: PostprocWorker record was
    created without disaggregated_params, causing /v1/chat/completions to
    return 400 in disaggregated serving with num_postprocess_workers > 0.
    """
    input_pipe = ZeroMqQueue(is_server=True)
    out_pipe = ZeroMqQueue(is_server=True, socket_type=zmq.PULL)

    pool = ProcessPoolExecutor(max_workers=1)
    fut = pool.submit(
        ResponsePostprocessWorker_worker_task, input_pipe.address,
        out_pipe.address,
        str(llm_models_root() / "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"))

    disagg_params = DisaggregatedParams(
        request_type="generation_only",
        first_gen_tokens=[7],
        ctx_request_id=12345,
    )

    # First response carries disaggregated_params (transmitted once)
    input_pipe.put(
        Input(rsp=create_rsp(42),
              sampling_params=SamplingParams(max_tokens=4),
              disaggregated_params=disagg_params,
              streaming=True))

    # Subsequent streaming response — no disaggregated_params in Input,
    # but the record should still have it from the first response
    input_pipe.put(
        Input(rsp=create_rsp(43, finished=True),
              sampling_params=None,
              disaggregated_params=None,
              streaming=None))

    for _ in range(2):
        out = out_pipe.get()
        assert isinstance(out, list)
        assert len(out) == 1
        output = out[0]
        assert isinstance(output, Output)
        assert output.disaggregated_params is not None, \
            "disaggregated_params was not propagated through PostprocWorker"
        assert output.disaggregated_params.request_type == "generation_only"
        assert output.disaggregated_params.first_gen_tokens == [7]
        assert output.disaggregated_params.ctx_request_id == 12345

    input_pipe.put(None)
    fut.result()
    pool.shutdown()
    input_pipe.close()
    out_pipe.close()


if __name__ == '__main__':
    test_FusedIpcQueue()
