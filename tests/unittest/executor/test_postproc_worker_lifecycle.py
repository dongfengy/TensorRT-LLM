# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import pickle
import threading
from concurrent.futures import Future
from unittest.mock import Mock

import pytest
import zmq

from tensorrt_llm.executor import postproc_worker
from tensorrt_llm.executor.ipc import ZeroMqQueue
from tensorrt_llm.executor.postproc_worker import PostprocWorker, PostprocWorkerFatal
from tensorrt_llm.executor.proxy import GenerationExecutorFrontendProxy, GenerationExecutorProxy
from tensorrt_llm.executor.worker import (
    _notify_postproc_workers_to_quit,
    _raise_if_postproc_worker_failed,
    _wait_for_postproc_workers,
)

pytestmark = pytest.mark.cpu_only


def test_pending_postproc_worker_is_healthy():
    _raise_if_postproc_worker_failed([Future()])


def test_pending_postproc_worker_with_failure_event_is_fatal():
    failure_event = threading.Event()
    failure_event.set()

    with pytest.raises(RuntimeError, match="Postprocessing worker 0 reported a fatal error"):
        _raise_if_postproc_worker_failed([Future()], [failure_event])


def test_postproc_worker_exception_is_fatal():
    future = Future()
    failure = ValueError("postprocessor failed")
    future.set_exception(failure)

    with pytest.raises(RuntimeError, match="Postprocessing worker 0 failed") as exc_info:
        _raise_if_postproc_worker_failed([future])

    assert exc_info.value.__cause__ is failure


def test_clean_postproc_worker_exit_is_fatal_before_shutdown():
    future = Future()
    future.set_result(None)

    with pytest.raises(RuntimeError, match="Postprocessing worker 0 exited unexpectedly"):
        _raise_if_postproc_worker_failed([future])


def test_wait_for_postproc_worker_detects_initialization_failure():
    future = Future()
    failure = RuntimeError("tokenizer initialization failed")
    future.set_exception(failure)

    with pytest.raises(RuntimeError, match="Postprocessing worker 0 failed") as exc_info:
        _wait_for_postproc_workers([threading.Event()], [future], [threading.Event()])

    assert exc_info.value.__cause__ is failure


def test_postproc_worker_signals_ready_before_entering_mainloop(monkeypatch):
    ready_event = threading.Event()
    entered_mainloop = threading.Event()
    closed = threading.Event()

    class FakePostprocWorker:
        def __init__(self, *args, **kwargs):
            pass

        def start(self, ready_event=None, failure_event=None):
            assert ready_event is not None
            assert failure_event is not None
            ready_event.set()
            assert ready_event.is_set()
            entered_mainloop.set()

        def close(self):
            closed.set()

    monkeypatch.setattr(postproc_worker, "PostprocWorker", FakePostprocWorker)

    postproc_worker.postproc_worker_main(
        ("ipc://feedin", b"key"),
        [("ipc://feedout", b"key")],
        tokenizer_dir="tokenizer",
        record_creator=lambda *_: None,
        ready_event=ready_event,
        failure_event=threading.Event(),
    )

    assert entered_mainloop.is_set()
    assert closed.is_set()


def test_shutdown_skips_failed_postproc_worker_feed():
    class RecordingQueue:
        def __init__(self):
            self.items = []

        def put_bounded(self, item, timeout):
            self.items.append(item)
            return True

    live_future = Future()
    failed_future = Future()
    failed_future.set_exception(RuntimeError("failed"))
    live_queue = RecordingQueue()
    failed_queue = RecordingQueue()
    live_shutdown = threading.Event()
    failed_shutdown = threading.Event()

    _notify_postproc_workers_to_quit(
        [live_queue, failed_queue], [live_future, failed_future], [live_shutdown, failed_shutdown]
    )

    assert live_queue.items == [None]
    assert failed_queue.items == []
    assert live_shutdown.is_set()
    assert failed_shutdown.is_set()


def test_shutdown_event_stops_worker_without_feed_sentinel():
    worker = object.__new__(PostprocWorker)
    worker._to_stop = asyncio.Event()
    worker._shutdown_event = threading.Event()
    worker._shutdown_event.set()

    async def first_batch():
        return await worker._mainloop().__anext__()

    assert asyncio.run(first_batch()) is None


def test_postproc_fatal_is_picklable_and_broadcast_to_every_frontend():
    class RecordingPipe:
        def __init__(self):
            self.items = []

        async def put_async_bounded(self, item, timeout):
            self.items.append((item, timeout))
            return True

    worker = object.__new__(PostprocWorker)
    worker._worker_id = 3
    worker._push_pipes = [RecordingPipe(), RecordingPipe(), RecordingPipe()]

    asyncio.run(worker._broadcast_fatal(ValueError("bad output")))

    fatals = [pipe.items[0][0] for pipe in worker._push_pipes]
    assert all(isinstance(fatal, PostprocWorkerFatal) for fatal in fatals)
    assert all(fatal.worker_id == 3 for fatal in fatals)
    assert all(fatal.error_type == "builtins.ValueError" for fatal in fatals)
    assert all(fatal.error_message == "bad output" for fatal in fatals)
    assert [pipe.items[0][1] for pipe in worker._push_pipes] == [1.0, 0.1, 0.1]
    assert pickle.loads(pickle.dumps(fatals[0])) == fatals[0]


@pytest.mark.parametrize("proxy_cls", [GenerationExecutorProxy, GenerationExecutorFrontendProxy])
def test_proxy_handles_postproc_fatal_before_accessing_client_id(proxy_cls):
    class FakeResultQueue:
        def get(self):
            return PostprocWorkerFatal(
                worker_id=2,
                error_type="builtins.RuntimeError",
                error_message="postprocessor died",
                traceback="traceback text",
            )

    proxy = object.__new__(proxy_cls)
    proxy.result_queue = FakeResultQueue()
    proxy.doing_shutdown = False
    proxy._set_fatal_error = Mock()
    proxy.pre_shutdown = Mock()

    assert proxy.dispatch_result_task() is False
    error = proxy._set_fatal_error.call_args.args[0]
    assert isinstance(error, RuntimeError)
    assert "Postprocessing worker 2 failed" in str(error)
    assert "traceback text" in str(error)
    proxy.pre_shutdown.assert_called_once_with()


def test_unexpected_result_lane_close_is_fatal():
    class FakeResultQueue:
        def get(self):
            return None

    proxy = object.__new__(GenerationExecutorProxy)
    proxy.result_queue = FakeResultQueue()
    proxy.doing_shutdown = False
    proxy._set_fatal_error = Mock()
    proxy.pre_shutdown = Mock()

    assert proxy.dispatch_result_task() is False
    error = proxy._set_fatal_error.call_args.args[0]
    assert "closed before proxy shutdown" in str(error)
    proxy.pre_shutdown.assert_called_once_with()


def test_sync_bounded_send_returns_when_peer_is_unavailable():
    queue = object.__new__(ZeroMqQueue)
    queue.name = "unavailable"
    queue.setup_lazily = Mock()
    queue._check_thread_safety = Mock()
    queue._prepare_data = Mock(return_value=b"payload")
    queue._send_data = Mock(side_effect=zmq.Again())

    assert queue.put_bounded("message", timeout=0) is False


def test_queue_close_sets_finite_linger():
    class FakeSocket:
        def __init__(self):
            self.options = []
            self.closed = False

        def __bool__(self):
            return True

        def setsockopt(self, option, value):
            self.options.append((option, value))

        def close(self):
            self.closed = True

    class FakeContext:
        def __init__(self):
            self.terminated = False

        def __bool__(self):
            return True

        def term(self):
            self.terminated = True

    queue = object.__new__(ZeroMqQueue)
    socket = FakeSocket()
    context = FakeContext()
    queue.socket = socket
    queue.context = context

    queue.close(linger_ms=250)

    assert socket.options == [(zmq.LINGER, 250)]
    assert socket.closed
    assert context.terminated
