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
from tensorrt_llm.executor.utils import EngineDeadError
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
    worker._fatal_broadcast_event = None

    async def first_batch():
        return await worker._mainloop().__anext__()

    assert asyncio.run(first_batch()) is None


def test_fatal_broadcast_suppresses_shutdown_event_terminal_marker():
    worker = object.__new__(PostprocWorker)
    worker._to_stop = asyncio.Event()
    worker._shutdown_event = threading.Event()
    worker._shutdown_event.set()
    worker._fatal_broadcast_event = threading.Event()
    worker._fatal_broadcast_event.set()

    with pytest.raises(StopAsyncIteration):
        asyncio.run(worker._mainloop().__anext__())


def test_fatal_broadcast_suppresses_feed_sentinel_terminal_marker():
    class SentinelPipe:
        async def get_async_noblock(self, timeout):
            return None

    worker = object.__new__(PostprocWorker)
    worker._to_stop = asyncio.Event()
    worker._shutdown_event = threading.Event()
    worker._fatal_broadcast_event = threading.Event()
    worker._fatal_broadcast_event.set()
    worker._pull_pipe = SentinelPipe()

    with pytest.raises(StopAsyncIteration):
        asyncio.run(worker._mainloop().__anext__())


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

    all_lanes_queued = asyncio.run(worker._broadcast_fatal(ValueError("bad output")))

    assert all_lanes_queued
    fatals = [pipe.items[0][0] for pipe in worker._push_pipes]
    assert all(isinstance(fatal, PostprocWorkerFatal) for fatal in fatals)
    assert all(fatal.worker_id == 3 for fatal in fatals)
    assert all(fatal.error_type == "builtins.ValueError" for fatal in fatals)
    assert all(fatal.error_message == "bad output" for fatal in fatals)
    assert [pipe.items[0][1] for pipe in worker._push_pipes] == [1.0, 0.1, 0.1]
    assert pickle.loads(pickle.dumps(fatals[0])) == fatals[0]


def test_postproc_fatal_broadcast_reports_partial_lane_failure():
    class RecordingPipe:
        def __init__(self, queued):
            self.queued = queued

        async def put_async_bounded(self, item, timeout):
            return self.queued

    worker = object.__new__(PostprocWorker)
    worker._worker_id = 3
    worker._push_pipes = [RecordingPipe(True), RecordingPipe(False)]

    assert not asyncio.run(worker._broadcast_fatal(ValueError("bad output")))


@pytest.mark.parametrize("broadcast_result", [True, False])
def test_postproc_worker_broadcasts_fatal_before_signaling_rank_zero(broadcast_result):
    failure_event = threading.Event()
    fatal_broadcast_event = threading.Event()
    observations = []

    class FakePipe:
        def setup_lazily(self):
            pass

    worker = object.__new__(PostprocWorker)
    worker._pull_pipe = FakePipe()
    worker._push_pipes = []
    worker._fatal_broadcast_event = fatal_broadcast_event

    async def fail():
        raise RuntimeError("postprocessor failed")

    async def record_broadcast(error):
        observations.append(("broadcast", failure_event.is_set(), str(error)))
        return broadcast_result

    async def record_drain():
        observations.append(("drain", failure_event.is_set(), fatal_broadcast_event.is_set()))

    worker._batched_put = fail
    worker._broadcast_fatal = record_broadcast
    worker._drain_until_shutdown = record_drain

    with pytest.raises(RuntimeError, match="postprocessor failed"):
        worker.start(failure_event=failure_event)

    assert observations == [
        ("broadcast", False, "postprocessor failed"),
        ("drain", True, broadcast_result),
    ]
    assert failure_event.is_set()
    assert fatal_broadcast_event.is_set() is broadcast_result


def test_postproc_worker_preserves_original_error_when_fatal_broadcast_raises():
    failure_event = threading.Event()
    drained = threading.Event()

    class FakePipe:
        def setup_lazily(self):
            pass

    worker = object.__new__(PostprocWorker)
    worker._pull_pipe = FakePipe()
    worker._push_pipes = []
    worker._fatal_broadcast_event = threading.Event()

    async def fail():
        raise RuntimeError("original postprocessor failure")

    async def fail_broadcast(error):
        raise OSError("result lane closed")

    async def record_drain():
        drained.set()

    worker._batched_put = fail
    worker._broadcast_fatal = fail_broadcast
    worker._drain_until_shutdown = record_drain

    with pytest.raises(RuntimeError, match="original postprocessor failure"):
        worker.start(failure_event=failure_event)

    assert failure_event.is_set()
    assert drained.is_set()
    assert not worker._fatal_broadcast_event.is_set()


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


def test_real_postproc_fatal_handler_wakes_client_without_request_socket():
    pending_queue = Mock()
    pending_result = Mock()
    pending_result.queue = pending_queue

    proxy = object.__new__(GenerationExecutorProxy)
    proxy.workers_started = True
    proxy.doing_shutdown = False
    proxy._engine_dead = False
    proxy._fatal_error = None
    proxy._deferred_request_shutdown = False
    proxy._results = {7: pending_result}
    proxy._worker_process_monitor = Mock()
    proxy._shutdown_event = threading.Event()
    proxy.request_queue = Mock()
    proxy.mpi_session = Mock()

    proxy._handle_postproc_worker_fatal(
        PostprocWorkerFatal(
            worker_id=2,
            error_type="builtins.RuntimeError",
            error_message="postprocessor died",
            traceback="traceback text",
        )
    )

    dead_error = pending_queue.put.call_args.args[0]
    assert isinstance(dead_error, EngineDeadError)
    assert "postprocessor died" in str(dead_error)
    assert proxy.doing_shutdown
    assert proxy._deferred_request_shutdown
    proxy.request_queue.put.assert_not_called()
    proxy.request_queue.put_noblock.assert_not_called()


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


def test_launcher_fatal_pre_shutdown_skips_request_socket():
    class OrderingProxy(GenerationExecutorProxy):
        @property
        def doing_shutdown(self):
            return self._test_doing_shutdown

        @doing_shutdown.setter
        def doing_shutdown(self, value):
            if value:
                assert self._deferred_request_shutdown
            self._test_doing_shutdown = value

    proxy = object.__new__(OrderingProxy)
    proxy.workers_started = True
    proxy._test_doing_shutdown = False
    proxy._engine_dead = True
    proxy._fatal_error = RuntimeError("engine dead")
    proxy._deferred_request_shutdown = False
    proxy._worker_process_monitor = Mock()
    proxy._shutdown_event = threading.Event()
    proxy._abort_all_requests = Mock()
    proxy.request_queue = Mock()
    proxy.mpi_futures = []

    proxy.pre_shutdown()

    assert proxy.doing_shutdown
    assert proxy._deferred_request_shutdown
    proxy._worker_process_monitor.close.assert_called_once_with()
    assert proxy._shutdown_event.is_set()
    proxy._abort_all_requests.assert_not_called()
    proxy.request_queue.put_noblock.assert_not_called()


def test_launcher_completes_deferred_request_shutdown_from_owner():
    proxy = object.__new__(GenerationExecutorProxy)
    proxy._deferred_request_shutdown = True
    proxy.request_queue = Mock()

    proxy._complete_deferred_request_shutdown()

    assert not proxy._deferred_request_shutdown
    proxy.request_queue.put_noblock.assert_called_once_with(None, retry=4)


def test_launcher_normal_pre_shutdown_retains_abort_and_sentinel():
    proxy = object.__new__(GenerationExecutorProxy)
    proxy.workers_started = True
    proxy.doing_shutdown = False
    proxy._engine_dead = False
    proxy._fatal_error = None
    proxy._deferred_request_shutdown = False
    proxy._worker_process_monitor = Mock()
    proxy._shutdown_event = threading.Event()
    proxy._abort_all_requests = Mock()
    proxy.request_queue = Mock()
    proxy.mpi_futures = []

    proxy.pre_shutdown()

    assert proxy.doing_shutdown
    proxy._abort_all_requests.assert_called_once_with()
    proxy.request_queue.put_noblock.assert_called_once_with(None, retry=4)


def test_attached_frontend_fatal_pre_shutdown_skips_request_socket():
    class OrderingFrontendProxy(GenerationExecutorFrontendProxy):
        @property
        def doing_shutdown(self):
            return self._test_doing_shutdown

        @doing_shutdown.setter
        def doing_shutdown(self, value):
            if value:
                assert self._deferred_request_shutdown
            self._test_doing_shutdown = value

    proxy = object.__new__(OrderingFrontendProxy)
    proxy._test_doing_shutdown = False
    proxy._engine_dead = True
    proxy._fatal_error = RuntimeError("engine dead")
    proxy._deferred_request_shutdown = False
    proxy._abort_all_requests = Mock()

    proxy.pre_shutdown()

    assert proxy.doing_shutdown
    assert proxy._deferred_request_shutdown
    proxy._abort_all_requests.assert_not_called()


def test_attached_frontend_completes_deferred_aborts_from_owner():
    proxy = object.__new__(GenerationExecutorFrontendProxy)
    proxy._deferred_request_shutdown = True
    proxy._results = {17: Mock()}
    proxy.request_queue = Mock()

    proxy._complete_deferred_request_shutdown()

    assert not proxy._deferred_request_shutdown
    (request,) = proxy.request_queue.put_noblock.call_args.args
    assert request.id == 17
    assert proxy.request_queue.put_noblock.call_args.kwargs == {"retry": 4}


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
