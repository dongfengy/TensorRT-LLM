import gc
import json
import math
import os
import selectors
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import List, Optional, Sequence

import zmq

from tensorrt_llm.logger import logger

from .._utils import mpi_comm, mpi_rank, print_all_stacks
from ..bindings import executor as tllm
from ..llmapi.llm_args import BaseLlmArgs
from ..llmapi.mpi_session import set_mpi_session_cpp, split_mpi_env
from ..llmapi.tokenizer import TokenizerBase
from ..llmapi.tracer import VizTracer, set_global_tracer
from ..llmapi.utils import ManagedThread, logger_debug, print_traceback_on_error
from ..sampling_params import BatchedLogitsProcessor
from .base_worker import BaseWorker, _init_hf_modules
from .ipc import FusedIpcQueue, IpcQueue
from .postproc_worker import (POSTPROC_WORKER_READY_FD_ENV,
                              POSTPROC_WORKER_SUBPROCESS_COMMAND,
                              PostprocWorkerConfig, encode_postproc_worker_spec)
from .request import CancellingRequest, GenerationRequest
from .rpc_worker_mixin import RpcWorkerMixin
from .utils import (ErrorResponse, IntraProcessQueue, RequestError,
                    WorkerCommIpcAddrs)
from .worker_process_monitor import capture_worker_process_identity

__all__ = [
    "GenerationExecutorWorker",
]


class GenerationExecutorWorker(RpcWorkerMixin, BaseWorker):

    def __init__(
        self,
        engine: Path,
        executor_config: Optional[tllm.ExecutorConfig] = None,
        batched_logits_processor: Optional[BatchedLogitsProcessor] = None,
        postproc_worker_config: Optional[PostprocWorkerConfig] = None,
        is_llm_executor: Optional[bool] = None,
        hf_model_dir: Optional[Path] = None,
        tokenizer: Optional[TokenizerBase] = None,
        llm_args: Optional[BaseLlmArgs] = None,
        rpc_addr: Optional[str] = None,
        hmac_key: bytes = b"",
    ) -> None:
        super().__init__(
            engine=engine,
            executor_config=executor_config,
            batched_logits_processor=batched_logits_processor,
            postproc_worker_config=postproc_worker_config,
            is_llm_executor=is_llm_executor,
            hf_model_dir=hf_model_dir,
            tokenizer=tokenizer,
            llm_args=llm_args,
        )

        if (self.llm_args is not None
                and getattr(self.llm_args, "enable_resource_governor", False)):
            self._resource_governor_queue = IntraProcessQueue()

        self.setup_engine()

        # Setup RPC server for stats (skip init_rpc_worker to keep IPC response queue)
        # Only set up if rpc_addr is provided (for stats RPC support)
        if rpc_addr is not None:
            assert hmac_key, "hmac_key is required when rpc_addr is set"
            self.rpc_addr = rpc_addr
            self.hmac_key = hmac_key
            self.start_rpc_server()  # Reuse from RpcWorkerMixin

        self.await_response_thread = ManagedThread(
            self.await_response_task,
            error_queue=self._error_queue,
            name="await_response_thread")

    def start_thread(self, thread: ManagedThread):
        if self.engine.can_enqueue_requests() and not thread.is_alive():
            thread.start()

    def await_response_task(self) -> bool:
        return self._await_response_helper()

    def start(self):
        # Stats and KV events are now fetched on-demand via RPC,
        # so we only need to start the response thread
        self.start_thread(self.await_response_thread)

    def shutdown(self):

        if self.doing_shutdown:
            return
        else:
            self.doing_shutdown = True

        logger_debug(f'Worker {mpi_rank()} shutdown...\n', "yellow")

        if self.engine is not None:
            if self.engine.can_enqueue_requests():
                if self.await_response_thread.is_alive():
                    self.await_response_thread.stop()
                    self.await_response_thread.join()

            self.engine.shutdown()
            self.engine = None

            if self.llm_args is not None:
                assert self._executor_config is None, "An empty executor_config is expected in shutdown when LLM arguments are defined."
                if (self.llm_args.backend == "pytorch"
                        and hasattr(self, "checkpoint_loader")
                        and self.checkpoint_loader is not None):
                    self.checkpoint_loader.cleanup()
                    self.checkpoint_loader = None
            else:
                if hasattr(
                        self._executor_config, "checkpoint_loader"
                ) and self._executor_config.checkpoint_loader is not None:
                    self._executor_config.checkpoint_loader.cleanup()
                    self._executor_config.checkpoint_loader = None

        # Destroy torch distributed process groups so that NCCL communicators
        # are torn down cleanly before MPI session shutdown and process exit.
        # This is done here (not in PyExecutor.shutdown()) because the MPI
        # worker owns the process group.  In the Ray path the process group
        # belongs to RayWorkerWrapper and must not be destroyed by the engine.
        import torch.distributed
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        # Return this rank's GPU memory to the driver. Under an external MPI
        # launch (mpirun/srun, e.g. CI), the worker process is long-lived and
        # shared across successive LLM instances: a new GenerationExecutorWorker
        # is built for each LLM, but the OS process -- and with it the CUDA
        # context and PyTorch caching allocator -- persists. Setting
        # `self.engine = None` above is not enough to free the GPU: reference
        # cycles keep the model tensors alive until a later GC, and the allocator
        # holds freed blocks as "reserved" instead of returning them. Without
        # this, the previous model's ~weights-sized reservation carries into the
        # next LLM built in this process and can OOM its load (e.g. back-to-back
        # tests in one CI shard).
        gc.collect()
        torch.cuda.empty_cache()

        # Check if there are any errors from the threads before shutdown.
        self._handle_background_error()

        logger_debug(f"Worker {mpi_rank()} shutdown done.\n", "yellow")

    def block_subordinates(self):
        if self.rank != 0:
            from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
            if isinstance(self.engine, PyExecutor):
                self.engine.wait_shutdown()


class _PostprocWorkerProcess:
    """Small, thread-safe process handle for an ``os.posix_spawn`` child."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self._returncode: Optional[int] = None
        self._waitpid_lock = threading.Lock()

    def _poll_locked(self) -> Optional[int]:
        if self._returncode is not None:
            return self._returncode
        try:
            waited_pid, status = os.waitpid(self.pid, os.WNOHANG)
        except ChildProcessError:
            # A foreign SIGCHLD handler may have reaped the child. Treat it as
            # failed instead of operating on a potentially recycled PID.
            self._returncode = 255
        else:
            if waited_pid:
                self._returncode = os.waitstatus_to_exitcode(status)
        return self._returncode

    def poll(self) -> Optional[int]:
        with self._waitpid_lock:
            return self._poll_locked()

    @property
    def returncode(self) -> Optional[int]:
        return self.poll()

    def wait(self, timeout: Optional[float] = None) -> int:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            returncode = self.poll()
            if returncode is not None:
                return returncode
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out waiting for postprocessing worker "
                        f"pid={self.pid}")
                time.sleep(min(remaining, 0.05))
            else:
                time.sleep(0.05)

    def signal_process_group(self, sig: signal.Signals) -> None:
        """Signal the private process group created by ``setpgroup=0``."""
        with self._waitpid_lock:
            if self._poll_locked() is not None:
                return
            try:
                os.killpg(self.pid, sig)
            except ProcessLookupError:
                pass


def _wait_postproc_worker_processes(
    processes: Sequence[_PostprocWorkerProcess],
    timeout: float,
) -> List[_PostprocWorkerProcess]:
    deadline = time.monotonic() + timeout
    remaining_processes: List[_PostprocWorkerProcess] = []
    for process in processes:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except TimeoutError:
            remaining_processes.append(process)
    return remaining_processes


def _terminate_postproc_workers(
        processes: Sequence[_PostprocWorkerProcess]) -> None:
    """Bounded termination of every postprocessor's private process group."""
    alive = [process for process in processes if process.poll() is None]
    for process in alive:
        process.signal_process_group(signal.SIGTERM)

    alive = _wait_postproc_worker_processes(alive, timeout=10)
    for process in alive:
        process.signal_process_group(signal.SIGKILL)

    alive = _wait_postproc_worker_processes(alive, timeout=10)
    for process in alive:
        logger.error(f"Could not reap postprocessing worker pid={process.pid} "
                     "after SIGKILL")


def _wait_postproc_workers_ready(processes: Sequence[_PostprocWorkerProcess],
                                 ready_fds: Sequence[int]) -> None:
    """Wait until every clean-exec sidecar reports usable IPC sockets."""
    if len(processes) != len(ready_fds):
        raise ValueError("Each postprocessing worker needs one READY pipe")

    timeout = float(os.getenv("TLLM_POSTPROC_WORKER_READY_TIMEOUT", "300"))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(
            "TLLM_POSTPROC_WORKER_READY_TIMEOUT must be finite and positive")

    deadline = time.monotonic() + timeout
    selector = selectors.DefaultSelector()
    pending = set(ready_fds)
    try:
        for ready_fd, process in zip(ready_fds, processes):
            selector.register(ready_fd, selectors.EVENT_READ, process)

        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"{len(pending)} postprocessing worker(s) not ready within "
                    f"{timeout:.0f}s")

            for key, _ in selector.select(timeout=min(remaining, 1.0)):
                process = key.data
                ready_fd = key.fd
                if os.read(ready_fd, 1) != b"R":
                    raise RuntimeError(
                        f"Postprocessing worker pid={process.pid} exited "
                        "before signaling READY")
                selector.unregister(ready_fd)
                pending.remove(ready_fd)
                logger.info(f"Postprocessing worker pid={process.pid} is ready")

            for ready_fd in pending:
                process = selector.get_key(ready_fd).data
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Postprocessing worker pid={process.pid} exited with "
                        f"code {process.returncode} before signaling READY")

        # A worker that wrote READY and then failed during the same startup
        # window is still a startup failure, not a healthy engine.
        for process in processes:
            if process.poll() is not None:
                raise RuntimeError(
                    f"Postprocessing worker pid={process.pid} exited with "
                    f"code {process.returncode} immediately after READY")
    finally:
        selector.close()


def _open_cloexec_pipe() -> tuple[int, int]:
    """Open a CLOEXEC pipe whose ends cannot alias standard descriptors."""
    import fcntl
    pipe_fds = list(os.pipe2(os.O_CLOEXEC))
    try:
        for index, fd in enumerate(pipe_fds):
            if fd <= 2:
                normalized_fd = fcntl.fcntl(fd, fcntl.F_DUPFD_CLOEXEC, 3)
                os.close(fd)
                pipe_fds[index] = normalized_fd
        return pipe_fds[0], pipe_fds[1]
    except BaseException:
        for fd in pipe_fds:
            try:
                os.close(fd)
            except OSError:
                pass
        raise


def _close_fd(fd: int) -> None:
    try:
        os.close(fd)
    except OSError:
        pass


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        try:
            written = os.write(fd, view)
        except InterruptedError:
            continue
        if written == 0:
            raise BrokenPipeError("Could not write postprocessing worker spec")
        view = view[written:]


_SPAWN_DEFAULT_SIGNALS = tuple(
    getattr(signal, name)
    for name in ("SIGPIPE", "SIGXFZ", "SIGXFSZ", "SIGTERM", "SIGINT")
    if hasattr(signal, name))


def _absolute_python_path() -> List[str]:
    """Capture import paths explicitly because ``-I -S`` ignores ambient ones."""
    package_root = str(Path(__file__).resolve().parents[2])
    paths = [package_root]
    for path in sys.path:
        # Empty and relative entries name the current working directory. Do
        # not turn them into absolute child import roots: that would undo -I's
        # protection against cwd module shadowing.
        if path and os.path.isabs(path) and path not in paths:
            paths.append(path)
    return paths


def _spawn_postproc_worker(
    spec: bytes,
    processes: List[_PostprocWorkerProcess],
) -> int:
    """Spawn one postprocessor without forking the live MPI Python runtime."""
    spec_read_fd = spec_write_fd = -1
    ready_read_fd = ready_write_fd = -1
    child_ready_fd = -1
    process: Optional[_PostprocWorkerProcess] = None
    keep_ready_read_fd = False
    try:
        import fcntl
        spec_read_fd, spec_write_fd = _open_cloexec_pipe()
        ready_read_fd, ready_write_fd = _open_cloexec_pipe()
        # Reserve a collision-free dynamic descriptor for READY in the child.
        child_ready_fd = fcntl.fcntl(ready_write_fd, fcntl.F_DUPFD_CLOEXEC, 3)
        child_env, _ = split_mpi_env()
        child_env["TLLM_DISABLE_MPI"] = "1"
        child_env[POSTPROC_WORKER_READY_FD_ENV] = str(child_ready_fd)
        executable = os.path.abspath(sys.executable)
        argv = [
            executable,
            "-I",
            "-S",
            "-c",
            POSTPROC_WORKER_SUBPROCESS_COMMAND,
            # This value is deliberately not taken from PYTHONPATH: isolated
            # mode ignores ambient Python configuration.
            json.dumps(_absolute_python_path()),
        ]
        file_actions = [
            (os.POSIX_SPAWN_DUP2, spec_read_fd, 0),
            (os.POSIX_SPAWN_DUP2, ready_write_fd, child_ready_fd),
            (os.POSIX_SPAWN_CLOSE, spec_read_fd),
            (os.POSIX_SPAWN_CLOSE, spec_write_fd),
            (os.POSIX_SPAWN_CLOSE, ready_read_fd),
            (os.POSIX_SPAWN_CLOSE, ready_write_fd),
        ]
        pid = os.posix_spawn(
            executable,
            argv,
            child_env,
            file_actions=file_actions,
            setpgroup=0,
            setsigmask=(),
            setsigdef=_SPAWN_DEFAULT_SIGNALS,
        )
        process = _PostprocWorkerProcess(pid)
        # Publish ownership before any parent-side operation can fail. The
        # worker_main finally block will then clean up this process group after
        # reporting the startup error to the proxy.
        processes.append(process)
        _write_all(spec_write_fd, spec)
        keep_ready_read_fd = True
        return ready_read_fd
    finally:
        _close_fd(spec_read_fd)
        _close_fd(spec_write_fd)
        if not keep_ready_read_fd:
            _close_fd(ready_read_fd)
        _close_fd(ready_write_fd)
        _close_fd(child_ready_fd)


def _start_postproc_workers(
    result_queues: Sequence[FusedIpcQueue],
    proxy_result_addrs: Sequence[tuple[str, Optional[bytes]]],
    config: PostprocWorkerConfig,
    processes: List[_PostprocWorkerProcess],
) -> None:
    """Start isolated postprocessors with a direct POSIX spawn/exec."""
    ready_fds: List[int] = []
    try:
        for worker_id, result_queue in enumerate(result_queues):
            ready_fd = _spawn_postproc_worker(
                encode_postproc_worker_spec(
                    result_queue.address,
                    list(proxy_result_addrs),
                    config.postprocess_tokenizer_dir,
                    config.post_processor_hook,
                ),
                processes,
            )
            ready_fds.append(ready_fd)
            process = processes[-1]
            logger.info(f"Launched postprocessing worker {worker_id} "
                        f"(pid={process.pid})")

        _wait_postproc_workers_ready(processes, ready_fds)
    finally:
        for fd in ready_fds:
            _close_fd(fd)


def _reap_postproc_workers(processes: Sequence[_PostprocWorkerProcess]) -> None:
    """Give sidecars a bounded graceful exit after their feed sentinel."""
    if _wait_postproc_worker_processes(processes, timeout=10):
        _terminate_postproc_workers(processes)


@print_traceback_on_error
def worker_main(
    engine: Path,
    worker_queues: WorkerCommIpcAddrs,
    log_level: str,
    executor_config: Optional[tllm.ExecutorConfig] = None,
    batched_logits_processor: Optional[BatchedLogitsProcessor] = None,
    worker_cls: type = GenerationExecutorWorker,
    tracer_init_kwargs: Optional[dict] = None,
    _torch_model_class_mapping: Optional[dict] = None,
    postproc_worker_config: Optional[PostprocWorkerConfig] = None,
    ready_signal: Optional[str] = None,
    is_llm_executor: Optional[
        bool] = True,  # whether it's the main executor instance
    hf_model_dir: Optional[Path] = None,
    tokenizer: Optional[TokenizerBase] = None,
    llm_args: Optional[BaseLlmArgs] = None,
    rpc_addr: Optional[str] = None,
    hmac_key: bytes = b"",
) -> None:

    def _print_stacks():
        counter = 0
        while True:
            time.sleep(print_stacks_period)
            counter += 1
            logger.error(f"Printing stacks {counter} times")
            print_all_stacks()

    print_stacks_period = int(
        os.getenv("TRTLLM_WORKER_PRINT_STACKS_PERIOD", "-1"))
    if print_stacks_period > 0:
        print_stacks_thread = threading.Thread(target=_print_stacks,
                                               daemon=True)
        print_stacks_thread.start()

    mpi_comm().barrier()

    if llm_args is not None and llm_args.env_overrides:
        # this is needed because MPI_Init seems to cache the env at import time.
        # The cached env snapshot is used to spawn workers.
        # Any env overrides to the main process after tensorrt_llm import
        # may not get reflected in the spawned worker process, no matter how early,
        # unless we update it explicitly here.
        os.environ.update(llm_args.env_overrides)

    if llm_args is not None and llm_args.trust_remote_code:
        _init_hf_modules()

    logger_debug(f"Worker {mpi_rank()} entering worker_main...\n", "green")

    result_queue: Optional[IpcQueue] = None
    result_queues: Optional[List[FusedIpcQueue]] = None
    resource_governor_queue: Optional[IpcQueue] = None

    postproc_worker_config = postproc_worker_config or PostprocWorkerConfig()

    is_leader: bool = mpi_rank() == 0
    # Multi-frontend serving: the worker binds the request ingress (PULL)
    # and pushes responses to per-frontend result lanes.
    multi_frontend_addrs = worker_queues.frontend_result_queue_addrs
    frontend_result_queues: Optional[List[FusedIpcQueue]] = None
    if tracer_init_kwargs is not None and is_leader:
        tracer = VizTracer(**tracer_init_kwargs)
        tracer.register_exit()
        tracer.start()
        set_global_tracer(tracer)

    if _torch_model_class_mapping is not None:
        from tensorrt_llm._torch.models.modeling_auto import MODEL_CLASS_MAPPING
        MODEL_CLASS_MAPPING.update(**_torch_model_class_mapping)

    set_mpi_session_cpp(mpi_comm())

    if is_leader:
        # Only set the log level for the leader process, the other processes will
        # inherit the log level from "TLLM_LOG_LEVEL" environment variable
        logger.set_level(log_level)
        request_queue = IpcQueue(worker_queues.request_queue_addr,
                                 is_server=multi_frontend_addrs is not None,
                                 socket_type=zmq.PULL if multi_frontend_addrs
                                 is not None else zmq.PAIR,
                                 name="worker_request_queue")
        worker_init_status_queue = IpcQueue(
            worker_queues.worker_init_status_queue_addr,
            is_server=False,
            socket_type=zmq.DEALER,
            name="worker_init_status_queue")
        resource_governor_queue = IpcQueue(
            worker_queues.resource_governor_queue_addr,
            is_server=False,
            name="worker_resource_governor_queue"
        ) if worker_queues.resource_governor_queue_addr else None

        if postproc_worker_config.enabled:
            # IPC queues for sending inputs to the postprocess parallel
            # processes, each one is a PAIR zmq socket
            result_queues = [
                FusedIpcQueue(is_server=True,
                              fuse_message=False,
                              name=f"postprocess_{i}_feedin_queue")
                for i in range(postproc_worker_config.num_postprocess_workers)
            ]
        elif multi_frontend_addrs is not None:
            # One PUSH lane per frontend (see base_worker._send_rsp).
            frontend_result_queues = [
                FusedIpcQueue(addr,
                              is_server=False,
                              fuse_message=False,
                              socket_type=zmq.PUSH,
                              name=f"worker_result_queue_{i}")
                for i, addr in enumerate(multi_frontend_addrs)
            ]
        else:
            # IPC queue for sending results back to the proxy, and let the
            # Proxy process to handle the postprocess
            result_queue = FusedIpcQueue(worker_queues.result_queue_addr,
                                         is_server=False,
                                         fuse_message=False,
                                         name="worker_result_queue")

    def notify_proxy_threads_to_quit():
        nonlocal postproc_shutdown_requested
        # Signal the dispatcher thread in every frontend proxy to quit
        if result_queue is not None:
            result_queue.put(None)
        elif frontend_result_queues is not None:
            for q in frontend_result_queues:
                q.put(None)
        else:
            assert result_queues is not None
            if postproc_shutdown_requested or not postproc_workers_ready:
                return
            postproc_shutdown_requested = True
            for q in result_queues:
                q.put(None)

    postproc_worker_processes: List[_PostprocWorkerProcess] = []
    postproc_workers_ready = False
    postproc_shutdown_requested = False

    # Error handling in the Worker/MPI process
    #   1. During Executor initialization, the errors will be captured and
    #      send back via request_error_queue.
    #   2. During execution, the errors will be captured by ManagedThreads
    #      a) For per-request error, the error will be send back via
    #         result_queue, and eventually raised in handle_response() in
    #         the main thread.
    #      b) For system error, the error will be raised in the MPI process
    #         and handled by future.done_callback, that will propagate the
    #         error to the error_queue in the main thread.

    mpi_comm().barrier()
    worker_process_identities = mpi_comm().allgather(
        capture_worker_process_identity(mpi_rank()))
    logger_debug(f"Worker {mpi_rank()} ready to setup backend...\n", "green")

    try:
        worker: GenerationExecutorWorker = worker_cls(
            engine,
            executor_config,
            batched_logits_processor,
            postproc_worker_config=postproc_worker_config,
            is_llm_executor=is_llm_executor,
            hf_model_dir=hf_model_dir,
            tokenizer=tokenizer,
            llm_args=llm_args,
            rpc_addr=rpc_addr,
            hmac_key=hmac_key)
    except Exception as e:
        logger.error(f"Failed to initialize executor on rank {mpi_rank()}: {e}")
        logger.error(traceback.format_exc())
        logger_debug(f"error: {traceback.format_exc()}", "red")
        if is_leader:
            # Send error message with confirmation
            error_msg = (e, traceback.format_exc())
            if not worker_init_status_queue.notify_with_retry(error_msg):
                logger.error("Failed to deliver error message to proxy")
        return

    # Optionally disable GC (default: not disabled)
    if os.getenv("TRTLLM_WORKER_DISABLE_GC", "0") == "1":
        gc.disable()

    with worker:
        try:
            worker.block_subordinates()

            if is_leader:
                if postproc_worker_config.enabled:
                    assert result_queues is not None
                    worker.set_postproc_queues(result_queues)
                    proxy_result_addrs = (multi_frontend_addrs if
                                          multi_frontend_addrs is not None else
                                          [worker_queues.result_queue_addr])
                    try:
                        _start_postproc_workers(
                            result_queues,
                            proxy_result_addrs,
                            postproc_worker_config,
                            postproc_worker_processes,
                        )
                        postproc_workers_ready = True
                    except Exception as e:
                        error_msg = (e, traceback.format_exc())
                        if not worker_init_status_queue.notify_with_retry(
                                error_msg):
                            logger.error(
                                "Failed to deliver postprocessing worker "
                                "initialization error to proxy")
                        raise
                elif frontend_result_queues is not None:
                    worker.set_frontend_result_queues(frontend_result_queues)
                else:
                    worker.set_result_queue(result_queue)

                # The proxy only sees READY after every postprocessing process
                # has loaded its tokenizer/hook and connected IPC.
                ready_msg = (ready_signal, None, worker_process_identities)
                if not worker_init_status_queue.notify_with_retry(ready_msg):
                    logger.warning(
                        "Failed to deliver ready signal to proxy, continuing anyway"
                    )
                if resource_governor_queue is not None:
                    # Swap rank 0 to the proxy IPC queue after construction.
                    # The resource-governor flag is already enabled on all
                    # ranks.
                    worker.engine.set_resource_governor_queue(
                        resource_governor_queue)

                while (req := request_queue.get()) is not None:
                    if isinstance(req, CancellingRequest):
                        worker.abort_request(req.id)
                    elif isinstance(req, GenerationRequest):
                        try:
                            worker.submit(req)
                        except RequestError as e:
                            logger.error(f"submit request failed: {e}")
                            logger.error(traceback.format_exc())
                            worker._await_response_helper.temp_error_responses.put(
                                ErrorResponse(req.id, e, req.id))
                    else:
                        raise ValueError(f"Unknown request type: {type(req)}")

                notify_proxy_threads_to_quit()
                _reap_postproc_workers(postproc_worker_processes)

        except GenerationExecutorWorker.WorkerExit as e:
            if is_leader:
                notify_proxy_threads_to_quit()
                if postproc_shutdown_requested:
                    _reap_postproc_workers(postproc_worker_processes)
            # This will capture by the with-statement and exit normally.
            raise e

        except Exception as e:  # other critical errors
            if is_leader:
                notify_proxy_threads_to_quit()
            logger.error(traceback.format_exc())
            # This will be captured by mpi4py and handled by future.done_callback
            raise e

        finally:
            # Includes BaseException paths: no postprocessor process group may
            # outlive the rank-0 worker that owns its ZeroMQ feed socket.
            if is_leader:
                _terminate_postproc_workers(postproc_worker_processes)
