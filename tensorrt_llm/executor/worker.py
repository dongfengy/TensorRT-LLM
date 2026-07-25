import gc
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import List, Optional

import zmq

from tensorrt_llm.logger import logger

from .. import serialization
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
from .postproc_worker import PostprocWorkerConfig
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


_POSTPROC_WORKER_COMMAND = """
import os, signal, sys
for _sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(_sig, signal.SIG_DFL)
signal.pthread_sigmask(signal.SIG_SETMASK, ())
sys.path[:] = sys.argv[1:]
from tensorrt_llm import serialization
from tensorrt_llm.executor.postproc_worker import PostprocWorker
_in, _outs, _tokenizer, _hook = serialization.load(sys.stdin.buffer)
_worker = PostprocWorker(
    _in, _outs, tokenizer_dir=_tokenizer,
    record_creator=PostprocWorker.default_record_creator,
    post_processor_hook=_hook)
try:
    _worker._pull_pipe.setup_lazily()
    for _pipe in _worker._push_pipes:
        _pipe.setup_lazily()
    os.write(0, b"R")
    _worker.start()
finally:
    for _pipe in [_worker._pull_pipe, *_worker._push_pipes]:
        _pipe.close()
"""


def _stop_postproc_workers(processes, grace):
    pending = list(processes)
    for method, timeout in ((None, grace), ("terminate", 10), ("kill", 10)):
        pending = [process for process in pending if process.poll() is None]
        if not pending:
            return
        if method:
            for process in pending:
                getattr(process, method)()
        deadline = time.monotonic() + timeout
        while pending and time.monotonic() < deadline:
            time.sleep(0.05)
            pending = [process for process in pending if process.poll() is None]
    pending = [process for process in pending if process.poll() is None]
    if pending:
        logger.error(f"Could not reap postprocessing workers: "
                     f"{[process.pid for process in pending]}")


def _start_postproc_workers(result_queues, result_addrs, config):
    processes, controls = [], []
    env, _ = split_mpi_env()
    env["TLLM_DISABLE_MPI"] = "1"
    executable = os.path.abspath(sys.executable)
    argv = [
        executable, "-I", "-S", "-c", _POSTPROC_WORKER_COMMAND,
        str(Path(__file__).resolve().parents[2]),
        *[p for p in sys.path if p and os.path.isabs(p)]
    ]
    try:
        for queue in result_queues:
            parent, child = socket.socketpair()
            controls.append(parent)
            try:
                # Popen closes inherited FDs in its native close-FDs/exec path;
                # no Python or inherited MPI runtime runs before exec.
                process = subprocess.Popen(argv,
                                           env=env,
                                           stdin=child,
                                           close_fds=True)  # nosec B603
                processes.append(process)
            finally:
                child.close()
            parent.sendall(
                serialization.dumps((
                    queue.address,
                    list(result_addrs),
                    str(config.postprocess_tokenizer_dir)
                    if config.postprocess_tokenizer_dir else None,
                    config.post_processor_hook,
                )))
            logger.info(f"Launched postprocessing worker pid={process.pid}")
        deadline = time.monotonic() + float(
            os.getenv("TLLM_POSTPROC_WORKER_READY_TIMEOUT", "300"))
        for control in controls:
            control.settimeout(max(0.001, deadline - time.monotonic()))
            if control.recv(1) != b"R":
                raise RuntimeError("Postprocessor exited before READY")
        logger.info("All postprocessing workers are ready")
        return processes
    except BaseException:
        _stop_postproc_workers(processes, 0)
        raise
    finally:
        for control in controls:
            control.close()


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
    result_queues: Optional[List[IpcQueue]] = None
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
        # Signal the dispatcher thread in every frontend proxy to quit
        if result_queue is not None:
            result_queue.put(None)
        elif frontend_result_queues is not None:
            for q in frontend_result_queues:
                q.put(None)
        else:
            assert result_queues is not None
            if not postproc_workers:
                return
            for q in result_queues:
                q.put(None)

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

    postproc_workers = []
    with worker:
        try:
            worker.block_subordinates()

            if is_leader:
                if postproc_worker_config.enabled:
                    assert result_queues is not None
                    worker.set_postproc_queues(result_queues)
                    result_addrs = (multi_frontend_addrs
                                    if multi_frontend_addrs is not None else
                                    [worker_queues.result_queue_addr])
                    try:
                        postproc_workers = _start_postproc_workers(
                            result_queues, result_addrs, postproc_worker_config)
                    except Exception as e:
                        if not worker_init_status_queue.notify_with_retry(
                            (e, traceback.format_exc())):
                            logger.error(
                                "Failed to report postprocessor startup error")
                        raise
                elif frontend_result_queues is not None:
                    worker.set_frontend_result_queues(frontend_result_queues)
                else:
                    worker.set_result_queue(result_queue)

                # Send ready signal with confirmation
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
                _stop_postproc_workers(postproc_workers, 10)

        except GenerationExecutorWorker.WorkerExit as e:
            # This will capture by the with-statement and exit normally.
            raise e

        except Exception as e:  # other critical errors
            if is_leader:
                notify_proxy_threads_to_quit()
            logger.error(traceback.format_exc())
            # This will be captured by mpi4py and handled by future.done_callback
            raise e
        finally:
            if is_leader:
                _stop_postproc_workers(postproc_workers, 0)
