import asyncio
import json
import os
import sys
import traceback
from collections import deque
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, BinaryIO, Callable, Dict, List,
                    NamedTuple, Optional, Union)

import zmq

from .._utils import nvtx_range_debug
from ..bindings import executor as tllm
from ..llmapi.tokenizer import TransformersTokenizer, load_hf_tokenizer
from ..llmapi.utils import print_traceback_on_error
from ..logger import logger
from ..sampling_params import SamplingParams
from .ipc import ZeroMqQueue
from .postprocessor_hook import load_post_processor_hook
from .utils import ErrorResponse, bucket_responses_by_frontend, is_llm_response

if TYPE_CHECKING:
    from ..disaggregated_params import DisaggregatedParams
    from .result import (DetokenizedGenerationResultBase, GenerationResult,
                         GenerationResultBase, ResponseWrapper)

__all__ = [
    "PostprocWorker",
    "PostprocWorkerConfig",
]

POSTPROC_WORKER_READY_FD_ENV = "TLLM_POSTPROC_WORKER_READY_FD"
POSTPROC_WORKER_SUBPROCESS_COMMAND = f"""
import os
import sys

_ready_fd = int(os.environ.pop({POSTPROC_WORKER_READY_FD_ENV!r}))
_keep_fds = {{0, 1, 2, _ready_fd}}
for _fd_root in ("/proc/self/fd", "/dev/fd"):
    try:
        _open_fds = os.listdir(_fd_root)
    except OSError:
        continue
    for _fd_name in _open_fds:
        try:
            _fd = int(_fd_name)
            if _fd not in _keep_fds:
                os.close(_fd)
        except (OSError, ValueError):
            pass
    break
else:
    _open_max = int(os.sysconf("SC_OPEN_MAX"))
    os.closerange(3, _ready_fd)
    os.closerange(_ready_fd + 1, _open_max)

import json
sys.path[:] = json.loads(sys.argv[1])
del sys.argv[1:]
from tensorrt_llm.executor.postproc_worker import postproc_worker_subprocess_main
postproc_worker_subprocess_main(_ready_fd)
"""


def _encode_ipc_addr(
        address: tuple[str, Optional[bytes]]) -> dict[str, Optional[str]]:
    endpoint, hmac_key = address
    return {
        "endpoint": endpoint,
        "hmac_key": hmac_key.hex() if hmac_key is not None else None,
    }


def _decode_ipc_addr(
        encoded: dict[str, Optional[str]]) -> tuple[str, Optional[bytes]]:
    endpoint = encoded["endpoint"]
    if endpoint is None:
        raise ValueError("IPC endpoint must not be None")
    hmac_key = encoded["hmac_key"]
    return endpoint, (bytes.fromhex(hmac_key) if hmac_key is not None else None)


def encode_postproc_worker_spec(
        feedin_ipc_addr: tuple[str, Optional[bytes]],
        feedout_ipc_addrs: List[tuple[str, Optional[bytes]]],
        tokenizer_dir: Optional[str],
        post_processor_hook: Optional[str] = None) -> bytes:
    """Serialize the private launch config for a clean-exec worker.

    The ZeroMQ HMAC keys travel over the child's private stdin pipe rather
    than appearing in argv or the environment.
    """
    return json.dumps({
        "feedin_ipc_addr":
        _encode_ipc_addr(feedin_ipc_addr),
        "feedout_ipc_addrs":
        [_encode_ipc_addr(address) for address in feedout_ipc_addrs],
        "tokenizer_dir":
        str(tokenizer_dir) if tokenizer_dir is not None else None,
        "post_processor_hook":
        post_processor_hook,
    }).encode()


@dataclass(kw_only=True)
class PostprocArgs:
    first_iteration: bool = True
    num_prompt_tokens: Optional[int] = None
    tokenizer: Optional[TransformersTokenizer] = None
    ctx_usage: Optional[Any] = None


@dataclass(kw_only=True)
class PostprocParams:
    post_processor: Callable[["GenerationResultBase", PostprocArgs], Any] = None
    postproc_args: PostprocArgs = None


@dataclass
class PostprocWorkerConfig:
    ''' The config for the postprocess worker. '''
    num_postprocess_workers: int = 0
    postprocess_tokenizer_dir: Optional[str] = None
    # Dotted import path of the user post-processing hook, or
    # None. NOTE: distinct from ``PostprocParams.post_processor``, which is the
    # per-endpoint response *formatter* (a Callable), not this hook.
    post_processor_hook: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return self.num_postprocess_workers > 0


class PostprocWorker:
    '''
    The worker to postprocess the responses from the executor's await_response.
    '''

    @dataclass
    class Input:
        rsp: Union["tllm.Response", "ResponseWrapper"]

        # The information necessary for creating a GenerationResult in the first Input for each request
        sampling_params: Optional[SamplingParams] = None
        postproc_params: Optional[PostprocParams] = None
        disaggregated_params: Optional["DisaggregatedParams"] = None
        streaming: Optional[bool] = None

    class Output(NamedTuple):
        client_id: int
        res: Any
        is_final: bool
        metrics: Optional[dict[str, float]] = None
        request_perf_metrics: Any = None
        disaggregated_params: Any = None
        should_abort: bool = False
        finish_reason: Optional[str] = None
        num_generated_tokens: Optional[int] = None

    def __init__(
        self,
        pull_pipe_addr: tuple[str, Optional[bytes]],
        push_pipe_addrs: List[tuple[str, Optional[bytes]]],
        tokenizer_dir: str,
        record_creator: Callable[
            ["PostprocWorker.Input", TransformersTokenizer], Any],
        post_processor_hook: Optional[str] = None,
    ):
        '''
        Args:
            pull_pipe_addr (tuple[str, Optional[bytes]]): The address and HMAC key of the input IPC.
            push_pipe_addrs: The addresses and HMAC keys of the output IPC
                lanes, one per frontend (a single-element list in
                single-frontend mode).
            tokenizer_dir (str): The directory to load tokenizer.
            record_creator (Callable[["ResponsePostprocessWorker.Input"], Any]): A creator for creating a record for a request.
            result_handler (Optional[Callable[[GenerationResultBase], Any]]): A callback handles the final result.
            post_processor_hook (Optional[str]): Import path of the user post-processing hook; built once and threaded onto each record.
        '''

        self._records: Dict[int, GenerationResult] = {}
        self._record_creator = record_creator
        self._pull_pipe = ZeroMqQueue(address=pull_pipe_addr,
                                      is_async=True,
                                      is_server=False,
                                      name="postprocess_pull_pipe")
        self._push_pipes = [
            ZeroMqQueue(address=addr,
                        is_async=True,
                        is_server=False,
                        socket_type=zmq.PUSH,
                        name=f"postprocess_push_pipe_{i}")
            for i, addr in enumerate(push_pipe_addrs)
        ]
        self._to_stop = asyncio.Event()

        self._q = deque()

        # Load the tokenizer and share in all records
        self._tokenizer = load_hf_tokenizer(tokenizer_dir)

        # Build the user post-processing hook once, like the
        # tokenizer above; threaded onto each record in ``_handle_input``.
        self._post_processor_hook = (
            load_post_processor_hook(post_processor_hook)
            if post_processor_hook else None)

    @staticmethod
    def default_record_creator(
            inp: "PostprocWorker.Input", tokenizer: TransformersTokenizer
    ) -> "DetokenizedGenerationResultBase":
        from .result import DetokenizedGenerationResultBase
        assert inp.sampling_params is not None
        return DetokenizedGenerationResultBase(
            inp.rsp.client_id,
            sampling_params=inp.sampling_params,
            postproc_params=inp.postproc_params,
            streaming=inp.streaming,
            tokenizer=tokenizer)

    async def _handle_input(
        self, input: Union["PostprocWorker.Input", "ResponseWrapper"]
    ) -> [Any, Optional[dict[str, float]]]:
        ''' Handle a single response from await_response worker. '''
        if input.rsp.result.context_logits is not None or \
              input.rsp.result.generation_logits is not None:
            raise ValueError(
                "Context logits or generation logits are not supposed to be "
                "sent to postprocessing workers.")

        with nvtx_range_debug("handle_input",
                              color="yellow",
                              category="Postproc"):
            req_id = input.rsp.client_id
            if req_id not in self._records:
                # TODO: support variant creation later
                self._records[req_id] = self._record_creator(
                    input, self._tokenizer)
                # Thread the hook onto the record here rather than
                # via record_creator, so custom record_creators keep working.
                self._records[
                    req_id]._post_processor_hook = self._post_processor_hook
                if input.disaggregated_params is not None:
                    self._records[
                        req_id]._disaggregated_params = input.disaggregated_params

            record = self._records[req_id]
            record._handle_response(input.rsp)  # inplace
            # Left the result_handler determine the final output dtype.
            # NOTE: This will change the CompletionOutput._postprocess_result
            metrics_dict = record.metrics_dict
            perf_metrics = None
            disaggregated_params = None
            if record.outputs:
                perf_metrics = record.outputs[0].request_perf_metrics
                disaggregated_params = record.outputs[0].disaggregated_params
            if postproc_params := record.postproc_params:
                result_handler, args = postproc_params.post_processor, postproc_params.postproc_args
                args.tokenizer = self._tokenizer
                out = result_handler(record, args)
            else:
                # This should only be called in streaming mode, and each time it
                # produces a single output.
                out = record.outputs[0]

            # TODO: Keep only the diff token_ids and text in streaming mode when
            # result_handler is not set
            return out, metrics_dict, perf_metrics, disaggregated_params

    async def _batched_put(self):
        ''' Batched IPC send. '''
        async for batch in self._mainloop():
            if batch is None:
                # notify the dispatch_result coroutine in every frontend to
                # quit
                for pipe in self._push_pipes:
                    await pipe.put_async(None)
                break
            assert isinstance(batch, list)
            if len(self._push_pipes) == 1:
                await self._push_pipes[0].put_async(batch)
                continue
            for frontend_id, sub_batch in enumerate(
                    bucket_responses_by_frontend(batch, len(self._push_pipes))):
                if sub_batch:
                    await self._push_pipes[frontend_id].put_async(sub_batch)

    async def _mainloop(self):
        ''' The loop for handle_response and keep producing outputs. '''

        async def handle_single_input(inp: PostprocWorker.Input,
                                      batch: List[PostprocWorker.Output]):
            assert isinstance(
                inp, PostprocWorker.Input
            ), f"Expect PostprocWorker.Input, got {type(inp)}."
            client_id = inp.rsp.client_id
            # ErrorResponse has no 'result' attribute; pass it through
            # directly so the proxy handles it via its ErrorResponse path.
            if isinstance(inp.rsp, ErrorResponse):
                batch.append(inp.rsp)
                self._records.pop(client_id, None)
                return
            try:
                is_final = inp.rsp.result.is_final if is_llm_response(
                    inp.rsp) else True
                res, metrics, perf_metrics, disaggregated_params = await self._handle_input(
                    inp)
                record = self._records.get(client_id)
                # A `terminate` verdict forces the record done;
                # honor it so the stream stops and the record is popped without
                # waiting for the engine's own is_final.
                if record is not None and record._done:
                    is_final = True
                should_abort = record._aborted if record else False
                finish_reason = record.outputs[0].finish_reason if (
                    record and record.outputs
                ) else None  # pass this through for _handle_response
                num_generated_tokens = len(record.outputs[0].token_ids) if (
                    record and record.outputs) else None
                batch.append(
                    PostprocWorker.Output(
                        client_id=client_id,
                        res=res,
                        is_final=is_final,
                        metrics=metrics,
                        request_perf_metrics=perf_metrics,
                        disaggregated_params=disaggregated_params,
                        should_abort=should_abort,
                        finish_reason=finish_reason,
                        num_generated_tokens=num_generated_tokens,
                    ))
                if is_final:
                    self._records.pop(client_id, None)
            except Exception as e:
                logger.error(
                    f"Postprocessing error for client {client_id}: {e}\n"
                    f"{traceback.format_exc()}")
                batch.append(
                    ErrorResponse(
                        client_id=client_id,
                        error_msg=f"Postprocessing error: {e}",
                        request_id=getattr(inp.rsp, 'request_id', -1),
                    ))
                self._records.pop(client_id, None)

        while not self._to_stop.is_set():
            batch = []
            inputs: Optional[List[PostprocWorker.Input]
                             | PostprocWorker.
                             Input] = await self._pull_pipe.get_async()

            if not isinstance(inputs, list):
                inputs = [inputs]

            for inp in inputs:
                if inp is None:
                    self._to_stop.set()
                    yield None
                    break
                await handle_single_input(inp, batch)

            yield batch

    def start(self, ready_pipe: Optional[BinaryIO] = None):
        ''' Start the workflow in the current thread. '''

        async def main():
            # Connect every socket before reporting READY. The launcher must
            # not publish engine readiness while a postprocessing sidecar can
            # still fail during tokenizer, hook, or IPC initialization.
            self._pull_pipe.setup_lazily()
            for pipe in self._push_pipes:
                pipe.setup_lazily()
            if ready_pipe is not None:
                ready_pipe.write(b"R")
                ready_pipe.close()
            await asyncio.gather(self._batched_put())

        try:
            asyncio.run(main())
        except Exception as e:
            print(traceback.format_exc())
            raise e


@print_traceback_on_error
def postproc_worker_main(feedin_ipc_addr: tuple[str, Optional[bytes]],
                         feedout_ipc_addrs: List[tuple[str, Optional[bytes]]],
                         tokenizer_dir: str,
                         record_creator: Callable,
                         post_processor_hook: Optional[str] = None):
    # Pass the hook import path; PostprocWorker builds it once.
    worker = PostprocWorker(feedin_ipc_addr,
                            feedout_ipc_addrs,
                            tokenizer_dir=tokenizer_dir,
                            record_creator=record_creator,
                            post_processor_hook=post_processor_hook)
    worker.start()


def postproc_worker_subprocess_main(ready_fd: int) -> None:
    """Run one postprocessor in a fresh interpreter launched with clean env."""
    ready_pipe = os.fdopen(ready_fd, "wb", buffering=0)
    try:
        spec = json.load(sys.stdin)
        worker = PostprocWorker(
            _decode_ipc_addr(spec["feedin_ipc_addr"]),
            [
                _decode_ipc_addr(address)
                for address in spec["feedout_ipc_addrs"]
            ],
            tokenizer_dir=spec["tokenizer_dir"],
            record_creator=PostprocWorker.default_record_creator,
            post_processor_hook=spec["post_processor_hook"],
        )
        worker.start(ready_pipe=ready_pipe)
    finally:
        # close() is idempotent if start() already sent READY and closed it.
        ready_pipe.close()
