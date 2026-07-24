(post-processor-hook)=

# Post-Processing Hook

`trtllm-serve` supports a user-supplied **post-processing hook**: a native, per-request seam that runs
on each generated output *after* detokenization and *before* the per-endpoint response formatter. It
lets a deployment rewrite, redact, suppress, or terminate model output — including stateful logic that
spans the chunks of a streamed response — without modifying TensorRT LLM source.

The hook is a plain Python callable class supplied by import path, mirroring `--custom_tokenizer`. It
is owned by the `LLM` instance (and built once in each post-processing worker thread when those are
enabled) and invoked once per output, per streaming chunk (plus a final call), so it can hold its own
per-request state. Independent `LLM` instances in one process each own separate hook instances.

```{note}
This feature is a prototype and its interface may change in a future release.
```

For the interface definitions referenced below, see
[`tensorrt_llm/executor/postprocessor_hook.py`](../../../tensorrt_llm/executor/postprocessor_hook.py).

## Enabling the hook

Pass the dotted import path of your hook class to `--post_processor_hook`:

```bash
trtllm-serve <model> --post_processor_hook my_pkg.guardrail.MyPostProcessorHook
```

Equivalently, set it in a YAML config passed via `--extra_llm_api_options`:

```yaml
post_processor_hook: my_pkg.guardrail.MyPostProcessorHook
```

The class must be:

- **Importable** — installed (`pip install`) or on `PYTHONPATH` when the server (and its
  rank-0 executor and post-processing worker threads, if any) start.
- **No-argument-constructible** — `__init__` takes no required arguments and is the one-time setup
  point for each hook instance. The hook object itself does not need to be picklable.

The hook works with or without parallel post-processing. When `--num_postprocess_workers` is greater
than zero, it selects that many supervised threads in the rank-0 executor process. It does not create
an out-of-process isolation boundary.

## The hook interface

A hook implements a single method, `__call__(self, chunk) -> verdict`:

```python
from tensorrt_llm.executor.postprocessor_hook import (
    PostProcessorHookChunk,
    PostProcessorHookVerdict,
    emit,
    suppress,
    terminate,
)


class MyPostProcessorHook:
    def __call__(self, chunk: PostProcessorHookChunk) -> PostProcessorHookVerdict:
        return emit(chunk.text_diff)  # pass through unchanged
```

### `PostProcessorHookChunk`

The payload handed to the hook for one output chunk — `request_id`, `output_index`, `text_diff`,
`text`, `token_ids_diff`, `is_final`, `aborted`, and `streaming`. See the `PostProcessorHookChunk`
dataclass in [`postprocessor_hook.py`](../../../tensorrt_llm/executor/postprocessor_hook.py)
for the authoritative field-by-field descriptions.

### Verdicts

Return one of the following helpers from `__call__`:

| Helper | Effect |
|--------|--------|
| `emit(text)` | Emit `text` for this chunk. Use it to pass output through unchanged (`emit(chunk.text_diff)`) or to rewrite/redact it. Affects the **text** channel only — it does not synthesize matching `token_ids` (see *Text vs. token ids* below). |
| `suppress()` | Withhold this chunk entirely across **all** client-visible channels — text, `token_ids`, and `logprobs` (so `detokenize=false` token output is withheld too). |
| `terminate(reason)` | Stop the stream for this request, withholding the terminating chunk on all channels. `reason` is surfaced as the response `stop_reason`, and the engine request is cancelled. |

Verdicts are **per chunk**: `suppress()` withholds the current chunk, and `terminate()` stops generation while keeping the chunks already emitted before it. This is consistent across streaming and non-streaming — a non-streaming response contains exactly the content the hook emitted before the first `suppress`/`terminate`, i.e. the same content a streaming client would have received. A hook that must withhold the *entire* output (all-or-nothing) should `suppress()` from the first chunk (it sees every chunk) rather than emitting and then terminating.

## Usage examples

### Rewrite output

A stateless hook that upper-cases every chunk:

```python
from tensorrt_llm.executor.postprocessor_hook import PostProcessorHookChunk, PostProcessorHookVerdict, emit


class UpperCaseHook:
    def __call__(self, chunk: PostProcessorHookChunk) -> PostProcessorHookVerdict:
        return emit(chunk.text_diff.upper())
```

### Stateful guardrail that terminates on a banned phrase

This hook accumulates text per request (keyed by `request_id`), stops the stream as soon as a banned
phrase appears, and releases its state when the request finishes:

```python
from tensorrt_llm.executor.postprocessor_hook import (
    PostProcessorHookChunk, PostProcessorHookVerdict, emit, terminate,
)


class BannedPhraseGuard:
    _BANNED = ("forbidden phrase",)

    def __init__(self):
        # Per-request accumulators owned entirely by the hook.
        self._buffers: dict[int, str] = {}

    def __call__(self, chunk: PostProcessorHookChunk) -> PostProcessorHookVerdict:
        buffer = self._buffers.get(chunk.request_id, "") + chunk.text_diff.lower()
        self._buffers[chunk.request_id] = buffer

        if any(phrase in buffer for phrase in self._BANNED):
            self._buffers.pop(chunk.request_id, None)
            return terminate("banned_phrase")

        if chunk.is_final:
            self._buffers.pop(chunk.request_id, None)
        return emit(chunk.text_diff)
```

### Suppress output

A hook that withholds all client-visible text:

```python
from tensorrt_llm.executor.postprocessor_hook import PostProcessorHookChunk, PostProcessorHookVerdict, suppress


class SuppressHook:
    def __call__(self, chunk: PostProcessorHookChunk) -> PostProcessorHookVerdict:
        return suppress()
```

## Per-request state

The hook instance is owned by the `LLM` (each rank-0 post-processing worker thread constructs its own
tokenizer and hook when parallel post-processing is enabled) and shared across all requests that
worker handles. Any per-request state must be keyed by `chunk.request_id` and released when
`chunk.is_final` is seen (or after a `terminate`). Hook instance attributes are not shared between
workers or separate `LLM` instances. The worker threads do share rank 0's process-global state,
including module globals, class variables, environment changes, and native libraries; access to
such state must be thread-safe, and hooks must not rely on process isolation. All chunks of a single
request are routed to the same worker, keeping its instance-local per-request state consistent.

Engine-level batching is transparent to the hook: even when requests are batched together in the
engine, the hook is invoked **once per request** (per output, per chunk) — there is no batched-call
form — so keying state on `chunk.request_id` is sufficient to keep concurrent requests isolated.

## Supported endpoints and limitations

- **Endpoints**: `chat/completions` and `completions`, both streaming and non-streaming. The hook is
  expected to apply to the `responses` endpoint for non-harmony models as well (shared detokenization
  path), though that endpoint is not covered by the current end-to-end tests.
- **Not client-bypassable**: the hook is a server-side guardrail, so it runs on every response even
  when a `completions` request sets `detokenize=false`. The server detokenizes for the hook regardless;
  the `detokenize` flag still controls only the returned channel (text vs. token ids). A
  `suppress`/`terminate` verdict withholds **all** client-visible channels — text, `token_ids`, and
  `logprobs` — on both the streaming and non-streaming paths, so a client cannot recover withheld
  content through any channel.
- **Requires a tokenizer**: the hook needs detokenized text to inspect, so `--post_processor_hook` combined
  with `skip_tokenizer_init` is rejected at startup rather than silently disabled.
- **harmony / gpt-oss models**: not supported. Because the harmony output path is reconstructed from
  raw token ids, it would bypass the text-based hook, so the server fails fast at startup when
  `--post_processor_hook` is combined with a harmony model.
- **Disaggregated serving**: the context and generation servers are separate processes, each running
  the hook on its own phase under a different `request_id`; per-request state cannot be correlated
  across the two. A `terminate` on one phase does not propagate to the other.
- **Text vs. token ids**: `emit` rewrites the **text** channel only — it does not rewrite the underlying
  `token_ids`/`logprobs`, so a client reading both should expect them to diverge. (`suppress`/`terminate`
  withhold all channels, so they do not diverge.)
- **Reasoning / tool parsing**: the hook runs before the reasoning and tool-call parsers. A hook that
  rewrites or suppresses text may desync those parsers; prefer `terminate`, or apply such hooks to
  plain-text requests.
- **Ordinary hook errors fail closed per request**: if the hook raises a Python `Exception`, the
  request fails with an error rather than serving the un-vetted chunk (the server and other requests
  stay alive). A returned verdict with an unknown action is rejected the same way. `BaseException`
  subclasses and native failures are engine-fatal instead.
- **Native failure isolation**: parallel post-processing workers are threads in rank 0, not child
  processes. CPU-bound pure-Python hooks share the Python GIL. This design avoids forking after
  MPI/CUDA initialization, but a native crash (for example, a segmentation fault in hook or
  tokenizer native code) can terminate rank 0. Pending requests are then failed by the executor
  proxy instead of waiting indefinitely.
- **`n` > 1 / beam search**: `emit` and `suppress` act per output sequence, but `terminate` cancels the
  **whole** request (all sequences), because the engine request is the unit of cancellation — a
  `terminate` on one candidate ends the others too. Hooks needing per-sequence state should key on
  `(request_id, output_index)`.

## Tests

The hook's unit and end-to-end tests double as runnable usage examples:

```bash
# Unit tests for the hook contract (rewrite / suppress / terminate, per-request state, loader)
pytest tests/unittest/executor/test_postprocessor_hook.py -v
```

- End-to-end serving tests across endpoints, streaming modes, and worker-pool settings:
  `tests/unittest/llmapi/apps/_test_openai_post_processor.py`.
- The deterministic sample hooks used by those tests:
  `tests/unittest/llmapi/apps/_postproc_hook_samples.py`.
