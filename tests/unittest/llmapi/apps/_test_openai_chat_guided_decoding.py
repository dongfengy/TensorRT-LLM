# Adapted from
# https://github.com/vllm-project/vllm/blob/aae6927be06dedbda39c6b0c30f6aa3242b84388/tests/entrypoints/openai/test_chat.py
import json
import os
import re
import tempfile

import jsonschema
import openai
import pytest
import yaml
from utils.llm_data import llm_datasets_root

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

pytestmark = pytest.mark.threadleak(enabled=False)
os.environ['TIKTOKEN_RS_CACHE_DIR'] = os.path.join(llm_datasets_root(),
                                                   'tiktoken_vocab')
os.environ['TIKTOKEN_ENCODINGS_BASE'] = os.path.join(llm_datasets_root(),
                                                     'tiktoken_vocab')

GUIDED_DECODING_REPEAT_ENV = "TRTLLM_GUIDED_DECODING_REPEAT"
GUIDED_DECODING_DEBUG_ENV = "TRTLLM_GUIDED_DECODING_DEBUG"
GUIDED_DECODING_EAGLE_ENV = "TRTLLM_GUIDED_DECODING_EAGLE"
GUIDED_DECODING_MOE_BACKEND_ENV = "TRTLLM_GUIDED_DECODING_MOE_BACKEND"
GUIDED_DECODING_SAMPLER_TYPE_ENV = "TRTLLM_GUIDED_DECODING_SAMPLER_TYPE"
GUIDED_DECODING_SAMPLING_MODE_ENV = "TRTLLM_GUIDED_DECODING_SAMPLING_MODE"
GUIDED_DECODING_PROGRESS_WIDTH = 30


def _guided_decoding_debug_enabled():
    value = os.environ.get(GUIDED_DECODING_DEBUG_ENV, "")
    return value.lower() not in ("", "0", "false", "no", "off")


def _model_dump_or_repr(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return repr(value)


def _debug_json(value):
    return json.dumps(value, default=str, indent=2)


def _guided_decoding_env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name, "")
    if value == "":
        return default
    value = value.lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def _guided_decoding_sampling_kwargs(default_temperature=None):
    mode = os.environ.get(GUIDED_DECODING_SAMPLING_MODE_ENV,
                          "baseline").lower()
    if mode in ("", "baseline"):
        if default_temperature is None:
            return {}
        return {"temperature": default_temperature}
    if mode == "greedy":
        return {"temperature": 0.0}
    if mode in ("top_p1", "top_p", "top-p", "topp"):
        return {"temperature": 1.0, "top_p": 1.0}
    raise ValueError(
        f"{GUIDED_DECODING_SAMPLING_MODE_ENV} must be baseline, greedy, or top_p1"
    )


def _print_chat_completion_debug(test_name: str, stage: str,
                                 chat_completion):
    if not _guided_decoding_debug_enabled():
        return

    choice = chat_completion.choices[0]
    message = choice.message
    print(
        f"[guided-debug] test={test_name} stage={stage} "
        f"finish_reason={choice.finish_reason} role={message.role} "
        f"content_repr={message.content!r}",
        flush=True,
    )
    print(
        f"[guided-debug] test={test_name} stage={stage} "
        f"message={_debug_json(_model_dump_or_repr(message))}",
        flush=True,
    )


def _guided_decoding_extra_llm_api_options(model_name: str):
    extra_llm_api_options_dict = {"guided_decoding_backend": "xgrammar"}
    if model_name != "openai/gpt-oss-120b":
        return extra_llm_api_options_dict

    moe_backend = os.environ.get(GUIDED_DECODING_MOE_BACKEND_ENV, "").strip()
    if moe_backend:
        extra_llm_api_options_dict["moe_config"] = {"backend": moe_backend}

    sampler_type = os.environ.get(GUIDED_DECODING_SAMPLER_TYPE_ENV,
                                  "").strip()
    if sampler_type:
        extra_llm_api_options_dict["sampler_type"] = sampler_type

    if _guided_decoding_env_bool(GUIDED_DECODING_EAGLE_ENV, True):
        extra_llm_api_options_dict["speculative_config"] = {
            "decoding_type":
            "Eagle",
            "max_draft_len":
            3,
            "speculative_model_dir":
            get_model_path("gpt_oss/gpt-oss-120b-Eagle3"),
        }

    return extra_llm_api_options_dict



@pytest.fixture(scope="module",
                params=[
                    "meta-llama/Llama-3.1-8B-Instruct",
                    "openai/gpt-oss-120b",
                    pytest.param("zai-org/GLM-5-FP8",
                                 marks=pytest.mark.skip_less_device(8)),
                ])
def model_name(request):
    return request.param


@pytest.fixture(scope="module")
def temp_extra_llm_api_options_file(model_name: str):
    temp_fd, temp_file_path = tempfile.mkstemp(
        prefix="extra_llm_api_options_", suffix=".yaml")
    os.close(temp_fd)
    try:
        extra_llm_api_options_dict = _guided_decoding_extra_llm_api_options(
            model_name)
        with open(temp_file_path, 'w') as f:
            yaml.dump(extra_llm_api_options_dict, f)
        print(
            f"[guided-config] model={model_name} "
            f"sampling_mode={os.environ.get(GUIDED_DECODING_SAMPLING_MODE_ENV, 'baseline')} "
            f"extra_llm_api_options={_debug_json(extra_llm_api_options_dict)}",
            flush=True,
        )

        yield temp_file_path
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@pytest.fixture(scope="module")
def server(model_name: str, temp_extra_llm_api_options_file: str):
    if model_name == "meta-llama/Llama-3.1-8B-Instruct":
        model_path = get_model_path("llama-3.1-model/Llama-3.1-8B-Instruct")
    elif model_name == "openai/gpt-oss-120b":
        model_path = "/tmp/models/gpt-oss-120b"
    elif model_name == "zai-org/GLM-5-FP8":
        model_path = get_model_path("GLM-5-FP8")

    args = [
        "--max_batch_size=8", "--max_seq_len=4096", "--max_num_tokens=4096",
        f"--extra_llm_api_options={temp_extra_llm_api_options_file}"
    ]

    if model_name == "zai-org/GLM-5-FP8":
        args.extend(["--tp_size=8", "--ep_size=8"])

    with RemoteOpenAIServer(model_path, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer):
    return server.get_client()


@pytest.fixture(scope="module")
def async_client(server: RemoteOpenAIServer):
    return server.get_async_client()


def _run_json_schema(client: openai.OpenAI, model_name: str):
    json_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "pattern": "^[\\w]+$"
            },
            "population": {
                # Keep numeric ranges finite so guided decoding cannot emit an
                # arbitrarily long integer and hit max_completion_tokens before
                # closing the JSON object.
                "type": "integer",
                "minimum": 0,
                "maximum": 100000000
            },
        },
        "required": ["name", "population"],
    }
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role":
            "user",
            "content":
            "Give me the information of the capital of France in the JSON format.",
        },
    ]
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=256,
        response_format={
            "type": "json",
            "schema": json_schema
        },
        **_guided_decoding_sampling_kwargs(),
    )

    message = chat_completion.choices[0].message
    _print_chat_completion_debug("json_schema", "response", chat_completion)
    assert message.content is not None
    assert message.role == "assistant"
    jsonschema.validate(json.loads(message.content), json_schema)


def _run_openai_compatible_json_schema(client: openai.OpenAI,
                                       model_name: str):
    json_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "pattern": "^[\\w]+$"
            },
            "population": {
                # Keep numeric ranges finite so guided decoding cannot emit an
                # arbitrarily long integer and hit max_completion_tokens before
                # closing the JSON object.
                "type": "integer",
                "minimum": 0,
                "maximum": 100000000
            },
        },
        "required": ["name", "population"],
    }
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role":
            "user",
            "content":
            "Give me the information of the capital of France in the JSON format.",
        },
    ]
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=256,
        response_format={
            "type": "json_schema",
            "json_schema": json_schema
        },
        **_guided_decoding_sampling_kwargs(default_temperature=0.0),
    )

    message = chat_completion.choices[0].message
    _print_chat_completion_debug("openai_compatible_json_schema", "response",
                                 chat_completion)
    assert message.content is not None
    assert message.role == "assistant"
    jsonschema.validate(json.loads(message.content), json_schema)


def _run_json_schema_user_profile(client: openai.OpenAI, model_name: str):
    json_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The full name of the user."
            },
            "age": {
                "type": "integer",
                # Keep numeric ranges finite so guided decoding cannot emit an
                # arbitrarily long integer and hit max_completion_tokens before
                # closing the JSON object.
                "minimum": 0,
                "maximum": 120,
                "description": "The age of the user, in years."
            },
        },
        "required": ["name", "age"],
    }
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role":
            "user",
            "content":
            f"Give an example JSON for an employee profile that fits this schema: {json_schema}",
        },
    ]
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=256,
        response_format={
            "type": "json",
            "schema": json_schema
        },
        **_guided_decoding_sampling_kwargs(),
    )

    message = chat_completion.choices[0].message
    _print_chat_completion_debug("json_schema_user_profile", "first_response",
                                 chat_completion)
    assert message.content is not None
    assert message.role == "assistant"
    first_json = json.loads(message.content)
    jsonschema.validate(first_json, json_schema)

    messages.extend([
        {
            "role": "assistant",
            "content": message.content,
        },
        {
            "role": "user",
            "content": "Give me another one with a different name and age.",
        },
    ])
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=256,
        response_format={
            "type": "json",
            "schema": json_schema
        },
        **_guided_decoding_sampling_kwargs(),
    )

    message = chat_completion.choices[0].message
    _print_chat_completion_debug("json_schema_user_profile", "second_response",
                                 chat_completion)
    assert message.content is not None
    assert message.role == "assistant"
    second_json = json.loads(message.content)
    jsonschema.validate(second_json, json_schema)

    assert (
        first_json["name"] != second_json["name"]
    ), "The model should have generated a different name in the second turn."
    assert (
        first_json["age"] != second_json["age"]
    ), "The model should have generated a different age in the second turn."


def _run_regex(client: openai.OpenAI, model_name: str):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "What is the capital of France?",
        },
    ]
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=256,
        response_format={
            "type": "regex",
            "regex": "(Paris|London)"
        },
        **_guided_decoding_sampling_kwargs(),
    )

    message = chat_completion.choices[0].message
    _print_chat_completion_debug("regex", "response", chat_completion)
    assert message.content is not None
    assert message.role == "assistant"
    assert re.match(r"(Paris|London)", message.content)


def _run_ebnf(client: openai.OpenAI, model_name: str):
    ebnf_grammar = """
root ::= description
city ::= "London" | "Paris" | "Berlin" | "Rome"
description ::= city " is " status
status ::= "the capital of " country
country ::= "England" | "France" | "Germany" | "Italy"
"""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful geography bot."
        },
        {
            "role": "user",
            "content": "What's the capital of France?",
        },
    ]
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=256,
        response_format={
            "type": "ebnf",
            "ebnf": ebnf_grammar
        },
        **_guided_decoding_sampling_kwargs(),
    )

    message = chat_completion.choices[0].message
    _print_chat_completion_debug("ebnf", "response", chat_completion)
    assert message.content is not None
    assert message.role == "assistant"
    assert message.content == "Paris is the capital of France"


def _run_structural_tag(client: openai.OpenAI, model_name: str):
    tool_get_current_weather = {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type":
                        "string",
                        "description":
                        "The city to find the weather for, e.g. 'San Francisco'",
                    },
                    "state": {
                        "type":
                        "string",
                        "description":
                        "the two-letter abbreviation for the state that the city is"
                        " in, e.g. 'CA' which would mean 'California'",
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit to fetch the temperature in",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city", "state", "unit"],
            },
        },
    }

    tool_get_current_date = {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": "Get the current date and time for a given timezone",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type":
                        "string",
                        "description":
                        "The timezone to fetch the current date and time for, e.g. 'America/New_York'",
                    }
                },
                "required": ["timezone"],
            },
        },
    }

    system_prompt = f"""# Tool Instructions
- Always execute python code in messages that you share.
- When looking for real time information use relevant functions if available else fallback to brave_search
You have access to the following functions:
Use the function 'get_current_weather' to: Get the current weather in a given location
{tool_get_current_weather["function"]}
Use the function 'get_current_date' to: Get the current date and time for a given timezone
{tool_get_current_date["function"]}
If a you choose to call a function ONLY reply in the following format:
<{{start_tag}}={{function_name}}>{{parameters}}{{end_tag}}
where
start_tag => `<function`
parameters => a JSON dict with the function argument name as key and function argument value as value.
end_tag => `</function>`
Here is an example,
<function=example_function_name>{{"example_name": "example_value"}}</function>
Reminder:
- Function calls MUST follow the specified format
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line
- Always add your sources when using search results to answer the user query
You are a helpful assistant."""
    user_prompt = "You are in New York. Please get the current date and time, and the weather."

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=256,
        response_format={
            "type": "structural_tag",
            "format": {
                "type":
                "triggered_tags",
                "triggers": ["<function="],
                "tags": [
                    {
                        "begin": "<function=get_current_weather>",
                        "content": {
                            "type":
                            "json_schema",
                            "json_schema":
                            tool_get_current_weather["function"]["parameters"]
                        },
                        "end": "</function>",
                    },
                    {
                        "begin": "<function=get_current_date>",
                        "content": {
                            "type":
                            "json_schema",
                            "json_schema":
                            tool_get_current_date["function"]["parameters"]
                        },
                        "end": "</function>",
                    },
                ],
            },
        },
        **_guided_decoding_sampling_kwargs(default_temperature=0.0),
    )

    message = chat_completion.choices[0].message
    _print_chat_completion_debug("structural_tag", "response", chat_completion)
    assert message.content is not None
    assert message.role == "assistant"

    match = re.search(r'<function=get_current_weather>([\S\s]+?)</function>',
                      message.content)
    params = json.loads(match.group(1))
    jsonschema.validate(params,
                        tool_get_current_weather["function"]["parameters"])

    match = re.search(r'<function=get_current_date>([\S\s]+?)</function>',
                      message.content)
    params = json.loads(match.group(1))
    jsonschema.validate(params, tool_get_current_date["function"]["parameters"])


def _guided_decoding_runners():
    return [
        ("json_schema", _run_json_schema),
        ("openai_compatible_json_schema", _run_openai_compatible_json_schema),
        ("json_schema_user_profile", _run_json_schema_user_profile),
        ("regex", _run_regex),
        ("ebnf", _run_ebnf),
        ("structural_tag", _run_structural_tag),
    ]


def _guided_decoding_repeat_count():
    repeat = int(os.environ.get(GUIDED_DECODING_REPEAT_ENV, "1"))
    assert repeat > 0, f"{GUIDED_DECODING_REPEAT_ENV} must be positive"
    return repeat


def _print_guided_decoding_progress(completed: int, total: int,
                                    iteration: int, repeat: int,
                                    test_name: str):
    filled = int(GUIDED_DECODING_PROGRESS_WIDTH * completed / total)
    bar = "#" * filled + "-" * (GUIDED_DECODING_PROGRESS_WIDTH - filled)
    print(
        f"[guided-stress {completed}/{total}] [{bar}] "
        f"iteration={iteration}/{repeat} test={test_name}",
        flush=True,
    )


def test_json_schema(client: openai.OpenAI, model_name: str):
    _run_json_schema(client, model_name)


def test_openai_compatible_json_schema(client: openai.OpenAI, model_name: str):
    _run_openai_compatible_json_schema(client, model_name)


def test_json_schema_user_profile(client: openai.OpenAI, model_name: str):
    _run_json_schema_user_profile(client, model_name)


def test_regex(client: openai.OpenAI, model_name: str):
    _run_regex(client, model_name)


def test_ebnf(client: openai.OpenAI, model_name: str):
    _run_ebnf(client, model_name)


def test_structural_tag(client: openai.OpenAI, model_name: str):
    _run_structural_tag(client, model_name)


@pytest.mark.skipif(
    GUIDED_DECODING_REPEAT_ENV not in os.environ,
    reason=f"set {GUIDED_DECODING_REPEAT_ENV}=N to run stress loop",
)
def test_guided_decoding_stress(client: openai.OpenAI, model_name: str):
    repeat = _guided_decoding_repeat_count()
    runners = _guided_decoding_runners()
    total = repeat * len(runners)
    completed = 0

    for iteration in range(1, repeat + 1):
        for test_name, runner in runners:
            completed += 1
            _print_guided_decoding_progress(completed, total, iteration,
                                            repeat, test_name)
            try:
                runner(client, model_name)
            except Exception:
                print(
                    f"[guided-stress failure] iteration={iteration}/{repeat} "
                    f"test={test_name}",
                    flush=True,
                )
                raise
