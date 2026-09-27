"""Provider-neutral chat call with fallback across free providers (todo.md section 8, extended).

LLM_PROVIDERS in .env lists provider:model pairs, tried in order:
    LLM_PROVIDERS=groq:llama-3.3-70b-versatile,openrouter:<model>:free,gemini:gemini-3.5-flash
A rate limit, outage or provider error moves to the next provider. When every provider is rate-limited, the whole
list is retried after 2 s, 4 s, 8 s, then LLMUnavailable. Groq and OpenRouter speak the OpenAI chat API (plain
httpx); Gemini uses google-genai. This is the only file that knows about providers.

Messages use our own format:
    {"role": "user", "text": ...}
    {"role": "assistant", "text": ..., "tool_calls": [ToolCall], "raw": (provider, turn) or None}
    {"role": "tool", "results": [{"id", "name", "result"}]}
why "raw": Gemini's thinking models sign their function-call parts (thought_signature) and reject a history that
drops the signature, so a Gemini turn is sent back to Gemini as it came. Other providers ignore it.
"""
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field

import httpx
from google import genai
from google.genai import errors, types

from . import config

logger = logging.getLogger(__name__)
RETRY_STATUS = {429, 500, 502, 503, 504}  # rate limit, temporary server errors
OPENAI_COMPATIBLE = {  # provider -> (chat completions URL, env var with the key)
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY"),
}
KEY_VARS = {**{name: env for name, (_, env) in OPENAI_COMPATIBLE.items()}, "gemini": "GEMINI_API_KEY"}
# Gemini accepts this in place of a real signature for function calls it didn't write (e.g. made by Groq earlier
# in the same message, before a fallback)
SKIP_SIGNATURE = b"skip_thought_signature_validator"


class LLMUnavailable(Exception):
    """Every provider is rate-limited or down, even after retrying."""


class LLMError(Exception):
    """No provider could answer for another reason (bad key, bad request, nothing configured)."""


class ProviderBusy(Exception):
    """One provider is rate-limited or temporarily down: try the next."""


@dataclass
class ToolCall:
    id: str            # generated when the provider doesn't give one
    name: str
    args: dict


@dataclass
class LLMResponse:
    text: str | None
    tool_calls: list[ToolCall]
    usage: dict = field(default_factory=dict)  # input/output token counts if available, plus the provider used
    raw: object = None                          # (provider, turn) to send back as-is (see module docstring)


def providers() -> list[tuple[str, str]]:
    """[(provider, model)] from LLM_PROVIDERS; falls back to GEMINI_MODEL alone (the old setting)."""
    spec = os.environ.get("LLM_PROVIDERS", "").strip()
    if not spec and os.environ.get("GEMINI_MODEL"):
        spec = f"gemini:{os.environ['GEMINI_MODEL']}"
    pairs = []
    for item in filter(None, (part.strip() for part in spec.split(","))):
        name, _, model = item.partition(":")  # why partition: OpenRouter model names contain ":" ("...:free")
        if name not in KEY_VARS or not model:
            raise LLMError(f"LLM_PROVIDERS: unknown entry {item!r} (use groq:, openrouter: or gemini:).")
        pairs.append((name, model))
    if not pairs:
        raise LLMError("LLM_PROVIDERS is not set (see .env.example).")
    return pairs


# ---------------------------------------------------------------- OpenAI-compatible (Groq, OpenRouter)

def _openai_messages(system: str, messages: list[dict]) -> list[dict]:
    """Neutral messages -> OpenAI chat messages (one "tool" message per result, in call order)."""
    converted = [{"role": "system", "content": system}]
    for message in messages:
        if message["role"] == "user":
            converted.append({"role": "user", "content": message["text"]})
        elif message["role"] == "assistant":
            turn = {"role": "assistant", "content": message.get("text") or ""}
            if message.get("tool_calls"):
                turn["tool_calls"] = [{"id": call.id, "type": "function",
                                       "function": {"name": call.name, "arguments": json.dumps(call.args)}}
                                      for call in message["tool_calls"]]
            converted.append(turn)
        else:
            converted += [{"role": "tool", "tool_call_id": result["id"],
                           "content": json.dumps(result["result"], separators=(",", ":"), ensure_ascii=False)}
                          for result in message["results"]]
    return converted


def _parse_args(text: str) -> dict:
    """Tool-call arguments JSON; unparseable -> an argument validation will reject (so the model sees why)."""
    try:
        args = json.loads(text or "{}")
    except ValueError:
        return {"unparseable_arguments": text}
    return args if isinstance(args, dict) else {"unparseable_arguments": text}


def _nullable(schema: dict) -> dict:
    """Copy of an object schema whose optional properties also accept null.
    why: models often send null for "leave empty", and Groq rejects the whole call if the schema doesn't allow it;
    recommender/validation.py drops nulls anyway."""
    properties = {}
    for key, prop in schema.get("properties", {}).items():
        prop = _nullable(prop) if prop.get("type") == "object" else dict(prop)
        if key not in schema.get("required", []):
            prop["type"] = [prop["type"], "null"]
            if "enum" in prop:
                prop["enum"] = [*prop["enum"], None]
        properties[key] = prop
    return {**schema, "properties": properties}


def _send_openai(provider: str, model: str, system: str, messages: list[dict], tools: list[dict] | None) -> LLMResponse:
    url, key_var = OPENAI_COMPATIBLE[provider]
    body: dict = {"model": model, "messages": _openai_messages(system, messages), "temperature": config.LLM_TEMPERATURE}
    if tools:
        body["tools"] = [{"type": "function", "function": {**tool, "parameters": _nullable(tool["parameters"])}} for tool in tools]
    try:
        response = httpx.post(url, json=body, timeout=config.LLM_TIMEOUT_SECONDS,
                              headers={"Authorization": f"Bearer {os.environ[key_var]}"})
    except httpx.TransportError as error:
        raise ProviderBusy(f"{provider}: {error}") from error
    if response.status_code in RETRY_STATUS:
        raise ProviderBusy(f"{provider} {response.status_code}: {response.text[:200]}")
    if response.status_code != 200:
        raise LLMError(f"{provider} {response.status_code}: {response.text[:300]}")
    data = response.json()
    message = data["choices"][0]["message"]
    calls = [ToolCall(call.get("id") or f"call_{uuid.uuid4().hex[:8]}", call["function"]["name"],
                      _parse_args(call["function"].get("arguments")))
             for call in message.get("tool_calls") or []]
    usage = data.get("usage") or {}
    return LLMResponse(text=message.get("content") or None, tool_calls=calls,
                       usage={"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)})


# ---------------------------------------------------------------- Gemini

_gemini_client: genai.Client | None = None


def _gemini() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _gemini_client


def _to_contents(messages: list[dict]) -> list[types.Content]:
    """Neutral messages -> Gemini contents (function calls and responses in the same order)."""
    contents = []
    for message in messages:
        if message["role"] == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=message["text"])]))
        elif message["role"] == "assistant":
            raw = message.get("raw")
            if raw and raw[0] == "gemini":
                contents.append(raw[1])
                continue
            parts = [types.Part(text=message["text"])] if message.get("text") else []
            parts += [types.Part(function_call=types.FunctionCall(id=call.id, name=call.name, args=call.args),
                                 thought_signature=SKIP_SIGNATURE)
                      for call in message.get("tool_calls", [])]
            contents.append(types.Content(role="model", parts=parts))
        else:
            parts = []
            for result in message["results"]:
                part = types.Part.from_function_response(name=result["name"], response=result["result"])
                part.function_response.id = result["id"]
                parts.append(part)
            contents.append(types.Content(role="user", parts=parts))
    return contents


def _to_tools(tools: list[dict]) -> list[types.Tool]:
    """Our JSON-schema tool list -> one Gemini Tool with a declaration per function."""
    return [types.Tool(function_declarations=[
        types.FunctionDeclaration(name=tool["name"], description=tool["description"], parameters_json_schema=tool["parameters"])
        for tool in tools
    ])]


def _send_gemini(model: str, system: str, messages: list[dict], tools: list[dict] | None) -> LLMResponse:
    generation_config = types.GenerateContentConfig(
        system_instruction=system, temperature=config.LLM_TEMPERATURE, tools=_to_tools(tools) if tools else None,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    try:
        response = _gemini().models.generate_content(model=model, contents=_to_contents(messages), config=generation_config)
    except errors.APIError as error:
        if error.code in RETRY_STATUS:
            raise ProviderBusy(f"gemini {error.code}: {error.message}") from error
        raise LLMError(f"gemini {error.code}: {error.message}") from error
    except httpx.TransportError as error:
        raise ProviderBusy(f"gemini: {error}") from error
    candidate = (response.candidates or [None])[0]
    parts = (candidate.content.parts if candidate and candidate.content else None) or []
    text = "".join(part.text for part in parts if part.text and not part.thought) or None
    calls = [ToolCall(id=part.function_call.id or f"call_{uuid.uuid4().hex[:8]}", name=part.function_call.name,
                      args=dict(part.function_call.args or {}))
             for part in parts if part.function_call]
    usage = {}
    if response.usage_metadata:
        usage = {"input_tokens": response.usage_metadata.prompt_token_count or 0,
                 "output_tokens": response.usage_metadata.candidates_token_count or 0}
    return LLMResponse(text=text, tool_calls=calls, usage=usage,
                       raw=("gemini", candidate.content) if candidate and candidate.content else None)


# ---------------------------------------------------------------- the one entry point

def _send(provider: str, model: str, system: str, messages: list[dict], tools: list[dict] | None) -> LLMResponse:
    """One call to one provider, no retries."""
    if provider == "gemini":
        return _send_gemini(model, system, messages, tools)
    return _send_openai(provider, model, system, messages, tools)


def chat(system: str, messages: list[dict], tools: list[dict] | None) -> LLMResponse:
    """Send the conversation; tools=None makes the model answer in text. Providers are tried in LLM_PROVIDERS order;
    a provider without a key is skipped, one that errors is dropped for this call, a busy one is retried after
    2 s / 4 s / 8 s together with the others."""
    candidates = [(name, model) for name, model in providers() if os.environ.get(KEY_VARS[name])]
    if not candidates:
        raise LLMError("No LLM API key is set for any provider in LLM_PROVIDERS (see .env.example).")
    failures: list[str] = []
    for attempt in range(config.LLM_MAX_RETRIES_ON_RATE_LIMIT + 1):
        busy = []
        for name, model in candidates:
            try:
                response = _send(name, model, system, messages, tools)
                response.usage["provider"] = f"{name}:{model}"
                return response
            except ProviderBusy as error:
                logger.warning("LLM provider busy: %s", error)
                busy.append((name, model))
                failures.append(str(error))
            except LLMError as error:  # why not raise: another provider may still work
                logger.warning("LLM provider failed: %s", error)
                failures.append(str(error))
        if not busy:
            raise LLMError("; ".join(failures))
        candidates = busy
        if attempt < config.LLM_MAX_RETRIES_ON_RATE_LIMIT:
            time.sleep(config.LLM_RETRY_BASE_SECONDS * 2 ** attempt)
    raise LLMUnavailable("; ".join(failures[-len(candidates):]))
