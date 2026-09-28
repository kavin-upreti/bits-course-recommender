"""Provider-neutral chat call with fallback across free providers.

LLM_PROVIDERS in .env lists provider:model pairs, tried in order:
    LLM_PROVIDERS=groq:llama-3.3-70b-versatile,openrouter:<model>:free,gemini:gemini-3.5-flash
A rate limit, outage or provider error moves to the next provider. When every provider is rate-limited, the whole
list is retried after 2 s, 4 s, 8 s, then LLMUnavailable. All three speak the OpenAI chat API (Gemini through its
OpenAI-compatible endpoint), so one plain httpx call serves them. This is the only file that knows about providers.

Messages use our own format:
    {"role": "user", "text": ...}
    {"role": "assistant", "text": ..., "tool_calls": [ToolCall], "raw": (provider, message) or None}
    {"role": "tool", "results": [{"id", "name", "result"}]}
why "raw": Gemini's thinking models sign their tool calls (extra_content.google.thought_signature) and reject a
history without the signature, so a Gemini turn goes back to Gemini exactly as it came.
"""
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field

import httpx

from . import config

logger = logging.getLogger(__name__)
RETRY_STATUS = {429, 500, 502, 503, 504}  # rate limit, temporary server errors
PROVIDERS = {  # provider -> (chat completions URL, env var with the key)
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "GEMINI_API_KEY"),
}
# Gemini accepts this in place of a real signature for tool calls it didn't write (e.g. Groq's, before a fallback)
SKIP_SIGNATURE = {"google": {"thought_signature": "skip_thought_signature_validator"}}


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
    raw: object = None                          # (provider, message) to send back as-is (see module docstring)


def providers() -> list[tuple[str, str]]:
    """[(provider, model)] from LLM_PROVIDERS."""
    pairs = []
    for item in filter(None, (part.strip() for part in os.environ.get("LLM_PROVIDERS", "").split(","))):
        name, _, model = item.partition(":")  # why partition: OpenRouter model names contain ":" ("...:free")
        if name not in PROVIDERS or not model:
            raise LLMError(f"LLM_PROVIDERS: unknown entry {item!r} (use groq:, openrouter: or gemini:).")
        pairs.append((name, model))
    if not pairs:
        raise LLMError("LLM_PROVIDERS is not set (see .env.example).")
    return pairs


def _openai_messages(provider: str, system: str, messages: list[dict]) -> list[dict]:
    """Neutral messages -> OpenAI chat messages (one "tool" message per result, in call order)."""
    converted = [{"role": "system", "content": system}]
    for message in messages:
        if message["role"] == "user":
            converted.append({"role": "user", "content": message["text"]})
        elif message["role"] == "assistant":
            raw = message.get("raw")
            if raw and raw[0] == provider == "gemini":  # why only Gemini: Groq rejects its own extra fields (reasoning)
                converted.append(raw[1])
                continue
            turn = {"role": "assistant", "content": message.get("text") or ""}
            if message.get("tool_calls"):
                turn["tool_calls"] = [{"id": call.id, "type": "function",
                                       "function": {"name": call.name, "arguments": json.dumps(call.args)},
                                       **({"extra_content": SKIP_SIGNATURE} if provider == "gemini" else {})}
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


def _send(provider: str, model: str, system: str, messages: list[dict], tools: list[dict] | None) -> LLMResponse:
    """One call to one provider, no retries."""
    url, key_var = PROVIDERS[provider]
    body: dict = {"model": model, "messages": _openai_messages(provider, system, messages), "temperature": config.LLM_TEMPERATURE}
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
    try:
        data = response.json()
        message = data["choices"][0]["message"]
    except (ValueError, KeyError, IndexError, TypeError) as error:
        # why: OpenRouter can answer 200 with {"error": {...}} (an upstream rate limit); that's "try the next one"
        raise ProviderBusy(f"{provider}: no reply in the response: {response.text[:200]}") from error
    calls = [ToolCall(call.get("id") or f"call_{uuid.uuid4().hex[:8]}", call["function"]["name"],
                      _parse_args(call["function"].get("arguments")))
             for call in message.get("tool_calls") or []]
    usage = data.get("usage") or {}
    return LLMResponse(text=message.get("content") or None, tool_calls=calls, raw=(provider, message),
                       usage={"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)})


def chat(system: str, messages: list[dict], tools: list[dict] | None) -> LLMResponse:
    """Send the conversation; tools=None makes the model answer in text. Providers are tried in LLM_PROVIDERS order;
    a provider without a key is skipped, one that errors is dropped for this call, a busy one is retried after
    2 s / 4 s / 8 s together with the others."""
    candidates = [(name, model) for name, model in providers() if os.environ.get(PROVIDERS[name][1])]
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
