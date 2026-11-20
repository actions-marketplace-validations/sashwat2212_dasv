"""Unified async OpenAI-compatible LLM client wired through BudgetGuard (ADR-005, SDD-02)."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Sequence
from typing import Any, Literal, Protocol, cast

import httpx
from pydantic import BaseModel, Field

from dasv.core.budget import BudgetGuard
from dasv.core.errors import RuntimeSandboxError

_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class LLMToolSpec(BaseModel):
    """OpenAI-compatible function tool specification."""

    name: str
    description: str
    parameters: dict[str, Any]


class LLMToolCall(BaseModel):
    """Structured tool invocation parsed from an assistant response."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict[str, Any])
    raw_arguments: str = ""


class LLMMessage(BaseModel):
    """Single message in an OpenAI Chat Completions conversation."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: list[LLMToolCall] = Field(default_factory=list[LLMToolCall])

    def to_wire_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            payload["name"] = self.name
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.raw_arguments or json.dumps(tc.arguments),
                    },
                }
                for tc in self.tool_calls
            ]
        return payload


class LLMUsage(BaseModel):
    """Token counts reported by or estimated from an inference response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMResponse(BaseModel):
    """Normalized chat completion result."""

    model: str
    content: str = ""
    tool_calls: list[LLMToolCall] = Field(default_factory=list[LLMToolCall])
    finish_reason: str | None = None
    usage: LLMUsage = Field(default_factory=LLMUsage)


class LLMClient(Protocol):
    """Narrow async LLM client protocol allowing deterministic test fakes."""

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        tools: Sequence[LLMToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> LLMResponse: ...


def _estimate_prompt_tokens(
    messages: Sequence[LLMMessage],
    tools: Sequence[LLMToolSpec] | None = None,
) -> int:
    total_chars = sum(len(m.content) + 16 for m in messages)
    for msg in messages:
        for tc in msg.tool_calls:
            total_chars += len(tc.name) + len(tc.raw_arguments or json.dumps(tc.arguments))
    if tools:
        for tool in tools:
            total_chars += len(tool.name) + len(tool.description) + len(json.dumps(tool.parameters))
    return max(1, total_chars // 3)


def _parse_tool_calls(raw_calls: Any) -> list[LLMToolCall]:
    if not isinstance(raw_calls, list):
        return []
    parsed: list[LLMToolCall] = []
    for idx, entry in enumerate(cast(list[Any], raw_calls)):
        if not isinstance(entry, dict):
            continue
        item = cast(dict[str, Any], entry)
        fn_raw = item.get("function")
        if not isinstance(fn_raw, dict):
            continue
        fn = cast(dict[str, Any], fn_raw)
        name = str(fn.get("name", "")).strip()
        if not name:
            continue
        raw_args = fn.get("arguments", "")
        args_str = raw_args if isinstance(raw_args, str) else json.dumps(raw_args)
        args_dict: dict[str, Any] = {}
        if args_str.strip():
            try:
                loaded = json.loads(args_str)
                if isinstance(loaded, dict):
                    args_dict = cast(dict[str, Any], loaded)
            except json.JSONDecodeError:
                args_dict = {}
        call_id = str(item.get("id") or f"call_{idx}")
        parsed.append(
            LLMToolCall(
                id=call_id,
                name=name,
                arguments=args_dict,
                raw_arguments=args_str,
            )
        )
    return parsed


class UnifiedLLMClient:
    """Async OpenAI-compatible HTTP client wrapped in atomic BudgetGuard accounting."""

    def __init__(
        self,
        *,
        base_url: str,
        budget: BudgetGuard,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_sec: float = 60.0,
        max_retries: int = 3,
        backoff_base_sec: float = 0.5,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._budget = budget
        self._api_key = api_key if api_key is not None else os.environ.get(api_key_env, "")
        self._timeout_sec = timeout_sec
        self._max_retries = max(0, max_retries)
        self._backoff_base_sec = max(0.0, backoff_base_sec)
        self._transport = transport

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def budget(self) -> BudgetGuard:
        return self._budget

    def _endpoint(self) -> str:
        if self._base_url.endswith("/chat/completions"):
            return self._base_url
        return f"{self._base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        tools: Sequence[LLMToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """Reserve budget, invoke /chat/completions with retries, and reconcile usage."""
        est_prompt = _estimate_prompt_tokens(messages, tools)
        reservation = await self._budget.reserve_or_raise(
            model,
            est_prompt,
            max_tokens,
        )

        body: dict[str, Any] = {
            "model": model,
            "messages": [m.to_wire_dict() for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
            body["tool_choice"] = "auto"

        endpoint = self._endpoint()
        headers = self._headers()
        last_error: str = "unknown transport failure"

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_sec,
                transport=self._transport,
            ) as client:
                for attempt in range(self._max_retries + 1):
                    try:
                        resp = await client.post(endpoint, headers=headers, json=body)
                    except (httpx.TimeoutException, httpx.TransportError) as exc:
                        last_error = f"{type(exc).__name__}: {exc}"
                        if attempt < self._max_retries:
                            await asyncio.sleep(self._backoff_base_sec * (2**attempt))
                            continue
                        raise RuntimeSandboxError(
                            f"LLM endpoint '{endpoint}' unreachable after "
                            f"{self._max_retries + 1} attempt(s): {last_error}"
                        ) from exc

                    if resp.status_code in _RETRYABLE_STATUS_CODES:
                        last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                        if attempt < self._max_retries:
                            await asyncio.sleep(self._backoff_base_sec * (2**attempt))
                            continue
                        raise RuntimeSandboxError(
                            f"LLM endpoint '{endpoint}' failed after "
                            f"{self._max_retries + 1} attempt(s): {last_error}"
                        )

                    if resp.status_code >= 400:
                        raise RuntimeSandboxError(
                            f"LLM endpoint '{endpoint}' returned HTTP {resp.status_code}: "
                            f"{resp.text[:400]}"
                        )

                    try:
                        data_raw: Any = resp.json()
                    except ValueError as exc:
                        raise RuntimeSandboxError(
                            f"LLM endpoint '{endpoint}' returned non-JSON payload: {exc}"
                        ) from exc

                    if not isinstance(data_raw, dict):
                        raise RuntimeSandboxError(
                            f"LLM endpoint '{endpoint}' returned unexpected JSON root type."
                        )
                    data = cast(dict[str, Any], data_raw)
                    parsed = self._parse_response(data, fallback_model=model, est_prompt=est_prompt)
                    await self._budget.reconcile(
                        reservation,
                        parsed.usage.prompt_tokens,
                        parsed.usage.completion_tokens,
                    )
                    return parsed
        except Exception:
            await self._budget.release(reservation)
            raise

        await self._budget.release(reservation)
        raise RuntimeSandboxError(f"LLM call to '{endpoint}' failed: {last_error}")

    @staticmethod
    def _parse_response(
        data: dict[str, Any],
        *,
        fallback_model: str,
        est_prompt: int,
    ) -> LLMResponse:
        choices_raw = data.get("choices")
        content = ""
        tool_calls: list[LLMToolCall] = []
        finish_reason: str | None = None

        if isinstance(choices_raw, list) and choices_raw:
            first = cast(list[Any], choices_raw)[0]
            if isinstance(first, dict):
                first_dict = cast(dict[str, Any], first)
                fr = first_dict.get("finish_reason")
                finish_reason = str(fr) if fr is not None else None
                msg_raw = first_dict.get("message")
                if isinstance(msg_raw, dict):
                    msg = cast(dict[str, Any], msg_raw)
                    raw_content = msg.get("content")
                    if isinstance(raw_content, str):
                        content = raw_content
                    tool_calls = _parse_tool_calls(msg.get("tool_calls"))

        usage_raw = data.get("usage")
        prompt_tokens = est_prompt
        completion_tokens = max(1, len(content) // 4) if content else 0
        if isinstance(usage_raw, dict):
            usage_dict = cast(dict[str, Any], usage_raw)
            pt = usage_dict.get("prompt_tokens")
            ct = usage_dict.get("completion_tokens")
            if isinstance(pt, int) and pt >= 0:
                prompt_tokens = pt
            if isinstance(ct, int) and ct >= 0:
                completion_tokens = ct

        model_name = str(data.get("model") or fallback_model)
        return LLMResponse(
            model=model_name,
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=LLMUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        )
