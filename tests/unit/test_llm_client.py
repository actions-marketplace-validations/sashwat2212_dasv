"""Unit tests for UnifiedLLMClient with httpx.MockTransport and BudgetGuard (SDD-02)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from dasv.core.budget import BudgetGuard
from dasv.core.errors import BudgetExceededError, RuntimeSandboxError
from dasv.redteam.llm_client import LLMMessage, LLMToolSpec, UnifiedLLMClient


async def test_llm_client_tool_calls_usage_and_budget_reconciliation() -> None:
    seen_payloads: list[dict[str, Any]] = []
    seen_auth: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers.get("authorization"))
        seen_payloads.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "run_cmd",
                                        "arguments": '{"cmd": "ls -la"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 120, "completion_tokens": 30},
            },
        )

    budget = BudgetGuard(max_cost_usd=1.0, max_tokens=10_000)
    client = UnifiedLLMClient(
        base_url="https://api.example.com/v1",
        budget=budget,
        api_key="sk-test-token",
        transport=httpx.MockTransport(handler),
        backoff_base_sec=0.0,
    )
    tool = LLMToolSpec(
        name="run_cmd",
        description="Run shell command",
        parameters={"type": "object", "properties": {"cmd": {"type": "string"}}},
    )
    resp = await client.complete(
        [LLMMessage(role="user", content="inspect sandbox")],
        model="gpt-4o-mini",
        tools=[tool],
        max_tokens=256,
    )

    assert seen_auth == ["Bearer sk-test-token"]
    assert seen_payloads[0]["tool_choice"] == "auto"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "run_cmd"
    assert resp.tool_calls[0].arguments == {"cmd": "ls -la"}
    assert resp.usage.total_tokens == 150

    snap = await budget.snapshot()
    assert snap.committed_tokens == 150
    assert snap.reserved_tokens == 0
    assert snap.committed_cost_usd > 0.0


async def test_llm_client_retries_429_and_503_then_succeeds() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, text="rate limited")
        if attempts == 2:
            return httpx.Response(503, text="overloaded")
        return httpx.Response(
            200,
            json={
                "model": "qwen2.5-coder:7b",
                "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 5},
            },
        )

    budget = BudgetGuard(
        max_cost_usd=0.0,
        max_tokens=5000,
        base_url="http://localhost:11434/v1",
    )
    client = UnifiedLLMClient(
        base_url="http://localhost:11434/v1",
        budget=budget,
        transport=httpx.MockTransport(handler),
        max_retries=3,
        backoff_base_sec=0.0,
    )
    resp = await client.complete(
        [LLMMessage(role="user", content="ping")],
        model="qwen2.5-coder:7b",
        max_tokens=64,
    )
    assert attempts == 3
    assert resp.content == "ok"
    snap = await budget.snapshot()
    assert snap.committed_cost_usd == 0.0
    assert snap.committed_tokens == 25


async def test_llm_client_releases_budget_reservation_on_transport_error() -> None:
    def failing_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    budget = BudgetGuard(max_cost_usd=1.0, max_tokens=5000)
    client = UnifiedLLMClient(
        base_url="https://api.example.com/v1",
        budget=budget,
        transport=httpx.MockTransport(failing_handler),
        max_retries=1,
        backoff_base_sec=0.0,
    )
    with pytest.raises(RuntimeSandboxError, match="unreachable"):
        await client.complete(
            [LLMMessage(role="user", content="hello")],
            model="gpt-4o-mini",
            max_tokens=128,
        )

    snap = await budget.snapshot()
    assert snap.committed_tokens == 0
    assert snap.reserved_tokens == 0
    assert snap.reserved_cost_usd == 0.0


async def test_llm_client_trips_budget_before_sending_http() -> None:
    http_called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal http_called
        http_called = True
        return httpx.Response(200, json={})

    budget = BudgetGuard(max_cost_usd=0.00001, max_tokens=100_000)
    client = UnifiedLLMClient(
        base_url="https://api.example.com/v1",
        budget=budget,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(BudgetExceededError):
        await client.complete(
            [LLMMessage(role="user", content="expensive prompt")],
            model="gpt-4o",
            max_tokens=10_000,
        )
    assert http_called is False
