"""Unit tests for atomic async BudgetGuard concurrency and cap enforcement."""

from __future__ import annotations

import asyncio

import pytest

from dasv.core.budget import BudgetGuard, ModelPricing
from dasv.core.errors import (
    EXIT_BUDGET_EXCEEDED,
    EXIT_CONFIG_ERROR,
    BudgetExceededError,
    ConfigError,
)


async def test_budget_guard_concurrent_reservations_never_exceed_cap() -> None:
    # $1.00 per 1M input and $1.00 per 1M output -> 100k tokens = $0.10 per reservation
    guard = BudgetGuard(
        max_cost_usd=0.25,
        max_tokens=1_000_000,
        pricing_table={"test/model": ModelPricing(input_per_1m_usd=1.0, output_per_1m_usd=1.0)},
    )

    async def attempt() -> bool:
        try:
            await guard.reserve_or_raise("test/model", 50_000, 50_000)
            return True
        except BudgetExceededError as exc:
            assert exc.exit_code == EXIT_BUDGET_EXCEEDED
            return False

    results = await asyncio.gather(*(attempt() for _ in range(10)))
    # At $0.10 each and $0.25 cap, strictly 2 reservations can succeed
    assert results.count(True) == 2
    assert results.count(False) == 8

    snap = await guard.snapshot()
    assert snap.reserved_cost_usd == pytest.approx(0.20)
    assert snap.reserved_cost_usd <= snap.max_cost_usd


async def test_budget_guard_reconcile_releases_unused_reservation() -> None:
    guard = BudgetGuard(
        max_cost_usd=0.15,
        max_tokens=200_000,
        pricing_table={"test/model": ModelPricing(input_per_1m_usd=1.0, output_per_1m_usd=1.0)},
    )
    h1 = await guard.reserve_or_raise("test/model", 50_000, 50_000)  # reserves $0.10
    # Second $0.10 call blocked while h1 is open
    with pytest.raises(BudgetExceededError):
        await guard.reserve_or_raise("test/model", 50_000, 50_000)

    # Reconcile actual usage of only 10k tokens ($0.01) -> frees $0.09
    await guard.reconcile(h1, actual_prompt_tokens=5_000, actual_completion_tokens=5_000)

    # Double reconcile must fail cleanly
    with pytest.raises(ConfigError):
        await guard.reconcile(h1, actual_prompt_tokens=1, actual_completion_tokens=1)

    # Now another $0.10 reservation succeeds
    h2 = await guard.reserve_or_raise("test/model", 50_000, 50_000)
    await guard.release(h2)
    snap = await guard.snapshot()
    assert snap.committed_tokens == 10_000
    assert snap.committed_cost_usd == pytest.approx(0.01)
    assert snap.reserved_tokens == 0


async def test_local_endpoint_zero_cost_still_enforces_max_tokens() -> None:
    guard = BudgetGuard(
        max_cost_usd=0.0,
        max_tokens=1_000,
        base_url="http://localhost:11434/v1",
    )
    h1 = await guard.reserve_or_raise("qwen2.5-coder:7b", 400, 400)
    await guard.reconcile(h1, 300, 200)
    snap = await guard.snapshot()
    assert snap.committed_cost_usd == 0.0
    assert snap.committed_tokens == 500

    with pytest.raises(BudgetExceededError):
        await guard.reserve_or_raise("qwen2.5-coder:7b", 400, 200)


async def test_unknown_cloud_model_raises_config_error_unless_opted_in() -> None:
    strict_guard = BudgetGuard(
        max_cost_usd=5.0,
        max_tokens=100_000,
        base_url="https://api.openai.com/v1",
    )
    with pytest.raises(ConfigError) as exc_info:
        await strict_guard.reserve_or_raise("mystery/model-x", 100, 100)
    assert exc_info.value.exit_code == EXIT_CONFIG_ERROR

    permissive_guard = BudgetGuard(
        max_cost_usd=5.0,
        max_tokens=100_000,
        base_url="https://api.openai.com/v1",
        allow_unknown_pricing=True,
    )
    handle = await permissive_guard.reserve_or_raise("mystery/model-x", 100, 100)
    assert handle.reserved_cost_usd == 0.0


async def test_budget_guard_pricing_aliases_and_openrouter_free_tier() -> None:
    guard = BudgetGuard(
        max_cost_usd=2.0,
        max_tokens=100_000,
        base_url="https://openrouter.ai/api/v1",
    )
    for model_id in ("gemini-2.5-flash", "deepseek-chat", "claude-3-5-haiku", "gpt-4.1-mini"):
        h = await guard.reserve_or_raise(model_id, 1_000, 1_000)
        assert h.reserved_cost_usd > 0.0
        await guard.release(h)

    free_h = await guard.reserve_or_raise("qwen/qwen-2.5-coder-32b-instruct:free", 2_000, 1_000)
    assert free_h.reserved_cost_usd == 0.0
    assert free_h.reserved_tokens == 3_000
    await guard.release(free_h)
