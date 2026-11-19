"""Atomic async BudgetGuard enforcing hard USD and token ceilings (SDD-01)."""

from __future__ import annotations

import asyncio
from urllib.parse import urlparse
from uuid import uuid4

from pydantic import BaseModel, Field

from dasv.core.errors import BudgetExceededError, ConfigError


class ModelPricing(BaseModel):
    input_per_1m_usd: float = Field(ge=0.0)
    output_per_1m_usd: float = Field(ge=0.0)


class ReservationHandle(BaseModel):
    reservation_id: str
    model: str
    reserved_tokens: int
    reserved_cost_usd: float
    is_local: bool = False


class BudgetSnapshot(BaseModel):
    committed_tokens: int
    committed_cost_usd: float
    reserved_tokens: int
    reserved_cost_usd: float
    max_tokens: int
    max_cost_usd: float


DEFAULT_PRICING_TABLE: dict[str, ModelPricing] = {
    "openai/gpt-4o-mini": ModelPricing(input_per_1m_usd=0.15, output_per_1m_usd=0.60),
    "gpt-4o-mini": ModelPricing(input_per_1m_usd=0.15, output_per_1m_usd=0.60),
    "openai/gpt-4o": ModelPricing(input_per_1m_usd=2.50, output_per_1m_usd=10.00),
    "gpt-4o": ModelPricing(input_per_1m_usd=2.50, output_per_1m_usd=10.00),
    "anthropic/claude-3-5-haiku": ModelPricing(input_per_1m_usd=0.80, output_per_1m_usd=4.00),
    "deepseek/deepseek-chat": ModelPricing(input_per_1m_usd=0.27, output_per_1m_usd=1.10),
    "google/gemini-2.5-flash": ModelPricing(input_per_1m_usd=0.15, output_per_1m_usd=0.60),
}

_ZERO_PRICING = ModelPricing(input_per_1m_usd=0.0, output_per_1m_usd=0.0)


def is_local_endpoint(base_url: str | None, model: str | None = None) -> bool:
    """Return True when the target model/endpoint runs locally (Ollama/MLX/localhost)."""
    if model is not None and model.lower().startswith(("ollama/", "mlx/")):
        return True
    if not base_url:
        return False
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}


class BudgetGuard:
    """Concurrency-safe USD & token circuit breaker using asyncio.Lock."""

    def __init__(
        self,
        max_cost_usd: float,
        max_tokens: int,
        pricing_table: dict[str, ModelPricing] | None = None,
        *,
        base_url: str | None = None,
        allow_unknown_pricing: bool = False,
    ) -> None:
        if max_cost_usd < 0.0:
            raise ConfigError("max_cost_usd must be non-negative.")
        if max_tokens < 1:
            raise ConfigError("max_tokens must be >= 1.")
        self._max_cost_usd = max_cost_usd
        self._max_tokens = max_tokens
        self._pricing_table: dict[str, ModelPricing] = (
            dict(DEFAULT_PRICING_TABLE) if pricing_table is None else dict(pricing_table)
        )
        self._base_url = base_url
        self._allow_unknown_pricing = allow_unknown_pricing
        self._lock = asyncio.Lock()
        self._committed_tokens = 0
        self._committed_cost_usd = 0.0
        self._active: dict[str, ReservationHandle] = {}

    def _resolve_pricing(
        self, model: str, *, local_override: bool | None
    ) -> tuple[ModelPricing, bool]:
        local = (
            local_override
            if local_override is not None
            else is_local_endpoint(self._base_url, model)
        )
        if local:
            return _ZERO_PRICING, True

        pricing = self._pricing_table.get(model)
        if pricing is None and "/" in model:
            pricing = self._pricing_table.get(model.split("/", 1)[1])

        if pricing is None:
            if self._allow_unknown_pricing:
                return _ZERO_PRICING, False
            raise ConfigError(
                f"Unknown pricing for cloud model '{model}'. Add it to the pricing table "
                "or explicitly opt in with allow_unknown_pricing=True."
            )
        return pricing, False

    @staticmethod
    def _estimate_cost(pricing: ModelPricing, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens / 1_000_000.0) * pricing.input_per_1m_usd + (
            completion_tokens / 1_000_000.0
        ) * pricing.output_per_1m_usd

    async def reserve_or_raise(
        self,
        model: str,
        est_prompt_tokens: int,
        max_completion_tokens: int,
        *,
        is_local: bool | None = None,
    ) -> ReservationHandle:
        """Reserve worst-case tokens and USD spend before dispatching an LLM HTTP call."""
        if est_prompt_tokens < 0 or max_completion_tokens < 0:
            raise ConfigError("Token estimates must be non-negative.")

        pricing, resolved_local = self._resolve_pricing(model, local_override=is_local)
        est_tokens = est_prompt_tokens + max_completion_tokens
        est_cost = self._estimate_cost(pricing, est_prompt_tokens, max_completion_tokens)

        async with self._lock:
            reserved_tokens = sum(r.reserved_tokens for r in self._active.values())
            reserved_cost = sum(r.reserved_cost_usd for r in self._active.values())

            projected_tokens = self._committed_tokens + reserved_tokens + est_tokens
            projected_cost = self._committed_cost_usd + reserved_cost + est_cost

            if projected_tokens > self._max_tokens:
                raise BudgetExceededError(
                    f"Token budget exceeded: projected {projected_tokens} > cap {self._max_tokens}."
                )
            if projected_cost > self._max_cost_usd + 1e-12:
                raise BudgetExceededError(
                    f"USD budget exceeded: projected ${projected_cost:.6f} > "
                    f"cap ${self._max_cost_usd:.6f}."
                )

            handle = ReservationHandle(
                reservation_id=uuid4().hex,
                model=model,
                reserved_tokens=est_tokens,
                reserved_cost_usd=est_cost,
                is_local=resolved_local,
            )
            self._active[handle.reservation_id] = handle
            return handle

    async def reconcile(
        self,
        handle: ReservationHandle,
        actual_prompt_tokens: int,
        actual_completion_tokens: int,
    ) -> None:
        """Release unused reservation and commit exact token and USD usage."""
        if actual_prompt_tokens < 0 or actual_completion_tokens < 0:
            raise ConfigError("Actual token counts must be non-negative.")

        pricing, _ = self._resolve_pricing(handle.model, local_override=handle.is_local)
        actual_tokens = actual_prompt_tokens + actual_completion_tokens
        actual_cost = self._estimate_cost(pricing, actual_prompt_tokens, actual_completion_tokens)

        async with self._lock:
            if handle.reservation_id not in self._active:
                raise ConfigError(
                    f"Reservation '{handle.reservation_id}' is unknown or already reconciled."
                )
            del self._active[handle.reservation_id]
            self._committed_tokens += actual_tokens
            self._committed_cost_usd += actual_cost

    async def release(self, handle: ReservationHandle) -> None:
        """Release an active reservation without committing spend (e.g., on transport error)."""
        async with self._lock:
            self._active.pop(handle.reservation_id, None)

    async def snapshot(self) -> BudgetSnapshot:
        """Return a consistent snapshot of committed and active reserved usage."""
        async with self._lock:
            return BudgetSnapshot(
                committed_tokens=self._committed_tokens,
                committed_cost_usd=round(self._committed_cost_usd, 10),
                reserved_tokens=sum(r.reserved_tokens for r in self._active.values()),
                reserved_cost_usd=round(
                    sum(r.reserved_cost_usd for r in self._active.values()), 10
                ),
                max_tokens=self._max_tokens,
                max_cost_usd=self._max_cost_usd,
            )
