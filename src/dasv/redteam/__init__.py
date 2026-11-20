"""Adversarial Hacker-Fixer red-team engine & unified OpenAI-compatible LLM client (SDD-02)."""

from __future__ import annotations

from dasv.redteam.fencing import DEFAULT_MAX_OBS_BYTES, fence_untrusted_output, truncate_utf8_bytes
from dasv.redteam.fixer_agent import FixerAgent, extract_unified_diff
from dasv.redteam.hacker_agent import (
    HACK_CLASSES,
    HACK_PLAYBOOKS,
    HACKER_TOOLS,
    HackClass,
    HackerAgent,
    HackerAttemptResult,
    format_exploit_jsonl_line,
)
from dasv.redteam.llm_client import (
    LLMClient,
    LLMMessage,
    LLMResponse,
    LLMToolCall,
    LLMToolSpec,
    LLMUsage,
    UnifiedLLMClient,
)
from dasv.redteam.runner import (
    RedTeamReport,
    RedTeamRunner,
    build_tier2_redteam_hook,
    parse_hack_classes,
)

__all__ = [
    "DEFAULT_MAX_OBS_BYTES",
    "FixerAgent",
    "HACK_CLASSES",
    "HACK_PLAYBOOKS",
    "HACKER_TOOLS",
    "HackClass",
    "HackerAgent",
    "HackerAttemptResult",
    "LLMClient",
    "LLMMessage",
    "LLMResponse",
    "LLMToolCall",
    "LLMToolSpec",
    "LLMUsage",
    "RedTeamReport",
    "RedTeamRunner",
    "UnifiedLLMClient",
    "build_tier2_redteam_hook",
    "extract_unified_diff",
    "fence_untrusted_output",
    "format_exploit_jsonl_line",
    "parse_hack_classes",
    "truncate_utf8_bytes",
]
