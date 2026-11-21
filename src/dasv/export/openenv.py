"""OpenEnv framework-neutral `TrainingTrace` JSONL exporter (SDD-05, FEAT-18)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, Field

from dasv.core.errors import ConfigError
from dasv.core.schemas import ExploitTrace
from dasv.export.common import (
    ExportResult,
    compute_export_manifest_sha,
    resolve_task_instruction,
)
from dasv.profile.signal_profiler import CurriculumEntry


class OpenEnvTraceTurn(BaseModel):
    """Single trajectory step carrying explicit token-level `loss_mask` semantics."""

    turn_index: int
    role: Literal["user", "assistant", "tool", "system"]
    content: str
    loss_mask: Literal[0, 1]
    metadata: dict[str, Any] = Field(default_factory=dict)


class OpenEnvTrainingTrace(BaseModel):
    """Framework-neutral OpenEnv `TrainingTrace` JSONL record."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    episode_id: str
    task_id: str
    source: Literal["exploit_trace", "curriculum_seed"]
    hack_class: str | None = None
    gold_sanity_status: str | None = None
    turns: list[OpenEnvTraceTurn] = Field(default_factory=list[OpenEnvTraceTurn])
    token_ids: list[int] = Field(default_factory=list[int])
    sampled_logprobs: list[float] = Field(default_factory=list[float])
    loss_mask: list[Literal[0, 1]] = Field(default_factory=list[Literal[0, 1]])
    reward: float
    tamper_flags: list[str] = Field(default_factory=list[str])


def _exploit_to_training_trace(trace: ExploitTrace) -> OpenEnvTrainingTrace:
    turns: list[OpenEnvTraceTurn] = []
    tamper_flags: list[str] = []
    mask_vec: list[Literal[0, 1]] = []

    for idx, step in enumerate(trace.steps):
        if step.get("event") == "clean_image_grade":
            raw_tampers = step.get("tamper_attempts")
            if isinstance(raw_tampers, list):
                tamper_flags.extend(str(p) for p in cast(list[Any], raw_tampers))
            continue

        if "action" in step:
            action_json = json.dumps(step["action"], sort_keys=True)
            turns.append(
                OpenEnvTraceTurn(
                    turn_index=idx,
                    role="assistant",
                    content=action_json,
                    loss_mask=1,
                    metadata={"tool_call_id": step.get("tool_call_id")},
                )
            )
            mask_vec.append(1)
            obs = str(step.get("observation_preview", ""))
            turns.append(
                OpenEnvTraceTurn(
                    turn_index=idx,
                    role="tool",
                    content=obs,
                    loss_mask=0,
                    metadata={"fence_nonce": step.get("fence_nonce")},
                )
            )
            mask_vec.append(0)
        else:
            role_raw = str(step.get("role", "assistant"))
            role: Literal["user", "assistant", "tool", "system"] = (
                "assistant" if role_raw == "assistant" else "tool"
            )
            step_mask: Literal[0, 1] = 1 if role == "assistant" else 0
            turns.append(
                OpenEnvTraceTurn(
                    turn_index=idx,
                    role=role,
                    content=str(step.get("content", "")),
                    loss_mask=step_mask,
                )
            )
            mask_vec.append(step_mask)

    return OpenEnvTrainingTrace(
        episode_id=trace.exploit_id,
        task_id=trace.task_id,
        source="exploit_trace",
        hack_class=trace.hack_class,
        gold_sanity_status=trace.gold_sanity_status,
        turns=turns,
        token_ids=[],
        sampled_logprobs=[],
        loss_mask=mask_vec,
        reward=trace.reward_obtained,
        tamper_flags=sorted(set(tamper_flags)),
    )


def _curriculum_to_seed_trace(entry: CurriculumEntry) -> OpenEnvTrainingTrace:
    instruction = resolve_task_instruction(entry)
    return OpenEnvTrainingTrace(
        episode_id=f"seed-{entry.task_id}",
        task_id=entry.task_id,
        source="curriculum_seed",
        turns=[
            OpenEnvTraceTurn(
                turn_index=0,
                role="user",
                content=instruction,
                loss_mask=0,
                metadata={
                    "bucket": entry.bucket.value,
                    "pass_at_1": entry.pass_at_1,
                    "sampler_weight": entry.recommended_sampler_weight,
                },
            )
        ],
        token_ids=[],
        sampled_logprobs=[],
        loss_mask=[0],
        reward=entry.pass_at_1,
        tamper_flags=[],
    )


def export_openenv_traces(
    *,
    exploits: Sequence[ExploitTrace] = (),
    entries: Sequence[CurriculumEntry] = (),
    out_path: Path,
) -> ExportResult:
    """Export `ExploitTrace` and/or curriculum records into OpenEnv `TrainingTrace` JSONL."""
    if not exploits and not entries:
        raise ConfigError(
            "OpenEnv TrainingTrace export requires at least one ExploitTrace (`--exploits`) "
            "or CurriculumEntry (`--curriculum`)."
        )

    records: list[OpenEnvTrainingTrace] = [
        *(_exploit_to_training_trace(ex) for ex in exploits),
        *(_curriculum_to_seed_trace(en) for en in entries),
    ]
    manifest_sha = compute_export_manifest_sha(records)

    target_jsonl = (
        out_path / "openenv_traces.jsonl"
        if (out_path.is_dir() or not out_path.suffix)
        else out_path
    )
    target_jsonl.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({**rec.model_dump(mode="json"), "manifest_sha": manifest_sha}, sort_keys=True)
        for rec in records
    ]
    target_jsonl.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")

    return ExportResult(
        format="openenv",
        output_path=target_jsonl,
        companion_paths=[],
        record_count=len(records),
        manifest_sha=manifest_sha,
    )
