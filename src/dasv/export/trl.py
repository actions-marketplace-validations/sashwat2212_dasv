"""TRL (`GRPOTrainer` / `AsyncGRPOTrainer`) + Unsloth dataset exporter (SDD-05, FEAT-18)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from dasv.export.common import (
    ExportResult,
    compute_export_manifest_sha,
    resolve_task_instruction,
    resolve_verifier_ref,
)
from dasv.profile.signal_profiler import CurriculumEntry


class TRLDatasetRow(BaseModel):
    """Hugging Face TRL GRPOTrainer-compatible JSONL row."""

    prompt: list[dict[str, str]]
    task_id: str
    root_path: str
    standard: str
    verifier_ref: str
    dasv_grade: str
    bucket: str
    baseline_pass_at_1: float
    advantage_variance: float
    sampler_weight: float


_TRL_REWARD_WRAPPER_TEMPLATE = '''"""TRL reward_func delegating to dasv CleanImageGrader."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from dasv.adapters.registry import AdapterRegistry
from dasv.profile.signal_profiler import _artifact_from_policy_response
from dasv.sandbox.docker_provider import DockerSandboxProvider
from dasv.sandbox.janitor import ContainerJanitor
from dasv.verifier.clean_grader import CleanImageGrader, ensure_container_image


async def _grade_completion(root_path: str, completion_text: str, run_id: str) -> float:
    path = Path(root_path)
    adapter = AdapterRegistry().resolve(path)
    spec = adapter.load(path)
    janitor = ContainerJanitor()
    provider = DockerSandboxProvider(janitor=janitor)
    try:
        built_spec = await ensure_container_image(spec, provider, run_id)
        artifact = _artifact_from_policy_response(built_spec, completion_text)
        outcome = await CleanImageGrader(provider).grade(built_spec, artifact, f"{run_id}-grade")
        return float(outcome.reward)
    finally:
        await janitor.reap_all(run_id)


def dasv_clean_image_reward_func(
    completions: list[Any],
    root_path: list[str],
    **kwargs: Any,
) -> list[float]:
    """TRL GRPOTrainer reward function enforcing clean-image out-of-container verification."""
    _ = kwargs
    rewards: list[float] = []
    for idx, (comp, env_root) in enumerate(zip(completions, root_path, strict=True)):
        if isinstance(comp, list) and comp and isinstance(comp[-1], dict):
            text = str(comp[-1].get("content", ""))
        else:
            text = str(comp)
        rewards.append(asyncio.run(_grade_completion(env_root, text, f"dasv-trl-{idx}")))
    return rewards
'''


def export_trl(
    entries: Sequence[CurriculumEntry],
    out_path: Path,
    *,
    grade_map: dict[str, str] | None = None,
    emit_reward_wrapper: bool = True,
) -> ExportResult:
    """Export calibrated curriculum entries to TRL/Unsloth JSONL + clean-image reward wrapper."""
    grades = grade_map or {}
    rows: list[TRLDatasetRow] = []
    for entry in entries:
        instruction = resolve_task_instruction(entry)
        rows.append(
            TRLDatasetRow(
                prompt=[{"role": "user", "content": instruction}],
                task_id=entry.task_id,
                root_path=entry.root_path,
                standard=entry.standard,
                verifier_ref=resolve_verifier_ref(entry),
                dasv_grade=grades.get(entry.task_id, "A"),
                bucket=entry.bucket.value,
                baseline_pass_at_1=entry.pass_at_1,
                advantage_variance=entry.advantage_variance,
                sampler_weight=entry.recommended_sampler_weight,
            )
        )

    manifest_sha = compute_export_manifest_sha(rows)
    target_jsonl = (
        out_path / "trl_dataset.jsonl" if (out_path.is_dir() or not out_path.suffix) else out_path
    )
    target_jsonl.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({**row.model_dump(mode="json"), "manifest_sha": manifest_sha}, sort_keys=True)
        for row in rows
    ]
    target_jsonl.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")

    companions: list[Path] = []
    if emit_reward_wrapper:
        wrapper_path = target_jsonl.parent / "trl_reward_func.py"
        wrapper_path.write_text(_TRL_REWARD_WRAPPER_TEMPLATE, encoding="utf-8")
        companions.append(wrapper_path)

    return ExportResult(
        format="trl",
        output_path=target_jsonl,
        companion_paths=companions,
        record_count=len(rows),
        manifest_sha=manifest_sha,
    )
