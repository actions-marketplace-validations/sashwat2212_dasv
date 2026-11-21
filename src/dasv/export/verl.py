"""veRL (HybridFlow) parquet-free JSONL dataset exporter (SDD-05, FEAT-18)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from dasv.export.common import (
    ExportResult,
    compute_export_manifest_sha,
    resolve_task_instruction,
    resolve_verifier_ref,
)
from dasv.profile.signal_profiler import CurriculumEntry


class VerlRewardModelSpec(BaseModel):
    """veRL `reward_model` block pointing to a deterministic rule/clean-image verifier."""

    style: Literal["rule"] = "rule"
    ground_truth: dict[str, Any]


class VerlDatasetRow(BaseModel):
    """veRL HybridFlow `RLHFDataset`-compatible JSONL row."""

    data_source: str
    prompt: list[dict[str, str]]
    ability: str
    reward_model: VerlRewardModelSpec
    extra_info: dict[str, Any]


def export_verl(
    entries: Sequence[CurriculumEntry],
    out_path: Path,
    *,
    data_source_prefix: str = "dasv",
    ability: str = "code_verification",
) -> ExportResult:
    """Export calibrated curriculum entries to veRL HybridFlow JSONL schema (zero pyarrow deps)."""
    provisional_rows: list[VerlDatasetRow] = []
    for entry in entries:
        instruction = resolve_task_instruction(entry)
        verifier_ref = resolve_verifier_ref(entry)
        provisional_rows.append(
            VerlDatasetRow(
                data_source=f"{data_source_prefix}/{entry.standard}",
                prompt=[{"role": "user", "content": instruction}],
                ability=ability,
                reward_model=VerlRewardModelSpec(
                    style="rule",
                    ground_truth={
                        "task_id": entry.task_id,
                        "root_path": entry.root_path,
                        "verifier_ref": verifier_ref,
                    },
                ),
                extra_info={
                    "task_id": entry.task_id,
                    "standard": entry.standard,
                    "bucket": entry.bucket.value,
                    "base_model": entry.base_model,
                    "rollouts_g": entry.rollouts_g,
                    "pass_at_1": entry.pass_at_1,
                    "advantage_variance": entry.advantage_variance,
                    "sampler_weight": entry.recommended_sampler_weight,
                },
            )
        )

    manifest_sha = compute_export_manifest_sha(provisional_rows)
    final_rows = [
        row.model_copy(update={"extra_info": {**row.extra_info, "dasv_manifest_sha": manifest_sha}})
        for row in provisional_rows
    ]

    target_jsonl = (
        out_path / "verl_dataset.jsonl" if (out_path.is_dir() or not out_path.suffix) else out_path
    )
    target_jsonl.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row.model_dump(mode="json"), sort_keys=True) for row in final_rows]
    target_jsonl.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")

    return ExportResult(
        format="verl",
        output_path=target_jsonl,
        companion_paths=[],
        record_count=len(final_rows),
        manifest_sha=manifest_sha,
    )
