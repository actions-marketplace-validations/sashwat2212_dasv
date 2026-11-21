"""Shared loaders and metadata helpers for RL trainer exporters (SDD-05, FEAT-18)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, Field, ValidationError

from dasv.core.errors import ConfigError
from dasv.core.schemas import CurriculumBucket, ExploitTrace
from dasv.profile.signal_profiler import CurriculumEntry

ExportFormat = Literal["trl", "verl", "prime_rl", "openenv"]
EXPORT_FORMATS: tuple[ExportFormat, ...] = ("trl", "verl", "prime_rl", "openenv")


class ExportResult(BaseModel):
    """Summary returned by all downstream RL framework exporters."""

    format: ExportFormat
    output_path: Path
    companion_paths: list[Path] = Field(default_factory=list[Path])
    record_count: int
    manifest_sha: str


def load_curriculum_entries(
    curriculum_path: Path,
    *,
    include_buckets: Sequence[CurriculumBucket] | None = None,
) -> list[CurriculumEntry]:
    """Load `CurriculumEntry` records from a `.jsonl` or `.yaml`/`.yml` curriculum manifest."""
    if not curriculum_path.is_file():
        raise ConfigError(f"Curriculum file not found: '{curriculum_path}'.")

    raw_text = curriculum_path.read_text(encoding="utf-8")
    raw_items: list[Any] = []
    suffix = curriculum_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            loaded: Any = yaml.safe_load(raw_text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"Malformed YAML curriculum '{curriculum_path}': {exc}") from exc
        if isinstance(loaded, dict):
            tasks_val = cast(dict[str, Any], loaded).get("tasks")
            if isinstance(tasks_val, list):
                raw_items = cast(list[Any], tasks_val)
        elif isinstance(loaded, list):
            raw_items = cast(list[Any], loaded)
    else:
        for lineno, line in enumerate(raw_text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw_items.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ConfigError(
                    f"Malformed JSON in curriculum '{curriculum_path}' line {lineno}: {exc}"
                ) from exc

    entries: list[CurriculumEntry] = []
    for idx, item in enumerate(raw_items, start=1):
        try:
            entries.append(CurriculumEntry.model_validate(item))
        except ValidationError as exc:
            raise ConfigError(
                f"Invalid CurriculumEntry at index {idx} in '{curriculum_path}': {exc}"
            ) from exc

    if include_buckets is not None:
        allowed = set(include_buckets)
        entries = [e for e in entries if e.bucket in allowed]

    if not entries:
        raise ConfigError(
            f"No matching curriculum entries found in '{curriculum_path}' "
            f"(filter={[b.value for b in include_buckets] if include_buckets else 'all'})."
        )
    return entries


def load_exploit_traces(exploits_path: Path) -> list[ExploitTrace]:
    """Load `ExploitTrace` records from an `exploits.jsonl` file."""
    if not exploits_path.is_file():
        raise ConfigError(f"Exploit traces JSONL file not found: '{exploits_path}'.")

    traces: list[ExploitTrace] = []
    for lineno, line in enumerate(
        exploits_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            raw: Any = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"Malformed JSON in exploit traces '{exploits_path}' line {lineno}: {exc}"
            ) from exc
        try:
            traces.append(ExploitTrace.model_validate(raw))
        except ValidationError as exc:
            raise ConfigError(
                f"Invalid ExploitTrace in '{exploits_path}' line {lineno}: {exc}"
            ) from exc

    if not traces:
        raise ConfigError(f"No ExploitTrace records found in '{exploits_path}'.")
    return traces


def resolve_task_instruction(entry: CurriculumEntry) -> str:
    """Load canonical task instruction text from disk when available, else synthesize fallback."""
    root = Path(entry.root_path)
    for filename in ("instruction.md", "README.md"):
        candidate = root / filename
        if candidate.is_file():
            try:
                text = candidate.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    return text
            except OSError:
                continue
    return (
        f"Solve RL environment task '{entry.task_id}' (standard={entry.standard}). "
        "Produce a clean, minimal solution satisfying all verifier specifications."
    )


def resolve_verifier_ref(entry: CurriculumEntry) -> str:
    """Resolve a stable verifier entrypoint reference string for downstream reward wrappers."""
    root = Path(entry.root_path)
    for rel in ("tests/test.sh", "tests/", "env.py", "tasks/grade.py"):
        if (root / rel).exists():
            return f"{entry.standard}://{entry.task_id}/{rel}"
    return f"{entry.standard}://{entry.task_id}/verifier"


def compute_export_manifest_sha(records: Sequence[BaseModel]) -> str:
    """Compute a deterministic SHA-256 fingerprint over serialized exported records."""
    hasher = hashlib.sha256()
    for rec in records:
        payload = json.dumps(rec.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        hasher.update(payload.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()
