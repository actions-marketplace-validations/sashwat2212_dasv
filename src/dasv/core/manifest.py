"""Reproducibility manifest builder (`manifest.json`, FEAT-21)."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from dasv import __version__
from dasv.core.schemas import RunManifest

_IGNORED_DIRS = {".git", "__pycache__", ".venv", ".pytest_cache", ".ruff_cache"}


def compute_dataset_sha256(path: Path) -> str:
    """Compute a deterministic SHA-256 digest of a file or environment directory tree."""
    hasher = hashlib.sha256()
    resolved = path.resolve()
    if resolved.is_file():
        hasher.update(resolved.name.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(resolved.read_bytes())
        return hasher.hexdigest()

    if not resolved.is_dir():
        return hasher.hexdigest()

    files: list[Path] = []
    for item in resolved.rglob("*"):
        if any(part in _IGNORED_DIRS for part in item.relative_to(resolved).parts):
            continue
        if item.is_file():
            files.append(item)

    for file_path in sorted(files, key=lambda p: p.relative_to(resolved).as_posix()):
        rel_posix = file_path.relative_to(resolved).as_posix()
        hasher.update(rel_posix.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(file_path.read_bytes())
        hasher.update(b"\x00")

    return hasher.hexdigest()


def detect_git_sha(cwd: Path) -> str | None:
    """Return current git HEAD SHA if `cwd` is inside a git repository, else None."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    sha = proc.stdout.strip()
    return sha or None


def build_run_manifest(
    dataset_path: Path,
    *,
    cli_args: Sequence[str] = (),
    docker_server_arch: str | None = None,
    oci_image_digests: dict[str, str] | None = None,
    models_used: Sequence[str] = (),
    rng_seed: int = 42,
    total_tokens: int = 0,
    total_cost_usd: float = 0.0,
    run_id: str | None = None,
) -> RunManifest:
    """Construct a pinned RunManifest for deterministic auditing and citation."""
    git_dir = dataset_path if dataset_path.is_dir() else dataset_path.parent
    return RunManifest(
        run_id=run_id or str(uuid4()),
        dasv_version=__version__,
        timestamp_utc=datetime.now(UTC).isoformat(),
        git_sha=detect_git_sha(git_dir),
        host_os=platform.system().lower(),
        host_arch=platform.machine().lower(),
        docker_server_arch=docker_server_arch,
        oci_image_digests=dict(oci_image_digests) if oci_image_digests else {},
        dataset_sha256=compute_dataset_sha256(dataset_path),
        models_used=list(models_used),
        rng_seed=rng_seed,
        total_tokens=total_tokens,
        total_cost_usd=total_cost_usd,
        cli_args=list(cli_args),
    )
