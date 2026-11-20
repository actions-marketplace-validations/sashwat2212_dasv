"""Trust-boundary artifact extraction and verifier-path tamper filtering (SDD-02 §1.1)."""

from __future__ import annotations

import base64
import hashlib
import json
import posixpath
import re
from collections.abc import Sequence
from typing import Any, Literal, cast

from pydantic import BaseModel, Field

from dasv.core.schemas import AuditFinding, EnvSpec, Severity
from dasv.sandbox.provider import SandboxHandle, SandboxProvider

_DIFF_HEADER_RE = re.compile(
    r'^diff --git (?:"?a/(?P<a_q>.+?)"?|a/(?P<a_u>.+?)) (?:"?b/(?P<b_q>.+?)"?|b/(?P<b_u>.+?))$'
)


class ExtractedArtifact(BaseModel):
    """Immutable artifact pulled across the Execution -> Verification Plane trust boundary."""

    mode: Literal["git_patch", "file_globs", "state_snapshot"]
    patch_bytes: bytes | None = None
    files: dict[str, bytes] = Field(default_factory=dict)
    sha256: str
    rejected_paths: list[str] = Field(default_factory=list)


def _unquote_git_path(raw: str) -> str:
    cleaned = raw.strip()
    if len(cleaned) >= 2 and cleaned[0] == '"' and cleaned[-1] == '"':
        cleaned = cleaned[1:-1]
    return cleaned


def _normalize_rel_path(raw: str) -> str:
    cleaned = _unquote_git_path(raw).replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    cleaned = cleaned.lstrip("/")
    return posixpath.normpath(cleaned) if cleaned else ""


def is_protected_verifier_path(candidate: str, verifier_paths: Sequence[str]) -> bool:
    """Return True if `candidate` touches any protected verifier path or escapes via '..'."""
    raw = _unquote_git_path(candidate).replace("\\", "/")
    if not raw:
        return False
    if "\x00" in raw or "\n" in raw or "\r" in raw:
        return True
    # Reject absolute path escapes outright (artifacts must stay workspace-relative).
    if raw.startswith("/"):
        return True
    # Reject any parent traversal attempts outright.
    raw_parts = [p for p in raw.split("/") if p]
    if ".." in raw_parts:
        return True

    norm = _normalize_rel_path(raw)
    if norm in {"", ".", ".."} or norm.startswith("../"):
        return True
    norm_lower = norm.lower()

    for vp in verifier_paths:
        vp_raw = _unquote_git_path(vp).replace("\\", "/")
        if ".." in [p for p in vp_raw.split("/") if p]:
            continue
        vp_norm = _normalize_rel_path(vp_raw).rstrip("/").lower()
        if not vp_norm or vp_norm in {".", ".."}:
            continue
        if norm_lower == vp_norm or norm_lower.startswith(f"{vp_norm}/"):
            return True
    return False


def _extract_diff_header_paths(header: str) -> tuple[str, str] | None:
    """Parse `diff --git` headers for both unquoted and quoted paths (including spaces)."""
    if not header.startswith("diff --git "):
        return None
    rest = header[len("diff --git ") :].strip()
    if rest.startswith('"a/'):
        end_a = rest.find('" ', 3)
        if end_a == -1:
            return None
        path_a = rest[3:end_a]
        rem = rest[end_a + 2 :].strip()
        if rem.startswith('"b/') and rem.endswith('"'):
            return path_a, rem[3:-1]
        if rem.startswith("b/"):
            return path_a, rem[2:]
        return None
    match = _DIFF_HEADER_RE.match(header)
    if not match:
        return None
    path_a = match.group("a_q") or match.group("a_u") or ""
    path_b = match.group("b_q") or match.group("b_u") or ""
    return path_a, path_b


def filter_git_patch(
    raw_patch: bytes,
    verifier_paths: Sequence[str],
) -> tuple[bytes, list[str]]:
    """Strip any git diff sections touching `verifier_paths` and record rejected paths."""
    if not raw_patch:
        return b"", []

    text = raw_patch.decode("utf-8", errors="surrogateescape")
    lines = text.splitlines(keepends=True)

    sections: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.startswith("diff --git "):
            if current:
                sections.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append(current)

    kept_chunks: list[str] = []
    rejected: set[str] = set()

    for section in sections:
        header = section[0].rstrip("\r\n") if section else ""
        if not header.startswith("diff --git "):
            # Non-diff preamble outside a diff section.
            kept_chunks.extend(section)
            continue

        parsed_pair = _extract_diff_header_paths(header)
        touched_paths: list[str] = list(parsed_pair) if parsed_pair is not None else []
        for sec_line in section[1:10]:
            stripped = sec_line.rstrip("\r\n")
            if stripped.startswith(("rename from ", "rename to ", "copy from ", "copy to ")):
                touched_paths.append(_unquote_git_path(stripped.split(" ", 2)[2]))
            elif stripped.startswith(("--- ", "+++ ")):
                raw_target = _unquote_git_path(stripped[4:].split("\t", 1)[0])
                if raw_target == "/dev/null":
                    continue
                if raw_target.startswith(("a/", "b/")):
                    touched_paths.append(raw_target[2:])
                else:
                    touched_paths.append(raw_target)

        if not touched_paths:
            rejected.add(header)
            continue

        blocked = [p for p in touched_paths if is_protected_verifier_path(p, verifier_paths)]
        if blocked:
            for bp in blocked:
                rejected.add(_normalize_rel_path(bp) or bp)
            continue

        kept_chunks.extend(section)

    sanitized_bytes = "".join(kept_chunks).encode("utf-8", errors="surrogateescape")
    return sanitized_bytes, sorted(rejected)


def compute_artifact_sha256(
    mode: str,
    patch_bytes: bytes | None,
    files: dict[str, bytes],
) -> str:
    """Compute a deterministic SHA-256 digest over extracted artifact payload."""
    hasher = hashlib.sha256()
    hasher.update(mode.encode("utf-8"))
    hasher.update(b"\x00")
    if patch_bytes is not None:
        hasher.update(patch_bytes)
    hasher.update(b"\x00")
    for path_key in sorted(files.keys()):
        hasher.update(path_key.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(files[path_key])
        hasher.update(b"\x00")
    return hasher.hexdigest()


def build_tamper_findings(artifact: ExtractedArtifact) -> list[AuditFinding]:
    """Emit RULE-BNDY-007 CRITICAL finding when an agent attempts to modify verifier paths."""
    if not artifact.rejected_paths:
        return []
    return [
        AuditFinding(
            rule_id="RULE-BNDY-007",
            title="Tamper attempt blocked: agent modified protected verifier path(s)",
            severity=Severity.CRITICAL,
            tier=1,
            source="dasv",
            file_path=artifact.rejected_paths[0],
            line_number=None,
            evidence=(
                "ArtifactExtractor stripped unauthorized modifications to protected verifier "
                f"path(s): {', '.join(artifact.rejected_paths)}."
            ),
            remediation=(
                "Keep verifier/test suites out of the Execution Plane and grade strictly via "
                "CleanImageGrader with read-only (:ro) verifier mounts."
            ),
        )
    ]


class ArtifactExtractor:
    """Extracts strictly scoped agent output across Execution -> Verification Plane."""

    async def extract(
        self,
        provider: SandboxProvider,
        handle: SandboxHandle,
        spec: EnvSpec,
        *,
        mode: Literal["git_patch", "file_globs", "state_snapshot"] = "git_patch",
        globs: Sequence[str] = (),
        workdir: str | None = None,
        timeout_sec: float = 30.0,
    ) -> ExtractedArtifact:
        """Pull scoped git diff or file globs, stripping any verifier-path modifications."""
        if mode == "git_patch":
            return await self._extract_git_patch(
                provider, handle, spec, workdir=workdir, timeout_sec=timeout_sec
            )
        if mode in {"file_globs", "state_snapshot"}:
            return await self._extract_file_globs(
                provider,
                handle,
                spec,
                mode=mode,
                globs=globs,
                workdir=workdir,
                timeout_sec=timeout_sec,
            )
        raise ValueError(f"Unsupported artifact extraction mode: {mode}")

    async def _extract_git_patch(
        self,
        provider: SandboxProvider,
        handle: SandboxHandle,
        spec: EnvSpec,
        *,
        workdir: str | None,
        timeout_sec: float,
    ) -> ExtractedArtifact:
        # Mark newly created untracked files intent-to-add so `git diff HEAD` captures them too.
        await provider.exec(
            handle,
            ["git", "add", "-N", "--", "."],
            timeout_sec=timeout_sec,
            workdir=workdir,
        )
        result = await provider.exec(
            handle,
            ["git", "diff", "--binary", "HEAD"],
            timeout_sec=timeout_sec,
            workdir=workdir,
        )
        if result.exit_code != 0:
            # Minimal images (e.g. python:3.12-slim) or .git-stripped workspaces lack git/HEAD;
            # fall back cleanly to scoped workspace file extraction with verifier-path filtering.
            return await self._extract_file_globs(
                provider,
                handle,
                spec,
                mode="file_globs",
                globs=(),
                workdir=workdir,
                timeout_sec=timeout_sec,
            )
        raw_patch = result.stdout.encode("utf-8", errors="surrogateescape")
        sanitized, rejected = filter_git_patch(raw_patch, spec.verifier_paths)
        digest = compute_artifact_sha256("git_patch", sanitized, {})
        return ExtractedArtifact(
            mode="git_patch",
            patch_bytes=sanitized,
            files={},
            sha256=digest,
            rejected_paths=rejected,
        )

    async def _extract_file_globs(
        self,
        provider: SandboxProvider,
        handle: SandboxHandle,
        spec: EnvSpec,
        *,
        mode: Literal["file_globs", "state_snapshot"],
        globs: Sequence[str],
        workdir: str | None,
        timeout_sec: float,
    ) -> ExtractedArtifact:
        patterns = list(globs) or ["**/*"]
        skip_dirs = sorted(
            {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "node_modules"}
        )
        script = (
            "import base64, glob, json, os\n"
            f"patterns = {json.dumps(patterns)}\n"
            f"skip_dirs = set({json.dumps(skip_dirs)})\n"
            "out = {}\n"
            "for pat in patterns:\n"
            "    for p in sorted(glob.glob(pat, recursive=True)):\n"
            "        parts = set(p.replace('\\\\', '/').split('/'))\n"
            "        if parts & skip_dirs:\n"
            "            continue\n"
            "        if os.path.isfile(p) and not os.path.islink(p):\n"
            "            try:\n"
            "                if os.path.getsize(p) > 2 * 1024 * 1024:\n"
            "                    continue\n"
            "                with open(p, 'rb') as fh:\n"
            "                    out[p] = base64.b64encode(fh.read()).decode('ascii')\n"
            "            except OSError:\n"
            "                continue\n"
            "print(json.dumps(out))\n"
        )
        result = await provider.exec(
            handle,
            ["python3", "-c", script],
            timeout_sec=timeout_sec,
            workdir=workdir,
        )
        raw_files: dict[str, Any] = {}
        for line in reversed(result.stdout.splitlines()):
            cleaned = line.strip()
            if cleaned.startswith("{"):
                try:
                    parsed: Any = json.loads(cleaned)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    raw_files = {str(k): v for k, v in cast(dict[Any, Any], parsed).items()}
                    break

        files: dict[str, bytes] = {}
        rejected: set[str] = set()
        for rel_path, b64_val in sorted(raw_files.items()):
            if not isinstance(b64_val, str):
                continue
            norm = _normalize_rel_path(rel_path)
            if is_protected_verifier_path(rel_path, spec.verifier_paths):
                rejected.add(norm or rel_path)
                continue
            files[norm] = base64.b64decode(b64_val.encode("ascii"))

        digest = compute_artifact_sha256(mode, None, files)
        return ExtractedArtifact(
            mode=mode,
            patch_bytes=None,
            files=files,
            sha256=digest,
            rejected_paths=sorted(rejected),
        )
