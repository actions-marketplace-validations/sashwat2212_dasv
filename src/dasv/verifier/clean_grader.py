"""Clean-Image Out-of-Container Verifier Grader (SDD-02 §1.2, FEAT-08)."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

from pydantic import BaseModel, Field

from dasv.core.config import SandboxConfig
from dasv.core.errors import RuntimeSandboxError
from dasv.core.schemas import EnvSpec
from dasv.sandbox.provider import (
    ContainerConfig,
    SandboxHandle,
    SandboxProvider,
)
from dasv.verifier.artifact_extractor import (
    ArtifactExtractor,
    ExtractedArtifact,
    is_protected_verifier_path,
)

VERIFIER_MOUNT_PATH = "/dasv_verifier"
_REWARD_KV_RE = re.compile(r"(?:^|\s)(?:DASV_REWARD|REWARD)\s*=\s*([0-9]+(?:\.[0-9]+)?)\b")


class GradeOutcome(BaseModel):
    """Deterministic grading outcome produced by a clean Verification Plane container."""

    reward: float
    passed: bool
    verifier_exit_code: int
    stdout_truncated: str
    stderr_truncated: str
    duration_ms: float
    peak_memory_mb: float | None = None
    tamper_attempts: list[str] = Field(default_factory=list)
    emulated_arch: bool = False


def resolve_verifier_host_dir(spec: EnvSpec) -> Path | None:
    """Locate the host-side verifier/tests directory to mount read-only (:ro)."""
    root = spec.root_path.resolve()
    tests_dir = root / "tests"
    if tests_dir.is_dir():
        return tests_dir

    for raw_vp in spec.verifier_paths:
        cleaned = raw_vp.strip().lstrip("./").lstrip("/")
        if not cleaned or ".." in Path(cleaned).parts:
            continue
        candidate = (root / cleaned).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if candidate.is_dir():
            return candidate
        if candidate.is_file() and candidate.parent != root:
            return candidate.parent
    return None


def resolve_verifier_command(
    spec: EnvSpec,
    host_verifier_dir: Path | None,
    mount_path: str = VERIFIER_MOUNT_PATH,
) -> list[str]:
    """Determine the verifier entrypoint command executed inside Verification Plane."""
    explicit_paths: list[str] = []
    harbor_cfg = spec.metadata.get("harbor_config")
    if isinstance(harbor_cfg, dict):
        verifier_tbl = cast(dict[str, Any], harbor_cfg).get("verifier")
        if isinstance(verifier_tbl, dict):
            v_dict = cast(dict[str, Any], verifier_tbl)
            cmd = v_dict.get("command")
            if isinstance(cmd, list) and cmd:
                cmd_items = cast(list[Any], cmd)
                if all(isinstance(x, str) for x in cmd_items):
                    return [str(x) for x in cmd_items]
            if isinstance(cmd, str) and cmd.strip():
                return ["sh", "-c", cmd.strip()]
            for key in ("test_path", "entrypoint"):
                val = v_dict.get(key)
                if isinstance(val, str) and val.strip():
                    explicit_paths.append(val.strip())

    def _cmd_for_candidate(raw_vp: str, resolved_verifier_dir: Path) -> list[str] | None:
        cleaned = raw_vp.strip().lstrip("./").lstrip("/")
        if not cleaned or ".." in Path(cleaned).parts:
            return None
        full = (spec.root_path / cleaned).resolve()
        if not full.is_file():
            return None
        try:
            rel = full.relative_to(resolved_verifier_dir).as_posix()
        except ValueError:
            return None
        if rel.endswith(".sh"):
            return ["sh", f"{mount_path}/{rel}"]
        if rel.endswith(".py"):
            return ["pytest", f"{mount_path}/{rel}"]
        return None

    if host_verifier_dir is not None:
        resolved_dir = host_verifier_dir.resolve()
        for raw_vp in explicit_paths:
            matched = _cmd_for_candidate(raw_vp, resolved_dir)
            if matched is not None:
                return matched
        if (resolved_dir / "test.sh").is_file():
            return ["sh", f"{mount_path}/test.sh"]
        for raw_vp in spec.verifier_paths:
            matched = _cmd_for_candidate(raw_vp, resolved_dir)
            if matched is not None:
                return matched
        return ["pytest", mount_path]

    return ["pytest", mount_path]


def parse_verifier_reward(stdout: str, exit_code: int, timed_out: bool) -> float:
    """Parse normalized reward in [0.0, 1.0] from verifier stdout and exit code."""
    if timed_out:
        return 0.0

    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed: Any = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                reward_raw = cast(dict[str, Any], parsed).get("reward")
                if isinstance(reward_raw, (int, float)) and not isinstance(reward_raw, bool):
                    return max(0.0, min(1.0, float(reward_raw)))
        match = _REWARD_KV_RE.search(stripped)
        if match:
            return max(0.0, min(1.0, float(match.group(1))))

    return 1.0 if exit_code == 0 else 0.0


def _resolve_container_image(spec: EnvSpec) -> str:
    image = spec.image_ref or spec.image_digest
    if not image or not image.strip():
        raise RuntimeSandboxError(
            f"EnvSpec '{spec.task_id}' declares neither image_ref nor image_digest; "
            "cannot launch clean-image container."
        )
    return image.strip()


def find_env_dockerfile(spec: EnvSpec) -> Path | None:
    """Locate a buildable Dockerfile under `spec.root_path` (`environment/Dockerfile` first)."""
    for candidate in (
        spec.root_path / "environment" / "Dockerfile",
        spec.root_path / "Dockerfile",
    ):
        if candidate.is_file():
            return candidate
    return None


def _resolve_build_context(spec: EnvSpec, dockerfile: Path) -> Path:
    """Choose Docker build context (`environment/` if it holds context files, else repo root)."""
    parent = dockerfile.parent
    if parent == spec.root_path:
        return parent
    siblings = [p for p in parent.iterdir() if p.name != dockerfile.name]
    if siblings:
        return parent
    return spec.root_path


async def ensure_container_image(
    spec: EnvSpec,
    provider: SandboxProvider,
    run_id: str,
    *,
    oci_digests: dict[str, str] | None = None,
    force_build: bool = False,
) -> EnvSpec:
    """Build `spec`'s Dockerfile when needed so containers contain the real task workspace.

    Skips building if `spec` declares an explicit prebuilt image (or has already been built),
    unless `force_build=True`.
    """
    if not force_build and spec.metadata.get("dasv_built_image"):
        if oci_digests is not None and spec.image_ref and spec.image_digest:
            oci_digests[spec.image_ref] = spec.image_digest
        return spec

    dockerfile = find_env_dockerfile(spec)
    needs_build = force_build or (
        dockerfile is not None
        and (
            (not spec.image_ref and not spec.image_digest)
            or spec.metadata.get("image_explicit") is False
        )
    )
    if not needs_build:
        if spec.image_ref or spec.image_digest:
            if oci_digests is not None and spec.image_ref and spec.image_digest:
                oci_digests[spec.image_ref] = spec.image_digest
            return spec
        raise RuntimeSandboxError(
            f"EnvSpec '{spec.task_id}' declares neither image_ref/image_digest nor a "
            "buildable Dockerfile; cannot execute container checks."
        )

    assert dockerfile is not None
    safe_task = re.sub(r"[^a-z0-9._-]+", "-", spec.task_id.lower()).strip(".-") or "env"
    safe_run = re.sub(r"[^a-z0-9._-]+", "-", run_id.lower()).strip(".-") or "run"
    tag = f"dasv-{safe_task}:{safe_run}"
    context_dir = _resolve_build_context(spec, dockerfile)
    built_digest = await provider.build_image(context_dir, dockerfile, tag, platform=None)
    if oci_digests is not None:
        oci_digests[tag] = built_digest
    new_meta = dict(spec.metadata)
    new_meta["dasv_built_image"] = tag
    new_meta["image_explicit"] = True
    return spec.model_copy(
        update={
            "image_ref": tag,
            "image_digest": built_digest,
            "metadata": new_meta,
        }
    )


class CleanImageGrader:
    """Grades extracted agent artifacts strictly inside a fresh Verification Plane container."""

    def __init__(
        self,
        provider: SandboxProvider,
        *,
        sandbox_config: SandboxConfig | None = None,
        extractor: ArtifactExtractor | None = None,
        strict_tamper_zero: bool = True,
    ) -> None:
        self._provider = provider
        self._sandbox_config = sandbox_config or SandboxConfig()
        self._extractor = extractor or ArtifactExtractor()
        self._strict_tamper_zero = strict_tamper_zero

    @property
    def provider(self) -> SandboxProvider:
        return self._provider

    @property
    def extractor(self) -> ArtifactExtractor:
        return self._extractor

    def _build_container_config(
        self,
        spec: EnvSpec,
        *,
        plane: Literal["execution", "verification"],
        bind_mounts: list[tuple[Path, str, Literal["ro", "rw"]]] | None = None,
        workdir: str | None = None,
    ) -> ContainerConfig:
        image = _resolve_container_image(spec)
        return ContainerConfig(
            image=image,
            network_mode="none",
            cpus=self._sandbox_config.cpus,
            memory=self._sandbox_config.memory,
            pids_limit=self._sandbox_config.pids_limit,
            bind_mounts=list(bind_mounts or []),
            plane=plane,
            workdir=workdir,
        )

    async def _apply_artifact(
        self,
        handle: SandboxHandle,
        spec: EnvSpec,
        artifact: ExtractedArtifact,
        *,
        workdir: str | None,
        timeout_sec: float,
    ) -> tuple[bool, str]:
        """Apply extracted patch or files into the clean Verification Plane workspace."""
        if artifact.mode == "git_patch":
            if not artifact.patch_bytes:
                return True, ""
            patch_b64 = base64.b64encode(artifact.patch_bytes).decode("ascii")
            apply_script = (
                "import base64, os, re, shutil, subprocess, sys, tempfile\n"
                f"data = base64.b64decode({patch_b64!r})\n"
                "if shutil.which('git'):\n"
                "    fd, p = tempfile.mkstemp(prefix='dasv_patch_', suffix='.diff', dir='/tmp')\n"
                "    os.write(fd, data); os.close(fd)\n"
                "    res = subprocess.run(['git', 'apply', '--whitespace=nowarn', p], "
                "capture_output=True, text=True)\n"
                "    os.unlink(p)\n"
                "    sys.stderr.write(res.stderr)\n"
                "    sys.exit(res.returncode)\n"
                "# Pure-Python unified diff fallback when image lacks `git` binary.\n"
                "text = data.decode('utf-8', errors='replace')\n"
                "lines = text.splitlines(keepends=True)\n"
                "i = 0; n = len(lines)\n"
                "hunk_re = re.compile(r'^@@ -(\\d+)(?:,(\\d+))? \\+(\\d+)(?:,(\\d+))? @@')\n"
                "while i < n:\n"
                "    if lines[i].startswith('--- '):\n"
                "        old_p = lines[i][4:].strip().split('\\t')[0].strip('\"')\n"
                "        i += 1\n"
                "        if i >= n or not lines[i].startswith('+++ '):\n"
                "            continue\n"
                "        new_p = lines[i][4:].strip().split('\\t')[0].strip('\"')\n"
                "        i += 1\n"
                "        target = new_p[2:] if new_p.startswith('b/') else new_p\n"
                "        if target == '/dev/null':\n"
                "            del_p = old_p[2:] if old_p.startswith('a/') else old_p\n"
                "            if os.path.exists(del_p):\n"
                "                os.unlink(del_p)\n"
                "            continue\n"
                "        orig = open(target, 'r', encoding='utf-8').splitlines(keepends=True) "
                "if os.path.exists(target) else []\n"
                "        out = []; cur = 0\n"
                "        while i < n and not lines[i].startswith(('diff --git ', '--- ')):\n"
                "            m = hunk_re.match(lines[i])\n"
                "            if not m:\n"
                "                i += 1; continue\n"
                "            start = max(0, int(m.group(1)) - 1)\n"
                "            out.extend(orig[cur:start])\n"
                "            cur = start; i += 1\n"
                "            stops = ('@@ ', 'diff --git ', '--- ')\n"
                "            while i < n and not lines[i].startswith(stops):\n"
                "                hl = lines[i]; i += 1\n"
                "                if hl.startswith(' '):\n"
                "                    out.append(hl[1:]); cur += 1\n"
                "                elif hl.startswith('-'):\n"
                "                    cur += 1\n"
                "                elif hl.startswith('+'):\n"
                "                    out.append(hl[1:])\n"
                "        out.extend(orig[cur:])\n"
                "        parent = os.path.dirname(target)\n"
                "        if parent:\n"
                "            os.makedirs(parent, exist_ok=True)\n"
                "        with open(target, 'w', encoding='utf-8') as fh:\n"
                "            fh.writelines(out)\n"
                "    else:\n"
                "        i += 1\n"
            )
            res = await self._provider.exec(
                handle,
                ["python3", "-c", apply_script],
                timeout_sec=min(30.0, timeout_sec),
                workdir=workdir,
            )
            if res.timed_out or res.exit_code != 0:
                return False, res.stderr or f"git apply failed with exit {res.exit_code}"
            return True, ""

        if artifact.files:
            encoded_files: dict[str, str] = {}
            for rel_path, content in sorted(artifact.files.items()):
                if is_protected_verifier_path(rel_path, spec.verifier_paths):
                    continue
                encoded_files[rel_path] = base64.b64encode(content).decode("ascii")
            if not encoded_files:
                return True, ""
            write_script = (
                "import base64, json, os\n"
                f"payload = json.loads({json.dumps(json.dumps(encoded_files))})\n"
                "for rel, b64 in payload.items():\n"
                "    parent = os.path.dirname(rel)\n"
                "    if parent:\n"
                "        os.makedirs(parent, exist_ok=True)\n"
                "    with open(rel, 'wb') as fh:\n"
                "        fh.write(base64.b64decode(b64))\n"
            )
            res = await self._provider.exec(
                handle,
                ["python3", "-c", write_script],
                timeout_sec=min(30.0, timeout_sec),
                workdir=workdir,
            )
            if res.timed_out or res.exit_code != 0:
                return False, res.stderr or f"artifact write failed with exit {res.exit_code}"
        return True, ""

    async def grade(
        self,
        spec: EnvSpec,
        artifact: ExtractedArtifact,
        run_id: str,
        *,
        timeout_sec: float | None = None,
        workdir: str | None = "/app",
    ) -> GradeOutcome:
        """Boot a fresh Verification Plane container, mount tests :ro, apply artifact & grade."""
        effective_timeout = (
            float(timeout_sec)
            if timeout_sec is not None
            else float(self._sandbox_config.episode_timeout_sec)
        )
        host_verifier_dir = resolve_verifier_host_dir(spec)
        bind_mounts: list[tuple[Path, str, Literal["ro", "rw"]]] = []
        if host_verifier_dir is not None:
            bind_mounts.append((host_verifier_dir, VERIFIER_MOUNT_PATH, "ro"))

        cfg = self._build_container_config(
            spec,
            plane="verification",
            bind_mounts=bind_mounts,
            workdir=workdir,
        )
        handle = await self._provider.start(cfg, run_id)
        try:
            applied, apply_err = await self._apply_artifact(
                handle,
                spec,
                artifact,
                workdir=workdir,
                timeout_sec=effective_timeout,
            )
            if not applied:
                return GradeOutcome(
                    reward=0.0,
                    passed=False,
                    verifier_exit_code=1,
                    stdout_truncated="",
                    stderr_truncated=apply_err,
                    duration_ms=0.0,
                    peak_memory_mb=None,
                    tamper_attempts=list(artifact.rejected_paths),
                    emulated_arch=handle.emulated_arch,
                )

            verifier_cmd = resolve_verifier_command(spec, host_verifier_dir, VERIFIER_MOUNT_PATH)
            exec_res = await self._provider.exec(
                handle,
                verifier_cmd,
                timeout_sec=effective_timeout,
                workdir=workdir,
            )
            reward = parse_verifier_reward(exec_res.stdout, exec_res.exit_code, exec_res.timed_out)

            if self._strict_tamper_zero and artifact.rejected_paths:
                reward = 0.0

            passed = (
                not exec_res.timed_out
                and reward >= 1.0 - 1e-9
                and not (self._strict_tamper_zero and bool(artifact.rejected_paths))
            )
            return GradeOutcome(
                reward=reward,
                passed=passed,
                verifier_exit_code=exec_res.exit_code,
                stdout_truncated=exec_res.stdout,
                stderr_truncated=exec_res.stderr,
                duration_ms=exec_res.duration_ms,
                peak_memory_mb=exec_res.peak_memory_mb,
                tamper_attempts=list(artifact.rejected_paths),
                emulated_arch=handle.emulated_arch,
            )
        finally:
            await self._provider.stop_and_remove(handle)

    async def execute_and_grade(
        self,
        spec: EnvSpec,
        run_id: str,
        *,
        agent_cmd: list[str] | None = None,
        script_bytes: bytes | None = None,
        timeout_sec: float | None = None,
        workdir: str | None = "/app",
        extract_mode: Literal["git_patch", "file_globs", "state_snapshot"] = "git_patch",
        globs: tuple[str, ...] = (),
    ) -> tuple[GradeOutcome, ExtractedArtifact]:
        """Run an episode in Execution Plane, extract scoped artifacts, and grade in clean plane."""
        effective_timeout = (
            float(timeout_sec)
            if timeout_sec is not None
            else float(self._sandbox_config.episode_timeout_sec)
        )
        exec_cfg = self._build_container_config(spec, plane="execution", workdir=workdir)
        exec_handle = await self._provider.start(exec_cfg, run_id)
        try:
            if script_bytes is not None:
                b64 = base64.b64encode(script_bytes).decode("ascii")
                runner_py = (
                    "import base64, os, subprocess, sys, tempfile; "
                    f"data = base64.b64decode({b64!r}); "
                    "fd, p = tempfile.mkstemp(prefix='dasv_ep_', suffix='.sh', dir='/tmp'); "
                    "os.write(fd, data); os.close(fd); os.chmod(p, 0o700); "
                    "res = subprocess.run(['sh', p], capture_output=False); "
                    "os.unlink(p); sys.exit(res.returncode)"
                )
                await self._provider.exec(
                    exec_handle,
                    ["python3", "-c", runner_py],
                    timeout_sec=effective_timeout,
                    workdir=workdir,
                )
            elif agent_cmd is not None:
                await self._provider.exec(
                    exec_handle,
                    agent_cmd,
                    timeout_sec=effective_timeout,
                    workdir=workdir,
                )

            artifact = await self._extractor.extract(
                self._provider,
                exec_handle,
                spec,
                mode=extract_mode,
                globs=globs,
                workdir=workdir,
                timeout_sec=min(30.0, effective_timeout),
            )
            exec_emulated = exec_handle.emulated_arch
        finally:
            await self._provider.stop_and_remove(exec_handle)

        outcome = await self.grade(
            spec,
            artifact,
            run_id,
            timeout_sec=effective_timeout,
            workdir=workdir,
        )
        if exec_emulated and not outcome.emulated_arch:
            outcome = outcome.model_copy(update={"emulated_arch": True})
        return outcome, artifact


class GradingContext:
    """Lightweight runtime bundle passed as `sandbox` into adapter oracle/noop hooks."""

    def __init__(
        self,
        grader: CleanImageGrader,
        *,
        run_id: str | None = None,
        timeout_sec: float | None = None,
        workdir: str | None = "/app",
    ) -> None:
        self.grader = grader
        self.run_id = run_id or f"dasv-{uuid4().hex[:12]}"
        self.timeout_sec = timeout_sec
        self.workdir = workdir


def coerce_grading_context(sandbox: Any) -> GradingContext:
    """Resolve `GradingContext`, `CleanImageGrader`, or `SandboxProvider` into GradingContext."""
    if isinstance(sandbox, GradingContext):
        return sandbox
    if isinstance(sandbox, CleanImageGrader):
        return GradingContext(sandbox)
    if isinstance(sandbox, SandboxProvider):
        return GradingContext(CleanImageGrader(sandbox))
    raise RuntimeSandboxError(
        "Adapter execution requires a GradingContext, CleanImageGrader, or SandboxProvider; "
        f"got {type(sandbox).__name__}."
    )


def outcome_to_details(outcome: GradeOutcome, artifact: ExtractedArtifact) -> dict[str, Any]:
    """Serialize GradeOutcome + ExtractedArtifact metadata into adapter hook detail dict."""
    return {
        "passed": outcome.passed,
        "verifier_exit_code": outcome.verifier_exit_code,
        "duration_ms": outcome.duration_ms,
        "peak_memory_mb": outcome.peak_memory_mb,
        "emulated_arch": outcome.emulated_arch,
        "tamper_attempts": list(outcome.tamper_attempts),
        "artifact_sha256": artifact.sha256,
        "stdout_truncated": outcome.stdout_truncated,
        "stderr_truncated": outcome.stderr_truncated,
    }
