"""Tier 0 static & Tier 1 runtime Verifier Boundary & Hermeticity probes (SDD-01)."""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

from dasv.core.schemas import AuditFinding, EnvSpec, EnvStandard, Severity
from dasv.sandbox.provider import SandboxHandle, SandboxProvider, current_host_arch

_COPY_ADD_RE = re.compile(r"^\s*(?:COPY|ADD)\s+(.+)$", re.IGNORECASE)
_FROM_RE = re.compile(
    r"^\s*FROM\s+(?:--platform=\S+\s+)?(?P<image>\S+)(?:\s+AS\s+(?P<alias>\S+))?",
    re.IGNORECASE,
)
_ENV_ARG_RE = re.compile(r"^\s*(?:ENV|ARG)\s+(.+)$", re.IGNORECASE)
_CMD_ENTRY_RE = re.compile(r"^\s*(?:CMD|ENTRYPOINT)\s+(.+)$", re.IGNORECASE)

_SENSITIVE_VERIFIER_TOKENS = ("tests", "test.sh", "grader", "solution", "eval")

_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("OpenAI secret key", re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")),
    ("GitHub PAT", re.compile(r"\bghp_[A-Za-z0-9]{10,}\b")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{10,}\b")),
    (
        "Hardcoded provider API key",
        re.compile(
            r"\b(?:OPENAI_API_KEY|ANTHROPIC_API_KEY|HF_TOKEN|GEMINI_API_KEY|DEEPSEEK_API_KEY)"
            r"\s*(?:=|\s+)\s*['\"]?(?![$\s\"']|YOUR_|REPLACE_|<)[^\s'\"]{6,}",
            re.IGNORECASE,
        ),
    ),
]

_LIVE_INSTALL_RE = re.compile(
    r"(?:\b(?:pip3?|uv\s+pip)\s+install\b|\bapt(?:-get)?\s+install\b|\bnpm\s+install\b"
    r"|\byarn\s+(?:add|install)\b|\b(?:curl|wget)\b[^|\n]+\|\s*(?:ba)?sh\b)",
    re.IGNORECASE,
)

_RUNTIME_SCRIPT_NAMES = {
    "entrypoint.sh",
    "start.sh",
    "run.sh",
    "setup.sh",
    "boot.sh",
}

_SAFE_NETWORK_MODES = {"none", "no-network"}
_SECRET_CONFIG_FILENAMES = (
    "task.toml",
    "openenv.yaml",
    "openenv.yml",
    "openenv.toml",
    "hud.toml",
)

_SENTINEL_PATHS = ("/tmp/.dasv_sentinel", ".dasv_dirty")


def _find_dockerfiles(root: Path) -> list[Path]:
    found: list[Path] = []
    for candidate_dir in (root, root / "environment"):
        if not candidate_dir.is_dir():
            continue
        for p in sorted(candidate_dir.iterdir()):
            if p.is_file() and p.name.startswith("Dockerfile"):
                found.append(p)
    return found


def _rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _probe_dockerfile_rules(spec: EnvSpec, dockerfiles: list[Path]) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    extra_verifier_tokens = [
        p.strip("/").split("/")[0].lower() for p in spec.verifier_paths if p.strip("/")
    ]
    forbidden_tokens = tuple(dict.fromkeys([*_SENSITIVE_VERIFIER_TOKENS, *extra_verifier_tokens]))

    for dockerfile in dockerfiles:
        rel_path = _rel(spec.root_path, dockerfile)
        lines = dockerfile.read_text(encoding="utf-8", errors="replace").splitlines()
        stage_aliases: set[str] = set()

        for lineno, raw_line in enumerate(lines, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            # RULE-HERM-001: Unpinned FROM image (no @sha256: digest or uses :latest)
            from_match = _FROM_RE.match(stripped)
            if from_match:
                image = from_match.group("image")
                alias = from_match.group("alias")
                if alias:
                    stage_aliases.add(alias.lower())
                if (
                    image.lower() != "scratch"
                    and image.lower() not in stage_aliases
                    and ("@sha256:" not in image or image.lower().endswith(":latest"))
                ):
                    findings.append(
                        AuditFinding(
                            rule_id="RULE-HERM-001",
                            title="Unpinned base image without @sha256 digest",
                            severity=Severity.HIGH,
                            tier=0,
                            source="dasv",
                            file_path=rel_path,
                            line_number=lineno,
                            evidence=f"FROM {image} does not pin an immutable @sha256 digest.",
                            remediation=(
                                "Pin the base image to an immutable OCI digest "
                                "(e.g., FROM python:3.12-slim@sha256:...)."
                            ),
                        )
                    )
                continue

            # RULE-BNDY-001: COPY/ADD verifier/test/solution paths into agent image
            copy_match = _COPY_ADD_RE.match(stripped)
            if copy_match:
                rest = copy_match.group(1)
                parts = [tok for tok in rest.split() if not tok.startswith("--")]
                sources = parts[:-1] if len(parts) >= 2 else parts
                for src in sources:
                    norm = src.strip("\"'").lstrip("./").lower()
                    if any(
                        norm == tok
                        or norm.startswith(f"{tok}/")
                        or f"/{tok}/" in f"/{norm}/"
                        or norm.endswith(f"/{tok}")
                        for tok in forbidden_tokens
                    ):
                        findings.append(
                            AuditFinding(
                                rule_id="RULE-BNDY-001",
                                title="Verifier or solution files copied into agent image",
                                severity=Severity.CRITICAL,
                                tier=0,
                                source="dasv",
                                file_path=rel_path,
                                line_number=lineno,
                                evidence=(f"Dockerfile copies sensitive path: {stripped}"),
                                remediation=(
                                    "Remove tests/, grader/, and solution/ from the agent image; "
                                    "mount verifiers read-only in a clean Verification Plane."
                                ),
                            )
                        )
                        break

            # RULE-BNDY-003: Hardcoded secret in Dockerfile ENV/ARG
            if _ENV_ARG_RE.match(stripped):
                for label, pattern in _SECRET_PATTERNS:
                    if pattern.search(stripped):
                        findings.append(
                            AuditFinding(
                                rule_id="RULE-BNDY-003",
                                title=f"Hardcoded secret detected in Dockerfile ({label})",
                                severity=Severity.CRITICAL,
                                tier=0,
                                source="dasv",
                                file_path=rel_path,
                                line_number=lineno,
                                evidence=f"Matched {label} pattern in {rel_path}:{lineno}.",
                                remediation=(
                                    "Remove API keys/tokens from Dockerfile ENV/ARG layers; "
                                    "never expose host credentials inside untrusted sandboxes."
                                ),
                            )
                        )
                        break

            # RULE-HERM-002: Live package install inside CMD/ENTRYPOINT
            if _CMD_ENTRY_RE.match(stripped) and _LIVE_INSTALL_RE.search(stripped):
                findings.append(
                    AuditFinding(
                        rule_id="RULE-HERM-002",
                        title="Live package installation executed at container boot",
                        severity=Severity.HIGH,
                        tier=0,
                        source="dasv",
                        file_path=rel_path,
                        line_number=lineno,
                        evidence=f"Runtime boot instruction installs packages: {stripped}",
                        remediation=(
                            "Move package installations to Docker build layers (RUN) and vendor "
                            "dependencies for offline --network=none execution."
                        ),
                    )
                )

    return findings


def _probe_network_isolation(spec: EnvSpec, dockerfiles: list[Path]) -> list[AuditFinding]:
    is_containerized = (
        spec.standard in {EnvStandard.HARBOR, EnvStandard.OPENENV}
        or bool(dockerfiles)
        or spec.image_ref is not None
    )
    if not is_containerized:
        return []

    raw_mode = spec.metadata.get("network_mode")
    normalized = str(raw_mode).strip().lower() if raw_mode is not None else ""
    if normalized in _SAFE_NETWORK_MODES:
        return []

    config_file: str | None = None
    for candidate in _SECRET_CONFIG_FILENAMES:
        if (spec.root_path / candidate).is_file():
            config_file = candidate
            break

    evidence = (
        "Environment config omits network_mode = 'no-network'/'none'."
        if not normalized
        else f"Environment config enables non-hermetic network_mode='{raw_mode}'."
    )
    return [
        AuditFinding(
            rule_id="RULE-BNDY-002",
            title="Missing or unsafe network isolation configuration",
            severity=Severity.CRITICAL,
            tier=0,
            source="dasv",
            file_path=config_file,
            line_number=1 if config_file else None,
            evidence=evidence,
            remediation=(
                "Configure network_mode = 'no-network' (Harbor) or 'none' (--network=none) "
                "to prevent agents from exfiltrating or fetching solutions online."
            ),
        )
    ]


def _probe_config_secrets(spec: EnvSpec) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    for filename in _SECRET_CONFIG_FILENAMES:
        cfg_path = spec.root_path / filename
        if not cfg_path.is_file():
            continue
        for lineno, line in enumerate(
            cfg_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            for label, pattern in _SECRET_PATTERNS:
                if pattern.search(stripped):
                    findings.append(
                        AuditFinding(
                            rule_id="RULE-BNDY-003",
                            title=f"Hardcoded secret detected in {filename} ({label})",
                            severity=Severity.CRITICAL,
                            tier=0,
                            source="dasv",
                            file_path=filename,
                            line_number=lineno,
                            evidence=f"Matched {label} pattern in {filename}:{lineno}.",
                            remediation=(
                                "Scrub API keys and tokens from task manifests and configs."
                            ),
                        )
                    )
                    break
    return findings


def _probe_git_history_leak(spec: EnvSpec) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    git_dirs: list[Path] = []
    for candidate in (
        spec.root_path / ".git",
        spec.root_path / "environment" / ".git",
        spec.root_path / "workspace" / ".git",
    ):
        if candidate.exists():
            git_dirs.append(candidate)

    base_commit = spec.metadata.get("base_commit")

    for git_dir in git_dirs:
        repo_dir = git_dir.parent
        rel_git = _rel(spec.root_path, git_dir)
        reasons: list[str] = []

        reflog_head = git_dir / "logs" / "HEAD"
        if (
            reflog_head.is_file()
            and reflog_head.read_text(encoding="utf-8", errors="replace").strip()
        ):
            reasons.append("non-empty .git/logs/HEAD reflog present")

        try:
            log_proc = subprocess.run(
                ["git", "log", "--all", "--name-only", "--pretty=format:%H"],
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if log_proc.returncode == 0:
                lines = [ln.strip() for ln in log_proc.stdout.splitlines() if ln.strip()]
                sensitive_hits = [
                    ln
                    for ln in lines
                    if any(
                        tok in ln.lower() for tok in ("solution/", "tests/", "grader/", "solve.sh")
                    )
                ]
                if sensitive_hits:
                    reasons.append(
                        f"git history references sensitive paths ({', '.join(sensitive_hits[:3])})"
                    )

            if isinstance(base_commit, str) and base_commit.strip():
                after_proc = subprocess.run(
                    ["git", "rev-list", "--count", f"{base_commit.strip()}..HEAD"],
                    cwd=str(repo_dir),
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if after_proc.returncode == 0 and int(after_proc.stdout.strip() or "0") > 0:
                    reasons.append(
                        f"{after_proc.stdout.strip()} commit(s) exist after base_commit "
                        f"{base_commit.strip()[:12]}"
                    )
        except (OSError, ValueError, subprocess.SubprocessError):
            pass

        if not reasons:
            reasons.append("unstripped .git directory found inside task workspace")

        findings.append(
            AuditFinding(
                rule_id="RULE-BNDY-004",
                title="Unstripped .git history or reflog leak in task workspace",
                severity=Severity.CRITICAL,
                tier=0,
                source="dasv",
                file_path=rel_git,
                line_number=None,
                evidence="; ".join(reasons) + ".",
                remediation=(
                    "Strip .git reflogs, tags, and future commits beyond base_commit "
                    "(or remove .git entirely from the agent container image)."
                ),
            )
        )

    return findings


def _probe_runtime_boot_installs(spec: EnvSpec) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    for script in sorted(spec.root_path.rglob("*.sh")):
        if script.name.lower() not in _RUNTIME_SCRIPT_NAMES:
            continue
        rel_path = _rel(spec.root_path, script)
        for lineno, line in enumerate(
            script.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if _LIVE_INSTALL_RE.search(stripped):
                findings.append(
                    AuditFinding(
                        rule_id="RULE-HERM-002",
                        title="Live package install inside runtime boot script",
                        severity=Severity.HIGH,
                        tier=0,
                        source="dasv",
                        file_path=rel_path,
                        line_number=lineno,
                        evidence=f"Runtime script executes live install: {stripped}",
                        remediation=(
                            "Install all dependencies during Docker image build (RUN) so episodes "
                            "run deterministically with --network=none."
                        ),
                    )
                )
                break
    return findings


def run_tier0_static_probes(spec: EnvSpec) -> list[AuditFinding]:
    """Execute all Tier 0 ($0) static boundary and hermeticity rules on `spec`."""
    dockerfiles = _find_dockerfiles(spec.root_path)
    findings: list[AuditFinding] = []
    findings.extend(_probe_dockerfile_rules(spec, dockerfiles))
    findings.extend(_probe_network_isolation(spec, dockerfiles))
    findings.extend(_probe_config_secrets(spec))
    findings.extend(_probe_git_history_leak(spec))
    findings.extend(_probe_runtime_boot_installs(spec))
    return findings


# ---------------------------------------------------------------------------
# Runtime Tier 0/1 Boundary & Reset Probes (SDD-01 §4.2)
# ---------------------------------------------------------------------------


def _parse_probe_json(stdout: str) -> dict[str, Any] | None:
    """Extract the last JSON object line emitted by a runtime container probe."""
    for line in reversed(stdout.splitlines()):
        cleaned = line.strip()
        if not cleaned or not cleaned.startswith("{"):
            continue
        try:
            parsed: Any = json.loads(cleaned)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return {str(k): v for k, v in cast(dict[Any, Any], parsed).items()}
    return None


def _candidate_verifier_paths(spec: EnvSpec) -> list[str]:
    candidates: list[str] = []
    for raw in spec.verifier_paths:
        cleaned = raw.strip().lstrip("./").rstrip("/")
        if not cleaned:
            continue
        for prefix in ("", "/app/", "/workspace/", "/"):
            candidate = f"{prefix}{cleaned}"
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


async def probe_writable_verifier_paths(
    provider: SandboxProvider,
    handle: SandboxHandle,
    spec: EnvSpec,
    *,
    timeout_sec: float = 15.0,
) -> list[AuditFinding]:
    """RULE-BNDY-005: Check if verifier paths exist or are writable inside Execution Plane."""
    candidates = _candidate_verifier_paths(spec)
    if not candidates:
        return []

    script = (
        "import json, os; "
        f"paths = {json.dumps(candidates)}; "
        "exists = [p for p in paths if os.path.exists(p)]; "
        "writable = [p for p in exists if os.access(p, os.W_OK)]; "
        "print(json.dumps({'exists': exists, 'writable': writable}))"
    )
    result = await provider.exec(handle, ["python3", "-c", script], timeout_sec=timeout_sec)
    payload = _parse_probe_json(result.stdout)
    if payload is None:
        return []

    exists = [str(p) for p in payload.get("exists", []) if isinstance(p, str)]
    writable = [str(p) for p in payload.get("writable", []) if isinstance(p, str)]
    if not exists and not writable:
        return []

    detail_parts: list[str] = []
    if writable:
        detail_parts.append(f"writable verifier path(s): {', '.join(writable)}")
    if exists:
        detail_parts.append(f"visible verifier path(s): {', '.join(exists)}")

    return [
        AuditFinding(
            rule_id="RULE-BNDY-005",
            title="Verifier or test paths exposed inside agent Execution Plane",
            severity=Severity.CRITICAL,
            tier=1,
            source="dasv",
            file_path=spec.verifier_paths[0] if spec.verifier_paths else None,
            line_number=None,
            evidence="; ".join(detail_parts) + ".",
            remediation=(
                "Remove all verifier/test files from the agent Execution Plane image and "
                "grade artifacts out-of-container with read-only (:ro) verifier mounts."
            ),
        )
    ]


async def probe_egress_and_env_scrub(
    provider: SandboxProvider,
    handle: SandboxHandle,
    spec: EnvSpec,
    *,
    timeout_sec: float = 15.0,
) -> list[AuditFinding]:
    """RULE-BNDY-006: Probe outbound network egress and /proc/self/environ secret leakage."""
    _ = spec
    script = (
        "import json, os, re, socket; "
        "targets = [('1.1.1.1', 53), ('github.com', 443)]; "
        "reachable = []; "
        "for h, p in targets:\n"
        "    try:\n"
        "        s = socket.create_connection((h, p), timeout=1.5)\n"
        "        s.close()\n"
        "        reachable.append(f'{h}:{p}')\n"
        "    except OSError:\n"
        "        pass\n"
        "env_pairs = []; "
        "try:\n"
        "    raw = open('/proc/self/environ', 'rb').read().split(b'\\x00')\n"
        "    for item in raw:\n"
        "        if b'=' in item:\n"
        "            k, v = item.split(b'=', 1)\n"
        "            env_pairs.append(("
        "k.decode('utf-8', 'replace'), v.decode('utf-8', 'replace')))\n"
        "except OSError:\n"
        "    env_pairs = list(os.environ.items())\n"
        "bad_keys = {'OPENAI_API_KEY','ANTHROPIC_API_KEY','HF_TOKEN','HUGGING_FACE_HUB_TOKEN',"
        "'GEMINI_API_KEY','GOOGLE_API_KEY','DEEPSEEK_API_KEY','OPENROUTER_API_KEY',"
        "'AZURE_OPENAI_API_KEY','MISTRAL_API_KEY','XAI_API_KEY',"
        "'AWS_ACCESS_KEY_ID','AWS_SECRET_ACCESS_KEY','AWS_SESSION_TOKEN',"
        "'GITHUB_TOKEN','GH_TOKEN'}; "
        "val_re = re.compile("
        "r'(?:sk-[A-Za-z0-9_-]{12,}|ghp_[A-Za-z0-9]{10,}|hf_[A-Za-z0-9]{10,}|AKIA[0-9A-Z]{16})'); "
        "leaked = sorted({"
        "k for k, v in env_pairs if k.upper() in bad_keys or val_re.search(v)"
        "}); "
        "print(json.dumps({'egress_reachable': reachable, 'leaked_env_keys': leaked}))"
    )
    result = await provider.exec(handle, ["python3", "-c", script], timeout_sec=timeout_sec)
    payload = _parse_probe_json(result.stdout)
    if payload is None:
        return []

    findings: list[AuditFinding] = []
    egress = [str(x) for x in payload.get("egress_reachable", []) if isinstance(x, str)]
    leaked_keys = [str(x) for x in payload.get("leaked_env_keys", []) if isinstance(x, str)]

    if egress:
        findings.append(
            AuditFinding(
                rule_id="RULE-BNDY-006",
                title="Live outbound network egress permitted inside Execution Plane",
                severity=Severity.CRITICAL,
                tier=1,
                source="dasv",
                file_path=None,
                line_number=None,
                evidence=(
                    f"Container established outbound socket connection to: {', '.join(egress)}."
                ),
                remediation=(
                    "Launch agent Execution Plane containers strictly with --network=none "
                    "to block data exfiltration and online answer lookup."
                ),
            )
        )

    if leaked_keys:
        findings.append(
            AuditFinding(
                rule_id="RULE-BNDY-006",
                title="Host/judge secret environment variables leaked into sandbox",
                severity=Severity.CRITICAL,
                tier=1,
                source="dasv",
                file_path=None,
                line_number=None,
                evidence=(
                    "Sensitive environment variable(s) visible in /proc/self/environ: "
                    f"{', '.join(leaked_keys)}."
                ),
                remediation=(
                    "Scrub all provider API keys and host tokens before starting "
                    "sandbox containers."
                ),
            )
        )

    return findings


async def probe_reset_residue(
    provider: SandboxProvider,
    handle: SandboxHandle,
    spec: EnvSpec,
    *,
    reset_fn: Callable[[], Awaitable[None]] | None = None,
    timeout_sec: float = 15.0,
) -> list[AuditFinding]:
    """RULE-HERM-003: Detect cross-episode state residue surviving environment reset()."""
    _ = spec
    # Episode 1: Plant deterministic sentinel files in /tmp and workspace.
    await provider.exec(
        handle,
        ["sh", "-c", f"touch {_SENTINEL_PATHS[0]} {_SENTINEL_PATHS[1]}"],
        timeout_sec=timeout_sec,
    )

    # Invoke reset hook between episodes if provided.
    if reset_fn is not None:
        await reset_fn()

    # Episode 2: Check whether sentinels survived and inspect container FS diff.
    check_script = (
        "import json, os; "
        f"paths = {json.dumps(list(_SENTINEL_PATHS))}; "
        "surviving = [p for p in paths if os.path.exists(p)]; "
        "print(json.dumps({'surviving': surviving}))"
    )
    check_res = await provider.exec(
        handle, ["python3", "-c", check_script], timeout_sec=timeout_sec
    )
    payload = _parse_probe_json(check_res.stdout) or {}
    surviving = [str(p) for p in payload.get("surviving", []) if isinstance(p, str)]

    fs_diffs = await provider.diff_fs(handle)
    residue_diffs = [
        entry
        for entry in fs_diffs
        if any(tok in entry for tok in (".dasv_sentinel", ".dasv_dirty"))
        or entry.startswith(("A ", "C "))
    ]

    if not surviving and not residue_diffs:
        return []

    evidence_parts: list[str] = []
    if surviving:
        evidence_parts.append(f"sentinel file(s) survived reset: {', '.join(surviving)}")
    if residue_diffs:
        evidence_parts.append(
            f"container FS diff residue after reset ({len(residue_diffs)} entry/entries): "
            f"{', '.join(residue_diffs[:5])}"
        )

    return [
        AuditFinding(
            rule_id="RULE-HERM-003",
            title="Cross-episode filesystem residue survived environment reset()",
            severity=Severity.HIGH,
            tier=1,
            source="dasv",
            file_path=None,
            line_number=None,
            evidence="; ".join(evidence_parts) + ".",
            remediation=(
                "Ensure environment reset() recreates a clean container or fully wipes "
                "workspace and /tmp state between episodes to prevent reward contamination."
            ),
        )
    ]


def probe_emulated_arch(handle: SandboxHandle, spec: EnvSpec | None = None) -> list[AuditFinding]:
    """RULE-HERM-004: Flag host/container architecture mismatch (e.g. amd64 under Rosetta/QEMU)."""
    _ = spec
    if not handle.emulated_arch:
        return []

    host_arch = current_host_arch()
    return [
        AuditFinding(
            rule_id="RULE-HERM-004",
            title="Emulated container architecture detected (host/container arch mismatch)",
            severity=Severity.MEDIUM,
            tier=1,
            source="dasv",
            file_path=None,
            line_number=None,
            evidence=(
                f"Container architecture '{handle.resolved_arch}' differs from host "
                f"architecture '{host_arch}' (emulated_arch_warning=True)."
            ),
            remediation=(
                "Latency and determinism metrics collected under Rosetta/QEMU emulation are "
                "non-authoritative; run official certifications on native ubuntu-latest CI."
            ),
        )
    ]


async def run_runtime_boundary_probes(
    provider: SandboxProvider,
    handle: SandboxHandle,
    spec: EnvSpec,
    *,
    reset_fn: Callable[[], Awaitable[None]] | None = None,
    run_reset_probe: bool = False,
    timeout_sec: float = 15.0,
) -> list[AuditFinding]:
    """Execute Tier 1 runtime boundary, egress, reset-residue, and arch-parity probes ($0)."""
    findings: list[AuditFinding] = []
    findings.extend(
        await probe_writable_verifier_paths(provider, handle, spec, timeout_sec=timeout_sec)
    )
    findings.extend(
        await probe_egress_and_env_scrub(provider, handle, spec, timeout_sec=timeout_sec)
    )
    if (
        run_reset_probe
        or reset_fn is not None
        or bool(spec.metadata.get("supports_interactive_reset"))
    ):
        findings.extend(
            await probe_reset_residue(
                provider, handle, spec, reset_fn=reset_fn, timeout_sec=timeout_sec
            )
        )
    findings.extend(probe_emulated_arch(handle, spec))
    return findings
