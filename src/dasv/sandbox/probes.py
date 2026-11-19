"""Tier 0 ($0, <1s) static Verifier Boundary & Hermeticity probes (SDD-01)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from dasv.core.schemas import AuditFinding, EnvSpec, EnvStandard, Severity

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
