"""KPI-RES-01 adversarial benchmark test: >= 90% recall and <= 5% false-alarm rate."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, cast

from dasv.adapters.registry import AdapterRegistry
from dasv.core.schemas import AuditFinding
from dasv.mutate import MutationSuiteRunner
from dasv.rubric import CallableRubricScorer, ProRubricAuditor
from dasv.sandbox.probes import run_tier0_static_probes
from dasv.sandbox.provider import (
    ContainerConfig,
    ExecResult,
    RuntimeHealth,
    SandboxHandle,
    current_host_arch,
)

CORPUS_ROOT = Path(__file__).resolve().parents[2] / "benchmarks" / "planted_vulns"


class _LocalVerifierSandboxProvider:
    """Ephemeral host-temp SandboxProvider executing real anti-patch diffs & Python verifiers.

    Used strictly for deterministic $0 CI verification of the trusted benchmark corpus in
    `benchmarks/planted_vulns/` without requiring Docker image builds for every synthetic fixture.
    """

    def __init__(self, env_root: Path) -> None:
        self._env_root = env_root.resolve()
        self._dirs: dict[str, tempfile.TemporaryDirectory[str]] = {}
        self._mounts: dict[str, list[tuple[Path, str]]] = {}
        self._counter = 0

    async def health_check(self) -> RuntimeHealth:
        arch = current_host_arch()
        return RuntimeHealth(healthy=True, runtime="local-bench", host_arch=arch, detail="ok")

    async def build_image(
        self,
        context_dir: Path,
        dockerfile: Path,
        tag: str,
        platform: str | None = None,
    ) -> str:
        _ = (context_dir, dockerfile, platform)
        return f"sha256:local-{tag}"

    async def start(self, cfg: ContainerConfig, run_id: str) -> SandboxHandle:
        self._counter += 1
        cid = f"bench-{cfg.plane}-{self._counter}"
        td = tempfile.TemporaryDirectory(prefix="dasv_bench_")
        ws = Path(td.name) / "app"
        shutil.copytree(self._env_root, ws, ignore=shutil.ignore_patterns(".git", "__pycache__"))
        self._dirs[cid] = td
        self._mounts[cid] = [(hp.resolve(), cp) for hp, cp, _m in cfg.bind_mounts]
        return SandboxHandle(
            container_id=cid,
            image_digest=cfg.image or "local",
            resolved_arch=current_host_arch(),
            emulated_arch=False,
            plane=cfg.plane,
            run_id=run_id,
        )

    async def exec(
        self,
        handle: SandboxHandle,
        cmd: list[str],
        timeout_sec: float,
        workdir: str | None = None,
    ) -> ExecResult:
        _ = workdir
        ws = Path(self._dirs[handle.container_id].name) / "app"
        mounts = self._mounts.get(handle.container_id, [])

        # Rewrite container mount paths (e.g. /dasv_verifier/test.sh) to host temp paths.
        rewritten: list[str] = []
        for tok in cmd:
            replaced = tok
            for host_dir, ctr_mount in mounts:
                if replaced == ctr_mount:
                    replaced = str(host_dir)
                elif replaced.startswith(f"{ctr_mount}/"):
                    replaced = str(host_dir / replaced[len(ctr_mount) + 1 :])
            rewritten.append(replaced)

        # For `pytest <test_file.py>`, execute the standalone Python verifier script directly.
        if rewritten and rewritten[0] == "pytest" and len(rewritten) >= 2:
            target = Path(rewritten[1])
            if target.is_file() and target.suffix == ".py":
                # Copy mounted verifier inside temp ws/tests so relative imports see mutated src/
                ws_test = ws / "tests" / target.name
                ws_test.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, ws_test)
                rewritten = [sys.executable, str(ws_test)]
        elif rewritten and rewritten[0] == "sh" and len(rewritten) >= 2:
            target = Path(rewritten[1])
            if target.is_file():
                # Point test.sh at temp ws/tests so sibling scripts see mutated ws/src/
                ws_tests_dir = ws / "tests"
                if ws_tests_dir.is_dir():
                    rewritten = ["sh", str(ws_tests_dir / target.name)]
        elif rewritten and rewritten[0] == "python3":
            rewritten = [sys.executable, *rewritten[1:]]

        # Isolate git apply inside ws from the parent dasv repository .git
        env: dict[str, str] = dict(os.environ)
        env["GIT_CEILING_DIRECTORIES"] = str(ws.parent)
        proc = subprocess.run(
            rewritten,
            cwd=str(ws),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        return ExecResult(
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            duration_ms=5.0,
        )

    async def copy_from(self, handle: SandboxHandle, container_path: str, host_dest: Path) -> None:
        _ = (handle, container_path, host_dest)

    async def diff_fs(self, handle: SandboxHandle) -> list[str]:
        _ = handle
        return []

    async def stop_and_remove(self, handle: SandboxHandle) -> None:
        td = self._dirs.pop(handle.container_id, None)
        self._mounts.pop(handle.container_id, None)
        if td is not None:
            td.cleanup()


def _noop_stub(*_args: object, **_kwargs: object) -> None:
    return None


def _load_env_score_fn(env_py: Path) -> Any:
    """Load `score_response(prompt, response)` from a benchmark verifiers v1 `env.py`."""
    stub = types.ModuleType("verifiers")
    for cls_name in ("JudgeRubric", "Rubric", "RubricGroup", "Taskset"):
        setattr(stub, cls_name, _noop_stub)
    prev = sys.modules.get("verifiers")
    sys.modules["verifiers"] = stub
    try:
        mod_spec = importlib.util.spec_from_file_location(
            f"bench_rubric_{env_py.parent.name}",
            env_py,
        )
        assert mod_spec is not None and mod_spec.loader is not None
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
        return cast(Any, module).score_response
    finally:
        if prev is None:
            sys.modules.pop("verifiers", None)
        else:
            sys.modules["verifiers"] = prev


async def _evaluate_corpus_case(case_dir: Path, audit_mode: str) -> list[AuditFinding]:
    registry = AdapterRegistry(load_entry_points=False)

    if audit_mode == "tier0_static_with_git_fixture":
        with tempfile.TemporaryDirectory(prefix="dasv_git_leak_") as tmp:
            copy_dir = Path(tmp) / case_dir.name
            shutil.copytree(case_dir, copy_dir)
            meta = json.loads((copy_dir / "planted_git_leak.json").read_text(encoding="utf-8"))
            subprocess.run(
                ["git", "init", "-b", "main"], cwd=str(copy_dir), check=True, capture_output=True
            )
            subprocess.run(
                ["git", "config", "user.email", "bench@example.com"],
                cwd=str(copy_dir),
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Bench"],
                cwd=str(copy_dir),
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "add", *meta["files_to_commit"]],
                cwd=str(copy_dir),
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "commit", "-m", meta["commit_message"]],
                cwd=str(copy_dir),
                check=True,
                capture_output=True,
            )
            adapter = registry.resolve(copy_dir)
            spec = adapter.load(copy_dir)
            return [*run_tier0_static_probes(spec), *adapter.static_findings(spec)]

    adapter = registry.resolve(case_dir)
    spec = adapter.load(case_dir)
    findings: list[AuditFinding] = [
        *run_tier0_static_probes(spec),
        *adapter.static_findings(spec),
    ]

    if audit_mode in {"tier1_mutation_local_exec", "tier0_and_tier1_mutation"}:
        provider = _LocalVerifierSandboxProvider(case_dir)
        mut_runner = MutationSuiteRunner(provider=provider)
        mut_report = await mut_runner.run(
            spec,
            adapter=adapter,
            include_impossible_bench=False,
            dry_run=False,
        )
        findings.extend(mut_report.findings)

    if audit_mode in {"tier2_prorubric_and_verbosity", "tier0_and_tier2_prorubric"}:
        score_fn = _load_env_score_fn(case_dir / "env.py")
        rubric_report = await ProRubricAuditor().audit(
            case_dir,
            spec=spec,
            scorer=CallableRubricScorer(score_fn),
            baseline_response="Merge sort runs in O(1) time.",
        )
        findings.extend(rubric_report.findings)

    return findings


async def test_kpi_res_01_planted_vulns_recall_and_false_alarm_rate() -> None:
    """Assert >= 90% detection recall across >= 6 vuln classes and <= 5% false-alarm rate."""
    corpus = cast(dict[str, Any], json.loads((CORPUS_ROOT / "corpus.json").read_text("utf-8")))
    vulnerable_cases = cast(list[dict[str, Any]], corpus["vulnerable"])
    hardened_cases = cast(list[dict[str, Any]], corpus["hardened"])

    vuln_classes = {str(c["vuln_class"]) for c in vulnerable_cases}
    assert len(vuln_classes) >= 6, f"Expected >= 6 vulnerability classes, found {len(vuln_classes)}"

    true_positives = 0
    missed: list[str] = []
    for case in vulnerable_cases:
        case_dir = CORPUS_ROOT / str(case["path"])
        findings = await _evaluate_corpus_case(case_dir, str(case["audit_mode"]))
        found_rules = {f.rule_id for f in findings}
        expected_rules = set(cast(list[str], case["expected_rules"]))
        if expected_rules.issubset(found_rules):
            true_positives += 1
        else:
            missed.append(
                f"{case['id']}: expected {sorted(expected_rules)}, got {sorted(found_rules)}"
            )

    false_positives = 0
    false_alarms: list[str] = []
    for ctrl in hardened_cases:
        ctrl_dir = CORPUS_ROOT / str(ctrl["path"])
        findings = await _evaluate_corpus_case(ctrl_dir, str(ctrl["audit_mode"]))
        if findings:
            false_positives += 1
            false_alarms.append(
                f"{ctrl['id']}: unexpected findings {[f.rule_id for f in findings]}"
            )

    recall = true_positives / len(vulnerable_cases)
    false_alarm_rate = false_positives / len(hardened_cases)

    assert recall >= 0.90, f"KPI-RES-01 recall {recall:.2%} < 90%; missed={missed}"
    assert false_alarm_rate <= 0.05, (
        f"KPI-RES-01 false-alarm rate {false_alarm_rate:.2%} > 5%; false_alarms={false_alarms}"
    )
    assert recall == 1.0, f"Expected 100% recall on planted corpus; missed={missed}"
    assert false_alarm_rate == 0.0, f"Expected 0% false alarms; false_alarms={false_alarms}"
