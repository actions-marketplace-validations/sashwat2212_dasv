"""Frozen `dasv` CLI surface (02-PRD-FEATURES.md & SDD-04)."""

from __future__ import annotations

import asyncio
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from dasv import __version__
from dasv.adapters.registry import AdapterRegistry
from dasv.core.config import load_config
from dasv.core.errors import (
    EXIT_AUDIT_FAIL,
    EXIT_CONFIG_ERROR,
    EXIT_PASS,
    DasvError,
)
from dasv.core.manifest import build_run_manifest
from dasv.core.schemas import (
    AuditFinding,
    AuditReport,
    Severity,
    meets_fail_threshold,
)
from dasv.profile.stats import compute_grade, compute_hackability_score
from dasv.reporting.json_writer import write_manifest_json, write_report_json
from dasv.reporting.sarif_writer import write_sarif
from dasv.sandbox.probes import run_tier0_static_probes

app = typer.Typer(
    name="dasv",
    help="Deterministic Adversarial Sandbox Verifier — pre-flight auditor for RL environments.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"dasv {__version__}")
        raise typer.Exit(code=EXIT_PASS)


@app.callback()
def _root_callback(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Show dasv version and exit.",
        ),
    ] = False,
) -> None:
    _ = version


def _probe_docker_arch() -> tuple[bool, str | None, str]:
    docker_bin = shutil.which("docker")
    if docker_bin is None:
        return False, None, "docker CLI not found on PATH"
    try:
        proc = subprocess.run(
            [docker_bin, "info", "--format", "{{.Architecture}}"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, None, str(exc)
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "Docker daemon unreachable").strip()
        return False, None, msg
    arch = proc.stdout.strip() or None
    return True, arch, "reachable"


def _parse_tiers(tiers_str: str) -> set[int]:
    tiers: set[int] = set()
    for part in tiers_str.split(","):
        cleaned = part.strip()
        if not cleaned:
            continue
        try:
            tier_num = int(cleaned)
        except ValueError as exc:
            raise DasvError(
                f"Invalid tier '{cleaned}' in --tiers '{tiers_str}'. Expected integers 0-3.",
                exit_code=EXIT_CONFIG_ERROR,
            ) from exc
        if tier_num not in {0, 1, 2, 3}:
            raise DasvError(
                f"Unsupported tier {tier_num}. Allowed tiers are 0, 1, 2, 3.",
                exit_code=EXIT_CONFIG_ERROR,
            )
        tiers.add(tier_num)
    if not tiers:
        raise DasvError("At least one tier must be specified.", exit_code=EXIT_CONFIG_ERROR)
    return tiers


def _parse_severity(raw: str) -> Severity:
    try:
        return Severity(raw.strip().lower())
    except ValueError as exc:
        valid = ", ".join(s.value for s in Severity)
        raise DasvError(
            f"Invalid --fail-on severity '{raw}'. Valid values: {valid}.",
            exit_code=EXIT_CONFIG_ERROR,
        ) from exc


@app.command()
def doctor() -> None:
    """Verify local Python, Docker/OCI runtime, arch parity, and optional native CLIs."""
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = sys.version_info >= (3, 11)
    host_os = platform.system().lower()
    host_arch = platform.machine().lower()
    docker_ok, docker_arch, docker_detail = _probe_docker_arch()

    table = Table(title=f"dasv doctor (v{__version__})")
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Details")

    table.add_row("Python >= 3.11", "OK" if py_ok else "FAIL", py_ver)
    table.add_row("Host Platform", "INFO", f"{host_os} / {host_arch}")
    table.add_row(
        "Docker Runtime",
        "OK" if docker_ok else "WARN",
        f"{docker_detail} (server arch: {docker_arch})" if docker_ok else docker_detail,
    )

    if host_os == "darwin" and host_arch in {"arm64", "aarch64"}:
        table.add_row(
            "Arch Parity",
            "WARN",
            "macOS arm64 host: avoid publishing QEMU-emulated linux/amd64 latency stats; "
            "run official certifications on ubuntu-latest.",
        )

    for binary in ("harbor", "openenv", "vf-validate", "hud"):
        found_path = shutil.which(binary)
        table.add_row(
            f"Native CLI: {binary}",
            "OK" if found_path else "OPTIONAL",
            found_path or "not installed (dasv fallback checks active)",
        )

    console.print(table)


@app.command()
def inspect(
    path: Annotated[Path, typer.Argument(help="Path to RL environment directory.")],
    env_format: Annotated[
        str | None,
        typer.Option("--format", help="Force adapter format (harbor, openenv, verifiers_v1, hud)."),
    ] = None,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit canonical EnvSpec as JSON."),
    ] = False,
) -> None:
    """Auto-detect environment format and print canonical EnvSpec."""
    registry = AdapterRegistry()
    try:
        adapter = registry.resolve(path, forced_format=env_format)
        spec = adapter.load(path)
    except DasvError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=exc.exit_code) from exc

    if as_json:
        typer.echo(spec.model_dump_json(indent=2))
        return

    table = Table(title=f"EnvSpec: {spec.task_id}")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Standard", spec.standard.value)
    table.add_row("Root Path", str(spec.root_path))
    table.add_row("Image Ref", spec.image_ref or "-")
    table.add_row("Image Digest", spec.image_digest or "-")
    table.add_row("Has Oracle", str(spec.has_oracle))
    table.add_row("Oracle Entrypoint", spec.oracle_entrypoint or "-")
    table.add_row("Verifier Paths", ", ".join(spec.verifier_paths) or "-")
    table.add_row("Rubric Type", spec.rubric_type)
    console.print(table)


@app.command()
def audit(
    path: Annotated[Path, typer.Argument(help="Path to RL environment directory.")],
    tiers: Annotated[
        str,
        typer.Option("--tiers", help="Comma-separated audit tiers to run (e.g. '0' or '0,1')."),
    ] = "0",
    env_format: Annotated[
        str | None,
        typer.Option("--format", help="Force adapter format (harbor, openenv, verifiers_v1, hud)."),
    ] = None,
    out_dir: Annotated[
        Path | None,
        typer.Option("--out-dir", help="Directory to write report.json and manifest.json."),
    ] = None,
    sarif: Annotated[
        Path | None,
        typer.Option("--sarif", help="Write SARIF 2.1.0 report to this path."),
    ] = None,
    fail_on: Annotated[
        str | None,
        typer.Option(
            "--fail-on", help="Minimum severity to fail audit (critical|high|medium|low)."
        ),
    ] = None,
    native: Annotated[
        bool,
        typer.Option("--native/--no-native", help="Delegate baseline checks to native CLI."),
    ] = True,
    require_native: Annotated[
        bool,
        typer.Option(
            "--require-native", help="Exit 5 if native validator binary is missing/fails."
        ),
    ] = False,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Optional path to dasv.toml."),
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Deterministic RNG seed.")] = 42,
) -> None:
    """Run the tiered audit pipeline (Stage 5 supports Tier 0 static + native delegation)."""
    try:
        cfg = load_config(config_path)
        parsed_tiers = _parse_tiers(tiers)
        threshold = _parse_severity(fail_on or cfg.project.fail_on)
        registry = AdapterRegistry()
        adapter = registry.resolve(path, forced_format=env_format)
        spec = adapter.load(path)
    except DasvError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=exc.exit_code) from exc

    findings: list[AuditFinding] = []
    if 0 in parsed_tiers:
        findings.extend(run_tier0_static_probes(spec))
        findings.extend(adapter.static_findings(spec))
        if native or require_native:
            try:
                native_findings = asyncio.run(
                    adapter.run_native_check(spec, require_native=require_native)
                )
            except DasvError as exc:
                typer.echo(f"Error: {exc}", err=True)
                raise typer.Exit(code=exc.exit_code) from exc
            findings.extend(native_findings)

    deferred_tiers = sorted(t for t in parsed_tiers if t > 0)
    if deferred_tiers:
        findings.append(
            AuditFinding(
                rule_id="RULE-STAGE-001",
                title="Requested higher tiers deferred to Stage 6",
                severity=Severity.INFO,
                tier=0,
                source="dasv",
                file_path=None,
                line_number=None,
                evidence=(
                    f"Tier(s) {', '.join(str(t) for t in deferred_tiers)} require Stage 6 "
                    "container/LLM runners; only Tier 0 static probes executed in Stage 5."
                ),
                remediation="Run with --tiers 0 in Stage 5 or upgrade to Stage 6 v0.1+.",
            )
        )

    _, docker_arch, _ = _probe_docker_arch()
    digests = {spec.image_ref: spec.image_digest} if (spec.image_ref and spec.image_digest) else {}
    manifest = build_run_manifest(
        spec.root_path,
        cli_args=sys.argv[1:],
        docker_server_arch=docker_arch,
        oci_image_digests=digests,
        rng_seed=seed,
    )
    hack_score = compute_hackability_score(findings)
    grade = compute_grade(findings, hackability_score=hack_score)
    report = AuditReport(
        manifest=manifest,
        env_spec=spec,
        grade=grade,
        hackability_score=hack_score,
        findings=findings,
    )

    if out_dir is not None:
        write_report_json(report, out_dir / "report.json")
        write_manifest_json(manifest, out_dir / "manifest.json")
    if sarif is not None:
        write_sarif(report, sarif)

    summary = Table(title=f"Audit Summary: {spec.task_id} (Grade {grade})")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value")
    summary.add_row("Standard", spec.standard.value)
    summary.add_row("Grade", grade)
    summary.add_row("Hackability Score", f"{hack_score:.2f} / 100.0")
    summary.add_row("Total Findings", str(len(findings)))
    summary.add_row("Fail Threshold", threshold.value)
    console.print(summary)

    if findings:
        ftable = Table(title="Findings")
        ftable.add_column("Rule", style="bold")
        ftable.add_column("Severity")
        ftable.add_column("Location")
        ftable.add_column("Title")
        for f in findings:
            loc = (
                f"{f.file_path}:{f.line_number}"
                if (f.file_path and f.line_number)
                else (f.file_path or "-")
            )
            ftable.add_row(f.rule_id, f.severity.value.upper(), loc, f.title)
        console.print(ftable)

    if meets_fail_threshold(findings, threshold):
        raise typer.Exit(code=EXIT_AUDIT_FAIL)


@app.command()
def report(
    report_json: Annotated[Path, typer.Argument(help="Path to an existing report.json file.")],
    sarif: Annotated[
        Path | None,
        typer.Option("--sarif", help="Write SARIF 2.1.0 report to this path."),
    ] = None,
    json_out: Annotated[
        Path | None,
        typer.Option("--json-out", help="Re-serialize validated AuditReport JSON to this path."),
    ] = None,
) -> None:
    """Re-render SARIF 2.1.0 or validated JSON artifacts from a saved report.json."""
    if not report_json.is_file():
        typer.echo(f"Error: Report file not found: {report_json}", err=True)
        raise typer.Exit(code=EXIT_CONFIG_ERROR)
    try:
        loaded = AuditReport.model_validate_json(report_json.read_text(encoding="utf-8"))
    except Exception as exc:
        typer.echo(f"Error: Invalid AuditReport JSON '{report_json}': {exc}", err=True)
        raise typer.Exit(code=EXIT_CONFIG_ERROR) from exc

    if sarif is not None:
        write_sarif(loaded, sarif)
        typer.echo(f"Wrote SARIF 2.1.0 report to {sarif}")
    if json_out is not None:
        write_report_json(loaded, json_out)
        typer.echo(f"Wrote AuditReport JSON to {json_out}")
    if sarif is None and json_out is None:
        typer.echo(
            f"Validated AuditReport for '{loaded.env_spec.task_id}' "
            f"(Grade {loaded.grade}, {len(loaded.findings)} finding(s))."
        )


def _stage6_stub(command_name: str, milestone: str) -> None:
    typer.echo(
        f"`dasv {command_name}` is scheduled for Stage 6 ({milestone}). "
        "Available in Stage 5: `doctor`, `inspect`, `audit --tiers 0`, `report`.",
        err=True,
    )
    raise typer.Exit(code=EXIT_CONFIG_ERROR)


@app.command()
def mutate(
    path: Annotated[Path, typer.Argument(help="Path to RL environment directory.")],
    impossible_bench: Annotated[
        bool,
        typer.Option(
            "--impossible-bench/--no-impossible-bench", help="Run spec-contradiction mutations."
        ),
    ] = True,
) -> None:
    """Run Tier 1 deterministic anti-patch & ImpossibleBench mutations ($0, Stage 6 M2)."""
    _ = (path, impossible_bench)
    _stage6_stub("mutate", "M2")


@app.command()
def redteam(
    path: Annotated[Path, typer.Argument(help="Path to RL environment directory.")],
    model: Annotated[
        str, typer.Option("--model", help="Adversarial attacker model ID.")
    ] = "qwen2.5-coder:7b",
    max_cost_usd: Annotated[float, typer.Option("--max-cost-usd", help="Hard USD cap.")] = 1.50,
    max_turns: Annotated[int, typer.Option("--max-turns", help="Max ReAct turns.")] = 12,
    propose_fix: Annotated[
        bool, typer.Option("--propose-fix", help="Generate Fixer unified diff.")
    ] = False,
    gold_sanity_replays: Annotated[
        int, typer.Option("--gold-sanity-replays", help="Oracle replays.")
    ] = 5,
) -> None:
    """Run Tier 2 budget-capped Hacker-Fixer loop & Gold-Sanity Gate (Stage 6 M2)."""
    _ = (path, model, max_cost_usd, max_turns, propose_fix, gold_sanity_replays)
    _stage6_stub("redteam", "M2")


@app.command()
def rubric(
    path: Annotated[Path, typer.Argument(help="Path to rubric-based RL environment.")],
    max_cost_usd: Annotated[float, typer.Option("--max-cost-usd", help="Hard USD cap.")] = 1.00,
) -> None:
    """Run ProRubric additive-compensation & CHERRL judge stress tests (Stage 6 M2/M4)."""
    _ = (path, max_cost_usd)
    _stage6_stub("rubric", "M2/M4")


@app.command()
def profile(
    path: Annotated[Path, typer.Argument(help="Path to RL taskset directory.")],
    base_url: Annotated[
        str, typer.Option("--base-url", help="OpenAI-compatible base URL.")
    ] = "http://localhost:11434/v1",
    model: Annotated[str, typer.Option("--model", help="Base policy model ID.")] = "qwen2.5:3b",
    rollouts: Annotated[int, typer.Option("--rollouts", help="Rollouts G per task.")] = 8,
    export_curriculum: Annotated[
        Path | None, typer.Option("--export-curriculum", help="Output JSONL path.")
    ] = None,
) -> None:
    """Run Tier 3 pass@k / pass^k & GRPO/DAPO zero-gradient signal profiler (Stage 6 M3)."""
    _ = (path, base_url, model, rollouts, export_curriculum)
    _stage6_stub("profile", "M3")


def main() -> None:
    """Console entry point (`dasv = dasv.cli.app:main`)."""
    app()
