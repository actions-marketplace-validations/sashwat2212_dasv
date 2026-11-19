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
    EXIT_NATIVE_ERROR,
    EXIT_PASS,
    EXIT_RUNTIME_ERROR,
    DasvError,
)
from dasv.core.orchestrator import AuditContext, TieredAuditOrchestrator
from dasv.core.schemas import (
    AuditReport,
    Severity,
)
from dasv.mutate import MutationSuiteRunner
from dasv.reporting.html_renderer import write_html_report
from dasv.reporting.json_writer import write_report_json
from dasv.reporting.sarif_writer import write_sarif
from dasv.sandbox.docker_provider import DockerSandboxProvider
from dasv.sandbox.janitor import ContainerJanitor

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
        str | None,
        typer.Option(
            "--tiers",
            help="Comma-separated audit tiers (e.g. '0' or '0,1'). Defaults to dasv.toml.",
        ),
    ] = None,
    env_format: Annotated[
        str | None,
        typer.Option("--format", help="Force adapter format (harbor, openenv, verifiers_v1, hud)."),
    ] = None,
    replays: Annotated[
        int | None,
        typer.Option("--replays", min=0, help="Tier 1 Oracle replay count N (overrides config)."),
    ] = None,
    noop_replays: Annotated[
        int | None,
        typer.Option(
            "--noop-replays", min=0, help="Tier 1 No-op replay count M (overrides config)."
        ),
    ] = None,
    out_dir: Annotated[
        Path | None,
        typer.Option("--out-dir", help="Directory to write report.json and manifest.json."),
    ] = None,
    sarif: Annotated[
        Path | None,
        typer.Option("--sarif", help="Write SARIF 2.1.0 report to this path."),
    ] = None,
    html: Annotated[
        Path | None,
        typer.Option("--html", help="Write zero-CDN offline HTML Certification Report."),
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
    short_circuit: Annotated[
        bool,
        typer.Option(
            "--short-circuit/--no-short-circuit",
            help="Skip Tier 2/3 spend when Tier 1 determinism fails.",
        ),
    ] = True,
    with_mutations: Annotated[
        bool,
        typer.Option(
            "--with-mutations/--no-mutations",
            help="Run $0 Anti-Patch & ImpossibleBench mutation suite inside Tier 1.",
        ),
    ] = False,
    max_cost_usd: Annotated[
        float | None,
        typer.Option("--max-cost-usd", min=0.0, help="Hard USD budget ceiling."),
    ] = None,
    max_tokens: Annotated[
        int | None,
        typer.Option("--max-tokens", min=1, help="Hard total token ceiling."),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Optional path to dasv.toml."),
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Deterministic RNG seed.")] = 42,
) -> None:
    """Run the tiered audit pipeline (Tier 0 static -> Tier 1 clean-image determinism)."""
    try:
        cfg = load_config(config_path)
        if max_cost_usd is not None or max_tokens is not None:
            budget_updates: dict[str, object] = {}
            if max_cost_usd is not None:
                budget_updates["max_cost_usd"] = max_cost_usd
            if max_tokens is not None:
                budget_updates["max_tokens"] = max_tokens
            cfg = cfg.model_copy(update={"budget": cfg.budget.model_copy(update=budget_updates)})
        parsed_tiers: set[int] = (
            _parse_tiers(tiers)
            if tiers is not None
            else {int(t) for t in cfg.project.default_tiers}
        )
        threshold = _parse_severity(fail_on or cfg.project.fail_on)
        registry = AdapterRegistry()
        adapter = registry.resolve(path, forced_format=env_format)
        spec = adapter.load(path)
    except DasvError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=exc.exit_code) from exc

    ctx = AuditContext.create(
        config=cfg,
        adapter=adapter,
        env_spec=spec,
        out_dir=out_dir,
        cli_args=sys.argv[1:],
        seed=seed,
        native=native,
        require_native=require_native,
        oracle_replays=replays,
        noop_replays=noop_replays,
        short_circuit=short_circuit,
        run_mutations=with_mutations,
    )
    orchestrator = TieredAuditOrchestrator()
    result = asyncio.run(
        orchestrator.execute(
            ctx,
            parsed_tiers,
            fail_on=threshold,
            sarif_path=sarif,
            html_path=html,
            install_signal_handlers=True,
        )
    )

    if result.error_message:
        typer.echo(f"Error: {result.error_message}", err=True)

    # Do not render a misleading Grade badge when runtime/native setup aborted the audit.
    if result.exit_code in {EXIT_RUNTIME_ERROR, EXIT_NATIVE_ERROR} or result.aborted_by_signal:
        raise typer.Exit(code=result.exit_code)

    rep = result.report
    hack_score = 0.0 if rep.hackability_score is None else rep.hackability_score
    title = (
        f"Audit Summary (Partial — Budget Cap Reached): {spec.task_id} "
        f"(Provisional Grade {rep.grade})"
        if result.budget_exceeded
        else f"Audit Summary: {spec.task_id} (Grade {rep.grade})"
    )
    summary = Table(title=title)
    summary.add_column("Metric", style="bold")
    summary.add_column("Value")
    summary.add_row("Standard", spec.standard.value)
    summary.add_row("Grade", f"{rep.grade} (provisional)" if result.budget_exceeded else rep.grade)
    summary.add_row("Hackability Score", f"{hack_score:.2f} / 100.0")
    summary.add_row("Total Findings", str(len(rep.findings)))
    summary.add_row("Fail Threshold", threshold.value)
    if rep.determinism is not None:
        det = rep.determinism
        summary.add_row(
            "Oracle Pass Rate", f"{det.oracle_pass_rate * 100.0:.1f}% ({det.oracle_trials} trials)"
        )
        summary.add_row(
            "No-op Pass Rate", f"{det.noop_pass_rate * 100.0:.1f}% ({det.noop_trials} trials)"
        )
        summary.add_row("Replay Variance", f"{det.replay_variance:.6f}")
        summary.add_row(
            "Latency p50 / p95", f"{det.latency_p50_ms:.0f} / {det.latency_p95_ms:.0f} ms"
        )
    if rep.mutation_survival_rate is not None:
        summary.add_row("Mutation Survival Rate", f"{rep.mutation_survival_rate * 100.0:.1f}%")
    console.print(summary)

    if rep.findings:
        ftable = Table(title="Findings")
        ftable.add_column("Rule", style="bold")
        ftable.add_column("Severity")
        ftable.add_column("Location")
        ftable.add_column("Title")
        for f in rep.findings:
            loc = (
                f"{f.file_path}:{f.line_number}"
                if (f.file_path and f.line_number)
                else (f.file_path or "-")
            )
            ftable.add_row(f.rule_id, f.severity.value.upper(), loc, f.title)
        console.print(ftable)

    if result.exit_code != EXIT_PASS:
        raise typer.Exit(code=result.exit_code)


@app.command()
def report(
    report_json: Annotated[Path, typer.Argument(help="Path to an existing report.json file.")],
    sarif: Annotated[
        Path | None,
        typer.Option("--sarif", help="Write SARIF 2.1.0 report to this path."),
    ] = None,
    html: Annotated[
        Path | None,
        typer.Option("--html", help="Write zero-CDN offline HTML Certification Report."),
    ] = None,
    json_out: Annotated[
        Path | None,
        typer.Option("--json-out", help="Re-serialize validated AuditReport JSON to this path."),
    ] = None,
) -> None:
    """Re-render SARIF 2.1.0, offline HTML, or validated JSON artifacts from report.json."""
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
    if html is not None:
        write_html_report(loaded, html)
        typer.echo(f"Wrote HTML Certification Report to {html}")
    if json_out is not None:
        write_report_json(loaded, json_out)
        typer.echo(f"Wrote AuditReport JSON to {json_out}")
    if sarif is None and html is None and json_out is None:
        typer.echo(
            f"Validated AuditReport for '{loaded.env_spec.task_id}' "
            f"(Grade {loaded.grade}, {len(loaded.findings)} finding(s))."
        )


def _stage6_stub(command_name: str, milestone: str) -> None:
    typer.echo(
        f"`dasv {command_name}` is scheduled for Stage 6 ({milestone}). "
        "Available now: `doctor`, `inspect`, `audit`, `mutate`, `report`.",
        err=True,
    )
    raise typer.Exit(code=EXIT_CONFIG_ERROR)


@app.command()
def mutate(
    path: Annotated[Path, typer.Argument(help="Path to RL environment directory.")],
    env_format: Annotated[
        str | None,
        typer.Option("--format", help="Force adapter format (harbor, openenv, verifiers_v1, hud)."),
    ] = None,
    impossible_bench: Annotated[
        bool,
        typer.Option(
            "--impossible-bench/--no-impossible-bench",
            help="Include ImpossibleBench One-Off and Conflicting spec-contradiction mutations.",
        ),
    ] = True,
    max_one_off: Annotated[
        int,
        typer.Option("--max-one-off", min=1, help="Max One-Off verifier assertion inversions."),
    ] = 5,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Synthesize & syntax-check all mutations at $0 without booting Docker containers.",
        ),
    ] = False,
    out_dir: Annotated[
        Path | None,
        typer.Option("--out-dir", help="Write mutations.json and generated diffs/variants here."),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Optional path to dasv.toml."),
    ] = None,
) -> None:
    """Run Tier 1 deterministic anti-patch & ImpossibleBench mutations ($0)."""
    try:
        cfg = load_config(config_path)
        registry = AdapterRegistry()
        adapter = registry.resolve(path, forced_format=env_format)
        spec = adapter.load(path)
    except DasvError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=exc.exit_code) from exc

    async def _run_mutations() -> int:
        janitor = ContainerJanitor()
        run_id = "dasv-mutate"
        provider: DockerSandboxProvider | None = None
        if not dry_run:
            provider = DockerSandboxProvider(janitor=janitor)
            health = await provider.health_check()
            if not health.healthy:
                typer.echo(
                    f"Error: Docker runtime unavailable ({health.detail}). "
                    "Start Docker/OrbStack or pass `--dry-run` to synthesize mutations locally.",
                    err=True,
                )
                return EXIT_RUNTIME_ERROR
        try:
            runner_obj = MutationSuiteRunner(
                provider=provider,
                sandbox_config=cfg.sandbox,
            )
            mut_report = await runner_obj.run(
                spec,
                adapter=adapter,
                run_id=run_id,
                include_impossible_bench=impossible_bench,
                max_one_off=max_one_off,
                dry_run=dry_run,
                out_dir=out_dir,
            )
        except DasvError as exc:
            typer.echo(f"Error: {exc}", err=True)
            return exc.exit_code
        finally:
            if not dry_run:
                await janitor.reap_all(run_id)

        table = Table(title=f"Mutation Suite: {spec.task_id}")
        table.add_column("Metric", style="bold")
        table.add_column("Value")
        table.add_row("Mode", "dry-run (synthesis only)" if dry_run else "clean-image grading")
        table.add_row("Anti-Patches Generated", str(mut_report.anti_patches_generated))
        if mut_report.mutation_survival_rate is not None:
            table.add_row(
                "Mutation Survival Rate", f"{mut_report.mutation_survival_rate * 100.0:.1f}%"
            )
        table.add_row("One-Off Verifier Mutations", str(mut_report.one_off_mutations_generated))
        table.add_row("Dead Assertions (RULE-MUT-001)", str(mut_report.dead_assertions_count))
        table.add_row("Conflicting Spec Variants", str(len(mut_report.conflicting_variants)))
        table.add_row("Findings", str(len(mut_report.findings)))
        console.print(table)
        return EXIT_PASS if mut_report.passed else EXIT_AUDIT_FAIL

    code = asyncio.run(_run_mutations())
    if code != EXIT_PASS:
        raise typer.Exit(code=code)


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
