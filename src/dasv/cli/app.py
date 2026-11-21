"""Frozen `dasv` CLI surface (02-PRD-FEATURES.md & SDD-04)."""

from __future__ import annotations

import asyncio
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import typer
from rich.console import Console
from rich.table import Table

from dasv import __version__
from dasv.adapters.registry import AdapterRegistry
from dasv.core.budget import BudgetGuard, is_local_endpoint
from dasv.core.config import load_config
from dasv.core.errors import (
    EXIT_AUDIT_FAIL,
    EXIT_BUDGET_EXCEEDED,
    EXIT_CONFIG_ERROR,
    EXIT_NATIVE_ERROR,
    EXIT_PASS,
    EXIT_RUNTIME_ERROR,
    BudgetExceededError,
    DasvError,
)
from dasv.core.orchestrator import AuditContext, TieredAuditOrchestrator
from dasv.core.schemas import (
    AuditReport,
    CurriculumBucket,
    Severity,
)
from dasv.export import (
    EXPORT_FORMATS,
    ExportResult,
    export_openenv_traces,
    export_prime_rl,
    export_trl,
    export_verl,
    load_curriculum_entries,
    load_exploit_traces,
)
from dasv.mutate import MutationSuiteRunner
from dasv.profile import (
    SignalProfiler,
    build_tier3_profile_hook,
    discover_taskset_specs,
    load_rewards_from_traces_jsonl,
)
from dasv.redteam import (
    RedTeamRunner,
    UnifiedLLMClient,
    build_tier2_redteam_hook,
    parse_hack_classes,
)
from dasv.reporting.html_renderer import write_html_report
from dasv.reporting.json_writer import write_report_json
from dasv.reporting.sarif_writer import write_sarif
from dasv.rubric import (
    DEFAULT_DELTA_THRESHOLD,
    DEFAULT_TAU_THRESHOLD,
    LLMRubricJudgeScorer,
    ProRubricAuditor,
    RubricJudgeScorer,
    load_judge_samples_jsonl,
)
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
    redteam_model: Annotated[
        str | None,
        typer.Option("--redteam-model", help="Override Tier 2 Hacker/Fixer model ID."),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option("--base-url", help="Override OpenAI-compatible inference base URL."),
    ] = None,
    api_key_env: Annotated[
        str,
        typer.Option("--api-key-env", help="Env var holding cloud API key (never enters sandbox)."),
    ] = "OPENAI_API_KEY",
    propose_fix: Annotated[
        bool,
        typer.Option("--propose-fix", help="Run Fixer + GoldSanityGate on Tier 2 exploits."),
    ] = False,
    hack_classes: Annotated[
        str | None,
        typer.Option(
            "--hack-classes",
            help="Comma-separated Tier 2 hack classes (default: all 6 SDD-02 strategies).",
        ),
    ] = None,
    gold_sanity_replays: Annotated[
        int,
        typer.Option("--gold-sanity-replays", min=1, help="Oracle replays in GoldSanityGate."),
    ] = 5,
    profile_model: Annotated[
        str | None,
        typer.Option("--profile-model", help="Override Tier 3 base policy model ID."),
    ] = None,
    rollouts: Annotated[
        int,
        typer.Option("--rollouts", min=1, help="Tier 3 rollouts G per task."),
    ] = 8,
    traces: Annotated[
        Path | None,
        typer.Option("--traces", help="Optional TrainingTrace JSONL for $0 Tier 3 profiling."),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Optional path to dasv.toml."),
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Deterministic RNG seed.")] = 42,
) -> None:
    """Run the tiered audit pipeline (Tier 0 -> Tier 1 -> Tier 2 red-team -> Tier 3 signal)."""
    try:
        cfg = load_config(config_path)
        if max_cost_usd is not None or max_tokens is not None:
            budget_updates: dict[str, object] = {}
            if max_cost_usd is not None:
                budget_updates["max_cost_usd"] = max_cost_usd
            if max_tokens is not None:
                budget_updates["max_tokens"] = max_tokens
            cfg = cfg.model_copy(update={"budget": cfg.budget.model_copy(update=budget_updates)})
        if base_url is not None or redteam_model is not None or profile_model is not None:
            llm_updates: dict[str, object] = {}
            if base_url is not None:
                llm_updates["base_url"] = base_url
            if redteam_model is not None:
                llm_updates["attacker_model"] = redteam_model
                llm_updates["fixer_model"] = redteam_model
            elif profile_model is not None:
                llm_updates["attacker_model"] = profile_model
            cfg = cfg.model_copy(update={"llm": cfg.llm.model_copy(update=llm_updates)})
        parsed_tiers: set[int] = (
            _parse_tiers(tiers)
            if tiers is not None
            else {int(t) for t in cfg.project.default_tiers}
        )
        selected_hack_classes = (
            parse_hack_classes(hack_classes) if (2 in parsed_tiers or hack_classes) else None
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
    tier2_hook = (
        build_tier2_redteam_hook(
            attacker_model=redteam_model,
            fixer_model=redteam_model,
            base_url=base_url,
            api_key_env=api_key_env,
            hack_classes=selected_hack_classes
            if selected_hack_classes is not None
            else parse_hack_classes(None),
            propose_fix=propose_fix,
            gold_sanity_replays=gold_sanity_replays,
        )
        if 2 in parsed_tiers
        else None
    )
    tier3_hook = (
        build_tier3_profile_hook(
            base_model=profile_model or redteam_model,
            base_url=base_url,
            api_key_env=api_key_env,
            rollouts_g=rollouts,
            traces_path=traces,
        )
        if 3 in parsed_tiers
        else None
    )
    orchestrator = TieredAuditOrchestrator(tier2_hook=tier2_hook, tier3_hook=tier3_hook)
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

    # Do not render a misleading Grade badge when config/runtime/native errors aborted the audit.
    if (
        result.exit_code in {EXIT_CONFIG_ERROR, EXIT_RUNTIME_ERROR, EXIT_NATIVE_ERROR}
        or result.aborted_by_signal
    ):
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
    if rep.exploits or 2 in parsed_tiers:
        unpatched = sum(1 for e in rep.exploits if e.gold_sanity_status != "verified_fix")
        summary.add_row(
            "Confirmed Exploits",
            f"{len(rep.exploits)} ({unpatched} unpatched)",
        )
    if rep.signal_profile is not None:
        sp = rep.signal_profile
        summary.add_row(
            "RLVR Signal Bucket",
            (
                f"{sp.bucket.value} "
                f"(pass@1={sp.pass_at_1 * 100.0:.1f}%, var={sp.advantage_variance:.4f})"
            ),
        )
    if rep.manifest.total_tokens > 0 or rep.manifest.total_cost_usd > 0.0:
        summary.add_row("Tokens Used", str(rep.manifest.total_tokens))
        summary.add_row("Spend (USD)", f"${rep.manifest.total_cost_usd:.6f}")
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
        run_id = f"dasv-mutate-{uuid4().hex[:10]}"
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
    env_format: Annotated[
        str | None,
        typer.Option("--format", help="Force adapter format (harbor, openenv, verifiers_v1, hud)."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Adversarial attacker model ID (overrides dasv.toml)."),
    ] = None,
    fixer_model: Annotated[
        str | None,
        typer.Option("--fixer-model", help="Fixer model ID (defaults to --model)."),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option(
            "--base-url",
            help="OpenAI-compatible inference base URL (default: http://localhost:11434/v1).",
        ),
    ] = None,
    api_key_env: Annotated[
        str,
        typer.Option("--api-key-env", help="Env var holding cloud API key (never enters sandbox)."),
    ] = "OPENAI_API_KEY",
    max_cost_usd: Annotated[
        float | None,
        typer.Option("--max-cost-usd", min=0.0, help="Hard USD budget cap."),
    ] = None,
    max_tokens: Annotated[
        int | None,
        typer.Option("--max-tokens", min=1, help="Hard total token cap."),
    ] = None,
    max_turns: Annotated[
        int | None,
        typer.Option("--max-turns", min=1, help="Max ReAct turns per hack class."),
    ] = None,
    hack_classes: Annotated[
        str | None,
        typer.Option(
            "--hack-classes",
            help="Comma-separated hack classes (default: all 6 SDD-02 strategies).",
        ),
    ] = None,
    propose_fix: Annotated[
        bool,
        typer.Option("--propose-fix", help="Generate Fixer unified diffs + run GoldSanityGate."),
    ] = False,
    gold_sanity_replays: Annotated[
        int,
        typer.Option("--gold-sanity-replays", min=1, help="Oracle replays in GoldSanityGate."),
    ] = 5,
    apply_fixes: Annotated[
        bool,
        typer.Option(
            "--apply",
            help="Apply ONLY GoldSanityGate-verified (`verified_fix`) patches to target env.",
        ),
    ] = False,
    out_dir: Annotated[
        Path | None,
        typer.Option("--out-dir", help="Write redteam.json, exploits.jsonl, and fixes/*.diff."),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Optional path to dasv.toml."),
    ] = None,
) -> None:
    """Run Tier 2 budget-capped adversarial Hacker-Fixer loop & 3-invariant GoldSanityGate."""
    try:
        cfg = load_config(config_path)
        selected_classes = parse_hack_classes(hack_classes)
        registry = AdapterRegistry()
        adapter = registry.resolve(path, forced_format=env_format)
        spec = adapter.load(path)
    except DasvError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=exc.exit_code) from exc

    effective_base_url = base_url or cfg.llm.base_url
    effective_attacker = model or cfg.llm.attacker_model
    effective_fixer = fixer_model or model or cfg.llm.fixer_model
    effective_max_cost = max_cost_usd if max_cost_usd is not None else cfg.budget.max_cost_usd
    effective_max_tokens = max_tokens if max_tokens is not None else cfg.budget.max_tokens
    effective_max_turns = max_turns if max_turns is not None else cfg.budget.max_redteam_turns

    async def _run_redteam() -> int:
        janitor = ContainerJanitor()
        run_id = f"dasv-redteam-{uuid4().hex[:10]}"
        provider = DockerSandboxProvider(janitor=janitor)
        health = await provider.health_check()
        if not health.healthy:
            typer.echo(
                f"Error: Docker runtime unavailable ({health.detail}). "
                "Start Docker/OrbStack to execute Tier 2 red-team sandboxes.",
                err=True,
            )
            return EXIT_RUNTIME_ERROR

        budget = BudgetGuard(
            max_cost_usd=effective_max_cost,
            max_tokens=effective_max_tokens,
            base_url=effective_base_url,
        )
        llm = UnifiedLLMClient(
            base_url=effective_base_url,
            budget=budget,
            api_key_env=api_key_env,
        )
        runner_obj = RedTeamRunner(
            llm=llm,
            provider=provider,
            budget=budget,
            sandbox_config=cfg.sandbox,
        )
        try:
            rt_report = await runner_obj.run(
                spec,
                adapter,
                attacker_model=effective_attacker,
                fixer_model=effective_fixer,
                hack_classes=selected_classes,
                max_turns=effective_max_turns,
                propose_fix=propose_fix,
                gold_sanity_replays=gold_sanity_replays,
                apply_verified_fixes=apply_fixes,
                run_id=run_id,
                out_dir=out_dir,
            )
        except DasvError as exc:
            typer.echo(f"Error: {exc}", err=True)
            return exc.exit_code
        finally:
            await janitor.reap_all(run_id)

        table = Table(title=f"Adversarial Red-Team: {spec.task_id}")
        table.add_column("Metric", style="bold")
        table.add_column("Value")
        table.add_row("Attacker Model", rt_report.attacker_model)
        table.add_row("Hack Classes Attempted", ", ".join(rt_report.hack_classes_attempted))
        table.add_row("Confirmed Exploits", str(len(rt_report.exploits)))
        table.add_row("Unpatched Exploits", str(rt_report.unpatched_exploit_count))
        table.add_row("Tokens Used", str(rt_report.total_tokens))
        table.add_row("Spend (USD)", f"${rt_report.total_cost_usd:.6f}")
        if rt_report.budget_exceeded:
            table.add_row("Budget Status", "EXCEEDED (partial results saved)")
        console.print(table)

        if rt_report.budget_exceeded:
            return EXIT_BUDGET_EXCEEDED
        return EXIT_PASS if rt_report.unpatched_exploit_count == 0 else EXIT_AUDIT_FAIL

    code = asyncio.run(_run_redteam())
    if code != EXIT_PASS:
        raise typer.Exit(code=code)


def _parse_curriculum_buckets(raw: str | None) -> list[CurriculumBucket]:
    if raw is None or not raw.strip():
        return [CurriculumBucket.GOLDILOCKS]
    selected: list[CurriculumBucket] = []
    for part in raw.split(","):
        cleaned = part.strip().lower()
        if not cleaned:
            continue
        try:
            bucket = CurriculumBucket(cleaned)
        except ValueError as exc:
            valid = ", ".join(b.value for b in CurriculumBucket)
            raise DasvError(
                f"Invalid curriculum bucket '{cleaned}'. Valid values: {valid}.",
                exit_code=EXIT_CONFIG_ERROR,
            ) from exc
        if bucket not in selected:
            selected.append(bucket)
    if not selected:
        raise DasvError("At least one curriculum bucket must be specified.", exit_code=2)
    return selected


def _parse_judge_configs(
    judges_raw: str | None,
    base_urls_raw: str | None,
    default_base_url: str,
) -> list[tuple[str, str]]:
    """Parse `--judges modelA,modelB` and optional `--judge-base-urls urlA,urlB`."""
    if judges_raw is None or not judges_raw.strip():
        return []
    models = [part.strip() for part in judges_raw.split(",") if part.strip()]
    if len(models) < 2:
        raise DasvError(
            "`--judges` requires at least 2 comma-separated judge models (e.g. modelA,modelB).",
            exit_code=EXIT_CONFIG_ERROR,
        )
    if base_urls_raw is None or not base_urls_raw.strip():
        return [(m, default_base_url) for m in models]
    urls = [part.strip() for part in base_urls_raw.split(",") if part.strip()]
    if len(urls) == 1:
        return [(m, urls[0]) for m in models]
    if len(urls) != len(models):
        raise DasvError(
            f"`--judge-base-urls` count ({len(urls)}) must be 1 or match `--judges` "
            f"count ({len(models)}).",
            exit_code=EXIT_CONFIG_ERROR,
        )
    return list(zip(models, urls, strict=True))


@app.command()
def rubric(
    path: Annotated[
        Path, typer.Argument(help="Path to rubric-based RL environment directory or rubric file.")
    ],
    env_format: Annotated[
        str | None,
        typer.Option("--format", help="Force adapter format (harbor, openenv, verifiers_v1, hud)."),
    ] = None,
    run_perturbations: Annotated[
        bool,
        typer.Option(
            "--run-perturbations/--static-only",
            help="Run 3 deterministic verbosity/fluff perturbations against a judge model.",
        ),
    ] = False,
    baseline_response: Annotated[
        str | None,
        typer.Option(
            "--baseline-response",
            help="Known-wrong or partial baseline response used for RULE-RUBR-002 perturbations.",
        ),
    ] = None,
    inflation_threshold: Annotated[
        float,
        typer.Option(
            "--inflation-threshold",
            min=0.0,
            max=1.0,
            help="Minimum score inflation delta triggering RULE-RUBR-002 (default: 0.15).",
        ),
    ] = 0.15,
    judges: Annotated[
        str | None,
        typer.Option(
            "--judges",
            help="Comma-separated >=2 judge model IDs for CHERRL cross-family divergence.",
        ),
    ] = None,
    judge_base_urls: Annotated[
        str | None,
        typer.Option(
            "--judge-base-urls",
            help="Comma-separated base URLs for `--judges` (1 shared or 1 per judge).",
        ),
    ] = None,
    samples_path: Annotated[
        Path | None,
        typer.Option(
            "--samples",
            help="Optional JSONL batch (`prompt`, `response`) for CHERRL judge divergence.",
        ),
    ] = None,
    tau_threshold: Annotated[
        float,
        typer.Option(
            "--tau-threshold",
            min=-1.0,
            max=1.0,
            help="Minimum Kendall tau-b agreement before flagging RULE-RUBR-003 (default: 0.70).",
        ),
    ] = DEFAULT_TAU_THRESHOLD,
    delta_threshold: Annotated[
        float,
        typer.Option(
            "--delta-threshold",
            min=0.0,
            max=1.0,
            help="Maximum mean absolute judge score delta before flagging (default: 0.15).",
        ),
    ] = DEFAULT_DELTA_THRESHOLD,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Fixed RNG seed for CHERRL bootstrap CI (default: 42)."),
    ] = 42,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Judge model ID for dynamic verbosity perturbations."),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option("--base-url", help="OpenAI-compatible inference base URL for judge LLM."),
    ] = None,
    api_key_env: Annotated[
        str,
        typer.Option("--api-key-env", help="Env var holding cloud API key (never enters sandbox)."),
    ] = "OPENAI_API_KEY",
    max_cost_usd: Annotated[
        float | None,
        typer.Option("--max-cost-usd", min=0.0, help="Hard USD cap for judge calls."),
    ] = None,
    max_tokens: Annotated[
        int | None,
        typer.Option("--max-tokens", min=1, help="Hard token cap for judge calls."),
    ] = None,
    out_dir: Annotated[
        Path | None,
        typer.Option("--out-dir", help="Write rubric_audit.json and prorubric_gate.yaml here."),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Optional path to dasv.toml."),
    ] = None,
) -> None:
    """Run ProRubric (RULE-RUBR-001/002) & CHERRL judge divergence (RULE-RUBR-003)."""
    if not path.exists():
        typer.echo(f"Error: Rubric target path not found: '{path}'", err=True)
        raise typer.Exit(code=EXIT_CONFIG_ERROR)

    try:
        cfg = load_config(config_path)
        effective_base_url = base_url or cfg.llm.base_url
        judge_specs = _parse_judge_configs(judges, judge_base_urls, effective_base_url)
        loaded_samples = (
            load_judge_samples_jsonl(samples_path) if samples_path is not None else None
        )
    except DasvError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=exc.exit_code) from exc

    spec = None
    if path.is_dir():
        try:
            adapter = AdapterRegistry().resolve(path, forced_format=env_format)
            spec = adapter.load(path)
        except DasvError:
            if env_format is not None:
                typer.echo(
                    f"Error: Path '{path}' could not be loaded with --format '{env_format}'.",
                    err=True,
                )
                raise typer.Exit(code=EXIT_CONFIG_ERROR) from None

    effective_model = model or cfg.llm.attacker_model
    effective_max_cost = max_cost_usd if max_cost_usd is not None else cfg.budget.max_cost_usd
    effective_max_tokens = max_tokens if max_tokens is not None else cfg.budget.max_tokens

    async def _run_rubric() -> int:
        auditor = ProRubricAuditor()
        scorer: RubricJudgeScorer | None = None
        divergence_scorers: list[tuple[str, RubricJudgeScorer]] = []
        budget: BudgetGuard | None = None

        needs_llm = run_perturbations or bool(judge_specs)
        if needs_llm:
            all_urls = [effective_base_url, *(u for _m, u in judge_specs)]
            budget_url = next(
                (u for u in all_urls if not is_local_endpoint(u)),
                effective_base_url,
            )
            budget = BudgetGuard(
                max_cost_usd=effective_max_cost,
                max_tokens=effective_max_tokens,
                base_url=budget_url,
            )
            if run_perturbations:
                llm = UnifiedLLMClient(
                    base_url=effective_base_url,
                    budget=budget,
                    api_key_env=api_key_env,
                )
                scorer = LLMRubricJudgeScorer(llm, model=effective_model)
            for j_model, j_url in judge_specs:
                j_llm = UnifiedLLMClient(
                    base_url=j_url,
                    budget=budget,
                    api_key_env=api_key_env,
                )
                divergence_scorers.append((j_model, LLMRubricJudgeScorer(j_llm, model=j_model)))

        budget_tripped = False
        try:
            rubric_report = await auditor.audit(
                path,
                spec=spec,
                scorer=scorer,
                baseline_response=baseline_response,
                inflation_threshold=inflation_threshold,
                divergence_judges=divergence_scorers or None,
                divergence_samples=loaded_samples,
                tau_threshold=tau_threshold,
                delta_threshold=delta_threshold,
                seed=seed,
                out_dir=out_dir,
            )
        except BudgetExceededError as exc:
            budget_tripped = True
            typer.echo(f"Error: {exc}", err=True)
            # Still flush $0 static analysis when budget aborts dynamic judge perturbations.
            rubric_report = await auditor.audit(path, spec=spec, scorer=None, out_dir=out_dir)
            rubric_report = rubric_report.model_copy(update={"budget_exceeded": True})
        except DasvError as exc:
            typer.echo(f"Error: {exc}", err=True)
            return exc.exit_code

        if budget is not None:
            snap = await budget.snapshot()
            rubric_report = rubric_report.model_copy(
                update={
                    "total_tokens": snap.committed_tokens,
                    "total_cost_usd": snap.committed_cost_usd,
                }
            )
            if out_dir is not None:
                ProRubricAuditor.write_artifacts(out_dir, rubric_report)

        table = Table(title=f"ProRubric Audit: {rubric_report.task_id}")
        table.add_column("Metric", style="bold")
        table.add_column("Value")
        table.add_row("Rubric Definitions Inspected", str(len(rubric_report.targets_inspected)))
        table.add_row(
            "Additive Loopholes (RULE-RUBR-001)",
            str(sum(1 for f in rubric_report.findings if f.rule_id == "RULE-RUBR-001")),
        )
        if rubric_report.verbosity_audit is not None:
            va = rubric_report.verbosity_audit
            table.add_row("Baseline Wrong Score", f"{va.baseline_score:.2f}")
            table.add_row("Max Fluff Inflation Delta", f"+{va.max_delta:.2f}")
            table.add_row(
                "Verbosity Inflation (RULE-RUBR-002)",
                "FAIL" if not va.passed else "PASS",
            )
        if rubric_report.judge_divergence is not None:
            jd = rubric_report.judge_divergence
            table.add_row("CHERRL Judges", ", ".join(jd.judges))
            table.add_row("CHERRL Samples Scored", str(jd.sample_count))
            table.add_row(
                "Reference-Free Pipeline",
                "YES (HIGH risk)" if jd.is_reference_free else "NO (anchored/gated)",
            )
            for pair in jd.pairs:
                table.add_row(
                    f"Pair {pair.judge_a} vs {pair.judge_b}",
                    (
                        f"tau={pair.kendall_tau:.3f} [{pair.tau_ci_low:.3f}, "
                        f"{pair.tau_ci_high:.3f}], rho={pair.spearman_rho:.3f}, "
                        f"mean|delta|={pair.mean_abs_delta:.3f}"
                    ),
                )
            table.add_row(
                "Judge Divergence (RULE-RUBR-003)",
                "FAIL" if not jd.passed else "PASS",
            )
        table.add_row("Recommended Gate Specs", str(len(rubric_report.recommended_gate_specs)))
        if rubric_report.total_tokens > 0 or rubric_report.total_cost_usd > 0.0:
            table.add_row("Tokens Used", str(rubric_report.total_tokens))
            table.add_row("Spend (USD)", f"${rubric_report.total_cost_usd:.6f}")
        console.print(table)

        if budget_tripped or rubric_report.budget_exceeded:
            return EXIT_BUDGET_EXCEEDED
        return EXIT_PASS if rubric_report.passed else EXIT_AUDIT_FAIL

    code = asyncio.run(_run_rubric())
    if code != EXIT_PASS:
        raise typer.Exit(code=code)


@app.command()
def profile(
    path: Annotated[
        Path, typer.Argument(help="Path to single RL environment or parent taskset directory.")
    ],
    env_format: Annotated[
        str | None,
        typer.Option("--format", help="Force adapter format (harbor, openenv, verifiers_v1, hud)."),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option("--base-url", help="OpenAI-compatible base URL (Ollama/MLX/vLLM/cloud)."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Base policy model ID (default: dasv.toml attacker_model)."),
    ] = None,
    api_key_env: Annotated[
        str,
        typer.Option("--api-key-env", help="Env var holding cloud API key (never enters sandbox)."),
    ] = "OPENAI_API_KEY",
    rollouts: Annotated[
        int,
        typer.Option("--rollouts", min=1, help="Rollouts G per task (default: 8)."),
    ] = 8,
    traces: Annotated[
        Path | None,
        typer.Option(
            "--traces",
            help="Optional TrainingTrace JSONL (`task_id`, `reward`) for $0 offline profiling.",
        ),
    ] = None,
    goldilocks_low: Annotated[
        float,
        typer.Option(
            "--goldilocks-low",
            min=0.0,
            max=1.0,
            help="Lower pass@1 band for GOLDILOCKS.",
        ),
    ] = 0.05,
    goldilocks_high: Annotated[
        float,
        typer.Option(
            "--goldilocks-high",
            min=0.0,
            max=1.0,
            help="Upper pass@1 band for GOLDILOCKS.",
        ),
    ] = 0.40,
    include_buckets: Annotated[
        str | None,
        typer.Option(
            "--include-buckets",
            help="Comma-separated CurriculumBuckets to export (default: goldilocks).",
        ),
    ] = None,
    export_curriculum: Annotated[
        Path | None,
        typer.Option(
            "--export-curriculum",
            help="Write calibrated curriculum manifest (.jsonl or .yaml).",
        ),
    ] = None,
    max_cost_usd: Annotated[
        float | None,
        typer.Option("--max-cost-usd", min=0.0, help="Hard USD budget ceiling."),
    ] = None,
    max_tokens: Annotated[
        int | None,
        typer.Option("--max-tokens", min=1, help="Hard token budget ceiling."),
    ] = None,
    out_dir: Annotated[
        Path | None,
        typer.Option("--out-dir", help="Write profile.json artifact to this directory."),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Optional path to dasv.toml."),
    ] = None,
) -> None:
    """Run Tier 3 pass@1/pass@k/pass^k & GRPO/DAPO zero-gradient signal profiler + export."""
    try:
        cfg = load_config(config_path)
        buckets = _parse_curriculum_buckets(include_buckets)
        discovered = discover_taskset_specs(path, forced_format=env_format)
        traces_map = load_rewards_from_traces_jsonl(traces) if traces is not None else None
    except DasvError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=exc.exit_code) from exc

    specs = [spec for _adapter, spec in discovered]
    effective_base_url = base_url or cfg.llm.base_url
    effective_model = model or cfg.llm.attacker_model
    effective_max_cost = max_cost_usd if max_cost_usd is not None else cfg.budget.max_cost_usd
    effective_max_tokens = max_tokens if max_tokens is not None else cfg.budget.max_tokens

    async def _run_profile() -> int:
        janitor = ContainerJanitor()
        run_id = f"dasv-profile-{uuid4().hex[:10]}"
        provider: DockerSandboxProvider | None = None
        llm: UnifiedLLMClient | None = None
        budget = BudgetGuard(
            max_cost_usd=effective_max_cost,
            max_tokens=effective_max_tokens,
            base_url=effective_base_url,
        )

        if traces_map is None:
            provider = DockerSandboxProvider(janitor=janitor)
            health = await provider.health_check()
            if not health.healthy:
                typer.echo(
                    f"Error: Docker runtime unavailable ({health.detail}). "
                    "Start Docker/OrbStack or pass `--traces <file.jsonl>` for offline profiling.",
                    err=True,
                )
                return EXIT_RUNTIME_ERROR
            llm = UnifiedLLMClient(
                base_url=effective_base_url,
                budget=budget,
                api_key_env=api_key_env,
            )

        profiler = SignalProfiler(
            llm=llm,
            provider=provider,
            budget=budget,
            sandbox_config=cfg.sandbox,
        )
        try:
            prof_report = await profiler.profile_specs(
                specs,
                base_model=effective_model,
                rollouts_g=rollouts,
                run_id=run_id,
                goldilocks_low=goldilocks_low,
                goldilocks_high=goldilocks_high,
                traces_by_task=traces_map,
                out_dir=out_dir,
                export_curriculum_path=export_curriculum,
                include_buckets=buckets,
            )
        except DasvError as exc:
            typer.echo(f"Error: {exc}", err=True)
            return exc.exit_code
        finally:
            if provider is not None:
                await janitor.reap_all(run_id)

        table = Table(title=f"RLVR Signal Profile ({prof_report.base_model}, G={rollouts})")
        table.add_column("Metric", style="bold")
        table.add_column("Value")
        table.add_row("Tasks Profiled", str(len(prof_report.tasks)))
        table.add_row("Mean pass@1", f"{prof_report.mean_pass_at_1 * 100.0:.1f}%")
        table.add_row("Mean Advantage Variance", f"{prof_report.mean_advantage_variance:.4f}")
        table.add_row("Zero-Grad Compute Waste", f"{prof_report.zero_grad_waste_pct:.1f}%")
        table.add_row(
            "Buckets",
            ", ".join(f"{k}={v}" for k, v in prof_report.bucket_counts.items() if v > 0) or "-",
        )
        if export_curriculum is not None:
            table.add_row("Exported Curriculum", str(export_curriculum))
        if prof_report.total_tokens > 0 or prof_report.total_cost_usd > 0.0:
            table.add_row("Tokens Used", str(prof_report.total_tokens))
            table.add_row("Spend (USD)", f"${prof_report.total_cost_usd:.6f}")
        console.print(table)

        if prof_report.budget_exceeded:
            return EXIT_BUDGET_EXCEEDED
        return EXIT_PASS

    code = asyncio.run(_run_profile())
    if code != EXIT_PASS:
        raise typer.Exit(code=code)


@app.command(name="export")
def export_cmd(
    export_format: Annotated[
        str,
        typer.Option(
            "--format",
            help="Target RL trainer format: trl | verl | prime_rl | openenv.",
        ),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", help="Output file path (or directory) for exported dataset."),
    ],
    curriculum: Annotated[
        Path | None,
        typer.Option(
            "--curriculum",
            help="Calibrated curriculum manifest (.jsonl or .yaml) from `dasv profile`.",
        ),
    ] = None,
    exploits: Annotated[
        Path | None,
        typer.Option(
            "--exploits",
            help="ExploitTrace JSONL (`exploits.jsonl`) from `dasv redteam` (for openenv).",
        ),
    ] = None,
    include_buckets: Annotated[
        str | None,
        typer.Option(
            "--include-buckets",
            help="Optional comma-separated CurriculumBucket filter (default: goldilocks).",
        ),
    ] = None,
) -> None:
    """Export certified GOLDILOCKS curricula & exploit traces to TRL/veRL/prime-rl/OpenEnv."""
    fmt = export_format.strip().lower()
    if fmt not in EXPORT_FORMATS:
        typer.echo(
            f"Error: Unsupported export format '{export_format}'. "
            f"Supported formats: {', '.join(EXPORT_FORMATS)}.",
            err=True,
        )
        raise typer.Exit(code=EXIT_CONFIG_ERROR)

    try:
        buckets = _parse_curriculum_buckets(include_buckets)
        result: ExportResult
        if fmt == "openenv":
            loaded_exploits = load_exploit_traces(exploits) if exploits is not None else []
            loaded_entries = (
                load_curriculum_entries(curriculum, include_buckets=buckets)
                if curriculum is not None
                else []
            )
            result = export_openenv_traces(
                exploits=loaded_exploits,
                entries=loaded_entries,
                out_path=out,
            )
        else:
            if curriculum is None:
                raise DasvError(
                    f"`--curriculum` is required for `--format {fmt}`.",
                    exit_code=EXIT_CONFIG_ERROR,
                )
            loaded_entries = load_curriculum_entries(curriculum, include_buckets=buckets)
            if fmt == "trl":
                result = export_trl(loaded_entries, out)
            elif fmt == "verl":
                result = export_verl(loaded_entries, out)
            else:
                result = export_prime_rl(loaded_entries, out)
    except DasvError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=exc.exit_code) from exc

    table = Table(title=f"RL Trainer Export ({result.format})")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Records Exported", str(result.record_count))
    table.add_row("Output Path", str(result.output_path))
    if result.companion_paths:
        table.add_row(
            "Companion Artifacts",
            ", ".join(str(p) for p in result.companion_paths),
        )
    table.add_row("Manifest SHA-256", result.manifest_sha[:16])
    console.print(table)


def main() -> None:
    """Console entry point (`dasv = dasv.cli.app:main`)."""
    app()
