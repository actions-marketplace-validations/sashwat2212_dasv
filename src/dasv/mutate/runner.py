"""Tier 1 Mutation Suite Runner orchestrating Anti-Patch & ImpossibleBench suites (SDD-02)."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path

from pydantic import BaseModel, Field

from dasv.adapters.base import EnvironmentAdapter
from dasv.core.config import SandboxConfig
from dasv.core.schemas import AuditFinding, EnvSpec
from dasv.mutate.anti_patches import (
    AntiPatchMutation,
    AntiPatchOutcome,
    generate_anti_patches,
    summarize_anti_patch_outcomes,
)
from dasv.mutate.impossible_bench import (
    ConflictingSpecVariant,
    OneOffMutation,
    OneOffOutcome,
    generate_conflicting_spec_variants,
    generate_one_off_mutations,
    summarize_one_off_outcomes,
)
from dasv.sandbox.provider import SandboxProvider
from dasv.verifier.clean_grader import CleanImageGrader, GradingContext, ensure_container_image

OneOffOracleEvaluator = Callable[[EnvSpec, OneOffMutation], Awaitable[float]]


class MutationReport(BaseModel):
    """Structured output of `dasv mutate` ($0 deterministic mutation suite)."""

    task_id: str
    dry_run: bool = False
    anti_patches_generated: int = 0
    anti_patch_outcomes: list[AntiPatchOutcome] = Field(default_factory=list[AntiPatchOutcome])
    mutation_survival_rate: float | None = None
    one_off_mutations_generated: int = 0
    one_off_outcomes: list[OneOffOutcome] = Field(default_factory=list[OneOffOutcome])
    dead_assertions_count: int = 0
    conflicting_variants: list[ConflictingSpecVariant] = Field(
        default_factory=list[ConflictingSpecVariant]
    )
    findings: list[AuditFinding] = Field(default_factory=list[AuditFinding])

    @property
    def passed(self) -> bool:
        survival_bad = self.mutation_survival_rate is not None and self.mutation_survival_rate > 0.0
        return not survival_bad and self.dead_assertions_count == 0


class MutationSuiteRunner:
    """Runs deterministic Anti-Patch and ImpossibleBench mutations at $0 API cost."""

    def __init__(
        self,
        *,
        provider: SandboxProvider | None = None,
        sandbox_config: SandboxConfig | None = None,
        one_off_evaluator: OneOffOracleEvaluator | None = None,
    ) -> None:
        self._provider = provider
        self._sandbox_config = sandbox_config or SandboxConfig()
        self._one_off_evaluator = one_off_evaluator

    async def run(
        self,
        spec: EnvSpec,
        *,
        adapter: EnvironmentAdapter | None = None,
        run_id: str = "dasv-mutate",
        include_impossible_bench: bool = True,
        max_source_files: int = 3,
        max_one_off: int = 5,
        dry_run: bool = False,
        out_dir: Path | None = None,
    ) -> MutationReport:
        """Synthesize and (unless `dry_run=True`) grade all deterministic mutations."""
        anti_patches: list[AntiPatchMutation] = generate_anti_patches(
            spec, max_files=max_source_files
        )
        one_off_muts: list[OneOffMutation] = (
            generate_one_off_mutations(spec, max_mutations=max_one_off)
            if include_impossible_bench
            else []
        )
        conflicting: list[ConflictingSpecVariant] = (
            generate_conflicting_spec_variants(spec) if include_impossible_bench else []
        )

        if out_dir is not None:
            self._write_generated_diffs(out_dir, anti_patches, one_off_muts, conflicting)

        if dry_run:
            report = MutationReport(
                task_id=spec.task_id,
                dry_run=True,
                anti_patches_generated=len(anti_patches),
                one_off_mutations_generated=len(one_off_muts),
                conflicting_variants=conflicting,
            )
            if out_dir is not None:
                self._write_report_json(out_dir, report)
            return report

        effective_spec = spec
        needs_container = bool(anti_patches) or (
            include_impossible_bench
            and bool(one_off_muts)
            and spec.has_oracle
            and self._one_off_evaluator is None
        )
        if self._provider is not None and needs_container:
            effective_spec = await ensure_container_image(spec, self._provider, run_id)

        anti_outcomes: list[AntiPatchOutcome] = []
        if self._provider is not None:
            grader = CleanImageGrader(self._provider, sandbox_config=self._sandbox_config)
            for idx, mut in enumerate(anti_patches):
                artifact = mut.to_extracted_artifact()
                outcome = await grader.grade(effective_spec, artifact, f"{run_id}-ap-{idx}")
                anti_outcomes.append(
                    AntiPatchOutcome(
                        mutator_id=mut.mutator_id,
                        name=mut.name,
                        target_file=mut.target_file,
                        reward=outcome.reward,
                        survived=outcome.reward > 0.0,
                        detail=outcome.stderr_truncated or outcome.stdout_truncated,
                    )
                )

        survival_rate, ap_findings = summarize_anti_patch_outcomes(anti_outcomes)

        one_off_outcomes: list[OneOffOutcome] = []
        if include_impossible_bench and one_off_muts and spec.has_oracle:
            for mut in one_off_muts:
                oracle_reward = await self._evaluate_one_off(effective_spec, mut, adapter, run_id)
                if oracle_reward is not None:
                    one_off_outcomes.append(
                        OneOffOutcome(
                            mutation_id=mut.mutation_id,
                            target_test_file=mut.target_test_file,
                            line_number=mut.line_number,
                            oracle_reward=oracle_reward,
                            dead_assertion=oracle_reward > 0.0,
                        )
                    )

        one_off_findings = summarize_one_off_outcomes(one_off_outcomes)
        dead_count = sum(1 for o in one_off_outcomes if o.dead_assertion)

        report = MutationReport(
            task_id=spec.task_id,
            dry_run=False,
            anti_patches_generated=len(anti_patches),
            anti_patch_outcomes=anti_outcomes,
            mutation_survival_rate=survival_rate if anti_outcomes else None,
            one_off_mutations_generated=len(one_off_muts),
            one_off_outcomes=one_off_outcomes,
            dead_assertions_count=dead_count,
            conflicting_variants=conflicting,
            findings=[*ap_findings, *one_off_findings],
        )
        if out_dir is not None:
            self._write_report_json(out_dir, report)
        return report

    async def _evaluate_one_off(
        self,
        spec: EnvSpec,
        mut: OneOffMutation,
        adapter: EnvironmentAdapter | None,
        run_id: str,
    ) -> float | None:
        if self._one_off_evaluator is not None:
            return await self._one_off_evaluator(spec, mut)
        if self._provider is None or adapter is None:
            return None

        # Copy environment into a clean temporary dir and overwrite ONLY the mutated test file.
        with tempfile.TemporaryDirectory(prefix="dasv_oneoff_") as tmp_dir:
            mutated_root = Path(tmp_dir) / spec.root_path.name
            shutil.copytree(spec.root_path, mutated_root)
            target_path = mutated_root / mut.target_test_file
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(mut.mutated_test_source, encoding="utf-8")

            mutated_spec = spec.model_copy(update={"root_path": mutated_root})
            grader = CleanImageGrader(self._provider, sandbox_config=self._sandbox_config)
            ctx = GradingContext(
                grader,
                run_id=f"{run_id}-{mut.mutation_id.lower()}",
                timeout_sec=float(self._sandbox_config.episode_timeout_sec),
            )
            reward, _details = await adapter.run_oracle(mutated_spec, ctx)
            return float(reward)

    @staticmethod
    def _write_generated_diffs(
        out_dir: Path,
        anti_patches: list[AntiPatchMutation],
        one_off_muts: list[OneOffMutation],
        conflicting: list[ConflictingSpecVariant],
    ) -> None:
        ap_dir = out_dir / "anti_patches"
        ap_dir.mkdir(parents=True, exist_ok=True)
        for idx, mut in enumerate(anti_patches, start=1):
            (ap_dir / f"{idx:02d}_{mut.mutator_id}.diff").write_text(
                mut.patch_diff, encoding="utf-8"
            )

        if one_off_muts:
            oo_dir = out_dir / "impossible_one_off"
            oo_dir.mkdir(parents=True, exist_ok=True)
            for mut in one_off_muts:
                (oo_dir / f"{mut.mutation_id}.py").write_text(
                    mut.mutated_test_source, encoding="utf-8"
                )

        if conflicting:
            cf_dir = out_dir / "impossible_conflicting"
            cf_dir.mkdir(parents=True, exist_ok=True)
            for var in conflicting:
                (cf_dir / f"{var.variant_id}.py").write_text(
                    var.mutated_test_source, encoding="utf-8"
                )

    @staticmethod
    def _write_report_json(out_dir: Path, report: MutationReport) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "mutations.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
