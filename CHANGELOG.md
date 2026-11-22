# Changelog

All notable changes to **`dasv`** (*Deterministic Adversarial Sandbox Verifier*) are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Multi-seed ablation aggregator & extended significance tests** in `recipes/lora_dr_grpo_ablation/plot_ablation.py` (`--seeds-dir`, `compute_multiseed_summary`, `Arm 2 vs Arm 0` / `Arm 1 vs Arm 0` paired bootstrap, and `pass@k` exact McNemar).
- **arXiv-style Technical Report** (`docs/TECHNICAL-REPORT.md`) consolidating `KPI-RES-01`, `KPI-RES-02`, and `KPI-RES-03` empirical results with embedded SVG telemetry.
- **Controlled 3-Arm LoRA-Dr.GRPO 2-Seed Ablation Artifacts** (`results/v1_ablation/{seed_42,seed_1337,multiseed_summary.json}`) proving `dasv` hardening eliminates weak-assertion reward hacking (`0/320` steps vs. `47/320` in Arm 1), outperforms Unhardened RLVR on both seeds (2/2), cuts `pass^4` variance by 2.2×, and achieves statistically significant pooled held-out lift (`+3.71 pp` `pass@1`, $p=0.0456$; `+7.03 pp` `pass^4`, $p=0.0072$; `+10.75 pp` MBPP+ `pass^4`, $p=0.0088$) on `Qwen/Qwen2.5-Coder-3B-Instruct`.
- **Production `README.md`** with 60-second `uvx dasv` quickstart, 9-command CLI reference, `dasv-action` workflow snippet, and empirical summary tables.

### Fixed
- **Repo-relative SARIF 2.1.0 artifact URIs** (`src/dasv/reporting/sarif_writer.py`): Automatically resolves audited subdirectory paths relative to the enclosing `.git` root so GitHub Code Scanning attaches inline PR diff annotations (`annotations_count >= 1`) directly on modified files, and corrected `tool.driver.informationUri`.

## [0.1.0] - 2026-11-21

### Added
- **4-Plane Core & CLI (`dasv`)**: Frozen Pydantic v2 schemas, strict `dasv.toml` config, deterministic exit codes `0–5`, atomic `BudgetGuard` with `\$0` local-endpoint bypass, and 9 CLI commands (`doctor`, `inspect`, `audit`, `mutate`, `redteam`, `rubric`, `profile`, `export`, `report`).
- **Cross-Standard Adapters (`dasv.adapters`)**: Harbor (`task.toml` v1.1 `allow_internet`/`docker_image`), OpenEnv (`server/Dockerfile`), Prime Intellect `verifiers` v1 (`vf.Taskset` + `@vf.reward` + `RULE-ADPT-001` legacy v0 detection), and HUD (`RULE-ADPT-002` + container-backed execution) with `NativeDelegateRunner`.
- **Tier 0–3 Verification Engines**:
  - Tier 0 static boundary (`RULE-BNDY-001..004`) and hermeticity (`RULE-HERM-001..002`) probes.
  - Tier 1 Clean-Image Out-of-Container Grader, $N$-trial Oracle/No-op determinism (`pass^k`), deterministic Anti-Patch & *ImpossibleBench* mutators (`RULE-MUT-001..002`).
  - Tier 2 budget-capped ReAct Hacker-Fixer loop + 3-invariant Gold-Sanity Gate + *ProRubric* (`RULE-RUBR-001..002`) & *CHERRL* cross-family judge divergence (`RULE-RUBR-003`).
  - Tier 3 RLVR `SignalProfiler` (`pass@1`, `pass@k`, `pass^k`, GRPO zero-gradient probability, `CurriculumBucket`) + `TRL`/`veRL`/`prime-rl`/`OpenEnv` exporters.
- **Reporting & CI/CD**: Synchronized JSON, **SARIF 2.1.0**, offline zero-CDN HTML Certification Cards, `manifest.json`, composite GitHub Action (`action/action.yml`), and OIDC PyPI/TestPyPI publishing workflow (`.github/workflows/publish.yml`).
- **Empirical Benchmarks & Recipes**:
  - `KPI-RES-01` Planted Vulnerability Corpus (`benchmarks/planted_vulns/`, 100% recall / 0% false alarms).
  - `KPI-RES-02` 45-task public RL environment audit (`results/public_audit/`, \$0.00 API spend).
  - `KPI-RES-03` VRAM-safe LoRA-Dr.GRPO/DAPO GPU ablation recipe (`recipes/lora_dr_grpo_ablation/`).
