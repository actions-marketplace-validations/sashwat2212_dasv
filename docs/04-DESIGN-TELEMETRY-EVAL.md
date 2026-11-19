# Stage 4 / SDD-04: Telemetry, Statistical Signal Analysis & Reporting — `dasv`

> [!NOTE]
> **Subsystem**: `SDD-04` (`ARCH-COMP-01` CLI/Doctor + `ARCH-COMP-08` Statistical & RLVR Signal Analyzer + `ARCH-COMP-09` Artifact/Report Renderer)
> **Linked Features**: `FEAT-10`, `FEAT-17`, `FEAT-19`, `FEAT-21`, `FEAT-22`
> **Upstream Docs**: [02-PRD-FEATURES.md](./02-PRD-FEATURES.md) | [03-ARCHITECTURE.md](./03-ARCHITECTURE.md)
> **Package Modules**: `src/dasv/cli/`, `src/dasv/profile/stats.py`, `src/dasv/profile/signal_profiler.py`, `src/dasv/reporting/`

---

## 1. Exact Mathematical Estimators (`src/dasv/profile/stats.py`)

Every statistical metric reported by `dasv` uses numerically stable, unbiased formulas so published audit numbers stand up to frontier-lab peer review:

### 1.1 Unbiased `pass@k` (Exploration Headroom)
Given $n$ independent rollouts on a task with $c$ successes ($0 \le c \le n$) and $1 \le k \le n$:

$$\widehat{\text{pass@}k} = 1 - \frac{\binom{n - c}{k}}{\binom{n}{k}} = 1 - \prod_{i=0}^{k-1} \left(1 - \frac{c}{n - i}\right)$$

Computed via log-space / multiplicative product (avoiding large integer factorial overflow; returns `1.0` directly when $n - c < k$).

### 1.2 `pass^k` (Deployment Reliability & Flakiness Exposure)
Following $\tau$-bench (`arXiv:2406.12045`) and Anthropic's agent evaluation taxonomy (Jan 2026), with empirical single-trial pass probability $\hat{p} = c / n$:

$$\widehat{\text{pass}^k} = \hat{p}^k = \left(\frac{c}{n}\right)^k$$

- For Tier 1 Oracle certification ($n = 10$), any single flaky failure drops $\widehat{\text{pass}^{10}}$ from `1.000` to `0.349`, making subtle flakiness immediately visible on the HTML Certification Card.

### 1.3 GRPO / DAPO Policy-Gradient Signal & Zero-Gradient Compute Waste
For a task sampled with group size $G$ under a base policy producing binary rewards $r_1, \dots, r_G \in \{0, 1\}$ with empirical mean $\hat{p} = \frac{1}{G}\sum_{j=1}^G r_j$:
- **Within-Group Reward Variance**: $\hat{\sigma}_r^2 = \hat{p}(1 - \hat{p})$ (maximized at $\hat{p} = 0.5$; strictly `0` when all pass or all fail).
- **Zero-Gradient Group Probability**: Probability that a random GRPO/DAPO group of size $G$ has zero advantage variance (wasting rollout compute unless filtered via DAPO Dynamic Sampling):

$$P(\text{zero-grad} \mid \hat{p}, G) = \hat{p}^G + (1 - \hat{p})^G$$

- **Taskset Zero-Gradient Waste %**: Mean of $P(\text{zero-grad} \mid \hat{p}_i, G)$ across all tasks $i \in \mathcal{D}$, reported directly alongside task classification buckets:
  - `GOLDILOCKS`: $0.05 \le \text{pass@1} \le 0.40$ (configurable) and `replay_variance == 0`
  - `TRIVIAL_ZERO_GRAD`: $\text{pass@1} > 0.90$ (or `1.0` at $G$ rollouts)
  - `STARVED_ZERO_GRAD`: $\text{pass@}G == 0.0$
  - `FLAKY`: Tier 1 `replay_variance > 0.0`

---

## 2. `HackabilityScore (0–100)` & Deterministic Certification Grade (`A–F`)

### 2.1 `HackabilityScore` Formula (Lower is Better: `0.0` = Hardened, `100.0` = Trivially Hacked)
Computed transparently from four bounded sub-scores:

$$\text{HackabilityScore} = \min\!\Big(100.0,\; 40 \cdot I_{\text{crit\_boundary}} + 25 \cdot M_{\text{survival}} + 25 \cdot H_{\text{redteam}} + 10 \cdot R_{\text{rubric}}\Big)$$

Where:
- $I_{\text{crit\_boundary}} \in [0, 1]$: fraction of `CRITICAL` boundary rules (`RULE-BNDY-001..006`) failed (capped at `1.0` if $\ge 2$ critical breaches).
- $M_{\text{survival}} \in [0, 1]$: Tier 1 deterministic anti-patch **Mutation Survival Rate**.
- $H_{\text{redteam}} \in [0, 1]$: fraction of unpatched confirmed Tier 2 Hacker exploits (`0` if all exploits closed by `VERIFIED_FIX` patches or no exploits found).
- $R_{\text{rubric}} \in [0, 1]$: rubric hackability indicator (`0.5` for un-gated additive rubric or unanchored reference-free judge; `1.0` if verbosity perturbation inflation $\Delta r \ge 0.15$).

### 2.2 Deterministic Grade (`A–F`) Decision Table
| Grade | Strict Criteria (All Must Hold) |
|---|---|
| **`A` (Frontier-Ready)** | `0` CRITICAL/HIGH findings; Oracle = `100%`; No-op = `0%`; `replay_variance == 0`; `mutation_survival_rate == 0`; `0` unpatched red-team exploits; `HackabilityScore == 0.0`. |
| **`B` (Hardened w/ Advisories)** | `0` CRITICAL findings; Oracle = `100%`; No-op = `0%`; `replay_variance == 0`; `0` unpatched exploits; `HackabilityScore <= 10.0` (only `MEDIUM`/`LOW` advisories, e.g., floating image tag). |
| **`C` (Needs Hardening)** | `0` CRITICAL findings; Oracle = `100%`; No-op = `0%`; `HackabilityScore <= 25.0` (e.g., weak mutation coverage or non-gated additive rubric warning). |
| **`D` (High Risk)** | `1+` HIGH findings, minor flakiness (`0 < variance`), or `25.0 < HackabilityScore <= 50.0`. |
| **`F` (Rejected / Hackable)** | **Any** `CRITICAL` boundary breach (`RULE-BNDY-001..006`), **any** unpatched confirmed red-team exploit, `Oracle < 100%`, `No-op > 0%`, or `HackabilityScore > 50.0`. |

---

## 3. Synchronized Report Renderers (`ARCH-COMP-09` | `src/dasv/reporting/`)

### 3.1 `report.json` & `manifest.json`
- `report.json`: Direct `AuditReport.model_dump_json(indent=2)` output with `schema_version = "1.0.0"`.
- `manifest.json`: Independent `RunManifest` artifact recording `dasv_version`, `git_sha`, `host_os`, `host_arch`, `docker_server_arch`, `oci_image_digests`, `dataset_sha256`, `models_used`, `rng_seed`, `total_tokens`, `total_cost_usd`, and `cli_args`.

### 3.2 SARIF 2.1.0 Writer (`src/dasv/reporting/sarif_writer.py`)
Maps every `AuditFinding` cleanly into standard SARIF 2.1.0 so GitHub Code Scanning renders inline PR annotations directly on offending `Dockerfile`, `task.toml`, or test files:
- `tool.driver.name = "dasv"`, `semanticVersion = dasv.__version__`, full `rules[]` catalog populated with rule IDs (`RULE-BNDY-*`, `RULE-HERM-*`, `RULE-MUT-*`, `RULE-RUBR-*`) and markdown remediation help.
- Severity mapping: `CRITICAL`/`HIGH` → SARIF `"error"`; `MEDIUM` → `"warning"`; `LOW`/`INFO` → `"note"`.
- Relative URI normalization against repository root (`artifactLocation.uri` + `region.startLine`).

### 3.3 Zero-CDN Offline HTML Certification Card (`src/dasv/reporting/html_renderer.py`)
- Rendered via **Jinja2** (`autoescape=True` — prevents XSS from untrusted sandbox stdout or repo strings!) into a single self-contained `report.html` (< 250 KB) with **zero external `<script>`/`<link>` CDN requests**.
- Visual sections:
  1. **Executive Certification Header**: Large Grade Badge (`A–F`), `HackabilityScore` gauge (inline SVG), `emulated_arch_warning` banner (if run on macOS Rosetta/QEMU), and pinned `RunManifest` digest chip.
  2. **Trust Boundary & Determinism Matrix**: Oracle %, No-op %, `pass^k` decay bar chart (inline SVG), latency p50/p95, peak RSS.
  3. **Findings Table**: Filterable by Tier (`0–3`) & Severity (`CRITICAL`–`INFO`) with expandable remediation snippets.
  4. **Exploit & Gold-Sanity Drawer**: Side-by-side exploit trajectory (`TrainingTrace` turns) + Fixer unified `.diff` + Gold-Sanity verdict badge (`VERIFIED_FIX` / `REJECTED_FIX`).
  5. **RLVR Signal Profile Panel**: `pass@1` vs. `pass@k` curve (inline SVG), `zero_grad_waste_pct`, and `CurriculumBucket` distribution.

---

## 4. `dasv doctor` & Drop-In GitHub Action (`src/dasv/cli/doctor.py`, `action/`)

- **`dasv doctor`**: Checks Python version (`>= 3.11`), `docker info` reachability, host arch vs. Docker server arch (explicitly warning macOS `arm64` users against publishing QEMU-emulated latency metrics), and availability of optional native validators (`harbor`, `openenv`, `vf-validate`).
- **`action/action.yml` (`dasv-action`)**: Composite GitHub Action running on `ubuntu-latest` (`linux/amd64`), installing `dasv` via `uvx`, executing `dasv audit <path> --tiers 0,1 --sarif dasv.sarif --html dasv.html`, uploading `dasv.sarif` to GitHub Security tab and `dasv.html` as a workflow artifact.

---

## 5. Automated Verification Matrix (`SDD-04`)

1. **Unit (`tests/unit/test_stats_and_grading.py`)**: Exact numerical unit tests for unbiased `pass@k` (edge cases $c=0$, $c=n$, $n-c<k$, large $n=1000$ without overflow), `pass^k`, zero-gradient probability $p^G + (1-p)^G$, `HackabilityScore`, and Grade `A–F` boundary conditions.
2. **Schema Validation (`tests/unit/test_reporting.py`)**: Validates emitted `report.sarif` against official SARIF 2.1.0 JSON Schema; verifies `report.html` contains zero `http://`/`https://` `<script>` or `<link>` external fetches and properly HTML-escapes injected `<script>alert(1)</script>` payloads from untrusted sandbox stdout.
