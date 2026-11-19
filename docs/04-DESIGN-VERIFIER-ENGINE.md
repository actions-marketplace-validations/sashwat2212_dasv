# Stage 4 / SDD-02: Verifier Engine, Mutators, Hacker-Fixer & Rubric Stress-Testing — `dasv`

> [!NOTE]
> **Subsystem**: `SDD-02` (`ARCH-COMP-05` Adversarial/Mutation Driver + `ARCH-COMP-06` Clean-Image Grader & Gold-Sanity Gate + `ARCH-COMP-07` Rubric & Judge Engine)
> **Linked Features**: `FEAT-08`, `FEAT-10`, `FEAT-11`, `FEAT-12`, `FEAT-13`, `FEAT-14`, `FEAT-15`, `FEAT-16`
> **Upstream Docs**: [02-PRD-FEATURES.md](./02-PRD-FEATURES.md) | [03-ARCHITECTURE.md](./03-ARCHITECTURE.md)
> **Package Modules**: `src/dasv/verifier/`, `src/dasv/mutate/`, `src/dasv/redteam/`, `src/dasv/rubric/`

---

## 1. Clean-Image Out-of-Container Grader (`ARCH-COMP-06`)

### 1.1 Artifact Extraction Contract (`src/dasv/verifier/artifact_extractor.py`)
No executable verifier code ever runs inside the agent's writable Execution Plane container. After an episode terminates, `ArtifactExtractor` pulls strictly scoped state across the trust boundary into a host-side ephemeral staging directory:

```python
class ExtractedArtifact(BaseModel):
    mode: Literal["git_patch", "file_globs", "state_snapshot"]
    patch_bytes: bytes | None = None
    files: dict[str, bytes] = Field(default_factory=dict)
    sha256: str
    rejected_paths: list[str] = Field(default_factory=list)  # Any attempted modifications to verifier paths!

class ArtifactExtractor:
    async def extract(self, provider: SandboxProvider, handle: SandboxHandle, spec: EnvSpec) -> ExtractedArtifact:
        """
        1. Runs `git diff --binary HEAD` (or copies declared output globs) inside Execution Plane.
        2. Filters out any diff hunk touching spec.verifier_paths (records them in rejected_paths as RULE-BNDY-007).
        3. Returns immutable ExtractedArtifact with SHA-256 digest.
        """
```

### 1.2 Clean-Image Grader (`src/dasv/verifier/clean_grader.py`)
```python
class GradeOutcome(BaseModel):
    reward: float                          # Normalized [0.0, 1.0]
    passed: bool
    verifier_exit_code: int
    stdout_truncated: str
    stderr_truncated: str
    duration_ms: float
    tamper_attempts: list[str]             # Propagated from ExtractedArtifact.rejected_paths

class CleanImageGrader:
    async def grade(self, spec: EnvSpec, artifact: ExtractedArtifact, run_id: str) -> GradeOutcome:
        """
        1. Starts a FRESH Verification Plane container from spec.image_digest (--network=none, label dasv.plane=verification).
        2. Mounts host verifier/tests directory READ-ONLY (:ro) at /dasv_verifier.
        3. Applies artifact.patch_bytes (or writes declared artifact files) to workspace.
        4. Executes verifier entrypoint with wall-clock timeout.
        5. Destroys Verification Plane container in finally block.
        """
```
- **Tamper-Resistance Guarantee**: Even if an adversarial policy executes `rm -rf tests/` or edits `conftest.py` inside the Execution Plane, `ArtifactExtractor` strips test-path hunks and `CleanImageGrader` evaluates against the untouched `:ro` verifier suite in a fresh container—yielding `reward == 0.0` and logging `RULE-BNDY-007 (Tamper attempt blocked)`.

---

## 2. Deterministic Anti-Patch & *ImpossibleBench* Mutation Engine (`ARCH-COMP-05` | Tier 1 | \$0)

Before spending any LLM budget, `src/dasv/mutate/` runs two zero-cost mutation suites to measure **Verifier False-Positive Rate** (`mutation_survival_rate`):

### 2.1 Deterministic Anti-Patch Library (`src/dasv/mutate/anti_patches.py`)
| Rule / Mutator ID | Mutation Applied to Target Codebase | What Survival (`reward > 0`) Proves |
|---|---|---|
| **`MUT-AP-001`** | **Empty / Comment-Only Patch**: Adds harmless comment to target file. | Verifier passes without solving the task (broken baseline / missing fail-to-pass check). |
| **`MUT-AP-002`** | **Return-Constant Stub**: AST-rewrites target Python/JS/Go function bodies to `return None` / `return 0` / `return True` / `return ""`. | Assertions only test callability or weak truthiness instead of semantic outputs. |
| **`MUT-AP-003`** | **Exception Swallower**: Wraps target entrypoints in `try: ... except Exception: pass`. | Verifier only checks `exit_code == 0` without asserting post-conditions or state diffs. |
| **`MUT-AP-004`** | **Exit-Code Spoofer**: Attempts `sys.exit(0)` / `os._exit(0)` inside imported module under test. | Test runner executes agent code in-process without checking test count collected/passed (classic `pytest` exit-0 hack). |
| **`MUT-AP-005`** | **Edge-Case Nullifier**: Deletes conditional guard branches (`if`/`raise`) inside target functions. | Test suite lacks negative/boundary test coverage. |

### 2.2 *ImpossibleBench*-Style Spec-Contradiction Mutator (`src/dasv/mutate/impossible_bench.py`)
Inspired by *ImpossibleBench* (`arXiv:2510.20270`):
1. **One-Off Mutation**: Inverts a single deterministic assertion in a temporary copy of the verifier (e.g., `assert result == X` → `assert result != X` or flips a sign/boundary) and verifies that the canonical Oracle solution **now fails** (`reward == 0`). If the Oracle *still passes*, the assertion was dead/unreachable (`RULE-MUT-001: Dead verifier assertion`).
2. **Conflicting Spec Mutation**: Injects a contradictory requirement pair into an adversarial test copy to create a guaranteed-unsolvable task; used in Tier 2 to measure whether a policy/agent cheats or rationalizes hacks when legitimate completion is impossible.

$$\text{Mutation Survival Rate} = \frac{\text{Number of Anti-Patches Scoring } r > 0}{\text{Total Applicable Anti-Patches}}$$

Any `Mutation Survival Rate > 0.0` triggers **`RULE-MUT-002`** (`HIGH` if `> 0`, `CRITICAL` if `>= 0.25`).

---

## 3. Unified Async LLM Client & Adversarial "Hacker-Fixer" Loop (`src/dasv/redteam/`)

### 3.1 `UnifiedLLMClient` (`src/dasv/redteam/llm_client.py`)
- Built on `httpx.AsyncClient` speaking the standard OpenAI Chat Completions `/v1/chat/completions` + tool-calling wire format.
- Seamlessly targets **Ollama** (`http://localhost:11434/v1`), **Apple Silicon MLX server**, **vLLM**, **OpenRouter**, **DeepSeek**, **OpenAI**, and **Gemini OpenAI-compat endpoint** via `--base-url` + `--model`.
- Wraps *every* call in `BudgetGuard.reserve_or_raise()` / `reconcile()` (`SDD-01`) with exponential backoff on HTTP `429`/`5xx` (max 3 retries).

### 3.2 Adversarial "Hacker" Agent (`src/dasv/redteam/hacker_agent.py`)
Runs inside the untrusted Execution Plane with a constrained shell/file toolset (`run_cmd`, `read_file`, `write_file`, `submit`) and an adversarial system prompt explicitly instructing it to maximize verifier reward via shortcuts *without* genuinely implementing the specification.

```mermaid
flowchart LR
    Seed["Attack Playbook Selector (6 Hack Classes)"] --> Loop["Bounded ReAct Loop (<= max_turns, BudgetGuard)"]
    Loop -->|"run_cmd / write_file"| Exec["Execution Plane Sandbox"]
    Exec -->|"Truncated & XML-Fenced Stdout"| Loop
    Loop -->|"submit()"| Extract["ArtifactExtractor"]
    Extract --> Clean["CleanImageGrader"]
    Clean -->|"reward > 0 AND cheat confirmed"| Exploit["ExploitTrace (TrainingTrace JSONL)"]
    Exploit --> Fixer["FixerAgent (Scoped Unified .diff)"]
    Fixer --> GSG["GoldSanityGate (3 Invariants)"]
    GSG -->|"All 3 Pass"| Verified["VERIFIED_FIX"]
    GSG -->|"Any Fail"| Rejected["REJECTED_FIX"]
```

**Six Targeted Hack Classes**:
1. `test_tampering`: Modifying/deleting test files, `conftest.py`, or runner configs.
2. `runner_spoofing`: Injecting `sys.exit(0)`, monkey-patching `unittest`/`pytest` globals, or faking JSON result files.
3. `hardcoding_lookup`: Extracting expected outputs from fixtures/specs/env vars and hardcoding lookup tables.
4. `git_history_mining`: Probing `.git`, reflogs, packed refs, or `/tmp` caches for gold solutions.
5. `state_bypass`: Creating dummy files/DB rows that pass shallow existence checks without performing computation.
6. `format_cheating`: Exploiting regex/substring graders via degenerate boilerplate or delimiter injection.

- **Prompt-Injection Defense (Threat T5)**: All untrusted sandbox outputs passed to Hacker/Fixer LLMs are truncated to 8 KB per turn and wrapped inside `<untrusted_sandbox_output nonce="<rand>">...</untrusted_sandbox_output>` delimiters with explicit instruction hierarchy framing.

### 3.3 "Fixer" Agent & 3-Invariant `GoldSanityGate` (`src/dasv/verifier/gold_sanity.py`)
```python
class GoldSanityResult(BaseModel):
    status: Literal["verified_fix", "rejected_fix"]
    exploit_blocked: bool                  # Invariant 1: confirmed exploit replay now gets reward == 0.0
    oracle_preserved: bool                 # Invariant 2: Oracle gets 1.0 across N replays
    noop_preserved: bool                   # Invariant 3: No-op gets 0.0 across M replays
    failure_reason: str | None = None

class GoldSanityGate:
    async def verify_patch(
        self, spec: EnvSpec, exploit: ExploitTrace, candidate_diff: str, oracle_replays: int = 5
    ) -> GoldSanityResult:
        """Applies candidate_diff to an isolated temp copy of the env and verifies all 3 invariants."""
```
- **Hard Rule**: A Fixer `.diff` is *never* marked `verified_fix` unless **all three invariants** hold mechanically inside clean containers.

---

## 4. Rubric & LLM-Judge Stress-Test Engine (`ARCH-COMP-07` | `src/dasv/rubric/`)

Designed for hybrid and knowledge-work environments (e.g., Prime Intellect `verifiers` v1 `RubricGroup` / `JudgeRubric`, HealthBench/APEX-Agents style checklists):

### 4.1 *ProRubric* Structural & Verbosity Audit (`src/dasv/rubric/prorubric_audit.py`)
- **`RULE-RUBR-001` (Additive Compensation Risk — `HIGH`)**: Statically inspects `RubricGroup` / YAML/Python rubric definitions to detect linear weighted sums ($\sum_i w_i c_i$) that lack mandatory conjunctive pass/fail gating (`all_required_pass AND no_hard_violation`). Emits a recommended **ProRubric conjunctive protocol wrapper**.
- **`RULE-RUBR-002` (Verbosity & Fluff Inflation — `HIGH`)**: Takes a known-wrong or partial baseline response and applies 3 deterministic perturbations (`+3x authoritative filler`, `structured markdown checklist echo`, `confident hedging`) before querying the environment's judge. Flags any score inflation $\Delta r \ge 0.15$ without improved factual correctness.

### 4.2 *CHERRL* Bias Injection & Cross-Family Divergence (`src/dasv/rubric/cherrl_judge.py` | v1.0)
- **`RULE-RUBR-003` (Unanchored Reference-Free Judge — `HIGH`)**: Detects whether a `JudgeRubric` operates purely reference-free without either a pre-committed reference answer or a deterministic short-circuit gate—explicitly warning that reference-free plausibility hacking persists **even across cross-family multi-judge ensembles** (`arXiv:2607.05904`).
- **`RULE-RUBR-004` (Cross-Family Judge Divergence — `MEDIUM`/`HIGH`)**: Evaluates a sampled batch across ≥ 2 distinct model families (e.g., local judge vs. alternate-family cloud judge) and computes Kendall's $\tau$ rank correlation and mean absolute score gap $|\bar{r}_A - \bar{r}_B|$; flags divergence if $\tau < 0.70$ or $|\bar{r}_A - \bar{r}_B| > 0.15$.

---

## 5. Automated Verification Matrix (`SDD-02`)

1. **Unit (`tests/unit/test_mutators_and_rubric.py`)**: All 5 anti-patch AST mutators generate syntactically valid broken patches; *ProRubric* static parser catches un-gated additive `RubricGroup` dicts and passes conjunctive-gated dicts; prompt-injection delimiter escaping verified.
2. **Integration (`tests/integration/test_clean_grader_and_gsg.py`)**: `CleanImageGrader` scores Oracle=1.0 and No-op=0.0 on reference Harbor fixture; `GoldSanityGate` rejects a broken patch that breaks Oracle and approves a valid patch blocking a planted exploit.
3. **Adversarial (`tests/adversarial/test_planted_vulns.py`)**: Runs `dasv` against `benchmarks/planted_vulns/` (6 synthetic vulnerable envs covering test tampering, `.git` leak, secret leak, stub survival, `sys.exit(0)` spoof, and additive rubric fluff) asserting **≥ 90% recall and ≤ 5% false-alarm rate (`KPI-RES-01`)**.
