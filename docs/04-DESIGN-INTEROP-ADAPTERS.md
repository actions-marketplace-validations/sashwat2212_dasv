# Stage 4 / SDD-03: Interop Adapters & Native Validator Delegation — `dasv`

> [!NOTE]
> **Subsystem**: `SDD-03` (`ARCH-COMP-03` Interop Adapter Registry & Native Delegator)
> **Linked Features**: `FEAT-01`, `FEAT-02`, `FEAT-03`, `FEAT-04`, `FEAT-05`
> **Upstream Docs**: [02-PRD-FEATURES.md](./02-PRD-FEATURES.md) | [03-ARCHITECTURE.md](./03-ARCHITECTURE.md)
> **Package Modules**: `src/dasv/adapters/base.py`, `src/dasv/adapters/registry.py`, `src/dasv/adapters/harbor.py`, `src/dasv/adapters/openenv.py`, `src/dasv/adapters/verifiers_v1.py`, `src/dasv/adapters/hud.py`, `src/dasv/adapters/native.py`

---

## 1. Adapter Protocol & Plugin Discovery (`src/dasv/adapters/base.py`, `registry.py`)

To insulate `dasv` core logic from rapidly evolving upstream RL environment frameworks, every ecosystem spec implements a single narrow `EnvironmentAdapter` protocol discovered dynamically via Python entry points:

```python
class AdapterCapabilities(BaseModel):
    supports_clean_image_grading: bool = True
    supports_interactive_reset: bool = False
    supports_rubric_introspection: bool = False
    native_cli_binary: str | None = None

class EnvironmentAdapter(Protocol):
    standard: EnvStandard
    capabilities: AdapterCapabilities

    def detect(self, path: Path) -> bool:
        """Fast (< 10 ms) deterministic detection; never executes untrusted code."""

    def load(self, path: Path) -> EnvSpec:
        """Parses manifest/config into canonical EnvSpec with zero side effects."""

    async def run_native_check(self, spec: EnvSpec, timeout_sec: float = 60.0) -> list[AuditFinding]:
        """Delegates to native CLI if installed; returns normalized findings."""

    async def build_execution_config(self, spec: EnvSpec, cfg: SandboxConfig) -> ContainerConfig:
        """Builds Execution Plane ContainerConfig enforcing dasv security defaults."""

    async def run_oracle(self, spec: EnvSpec, provider: SandboxProvider, grader: CleanImageGrader, run_id: str) -> GradeOutcome: ...
    async def run_noop(self, spec: EnvSpec, provider: SandboxProvider, grader: CleanImageGrader, run_id: str) -> GradeOutcome: ...
```

### Dynamic Entry-Point Registry (`pyproject.toml`)
```toml
[project.entry-points."dasv.adapters"]
harbor = "dasv.adapters.harbor:HarborAdapter"
openenv = "dasv.adapters.openenv:OpenEnvAdapter"
verifiers_v1 = "dasv.adapters.verifiers_v1:VerifiersV1Adapter"
hud = "dasv.adapters.hud:HudAdapter"
```
- **Resolution Order**: When `dasv audit <path>` runs without `--format`, `AdapterRegistry.resolve(path)` evaluates `detect(path)` across built-in adapters in deterministic priority (`harbor` → `openenv` → `verifiers_v1` → `hud` → third-party entry points). If > 1 adapter matches, raises `ConfigError` (`EXIT_CONFIG_ERROR = 2`) prompting explicit `--format <name>`.

---

## 2. Standard-by-Standard Schema Mapping

### 2.1 `HarborAdapter` (`src/dasv/adapters/harbor.py` — v0.1 P0)
- **Detection**: Presence of `task.toml` + `instruction.md` (and `environment/Dockerfile` or `[environment]` block).
- **Canonical Field Mapping**:
  - `task_id` ← `[metadata].name` or directory basename
  - `instruction_text` ← contents of `instruction.md`
  - `has_oracle` / `oracle_entrypoint` ← `solution/solve.sh` (or `[solution].entrypoint`)
  - `verifier_paths` ← `["tests/", "tests/test.sh"]` (plus any path configured under `[verifier]`)
  - `metadata["network_mode"]` ← `[environment].network_mode` (audited against `RULE-BNDY-002` — expects `"no-network"`)
- **Native Delegation**: Executes `harbor check <task-dir>` *(explicitly using the modern `harbor check` subcommand; never the deprecated `harbor tasks check`)*.
- **Oracle & No-op Execution**:
  - *Oracle*: Boots Execution Plane container, runs `solution/solve.sh`, extracts patch/state diff via `ArtifactExtractor`, grades via `CleanImageGrader` mounting host `tests/` `:ro`.
  - *No-op*: Boots Execution Plane container, applies zero actions, extracts empty diff, grades via `CleanImageGrader`.

### 2.2 `OpenEnvAdapter` (`src/dasv/adapters/openenv.py` — v0.1 P0)
- **Detection**: Presence of `openenv.yaml` / `openenv.toml` or `pyproject.toml` declaring `openenv` entrypoint / HTTP Gymnasium server contract.
- **Canonical Field Mapping**:
  - `image_ref` ← declared OCI image or build context Dockerfile
  - `capabilities.supports_interactive_reset = True` (enables live HTTP/WS `POST /reset` → `POST /step` → `GET /state` episode lifecycle & `RULE-HERM-003` cross-episode reset residue probing)
  - Supports ingesting OpenEnv **`TrainingTrace`** JSONL artifacts (token IDs, sampled logprobs, per-token loss masks, rewards) directly into Tier 3 signal profiling & `SDD-05` exporters.
- **Native Delegation**: Executes `openenv validate --level static --skip-build <path>` (delegating static schema linting cleanly).
- **Sandbox Network Note**: Because OpenEnv communicates with the host controller over container loopback/HTTP, `OpenEnvAdapter` binds the container HTTP server to an isolated host-only Unix socket or `127.0.0.1` ephemeral port while still verifying outbound internet egress is blocked (`RULE-BNDY-006`).

### 2.3 `VerifiersV1Adapter` (`src/dasv/adapters/verifiers_v1.py` — v0.1 P0)
- **Detection**: `pyproject.toml` with `verifiers` dependency and a Python module exporting `load_environment()` / `vf.Taskset` (`verifiers` v1 contract).
- **Legacy v0 Deprecation Guard**: Statically inspects AST imports/signatures; if legacy deprecated v0 environment patterns are detected without v1 `Taskset`, emits `AuditFinding(rule_id="RULE-ADPT-001", severity="HIGH")` advising upgrade to `verifiers` v1.
- **Rubric Introspection (Safe Static-First)**:
  - Parses rubric definitions (`Rubric`, `JudgeRubric`, `RubricGroup`) via Python `ast` first (zero untrusted code execution during `inspect` / Tier 0) to detect:
    - Programmatic check functions vs. LLM-judge functions
    - Presence/absence of conjunctive gating (`RULE-RUBR-001`)
    - Presence/absence of reference answers or deterministic short-circuit gates (`RULE-RUBR-003`)
  - Marks `EnvSpec.rubric_type` as `"deterministic"`, `"rubric_llm"`, or `"hybrid"`.
- **Native Delegation**: Executes `uv run vf-validate <taskset> -n 1` inside an isolated subprocess with scrubbed env vars and timeout cap.

### 2.4 `HudAdapter` (`src/dasv/adapters/hud.py` — v1.0 Plugin)
- **Detection**: Presence of HUD project layout (`hud.toml` / `tasks.grade` / MCP server manifest targeting `hud-python`).
- **Mapping**: Normalizes MCP tool schemas, container build context, and `tasks.grade` verifier references into `EnvSpec`, allowing `dasv` Certification Reports to attach directly to **HUD DataVendor** submissions.

---

## 3. `NativeDelegateRunner` & Graceful Degradation (`src/dasv/adapters/native.py`)

```python
class NativeDelegateResult(BaseModel):
    binary: str
    found: bool
    exit_code: int | None = None
    duration_ms: float = 0.0
    findings: list[AuditFinding] = Field(default_factory=list)

class NativeDelegateRunner:
    async def run(self, binary: str, args: list[str], cwd: Path, timeout_sec: float, require_native: bool) -> NativeDelegateResult:
        """
        1. Checks shutil.which(binary).
        2. If missing and require_native=True -> raises NativeDelegateError (EXIT_NATIVE_ERROR = 5).
        3. If missing and require_native=False -> emits INFO advisory finding and lets dasv continue.
        4. Runs subprocess with scrubbed environment & hard timeout; maps non-zero exit into HIGH AuditFinding(source="native_delegate").
        """
```

### Stage 5 Pre-Code Schema Pinning Gate
Before coding `harbor.py`, `openenv.py`, and `verifiers_v1.py` in Stage 5/6, we will install the exact upstream packages inside the `dasv` dev virtualenv (or pin optional extras `[adapters]`) and capture snapshot fixture tests against their real Pydantic/TOML schemas so zero guessed field names enter production code.

---

## 4. Automated Verification Matrix (`SDD-03`)

1. **Unit (`tests/unit/test_adapters.py`)**: Auto-detection accuracy across minimal Harbor, OpenEnv, `verifiers` v1, legacy v0 (triggers `RULE-ADPT-001`), and ambiguous multi-manifest fixtures (`EXIT_CONFIG_ERROR = 2`); `entry_points` discovery test with a mock third-party adapter; `NativeDelegateRunner` behavior when binary is present (exit 0 / exit 1 / timeout) vs. absent (`require_native=False` vs. `True`).
2. **Integration (`tests/integration/test_harbor_openenv_e2e.py`)**: End-to-end `dasv inspect` + Tier 0/1 run on synthetic minimal Harbor and OpenEnv Docker fixtures, confirming clean-image Oracle=1.0 and No-op=0.0.
