# Contributing to `dasv`

Thank you for your interest in contributing to **`dasv`** (*Deterministic Adversarial Sandbox Verifier*). Contributions of all kinds are welcome—bug reports, new ecosystem adapters, audit probes, documentation improvements, and reproducible RLVR benchmarks.

## Development Setup

`dasv` uses [`uv`](https://docs.astral.sh/uv/) for deterministic Python `>= 3.11` dependency management.

```bash
git clone https://github.com/sashwat2212/dasv.git
cd dasv
uv sync --frozen
uv run dasv doctor
```

## Quality Gate (Required Before Opening a PR)

Run the exact same checks enforced in `.github/workflows/ci.yml`:

```bash
uv run ruff format --check .
uv run ruff check src tests recipes scripts
uv run pyright
uv run pytest -v
```

- **Formatting & Linting**: `ruff` with `line-length = 100`.
- **Static Typing**: Strict `pyright` across `src/` and `tests/` (`0` errors).
- **Tests**: Unit tests, `KPI-RES-01` planted vulnerability corpus tests (`tests/adversarial/`), and container integration tests (requires Docker or OrbStack).

## Adding a New Ecosystem Adapter Plugin

`dasv` discovers environment adapters via Python entry points under `dasv.adapters`:

```toml
[project.entry-points."dasv.adapters"]
my_gym = "my_package.adapter:MyGymAdapter"
```

1. Subclass `EnvironmentAdapter` from `dasv.adapters.base` (see `src/dasv/adapters/harbor.py` or `src/dasv/adapters/openenv.py` for reference).
2. Implement `detect(path)`, `load(path) -> EnvSpec`, `static_findings(spec)`, and clean-image oracle/no-op execution hooks.
3. Add unit tests under `tests/unit/test_adapters.py` covering both a hardened control environment (`0` findings) and a vulnerable fixture.

## Reporting Vulnerabilities

Please **do not** open public issues for security vulnerabilities in `dasv` itself. See [SECURITY.md](./SECURITY.md) for responsible disclosure instructions.

## Code of Conduct

All contributors and maintainers agree to follow our [Code of Conduct](./CODE_OF_CONDUCT.md).
