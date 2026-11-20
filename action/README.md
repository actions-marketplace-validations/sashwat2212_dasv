# `dasv-action` — Drop-In GitHub Action (`FEAT-22`)

Composite GitHub Action running natively on `ubuntu-latest` (`linux/amd64`) to gate RL environment pull requests on Tier 0–1 boundary security, clean-image Oracle/No-op determinism, optional `$0` deterministic mutations, inline SARIF 2.1.0 Code Scanning annotations, and zero-CDN offline HTML Certification Cards.

## 10-Line Quickstart Workflow

```yaml
name: Verify RL Environment
on: [pull_request]

permissions:
  contents: read
  security-events: write

jobs:
  dasv-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: ./action   # or <org>/dasv/action@v0.2
        with:
          path: ./tasks/my-env
          tiers: "0,1"
          replays: "10"
          with-mutations: "true"
          fail-on: high
```

## Key Inputs

| Input | Default | Description |
|---|---|---|
| `path` | `.` | Path to RL environment directory (Harbor, OpenEnv, `verifiers` v1, or HUD). |
| `tiers` | `0,1` | Comma-separated audit tiers (`0`, `0,1`, etc.). |
| `replays` | `10` | Tier 1 Oracle replay count `N`. |
| `noop-replays` | `3` | Tier 1 No-op replay count `M`. |
| `with-mutations` | `false` | Enable `$0` deterministic Anti-Patch & *ImpossibleBench* mutations. |
| `fail-on` | `high` | Minimum finding severity to fail the PR gate (`critical\|high\|medium\|low`). |
| `sarif-file` | `dasv-artifacts/dasv.sarif` | Output path for SARIF 2.1.0 annotations. |
| `html-file` | `dasv-artifacts/dasv.html` | Output path for offline HTML Certification Report. |
| `upload-sarif` | `true` | Upload SARIF to GitHub Security / Code Scanning (`github/codeql-action/upload-sarif@v3`). |
| `upload-html-artifact` | `true` | Upload HTML/JSON artifacts (`actions/upload-artifact@v4`). |

## Outputs

- `grade`: Deterministic certification grade (`A`–`F`)
- `hackability-score`: Bounded `HackabilityScore` (`0.0`–`100.0`)
- `sarif-file`, `html-file`, `report-json`, `exit-code`

See [docs/04-DESIGN-TELEMETRY-EVAL.md](../docs/04-DESIGN-TELEMETRY-EVAL.md) for the full specification.
