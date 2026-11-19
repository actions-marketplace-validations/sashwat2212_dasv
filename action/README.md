# `dasv-action` — Drop-In GitHub Action (`FEAT-22`)

Scheduled for **Stage 6 (M2)**.

This directory will house `action.yml`, a composite GitHub Action running natively on `ubuntu-latest` (`linux/amd64`) to execute `dasv audit --tiers 0,1 --sarif dasv.sarif --html dasv.html` and upload inline PR Code Scanning annotations.

See [docs/04-DESIGN-TELEMETRY-EVAL.md](../docs/04-DESIGN-TELEMETRY-EVAL.md).
