# Planted Vulnerability Benchmark Corpus (`KPI-RES-01`)

Scheduled for **Stage 6 (M1–M2)**.

This directory will house ≥ 6 minimal synthetic RL environments with known planted verifier defects (writable test files, `.git` history leak, secret leak, trivial return-constant survival, `sys.exit(0)` runner spoof, and un-gated additive rubric fluff) plus hardened control counterparts to continuously verify **≥ 90% detection recall** and **≤ 5% false-alarm rate**.

See [docs/01-BRD.md](../../docs/01-BRD.md) and [docs/04-DESIGN-VERIFIER-ENGINE.md](../../docs/04-DESIGN-VERIFIER-ENGINE.md).
