# Security Policy

## Scope & Threat Model

`dasv` audits untrusted RL post-training environments and (in Stage 6 Tier 2+) executes adversarial "Hacker-Fixer" loops inside hermetic sandboxes. All security invariants are specified in [docs/03-ARCHITECTURE.md](./docs/03-ARCHITECTURE.md) and [docs/04-DESIGN-SANDBOX-RUNTIME.md](./docs/04-DESIGN-SANDBOX-RUNTIME.md).

### Mandatory Execution Guardrails
- **Strict Trust-Plane Separation**: Untrusted agent code runs strictly inside the Execution Plane (`--network=none`, `--cap-drop=ALL`, `--security-opt=no-new-privileges:true`, CPU/RAM/PID/wall-clock caps). Grading executes strictly inside a pristine Verification Plane container with read-only (`:ro`) verifier mounts.
- **No Host Secrets in Sandboxes**: Host API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `HF_TOKEN`, `GITHUB_TOKEN`, etc.) are scrubbed from native delegate subprocesses and never injected into sandbox containers.
- **Host Mount Guardrail**: Mounting `$HOME`, `/`, `/etc`, `/var/run/docker.sock`, or `~/.ssh` into Execution Plane containers is rejected unconditionally.
- **Authorized Local Targets Only**: `dasv` is designed exclusively for auditing user-owned or open-license RL environments inside local/CI hermetic sandboxes—never live or unauthorized external systems.

## Reporting a Vulnerability

If you discover a security bug in `dasv` (e.g., sandbox escape vector, secret leakage, SARIF/HTML XSS injection, or budget circuit-breaker bypass), please open a private security advisory on GitHub rather than filing a public issue with exploit details.
