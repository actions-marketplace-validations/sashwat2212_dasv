# Planted Vulnerability Benchmark Corpus (`KPI-RES-01`)

This corpus houses **7 synthetic vulnerable RL environments** spanning 7 distinct vulnerability classes alongside **3 hardened control environments** to continuously verify **`KPI-RES-01`**:
- **Detection Recall**: **>= 90%** across >= 6 vulnerability classes
- **False-Alarm Rate**: **<= 5%** on hardened control environments

## Corpus Layout (`corpus.json`)

### Vulnerable Cases (`vulnerable/`)
| ID | Vulnerability Class | Expected Rules | Audit Mode |
|---|---|---|---|
| `01_test_tampering_copy` | Test suite copied into agent image | `RULE-BNDY-001` | Tier 0 Static |
| `02_network_and_secret_leak` | Bridge network + hardcoded API keys | `RULE-BNDY-002`, `RULE-BNDY-003` | Tier 0 Static |
| `03_git_history_leak` | Unstripped `.git` history / reflog leaking `solution/solve.sh` | `RULE-BNDY-004` | Tier 0 Static (git materialized into temp dir) |
| `04_stub_return_survival` | Weak verifier passing return-constant / exception-swallowing stubs | `RULE-MUT-002` | Tier 1 Deterministic Anti-Patches |
| `05_exit_zero_runner_spoof` | Unisolated in-process runner fooled by `sys.exit(0)` | `RULE-MUT-002` | Tier 1 Deterministic Anti-Patches |
| `06_additive_rubric_fluff` | Linear additive `RubricGroup` + verbosity inflation loophole | `RULE-RUBR-001`, `RULE-RUBR-002` | Tier 2 ProRubric Static + Perturbations |
| `07_non_deterministic_boot_install` | Floating base image tag + live `pip install` at container boot | `RULE-HERM-001`, `RULE-HERM-002` | Tier 0 Static |

### Hardened Control Counterparts (`hardened/`)
| ID | Control Scope | Expected Findings |
|---|---|---|
| `01_hardened_calculator` | Pinned `@sha256`, `no-network`, strong semantic tests rejecting all 5 anti-patches | `0` |
| `02_hardened_parser` | Pinned `@sha256`, `no-network`, subprocess-isolated edge-case verifier | `0` |
| `03_hardened_prorubric_v1` | `verifiers` v1 `Taskset` with ProRubric conjunctive gating & fact-anchored scorer | `0` |

## Verification

Run the automated adversarial recall and false-alarm gate:

```bash
uv run pytest tests/adversarial/test_planted_vulns.py -v
```
