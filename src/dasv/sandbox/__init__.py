"""Sandbox runtime abstractions, hardened Docker provider, janitor, and boundary probes."""

from __future__ import annotations

from dasv.sandbox.docker_provider import (
    DockerSandboxProvider,
    build_docker_run_args,
    validate_bind_mount,
    validate_env_vars,
)
from dasv.sandbox.janitor import ContainerJanitor, ReapReport
from dasv.sandbox.probes import (
    probe_egress_and_env_scrub,
    probe_emulated_arch,
    probe_reset_residue,
    probe_writable_verifier_paths,
    run_runtime_boundary_probes,
    run_tier0_static_probes,
)
from dasv.sandbox.provider import (
    DEFAULT_MAX_OUTPUT_BYTES,
    ContainerConfig,
    ExecResult,
    RuntimeHealth,
    SandboxHandle,
    SandboxProvider,
    current_host_arch,
    normalize_arch,
    truncate_output,
)

__all__ = [
    "ContainerConfig",
    "ContainerJanitor",
    "DEFAULT_MAX_OUTPUT_BYTES",
    "DockerSandboxProvider",
    "ExecResult",
    "ReapReport",
    "RuntimeHealth",
    "SandboxHandle",
    "SandboxProvider",
    "build_docker_run_args",
    "current_host_arch",
    "normalize_arch",
    "probe_egress_and_env_scrub",
    "probe_emulated_arch",
    "probe_reset_residue",
    "probe_writable_verifier_paths",
    "run_runtime_boundary_probes",
    "run_tier0_static_probes",
    "truncate_output",
    "validate_bind_mount",
    "validate_env_vars",
]
