"""Deterministic exit codes and domain exceptions for `dasv`."""

from __future__ import annotations

EXIT_PASS: int = 0
EXIT_AUDIT_FAIL: int = 1
EXIT_CONFIG_ERROR: int = 2
EXIT_BUDGET_EXCEEDED: int = 3
EXIT_RUNTIME_ERROR: int = 4
EXIT_NATIVE_ERROR: int = 5


class DasvError(Exception):
    """Base exception carrying a deterministic CLI exit code."""

    exit_code: int = EXIT_RUNTIME_ERROR

    def __init__(self, message: str, *, exit_code: int | None = None) -> None:
        super().__init__(message)
        if exit_code is not None:
            self.exit_code = exit_code


class AuditFailedError(DasvError):
    """Raised when an audit run fails certification thresholds."""

    exit_code: int = EXIT_AUDIT_FAIL


class ConfigError(DasvError):
    """Raised on invalid CLI arguments, malformed dasv.toml, or unreadable env layout."""

    exit_code: int = EXIT_CONFIG_ERROR


class BudgetExceededError(DasvError):
    """Raised when --max-cost-usd or --max-tokens circuit breaker trips."""

    exit_code: int = EXIT_BUDGET_EXCEEDED


class RuntimeSandboxError(DasvError):
    """Raised on OCI/Docker runtime failures or unrecoverable sandbox errors."""

    exit_code: int = EXIT_RUNTIME_ERROR


class SandboxSecurityError(RuntimeSandboxError):
    """Raised when a forbidden host mount or unsafe sandbox option is attempted."""

    exit_code: int = EXIT_RUNTIME_ERROR


class NativeDelegateError(DasvError):
    """Raised when --require-native is set and the native validator binary fails or is absent."""

    exit_code: int = EXIT_NATIVE_ERROR
