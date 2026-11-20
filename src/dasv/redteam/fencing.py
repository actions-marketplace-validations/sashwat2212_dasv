"""Prompt-injection containment and output fencing for untrusted sandbox data (Threat T5)."""

from __future__ import annotations

import re
import secrets

DEFAULT_MAX_OBS_BYTES: int = 8192


def truncate_utf8_bytes(text: str, max_bytes: int = DEFAULT_MAX_OBS_BYTES) -> str:
    """Truncate `text` to at most `max_bytes` UTF-8 bytes without splitting code points."""
    if max_bytes < 1:
        return ""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    omitted = len(encoded) - max_bytes
    clipped = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return f"{clipped}\n[... truncated {omitted} bytes ...]"


def fence_untrusted_output(
    text: str,
    *,
    label: str = "untrusted_sandbox_output",
    max_bytes: int = DEFAULT_MAX_OBS_BYTES,
    nonce: str | None = None,
) -> tuple[str, str]:
    """Wrap untrusted sandbox stdout/files inside an XML delimiter carrying a random nonce.

    Any closing/opening delimiter tags matching `label` inside the untrusted payload are
    neutralized so prompt-injected repo files cannot break out of the data fence.
    Returns `(fenced_text, nonce)`.
    """
    safe_label = re.sub(r"[^a-zA-Z0-9_-]", "_", label) or "untrusted_sandbox_output"
    token = nonce if (nonce and re.fullmatch(r"[0-9a-fA-F]{8,32}", nonce)) else secrets.token_hex(8)
    clipped = truncate_utf8_bytes(text, max_bytes=max_bytes)

    # Neutralize any forged tags referencing the fence label (with or without nonce attributes).
    tag_pattern = re.compile(rf"</?\s*{re.escape(safe_label)}\b[^>]*>", re.IGNORECASE)
    sanitized = tag_pattern.sub("[FENCE_TAG_NEUTRALIZED]", clipped)
    # Also neutralize bare nonce echoes so the closing tag cannot be spoofed verbatim.
    sanitized = sanitized.replace(token, "[NONCE_REDACTED]")

    wrapped = f'<{safe_label} nonce="{token}">\n{sanitized}\n</{safe_label} nonce="{token}">'
    return wrapped, token
