"""Validation tests for the drop-in Composite GitHub Action `action/action.yml` (FEAT-22)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

ACTION_YML = Path(__file__).resolve().parents[2] / "action" / "action.yml"


def test_github_action_manifest_structure_and_injection_safety() -> None:
    assert ACTION_YML.is_file()
    data = cast(dict[str, Any], yaml.safe_load(ACTION_YML.read_text(encoding="utf-8")))

    assert data["name"]
    assert data["description"]

    inputs = cast(dict[str, Any], data["inputs"])
    for req_input in (
        "path",
        "tiers",
        "replays",
        "noop-replays",
        "with-mutations",
        "fail-on",
        "out-dir",
        "sarif-file",
        "html-file",
        "upload-sarif",
        "upload-html-artifact",
    ):
        assert req_input in inputs, f"Missing action input: {req_input}"

    outputs = cast(dict[str, Any], data["outputs"])
    for req_output in ("grade", "hackability-score", "sarif-file", "html-file", "exit-code"):
        assert req_output in outputs, f"Missing action output: {req_output}"

    runs = cast(dict[str, Any], data["runs"])
    assert runs["using"] == "composite"
    steps = cast(list[dict[str, Any]], runs["steps"])
    assert len(steps) >= 4

    uses_list = [str(s.get("uses", "")) for s in steps]
    assert any(u.startswith("astral-sh/setup-uv@") for u in uses_list)
    assert any(u.startswith("github/codeql-action/upload-sarif@") for u in uses_list)
    assert any(u.startswith("actions/upload-artifact@") for u in uses_list)

    # Security guardrail: NEVER interpolate `${{ inputs.* }}` directly inside `run:` shell blocks
    for step in steps:
        run_script = step.get("run")
        if isinstance(run_script, str):
            assert "${{ inputs." not in run_script, (
                f"Unsafe direct inputs interpolation in step '{step.get('name')}': "
                "pass inputs through step env vars instead."
            )
