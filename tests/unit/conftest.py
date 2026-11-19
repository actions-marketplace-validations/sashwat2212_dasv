"""Synthetic Harbor, OpenEnv, verifiers v1/v0, and HUD fixtures for unit tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


@pytest.fixture()
def hardened_harbor_env(tmp_path: Path) -> Path:
    env_dir = tmp_path / "hardened_harbor"
    (env_dir / "environment").mkdir(parents=True)
    (env_dir / "solution").mkdir()
    (env_dir / "tests").mkdir()

    (env_dir / "task.toml").write_text(
        """
[metadata]
name = "hardened-calculator"

[environment]
network_mode = "no-network"

[verifier]
test_path = "tests/test.sh"
""".lstrip(),
        encoding="utf-8",
    )
    (env_dir / "instruction.md").write_text(
        "Implement add(a, b) in calculator.py.\n", encoding="utf-8"
    )
    digest = "0123456789abcdef" * 4
    (env_dir / "environment" / "Dockerfile").write_text(
        (
            f"FROM python:3.12-slim@sha256:{digest} AS builder\n"
            "RUN pip install --no-cache-dir pytest==8.3.0\n"
            "FROM builder\n"
            "WORKDIR /app\n"
            "COPY src/ /app/src/\n"
            'CMD ["python", "-m", "app"]\n'
        ),
        encoding="utf-8",
    )
    (env_dir / "solution" / "solve.sh").write_text("#!/bin/sh\necho solved\n", encoding="utf-8")
    (env_dir / "tests" / "test.sh").write_text("#!/bin/sh\npytest\n", encoding="utf-8")
    return env_dir


@pytest.fixture()
def vulnerable_harbor_env(tmp_path: Path) -> Path:
    env_dir = tmp_path / "vulnerable_harbor"
    (env_dir / "environment").mkdir(parents=True)
    (env_dir / "solution").mkdir()
    (env_dir / "tests").mkdir()

    (env_dir / "task.toml").write_text(
        """
[metadata]
name = "leaky-task"

[environment]
network_mode = "bridge"
api_token = "hf_abcdef1234567890"
""".lstrip(),
        encoding="utf-8",
    )
    (env_dir / "instruction.md").write_text("Fix the bug.\n", encoding="utf-8")
    (env_dir / "environment" / "Dockerfile").write_text(
        """
FROM python:3.12-slim
ENV OPENAI_API_KEY=sk-proj-supersecret1234567890
COPY tests/ /app/tests/
CMD ["sh", "-c", "pip install requests && python app.py"]
""".lstrip(),
        encoding="utf-8",
    )
    (env_dir / "entrypoint.sh").write_text(
        "#!/bin/sh\napt-get install -y curl\npython app.py\n", encoding="utf-8"
    )
    (env_dir / "solution" / "solve.sh").write_text("#!/bin/sh\necho gold\n", encoding="utf-8")
    (env_dir / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    # Initialize a real .git repo containing a commit referencing solution/solve.sh
    subprocess.run(["git", "init", "-b", "main"], cwd=str(env_dir), check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=str(env_dir),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(env_dir),
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=str(env_dir), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "add gold solution"],
        cwd=str(env_dir),
        check=True,
        capture_output=True,
    )
    return env_dir


@pytest.fixture()
def minimal_openenv(tmp_path: Path) -> Path:
    env_dir = tmp_path / "openenv_grid"
    env_dir.mkdir()
    digest = "abcdef0123456789" * 4
    (env_dir / "openenv.yaml").write_text(
        (
            "name: gridworld-v1\n"
            "description: Navigate the 5x5 grid to the goal.\n"
            f"image: ghcr.io/example/gridworld@sha256:{digest}\n"
            "network_mode: none\n"
        ),
        encoding="utf-8",
    )
    (env_dir / "tests").mkdir()
    return env_dir


@pytest.fixture()
def minimal_verifiers_v1(tmp_path: Path) -> Path:
    env_dir = tmp_path / "vf_v1_env"
    env_dir.mkdir()
    (env_dir / "pyproject.toml").write_text(
        """
[project]
name = "math-rubric-v1"
version = "0.1.0"
dependencies = ["verifiers>=1.0.0"]
""".lstrip(),
        encoding="utf-8",
    )
    (env_dir / "README.md").write_text("Solve competition algebra problems.\n", encoding="utf-8")
    (env_dir / "env.py").write_text(
        """
from verifiers import JudgeRubric, Rubric, RubricGroup, Taskset


def load_environment():
    rubric = RubricGroup([Rubric(), JudgeRubric()])
    return Taskset(name="math-rubric-v1", rubric=rubric)
""".lstrip(),
        encoding="utf-8",
    )
    return env_dir


@pytest.fixture()
def legacy_verifiers_v0(tmp_path: Path) -> Path:
    env_dir = tmp_path / "vf_v0_env"
    env_dir.mkdir()
    (env_dir / "pyproject.toml").write_text(
        """
[project]
name = "legacy-v0-env"
version = "0.1.0"
dependencies = ["verifiers==0.1.5"]
""".lstrip(),
        encoding="utf-8",
    )
    (env_dir / "legacy.py").write_text(
        """
from verifiers.envs import SingleTurnEnv


class OldEnv(SingleTurnEnv):
    pass
""".lstrip(),
        encoding="utf-8",
    )
    return env_dir
