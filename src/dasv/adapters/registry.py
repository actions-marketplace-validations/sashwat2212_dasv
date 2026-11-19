"""Dynamic `EnvironmentAdapter` registry with deterministic priority & entry_points (SDD-03)."""

from __future__ import annotations

from collections.abc import Iterable
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import Any, cast

from dasv.adapters.base import EnvironmentAdapter
from dasv.adapters.harbor import HarborAdapter
from dasv.adapters.hud import HudAdapter
from dasv.adapters.openenv import OpenEnvAdapter
from dasv.adapters.verifiers_v1 import VerifiersV1Adapter
from dasv.core.errors import ConfigError

ENTRY_POINT_GROUP = "dasv.adapters"


class AdapterRegistry:
    """Resolves environment directories to a canonical EnvironmentAdapter."""

    def __init__(self, *, load_entry_points: bool = True) -> None:
        self._adapters: dict[str, EnvironmentAdapter] = {}
        # Deterministic built-in resolution order: harbor -> openenv -> verifiers_v1 -> hud
        self.register("harbor", HarborAdapter())
        self.register("openenv", OpenEnvAdapter())
        self.register("verifiers_v1", VerifiersV1Adapter())
        self.register("hud", HudAdapter())

        if load_entry_points:
            self._discover_entry_points(entry_points(group=ENTRY_POINT_GROUP))

    def _discover_entry_points(self, eps: Iterable[EntryPoint]) -> None:
        for ep in eps:
            if ep.name in self._adapters:
                continue
            loaded: Any = ep.load()
            instance: Any = loaded() if callable(loaded) else loaded
            self._adapters[ep.name] = cast(EnvironmentAdapter, instance)

    def register(self, name: str, adapter: EnvironmentAdapter) -> None:
        self._adapters[name] = adapter

    def list_names(self) -> list[str]:
        return list(self._adapters.keys())

    def get(self, name: str) -> EnvironmentAdapter:
        normalized = name.strip().lower()
        if normalized in self._adapters:
            return self._adapters[normalized]
        for adapter in self._adapters.values():
            if adapter.standard.value == normalized:
                return adapter
        raise ConfigError(
            f"Unknown environment format '{name}'. Available adapters: {', '.join(self._adapters)}."
        )

    def resolve(self, path: Path, forced_format: str | None = None) -> EnvironmentAdapter:
        if not path.exists():
            raise ConfigError(f"Environment path does not exist: {path}")

        if forced_format is not None:
            return self.get(forced_format)

        matches: list[tuple[str, EnvironmentAdapter]] = [
            (name, adapter) for name, adapter in self._adapters.items() if adapter.detect(path)
        ]
        if not matches:
            raise ConfigError(
                f"Could not auto-detect environment standard at '{path}'. "
                f"Checked adapters: {', '.join(self._adapters)}. "
                "Pass --format <name> explicitly if using a custom layout."
            )
        if len(matches) > 1:
            names = ", ".join(name for name, _ in matches)
            raise ConfigError(
                f"Ambiguous environment at '{path}': multiple adapters matched ({names}). "
                "Specify --format explicitly."
            )
        return matches[0][1]
