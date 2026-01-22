from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class SkillRegistryEntry:
    path: str
    format: str | None = None
    mode: str | None = None
    template: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SkillRegistryEntry":
        return cls(
            path=str(data.get("path", "")),
            format=data.get("format"),
            mode=data.get("mode"),
            template=data.get("template"),
        )


@dataclass(frozen=True)
class SkillsConfig:
    enabled: bool = False
    allowed_roots: list[str] | None = None
    registry: dict[str, SkillRegistryEntry] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "SkillsConfig":
        if not data:
            return cls()

        enabled = bool(data.get("enabled", False))
        allowed_roots = data.get("allowed_roots")
        registry_raw = data.get("registry", {})
        registry: dict[str, SkillRegistryEntry] = {}
        for name, entry in registry_raw.items():
            if isinstance(entry, SkillRegistryEntry):
                registry[name] = entry
                continue
            if isinstance(entry, str):
                registry[name] = SkillRegistryEntry(path=entry)
                continue
            if isinstance(entry, Mapping):
                registry[name] = SkillRegistryEntry.from_dict(entry)
                continue
            raise TypeError(f"Unsupported skill registry entry for {name}: {type(entry)!r}")

        return cls(enabled=enabled, allowed_roots=allowed_roots, registry=registry)
