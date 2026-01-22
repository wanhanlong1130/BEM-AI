from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from automa_ai.skills.config import SkillsConfig, SkillRegistryEntry

logger = logging.getLogger(__name__)

_ALLOWED_SKILL_NAME = re.compile(r"^[a-zA-Z0-9_.-]+$")
_SUPPORTED_EXTENSIONS = (".md", ".txt")


@dataclass
class SkillCacheEntry:
    mtime: float
    prompt: str


class SkillManager:
    def __init__(self, config: SkillsConfig):
        self.config = config
        self.enabled = bool(config.enabled)
        self._allowed_roots = [Path(root).resolve() for root in (config.allowed_roots or [])]
        self._registry = config.registry
        self._cache: dict[tuple[str, Path], SkillCacheEntry] = {}

    @classmethod
    def from_config(cls, config: SkillsConfig | dict | None) -> "SkillManager":
        if isinstance(config, SkillsConfig):
            return cls(config)
        return cls(SkillsConfig.from_dict(config))

    def available_skills(self) -> list[str]:
        skills: set[str] = set()
        for name, entry in self._registry.items():
            path = Path(entry.path)
            if self._is_directory_entry(entry, path):
                skills.update(self._list_directory_skills(path))
            else:
                skills.add(name)
        return sorted(skills)

    def load(self, skill_name: str) -> str:
        if not self.enabled:
            return "Skills are not enabled for this agent."
        if not _ALLOWED_SKILL_NAME.match(skill_name):
            return self._format_error(
                f"Invalid skill name '{skill_name}'. Skill names may only include letters, numbers, underscores, dashes, and periods."
            )

        resolution = self._resolve_skill_path(skill_name)
        if resolution.error:
            return self._format_error(resolution.error)
        path = resolution.path
        entry = resolution.entry

        if not path.exists() or not path.is_file():
            return self._format_error(
                f"Skill '{skill_name}' could not be loaded because the file was not found: {path}"
            )

        if not self._is_allowed_path(path):
            return self._format_error(
                f"Skill '{skill_name}' is outside the allowed roots and cannot be loaded: {path}"
            )

        try:
            mtime = path.stat().st_mtime
        except OSError as exc:
            logger.warning("Failed to stat skill file %s: %s", path, exc)
            return self._format_error(
                f"Skill '{skill_name}' could not be read (unable to stat file): {path}"
            )

        cache_key = (skill_name, path)
        cached = self._cache.get(cache_key)
        if cached and cached.mtime == mtime:
            return cached.prompt

        parse_result = self._parse_skill_file(path, skill_name, entry)
        if parse_result.error:
            return self._format_error(parse_result.error)

        wrapped = self._wrap_prompt(skill_name, path, parse_result.body, entry)
        self._cache[cache_key] = SkillCacheEntry(mtime=mtime, prompt=wrapped)
        return wrapped

    def _is_allowed_path(self, path: Path) -> bool:
        if not self._allowed_roots:
            return True
        resolved = path.resolve()
        return any(root in resolved.parents or resolved == root for root in self._allowed_roots)

    def _resolve_skill_path(self, skill_name: str) -> "SkillResolution":
        if skill_name in self._registry:
            entry = self._registry[skill_name]
            path = Path(entry.path)
            if self._is_directory_entry(entry, path):
                return self._resolve_from_directory(entry, path, skill_name)
            return SkillResolution(path=path.resolve(), entry=entry)

        directory_matches: list[SkillResolution] = []
        for entry in self._directory_entries():
            path = Path(entry.path)
            resolution = self._resolve_from_directory(entry, path, skill_name)
            if not resolution.error:
                directory_matches.append(resolution)

        if not directory_matches:
            return SkillResolution(error=f"Skill '{skill_name}' is not registered.")
        if len(directory_matches) > 1:
            paths = ", ".join(str(match.path) for match in directory_matches)
            return SkillResolution(
                error=(
                    f"Skill '{skill_name}' is ambiguous across multiple directories. Matches found at: {paths}"
                )
            )
        return directory_matches[0]

    def _directory_entries(self) -> Iterable[SkillRegistryEntry]:
        for entry in self._registry.values():
            path = Path(entry.path)
            if self._is_directory_entry(entry, path):
                yield entry

    def _resolve_from_directory(
        self,
        entry: SkillRegistryEntry,
        directory: Path,
        skill_name: str,
    ) -> "SkillResolution":
        if not directory.exists() or not directory.is_dir():
            return SkillResolution(
                error=f"Directory registry entry is not a valid directory: {directory}"
            )

        for extension in _SUPPORTED_EXTENSIONS:
            candidate = (directory / f"{skill_name}{extension}").resolve()
            if candidate.exists() and candidate.is_file():
                return SkillResolution(path=candidate, entry=entry)

        return SkillResolution(
            error=f"Skill '{skill_name}' was not found in directory: {directory}"
        )

    def _is_directory_entry(self, entry: SkillRegistryEntry, path: Path) -> bool:
        if entry.mode:
            return entry.mode.lower() == "directory"
        return path.exists() and path.is_dir()

    def _list_directory_skills(self, directory: Path) -> set[str]:
        if not directory.exists() or not directory.is_dir():
            return set()
        skills = set()
        for child in directory.iterdir():
            if child.suffix.lower() in _SUPPORTED_EXTENSIONS:
                skills.add(child.stem)
        return skills

    def _parse_skill_file(
        self,
        path: Path,
        skill_name: str,
        entry: SkillRegistryEntry,
    ) -> "SkillParseResult":
        try:
            raw = path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            logger.warning("Failed to read skill file %s: %s", path, exc)
            return SkillParseResult(error=f"Skill '{skill_name}' could not be read: {path}")

        normalized = _normalize_newlines(raw)
        format_name = self._resolve_format(path, entry)
        if format_name == "markdown":
            body = _strip_front_matter(normalized)
        elif format_name == "text":
            body = normalized
        else:
            return SkillParseResult(
                error=(
                    f"Skill '{skill_name}' has unsupported format '{format_name}'."
                    " Supported formats are 'markdown' and 'text'."
                )
            )

        body = body.lstrip("\n")
        return SkillParseResult(body=body)

    def _resolve_format(self, path: Path, entry: SkillRegistryEntry) -> str:
        if entry.format:
            return entry.format.lower()
        extension = path.suffix.lower()
        if extension == ".md":
            return "markdown"
        if extension == ".txt":
            return "text"
        return "unknown"

    def _wrap_prompt(
        self,
        skill_name: str,
        path: Path,
        body: str,
        entry: SkillRegistryEntry,
    ) -> str:
        if entry.template:
            try:
                return entry.template.format(
                    skill_name=skill_name,
                    source=str(path),
                    body=body,
                )
            except KeyError as exc:
                return self._format_error(
                    f"Skill '{skill_name}' template is missing key: {exc}"
                )

        return (
            f"SKILL: {skill_name}\n"
            f"SOURCE: {path}\n"
            "---\n"
            f"{body}\n"
            "---\n"
            "USAGE: Follow this skill precisely when relevant."
        )

    def _format_error(self, message: str) -> str:
        available = self.available_skills()
        hint = "Configure skills in AgentFactory with a skills registry." 
        if available:
            preview = ", ".join(available[:10])
            if len(available) > 10:
                preview += ", ..."
            return f"{message} Available skills: {preview}. {hint}"
        return f"{message} {hint}"


@dataclass
class SkillResolution:
    path: Path | None = None
    entry: SkillRegistryEntry | None = None
    error: str | None = None


@dataclass
class SkillParseResult:
    body: str | None = None
    error: str | None = None


def _normalize_newlines(text: str) -> str:
    text = text.replace("\r\n", "\n")
    return text.replace("\r", "\n")


def _strip_front_matter(text: str) -> str:
    if not text.startswith("---"):
        return text
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return text
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            return "\n".join(lines[idx + 1 :])
    return text
