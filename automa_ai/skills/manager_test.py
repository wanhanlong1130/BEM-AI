import os
import time

from automa_ai.skills.manager import SkillManager


def test_load_markdown_skill(tmp_path):
    skill_path = tmp_path / "write_sql.md"
    skill_path.write_text(
        "---\nname: write_sql\n---\nUse parameterized queries.\n",
        encoding="utf-8",
    )
    manager = SkillManager.from_config(
        {
            "enabled": True,
            "registry": {
                "write_sql": {
                    "path": str(skill_path),
                    "format": "markdown",
                }
            },
        }
    )

    result = manager.load("write_sql")

    assert "SKILL: write_sql" in result
    assert "Use parameterized queries." in result
    assert "name: write_sql" not in result


def test_unknown_skill_returns_error(tmp_path):
    skill_path = tmp_path / "write_sql.md"
    skill_path.write_text("Use SQL.", encoding="utf-8")
    manager = SkillManager.from_config(
        {
            "enabled": True,
            "registry": {"write_sql": {"path": str(skill_path)}},
        }
    )

    result = manager.load("missing_skill")

    assert "missing_skill" in result
    assert "Available skills" in result
    assert "write_sql" in result


def test_directory_mode_resolves_skill(tmp_path):
    skill_dir = tmp_path / "skills"
    skill_dir.mkdir()
    (skill_dir / "alpha.md").write_text("Alpha skill body", encoding="utf-8")

    manager = SkillManager.from_config(
        {
            "enabled": True,
            "registry": {
                "dir": {"path": str(skill_dir), "mode": "directory"}
            },
        }
    )

    result = manager.load("alpha")

    assert "Alpha skill body" in result


def test_path_traversal_blocked(tmp_path):
    skill_path = tmp_path / "safe.md"
    skill_path.write_text("Safe", encoding="utf-8")
    manager = SkillManager.from_config(
        {"enabled": True, "registry": {"safe": {"path": str(skill_path)}}}
    )

    result = manager.load("../secret")

    assert "Invalid skill name" in result


def test_allowed_roots_enforced(tmp_path):
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    skill_path = tmp_path / "outside.md"
    skill_path.write_text("Outside", encoding="utf-8")

    manager = SkillManager.from_config(
        {
            "enabled": True,
            "allowed_roots": [str(allowed_root)],
            "registry": {"outside": {"path": str(skill_path)}},
        }
    )

    result = manager.load("outside")

    assert "outside the allowed roots" in result


def test_cache_invalidation_on_mtime(tmp_path):
    skill_path = tmp_path / "cache.md"
    skill_path.write_text("First", encoding="utf-8")

    manager = SkillManager.from_config(
        {"enabled": True, "registry": {"cache": {"path": str(skill_path)}}}
    )

    first = manager.load("cache")
    time.sleep(1.1)
    skill_path.write_text("Second", encoding="utf-8")
    os.utime(skill_path, None)

    second = manager.load("cache")

    assert "First" in first
    assert "Second" in second
