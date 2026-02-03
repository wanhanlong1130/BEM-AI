import pytest

from automa_ai.skills.manager import SkillManager

pytest.importorskip("langchain")

from automa_ai.skills.tools import build_load_skill_tool


def _invoke_tool(tool, skill_name: str):
    if hasattr(tool, "invoke"):
        return tool.invoke({"skill_name": skill_name})
    if hasattr(tool, "run"):
        return tool.run(skill_name)
    return tool(skill_name)


def test_build_load_skill_tool_wraps_manager(tmp_path):
    skill_path = tmp_path / "foo.md"
    skill_path.write_text("Hello", encoding="utf-8")
    manager = SkillManager.from_config(
        {"enabled": True, "registry": {"foo": {"path": str(skill_path)}}}
    )

    tool = build_load_skill_tool(manager)

    assert tool.name == "load_skill"
    result = _invoke_tool(tool, "foo")
    assert "Hello" in result
