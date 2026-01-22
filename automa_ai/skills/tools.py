from __future__ import annotations

from langchain.tools import tool

from automa_ai.skills.manager import SkillManager


def build_load_skill_tool(skill_manager: SkillManager):
    @tool("load_skill")
    def load_skill(skill_name: str) -> str:
        """Load a skill prompt by name from the configured skill registry."""
        return skill_manager.load(skill_name)

    return load_skill
