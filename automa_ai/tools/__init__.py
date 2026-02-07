"""Built-in default tools registry."""

from automa_ai.tools.registry import DEFAULT_TOOL_REGISTRY, build_langchain_tools
from automa_ai.tools.web_search import build_web_search_tool

try:
    DEFAULT_TOOL_REGISTRY.register("web_search", build_web_search_tool)
except ValueError:
    # Module reloads (e.g. in tests) may re-run registration.
    pass

__all__ = ["DEFAULT_TOOL_REGISTRY", "build_langchain_tools"]
