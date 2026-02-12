from pathlib import Path

from automa_ai.blackboard.backends.local_json import LocalJSONBlackboardStore
from automa_ai.blackboard.schema import BlackboardSchemaRegistry, BlackboardSchemaValidator
from automa_ai.blackboard.tools import build_blackboard_tools


def build_store(tmp_path: Path):
    registry = BlackboardSchemaRegistry()
    registry.register(
        "test",
        "1",
        {
            "type": "object",
            "properties": {"items": {"type": "array", "items": {"type": "string"}}},
            "required": ["items"],
        },
    )
    validator = BlackboardSchemaValidator(registry)
    store = LocalJSONBlackboardStore(str(tmp_path), validator)
    store.create("session-tools", "test", "1", {"items": []})
    return store


def test_tool_wrappers_happy_path(tmp_path: Path):
    tools = {t.name: t for t in build_blackboard_tools(build_store(tmp_path))}

    write_result = tools["blackboard_write"].func(
        session_id="session-tools",
        ops=[{"op": "append", "path": "items", "value": "a"}],
        expected_revision=1,
        actor="tester",
    )
    assert write_result["revision"] == 2

    read_result = tools["blackboard_read"].func(session_id="session-tools", path="items")
    assert read_result["data"] == ["a"]

    rev_result = tools["blackboard_get_revision"].func(session_id="session-tools")
    assert rev_result["revision"] == 2
