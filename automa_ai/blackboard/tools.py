from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from automa_ai.blackboard.models import BlackboardPatch
from automa_ai.blackboard.store import BlackboardStore, get_path_value


class BlackboardReadInput(BaseModel):
    session_id: str
    path: str | None = None


class BlackboardWriteInput(BaseModel):
    session_id: str
    ops: list[dict[str, Any]] = Field(default_factory=list)
    expected_revision: int | None = None
    actor: str | None = None
    note: str | None = None


class BlackboardRevisionInput(BaseModel):
    session_id: str


def build_blackboard_tools(store: BlackboardStore) -> list[StructuredTool]:
    def blackboard_read(session_id: str, path: str | None = None) -> dict[str, Any]:
        doc = store.load(session_id)
        data = get_path_value(doc.data, path)
        return {
            "session_id": session_id,
            "revision": doc.revision,
            "updated_at": doc.updated_at.isoformat(),
            "path": path,
            "data": data,
        }

    def blackboard_write(
        session_id: str,
        ops: list[dict[str, Any]],
        expected_revision: int | None = None,
        actor: str | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        patch = BlackboardPatch(ops=ops, actor=actor, note=note)
        doc = store.apply_patch(session_id=session_id, patch=patch, expected_revision=expected_revision)
        return {
            "session_id": session_id,
            "revision": doc.revision,
            "updated_at": doc.updated_at.isoformat(),
            "event_count": len(doc.events),
        }

    def blackboard_get_revision(session_id: str) -> dict[str, Any]:
        doc = store.load(session_id)
        return {"session_id": session_id, "revision": doc.revision, "updated_at": doc.updated_at.isoformat()}

    return [
        StructuredTool.from_function(
            name="blackboard_read",
            description="Read the session blackboard document or a specific path.",
            func=blackboard_read,
            args_schema=BlackboardReadInput,
        ),
        StructuredTool.from_function(
            name="blackboard_write",
            description="Apply deterministic write operations to the session blackboard.",
            func=blackboard_write,
            args_schema=BlackboardWriteInput,
        ),
        StructuredTool.from_function(
            name="blackboard_get_revision",
            description="Return the current revision for session blackboard.",
            func=blackboard_get_revision,
            args_schema=BlackboardRevisionInput,
        ),
    ]
