from typing import Any, Optional, Callable
from uuid import uuid4

from a2a.client.transports import JsonRpcTransport
from a2a.types import AgentCard, Task, Message, MessageSendParams
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from dataclasses import dataclass
from automa_ai.common.base_agent import BaseAgent
import httpx


def build_subagent_delegation_instruction(subagents) -> str:
    lines = [
        "## AGENT DELEGATION",
        "You can delegate tasks to the following agent tools when appropriate:",
        "",
    ]

    for spec in subagents:
        lines.append(f"- **{spec.name}**: {spec.description}")

    return "\n".join(lines)

def compute_final(chunks: list[str]) -> str:
    # Last numeric expression result
    for chunk in reversed(chunks):
        if chunk.strip().isdigit():
            return chunk.strip()
    return chunks[-1]

@dataclass
class SubAgentSpec:
    name: str
    description: str
    agent_card: AgentCard

@dataclass
class A2AToolResult:
    final: Optional[str]
    chunks: list[str]
    state: str
    task_id: str

@dataclass
class StreamEvent:
    source: str            # "coordinator" | "subagent:math"
    type: str              # "text" | "tool" | "subagent_chunk"
    content: str
    metadata: dict | None = None

class A2AToolAdapter:
    def __init__(self, *, subagent, emit_event: Callable[[StreamEvent], None],):
        """
            subagent: RemoteAgent
            on_chunk: optional callback for streaming text chunks
        """
        self.subagent = subagent
        self.emit_event = emit_event

    async def run(self, task: str) -> A2AToolResult:
        #TODO check if it is needed to resolve the context_id to the same as the task.
        a2a_task = await self.subagent.invoke(task, uuid4().hex)
        print(a2a_task)
        chunks: list[str] = []

        # --- collect from history ---
        for msg in a2a_task.history:
            for part in msg.parts:
                if part.root.kind == "text":
                    text = part.root.text
                    chunks.append(text)
                    self.emit_event(
                        StreamEvent(
                            source=f"subagent:{self.subagent.agent_name}",
                            type="subagent_chunk",
                            content=part.root.text.rstrip() + "\n",
                            metadata=None
                        )
                    )

        final = ""
        # --- extract final answer from artifact ---
        artifact = a2a_task.artifacts[0] if a2a_task.artifacts else None
        if artifact:
            # if DataPart, take its content
            if hasattr(artifact.parts[0].root, "data"):
                final = artifact.parts[0].root.data
            elif hasattr(artifact.parts[0].root, "text"):
                final = artifact.parts[0].root.text

        self.emit_event(
            StreamEvent(
                source=f"subagent:{self.subagent.agent_name}",
                type="subagent_chunk",
                content=final,
                metadata={"final": True},
            )
        )

        return A2AToolResult(
            final=final,
            chunks=chunks,
            state=a2a_task.status.state.value,
            task_id=a2a_task.id,
        )


class SubAgentInput(BaseModel):
    task: str = Field(description="Task description to delegate to the sub-agent")

def make_subagent_tool(
    spec: SubAgentSpec,
    emitter: Callable[[StreamEvent], None] = None,
):
    subagent = RemoteAgent(
        agent_name=spec.name,
        subagent_card=spec.agent_card,
        description=spec.description,
    )

    adapter = A2AToolAdapter(subagent=subagent, emit_event=emitter)

    async def _run(task: str) -> dict:
        result = await adapter.run(task)
        return {
            "final": result.final,
            "chunks": result.chunks,
            "state": result.state,
            "task_id": result.task_id,
        }

    return StructuredTool.from_function(
        name=spec.name,
        description=spec.description,
        coroutine=_run,
        args_schema=SubAgentInput,
    )


class RemoteAgent(BaseAgent):
    """An interface to stream connections to a hands off agent"""
    def __init__(
            self,
            agent_name: str,
            subagent_card: AgentCard,
            description: str,
    ):
        super().__init__(
            agent_name=agent_name,
            description=description,
            content_types=["text", "text/plain"],
        )

        self._client = httpx.AsyncClient(timeout=httpx.Timeout(None))
        self._transport = JsonRpcTransport(
            httpx_client=self._client,
            agent_card=subagent_card,
        )

    async def invoke(self, message:str, sessionId: str) -> Task | Message:
        payload: dict[str, Any] = {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": message}],
                "message_id": uuid4().hex
            }
        }

        return await self._transport.send_message(
            request=MessageSendParams(**payload)
        )

    async def close(self):
        await self._client.aclose()