from typing import Any, Optional, Callable, AsyncIterable, AsyncGenerator, Awaitable
from uuid import uuid4

from a2a.client.transports import JsonRpcTransport
from a2a.types import AgentCard, Task, Message, MessageSendParams, TaskStatusUpdateEvent, TaskArtifactUpdateEvent, \
    TaskState, TextPart, DataPart
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from dataclasses import dataclass
from automa_ai.common.base_agent import BaseAgent
import httpx
import re


def build_subagent_delegation_instruction(subagents) -> str:
    lines = [
        "## AGENT DELEGATION",
        "You can delegate tasks to the following agent tools when appropriate:",
        "",
    ]

    for spec in subagents:
        lines.append(f"- **{spec.tool_name}**: {spec.description}")

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

    @property
    def tool_name(self) -> str:
        """
        Tool-safe name for LLM function calling.
        """
        return re.sub(r"[^a-zA-Z0-9_]", "_", self.name).lower()

@dataclass
class A2AToolResult:
    final: Optional[str]
    chunks: list[str]
    task_id: str

@dataclass
class StreamEvent:
    source: str            # "coordinator" | "subagent:math"
    type: str              # "text" | "tool" | "subagent_chunk"
    content: str
    metadata: dict | None = None

class A2AToolAdapter:
    def __init__(self, *, subagent, emit_event: Callable[[StreamEvent], Awaitable[None]],):
        """
            subagent: RemoteAgent
            on_chunk: optional callback for streaming text chunks
        """
        self.subagent = subagent
        self.emit_event = emit_event

    async def run(self, task: str) -> A2AToolResult:
        #TODO check if it is needed to resolve the context_id to the same as the task.
        a2a_task = await self.subagent.invoke(task, uuid4().hex)
        chunks: list[str] = []

        # --- collect from history ---
        for msg in a2a_task.history:
            for part in msg.parts:
                if part.root.kind == "text":
                    text = part.root.text
                    chunks.append(text)
                    await self.emit_event(
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

        await self.emit_event(
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
            task_id=a2a_task.id,
        )

    async def stream(self, task: str) -> AsyncIterable[A2AToolResult]:
        chunks: list[str] = []
        async for chunk in self.subagent.stream(task):
            # print(f"Remote receiving chunk, type: {type(chunk)}, chunk: {chunk}")
            if isinstance(chunk, TaskStatusUpdateEvent):
                # context_id = chunk.context_id
                # If the node is completed, then move to the next node
                if chunk.status.state == TaskState.completed:
                    # Task status update provides status but no artifacts
                    continue
                if chunk.status.state == TaskState.input_required:
                    question = chunk.status.message.parts[
                        0
                    ].root.text
                    #TODO question needs to be looped back to orchestrator
                    # This status should be aligned with orchestrator status
                    await self.emit_event(
                        StreamEvent(
                            source=f"subagent:{self.subagent.agent_name}",
                            type="subagent_chunk",
                            content=question + "\n",
                            metadata=None
                        )
                    )
                    chunks.append(question)
                if chunk.status.state == TaskState.working:
                    message = chunk.status.message.parts[0].root.text
                    # print(f"emitting event: {message}")
                    await self.emit_event(
                        StreamEvent(
                            source=f"subagent:{self.subagent.agent_name}",
                            type="subagent_chunk",
                            content=message + "\n",
                            metadata=None
                        )
                    )
                    chunks.append(message)
            if isinstance(chunk, TaskArtifactUpdateEvent):
                artifact = chunk.artifact
                # self.results.append(artifact)
                if isinstance(artifact.parts[0].root, TextPart):
                    text = artifact.parts[0].root.text
                    await self.emit_event(
                        StreamEvent(
                            source=f"subagent:{self.subagent.agent_name}",
                            type="subagent_chunk",
                            content=text + "\n",
                            metadata=None
                        )
                    )
                    yield A2AToolResult(
                        final=text,
                        chunks=chunks,
                        task_id=chunk.task_id,
                    )
                if isinstance(artifact.parts[0].root, DataPart):
                    artifact_data = artifact.parts[0].root.data
                    await self.emit_event(
                        StreamEvent(
                            source=f"subagent:{self.subagent.agent_name}",
                            type="subagent_chunk",
                            content=artifact_data,
                            metadata=None
                        )
                    )
                    yield A2AToolResult(
                        final=artifact_data,
                        chunks=chunks,
                        task_id=chunk.task_id,
                    )


class SubAgentInput(BaseModel):
    task: str = Field(description="Task description to delegate to the sub-agent")

def make_subagent_tool(
    spec: SubAgentSpec,
    emitter: Callable[[StreamEvent], Awaitable[None]] = None,
):
    subagent = RemoteAgent(
        agent_name=spec.tool_name,
        subagent_card=spec.agent_card,
        description=spec.description,
    )

    adapter = A2AToolAdapter(subagent=subagent, emit_event=emitter)

    async def _run(task: str) -> dict:
        chunks: list[A2AToolResult] = []
        agent_card: AgentCard = adapter.subagent.agent_card
        if agent_card.capabilities.streaming:
            async for chunk in adapter.stream(task):
                chunks.append(chunk)
        else:
            result = await adapter.run(task)
            chunks.append(result)

        result = None
        if chunks:
            result = chunks[0]

        if result:
            return {
                "final": chunks[0].final,
                "chunks": chunks[0].chunks,
                "task_id": chunks[0].task_id,
            }
        else:
            return {
                "final": f"No result produced by the subagent {adapter.subagent.agent_name}",
                "chunks": "",
                "task_id": "",
            }

    return StructuredTool.from_function(
        name=spec.tool_name,
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
        self.agent_card = subagent_card
        self._transport = JsonRpcTransport(
            httpx_client=self._client,
            agent_card=self.agent_card,
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

    async def stream(self, message:str) -> AsyncGenerator[Message | Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent, None]:
        payload: dict[str, Any] = {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": message}],
                "message_id": uuid4().hex
            }
        }
        async for chunk in self._transport.send_message_streaming(request=MessageSendParams(**payload)):
            yield chunk

    async def close(self):
        await self._client.aclose()