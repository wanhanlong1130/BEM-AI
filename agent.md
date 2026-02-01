# BEM-AI Agent Guide

This document provides an AI-focused orientation to BEM-AI’s agent architecture, with emphasis on subagents (`remote_agent`), skills, and memory. It is meant to help an LLM understand how the runtime is wired and where to look when extending or debugging behavior.

## Repository mental model

- **Core agent runtime** lives in `automa_ai/agents` (LangGraph-based chat agent, A2A subagents, agent factory).【F:automa_ai/agents/langgraph_chatagent.py†L1-L132】【F:automa_ai/agents/remote_agent.py†L1-L244】
- **Skills** are filesystem prompt snippets managed by `automa_ai/skills` and loaded at runtime via a tool call (`load_skill`).【F:automa_ai/skills/README.md†L1-L52】【F:automa_ai/skills/manager.py†L23-L239】
- **Memory** is handled by `automa_ai/memory`, with a configurable manager, store interfaces, and concrete stores (SQLite + Chroma vector store).【F:automa_ai/memory/manager.py†L1-L218】【F:automa_ai/memory/memory_stores.py†L1-L59】【F:automa_ai/memory/sqlite_memory_store.py†L1-L141】

## Subagents (`remote_agent`)

BEM-AI uses A2A (agent-to-agent) calls to delegate work to subagents. Subagents are defined by `SubAgentSpec`, which wraps an A2A `AgentCard` and provides a tool-safe name for function calling.【F:automa_ai/agents/remote_agent.py†L29-L65】

### How delegation works

1. **Define subagent specs** with names, descriptions, and `AgentCard`s.
2. **Register subagents** when building the main agent (via `AgentFactory` or directly in `GenericLangGraphChatAgent`).
3. **Tool creation**: each subagent spec is turned into a `StructuredTool` via `make_subagent_tool`, so the LLM can call it like any other tool.【F:automa_ai/agents/remote_agent.py†L150-L223】
4. **Instruction injection**: the coordinator’s system prompt is augmented with a delegation section that lists available subagents and their descriptions, ensuring the LLM knows when to delegate.【F:automa_ai/agents/remote_agent.py†L12-L27】【F:automa_ai/agents/langgraph_chatagent.py†L93-L106】

### Streaming vs non-streaming

- `A2AToolAdapter.run` performs a single request and synthesizes a final response from artifacts and message history when the subagent does **not** stream.【F:automa_ai/agents/remote_agent.py†L72-L129】
- `A2AToolAdapter.stream` yields intermediate events from `TaskStatusUpdateEvent` and `TaskArtifactUpdateEvent` when the subagent **does** stream, forwarding both text and data chunks to the parent agent’s event queue.【F:automa_ai/agents/remote_agent.py†L131-L200】

### Remote agent transport

`RemoteAgent` uses A2A JSON-RPC transport (`JsonRpcTransport`) over HTTPX. It constructs A2A `MessageSendParams` and supports both invoke and streaming calls for each delegated task.【F:automa_ai/agents/remote_agent.py†L225-L265】

### Example usage

The `examples/subagent_example` shows a coordinator delegating arithmetic to a math subagent using `SubAgentSpec` and A2A servers.【F:examples/subagent_example/chatbot.py†L1-L120】

## Skills

Skills are optional prompt snippets that can be dynamically loaded at runtime to provide reusable instructions. They are not “tools” in the functional sense; instead they are injected as text into the agent’s context.

### How skills are configured

- Skills are configured via `AgentFactory` by passing a `skills_config` block. When enabled, a `load_skill` tool is registered with the agent, allowing LLM calls like `load_skill("write_sql")` to return a prompt snippet.【F:automa_ai/skills/README.md†L3-L52】【F:automa_ai/agents/langgraph_chatagent.py†L108-L123】
- Skills can be registered as:
  - **single files** (explicit registry entry), or
  - **directory mode** (skill name resolved to `foo.md`/`foo.txt`).【F:automa_ai/skills/README.md†L10-L35】

### Skill loading behavior

- The `SkillManager` enforces allowed roots, validates skill names, resolves ambiguity, and caches results for performance.【F:automa_ai/skills/manager.py†L23-L239】
- Loaded skills are wrapped with a standardized header that includes the skill name and source path, making the content traceable for the LLM during reasoning.【F:automa_ai/skills/README.md†L37-L50】

## Memory

BEM-AI supports a short-term + long-term memory architecture via `DefaultMemoryManager`.

### Memory manager responsibilities

- **Short-term writes:** every new message is stored as a `MemoryEntry` (default `MemoryType.SHORT_TERM`).【F:automa_ai/memory/manager.py†L103-L130】【F:automa_ai/memory/memory_types.py†L8-L69】
- **Size management:** when short-term memory exceeds configured thresholds, older/less important entries are moved into long-term storage (or summarized, depending on strategy).【F:automa_ai/memory/manager.py†L131-L154】
- **Retrieval:** both short-term and long-term memories are retrieved, merged, and re-ranked by a combined importance/recency score.【F:automa_ai/memory/manager.py†L156-L196】

### Memory types and entries

Memory entries are typed (`SHORT_TERM`, `LONG_TERM`, `EPISODIC`, `SEMANTIC`) and carry metadata like timestamps, importance, and access counts for ranking.【F:automa_ai/memory/memory_types.py†L8-L69】

### Memory stores and plugins

- `BaseMemoryStore` defines synchronous + asynchronous read/write/delete/clear interfaces.【F:automa_ai/memory/memory_stores.py†L1-L59】
- Stores are registered via `MemoryStoreRegistry` (e.g., SQLite or Chroma vector store) and then wired into `DefaultMemoryManager` via config.【F:automa_ai/memory/memory_stores.py†L44-L59】【F:automa_ai/memory/manager.py†L46-L82】

### Example memory configuration

See `examples/sim_chat_stream_demo/chatbot.py` for a concrete example that wires a SQLite short-term store and a Chroma long-term store into the agent factory’s `memory_config`.【F:examples/sim_chat_stream_demo/chatbot.py†L185-L227】

## Extension tips

- **Add a new subagent**: build an `AgentCard`, wrap it in `SubAgentSpec`, and pass it to the main agent. Ensure the name is unique after normalization (`tool_name`).【F:automa_ai/agents/remote_agent.py†L29-L65】【F:automa_ai/agents/langgraph_chatagent.py†L93-L106】
- **Add a new skill**: drop a `.md` or `.txt` file under an allowed root and register it in `skills_config` (or add a directory registry entry).【F:automa_ai/skills/README.md†L10-L35】
- **Add a new memory store**: implement `BaseMemoryStore`, register with `MemoryStoreRegistry`, then update the `memory_config` for `DefaultMemoryManager`.【F:automa_ai/memory/memory_stores.py†L1-L59】【F:automa_ai/memory/manager.py†L46-L82】

## Building and Testing

The repository does not include a dedicated build step in this document, but you can run focused tests for the key subsystems below.

- **Memory**: `pytest automa_ai/memory`. 
- **Skills**: `pytest automa_ai/skills`. 
- **Subagents**: `pytest examples/subagent_example`. 

## Commit Messages and Pull Requests

Follow the Chris Beams guidelines for commit messages: https://cbea.ms/git-commit/. 
Every pull request should answer the following questions: what changed, why, and whether there are breaking changes. 
All pull request comments should be complete sentences and end with a period. 

## Review Checklist

- All tests must succeed. 
- Add new tests for any new feature or bug fix. 
- Update documentation for user-facing changes. 
