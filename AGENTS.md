# BEM-AI Agent Guide

This document provides an AI-focused orientation to BEM-AI’s agent architecture, with emphasis on retrieval, subagents (`remote_agent`), skills, and memory. It is meant to help an LLM understand how the runtime is wired and where to look when extending or debugging behavior.

## Repository mental model

- **Core agent runtime** lives in `automa_ai/agents` (LangGraph-based chat agent, A2A subagents, agent factory).【F:automa_ai/agents/langgraph_chatagent.py†L1-L132】【F:automa_ai/agents/remote_agent.py†L1-L244】
- **Retrieval** is handled by `automa_ai/retrieval` (retriever interface, provider registry/resolver, embedding factory) and is wired into chat agents through `AgentFactory`.【F:automa_ai/retrieval/base.py†L1-L59】【F:automa_ai/retrieval/resolve.py†L1-L52】【F:automa_ai/agents/agent_factory.py†L137-L185】
- **Skills** are filesystem prompt snippets managed by `automa_ai/skills` and loaded at runtime via a tool call (`load_skill`).【F:automa_ai/skills/README.md†L1-L52】【F:automa_ai/skills/manager.py†L23-L239】
- **Memory** is handled by `automa_ai/memory`, with a configurable manager, store interfaces, and concrete stores (SQLite + Chroma vector store).【F:automa_ai/memory/manager.py†L1-L218】【F:automa_ai/memory/memory_stores.py†L1-L59】【F:automa_ai/memory/sqlite_memory_store.py†L1-L141】

## Retrieval

BEM-AI retrieval uses a provider abstraction resolved at agent construction time.

### Retrieval contracts and resolution

- `BaseRetriever` defines sync text/vector search plus async wrappers (`asimilarity_search*`) that default to `asyncio.to_thread(...)`.【F:automa_ai/retrieval/base.py†L8-L59】
- `RetrieverProviderSpec` requires exactly one of `provider` or `impl` when enabled; `enabled=False` short-circuits retrieval entirely.【F:automa_ai/retrieval/config.py†L21-L37】
- `resolve_retriever(...)` supports:
  - **registry path** (`provider`) via `register_retriever_provider`, or
  - **direct import path** (`impl`) in `module:ClassName` format.
  In both cases, the resolved class must expose `from_config(spec)`.【F:automa_ai/retrieval/registry.py†L1-L14】【F:automa_ai/retrieval/resolve.py†L12-L52】
- Embeddings are resolved separately via `resolve_embeddings(...)`; currently supported providers are `ollama` and `openai`.【F:automa_ai/retrieval/embedding_factory.py†L12-L44】

### Runtime integration in chat agents

- `AgentFactory` resolves `retriever_spec` and injects the resulting retriever into `GenericLangGraphChatAgent`.【F:automa_ai/agents/agent_factory.py†L171-L185】
- During request handling, `GenericLangGraphChatAgent._build_stream_inputs(...)`:
  1. calls `retriever.asimilarity_search(query)` when configured,
  2. serializes returned context into an additional system prompt block,
  3. prepends that system message before the user message.【F:automa_ai/agents/langgraph_chatagent.py†L331-L367】

### Energy Codes example (`examples/energycodes_chatbot`)

- `helpdesk_retriever.py` implements a Chroma-backed retriever + provider:
  - provider key: `helpdesk_chroma`,
  - requires `db_path` or `persist_directory`,
  - optional `collection_name` (default `helpdesk_qna`) and `chroma_kwargs`,
  - optional embeddings via `spec.embedding`,
  - normalizes output as `{"relevant_context", "metadata", "score?"}` records.【F:examples/energycodes_chatbot/helpdesk_retriever.py†L11-L94】
- `energycode_bot.py` registers the provider, builds `RetrieverProviderSpec`, and passes it to `AgentFactory`:
  - top-k retrieval set to `3`,
  - embedding provider is `ollama` with model `mxbai-embed-large`,
  - persisted Chroma location: `examples/energycodes_chatbot/pipeline/chroma_persist`,
  - collection: `helpdesk_qna`.【F:examples/energycodes_chatbot/energycode_bot.py†L93-L119】
- Practical requirement: provider registration (`register_retriever_provider`) must happen before `AgentFactory(...)` resolves `retriever_spec`, otherwise provider lookup fails.【F:automa_ai/retrieval/resolve.py†L45-L51】【F:examples/energycodes_chatbot/energycode_bot.py†L93-L110】

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
- **Serve an agent under a base path**: pass `base_url_path="/my-path"` to `A2AAgentServer`, and ensure the agent card `url` and client URLs include the same prefix (trailing slash recommended to avoid SSE redirects).【F:automa_ai/common/agent_registry.py†L21-L98】
- **Add a new retriever provider**: implement a provider with `from_config(spec) -> BaseRetriever`, register it with `register_retriever_provider`, and pass a `RetrieverProviderSpec` into `AgentFactory(retriever_spec=...)`.【F:automa_ai/retrieval/providers/base.py†L1-L13】【F:automa_ai/retrieval/registry.py†L1-L14】【F:automa_ai/agents/agent_factory.py†L107-L185】
- **Add a new skill**: drop a `.md` or `.txt` file under an allowed root and register it in `skills_config` (or add a directory registry entry).【F:automa_ai/skills/README.md†L10-L35】
- **Add a new memory store**: implement `BaseMemoryStore`, register with `MemoryStoreRegistry`, then update the `memory_config` for `DefaultMemoryManager`.【F:automa_ai/memory/memory_stores.py†L1-L59】【F:automa_ai/memory/manager.py†L46-L82】


## Default tools (non-MCP)

BEM-AI now supports first-class default tools configured directly in `AgentFactory` via `tools_config` (not MCP).

- Core configuration models are in `automa_ai/config/tools.py` (`ToolsConfig`, `ToolSpec`).
- Registry/factory and internal tool interface are in `automa_ai/tools/base.py` and `automa_ai/tools/registry.py`.
- Built-in tools are registered in `automa_ai/tools/__init__.py`.
- `AgentFactory` accepts `tools_config` and passes tool specs to `GenericLangGraphChatAgent`, which binds tools only when configured.
- Built-in `web_search` implementation is under `automa_ai/tools/web_search/` with Serper/OSS search, Firecrawl/OSS scraping, and Jina/Cohere/OSS reranking.

## Building and Testing

The repository does not include a dedicated build step in this document, but you can run focused tests for the key subsystems below.

- **Memory**: `pytest automa_ai/memory`. 
- **Retrieval**: `pytest tests/test_retrieval_resolve.py`.
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

## Learning Mode
When a user explicitly expressed that they are currently onboarding or learning this repository, the agent shall follow the additional instructions in the learning mode.

### BEFORE AGENT WRITING CODE
- Explain what you're about to do and why
- Break it down into steps the user can follow
- Wait for the user's OK before proceeding

### AFTER WRITING CODE
- Explain what each part does
- Ask the user **3 questions** to verify their understanding
- If the user answer wrong, explain again until the user get it
- **Do NOT let the user commit** until the user pass your questions

### GENERAL RULES FOR LEARNING MODE
- **Never** generate code the user can't explain
- If the user asks for something complex, **suggest simpler alternatives**
- Treat every session as a **teaching opportunity**
- Be direct, **Tell the user when they are doing something wrong**
