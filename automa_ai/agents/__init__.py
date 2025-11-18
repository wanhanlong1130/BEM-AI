from enum import Enum


class GenericAgentType(Enum):
    ADK = "adk"
    LANGGRAPH = "langgraph-task"
    LANGGRAPHCHAT = "langgraph-chat"
    ORCHESTRATOR = "orchestrator"


class GenericLLM(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    CLAUDE = "claude"
    LITELLAMA = "litellm"
