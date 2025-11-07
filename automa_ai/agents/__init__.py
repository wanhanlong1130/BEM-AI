from enum import Enum


class GenericAgentType(Enum):
    ADK = "adk"
    LANGGRAPH = "langgraph"
    ORCHESTRATOR = "orchestrator"


class GenericLLM(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    CLAUDE = "claude"
    LITELLAMA = "litellm"
