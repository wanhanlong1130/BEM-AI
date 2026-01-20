import asyncio

from a2a.types import AgentCard, AgentSkill, AgentCapabilities

from automa_ai.agents import GenericAgentType, GenericLLM
from automa_ai.agents.agent_factory import AgentFactory
from automa_ai.agents.remote_agent import SubAgentSpec
from automa_ai.common.agent_registry import A2AServerManager, A2AAgentServer

#### MATH AGENT
math_skill = AgentSkill(
    id="basic_math",
    name="Basic Math",
    description="Performs simple arithmetic calculations",
    tags=["math", "calculation"],
    examples=["What is 3 * 7?", "Calculate 12 x 7"],
)

MATH_AGENT_URL = "http://localhost:31000"

math_agent_card = AgentCard(
    name="Math Subagent",
    description="A subagent that performs basic arithmetic calculations.",
    url=MATH_AGENT_URL,
    version="1.0.0",
    default_input_modes=["text"],
    default_output_modes=["text"],
    capabilities=AgentCapabilities(streaming=True),
    skills=[math_skill],
    supports_authenticated_extended_card=False,
)

MATH_AGENT_COT = """
You are a math subagent.
Your job is to compute the result of arithmetic expressions.
Return only the final numeric answer.
"""

math_agent = AgentFactory(
    card=math_agent_card,
    instructions=MATH_AGENT_COT,
    agent_type=GenericAgentType.LANGGRAPHCHAT,
    chat_model=GenericLLM.OLLAMA,
    model_name="qwen3:4b",
    enable_metrics=True,
    debug=True,
)

### COORDINATOR
coordinator_skill = AgentSkill(
    id="task_coordination",
    name="Task Coordination",
    description="Coordinates tasks and delegates calculations to subagents",
    tags=["coordination"],
    examples=["What is 12 * 7?"],
)

COORD_AGENT_URL = "http://localhost:30000"

coordinator_card = AgentCard(
    name="Coordinator Agent",
    description="Main agent that delegates calculations to subagents.",
    url=COORD_AGENT_URL,
    version="1.0.0",
    default_input_modes=["text"],
    default_output_modes=["text"],
    capabilities=AgentCapabilities(streaming=True),
    skills=[coordinator_skill],
    supports_authenticated_extended_card=False,
)

COORDINATOR_COT = """
You are a coordinator agent.

## AGENT DELEGATION
- Math Subagent: Performs arithmetic calculations

Delegate calculation tasks to the Math Subagent when needed.
"""

coordinator_agent = AgentFactory(
    card=coordinator_card,
    instructions=COORDINATOR_COT,
    agent_type=GenericAgentType.LANGGRAPHCHAT,
    chat_model=GenericLLM.OLLAMA,
    model_name="qwen3:4b",
    subagent_config=[SubAgentSpec(
        name=math_agent_card.name,
        description=math_agent_card.description,
        agent_card=math_agent_card,
    )],
    enable_metrics=True,
    debug=True,
)


###### Add servers

math_a2a = A2AAgentServer(math_agent, math_agent_card)
coordinator_a2a = A2AAgentServer(coordinator_agent, coordinator_card)

server_manager = A2AServerManager()
server_manager.add_server(math_a2a)
server_manager.add_server(coordinator_a2a)


async def main():
    await server_manager.start_all()
    print("✅ A2A Server started at http://localhost:30000/")
    print("Type 'exit' or 'stop' to shut down.")

    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    async def wait_for_input():
        while True:
            cmd = await loop.run_in_executor(None, input, "> ")
            if cmd.strip().lower() in {"exit", "stop", "quit"}:
                stop_event.set()
                break

    await wait_for_input()
    print("🛑 Stopping server...")
    await server_manager.stop_all()
    print("🧹 Server stopped cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
