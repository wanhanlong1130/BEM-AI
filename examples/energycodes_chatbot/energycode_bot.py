import asyncio
import os
from pathlib import Path

from a2a.types import AgentCard, AgentSkill, AgentCapabilities
from dotenv import load_dotenv

from automa_ai.agents import GenericAgentType, GenericLLM
from automa_ai.agents.agent_factory import AgentFactory
from automa_ai.common.agent_registry import A2AServerManager, A2AAgentServer
from automa_ai.common.mcp_registry import MCPServerConfig, MCPServerManager
from examples.sim_chat_stream_demo.mcp_server.mcp_server import serve

base_dir = Path(__file__).resolve().parent
env_path = base_dir / '.env'
load_dotenv(dotenv_path=env_path)

CHAT_COT = """
You are an **Energy Code Assistant**. Your role is to assist users by providing guidance based on past user tickets and answers stored in a knowledge base.

**Important rules**:
1. For every interaction, you **must first review the retrieved content from the knowledge base**, which contains relevant past tickets (questions and answers).
2. Follow a **CHAIN-OF-THOUGHT reasoning process** before generating any response.
3. Decide whether you have enough information to answer the user fully or if you need to ask the user for additional input.

CHAIN-OF-THOUGHT PROCESS:
Before each response, reason through:
1. Understand the user question:  What exactly does the user want to know? Identify the key goal or problem.
2. Assess available information: What information do you already have from past tickets, answers, or other sources that can help? List all known information.
3. Identify gaps: What information is missing or unknown that is necessary to answer the question?
4. Search for missing information: Can the unknown information be found in the provided SAMPLE CODE or reference data? Attempt to locate it.
4. Request additional input if needed: If you cannot find the missing information, formulate a question for the user to obtain it.
5. Provide final answer: If all necessary information is available, generate a clear, accurate, and complete response to the user’s question.

USER QUESTION FORMAT
If you need to ask the user for more information, format the question exactly as follows:
{{ "status": "input_required", "question": "<insert your question here>" }}

"""

CHATBOT_SERVER_URL = os.environ.get("CHATBOT_SERVER_URL")

# Define the primary skill
skill = AgentSkill(
    id="energy_code_assistant",
    name="Energy Code Assistant",
    description=(
        "Assistant to help users with energy code questions, leveraging past user tickets "
        "and answers stored in the knowledge base. Can reason step-by-step to determine "
        "if more information is needed before providing a complete answer."
    ),
    tags=["assistant", "energy code", "compliance", "helpdesk"],
    examples=[
        "How do I set wall orientation in COMcheck?",
        "Can you explain why my exterior wall fails IECC compliance?",
        "What insulation requirements apply to above-grade walls?"
    ],
)

# --8<-- [start:AgentCard]
# Public-facing agent card
public_agent_card = AgentCard(
    name="Energy Code Assistant Agent",
    description=(
        "Provides guidance on energy code compliance questions, referencing past user tickets "
        "and answers from the knowledge base. Uses chain-of-thought reasoning to decide whether "
        "more information is needed from the user before responding."
    ),
    url=CHATBOT_SERVER_URL,
    version="1.0.0",
    default_input_modes=["text"],
    default_output_modes=["text"],
    capabilities=AgentCapabilities(streaming=True),
    skills=[skill],  # Only the primary skill for the public card
    supports_authenticated_extended_card=False,
)

chat_bot_model_name = os.environ.get("CHAT_BOT_MODEL_NAME")
chat_bot_base_url = os.environ.get("CHAT_BOT_MODEL_BASE_URL") or None

# Initialize chatbot agent
chatbot = AgentFactory(
    card=public_agent_card,
    instructions=CHAT_COT,
    model_name=chat_bot_model_name,
    agent_type=GenericAgentType.LANGGRAPHCHAT,
    chat_model=GenericLLM.OLLAMA,
    model_base_url=chat_bot_base_url,
    enable_metrics=True,
    debug=True
)

# Wrap chatbot agent in A2A agent server
chatbot_a2a = A2AAgentServer(chatbot, public_agent_card)

# Initialize A2A server manager
server_manager = A2AServerManager()
# Add server
server_manager.add_server(chatbot_a2a)
# Start network


async def main():
    await server_manager.start_all()
    print("✅ A2A Server started at http://localhost:9999/")
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
