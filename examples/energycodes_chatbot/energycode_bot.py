import asyncio
import os
from pathlib import Path

from a2a.types import AgentCard, AgentSkill, AgentCapabilities
from dotenv import load_dotenv

from automa_ai.agents import GenericAgentType, GenericLLM, GenericEmbedModel
from automa_ai.agents.agent_factory import AgentFactory
from automa_ai.common.agent_registry import A2AServerManager, A2AAgentServer
from automa_ai.common.retriever import RetrieverConfig

base_dir = Path(__file__).resolve().parent
env_path = base_dir / '.env'
load_dotenv(dotenv_path=env_path)

ENERGY_CODE_ASSISTANT_COT = """
You are an **Energy Code Assistant**. Your role is to assist users by providing guidance.
Keep your answer concise and address the user question only.

Some General Information that can give you background on the software tools that user mostly asking for.

### GENERAL INFORMATION
You may encounter user asking general questions that the past information may not be helpful. This section provides some general information.

Currently, there are three check tools, the new COMcheck-Web, legacy COMcheck-Web, REScheck-Web, COMcheck Desktop and REScheck Desktop.
For questions relate to COMcheck and REScheck desktop versions, you should politely respond that both versions are no longer supported.

The new COMcheck-Web:
Description: The new version of COMcheck-Web supports commercial and high-rise residential energy code compliance while providing better user navigation and experience. All future national or state energy codes will be implemented in the new version of COMcheck-Web.
Supported energy codes: 2015. 2018, 2021, and 2024 IECC; ASHRAE Standard 90.1-2013, 2016, 2019, and 2022, and State-specific: Colorado (Boulder and Denver), Louisiana, Massachusetts, Minnesota, New York City (NYCECC), NYSTRETCH, Vermont, Ontario, and Puerto Rico

The legacy COMcheck-Web:
Description: The legacy COMcheck-Web supports commercial and high-rise residential energy code compliance while providing better user navigation and experience. All future national or state energy codes will be implemented in the new version of COMcheck-Web.
Supported energy codes: 2009, 2012, 2015. 2018, and 2021 IECC; ASHRAE Standard 90.1-2007, 2013, 2016, and 2019, and State-specific: Colorado (Boulder and Denver), Louisiana, Massachusetts, Minnesota, New York City (NYCECC), NYSTRETCH, Vermont, Ontario, and Puerto Rico

The REScheck-Web:
Description: The REScheck product group makes it fast and easy for builders, designers, and contractors to determine whether new homes, additions, and alterations meet the requirements of the IECC or a number of state energy codes. REScheck also simplifies compliance determinations for building officials, plan checkers, and inspectors by allowing them to quickly determine if a low-rise residence meets the code.
Supported energy codes: 2009, 2012, 2015, 2018, 2021, and 2024 IECC, and State Energy Codes: Florida (2017), Louisiana (2021). Massachusetts (2023), New York City, DC (2017), Denver (2019), Puerto Rico, Utah (2012), Vermont (2020, 2024) 

"""

# retriever_mcp_config = MCPServerConfig(
#    name="retriever_mcp",
#    host="localhost",
#    port=10000,
#    serve=serve,
#    transport="sse"
#)

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

retriever_config = RetrieverConfig(
    db_path=str(Path(__file__).parent / "pipeline/chroma_persist"),
    embeddings="mxbai-embed-large",
    type=GenericEmbedModel.OLLAMA,
    api_key=None,
    top_k=3,
    collection_name="helpdesk_qna",
)

# Initialize chatbot agent
chatbot = AgentFactory(
    card=public_agent_card,
    instructions=ENERGY_CODE_ASSISTANT_COT,
    model_name=chat_bot_model_name,
    agent_type=GenericAgentType.LANGGRAPHCHAT,
    chat_model=GenericLLM.OLLAMA,
    model_base_url=chat_bot_base_url,
    # mcp_configs={"retriever":retriever_mcp_config},
    retriever_config=retriever_config,
    enable_metrics=True,
    debug=True
)

# mcp_manager = MCPServerManager()
# mcp_manager.add_server(retriever_mcp_config)

# Wrap chatbot agent in A2A agent server
chatbot_a2a = A2AAgentServer(chatbot, public_agent_card)

# Initialize A2A server manager
server_manager = A2AServerManager()
# Add server
server_manager.add_server(chatbot_a2a)
# Start network


async def main():
    # await mcp_manager.start_all()
    # print("✅ MCP Server started at http://localhost:10000/")
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
    # await mcp_manager.stop_all()
    print("🧹 Server stopped cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
