import logging
import os
import sys

from mcp.server import FastMCP

from examples.energycodes_chatbot.helpdesk_retriever import EnergyCodesHelpdeskRetriever
logger = logging.getLogger(__name__)

MCP_NAME = "knowledge_base_mcp"

# counties_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'resources', 'counties.dat')

def serve(host, port, transport):
    """Initialize and runs the agent cards mcp_servers server.
    Args:
        host: The hostname or IP address to bind the server to.
        port: The port number to bind the server to.
        transport: The transport mechanism for the MCP server (e.g., 'stdio', 'sse')

    Raises:
        ValueError
    """
    logger.info("Starting MCP Model Server")
    mcp = FastMCP(MCP_NAME, host=host, port=port)

    log_file = os.path.join("./logs", f"{MCP_NAME}_server_{port}.log")
    os.makedirs("./logs", exist_ok=True)

    # Redirect stdout and stderr to log file — like `> logfile 2>&1`
    sys.stdout = open(log_file, "a", buffering=1)
    sys.stderr = sys.stdout
    retriever = EnergyCodesHelpdeskRetriever(k=3)

    @mcp.tool(
        name="get_similar_past_tickets",
        description="Retrieve past user tickets that contains question and answers."
    )
    def get_similar_past_tickets(user_query: str) -> str:
        """
        Get the current weather condition by city and state
        :param user_query: user inputs, For example: "How do I set wall orientation in COMcheck?"
        :return: str: A formatted string
        """
        print(user_query)
        retrieved = retriever.similarity_search_by_vector(user_query)
        formatted = "You are given the following context from the knowledge base: \n\n"
        for r in retrieved:
            formatted += f"    question: {r["question"]}, answer: {r["answer"]}, score: {r["score"]} \n"
        return formatted

    logger.info(f"MCP Server at {host}:{port} and transport {transport}")
    mcp.run(transport=transport)