import logging
import os
import sys

from mcp.server import FastMCP
from mcp.server.fastmcp.utilities.logging import get_logger

logger = get_logger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MCP_NAME = "chat_mcp"

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

    @mcp.tool(
        name="get_weather_by_city_and_state",
        description="Get current weather condition by city and state"
    )
    def get_weather_by_city(city: str, state: str) -> str:
        """
        Get the current weather condition by city and state
        :param city: name of the city, For example: "Boulder"
        :param state: name of the State, For example: "Colorado"
        :return: weather condition: Sunny | Windy | Cloudy
        """
        if city.lower() == "tampa" and state.lower() == "florida":
            return "Sunny"
        return "Cloudy"

    logger.info(f"MCP Server at {host}:{port} and transport {transport}")
    mcp.run(transport=transport)