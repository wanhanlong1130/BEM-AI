# AUTOMA-AI - Autonomous Multi-Agent Network

AUTOMA-AI is a dynamic multi-agent network system built on Google's A2A (Agent-to-Agent) and Anthropic's MCP (Model Context Protocol) protocols, combining the power of LangChain, Google GenAI, and modern agent orchestration for engineering task orchestration.


## ⚠️ Project Status

This project is in its **early development phase** and is considered **highly unstable**. APIs, interfaces, and core functionality are subject to significant changes. Use for development and experimentation only.

## 🚀 Overview

AUTOMA-AI creates a distributed multi-agent system that enables intelligent agents to communicate, collaborate, and coordinate using industry-standard protocols. The system leverages:

- **Google A2A Protocol**: For agent-to-agent communication
- **Anthropic MCP Protocol**: For model context management
- **LangChain / LangGraph**: For LLM-based agent orchestration and workflow management
- **Google GenAI**: For AI model integration

## 🛠️ Technology Stack

### Core Dependencies
- **LangChain / LangGraph**: Agent framework and orchestration
- **Google GenAI**: AI model integration
- **Google A2A**: Agent-to-agent communication protocol
- **Anthropic MCP**: Model context protocol implementation

### Development Tools
- **uv**: Modern Python package management
- **Python 3.12**: Runtime environment

## 📁 Project Structure

```
BEM-AI/
├── examples/                           # Example engineering applications built with the foundational framework
├── automa_ai/
│   ├── agent_test/                     # Test implementations and examples
│   ├── agents/                         # Generic agent classes
│   │   ├── react_langgraph_agent.py    # langchain/langgraph based agent
│   │   ├── agent_factor.py             # Agent factory - recommend utility to initialize an agent
│   │   ├── orchestrator_agent.py       # An agent that orchestrates the task workflow
│   │   └── adk_agent.py                # Google ADK based agent
│   ├── client/                         # Under development
│   ├── mcp_servers/                    # MCP library
│   ├── network/                        # Network
│   ├── common/                         # Common utilities
│   └── prompt_engineering/             # Under development
├── pyproject.toml                      # Project configuration
├── uv.lock                             # Dependency lock file
└── README.md                           # This file
```

## 🔧 Installation
We recommend install AUTOMA-AI through PYPI:
```shell
pip install automa-ai
```
This will install all packages needed under automa_ai folder.


### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd bem-ai
   ```

2. **Install dependencies using uv**
   ```bash
   uv sync
   ```

3. **Activate the virtual environment**
   ```bash
   uv shell
   ```

## 🧪 Running Tests
TBD

## 🏗️ Architecture
<img src="sources/architecture.png" alt="System Architecture" width="600">

- **Orchestrator**: Assemble workflow, access agent card storage
- **Task Memory**: Task memory including shared blackboard and conversation history
- **Planner**: A planner agent
- **Summary**: A summary agent
- **Specialized agents**: Domain specific agents
- **Agent Card Service**: A RAG pipeline stores agent cards
- **Tool and Resources**: External tool and resource access through MCPs

## 📝 Configuration

Project configuration is managed through `pyproject.toml`. Key configuration areas include:

- **Dependencies**: Core and development packages
- **Build System**: uv-based build configuration
- **Project Metadata**: Version, description, and author information
- **Optional**: optional packages to use for UI integration and running examples.

## Examples
#### Single Agent Chatbot with Streamlit UI interface
This example demonstrates the use of automa-ai for creating a live-streaming chatbot.
The example uses QWEN3:4B as the language model and a sample MCP server is built to connect with the agent, demonstrating the capabilities of streaming and tool calling using a single chat bot.
See [README](./examples/sim_chat_demo/README.md)

#### Simple BEM typical building Network
This example is the prototype of BEM-AI, which consists of multiple agents collaboratively completing a building energy modeling task together.
See [README](./examples/sim_bem_network/README.md)

### EnergyPlus Chatbot with EnergyPlus MCP server
This example shows automa-ai integrates with EnergyPlus MCP, developed by LBNL.
See [README](./examples/eplus_mcp_demo/README.md)


## 🔍 Development Guidelines

### Code Organization
TBD

### Dependency Management
- Use `uv add <package>` to add new dependencies
- Update `uv.lock` with `uv lock` after dependency changes
- Keep dependencies minimal and focused

### Testing Strategy
TBD

## 🤝 Contributing
TBD

## 📄 License

see [LICENSE](/LICENSE.md)

---

**Note**: This project is experimental and under active development. Use in production environments is not recommended at this time.

## 📚 Citation

If you use this framework in your research or projects, please cite the following paper:

```bibtex
@article{xu5447218development,
  title={Development of a dynamic multi-agent network for building energy modeling: A case study towards scalable and autonomous energy modeling},
  author={Xu, Weili and Wan, Hanlong and Antonopoulos, Chrissi and Goel, Supriya},
  journal={Available at SSRN 5447218}
}
