# 💬 BEM-AI (Autonomous OpenStudio Energy Modeling)

This project provides a **BEM-AI**, a dynamic multi-agent agentic AI system for building energy modeling using OpenStudio.
BEM-AI consists of three parts:

1. **Agents** — an async A2A servers (supports JSON-RPC)
   1. Envelope agent
   2. Lighting agent
   3. Output analysis agent
   4. Planner agent
   5. Simulation agent
   6. Model template agent
   7. Orchestrator
2. **MCP Servers** - async MCP servers (supports JSON-RPC)
   1. OpenStudio MCP, including tools for energy model simulation and model manipulation.
   2. Model MCP, loads template openstudio models.
3. **Client** — a Streamlit-based chat UI built with React-style components and streaming support.

The client communicates with the server in real time, displaying streamed model responses as they arrive.


<img src="../../sources/bem-agentic.png" alt="BEM-Agents" width="500">

To run this example, the user will need to provide your own language models when creating agents
```python
planner = AgentFactory(
    card=agent_card,
    instructions=PLANNER_COT,
    model_name="llama3.3:70b", # need to replace this model to a user accessible language model
    agent_type=GenericAgentType.LANGGRAPH,
    chat_model=GenericLLM.OLLAMA,
    response_format=ResponseFormat,
    model_base_url="http://..." # if needed, provide the base URL.
)
```
It is recommended using a large size model, for example, llama3.3:70b for planner agent and use reasoning models such as qwen3:4b for the specialized agents.

See Live Demo of BEM-AI on [Youtube](https://youtu.be/eYhvig792Sc).

## ⚙️ Prerequisites

Make sure you have the following installed:

- **Python 3.12+**
- **Streamlit**
- **automa_ai 0.2.0**
- **Async libraries** used in your BEM-AI (e.g. `httpx`, etc.)

To install dependencies:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:
```txt
streamlit
openstudio
automa_ai
```

or you can install the package directly with:
```bash
pip install automa_ai[sim_bem_network]
```

---

## ▶️ How to Run

### Option 1: Run both (recommended)
Use the provided shell script to start both server and Streamlit client together:

```bash
chmod +x run_all.sh
./run_all.sh
```

This will:
- Start the BEM-AI server on `http://localhost:10100`
- Wait a few seconds
- Launch the Streamlit chat UI on `http://localhost:8501`

> Press **Ctrl+C** to stop both processes cleanly.

---

### Option 2: Run manually

#### 1️⃣ Start the server
```bash
python sim_bem_netowrk_orchestrator.py
```

Once you see:
```
✅ Agent servers started at http://localhost:10100/
```

#### 2️⃣ In a new terminal, start the client
```bash
streamlit run streamlit_ui.py
```

---

## 💻 Access the Chatbot

After starting, open your browser and go to:

👉 [http://localhost:8501](http://localhost:8501)

You should see the Streamlit-based chat interface.  
Type a message and watch the assistant stream its response in real time.

---

## 🧹 Stopping the App

Press **Ctrl+C** in the terminal where you ran `run_all.sh`.  
Both the server and Streamlit client will shut down gracefully.

---
