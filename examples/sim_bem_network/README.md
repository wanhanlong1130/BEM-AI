# 💬 BEM-AI (Autonomous OpenStudio Energy Modeling)

## Problem Statement: Complexity and Opportunity in Building Energy Modeling (BEM)

Building Energy Modeling (BEM) is one of the most powerful tools we have for predicting building energy performance.  
However, traditional simulation workflows are **complex, time-consuming, and expertise-intensive**.  
Producing accurate results often requires coordination between architects, engineers, and simulation specialists — each managing different tools, file formats, and datasets.

Meanwhile, **rapid advances in Artificial Intelligence (AI)** are reshaping how complex technical workflows can be automated and scaled.  
AI offers new ways to bring **autonomy, reasoning, and task coordination** to processes that are often iterative, data-heavy, and error-prone.  

One emerging direction is the use of **agentic AI** — systems of specialized AI agents coordinated by a central planner.  
These agents can reason about modeling objectives, invoke domain-specific tools, and communicate dynamically across otherwise incompatible systems.  
In this work, we explore how such an agentic approach can **simplify and scale** BEM through automation and adaptive orchestration.

---

### Frontiers of AI in Building Energy Modeling

Recent research has explored two major directions for integrating AI into BEM:

#### 1. Static Multi-Agent Systems  
Zhang et al. (2025) demonstrated a **static multi-agent framework** where specialized agents follow a fixed workflow to generate EnergyPlus models from text descriptions.  
#### 2. Fine-Tuned Domain Models  
Jiang et al. (2024) proposed **EPlus-LLM**, a fine-tuned large language model that directly translates natural language into EnergyPlus input files.  
#### 3. Usability and Accessibility Frameworks  
Elsayed et al. (2025) emphasized **usability** through a Grasshopper-based framework that translates natural language and images into structured model specifications.  

Despite the progress, significant challenges remain in **generalization, interpretability, and workflow integration**.  
There is a growing need for an AI-driven framework that is **both adaptive and practical** — capable of reasoning about modeling tasks while integrating directly with existing engineering workflows.

---

### Toward a Dynamic, Agentic Framework

#### What is Agent?
> “It started with perception AI… Then generative AI … Now we’re entering the era of ‘physical AI, AI that can perceive, reason, plan and act.”  
> — *Jensen Huang, NVIDIA GTC Keynote, 2025*

In recent years, the term **AI agent** has become central to how we think about the future of artificial intelligence.  
An *agent* can be thought of as an autonomous system — one that **perceives its environment, makes decisions, and takes actions** to achieve specific goals.  
Unlike a simple program, an agent can reason, adapt, and collaborate with other agents or tools.

Most of us have already used agents — often without realizing it.  
When you interact with **ChatGPT in a browser**, or **Claude through the desktop app**, you’re engaging with one form of agent.  
These are conversational or **chat-based agents**, designed to respond to human queries in natural language.

But AI agents can go further.  
The next generation — known as **tool-using agents** — can call APIs, run code, access databases, or control simulation environments to accomplish tasks autonomously.  
They don’t just *talk* about doing something — they actually *do* it.
-----
### Agentic Architectures
The development of agentic AI has progressed through **three major architectural** paradigms, each reflecting a different balance between autonomy, control, and scalability.

<img src="../../sources/typical_agentic_sys_architecture.png" alt="Agent-Archi" width="500">

#### 1. All-in-One (Single Agent) Architecture
This is the simplest form, where a single, powerful model performs all reasoning and execution tasks.
Examples include **Suna**, **Search-R1**, and commercial systems such as **OpenAI Deep Research** and **Perplexity Deep Research**.
This design reduces orchestration overhead and simplifies management but depends heavily on the reasoning capacity of one foundation model, limiting flexibility for complex or evolving tasks.

#### 2. Static Multi-Agent Architecture
To overcome these limits, the static multi-agent approach distributes responsibilities across multiple specialized agents arranged in predefined workflows.
Frameworks such as **AWS Step Functions** and **Google Cloud Workflows** exemplify this paradigm.
Systems like **CoSearchAgent**, **AI Scientist**, and **Agent Laboratory** demonstrate strong predictability and transparency — yet they remain rigid, requiring manual definitions and struggling to adapt to changing goals or unexpected dependencies.

#### 3. Dynamic Multi-Agent Architecture

Dynamic systems represent the newest frontier.
Here, agents are **modular and reconfigurable**, capable of forming or dissolving connections at runtime.
Frameworks such as **LangGraph Multi-Agent Swarm**, **OpenManus AI**, **OWL**, and **DRAMA** enable this flexibility through runtime orchestration, agent discovery, and evolving graph structures.
Such architectures allow workflows to emerge dynamically based on user intent, making them well-suited to domains like **Building Energy Modeling (BEM)** — where both modeling tools and analytical goals are continuously evolving.

---
### Why Dynamic Matters

Dynamic multi-agent systems are **future-proof** by design.
As frontier LLMs evolve and frameworks like LangChain or LangGraph undergo frequent structural changes, static pipelines or monolithic “know-it-all” agents risk obsolescence.
By contrast, dynamic systems emphasize **context engineering** — defining interactions through high-level task logic rather than fixed node connections.
This abstraction isolates changes to individual agents, enabling seamless upgrades, replacements, or specialization without disrupting the overall workflow.

However, this flexibility comes with new challenges: **effective coordination**, **robust communication**, and **standardized protocols** are essential for managing tool access, secure data exchange, and structured interaction across heterogeneous environments.
These challenges motivate frameworks like **BEM-AI**, which integrate standardized communication protocols — such as **Agent-to-Agent (A2A)** and **Model Context Protocol (MCP)** — to orchestrate dynamic, interoperable agent networks for building energy modeling and beyond.

---
### BEM-AI
Now, imagine applying that same paradigm to **building energy modeling (BEM)** — a process that typically requires coordination among multiple domain experts, software tools, and large datasets.  
This is where our work begins.

To bridge this gap, we introduce **BEM-AI** — a **dynamic multi-agent framework** for autonomous building energy modeling.

BEM-AI decomposes modeling tasks into smaller, specialized agents coordinated by a **central planner**.  
Unlike static multi-agent systems, workflows **emerge dynamically** based on user intent and system context.  
Each agent can invoke **programmatic tools, predefined functions, or automation scripts**, ensuring reliable execution while minimizing token and energy costs.

By supporting **smaller, domain-focused language models**, BEM-AI improves efficiency, scalability, and interpretability.  
Most importantly, it connects directly to tools like **OpenStudio** and **EnergyPlus**, making AI integration feasible in real-world engineering practice.

The following sections describe the **BEM-AI architecture**, the **proof of concept design**, and case studies that demonstrate how this system performs key modeling tasks autonomously.

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

To run this example, users will need to provide your own language models when creating agents
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
