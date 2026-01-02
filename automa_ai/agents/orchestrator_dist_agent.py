from typing import List

import httpx
from a2a.client import A2ACardResolver
from automa_ai.common.workflow import WorkflowGraph
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel

memory = MemorySaver()
ORCHESTRATOR_INSTRUCTION = "You are an orchestrator agent. Your sole responsibility is to analyze the incoming user request, determine the user's intent, and route the task to exactly one of your expert subagents"

class OrchestratorAgent(BaseAgent):
    """
    Basic Orchestrator that resolves the agent to agent connections
    """

    def __init__(
            self,
            agent_name: str,
            description: str,
            subagent_urls: List[str],
            chat_model: BaseChatModel,
    ):
        super().__init__(
            agent_name=agent_name,
            description=description,
            content_types=["text", "text/plain"],
        )
        self.subagents = []
        self.model = chat_model
        self.context_id = None

        self.agent = create_agent(
            self.model,
            name=self.agent_name,
            checkpointer=memory,
            system_prompt=ORCHESTRATOR_INSTRUCTION
        )

    async def stream(self, query, context_id) -> AsyncIterable[dict[str, Any]]:
        """Execute and stream response."""
        logger.info(
            f"Running {self.agent_name} stream for session {context_id}, {query}"
        )
        if not query:
            raise ValueError("Query cannot be empty")
        if self.context_id != context_id:
            # Clear state when the context changes
            self.clear_state()
            self.context_id = context_id
        while True:
            pass

    async def _build_agent(self, subagent_urls: List[str]) -> List[RemoteA2aAgent]:
        for subagent_url in subagent_urls:
            async with httpx.AsyncClient() as client:
                resolver = A2ACardResolver(
                    httpx_client=client,
                    base_url=subagent_url,
                )
                subagent_card = await resolver.get_agent_card()
                logger.info('Successfully fetched public agent card:' + subagent_card.model_dump_json(indent=2, exclude_none=True))
                # clean name for adk
                clean_name = re.sub(r'[^0-9a-zA-Z_]+', '_', subagent_card.name)
                if clean_name == "":
                    clean_name = "_"
                if clean_name[0].isdigit():
                    clean_name = f"_{clean_name}"

                # make remote agent
                description = json.dumps({
                    "id": clean_name,
                    "name": subagent_card.name,
                    "description": subagent_card.description,
                    "skills": [
                        {
                            "name": skill.name,
                            "description": skill.description,
                            "examples": skill.examples,
                            "tags": skill.tags
                        } for skill in subagent_card.skills
                    ]
                }, indent=2)

                remote_a2a_agent = RemoteA2aAgent(
                    clean_name,
                    subagent_card,
                    description=description,
                )
                self.subagents.append(remote_a2a_agent)
