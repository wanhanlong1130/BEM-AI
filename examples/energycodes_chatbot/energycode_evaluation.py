import asyncio
from pathlib import Path

from langchain_ollama import ChatOllama

from automa_ai.retrieval import EmbeddingConfig, RetrieverProviderSpec, resolve_retriever
from automa_ai.retrieval.registry import register_retriever_provider
from examples.energycodes_chatbot.helpdesk_retriever import EnergyCodesHelpdeskRetrieverProvider
from automa_ai.agents.langgraph_chatagent import GenericLangGraphChatAgent
from automa_ai.metrics.eval_pipeline import run_deepeval
from examples.energycodes_chatbot.energycode_bot import ENERGY_CODE_ASSISTANT_COT

register_retriever_provider("helpdesk_chroma", EnergyCodesHelpdeskRetrieverProvider)

retriever_spec = RetrieverProviderSpec(
    provider="helpdesk_chroma",
    top_k=2,
    embedding=EmbeddingConfig(
        provider="ollama",
        model="mxbai-embed-large",
        base_url=None,
    ),
    retrieval_provider_config={
        "db_path": str(Path(__file__).parent / "pipeline/chroma_persist"),
        "collection_name": "helpdesk_qna",
    },
)

retriever = resolve_retriever(retriever_spec)


async def main():

    agent = GenericLangGraphChatAgent(
        agent_name="energy_code_assistant",
        description=("Assistant to help users with energy code questions, leveraging past user tickets "
                     "and answers stored in the knowledge base. Can reason step-by-step to determine "
                     "if more information is needed before providing a complete answer."),
        instructions=ENERGY_CODE_ASSISTANT_COT,
        chat_model=ChatOllama(model="llama3.1:8b", temperature=0),
        response_format=None,
        retriever=retriever
    )

    await run_deepeval(
        agent,
        "llama3.1:8b",
        10,
        "/Users/xuwe123/github/BEM-AI/examples/energycodes_chatbot/pipeline/combined_outputs_v2.json",
        output_path="/Users/xuwe123/github/BEM-AI/examples/energycodes_chatbot/output.json"
    )


if __name__ == "__main__":
    asyncio.run(main())
