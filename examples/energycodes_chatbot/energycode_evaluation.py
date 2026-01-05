import asyncio
from pathlib import Path

from langchain_ollama import ChatOllama

from automa_ai.agents import GenericEmbedModel
from automa_ai.agents.agent_factory import resolve_retriever
from automa_ai.agents.langgraph_chatagent import GenericLangGraphChatAgent
from automa_ai.common.retriever import RetrieverConfig
from automa_ai.metrics.eval_pipeline import run_deepeval
from examples.energycodes_chatbot.energycode_bot import ENERGY_CODE_ASSISTANT_COT

retriever_config = RetrieverConfig(
    db_path=str(Path(__file__).parent / "pipeline/chroma_persist"),
    embeddings="mxbai-embed-large",
    type=GenericEmbedModel.OLLAMA,
    api_key=None,
    top_k=2,
    collection_name="helpdesk_qna",
)

retriever = resolve_retriever(retriever_config)


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