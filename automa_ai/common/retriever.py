from dataclasses import dataclass
from typing import List, Any
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from automa_ai.agents import GenericEmbedModel

@dataclass
class RetrieverConfig:
    db_path: str # Path to the store or retrieve vector db
    embeddings: str # Embedding model name
    type: GenericEmbedModel
    top_k: int # top k retrieval
    collection_name: str # Name of the collection in the db
    api_key: str | None # API key to connect with the embedding model

class BaseRetriever:
    def similarity_search_by_vector(self, query) -> list[Any]:
        raise NotImplementedError()

    async def asimilarity_search_by_vector(self, query) -> list[Any]:
        raise NotImplementedError()

class ChromaRetriever(BaseRetriever):
    def __init__(self, db_path: str, collection_name: str, k=4, embeddings: Embeddings | None = None):

        self.embeddings = embeddings
        # import os
        # print("SERVER CWD:", os.getcwd())
        # print(collection_name)
        # print(db_path)
        self.store = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
        # collections = self.store._client.list_collections()
        # print("Collections:", [c.name for c in collections])
        # 3. Check collection count
        # raw = self.store._collection
        # print("Document count:", raw.count())
        self.k = k

    async def asimilarity_search_by_vector(self, query) -> list[Any]:
        if self.embeddings:
            query_embedding = await self.embeddings.aembed_query(query)
        else:
            # Use the chromaDB default embeddings
            query_embedding = query

        results = await self.store.asimilarity_search_by_vector(query_embedding, k=self.k)
        formatted = []
        # sample results format:
        # [Document(id='e30084f6-469c-4e40-9546-2bd39e1759e8', metadata={'source_file': '/Users/xuwe123/github/BEM-AI/examples/energycodes_chatbot/pipeline/combined_outputs_v2.json', 'answer': "Your issue is likely because yes. After logging in, click 'My Projects', then 'Upload projects...', select your .cck file, and click Upload. Save with the blue Save button at the top., here is the solution: Yes. After logging in, click 'My Projects', then 'Upload projects...', select your .cck file, and click Upload. Save with the blue Save button at the top.. Let us know if this resolved your issue. Thank you for contacting the helpdesk."}, page_content='How can I import data from COMcheck Desktop files into the web version?'),
        # Document(id='9bd48195-86fa-4af9-a68b-5cbdce842ad8', metadata={'answer': "Your issue is likely because yes. After logging in, click 'My Projects', then 'Upload projects...', select your .cck file, and click Upload. Save with the blue Save button at the top., here is the solution: Yes. After logging in, click 'My Projects', then 'Upload projects...', select your .cck file, and click Upload. Save with the blue Save button at the top.. Let us know if this resolved your issue. Thank you for contacting the helpdesk.", 'source_file': '/Users/xuwe123/github/BEM-AI/examples/energycodes_chatbot/pipeline/combined_outputs_v2.json'}, page_content='How can I import data from COMcheck Desktop files into the web version?'),
        # Document(id='a27b4f5f-d1dd-46ee-b402-ce9b68d6891b', metadata={'source_file': '/Users/xuwe123/github/BEM-AI/examples/energycodes_chatbot/pipeline/combined_outputs_v2.json', 'answer': "Your issue is likely because in desktop: File → Save as .cck file. In web: Login → My Projects → 'Upload file from...' → select your .cck file., here is the solution: In desktop: File → Save as .cck file. In web: Login → My Projects → 'Upload file from...' → select your .cck file.. Let us know if this resolved your issue. Thank you for contacting the helpdesk."}, page_content='How do I import my desktop COMcheck projects into the web version?')]
        for doc in results:
            metadata = doc.metadata
            page_content = doc.page_content
            formatted.append({
                "relevant_context": page_content,
                "metadata": metadata,
            })
        return formatted

    def similarity_search_by_vector(self, query) -> list[Any]:
        if self.embeddings:
            query_embedding = self.embeddings.embed_query(query)
        else:
            # Use the chromaDB default embeddings
            query_embedding = query
        results = self.store.similarity_search_by_vector_with_relevance_scores(query_embedding, k=self.k)
        formatted = []
        for doc, score in results:
            metadata = doc.metadata
            page_content = doc.page_content
            formatted.append({
                "relevant_context": page_content,
                "metadata": metadata,
                "score": score
            })

        return formatted


class MultiRetriever(BaseRetriever):
    def __init__(self, retrievers: List[BaseRetriever], max_chunks=15):
        self.retrievers = retrievers
        self.max_chunks = max_chunks

    def rank(self, query: str, items: list[Any]) -> list[Any]:
        # Sort by score descending (higher = more relevant)
        ranked = sorted(items, key=lambda x: x.score, reverse=True)
        return ranked[:self.max_chunks]

    def search(self, query: str) -> str:
        # Run all retrievers in parallel
        results =sum([ret.similarity_search_by_vector(query) for ret in self.retrievers])


        ranked = self.rank(query, results)

        context = "\n\n".join([item.content for item in ranked])
        return context

