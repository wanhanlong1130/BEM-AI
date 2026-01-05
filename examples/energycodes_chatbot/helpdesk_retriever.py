from typing import Any

from langchain_ollama import OllamaEmbeddings

from automa_ai.common.retriever import ChromaRetriever

class EnergyCodesHelpdeskRetriever(ChromaRetriever):
    def __init__(self, db_path="/Users/xuwe123/github/BEM-AI/examples/energycodes_chatbot/pipeline/chroma_persist", collection_name="helpdesk_qna", k=4):
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        super().__init__(
            db_path=db_path,
            collection_name=collection_name,
            embeddings=embeddings,
            k=k
        )

    def similarity_search_by_vector(self, query: str) -> list[Any]:
        if self.embeddings:
            query_embedding = self.embeddings.embed_query(query)
        else:
            # Use the chromaDB default embeddings
            query_embedding = query
        # Chroma is sync, so we run in thread to avoid blocking
        results = self.store.similarity_search_by_vector_with_relevance_scores(query_embedding, k=self.k)
        formatted = []
        for doc, score in results:
            metadata = doc.metadata
            page_content = doc.page_content
            formatted.append({
                "question": page_content,
                "answer": metadata["answer"],
                "score": 1 - score
            })

        return formatted

if __name__ == "__main__":
   embedder = OllamaEmbeddings(model="mxbai-embed-large")
   retriever = EnergyCodesHelpdeskRetriever()
   doc = retriever.similarity_search_by_vector("How do I set wall orientation in COMcheck?")
   print(doc)

   ollama_retriever = ChromaRetriever(
       db_path="/Users/xuwe123/github/BEM-AI/examples/energycodes_chatbot/pipeline/chroma_persist",
       collection_name="helpdesk_qna",
       k=3,
       embeddings=embedder
   )
   docs = ollama_retriever.similarity_search_by_vector("How do I set wall orientation in COMcheck?")
   print(docs)
