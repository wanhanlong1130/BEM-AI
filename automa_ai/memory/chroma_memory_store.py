from pathlib import Path
from typing import Optional, Dict, List, Any

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from automa_ai.memory.memory_stores import BaseMemoryStore
from automa_ai.memory.memory_types import MemoryEntry, MemoryType


class ChromaVectorMemoryStore(BaseMemoryStore):
    """Vector-based memory storage using embeddings for semantic search."""

    @classmethod
    def from_config(cls, config: dict) -> "BaseMemoryStore":
        """
        store: {
            "db_path": str, Path to the database file,
            "collection_name": str, optional
        }
        """
        db_path = config.get("db_path")
        assert db_path, "db_path must be defined for ChromaVectorMemoryStore."

        return cls(
            persist_directory=db_path
        )

    def __init__(
            self,
            persist_directory: str,
            collection_name: str = "memory_store",
            embeddings: Optional[Embeddings] = None,
    ):
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)

        # Initialize vector store
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_directory)
        )

        # Keep a mapping of document IDs to memory entries
        self.memory_mapping: Dict[str, MemoryEntry] = {}

    def write_memory(self, entries: List[MemoryEntry]) -> None:
        """Write a memory entry to vector storage."""
        # Add to vector store
        """Write multiple memory entries to Chroma vector store (bulk insert)."""
        if not entries:
            return  # nothing to insert

        texts = [entry.content for entry in entries]
        metadatas = [
            {
                "session_id": entry.session_id,
                "user_id": entry.user_id,
                "memory_id": entry.id,
                "memory_type": entry.memory_type.value,
                "importance_score": entry.importance_score,
                "timestamp": entry.timestamp.isoformat(),
                **entry.metadata
            }
            for entry in entries
        ]
        record_ids = [entry.record_id for entry in entries]

        # Add all entries at once
        self.vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=record_ids
        )

        # Keep in memory mapping
        for entry in entries:
            self.memory_mapping[entry.record_id] = entry

    def read_memories(
            self,
            query: Optional[str] = None,
            session_id: Optional[str] = None,
            user_id: Optional[str] = None,
            memory_type: Optional[MemoryType] = None,
            limit: int = 10
    ) -> List[MemoryEntry]:
        """Read memory entries using semantic search."""
        filter_dict = build_chroma_filter(session_id, user_id)
        if query:
            # Semantic search
            if filter_dict:
                results = self.vectorstore.similarity_search_with_score(
                    query=query,
                    k=1000,  # Get more results to filter
                    filter=filter_dict,
                )
            else:
                results = self.vectorstore.similarity_search_with_score(
                    query=query,
                    k=1000,  # Get more results to filter
                )

            memories = []
            for doc, score in results:
                memory_id = doc.metadata.get("memory_id")
                if memory_id in self.memory_mapping:
                    memory = self.memory_mapping[memory_id]
                    # Filter by memory type if specified
                    if memory_type is None or memory.memory_type == memory_type:
                        memories.append(memory)

                if len(memories) >= limit:
                    break

            return memories
        else:
            # Return recent memories
            memories = list(self.memory_mapping.values())
            if memory_type:
                memories = [m for m in memories if m.memory_type == memory_type]

            memories.sort(key=lambda x: x.timestamp, reverse=True)
            return memories[:limit]

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory entry."""
        if memory_id in self.memory_mapping:
            # Remove from vector store
            self.vectorstore.delete([memory_id])
            # Remove from mapping
            del self.memory_mapping[memory_id]
            return True
        return False

    def clear_memories(self, memory_type: Optional[MemoryType] = None) -> None:
        """Clear memories of a specific type or all memories."""
        if memory_type is None:
            # Clear everything
            self.vectorstore.delete_collection()
            self.memory_mapping.clear()
            # Reinitialize
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory)
            )
        else:
            # Clear specific memory type
            to_delete = [
                mid for mid, memory in self.memory_mapping.items()
                if memory.memory_type == memory_type
            ]
            for mid in to_delete:
                self.delete_memory(mid)



def build_chroma_filter(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    clauses = []

    if session_id is not None:
        clauses.append({"session_id": {"$eq": session_id}})

    if user_id is not None:
        clauses.append({"user_id": {"$eq": user_id}})

    if not clauses:
        return None  # no filter at all

    if len(clauses) == 1:
        return clauses[0]  # avoid unnecessary $and

    return {"$and": clauses}