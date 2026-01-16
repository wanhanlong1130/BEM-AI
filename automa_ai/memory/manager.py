"""Memory module for managing conversation history and context retrieval"""
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from automa_ai.memory.memory_stores import BaseMemoryStore, MemoryStoreRegistry
from automa_ai.memory.memory_types import MemoryType, MemoryEntry

DEFAULT_SHORT_TERM_LIMIT = 10
DEFAULT_LONG_TERM_STRATEGY = "summarize"
DEFAULT_SHORT_TERM_MAX = 30

from dataclasses import dataclass
from typing import Optional
from langchain_core.messages import BaseMessage

@dataclass
class MemoryWriteEvent:
    # data class define memory writing event
    message: BaseMessage
    session_id: str
    user_id: Optional[str] = None


class DefaultMemoryManager:
    """
    Memory Manager Configuration
    short_term_limit: int the max number of active short-term memory, default is 10
    short_term_buffer: int the buffer for flushing the short-term active memory to a long-term storage
    long_term_strategy: Literal["messages" | "summarize"]: choose messages will convert the short-term memory to a long-term memory, summarize will convert the short-term memories to one single summarized long-term memories
    stores: List[MemoryStore]
        {
            "name": str: the store name that is created by default or otherwise customized,
            "memory_type": MemoryType,
            "store_config: {
                wildcards that can be read by the memory store -> see instruction in the memory store.
            },
        }
    """
    @classmethod
    def from_config(cls, config: dict) -> "DefaultMemoryManager":
        short_term_limit = config.get("short_term_limit") or DEFAULT_SHORT_TERM_LIMIT
        long_term_strategy = config.get("long_term_strategy") or DEFAULT_LONG_TERM_STRATEGY
        assert long_term_strategy in ["messages", "summarize"], "Long term memory strategy must be one of messages, summarize"

        short_term_max = config.get("short_term_max") or DEFAULT_SHORT_TERM_MAX
        stores = config.get("stores") or []

        short_term_store = None
        long_term_store = None

        for store in stores:
            memory_type = store.get("memory_type")
            store_name = store.get("name")

            assert store_name, "Missing store name."
            assert isinstance(memory_type, MemoryType), "Memory type must be one of the MemoryType"

            store_cls = MemoryStoreRegistry.get(store_name)

            if memory_type == MemoryType.SHORT_TERM:
                short_term_store = store_cls.from_config(store["store_config"])
            elif memory_type == MemoryType.LONG_TERM:
                long_term_store = store_cls.from_config(store["store_config"])
            else:
                raise ValueError("Manager only supports long-term and short-term memories right now. For future releases, please check back on the repo.")

        return cls(
            short_term_store=short_term_store,
            long_term_store=long_term_store,
            short_term_limit=short_term_limit,
            max_short_term_memories=short_term_max,
        )


    """Main memory manager that orchestrates different memory stores and strategies."""
    def __init__(
            self,
            short_term_store: Optional[BaseMemoryStore] = None,
            long_term_store: Optional[BaseMemoryStore] = None,
            short_term_limit: int = DEFAULT_SHORT_TERM_LIMIT,
            max_short_term_memories: int = DEFAULT_SHORT_TERM_MAX,
            memory_decay_hours: int = 24,
    ):
        # Data validation
        self.short_term_store = short_term_store
        self.long_term_store = long_term_store
        self.short_term_limit = short_term_limit

        # buffer number set to 50%.
        self.max_short_term_memories =max_short_term_memories
        self.memory_decay_hours = memory_decay_hours

    async def add_memory(
            self,
            message: AIMessage | HumanMessage | ToolMessage,
            session_id: str,
            user_id: Optional[str] = None,
            importance_score: float = 0.5
    ) -> None:
        timestamp = datetime.now()

        role_map = {AIMessage: "agent", HumanMessage: "human", ToolMessage: "tool"}
        role = role_map.get(type(message))
        if not role:
            raise ValueError("Message must be an AIMessage, HumanMessage, or ToolMessage")

        if user_id is None:
            user_id = " "

        entry = MemoryEntry(
            content=message.content,  # Use instance content
            metadata={**getattr(message, 'response_metadata', {}), "role": role},
            timestamp=timestamp,
            memory_type=MemoryType.SHORT_TERM,
            importance_score=importance_score,
            session_id=session_id,
            user_id=user_id,
        )
        await self.short_term_store.awrite_memory([entry])

    async def manage_memory_size(self) -> None:
        """Manage memory size by moving old memories to long-term storage."""
        short_memories = await self.short_term_store.aread_memories(
            memory_type=MemoryType.SHORT_TERM,
            limit= self.max_short_term_memories * 2 # Get all short-term memories
        )
        if len(short_memories) > self.max_short_term_memories:
            # Sort by importance and age
            short_memories.sort(
                key=lambda x: (x.importance_score, x.timestamp))

            # Keep the most important/recent ones in short-term
            to_move = short_memories[self.short_term_limit:]
            move_ids = [m.id for m in to_move]

            # Wrap in a background task that deletes only ON SUCCESS
            async def safe_transfer():
                try:
                    await self.long_term_store.awrite_memory(to_move)
                    await self.short_term_store.adelete_memory(move_ids)
                except Exception as e:
                    print(f"FAILED to move memories to LTM: {e}")

            asyncio.create_task(safe_transfer())

    async def retrieve_memories(
            self,
            query: str,
            session_id: str | None = None,
            memory_types: Optional[List[MemoryType]] = None,
            limit: int = 10,
            include_short_term: bool = True,
            include_long_term: bool = True
    ):
        """Retrieve relevant memories based on query."""
        all_memories = []

        if include_short_term and (not memory_types or MemoryType.SHORT_TERM in memory_types):
            short_memories = await self.short_term_store.aread_memories(
                query=query,
                memory_type=MemoryType.SHORT_TERM,
                limit=limit,
                session_id=session_id,
            )
            if short_memories:
                all_memories.extend(short_memories)

        if include_long_term and (not memory_types or any(mt != MemoryType.SHORT_TERM for mt in (memory_types or []))):
            for memory_type in (memory_types or [MemoryType.LONG_TERM, MemoryType.EPISODIC, MemoryType.SEMANTIC]):
                if memory_type != MemoryType.SHORT_TERM:
                    long_memories = await self.long_term_store.aread_memories(
                        query=query,
                        memory_type=memory_type,
                        limit=limit,
                        session_id=session_id,
                    )
                    if long_memories:
                        all_memories.extend(long_memories)

        # Sort by relevance (importance + recency)
        all_memories.sort(
            key=lambda x: (x.importance_score * 0.7 + self.calculate_recency_score(x) * 0.3),
            reverse=True
        )

        return all_memories[:limit]

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about current memory usage."""
        short_term_count = len(self.short_term_store.read_memories(limit=1000))
        long_term_count = len(self.long_term_store.read_memories(limit=1000))

        return {
            "short_term_memories": short_term_count,
            "long_term_memories": long_term_count,
            "total_memories": short_term_count + long_term_count,
        }

    @staticmethod
    def calculate_recency_score(memory: MemoryEntry) -> float:
        """Calculate a recency score (0-1) based on how recent the memory is."""
        now = datetime.now()
        age_hours = (now - memory.timestamp).total_seconds() / 3600

        if age_hours <= 1:
            return 1.0
        elif age_hours >= 24:
            return 0.1
        else:
            return 1.0 - (age_hours / 24.0) * 0.9


