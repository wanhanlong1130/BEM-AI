"""Base retrieval interface for Automa-AI."""
from __future__ import annotations

import asyncio
from typing import Any


class BaseRetriever:
    """Interface for retrievers with sync and async search methods.

    Implementations may override either sync or async methods. The default
    async methods wrap sync implementations using asyncio.to_thread.
    """

    def similarity_search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        raise NotImplementedError

    def similarity_search_by_vector(
        self,
        vector: list[float],
        *,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        raise NotImplementedError

    async def asimilarity_search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        return await asyncio.to_thread(
            self.similarity_search,
            query,
            top_k=top_k,
            **kwargs,
        )

    async def asimilarity_search_by_vector(
        self,
        vector: list[float],
        *,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        return await asyncio.to_thread(
            self.similarity_search_by_vector,
            vector,
            top_k=top_k,
            **kwargs,
        )
