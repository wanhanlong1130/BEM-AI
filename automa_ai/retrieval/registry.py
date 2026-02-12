"""Registry for retriever providers."""
from __future__ import annotations

from automa_ai.retrieval.providers.base import RetrieverProvider

_RETRIEVER_PROVIDERS: dict[str, type[RetrieverProvider]] = {}


def register_retriever_provider(name: str, cls: type[RetrieverProvider]) -> None:
    _RETRIEVER_PROVIDERS[name] = cls


def get_retriever_provider(name: str) -> type[RetrieverProvider] | None:
    return _RETRIEVER_PROVIDERS.get(name)
