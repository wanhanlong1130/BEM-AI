"""Registry for retriever providers."""
from __future__ import annotations

from typing import Dict

from automa_ai.common.retrieval.providers.base import RetrieverProvider

_RETRIEVER_PROVIDERS: Dict[str, type[RetrieverProvider]] = {}


def register_retriever_provider(name: str, cls: type[RetrieverProvider]) -> None:
    _RETRIEVER_PROVIDERS[name] = cls


def get_retriever_provider(name: str) -> type[RetrieverProvider] | None:
    return _RETRIEVER_PROVIDERS.get(name)
