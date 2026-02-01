"""Retrieval module exports."""
from automa_ai.common.retrieval.base import BaseRetriever
from automa_ai.common.retrieval.config import EmbeddingConfig, RetrieverProviderSpec
from automa_ai.common.retrieval.resolve import resolve_retriever
from automa_ai.common.retrieval.registry import register_retriever_provider, get_retriever_provider

__all__ = [
    "BaseRetriever",
    "EmbeddingConfig",
    "RetrieverProviderSpec",
    "resolve_retriever",
    "register_retriever_provider",
    "get_retriever_provider",
]
