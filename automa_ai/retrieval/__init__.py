"""Retrieval module exports."""
from automa_ai.retrieval.base import BaseRetriever
from automa_ai.retrieval.config import EmbeddingConfig, RetrieverProviderSpec
from automa_ai.retrieval.resolve import resolve_retriever
from automa_ai.retrieval.registry import register_retriever_provider, get_retriever_provider

__all__ = [
    "BaseRetriever",
    "EmbeddingConfig",
    "RetrieverProviderSpec",
    "resolve_retriever",
    "register_retriever_provider",
    "get_retriever_provider",
]
