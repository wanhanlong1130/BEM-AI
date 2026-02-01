from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from automa_ai.common.retrieval.base import BaseRetriever
from automa_ai.common.retrieval.config import RetrieverProviderSpec
from automa_ai.common.retrieval.registry import register_retriever_provider
from automa_ai.common.retrieval.resolve import resolve_retriever


@dataclass
class DummyDoc:
    page_content: str
    metadata: dict[str, Any]


class DummyRetriever(BaseRetriever):
    def similarity_search(self, query: str, *, top_k: int | None = None, **kwargs: Any) -> list[Any]:
        return [query, top_k, kwargs]

    def similarity_search_by_vector(
        self,
        vector: list[float],
        *,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        return [vector, top_k, kwargs]


class CustomRetrieverProvider:
    @classmethod
    def from_config(cls, spec: RetrieverProviderSpec) -> BaseRetriever:
        return DummyRetriever()


class RegistryRetrieverProvider:
    @classmethod
    def from_config(cls, spec: RetrieverProviderSpec) -> BaseRetriever:
        return DummyRetriever()


def test_resolve_registry_provider() -> None:
    register_retriever_provider("dummy", RegistryRetrieverProvider)

    spec = RetrieverProviderSpec(
        provider="dummy",
        top_k=5,
    )

    retriever = resolve_retriever(spec)

    assert isinstance(retriever, DummyRetriever)


def test_resolve_custom_provider_by_impl() -> None:
    spec = RetrieverProviderSpec(
        impl="tests.test_retrieval_resolve:CustomRetrieverProvider",
        top_k=2,
    )

    retriever = resolve_retriever(spec)

    assert isinstance(retriever, DummyRetriever)


def test_resolve_disabled_returns_none() -> None:
    assert resolve_retriever({"enabled": False}) is None


def test_resolve_missing_provider_or_impl() -> None:
    with pytest.raises(ValueError, match="provider|impl"):
        resolve_retriever({"enabled": True})
