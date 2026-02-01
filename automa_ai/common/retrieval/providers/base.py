"""Base retriever provider interface."""
from __future__ import annotations

from typing import Protocol

from automa_ai.common.retrieval.base import BaseRetriever
from automa_ai.common.retrieval.config import RetrieverProviderSpec


class RetrieverProvider(Protocol):
    @classmethod
    def from_config(cls, spec: RetrieverProviderSpec) -> BaseRetriever:
        ...
