"""Resolver for retriever providers."""
from __future__ import annotations

from importlib import import_module
from typing import Any

from automa_ai.common.retrieval.base import BaseRetriever
from automa_ai.common.retrieval.config import RetrieverProviderSpec
from automa_ai.common.retrieval.registry import get_retriever_provider


def _import_from_path(path: str) -> Any:
    if ":" not in path:
        raise ValueError(
            f"Invalid retriever impl '{path}'. Expected format 'module:ClassName'."
        )
    module_path, attr = path.split(":", 1)
    module = import_module(module_path)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ValueError(
            f"Retriever impl '{path}' could not be imported. Missing attribute '{attr}'."
        ) from exc


def resolve_retriever(spec: RetrieverProviderSpec | dict | None) -> BaseRetriever | None:
    if spec is None:
        return None
    if isinstance(spec, dict):
        spec = RetrieverProviderSpec.model_validate(spec)

    if not spec.enabled:
        return None

    if spec.impl:
        impl = _import_from_path(spec.impl)
        if hasattr(impl, "from_config") and callable(getattr(impl, "from_config")):
            return impl.from_config(spec)
        raise ValueError(
            "Retriever impl must expose a 'from_config' classmethod. "
            f"Got '{spec.impl}'."
        )

    if spec.provider:
        provider_cls = get_retriever_provider(spec.provider)
        if not provider_cls:
            raise ValueError(
                f"Unknown retriever provider '{spec.provider}'. "
                "Register it with register_retriever_provider or provide 'impl'."
            )
        return provider_cls.from_config(spec)

    raise ValueError("Retriever spec must include either 'provider' or 'impl'.")
