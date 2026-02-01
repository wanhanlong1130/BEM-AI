"""Embedding factory for retrieval providers."""
from __future__ import annotations

from typing import Any

from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from automa_ai.common.retrieval.config import EmbeddingConfig


def resolve_embeddings(cfg: EmbeddingConfig) -> Any:
    """Resolve an embeddings client based on the provider name.

    Providers can consume cfg.extra for provider-specific settings.
    """
    provider = cfg.provider.lower()
    extra = dict(cfg.extra or {})

    if provider == "ollama":
        if not cfg.model:
            raise ValueError("EmbeddingConfig.model is required for the 'ollama' provider.")
        return OllamaEmbeddings(model=cfg.model, base_url=cfg.base_url, **extra)

    if provider == "openai":
        if not cfg.api_key:
            raise ValueError("EmbeddingConfig.api_key is required for the 'openai' provider.")
        if not cfg.model:
            raise ValueError("EmbeddingConfig.model is required for the 'openai' provider.")
        return OpenAIEmbeddings(
            model=cfg.model,
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            organization=cfg.organization,
            project=cfg.project,
            headers=cfg.headers or None,
            timeout=cfg.timeout_s,
            **extra,
        )

    raise ValueError(
        f"Unsupported embedding provider '{cfg.provider}'. "
        "Supported providers: 'ollama', 'openai'."
    )
