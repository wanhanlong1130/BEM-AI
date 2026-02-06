from __future__ import annotations

from typing import Any

import pytest

from automa_ai.tools.web_search.rerank import bm25_scores
from automa_ai.tools.web_search.tool import WebSearchTool


@pytest.mark.asyncio
async def test_provider_selection_auto_uses_serper_when_key_present() -> None:
    from automa_ai.tools.web_search.config import WebSearchToolConfig

    tool = WebSearchTool(
        WebSearchToolConfig.model_validate(
            {"provider": "auto", "serper": {"api_key": "x"}}
        )
    )
    assert tool._select_search_provider() == "serper"


@pytest.mark.asyncio
async def test_url_dedupe_and_max_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    from automa_ai.tools.web_search.config import WebSearchToolConfig
    from automa_ai.tools.web_search import tool as module

    web_tool = WebSearchTool(
        WebSearchToolConfig.model_validate(
            {
                "provider": "opensource",
                "scrape": {"enabled": True, "max_pages": 2},
                "rerank": {"provider": "none", "top_k": 3},
            }
        )
    )

    async def fake_search(
        query: str, max_results: int, region: str | None = None
    ) -> list[dict[str, Any]]:
        return [
            {
                "title": "a",
                "url": "https://a.com",
                "snippet": "one",
                "source": "duckduckgo",
            },
            {
                "title": "a2",
                "url": "https://a.com",
                "snippet": "dup",
                "source": "duckduckgo",
            },
            {
                "title": "b",
                "url": "https://b.com",
                "snippet": "two",
                "source": "duckduckgo",
            },
            {
                "title": "c",
                "url": "https://c.com",
                "snippet": "three",
                "source": "duckduckgo",
            },
        ]

    scraped: list[str] = []

    async def fake_scrape(client: Any, url: str, max_chars: int) -> str:
        scraped.append(url)
        return f"content for {url}"

    monkeypatch.setattr(module, "duckduckgo_search", fake_search)
    monkeypatch.setattr(module, "oss_scrape", fake_scrape)

    result = await web_tool.invoke(
        {"query": "test", "top_k": 3, "include_raw_content": True}
    )
    assert len(scraped) == 2
    assert set(scraped) == {"https://a.com", "https://b.com"}
    assert len(result["results"]) == 3


def test_bm25_scores_deterministic_order() -> None:
    rows = [
        {"title": "alpha beta", "snippet": "", "content": ""},
        {"title": "gamma", "snippet": "alpha", "content": ""},
        {"title": "delta", "snippet": "", "content": ""},
    ]
    scores = bm25_scores("alpha beta", rows)
    assert len(scores) == 3
    assert scores[0] >= scores[1] >= scores[2]
