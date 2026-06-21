"""Tests for web-search provider fallback behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agents import research_agent


class _NoopSpan:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def log(self, **kwargs):
        pass


class _FailingTavilyClient:
    def search(self, **kwargs):
        raise RuntimeError("Tavily quota exhausted")


class _WorkingExaClient:
    def search(self, query, **kwargs):
        assert query == "latest AI safety news"
        assert kwargs["type"] == "auto"
        assert kwargs["num_results"] == 2
        assert kwargs["contents"] == {"highlights": True}
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    title="AI safety update",
                    url="https://example.com/ai-safety",
                    highlights=["A concise highlighted excerpt."],
                )
            ]
        )


def test_tavily_search_falls_back_to_exa(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(research_agent, "start_span", lambda **kwargs: _NoopSpan())
    monkeypatch.setattr(research_agent, "_get_tavily_client", lambda: _FailingTavilyClient())
    monkeypatch.setattr(research_agent, "_get_exa_client", lambda: _WorkingExaClient())

    output = research_agent.tavily_search("latest AI safety news", max_results=2)

    assert "AI safety update" in output
    assert "https://example.com/ai-safety" in output
    assert "A concise highlighted excerpt." in output


def test_tavily_search_raises_without_exa_key(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.setattr(research_agent, "start_span", lambda **kwargs: _NoopSpan())
    monkeypatch.setattr(research_agent, "_get_tavily_client", lambda: _FailingTavilyClient())

    with pytest.raises(RuntimeError, match="Tavily quota exhausted"):
        research_agent.tavily_search("latest AI safety news", max_results=2)
