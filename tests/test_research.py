"""
Tests for the research agent — agent init, search fallback, provider status.

Run with:
    pytest tests/ -v
"""
import asyncio
import sys
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.research_agent import ResearchAgent, ResearchTask, ResearchResult
from src.search.providers import SearchProviderManager, SearchResult, DuckDuckGoProvider
from src.config import Settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_task(**kwargs) -> ResearchTask:
    defaults = dict(topic="AI agents", keywords=["LLM", "agent"], max_results=3)
    defaults.update(kwargs)
    return ResearchTask(**defaults)


# ---------------------------------------------------------------------------
# 1. Agent initialises cleanly with no API keys configured
# ---------------------------------------------------------------------------

def test_agent_init_no_keys(monkeypatch):
    """Agent should initialise without raising even when all API keys are absent."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SERP_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)

    # Reset cached settings singleton so monkeypatched env is picked up
    import src.config as cfg
    cfg._settings = None

    agent = ResearchAgent()

    assert agent.llm is None
    assert agent.agent_executor is None
    assert len(agent.tools) == 2  # web_search + analyze_content always registered

    cfg._settings = None  # clean up


# ---------------------------------------------------------------------------
# 2. get_agent_status returns expected keys
# ---------------------------------------------------------------------------

def test_agent_status_keys(monkeypatch):
    """get_agent_status must return the documented keys."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import src.config as cfg
    cfg._settings = None

    agent = ResearchAgent()
    status = agent.get_agent_status()

    assert "agent_available" in status
    assert "llm_model" in status
    assert "available_tools" in status
    assert "search_providers" in status
    assert "llm_providers" in status
    assert "memory_window" in status

    cfg._settings = None


# ---------------------------------------------------------------------------
# 3. SearchProviderManager falls back to DuckDuckGo when no keys are set
# ---------------------------------------------------------------------------

def test_provider_manager_duckduckgo_fallback(monkeypatch):
    """With no paid API keys, SearchProviderManager should use DuckDuckGo."""
    monkeypatch.delenv("SERP_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)

    import src.config as cfg
    cfg._settings = None

    manager = SearchProviderManager()
    provider_names = [p.name for p in manager.providers]

    assert "DuckDuckGo" in provider_names
    assert len(manager.providers) == 1  # only the fallback

    cfg._settings = None


# ---------------------------------------------------------------------------
# 4. SearchProviderManager.search deduplicates results
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_deduplication(monkeypatch):
    """
    If two providers return the same URL, the manager should deduplicate it
    so the final result list contains it only once.
    """
    monkeypatch.delenv("SERP_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    import src.config as cfg
    cfg._settings = None

    duplicate_result = SearchResult(
        title="Duplicate Article",
        url="https://example.com/article",
        snippet="Same article returned by two providers",
        source="MockProvider",
        relevance_score=8.0,
        authority_score=7.0
    )

    manager = SearchProviderManager()

    # Patch _search_with_provider to return the same result twice
    async def fake_search(provider, query, max_results):
        return [duplicate_result]

    manager._search_with_provider = fake_search

    # Add a second fake provider so the loop runs twice
    fake_provider_2 = MagicMock()
    fake_provider_2.name = "FakeProvider2"
    fake_provider_2.can_search.return_value = True
    manager.providers = [manager.providers[0], fake_provider_2]

    results = await manager.search("test query", max_results=10)

    # The duplicate URL should appear exactly once
    urls = [r.url for r in results]
    assert urls.count("https://example.com/article") == 1

    cfg._settings = None


# ---------------------------------------------------------------------------
# 5. Fallback research returns results without an LLM
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fallback_research_without_llm(monkeypatch):
    """
    _fallback_research should return ResearchResult objects using only the
    search layer — no LLM call required.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SERP_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    import src.config as cfg
    cfg._settings = None

    agent = ResearchAgent()

    # Patch search manager to return a controlled result
    mock_result = SearchResult(
        title="Test Article",
        url="https://example.com/test",
        snippet="This is a test article snippet.",
        source="DuckDuckGo",
        relevance_score=7.0,
        authority_score=6.0
    )
    agent.search_manager.search = AsyncMock(return_value=[mock_result])

    task = make_task(max_results=1)
    results = await agent._fallback_research(task)

    assert len(results) == 1
    assert isinstance(results[0], ResearchResult)
    assert results[0].title == "Test Article"
    assert results[0].url == "https://example.com/test"

    cfg._settings = None
