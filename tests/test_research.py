"""
Tests for research agent, LLM router, and search layer.
Run: pytest tests/ -v
"""
import asyncio
import sys
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.config as cfg
from src.search.providers import SearchProviderManager, SearchResult, DuckDuckGoProvider
from src.llm.router import LLMRouter
from src.agents.research_agent import ResearchAgent, ResearchTask, ResearchReport


def reset_settings():
    cfg._settings = None


# ---------------------------------------------------------------------------
# LLMRouter tests
# ---------------------------------------------------------------------------

ALL_PROVIDER_KEYS = [
    "SARVAM_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY",
    "MINIMAX_API_KEY", "OPENROUTER_API_KEY", "NIM_API_KEY", "SILICONFLOW_API_KEY",
    "DEEPSEEK_API_KEY",
]


def test_router_no_keys_returns_unavailable(monkeypatch):
    """Router with no cloud keys and Ollama disabled should report unavailable."""
    for key in ALL_PROVIDER_KEYS:
        monkeypatch.delenv(key, raising=False)
    # Patch _available to return False for ollama too (simulate no local instance)
    from src.llm import router as r
    original = r._available
    r._available = lambda p: False
    try:
        router = LLMRouter()
        assert router.available is False
        assert router.provider_name is None
        assert router.tier == "none"
    finally:
        r._available = original


def test_router_detects_anthropic_key(monkeypatch):
    """Router should pick Anthropic when ANTHROPIC_API_KEY is set and others are absent."""
    for key in ALL_PROVIDER_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    router = LLMRouter(preferred_provider="anthropic")
    assert router.available is True
    assert router.provider_name == "anthropic"
    assert router.tier == "paid"


def test_router_detects_sarvam_key(monkeypatch):
    """Router should pick Sarvam (sovereign-first) when SARVAM_API_KEY is set."""
    for key in ALL_PROVIDER_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SARVAM_API_KEY", "sk_test")
    router = LLMRouter()
    assert router.available is True
    assert router.provider_name == "sarvam"
    assert router.tier == "free"


def test_router_falls_back_to_groq_when_no_paid_key(monkeypatch):
    """Router should use Groq (free) when only GROQ_API_KEY is set."""
    for key in ALL_PROVIDER_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    router = LLMRouter(preferred_provider="groq")
    assert router.provider_name == "groq"
    assert router.tier == "free"


def test_router_status_shows_all_providers(monkeypatch):
    """status() should list all 11 providers (10 cloud + ollama)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    router = LLMRouter()
    status = router.status()
    assert "active_provider" in status
    assert "available_providers" in status
    assert len(status["available_providers"]) == 11
    names = [p["name"] for p in status["available_providers"]]
    assert "sarvam" in names
    assert "anthropic" in names
    assert "groq" in names
    assert "openrouter" in names
    assert "minimax" in names
    assert "ollama" in names


def test_router_call_returns_none_when_unavailable(monkeypatch):
    """call() should return None gracefully when no provider is configured."""
    for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY"]:
        monkeypatch.delenv(key, raising=False)
    router = LLMRouter()
    result = router.call([{"role": "user", "content": "hello"}])
    assert result is None


# ---------------------------------------------------------------------------
# Search provider tests
# ---------------------------------------------------------------------------

def test_provider_manager_uses_duckduckgo_when_no_keys(monkeypatch):
    """With no paid search keys, manager should use DuckDuckGo."""
    for key in ["SERP_API_KEY", "TAVILY_API_KEY", "BRAVE_SEARCH_API_KEY"]:
        monkeypatch.delenv(key, raising=False)
    reset_settings()
    manager = SearchProviderManager()
    names = [p.name for p in manager.providers]
    assert "DuckDuckGo" in names
    reset_settings()


@pytest.mark.asyncio
async def test_search_deduplicates_urls(monkeypatch):
    """Manager should return each URL only once across providers."""
    for key in ["SERP_API_KEY", "TAVILY_API_KEY", "BRAVE_SEARCH_API_KEY"]:
        monkeypatch.delenv(key, raising=False)
    reset_settings()

    dup = SearchResult(
        title="Dup", url="https://example.com/a",
        snippet="x", source="Mock",
        relevance_score=8.0, authority_score=7.0,
    )
    manager = SearchProviderManager()

    async def fake_search(provider, query, max_results):
        return [dup]

    manager._search_with_provider = fake_search
    fake = MagicMock()
    fake.name = "FakeProvider2"
    fake.can_search.return_value = True
    manager.providers = [manager.providers[0], fake]

    results = await manager.search("test", max_results=10)
    assert [r.url for r in results].count("https://example.com/a") == 1
    reset_settings()


# ---------------------------------------------------------------------------
# ResearchAgent integration tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_returns_raw_report_without_llm(monkeypatch):
    """Agent should return a raw ResearchReport when no LLM is configured."""
    for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY",
                "SERP_API_KEY", "TAVILY_API_KEY", "BRAVE_SEARCH_API_KEY"]:
        monkeypatch.delenv(key, raising=False)
    reset_settings()

    mock_result = SearchResult(
        title="AI Agents in Production",
        url="https://example.com/ai-agents",
        snippet="How teams are deploying LLM agents at scale.",
        source="DuckDuckGo",
        relevance_score=8.5,
        authority_score=7.0,
    )

    agent = ResearchAgent()
    agent.search.search = AsyncMock(return_value=[mock_result])
    agent.analyzer.analyze_url = AsyncMock(return_value={
        "summary": "Covers production deployment patterns for LLM agents.",
        "key_insights": ["Routing is key", "Fallbacks prevent downtime"],
        "sentiment": "positive",
        "content_type": "article",
    })

    task = ResearchTask(topic="AI agent deployment", keywords=["LLM", "production"])
    report = await agent.research(task)

    assert isinstance(report, ResearchReport)
    assert report.raw_mode is True
    assert report.llm_provider is None
    assert len(report.sources) == 1
    reset_settings()


@pytest.mark.asyncio
async def test_agent_synthesizes_report_with_llm(monkeypatch):
    """Agent should produce a synthesized report when an LLM is available."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    for key in ["OPENAI_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY",
                "SERP_API_KEY", "TAVILY_API_KEY", "BRAVE_SEARCH_API_KEY"]:
        monkeypatch.delenv(key, raising=False)
    reset_settings()

    mock_result = SearchResult(
        title="RAG Systems 2025",
        url="https://example.com/rag",
        snippet="Best practices for RAG pipelines.",
        source="DuckDuckGo",
        relevance_score=9.0,
        authority_score=8.0,
    )

    synthesis_json = '{"executive_summary": "RAG systems are maturing.", "key_findings": ["Chunking strategy matters", "Hybrid search outperforms dense-only"], "confidence": 0.88}'

    agent = ResearchAgent()
    agent.search.search = AsyncMock(return_value=[mock_result])
    agent.analyzer.analyze_url = AsyncMock(return_value={
        "summary": "Best practices for RAG.",
        "key_insights": ["Chunking matters"],
        "sentiment": "positive",
        "content_type": "article",
    })
    agent.llm.call = MagicMock(return_value=synthesis_json)

    task = ResearchTask(topic="RAG pipelines", keywords=["retrieval", "LLM"])
    report = await agent.research(task)

    assert report.raw_mode is False
    assert report.executive_summary == "RAG systems are maturing."
    assert len(report.key_findings) == 2
    assert report.confidence == 0.88
    reset_settings()
