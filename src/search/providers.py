"""
Production-ready search provider system
Integrates with SerpAPI, Tavily, Brave Search, and other real search APIs
"""
import asyncio
import logging
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import requests
import json
from datetime import datetime, timedelta

from src.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Standardized search result format"""
    title: str
    url: str
    snippet: str
    source: str
    published_date: Optional[datetime] = None
    authority_score: float = 0.0
    relevance_score: float = 0.0


class BaseSearchProvider(ABC):
    """Base class for search providers"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.rate_limit_remaining = 100
        self.rate_limit_reset = datetime.now()

    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search for content"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name"""
        pass

    def can_search(self) -> bool:
        """Check if provider can handle a search request"""
        if datetime.now() < self.rate_limit_reset and self.rate_limit_remaining <= 0:
            return False
        return True


class SerpAPIProvider(BaseSearchProvider):
    """SerpAPI Google search provider"""

    @property
    def name(self) -> str:
        return "SerpAPI"

    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search using SerpAPI"""
        if not self.can_search():
            logger.warning("SerpAPI rate limit exceeded")
            return []

        try:
            params = {
                "engine": "google",
                "q": query,
                "api_key": self.api_key,
                "num": min(max_results, 20),
                "gl": "us",
                "hl": "en"
            }

            response = await asyncio.to_thread(
                self.session.get,
                "https://serpapi.com/search",
                params=params,
                timeout=10
            )

            response.raise_for_status()
            data = response.json()

            # Update rate limiting info
            self.rate_limit_remaining -= 1

            results = []
            for item in data.get("organic_results", [])[:max_results]:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    source="Google (SerpAPI)",
                    published_date=self._parse_date(item.get("date")),
                    authority_score=self._calculate_authority(item.get("link", "")),
                    relevance_score=8.5  # SerpAPI provides high-quality results
                ))

            logger.info(f"SerpAPI returned {len(results)} results for query: {query}")
            return results

        except Exception as e:
            logger.error(f"SerpAPI search failed: {e}")
            return []

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date from SerpAPI response"""
        if not date_str:
            return None
        try:
            # Handle various date formats from SerpAPI
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except Exception:
            return None

    def _calculate_authority(self, url: str) -> float:
        """Calculate domain authority score"""
        # Simple authority scoring based on domain
        high_authority_domains = [
            'wikipedia.org', 'reuters.com', 'bbc.com', 'cnn.com',
            'nytimes.com', 'wsj.com', 'bloomberg.com', 'techcrunch.com',
            'arxiv.org', 'nature.com', 'science.org', 'mit.edu'
        ]

        for domain in high_authority_domains:
            if domain in url:
                return 9.0

        if any(ext in url for ext in ['.edu', '.gov', '.org']):
            return 8.0

        return 6.0


class TavilyProvider(BaseSearchProvider):
    """Tavily AI-powered search provider"""

    @property
    def name(self) -> str:
        return "Tavily"

    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search using Tavily API"""
        if not self.can_search():
            logger.warning("Tavily rate limit exceeded")
            return []

        try:
            payload = {
                "api_key": self.api_key,
                "query": query,
                "search_depth": "advanced",
                "include_answer": False,
                "include_images": False,
                "include_raw_content": False,
                "max_results": max_results,
                "include_domains": [],
                "exclude_domains": []
            }

            response = await asyncio.to_thread(
                self.session.post,
                "https://api.tavily.com/search",
                json=payload,
                timeout=15
            )

            response.raise_for_status()
            data = response.json()

            self.rate_limit_remaining -= 1

            results = []
            for item in data.get("results", [])[:max_results]:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    source="Tavily AI",
                    published_date=self._parse_tavily_date(item.get("published_date")),
                    authority_score=item.get("score", 7.0),
                    relevance_score=9.0  # Tavily provides AI-curated results
                ))

            logger.info(f"Tavily returned {len(results)} results for query: {query}")
            return results

        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return []

    def _parse_tavily_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date from Tavily response"""
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except Exception:
            return None


class BraveSearchProvider(BaseSearchProvider):
    """Brave Search API provider"""

    @property
    def name(self) -> str:
        return "Brave Search"

    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search using Brave Search API"""
        if not self.can_search():
            logger.warning("Brave Search rate limit exceeded")
            return []

        try:
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self.api_key
            }

            params = {
                "q": query,
                "count": min(max_results, 20),
                "safesearch": "moderate",
                "freshness": "pw"  # Past week for fresh content
            }

            response = await asyncio.to_thread(
                self.session.get,
                "https://api.search.brave.com/res/v1/web/search",
                headers=headers,
                params=params,
                timeout=10
            )

            response.raise_for_status()
            data = response.json()

            self.rate_limit_remaining -= 1

            results = []
            for item in data.get("web", {}).get("results", [])[:max_results]:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("description", ""),
                    source="Brave Search",
                    published_date=self._parse_brave_date(item.get("age")),
                    authority_score=7.5,
                    relevance_score=8.0
                ))

            logger.info(f"Brave Search returned {len(results)} results for query: {query}")
            return results

        except Exception as e:
            logger.error(f"Brave Search failed: {e}")
            return []

    def _parse_brave_date(self, age_str: Optional[str]) -> Optional[datetime]:
        """Parse relative date from Brave Search"""
        if not age_str:
            return None
        try:
            # Parse relative dates like "2 days ago"
            if "day" in age_str:
                days = int(age_str.split()[0])
                return datetime.now() - timedelta(days=days)
            elif "week" in age_str:
                weeks = int(age_str.split()[0])
                return datetime.now() - timedelta(weeks=weeks)
            elif "hour" in age_str:
                hours = int(age_str.split()[0])
                return datetime.now() - timedelta(hours=hours)
        except Exception:
            pass
        return None


class DuckDuckGoProvider:
    """
    Free DuckDuckGo fallback — no API key required.
    Used automatically when no paid providers are configured.
    """

    @property
    def name(self) -> str:
        return "DuckDuckGo"

    def can_search(self) -> bool:
        return True

    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search using LangChain's DuckDuckGoSearchRun wrapper"""
        try:
            from langchain_community.tools import DuckDuckGoSearchRun
            ddg = DuckDuckGoSearchRun()
            raw = await asyncio.to_thread(ddg.run, query)

            # DuckDuckGoSearchRun returns a single text blob; wrap it as one result
            return [SearchResult(
                title=f"DuckDuckGo results for: {query}",
                url="https://duckduckgo.com/?q=" + query.replace(" ", "+"),
                snippet=raw[:1000] if raw else "No results returned",
                source="DuckDuckGo",
                authority_score=5.0,
                relevance_score=6.0
            )]
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []


class SearchProviderManager:
    """
    Manages multiple search providers with intelligent routing and fallback.
    Falls back to DuckDuckGo automatically when no paid keys are set.
    """

    def __init__(self):
        self.settings = get_settings()
        self.providers = self._initialize_providers()
        self.provider_performance: Dict[str, Any] = {}

        logger.info(f"Initialized {len(self.providers)} search providers: {[p.name for p in self.providers]}")

    def _initialize_providers(self) -> list:
        """Initialize available search providers"""
        providers = []

        # SerpAPI (highest priority - Google results)
        if self.settings.serp_api_key:
            providers.append(SerpAPIProvider(self.settings.serp_api_key))

        # Tavily (AI-powered search)
        if self.settings.tavily_api_key:
            providers.append(TavilyProvider(self.settings.tavily_api_key))

        # Brave Search (privacy-focused)
        if self.settings.brave_search_api_key:
            providers.append(BraveSearchProvider(self.settings.brave_search_api_key))

        # DuckDuckGo as free fallback when no paid providers are configured
        if not providers:
            logger.info("No paid search providers configured — using DuckDuckGo fallback")
            providers.append(DuckDuckGoProvider())

        return providers

    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """
        Intelligent search across multiple providers.
        Runs all available providers concurrently, deduplicates by URL,
        then ranks by a weighted relevance + authority score.
        """
        if not self.providers:
            logger.error("No search providers available")
            return []

        all_results = []
        results_per_provider = max(1, max_results // len(self.providers))

        # Search with all available providers concurrently
        search_tasks = []
        for provider in self.providers:
            if provider.can_search():
                search_tasks.append(
                    self._search_with_provider(provider, query, results_per_provider)
                )

        if not search_tasks:
            logger.warning("All providers rate limited or unavailable")
            return []

        # Execute searches concurrently
        provider_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Combine and deduplicate results
        seen_urls: set = set()
        for results in provider_results:
            if isinstance(results, list):
                for result in results:
                    if result.url not in seen_urls:
                        all_results.append(result)
                        seen_urls.add(result.url)

        # Sort by relevance and authority
        all_results.sort(
            key=lambda x: (x.relevance_score * 0.6 + x.authority_score * 0.4),
            reverse=True
        )

        logger.info(f"Combined search returned {len(all_results)} unique results")
        return all_results[:max_results]

    async def _search_with_provider(self, provider, query: str, max_results: int) -> List[SearchResult]:
        """Search with a single provider and track performance"""
        start_time = datetime.now()

        try:
            results = await provider.search(query, max_results)

            # Track performance
            elapsed = (datetime.now() - start_time).total_seconds()
            self.provider_performance[provider.name] = {
                "last_response_time": elapsed,
                "last_result_count": len(results),
                "last_success": True,
                "timestamp": datetime.now()
            }

            return results

        except Exception as e:
            logger.error(f"{provider.name} search failed: {e}")

            # Track failure
            elapsed = (datetime.now() - start_time).total_seconds()
            self.provider_performance[provider.name] = {
                "last_response_time": elapsed,
                "last_result_count": 0,
                "last_success": False,
                "timestamp": datetime.now(),
                "error": str(e)
            }

            return []

    def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all search providers"""
        return {
            "available_providers": [p.name for p in self.providers],
            "provider_count": len(self.providers),
            "performance_data": self.provider_performance,
            "can_search": any(p.can_search() for p in self.providers)
        }
