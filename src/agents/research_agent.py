"""
Research Agent — multi-provider LLM + multi-provider search.

No LangChain required. Uses the LLMRouter directly for provider-agnostic
LLM calls (Anthropic / OpenAI / Groq / Gemini) and SearchProviderManager
for concurrent multi-source search.

Two modes:
  - With LLM key: search → analyze top sources → synthesize into ResearchReport
  - Without LLM key: search → return raw ranked results with no synthesis
"""
import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.config import get_settings
from src.content.analyzer import ContentAnalyzer
from src.llm.router import LLMRouter
from src.search.providers import SearchProviderManager, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class ResearchTask:
    topic: str
    keywords: List[str]
    max_results: int = 8
    focus_areas: List[str] = field(default_factory=list)
    depth: str = "standard"  # "quick" | "standard" | "deep"


@dataclass
class SourceSummary:
    """One analyzed source."""
    title: str
    url: str
    summary: str
    key_insights: List[str]
    relevance_score: float
    authority_score: float
    sentiment: str
    content_type: str


@dataclass
class ResearchReport:
    """
    Structured research output — not just a list of URLs.

    executive_summary: 2-3 sentence synthesis of findings.
    key_findings: top 5-7 insights across all sources.
    sources: individually analyzed source summaries.
    confidence: overall confidence in the research (0-1).
    llm_provider: which LLM synthesized this report (or None).
    """
    topic: str
    executive_summary: str
    key_findings: List[str]
    sources: List[SourceSummary]
    confidence: float
    generated_at: datetime
    llm_provider: Optional[str]
    search_providers: List[str]
    raw_mode: bool = False  # True when no LLM available


class ResearchAgent:
    """
    Researches any topic using concurrent multi-source search and LLM synthesis.

    Free tier (no LLM key set):
      - Uses DuckDuckGo for search
      - Returns ranked results without synthesis
      - Set GROQ_API_KEY or GEMINI_API_KEY for free LLM synthesis

    Paid tier (with LLM key):
      - SerpAPI + Tavily for best search coverage
      - Claude or GPT-4o for deep synthesis
      - Full ResearchReport with executive summary and key findings

    Provider priority: Anthropic > OpenAI > Groq > Gemini
    """

    def __init__(self, preferred_llm: Optional[str] = None):
        self.settings = get_settings()
        self.llm = LLMRouter(
            preferred_provider=preferred_llm or self.settings.preferred_llm,
            max_tokens=self.settings.max_llm_tokens,
        )
        self.search = SearchProviderManager()
        self.analyzer = ContentAnalyzer()
        logger.info(
            f"ResearchAgent ready. LLM: {self.llm.provider_name or 'none (raw mode)'}. "
            f"Search: {self.settings.available_search_providers}"
        )

    async def research(self, task: ResearchTask) -> ResearchReport:
        """
        Main entry point. Returns a ResearchReport.

        Steps:
          1. Concurrent search across all configured providers
          2. Analyze top N sources (fetch content + AI analysis)
          3. Synthesize findings into a structured report via LLM
        """
        logger.info(f"Researching: '{task.topic}' | depth={task.depth} | max={task.max_results}")

        # 1. Search
        query = f"{task.topic} {' '.join(task.keywords)}"
        search_results = await self.search.search(query, max_results=task.max_results)

        if not search_results:
            logger.warning("No search results found.")
            return self._empty_report(task)

        # 2. Analyze sources (limit analysis depth by task.depth)
        analyze_count = {"quick": 3, "standard": 5, "deep": len(search_results)}.get(task.depth, 5)
        sources = await self._analyze_sources(search_results[:analyze_count])

        # 3. Synthesize
        if self.llm.available:
            return await self._synthesize(task, sources)
        else:
            return self._raw_report(task, sources)

    async def _analyze_sources(self, results: List[SearchResult]) -> List[SourceSummary]:
        """Analyze each search result concurrently."""
        tasks = [self._analyze_one(r) for r in results]
        summaries = await asyncio.gather(*tasks, return_exceptions=True)
        return [s for s in summaries if isinstance(s, SourceSummary)]

    async def _analyze_one(self, result: SearchResult) -> SourceSummary:
        """Analyze a single search result."""
        try:
            analysis = await self.analyzer.analyze_url(result.url)
            return SourceSummary(
                title=result.title,
                url=result.url,
                summary=analysis.get("summary", result.snippet),
                key_insights=analysis.get("key_insights", [result.snippet]),
                relevance_score=result.relevance_score,
                authority_score=result.authority_score,
                sentiment=analysis.get("sentiment", "neutral"),
                content_type=analysis.get("content_type", "article"),
            )
        except Exception as e:
            logger.warning(f"Analysis failed for {result.url}: {e}")
            return SourceSummary(
                title=result.title,
                url=result.url,
                summary=result.snippet,
                key_insights=[result.snippet],
                relevance_score=result.relevance_score,
                authority_score=result.authority_score,
                sentiment="neutral",
                content_type="article",
            )

    async def _synthesize(self, task: ResearchTask, sources: List[SourceSummary]) -> ResearchReport:
        """Use the LLM to synthesize sources into a structured report."""
        sources_text = "\n\n".join([
            f"Source {i+1}: {s.title}\nURL: {s.url}\nSummary: {s.summary}\n"
            f"Key Insights: {'; '.join(s.key_insights)}\n"
            f"Relevance: {s.relevance_score:.1f} | Authority: {s.authority_score:.1f}"
            for i, s in enumerate(sources)
        ])

        focus_context = ""
        if task.focus_areas:
            focus_context = f"\nFocus specifically on: {', '.join(task.focus_areas)}"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert research analyst. You synthesize information from "
                    "multiple sources into structured, actionable research reports. "
                    "Be concise, specific, and cite sources by their index number."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Topic: {task.topic}\n"
                    f"Keywords: {', '.join(task.keywords)}{focus_context}\n\n"
                    f"Sources:\n{sources_text}\n\n"
                    "Produce a JSON research report with exactly these keys:\n"
                    "{\n"
                    '  "executive_summary": "2-3 sentence synthesis of all findings",\n'
                    '  "key_findings": ["finding 1", "finding 2", ... (5-7 items)"],\n'
                    '  "confidence": 0.85\n'
                    "}\n\n"
                    "Rules: be specific, reference sources by index, no filler phrases. "
                    "Confidence should reflect source quality and agreement (0.0-1.0). "
                    "Return only valid JSON."
                ),
            },
        ]

        raw = await asyncio.to_thread(self.llm.call, messages)

        try:
            # Strip markdown code fences if present
            clean = re.sub(r"```(?:json)?\n?", "", raw or "").strip().rstrip("`")
            data = json.loads(clean)
            return ResearchReport(
                topic=task.topic,
                executive_summary=data.get("executive_summary", ""),
                key_findings=data.get("key_findings", []),
                sources=sources,
                confidence=float(data.get("confidence", 0.7)),
                generated_at=datetime.now(),
                llm_provider=self.llm.provider_name,
                search_providers=self.settings.available_search_providers,
            )
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse LLM synthesis response: {e}")
            # Return a partial report with the raw LLM text as summary
            return ResearchReport(
                topic=task.topic,
                executive_summary=raw or "Synthesis failed — see raw sources below.",
                key_findings=[],
                sources=sources,
                confidence=0.5,
                generated_at=datetime.now(),
                llm_provider=self.llm.provider_name,
                search_providers=self.settings.available_search_providers,
            )

    def _raw_report(self, task: ResearchTask, sources: List[SourceSummary]) -> ResearchReport:
        """No LLM available — return ranked sources without synthesis."""
        return ResearchReport(
            topic=task.topic,
            executive_summary=(
                f"Raw results for '{task.topic}'. "
                "Set SARVAM_API_KEY, GROQ_API_KEY (free), or ANTHROPIC_API_KEY for synthesized reports."
            ),
            key_findings=[s.summary for s in sources[:5] if s.summary],
            sources=sources,
            confidence=0.5,
            generated_at=datetime.now(),
            llm_provider=None,
            search_providers=self.settings.available_search_providers,
            raw_mode=True,
        )

    def _empty_report(self, task: ResearchTask) -> ResearchReport:
        return ResearchReport(
            topic=task.topic,
            executive_summary="No results found. Try different keywords.",
            key_findings=[],
            sources=[],
            confidence=0.0,
            generated_at=datetime.now(),
            llm_provider=None,
            search_providers=self.settings.available_search_providers,
        )

    def status(self) -> Dict[str, Any]:
        return {
            "llm": self.llm.status(),
            "search_providers": self.settings.available_search_providers,
            "ready": True,
        }
