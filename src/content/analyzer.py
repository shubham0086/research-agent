"""
AI-powered content analysis system
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import json
from urllib.parse import urlparse

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

from src.config import get_settings

logger = logging.getLogger(__name__)


class ContentAnalyzer:
    """AI-powered content analysis and insights generation"""

    def __init__(self):
        self.settings = get_settings()
        self.llm = self._initialize_llm()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def _initialize_llm(self) -> Optional[ChatOpenAI]:
        """Initialize LLM for content analysis"""
        try:
            if self.settings.has_openai:
                return ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.3,
                    max_tokens=1000,
                    openai_api_key=self.settings.openai_api_key
                )
        except Exception as e:
            logger.error(f"Failed to initialize LLM for content analysis: {e}")
        return None

    async def analyze_url(self, url: str) -> Dict[str, Any]:
        """Analyze content from a URL"""
        try:
            # Extract content from URL
            content = await self._extract_content(url)
            if not content:
                return self._default_analysis(url)

            # Analyze with AI
            analysis = await self._ai_analyze_content(content, url)
            return analysis

        except Exception as e:
            logger.error(f"URL analysis failed for {url}: {e}")
            return self._default_analysis(url)

    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze raw text content"""
        try:
            if not text or len(text.strip()) < 50:
                return self._default_analysis()

            analysis = await self._ai_analyze_content(text)
            return analysis

        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            return self._default_analysis()

    async def _extract_content(self, url: str) -> Optional[str]:
        """Extract clean content from URL"""
        try:
            response = await asyncio.to_thread(
                self.session.get,
                url,
                timeout=10
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()

            # Extract main content — try semantic selectors first
            content_selectors = [
                'article', 'main', '.content', '#content',
                '.post-content', '.entry-content', '.article-content'
            ]

            content = ""
            for selector in content_selectors:
                elem = soup.select_one(selector)
                if elem:
                    content = elem.get_text(strip=True)
                    break

            if not content:
                content = soup.get_text(strip=True)

            # Clean and limit content
            content = ' '.join(content.split())
            return content[:5000]  # Limit for token efficiency

        except Exception as e:
            logger.error(f"Content extraction failed for {url}: {e}")
            return None

    async def _ai_analyze_content(self, content: str, url: Optional[str] = None) -> Dict[str, Any]:
        """Use AI to analyze content for insights"""
        if not self.llm:
            return self._default_analysis(url)

        try:
            analysis_prompt = f"""
            Analyze this content for key insights, relevance, and sentiment:

            Content: {content[:2000]}...

            Provide analysis as JSON:
            {{
                "key_insights": ["insight 1", "insight 2", "insight 3"],
                "sentiment": "positive/negative/neutral",
                "relevance_score": 8.5,
                "content_type": "news/analysis/tutorial/report",
                "main_topics": ["topic1", "topic2"],
                "quality_score": 7.5,
                "summary": "Brief 1-sentence summary"
            }}

            Return only valid JSON.
            """

            response = await asyncio.to_thread(
                self.llm.invoke,
                [HumanMessage(content=analysis_prompt)]
            )

            try:
                analysis = json.loads(response.content)

                # Add additional metadata
                analysis.update({
                    "analyzed_at": datetime.now().isoformat(),
                    "content_length": len(content),
                    "url": url,
                    "domain": urlparse(url).netloc if url else None
                })

                return analysis

            except json.JSONDecodeError:
                logger.warning("Failed to parse AI analysis response")
                return self._default_analysis(url)

        except Exception as e:
            logger.error(f"AI content analysis failed: {e}")
            return self._default_analysis(url)

    def _default_analysis(self, url: Optional[str] = None) -> Dict[str, Any]:
        """Default analysis when AI analysis fails or LLM is not configured"""
        return {
            "key_insights": [
                "Content requires manual review",
                "Analysis system temporarily unavailable",
                "May contain valuable information"
            ],
            "sentiment": "neutral",
            "relevance_score": 7.0,
            "content_type": "article",
            "main_topics": ["general"],
            "quality_score": 6.0,
            "summary": "Content analysis pending",
            "analyzed_at": datetime.now().isoformat(),
            "content_length": 0,
            "url": url,
            "domain": urlparse(url).netloc if url else None,
            "fallback": True
        }
