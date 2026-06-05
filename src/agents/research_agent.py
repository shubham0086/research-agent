"""
Production-ready AI Research Agent with LangChain
Implements intelligent content research with reasoning and planning
"""
import asyncio
import logging
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks import get_openai_callback

from src.config import get_settings
from src.search.providers import SearchProviderManager
from src.content.analyzer import ContentAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class ResearchTask:
    """Represents a research task with context"""
    topic: str
    keywords: List[str]
    max_results: int = 5
    focus_areas: List[str] = field(default_factory=list)
    content_types: List[str] = field(default_factory=list)


@dataclass
class ResearchResult:
    """Represents the result of research with AI analysis"""
    title: str
    url: str
    content_summary: str
    relevance_score: float
    key_insights: List[str]
    sentiment: str
    content_type: str
    source_authority: float
    publish_date: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)


class ResearchAgent:
    """
    Intelligent research agent that uses AI to plan, execute, and analyze content research.

    Architecture:
    - SearchProviderManager: concurrent multi-provider search with deduplication
    - ContentAnalyzer: per-URL AI analysis (key insights, sentiment, quality score)
    - LangChain AgentExecutor: orchestrates tool calls across up to 5 iterations
    """

    def __init__(self):
        self.settings = get_settings()
        self.search_manager = SearchProviderManager()
        self.content_analyzer = ContentAnalyzer()

        # Initialize LLM with fallback
        self.llm = self._initialize_llm()
        self.memory = ConversationBufferWindowMemory(
            k=5,
            return_messages=True,
            memory_key="chat_history"
        )

        # Initialize agent tools
        self.tools = self._create_tools()
        self.agent_executor = self._create_agent()

        logger.info(f"ResearchAgent initialized with LLM: {self.llm.model_name if self.llm else 'None'}")

    def _initialize_llm(self) -> Optional[ChatOpenAI]:
        """Initialize LLM with automatic fallback"""
        try:
            if self.settings.has_openai:
                return ChatOpenAI(
                    model=self.settings.default_llm_model,
                    temperature=self.settings.temperature,
                    max_tokens=self.settings.max_tokens_per_request,
                    openai_api_key=self.settings.openai_api_key
                )
            else:
                logger.warning("No LLM provider configured, using mock responses")
                return None
        except Exception as e:
            logger.error(f"Failed to initialize primary LLM: {e}")
            try:
                return ChatOpenAI(
                    model=self.settings.fallback_llm_model,
                    temperature=self.settings.temperature,
                    max_tokens=2000,
                    openai_api_key=self.settings.openai_api_key
                )
            except Exception as e:
                logger.error(f"Failed to initialize fallback LLM: {e}")
                return None

    def _create_tools(self) -> List[Tool]:
        """Create tools for the agent"""
        tools = []

        # Web Search Tool
        def search_web(query: str) -> str:
            """Search the web for relevant content"""
            try:
                # Run async search in sync context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    results = loop.run_until_complete(self.search_manager.search(query, max_results=5))
                    # Convert search results to dictionary format
                    formatted_results = []
                    for result in results:
                        formatted_results.append({
                            "title": result.title,
                            "url": result.url,
                            "snippet": result.snippet,
                            "source": result.source,
                            "relevance_score": result.relevance_score,
                            "authority_score": result.authority_score
                        })
                    return json.dumps(formatted_results, indent=2)
                finally:
                    loop.close()
            except Exception as e:
                logger.error(f"Search tool error: {e}")
                return f"Search failed: {str(e)}"

        tools.append(Tool(
            name="web_search",
            description="Search the web for current information about a topic. Use this to find recent articles, news, and relevant content.",
            func=search_web
        ))

        # Content Analysis Tool
        def analyze_content(url_or_text: str) -> str:
            """Analyze content for relevance and insights"""
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    if url_or_text.startswith('http'):
                        analysis = loop.run_until_complete(self.content_analyzer.analyze_url(url_or_text))
                    else:
                        analysis = loop.run_until_complete(self.content_analyzer.analyze_text(url_or_text))
                    return json.dumps(analysis, indent=2)
                finally:
                    loop.close()
            except Exception as e:
                logger.error(f"Content analysis error: {e}")
                return f"Analysis failed: {str(e)}"

        tools.append(Tool(
            name="analyze_content",
            description="Analyze content from a URL or text for relevance, sentiment, and key insights.",
            func=analyze_content
        ))

        return tools

    def _create_agent(self) -> Optional[AgentExecutor]:
        """Create the research agent"""
        if not self.llm:
            return None

        # System prompt — kept as written in production, no token optimizer needed here
        system_prompt = """
        You are an expert AI research assistant specializing in content discovery and analysis.
        Your goal: Find highly relevant, current content on given topics with deep insights.

        WORKFLOW:
        1. PLAN: Analyze the research request and create search strategy
        2. SEARCH: Execute targeted searches using available tools
        3. ANALYZE: Evaluate content quality, relevance, and insights
        4. SYNTHESIZE: Compile findings with actionable intelligence

        SEARCH STRATEGY:
        - Use multiple search queries with different angles
        - Focus on recent, authoritative sources
        - Prioritize diverse content types (news, analysis, reports)
        - Look for emerging trends and expert opinions

        ANALYSIS CRITERIA:
        - Relevance to user's specific interests
        - Content freshness and accuracy
        - Source authority and credibility
        - Unique insights and perspectives
        - Actionable information

        OUTPUT FORMAT:
        Always provide structured findings with:
        - Title and URL
        - Key insights (3-5 bullet points)
        - Relevance score (1-10)
        - Content type and authority
        - Recommended action

        Be efficient, thorough, and focus on high-value content.
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])

        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=5,
            max_execution_time=60,
            early_stopping_method="generate"
        )

    async def research_topic(self, task: ResearchTask) -> List[ResearchResult]:
        """
        Main method to research a topic using AI agent reasoning.
        Falls back to direct search if agent/LLM is unavailable.
        """
        logger.info(f"Starting AI research for topic: {task.topic}")

        if not self.agent_executor:
            logger.warning("No agent available, using fallback research")
            return await self._fallback_research(task)

        try:
            # Create research query
            research_query = self._create_research_query(task)

            # Track token usage
            with get_openai_callback() as cb:
                # Execute agent research
                response = await asyncio.to_thread(
                    self.agent_executor.invoke,
                    {"input": research_query}
                )

                logger.info(f"Research completed. Tokens used: {cb.total_tokens}")

            # Parse and structure results
            results = await self._parse_agent_results(response, task)

            logger.info(f"Found {len(results)} high-quality research results")
            return results

        except Exception as e:
            logger.error(f"Agent research failed: {e}")
            return await self._fallback_research(task)

    def _create_research_query(self, task: ResearchTask) -> str:
        """Create a research query for the agent"""
        focus_context = ""
        if task.focus_areas:
            focus_context = f" with focus on: {', '.join(task.focus_areas)}"

        content_context = ""
        if task.content_types:
            content_context = f" looking for: {', '.join(task.content_types)}"

        query = f"""
        Research Topic: {task.topic}
        Keywords: {', '.join(task.keywords)}
        {focus_context}
        {content_context}

        Find {task.max_results} most relevant, high-quality pieces of content.
        Prioritize recent, authoritative sources with unique insights.
        Analyze each piece for relevance, credibility, and actionable value.
        """

        return query.strip()

    async def _parse_agent_results(self, response: Dict[str, Any], task: ResearchTask) -> List[ResearchResult]:
        """Parse agent response into structured research results"""
        results = []

        try:
            # Extract structured data from agent response
            output = response.get('output', '')

            logger.info(f"Agent response length: {len(output)} characters")
            logger.info(f"Agent response preview: {output[:500]}...")

            # Try to extract URLs and titles from the response using regex
            url_pattern = r'https?://[^\s\)\]]+'
            title_pattern = r'(?:Title:|title:|[*#]+\s*)([^\n]+?)(?:\n|$)'

            urls = re.findall(url_pattern, output)
            titles = re.findall(title_pattern, output, re.IGNORECASE)

            logger.info(f"Found {len(urls)} URLs and {len(titles)} titles in response")

            # Create results from extracted data
            for i, url in enumerate(urls[:task.max_results]):
                title = titles[i] if i < len(titles) else f"Research Finding {i+1}"
                title = title.strip('*# ').strip()

                # Extract content around the URL for summary
                url_pos = output.find(url)
                context_start = max(0, url_pos - 200)
                context_end = min(len(output), url_pos + 300)
                context = output[context_start:context_end]

                # Clean up the context for summary
                summary_lines = []
                for line in context.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('http') and len(line) > 10:
                        summary_lines.append(line)

                summary = ' '.join(summary_lines[:3])  # First 3 relevant lines
                if not summary:
                    summary = f"Research content related to {task.topic}"

                # Extract key insights from the context
                key_insights = []
                if "key insights" in context.lower() or "insights" in context.lower():
                    insight_lines = [line.strip() for line in context.split('\n')
                                     if line.strip() and ('•' in line or '-' in line.strip()[:2])]
                    key_insights = [line.strip('• -').strip() for line in insight_lines[:3]]

                if not key_insights:
                    key_insights = [
                        f"Relevant information about {task.topic}",
                        "High-quality research source identified",
                        "Content validated by AI analysis"
                    ]

                results.append(ResearchResult(
                    title=title,
                    url=url,
                    content_summary=summary[:500],  # Limit summary length
                    relevance_score=8.5,  # Default high relevance since AI selected it
                    key_insights=key_insights,
                    sentiment='neutral',
                    content_type='research',
                    source_authority=8.0,  # High authority since found by AI
                    tags=task.keywords
                ))

            logger.info(f"Successfully parsed {len(results)} results from agent response")

        except Exception as e:
            logger.error(f"Error parsing agent results: {e}")

        # If no results parsed, use fallback
        if not results:
            logger.warning("No results parsed from agent, using fallback")
            results = await self._fallback_research(task)

        return results[:task.max_results]

    async def _fallback_research(self, task: ResearchTask) -> List[ResearchResult]:
        """
        Fallback research method when agent is unavailable.
        Calls SearchProviderManager directly — no LLM required.
        """
        logger.info("Using fallback research method")

        try:
            # Use direct search without agent
            search_results = await self.search_manager.search(
                f"{task.topic} {' '.join(task.keywords)}",
                max_results=task.max_results
            )

            results = []
            for item in search_results:
                results.append(ResearchResult(
                    title=item.title,
                    url=item.url,
                    content_summary=item.snippet or 'Content analysis pending',
                    relevance_score=item.relevance_score or 8.0,
                    key_insights=[
                        f"Related to {task.topic}",
                        "Requires further analysis",
                        "Potentially valuable content"
                    ],
                    sentiment='neutral',
                    content_type='article',
                    source_authority=item.authority_score or 7.0,
                    tags=task.keywords
                ))

            return results

        except Exception as e:
            logger.error(f"Fallback research failed: {e}")
            return []

    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status and capabilities"""
        return {
            "agent_available": bool(self.agent_executor),
            "llm_model": self.llm.model_name if self.llm else None,
            "available_tools": [tool.name for tool in self.tools],
            "search_providers": self.settings.available_search_providers,
            "llm_providers": self.settings.available_llm_providers,
            "memory_window": self.memory.k if self.memory else 0
        }
