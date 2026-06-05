"""
Demo: Research Agent

Works with any combination of API keys you have.
No keys at all: uses DuckDuckGo + returns raw results.
Free keys (Groq/Gemini): full synthesized report at no cost.
Paid keys (Anthropic/OpenAI): highest quality research.

Usage:
    pip install -r requirements.txt
    cp .env.example .env          # fill in at least OPENAI_API_KEY
    python demo/run.py

With no API keys set the agent falls back to DuckDuckGo search
and skips AI analysis — useful for a quick smoke test.
"""
import asyncio
import logging
import sys
import os

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
)

from src.agents.research_agent import ResearchAgent, ResearchTask


async def main():
    topic = "LangChain multi-agent systems"
    keywords = ["LangChain", "agent", "tool use", "LLM orchestration"]

    print("\n" + "=" * 60)
    print(f"Research Agent Demo")
    print(f"Topic   : {topic}")
    print(f"Keywords: {', '.join(keywords)}")
    print("=" * 60 + "\n")

    agent = ResearchAgent()

    # Print what providers and tools are available
    status = agent.get_agent_status()
    print("Agent status:")
    print(f"  LLM model        : {status['llm_model'] or 'Not configured (fallback mode)'}")
    print(f"  Search providers : {', '.join(status['search_providers'])}")
    print(f"  Tools            : {', '.join(status['available_tools'])}")
    print(f"  Agent available  : {status['agent_available']}")
    print()

    task = ResearchTask(
        topic=topic,
        keywords=keywords,
        max_results=3,
        focus_areas=["practical examples", "production use cases"],
        content_types=["tutorials", "case studies", "technical articles"]
    )

    print("Running research...\n")
    results = await agent.research_topic(task)

    if not results:
        print("No results returned. Check your env vars and API keys.")
        return

    print(f"\nFound {len(results)} result(s):\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r.title}")
        print(f"    URL              : {r.url}")
        print(f"    Relevance score  : {r.relevance_score}")
        print(f"    Source authority : {r.source_authority}")
        print(f"    Sentiment        : {r.sentiment}")
        print(f"    Content type     : {r.content_type}")
        print(f"    Summary          : {r.content_summary[:120]}...")
        print(f"    Key insights     :")
        for insight in r.key_insights:
            print(f"      - {insight}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
