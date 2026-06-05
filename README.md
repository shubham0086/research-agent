# Research Agent

> Multi-provider web research with LangChain, concurrent search, and automatic fallback.

Built from production code inside Agency OS. The research pipeline that powers the second agent in a 6-agent content production system.

## What It Does

Takes a topic + keywords. Searches across SerpAPI, Tavily, and Brave Search concurrently. Deduplicates and ranks results by relevance and authority. Returns structured research with key insights per source.

If no paid search keys are configured, falls back to DuckDuckGo at no cost.

## Architecture

- `src/search/providers.py` — four providers (SerpAPI, Tavily, Brave, DuckDuckGo fallback), each with rate-limit tracking. SearchProviderManager runs them concurrently via asyncio.gather and deduplicates results.
- `src/content/analyzer.py` — fetches and parses URLs, runs AI analysis (key insights, sentiment, quality score). Falls back gracefully if LLM unavailable.
- `src/agents/research_agent.py` — LangChain agent with two tools: web_search and analyze_content. Runs max 5 iterations. Parses structured output from agent response.

## Quick Start

```bash
git clone https://github.com/shubham0086/research-agent
cd research-agent
pip install -r requirements.txt

cp .env.example .env
# edit .env — at minimum set OPENAI_API_KEY
# leave all keys blank to run in free DuckDuckGo fallback mode

python demo/run.py
```

Run tests:

```bash
pytest tests/ -v
```

## Environment Variables

| Variable | Required | Notes |
|---|---|---|
| `OPENAI_API_KEY` | No | Enables AI analysis and the full agent loop |
| `SERP_API_KEY` | No | SerpAPI for Google results |
| `TAVILY_API_KEY` | No | Tavily AI-curated search |
| `BRAVE_SEARCH_API_KEY` | No | Brave Search |

With no keys set: DuckDuckGo fallback + no AI analysis. With only `OPENAI_API_KEY`: AI analysis but DuckDuckGo search. Paid search keys are fully optional.

## What Agency OS Builds On Top

This agent is the research stage of a 6-agent pipeline. See [Agency OS](https://github.com/shubham0086/agency-os) for how Research -> Content Strategist -> Creator -> QA -> Formatter are chained together with disk-persistent DAG orchestration.
