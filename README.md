# Research Agent

> **IMPORTANT**: This repository contains real, production-ready, battle-tested code extracted directly from active commercial systems (like Agency OS or Founder Growth OS), rather than simplified mock learning artifacts.
>
> For project walkthroughs, architecture flowcharts, and system context, visit the live landing page: [my-portfolio-github-io-beta-five.vercel.app/projects/research-agent.html](https://my-portfolio-github-io-beta-five.vercel.app/projects/research-agent.html)

> Multi-provider research pipeline. Search the web, analyze sources, synthesize findings: with your LLM of choice.

Extracted from Agency OS: the research stage of a 6-agent content production system. No LangChain. Works free out of the box. Plug in Anthropic, OpenAI, Groq, or Gemini for full synthesis.

---

## What It Does

Input: topic + keywords.
Output: a structured `ResearchReport` with:
- **Executive summary**: 2-3 sentence synthesis across all sources
- **Key findings**: 5-7 actionable insights distilled by the LLM
- **Analyzed sources**: relevance score, authority score, sentiment, content type per source
- **Confidence score**: how well sources agree on the topic

---

## LLM Providers

Set any key. The router auto-detects and uses the best available.

| Provider | Key | Tier | Notes |
|----------|-----|------|-------|
| Anthropic Claude | `ANTHROPIC_API_KEY` | Paid | Best synthesis. `claude-sonnet-4-6` default. |
| OpenAI | `OPENAI_API_KEY` | Paid | Strong. `gpt-4o-mini` default. Set `OPENAI_MODEL=gpt-4o` for best quality. |
| Groq | `GROQ_API_KEY` | **Free tier** | Fast. `llama-3.1-8b-instant`. Good for quick research. |
| Google Gemini | `GEMINI_API_KEY` | **Free tier** | `gemini-1.5-flash`. Solid free option. |
| None |: | Free | Returns raw ranked results without synthesis. |

Priority order: Anthropic > OpenAI > Groq > Gemini. Override with `PREFERRED_LLM=Groq`.

---

## Search Providers

| Provider | Key | Notes |
|----------|-----|-------|
| SerpAPI | `SERP_API_KEY` | Real Google results. Best coverage. |
| Tavily | `TAVILY_API_KEY` | AI-curated. High relevance. |
| Brave Search | `BRAVE_SEARCH_API_KEY` | Privacy-focused. Good fallback. |
| DuckDuckGo |: | No key needed. Auto-used when no paid key is set. |

Multiple providers run concurrently. Results are deduplicated and ranked.

---

## Quick Start

```bash
git clone https://github.com/shubham0086/research-agent
cd research-agent
pip install -r requirements.txt
cp .env.example .env

# Zero-key mode (DuckDuckGo + raw results: works immediately):
python demo/run.py

# Free synthesis via Groq (console.groq.com):
echo "GROQ_API_KEY=gsk_yourkey" >> .env
python demo/run.py

# Best quality via Anthropic:
echo "ANTHROPIC_API_KEY=sk-ant-yourkey" >> .env
python demo/run.py --topic "autonomous agent memory systems" --depth deep
```

---

## Architecture

```
ResearchAgent.research(task)
    ├── SearchProviderManager     (concurrent across all providers)
    ├── ContentAnalyzer × N       (concurrent URL fetch + analysis)
    └── LLMRouter.call()          (Anthropic > OpenAI > Groq > Gemini)
              → ResearchReport
```

- `src/llm/router.py`: provider-agnostic LLM router, auto-cascade on failure
- `src/search/providers.py`: four providers, concurrent search, URL dedup + ranking
- `src/content/analyzer.py`: URL fetch, noise removal, per-source AI analysis
- `src/agents/research_agent.py`: pipeline orchestration and report synthesis

---

## Tests

```bash
pytest tests/ -v  # 9 tests
```

Covers: LLM router detection + cascade, search deduplication, raw mode, synthesized report structure.

---

## Where This Fits in Agency OS

Stage 2 of the 6-agent pipeline:
```
Brief Intake → [Research Agent] → Content Strategist → Creator → QA → Formatter
```

See [Agency OS](https://github.com/shubham0086/agency-os) for the full orchestration layer. The SaaS engine that runs the pipeline: [Agentic SaaS Boilerplate](https://github.com/shubham0086/agentic-saas-boilerplate).
