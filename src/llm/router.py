"""
Multi-provider LLM router — extracted from production ace-engine/provider_router.py.

Key design:
  - Most providers (MiniMax, OpenRouter, NVIDIA NIM, SiliconFlow, DeepSeek, OpenAI)
    use the OpenAI-compatible /v1/chat/completions API — one HTTP call handles all.
  - Gemini, Anthropic, and Ollama have their own native API shapes.
  - Task-class chains: each task type (research, code, content, simple) has its own
    provider priority order. Research routes through content-optimized providers first.
  - Session-level circuit breaker: once a provider exhausts all its models in a session,
    it is skipped for the remainder — no wasted budget retrying known-down providers.
  - Cost tracking: each call returns provider, model, estimated cost, and latency.

Provider tiers:
  Free:   OpenRouter (:free suffix models), MiniMax (OpenCode Zen), Groq, Gemini Flash, Ollama
  Paid:   Anthropic, OpenAI, NVIDIA NIM, SiliconFlow, DeepSeek, Bedrock

Set the env vars for any providers you have. Router auto-detects and cascades.
"""
import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional, Set

import httpx

logger = logging.getLogger(__name__)

# Session-level circuit breaker: providers that exhausted all models this session
_down_providers: Set[str] = set()

ATTEMPTS_PER_PAIR = 2
RETRY_DELAY = 2.0


# ---------------------------------------------------------------------------
# Model chains — ordered: best/cheapest first, fallback last
# ---------------------------------------------------------------------------

def _build_model_chains() -> Dict[str, List[str]]:
    return {
        # MiniMax via OpenCode Zen (free)
        "minimax": [
            os.getenv("MINIMAX_MODEL", "MiniMax-Text-01"),
        ],
        # OpenRouter — append :free for zero-cost models
        "openrouter": [
            os.getenv("OPENROUTER_MODEL", "qwen/qwen3-coder:free"),
            "meta-llama/llama-3.3-70b-instruct:free",
            "mistralai/mistral-7b-instruct:free",
        ],
        # NVIDIA NIM
        "nim": [
            os.getenv("NIM_MODEL", "meta/llama-3.1-70b-instruct"),
            "nvidia/nemotron-nano-12b-v2",
        ],
        # SiliconFlow (Chinese market, fast and cheap)
        "siliconflow": [
            os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen3-8B"),
            "deepseek-ai/DeepSeek-V3",
        ],
        # DeepSeek
        "deepseek": [
            os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            "deepseek-reasoner",
        ],
        # Groq (free tier, fast)
        "groq": [
            os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        # Google Gemini
        "gemini": [
            os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ],
        # OpenAI
        "openai": [
            os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "gpt-4o",
        ],
        # Anthropic
        "anthropic": [
            os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            "claude-haiku-4-5",
        ],
        # Ollama (local, always available if running)
        "ollama": [
            os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b"),
            "llama3.2:3b",
        ],
    }


# ---------------------------------------------------------------------------
# Provider chains by task class
# Content/research: free providers first, then paid, then local
# ---------------------------------------------------------------------------

PROVIDER_CHAINS: Dict[str, List[str]] = {
    "research": ["minimax", "openrouter", "groq", "gemini", "deepseek", "siliconflow", "nim", "openai", "anthropic", "ollama"],
    "content":  ["minimax", "gemini", "openrouter", "groq", "openai", "anthropic", "siliconflow", "nim", "ollama"],
    "code":     ["deepseek", "siliconflow", "nim", "openrouter", "openai", "gemini", "anthropic", "ollama"],
    "simple":   ["minimax", "groq", "openrouter", "gemini", "openai", "ollama"],
}


# ---------------------------------------------------------------------------
# Endpoint + auth config for OpenAI-compatible providers
# ---------------------------------------------------------------------------

_OAI_COMPAT: Dict[str, str] = {
    "minimax":    os.getenv("MINIMAX_BASE_URL", "https://api.minimax.chat/v1") + "/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "nim":        "https://integrate.api.nvidia.com/v1/chat/completions",
    "siliconflow":"https://api.siliconflow.cn/v1/chat/completions",
    "deepseek":   "https://api.deepseek.com/v1/chat/completions",
    "groq":       "https://api.groq.com/openai/v1/chat/completions",
    "openai":     "https://api.openai.com/v1/chat/completions",
}

_ENV_KEYS: Dict[str, str] = {
    "minimax":    "MINIMAX_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "nim":        "NIM_API_KEY",
    "siliconflow":"SILICONFLOW_API_KEY",
    "deepseek":   "DEEPSEEK_API_KEY",
    "groq":       "GROQ_API_KEY",
    "openai":     "OPENAI_API_KEY",
    "anthropic":  "ANTHROPIC_API_KEY",
    "gemini":     "GEMINI_API_KEY",
    # ollama needs no key
}

PROVIDER_TIMEOUTS: Dict[str, int] = {
    "minimax":    60,
    "openrouter": 45,
    "nim":        45,
    "siliconflow":30,
    "deepseek":   45,
    "groq":       30,
    "gemini":     45,
    "openai":     30,
    "anthropic":  60,
    "ollama":     120,
}

# Cost per 1M tokens (input, output) — 0.0 = free tier
MODEL_PRICING: Dict[str, Dict[str, Dict[str, float]]] = {
    "openai":    {"gpt-4o-mini": {"in": 0.15, "out": 0.60}, "gpt-4o": {"in": 2.50, "out": 10.00}},
    "anthropic": {"claude-sonnet-4-6": {"in": 3.00, "out": 15.00}, "claude-haiku-4-5": {"in": 0.80, "out": 4.00}},
    "gemini":    {"gemini-2.0-flash": {"in": 0.075, "out": 0.30}, "gemini-1.5-flash": {"in": 0.075, "out": 0.30}},
    "deepseek":  {"deepseek-chat": {"in": 0.14, "out": 0.28}},
    "groq":      {"llama-3.1-8b-instant": {"in": 0.0, "out": 0.0}},  # free tier
    "minimax":   {"MiniMax-Text-01": {"in": 0.0, "out": 0.0}},        # free via OpenCode Zen
    "openrouter":{"qwen/qwen3-coder:free": {"in": 0.0, "out": 0.0}},  # free tier
    "ollama":    {"qwen2.5-coder:7b": {"in": 0.0, "out": 0.0}},       # local, free
}


# ---------------------------------------------------------------------------
# Provider availability check
# ---------------------------------------------------------------------------

def _available(provider: str) -> bool:
    if provider in _down_providers:
        return False
    if provider == "ollama":
        return True  # always available if running locally
    env_key = _ENV_KEYS.get(provider)
    if not env_key:
        return False
    val = os.getenv(env_key, "")
    return bool(val and val.strip())


# ---------------------------------------------------------------------------
# HTTP dispatch functions
# ---------------------------------------------------------------------------

async def _call_oai_compat(
    provider: str, model: str, messages: List[Dict], temperature: float, max_tokens: int
) -> tuple[str, int, int]:
    """One function handles all OpenAI-compatible providers."""
    key = os.getenv(_ENV_KEYS[provider], "")
    endpoint = _OAI_COMPAT[provider]
    timeout = PROVIDER_TIMEOUTS.get(provider, 30)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }
    if provider == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/shubham0086/research-agent"
        headers["X-Title"] = "Research-Agent"

    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(endpoint, headers=headers, json=payload)

    if not resp.is_success:
        raise RuntimeError(f"{provider.upper()} HTTP {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    if data.get("error"):
        raise RuntimeError(str(data["error"]))

    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
    if not content:
        raise RuntimeError(f"{provider} returned empty response")

    usage = data.get("usage", {})
    return content, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


async def _call_anthropic(
    model: str, messages: List[Dict], temperature: float, max_tokens: int
) -> tuple[str, int, int]:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    system = "\n".join(m["content"] for m in messages if m.get("role") == "system")
    conv = [m for m in messages if m.get("role") != "system"]

    payload: Dict[str, Any] = {"model": model, "max_tokens": max_tokens, "temperature": temperature, "messages": conv}
    if system:
        payload["system"] = system

    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
    timeout = PROVIDER_TIMEOUTS.get("anthropic", 60)

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)

    if not resp.is_success:
        raise RuntimeError(f"Anthropic HTTP {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    content = (data.get("content") or [{}])[0].get("text", "").strip()
    if not content:
        raise RuntimeError("Anthropic returned empty response")

    usage = data.get("usage", {})
    return content, usage.get("input_tokens", 0), usage.get("output_tokens", 0)


async def _call_gemini(
    model: str, messages: List[Dict], temperature: float, max_tokens: int
) -> tuple[str, int, int]:
    key = os.getenv("GEMINI_API_KEY", "")
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    user_parts = [{"text": m["content"]} for m in messages if m.get("role") != "system"]

    is_thinking = "2.5" in model
    body: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": user_parts}],
        "generationConfig": {
            "maxOutputTokens": 8192 if is_thinking else max_tokens,
            "temperature": 1.0 if is_thinking else temperature,
        },
    }
    if system_parts:
        body["system_instruction"] = {"parts": [{"text": "\n".join(system_parts)}]}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    timeout = PROVIDER_TIMEOUTS.get("gemini", 45)

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=body)

    if not resp.is_success:
        raise RuntimeError(f"Gemini HTTP {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    content = (
        (data.get("candidates") or [{}])[0]
        .get("content", {}).get("parts", [{}])[0].get("text", "").strip()
    )
    if not content:
        raise RuntimeError("Gemini returned empty response")

    # Gemini doesn't always return token counts in standard output
    in_tok = len(str(messages)) // 4
    out_tok = len(content) // 4
    return content, in_tok, out_tok


async def _call_ollama(
    model: str, messages: List[Dict], temperature: float
) -> tuple[str, int, int]:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    timeout = PROVIDER_TIMEOUTS.get("ollama", 120)

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{base_url}/api/chat",
            json={"model": model, "messages": messages, "stream": False, "options": {"temperature": temperature}},
        )

    if not resp.is_success:
        raise RuntimeError(f"Ollama HTTP {resp.status_code}")

    data = resp.json()
    content = data.get("message", {}).get("content", "").strip()
    if not content:
        raise RuntimeError("Ollama returned empty response")

    return content, data.get("prompt_eval_count", 0), data.get("eval_count", 0)


async def _dispatch(
    provider: str, model: str, messages: List[Dict], temperature: float, max_tokens: int
) -> tuple[str, int, int]:
    if provider in _OAI_COMPAT:
        return await _call_oai_compat(provider, model, messages, temperature, max_tokens)
    if provider == "anthropic":
        return await _call_anthropic(model, messages, temperature, max_tokens)
    if provider == "gemini":
        return await _call_gemini(model, messages, temperature, max_tokens)
    if provider == "ollama":
        return await _call_ollama(model, messages, temperature)
    raise ValueError(f"Unknown provider: {provider}")


def _estimate_cost(provider: str, model: str, in_tokens: int, out_tokens: int) -> float:
    pricing = MODEL_PRICING.get(provider, {}).get(model)
    if not pricing:
        return 0.0
    return (in_tokens / 1_000_000) * pricing["in"] + (out_tokens / 1_000_000) * pricing["out"]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

async def chat(
    messages: List[Dict[str, str]],
    task_class: str = "research",
    temperature: float = 0.3,
    max_tokens: int = 3000,
    preferred_provider: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Multi-provider fallback chat.

    Returns:
        {"content": str, "provider": str, "model": str, "cost": float, "latency": float}

    Raises RuntimeError if every provider in the chain fails.
    """
    model_chains = _build_model_chains()
    chain = PROVIDER_CHAINS.get(task_class, PROVIDER_CHAINS["research"])

    if preferred_provider:
        chain = [preferred_provider] + [p for p in chain if p != preferred_provider]

    usable = [p for p in chain if _available(p)]
    if not usable:
        raise RuntimeError(
            f"No providers available for task_class='{task_class}'. "
            "Set at least one of: GROQ_API_KEY (free), GEMINI_API_KEY (free), "
            "ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY, or run Ollama locally."
        )

    failures = []
    for provider in usable:
        models = model_chains.get(provider, [])
        provider_succeeded = False

        for model in models:
            for attempt in range(ATTEMPTS_PER_PAIR):
                try:
                    logger.info(f"-> {provider.upper()} / {model}")
                    t0 = time.time()
                    content, in_tok, out_tok = await _dispatch(provider, model, messages, temperature, max_tokens)
                    latency = time.time() - t0
                    cost = _estimate_cost(provider, model, in_tok, out_tok)
                    logger.info(f"OK {provider.upper()} / {model} in {latency:.2f}s (${cost:.5f})")
                    provider_succeeded = True
                    return {"content": content, "provider": provider, "model": model, "cost": cost, "latency": latency}
                except Exception as e:
                    msg = str(e)[:150]
                    failures.append(f"{provider}/{model}: {msg}")
                    logger.warning(f"{provider}/{model} attempt {attempt+1} failed: {msg}")
                    if attempt + 1 < ATTEMPTS_PER_PAIR:
                        await asyncio.sleep(RETRY_DELAY)

        if not provider_succeeded:
            _down_providers.add(provider)
            logger.warning(f"CIRCUIT OPEN: {provider} marked down for this session.")

    raise RuntimeError(f"All providers failed: {' | '.join(failures[-5:])}")


def available_providers() -> List[Dict[str, Any]]:
    """Return all configured providers and their status."""
    return [
        {
            "name": p,
            "configured": _available(p),
            "env_key": _ENV_KEYS.get(p, "none required"),
            "tier": "free" if p in {"minimax", "openrouter", "groq", "ollama"} else "paid",
            "in_chain": {tc: (p in chain) for tc, chain in PROVIDER_CHAINS.items()},
        }
        for p in list(_OAI_COMPAT.keys()) + ["anthropic", "gemini", "ollama"]
    ]


def reset_circuit_breakers() -> None:
    """Reset session circuit breakers — useful after recovering from an outage."""
    _down_providers.clear()


# ---------------------------------------------------------------------------
# Thin sync wrapper kept for backwards compatibility with non-async callers
# ---------------------------------------------------------------------------

class LLMRouter:
    """Sync wrapper around the async chat() function. For backwards compatibility."""

    def __init__(self, preferred_provider: Optional[str] = None, max_tokens: int = 3000):
        self.preferred = preferred_provider
        self.max_tokens = max_tokens

    @property
    def available(self) -> bool:
        return any(_available(p) for chain in PROVIDER_CHAINS.values() for p in chain)

    @property
    def provider_name(self) -> Optional[str]:
        for chain in [PROVIDER_CHAINS["research"]]:
            for p in chain:
                if _available(p):
                    return p
        return None

    @property
    def tier(self) -> str:
        name = self.provider_name
        if not name:
            return "none"
        return "free" if name in {"minimax", "openrouter", "groq", "ollama"} else "paid"

    def call(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> Optional[str]:
        """Synchronous call — runs the async chat() in a new event loop."""
        if not self.available:
            return None
        try:
            result = asyncio.run(chat(
                messages=messages,
                task_class="research",
                preferred_provider=self.preferred,
                max_tokens=self.max_tokens,
            ))
            return result["content"]
        except Exception as e:
            logger.error(f"LLMRouter.call failed: {e}")
            return None

    def status(self) -> Dict[str, Any]:
        return {
            "active_provider": self.provider_name,
            "tier": self.tier,
            "available_providers": available_providers(),
        }
