"""
Multi-provider LLM router. No LangChain required.

Priority cascade:
  1. Anthropic Claude (best reasoning, set ANTHROPIC_API_KEY)
  2. OpenAI GPT-4o (strong, set OPENAI_API_KEY)
  3. Groq (fast + free tier, set GROQ_API_KEY)
  4. Google Gemini (free tier, set GEMINI_API_KEY)
  5. None — agent degrades gracefully to raw search results

Each provider follows the same interface: call(messages) -> str
so you can swap providers without touching any other code.
"""
import logging
import os
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


def _call_anthropic(messages: List[Dict], model: str, max_tokens: int) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Separate system message from conversation messages
    system = ""
    conv = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            conv.append(m)

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=conv,
    )
    return response.content[0].text


def _call_openai(messages: List[Dict], model: str, max_tokens: int) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return response.choices[0].message.content


def _call_groq(messages: List[Dict], model: str, max_tokens: int) -> str:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return response.choices[0].message.content


def _call_gemini(messages: List[Dict], model: str, max_tokens: int) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    # Gemini uses a different message format — flatten to a single prompt
    prompt = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages
    )
    gemini_model = genai.GenerativeModel(model)
    response = gemini_model.generate_content(
        prompt,
        generation_config={"max_output_tokens": max_tokens, "temperature": 0.3},
    )
    return response.text


class LLMRouter:
    """
    Routes LLM calls to the best available provider.

    Tries providers in priority order. Skips any provider whose
    API key is missing. Falls back to None if nothing is configured.

    Usage:
        router = LLMRouter()
        response = router.call([
            {"role": "system", "content": "You are a research assistant."},
            {"role": "user", "content": "Summarise these findings: ..."}
        ])
    """

    PROVIDERS = [
        {
            "name": "Anthropic",
            "env_key": "ANTHROPIC_API_KEY",
            "default_model": "claude-sonnet-4-6",
            "fn": _call_anthropic,
            "tier": "paid",
            "notes": "Best reasoning. Great for synthesis and citation analysis.",
        },
        {
            "name": "OpenAI",
            "env_key": "OPENAI_API_KEY",
            "default_model": "gpt-4o-mini",
            "fn": _call_openai,
            "tier": "paid",
            "notes": "Strong all-around. gpt-4o for best quality, gpt-4o-mini for speed.",
        },
        {
            "name": "Groq",
            "env_key": "GROQ_API_KEY",
            "default_model": "llama-3.1-8b-instant",
            "fn": _call_groq,
            "tier": "free",
            "notes": "Free tier, very fast. Good for quick research summaries.",
        },
        {
            "name": "Gemini",
            "env_key": "GEMINI_API_KEY",
            "default_model": "gemini-1.5-flash",
            "fn": _call_gemini,
            "tier": "free",
            "notes": "Google free tier. gemini-1.5-flash is fast and capable.",
        },
    ]

    def __init__(self, preferred_provider: Optional[str] = None, max_tokens: int = 2000):
        self.max_tokens = max_tokens
        self.preferred = preferred_provider  # e.g. "Anthropic" to force a provider
        self.active_provider = self._detect_provider()

    def _detect_provider(self) -> Optional[Dict]:
        """Return the first provider whose API key is set, respecting preference."""
        if self.preferred:
            for p in self.PROVIDERS:
                if p["name"] == self.preferred and os.getenv(p["env_key"]):
                    return p
            logger.warning(f"Preferred provider '{self.preferred}' not available. Falling back.")

        for p in self.PROVIDERS:
            if os.getenv(p["env_key"]):
                logger.info(f"LLMRouter: using {p['name']} ({p['tier']} tier)")
                return p

        logger.warning("LLMRouter: no LLM provider configured. Research will return raw results.")
        return None

    @property
    def available(self) -> bool:
        return self.active_provider is not None

    @property
    def provider_name(self) -> Optional[str]:
        return self.active_provider["name"] if self.active_provider else None

    @property
    def tier(self) -> str:
        if not self.active_provider:
            return "none"
        return self.active_provider["tier"]

    def call(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> Optional[str]:
        """
        Call the active LLM provider. Returns None if no provider is configured.
        Tries the next provider in cascade on failure.
        """
        if not self.active_provider:
            return None

        providers_to_try = [self.active_provider] + [
            p for p in self.PROVIDERS
            if p["name"] != self.active_provider["name"] and os.getenv(p["env_key"])
        ]

        for provider in providers_to_try:
            try:
                resolved_model = model or provider["default_model"]
                logger.debug(f"Calling {provider['name']} / {resolved_model}")
                result = provider["fn"](messages, resolved_model, self.max_tokens)
                return result
            except Exception as e:
                logger.warning(f"{provider['name']} call failed: {e}. Trying next provider.")

        logger.error("All LLM providers failed.")
        return None

    def status(self) -> Dict[str, Any]:
        return {
            "active_provider": self.provider_name,
            "tier": self.tier,
            "available_providers": [
                {"name": p["name"], "tier": p["tier"], "configured": bool(os.getenv(p["env_key"]))}
                for p in self.PROVIDERS
            ],
        }
