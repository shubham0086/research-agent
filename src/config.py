"""
Config — reads from environment variables.
Supports all four LLM providers and three search providers.

Free tier: set GROQ_API_KEY or GEMINI_API_KEY (both have free tiers).
Paid tier: set ANTHROPIC_API_KEY or OPENAI_API_KEY for best research quality.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Settings:
    # LLM providers — set whichever you have (priority: Anthropic > OpenAI > Groq > Gemini)
    anthropic_api_key: Optional[str] = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    openai_api_key:    Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    groq_api_key:      Optional[str] = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    gemini_api_key:    Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))

    # Model overrides (optional — defaults are set in the router per provider)
    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"))
    openai_model:    str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    groq_model:      str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
    gemini_model:    str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))

    # Preferred provider — leave empty to auto-detect from available keys
    preferred_llm: Optional[str] = field(default_factory=lambda: os.getenv("PREFERRED_LLM"))

    # Search providers — leave empty to fall back to DuckDuckGo (free, no key needed)
    serp_api_key:        Optional[str] = field(default_factory=lambda: os.getenv("SERP_API_KEY"))
    tavily_api_key:      Optional[str] = field(default_factory=lambda: os.getenv("TAVILY_API_KEY"))
    brave_search_api_key: Optional[str] = field(default_factory=lambda: os.getenv("BRAVE_SEARCH_API_KEY"))

    # Research settings
    max_results:     int = int(os.getenv("MAX_RESULTS", "8"))
    max_llm_tokens:  int = int(os.getenv("MAX_LLM_TOKENS", "3000"))

    @property
    def has_any_llm(self) -> bool:
        return any([
            self.anthropic_api_key,
            self.openai_api_key,
            self.groq_api_key,
            self.gemini_api_key,
        ])

    @property
    def available_llm_providers(self) -> List[str]:
        providers = []
        if self.anthropic_api_key: providers.append("Anthropic")
        if self.openai_api_key:    providers.append("OpenAI")
        if self.groq_api_key:      providers.append("Groq")
        if self.gemini_api_key:    providers.append("Gemini")
        return providers or ["none — raw results only"]

    @property
    def available_search_providers(self) -> List[str]:
        providers = []
        if self.serp_api_key:         providers.append("SerpAPI")
        if self.tavily_api_key:        providers.append("Tavily")
        if self.brave_search_api_key:  providers.append("Brave Search")
        return providers or ["DuckDuckGo (free fallback)"]


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
