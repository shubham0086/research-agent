"""
Local config — reads settings from environment variables.
Replaces the internal get_settings() / pydantic-settings wiring from Agency OS.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Settings:
    # LLM
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    default_llm_model: str = field(default_factory=lambda: os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini"))
    fallback_llm_model: str = "gpt-3.5-turbo"
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    max_tokens_per_request: int = int(os.getenv("MAX_TOKENS", "2000"))

    # Search providers — leave empty to fall back to DuckDuckGo at no cost
    serp_api_key: Optional[str] = field(default_factory=lambda: os.getenv("SERP_API_KEY"))
    tavily_api_key: Optional[str] = field(default_factory=lambda: os.getenv("TAVILY_API_KEY"))
    brave_search_api_key: Optional[str] = field(default_factory=lambda: os.getenv("BRAVE_SEARCH_API_KEY"))

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def available_search_providers(self) -> List[str]:
        providers = []
        if self.serp_api_key:
            providers.append("SerpAPI")
        if self.tavily_api_key:
            providers.append("Tavily")
        if self.brave_search_api_key:
            providers.append("Brave Search")
        if not providers:
            providers.append("DuckDuckGo (fallback)")
        return providers

    @property
    def available_llm_providers(self) -> List[str]:
        providers = []
        if self.has_openai:
            providers.append("OpenAI")
        return providers


# Module-level singleton — mirrors the get_settings() pattern used in production
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
