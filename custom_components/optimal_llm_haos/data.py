"""Custom types for optimal_llm_haos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry

    from .cache_coordinator import OllamaCacheCoordinator
    from .ollama_client import OllamaClient
    from .prompt_builder import PromptBuilder


type OllamaConfigEntry = ConfigEntry[OllamaConfigEntryData]


@dataclass
class OllamaConfigEntryData:
    """Runtime data for the Optimal LLM HAOS integration."""

    client: OllamaClient
    coordinator: OllamaCacheCoordinator
    prompt_builder: PromptBuilder
    model: str
    aggressive_caching: bool
