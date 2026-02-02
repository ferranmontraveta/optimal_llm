"""Custom types for optimal_agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry

    from .cache_coordinator import DualModelCacheCoordinator
    from .router import IntentRouter


type OptimalAgentConfigEntry = ConfigEntry[OptimalAgentConfigEntryData]


@dataclass
class OptimalAgentConfigEntryData:
    """Runtime data for the Optimal Agent integration."""

    coordinator: DualModelCacheCoordinator
    router: IntentRouter
    router_model: str
    chat_model: str
    keep_alive_persistent: bool
