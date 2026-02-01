"""
Optimal LLM HAOS - Context-Cached Conversation Agent for Home Assistant.

This integration connects Home Assistant to a local Ollama server,
utilizing persistent KV caching to eliminate re-embedding delays
on low-powered hardware.

For more details about this integration, please refer to
https://github.com/optimal_llm_haos
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.event import async_track_time_change

from .cache_coordinator import OllamaCacheCoordinator
from .const import (
    CACHE_REFRESH_HOUR,
    CACHE_REFRESH_MINUTE,
    CONF_AGGRESSIVE_CACHING,
    CONF_MODEL,
    CONF_OLLAMA_URL,
    DOMAIN,
    LOGGER,
)
from .data import OllamaConfigEntryData
from .ollama_client import OllamaClient
from .prompt_builder import PromptBuilder

if TYPE_CHECKING:
    from .data import OllamaConfigEntry

PLATFORMS: list[Platform] = [
    Platform.CONVERSATION,
]


async def async_setup_entry(
    hass: HomeAssistant,
    entry: OllamaConfigEntry,
) -> bool:
    """
    Set up Optimal LLM HAOS from a config entry.

    Args:
        hass: Home Assistant instance
        entry: The config entry

    Returns:
        True if setup was successful

    """
    LOGGER.info("Setting up Optimal LLM HAOS integration")

    # Extract configuration
    ollama_url = entry.data[CONF_OLLAMA_URL]
    model = entry.data[CONF_MODEL]
    aggressive_caching = entry.data.get(CONF_AGGRESSIVE_CACHING, True)

    # Create the Ollama client
    client = OllamaClient(
        base_url=ollama_url,
        session=async_get_clientsession(hass),
    )

    # Create the prompt builder
    prompt_builder = PromptBuilder(hass)

    # Create the cache coordinator
    coordinator = OllamaCacheCoordinator(
        hass=hass,
        client=client,
        prompt_builder=prompt_builder,
        model=model,
        aggressive_caching=aggressive_caching,
    )

    # Store runtime data
    entry.runtime_data = OllamaConfigEntryData(
        client=client,
        coordinator=coordinator,
        prompt_builder=prompt_builder,
        model=model,
        aggressive_caching=aggressive_caching,
    )

    # Register event listeners for cache invalidation
    entry.async_on_unload(
        hass.bus.async_listen(
            "device_registry_updated",
            coordinator.async_handle_registry_update,
        )
    )
    entry.async_on_unload(
        hass.bus.async_listen(
            "area_registry_updated",
            coordinator.async_handle_registry_update,
        )
    )

    # Schedule daily cache refresh at 03:00
    entry.async_on_unload(
        async_track_time_change(
            hass,
            _async_scheduled_cache_refresh(coordinator),
            hour=CACHE_REFRESH_HOUR,
            minute=CACHE_REFRESH_MINUTE,
            second=0,
        )
    )

    # Try to load cache from disk, or warm fresh
    try:
        loaded = await coordinator.async_load_from_disk()
        if not loaded:
            LOGGER.info("No cached context found, warming cache...")
            await coordinator.async_warm_cache()
    except Exception as exc:
        LOGGER.warning(
            "Failed to initialize cache, will retry on first request: %s",
            exc,
        )

    # Forward setup to conversation platform
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Register reload listener
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))

    LOGGER.info("Optimal LLM HAOS integration setup complete")
    return True


def _async_scheduled_cache_refresh(
    coordinator: OllamaCacheCoordinator,
):
    """
    Create callback for scheduled cache refresh.

    Args:
        coordinator: The cache coordinator

    Returns:
        Async callback function

    """
    async def _callback(_now) -> None:
        """Handle scheduled cache refresh."""
        LOGGER.info("Running scheduled cache refresh")
        try:
            await coordinator.async_warm_cache()
        except Exception as exc:
            LOGGER.error("Scheduled cache refresh failed: %s", exc)

    return _callback


async def async_unload_entry(
    hass: HomeAssistant,
    entry: OllamaConfigEntry,
) -> bool:
    """
    Handle removal of an entry.

    Args:
        hass: Home Assistant instance
        entry: The config entry being unloaded

    Returns:
        True if unload was successful

    """
    LOGGER.info("Unloading Optimal LLM HAOS integration")

    # Save cache to disk before unloading
    coordinator: OllamaCacheCoordinator = entry.runtime_data.coordinator
    if coordinator.is_cache_valid:
        try:
            await coordinator.async_save_to_disk()
            LOGGER.debug("Saved cache to disk before unload")
        except Exception as exc:
            LOGGER.warning("Failed to save cache on unload: %s", exc)

    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_reload_entry(
    hass: HomeAssistant,
    entry: OllamaConfigEntry,
) -> None:
    """
    Reload config entry.

    Args:
        hass: Home Assistant instance
        entry: The config entry to reload

    """
    await hass.config_entries.async_reload(entry.entry_id)


async def async_remove_entry(
    hass: HomeAssistant,
    entry: OllamaConfigEntry,
) -> None:
    """
    Handle removal of an entry.

    Called when the config entry is removed.

    Args:
        hass: Home Assistant instance
        entry: The config entry being removed

    """
    # Clean up disk cache
    try:
        coordinator: OllamaCacheCoordinator = entry.runtime_data.coordinator
        await coordinator.async_clear_disk_cache()
        LOGGER.debug("Cleared disk cache on removal")
    except Exception as exc:
        LOGGER.debug("Could not clear disk cache: %s", exc)
