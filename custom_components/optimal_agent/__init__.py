"""
Optimal Agent - Dual-Model Routing Integration for Home Assistant.

This integration uses a dual-model architecture with Ollama:
- Router Model: Small model for intent classification (tool vs conversation)
- Chat Model: Larger model for natural language responses

Features:
- Persistent model loading with keep_alive: -1
- Separate context tracking for each model
- Fast tool execution with Jinja2 response templates
- Scheduled 3:00 AM cache refresh
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from homeassistant.const import Platform
from homeassistant.helpers.event import async_track_time_change

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from homeassistant.core import HomeAssistant

from .cache_coordinator import DualModelCacheCoordinator
from .const import (
    CACHE_REFRESH_HOUR,
    CACHE_REFRESH_MINUTE,
    CONF_CHAT_MODEL,
    CONF_KEEP_ALIVE_PERSISTENT,
    CONF_OLLAMA_URL,
    CONF_ROUTER_MODEL,
    LOGGER,
)
from .const import (
    DOMAIN as DOMAIN,
)
from .data import OptimalAgentConfigEntryData
from .ollama_client import OllamaClient
from .router import IntentRouter

if TYPE_CHECKING:
    from .data import OptimalAgentConfigEntry

PLATFORMS: list[Platform] = [
    Platform.CONVERSATION,
]


async def async_setup_entry(
    hass: HomeAssistant,
    entry: OptimalAgentConfigEntry,
) -> bool:
    """
    Set up Optimal Agent from a config entry.

    Args:
        hass: Home Assistant instance
        entry: The config entry

    Returns:
        True if setup was successful

    """
    LOGGER.info("Setting up Optimal Agent integration")

    # Extract configuration
    ollama_url = entry.data[CONF_OLLAMA_URL]
    router_model = entry.options.get(
        CONF_ROUTER_MODEL,
        entry.data.get(CONF_ROUTER_MODEL),
    )
    chat_model = entry.options.get(
        CONF_CHAT_MODEL,
        entry.data.get(CONF_CHAT_MODEL),
    )
    keep_alive_persistent = entry.options.get(
        CONF_KEEP_ALIVE_PERSISTENT,
        entry.data.get(CONF_KEEP_ALIVE_PERSISTENT, True),
    )

    LOGGER.info(
        "Configuration: router=%s, chat=%s, keep_alive_persistent=%s",
        router_model,
        chat_model,
        keep_alive_persistent,
    )

    # Create Ollama client
    ollama_client = OllamaClient(base_url=ollama_url)

    # Create the dual-model cache coordinator
    coordinator = DualModelCacheCoordinator(
        hass=hass,
        ollama_client=ollama_client,
        router_model=router_model,
        chat_model=chat_model,
        keep_alive_persistent=keep_alive_persistent,
    )

    # Create the intent router
    router = IntentRouter(
        hass=hass,
        ollama_url=ollama_url,
        model=router_model,
        keep_alive=-1 if keep_alive_persistent else "5m",
    )

    # Store runtime data
    entry.runtime_data = OptimalAgentConfigEntryData(
        coordinator=coordinator,
        router=router,
        router_model=router_model,
        chat_model=chat_model,
        keep_alive_persistent=keep_alive_persistent,
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

    # Schedule daily cache refresh at 03:00 AM
    entry.async_on_unload(
        async_track_time_change(
            hass,
            _async_scheduled_cache_refresh(coordinator, router),
            hour=CACHE_REFRESH_HOUR,
            minute=CACHE_REFRESH_MINUTE,
            second=0,
        )
    )

    # Initialize: Load from disk or warm models
    try:
        loaded = await coordinator.async_load_from_disk()
        if loaded:
            # Initialize router with cached device list
            await router.async_initialize(coordinator.device_list)
            LOGGER.info("Loaded state from disk")
        else:
            # Build device list and warm models
            LOGGER.info("No cached state, initializing...")
            await coordinator.async_build_device_list()
            await router.async_initialize(coordinator.device_list)

            if keep_alive_persistent:
                await coordinator.async_warm_both_models()
                await router.async_warm_model()

    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "Failed to initialize, will retry on first request: %s",
            exc,
        )

    # Forward setup to conversation platform
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Register reload listener
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))

    LOGGER.info("Optimal Agent integration setup complete")
    return True


def _async_scheduled_cache_refresh(
    coordinator: DualModelCacheCoordinator,
    router: IntentRouter,
) -> Callable[[Any], Coroutine[Any, Any, None]]:
    """Create callback for scheduled cache refresh at 3:00 AM."""

    async def _callback(_now: Any) -> None:
        """Handle scheduled cache refresh."""
        LOGGER.info("Running scheduled 3:00 AM cache refresh")
        try:
            # Rebuild device list with fresh data
            await coordinator.async_invalidate_cache()

            # Re-initialize router with updated device list
            await router.async_initialize(coordinator.device_list)
            await router.async_warm_model()

            LOGGER.info("Scheduled cache refresh completed")
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Scheduled cache refresh failed: %s", exc)

    return _callback


async def async_unload_entry(
    hass: HomeAssistant,
    entry: OptimalAgentConfigEntry,
) -> bool:
    """
    Handle removal of an entry.

    Args:
        hass: Home Assistant instance
        entry: The config entry being unloaded

    Returns:
        True if unload was successful

    """
    LOGGER.info("Unloading Optimal Agent integration")

    # Save state to disk before unloading
    coordinator: DualModelCacheCoordinator = entry.runtime_data.coordinator
    if coordinator.is_cache_valid:
        try:
            await coordinator.async_save_to_disk()
            LOGGER.debug("Saved state to disk before unload")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to save state on unload: %s", exc)

    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_reload_entry(
    hass: HomeAssistant,
    entry: OptimalAgentConfigEntry,
) -> None:
    """
    Reload config entry.

    Args:
        hass: Home Assistant instance
        entry: The config entry to reload

    """
    await hass.config_entries.async_reload(entry.entry_id)


async def async_remove_entry(
    hass: HomeAssistant,  # noqa: ARG001
    entry: OptimalAgentConfigEntry,
) -> None:
    """Handle removal of an entry."""
    # Clean up disk cache
    try:
        coordinator: DualModelCacheCoordinator = entry.runtime_data.coordinator
        await coordinator.async_clear_disk_cache()
        LOGGER.debug("Cleared disk cache on removal")
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Could not clear disk cache: %s", exc)
