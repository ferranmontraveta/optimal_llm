"""
Optimal LLM HAOS - Context-Cached Conversation Agent for Home Assistant.

This integration connects Home Assistant to a local Ollama server,
utilizing persistent KV caching to eliminate re-embedding delays
on low-powered hardware.

For more details about this integration, please refer to
https://github.com/optimal_llm_haos
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

import voluptuous as vol
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, ServiceCall, ServiceResponse, SupportsResponse
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.event import async_track_time_change

from .cache_coordinator import OllamaCacheCoordinator
from .const import (
    CACHE_REFRESH_HOUR,
    CACHE_REFRESH_MINUTE,
    CONF_AGGRESSIVE_CACHING,
    CONF_CONTEXT_PROMPT,
    CONF_MODEL,
    CONF_OLLAMA_URL,
    CONF_SYSTEM_PROMPT,
    LOGGER,
)
from .const import (
    DOMAIN as DOMAIN,
)
from .data import OllamaConfigEntryData
from .ollama_client import OllamaClient
from .prompt_builder import PromptBuilder

if TYPE_CHECKING:
    from .data import OllamaConfigEntry

# Service constants
SERVICE_PREVIEW_PROMPTS = "preview_prompts"
ATTR_CONFIG_ENTRY_ID = "config_entry_id"
ATTR_INCLUDE_CONTEXT = "include_context"

SERVICE_PREVIEW_PROMPTS_SCHEMA = vol.Schema({
    vol.Optional(ATTR_CONFIG_ENTRY_ID): str,
    vol.Optional(ATTR_INCLUDE_CONTEXT, default=True): bool,
})

PLATFORMS: list[Platform] = [
    Platform.CONVERSATION,
]


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """Set up the Optimal LLM HAOS integration."""
    # Register services
    async def async_preview_prompts(call: ServiceCall) -> ServiceResponse:
        """Handle preview_prompts service call."""
        config_entry_id = call.data.get(ATTR_CONFIG_ENTRY_ID)
        include_context = call.data.get(ATTR_INCLUDE_CONTEXT, True)

        # Find the config entry
        entries = hass.config_entries.async_entries(DOMAIN)
        if not entries:
            return {
                "error": "No Optimal LLM HAOS integrations configured",
                "system_prompt": None,
                "context_prompt": None,
            }

        entry = None
        if config_entry_id:
            for e in entries:
                if e.entry_id == config_entry_id:
                    entry = e
                    break
            if not entry:
                return {
                    "error": f"Config entry '{config_entry_id}' not found",
                    "system_prompt": None,
                    "context_prompt": None,
                }
        else:
            entry = entries[0]

        # Get the prompt builder from runtime data
        if not hasattr(entry, "runtime_data") or entry.runtime_data is None:
            return {
                "error": "Integration not fully loaded yet",
                "system_prompt": None,
                "context_prompt": None,
            }

        prompt_builder: PromptBuilder = entry.runtime_data.prompt_builder

        # Build the system prompt
        try:
            system_prompt = await prompt_builder.async_build_base_prompt()
        except Exception as exc:  # noqa: BLE001
            system_prompt = f"Error building system prompt: {exc}"

        # Build the context prompt if requested
        context_prompt = None
        if include_context:
            try:
                # Get some sample exposed entities for preview
                sample_entities = _get_sample_entities(hass)
                context_prompt = prompt_builder.build_volatile_context(sample_entities)
            except Exception as exc:  # noqa: BLE001
                context_prompt = f"Error building context prompt: {exc}"

        # Calculate lengths
        system_len = len(system_prompt) if system_prompt else 0
        context_len = len(context_prompt) if context_prompt else 0

        return {
            "system_prompt": system_prompt,
            "system_prompt_length": system_len,
            "context_prompt": context_prompt,
            "context_prompt_length": context_len,
            "total_length": system_len + context_len,
            "config_entry_id": entry.entry_id,
            "config_entry_title": entry.title,
        }

    hass.services.async_register(
        DOMAIN,
        SERVICE_PREVIEW_PROMPTS,
        async_preview_prompts,
        schema=SERVICE_PREVIEW_PROMPTS_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )

    return True


def _get_sample_entities(hass: HomeAssistant) -> list[dict[str, Any]]:
    """Get a sample of entities for context preview."""
    sample_entities = []
    # Get up to 20 entities from common domains
    priority_domains = ["light", "switch", "climate", "sensor", "binary_sensor", "cover"]
    seen = 0
    max_entities = 20

    for domain in priority_domains:
        if seen >= max_entities:
            break
        for entity_id in hass.states.async_entity_ids(domain):
            if seen >= max_entities:
                break
            sample_entities.append({"entity_id": entity_id})
            seen += 1

    return sample_entities


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

    # Extract configuration from data and options
    # Options can override data for configurable settings
    ollama_url = entry.data[CONF_OLLAMA_URL]
    model = entry.options.get(CONF_MODEL, entry.data.get(CONF_MODEL))
    aggressive_caching = entry.options.get(
        CONF_AGGRESSIVE_CACHING,
        entry.data.get(CONF_AGGRESSIVE_CACHING, True),
    )

    # Get custom prompts from options (if configured)
    system_prompt = entry.options.get(CONF_SYSTEM_PROMPT)
    context_prompt = entry.options.get(CONF_CONTEXT_PROMPT)

    # Create the Ollama client
    client = OllamaClient(
        base_url=ollama_url,
        session=async_get_clientsession(hass),
    )

    # Create the prompt builder with custom prompts
    prompt_builder = PromptBuilder(
        hass,
        system_prompt_template=system_prompt,
        context_prompt_template=context_prompt,
    )

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
    except Exception as exc:  # noqa: BLE001
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
) -> Callable[[Any], Coroutine[Any, Any, None]]:
    """Create callback for scheduled cache refresh."""

    async def _callback(_now: Any) -> None:
        """Handle scheduled cache refresh."""
        LOGGER.info("Running scheduled cache refresh")
        try:
            await coordinator.async_warm_cache()
        except Exception as exc:  # noqa: BLE001
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
        except Exception as exc:  # noqa: BLE001
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
    hass: HomeAssistant,  # noqa: ARG001
    entry: OllamaConfigEntry,
) -> None:
    """Handle removal of an entry."""
    # Clean up disk cache
    try:
        coordinator: OllamaCacheCoordinator = entry.runtime_data.coordinator
        await coordinator.async_clear_disk_cache()
        LOGGER.debug("Cleared disk cache on removal")
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Could not clear disk cache: %s", exc)
