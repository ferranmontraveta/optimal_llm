"""
Dual-Model Cache Coordinator for optimal_agent.

Manages separate context caches for Router and Chat models, including:
- Building and maintaining device lists for routing
- Persisting state to disk for HA restarts
- Handling cache invalidation on device/area registry changes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from homeassistant.core import Event, callback
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.storage import Store

from .const import (
    CHAT_CONTEXT_LIMIT,
    LOGGER,
    ROUTER_CONTEXT_LIMIT,
    STORAGE_KEY,
    STORAGE_VERSION,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from .ollama_client import OllamaClient


@dataclass
class ModelContext:
    """Context state for a single model."""

    model: str
    messages: list[dict[str, str]] = field(default_factory=list)
    context_limit: int = 4096
    warmed_at: datetime | None = None

    def estimate_tokens(self) -> int:
        """Estimate token count (~4 chars per token)."""
        total_chars = sum(len(m.get("content", "")) for m in self.messages)
        return total_chars // 4

    def trim_to_limit(self) -> None:
        """Trim oldest messages to stay within context limit."""
        while self.estimate_tokens() > self.context_limit and len(self.messages) > 1:
            # Keep system message, remove oldest user/assistant messages
            if self.messages[0].get("role") == "system":
                if len(self.messages) > 2:
                    self.messages.pop(1)
                else:
                    break
            else:
                self.messages.pop(0)


class DualModelCacheCoordinator:
    """Coordinator for managing separate context caches for Router and Chat models."""

    def __init__(
        self,
        hass: HomeAssistant,
        ollama_client: OllamaClient,
        router_model: str,
        chat_model: str,
        keep_alive_persistent: bool = False,
    ) -> None:
        """
        Initialize the dual-model cache coordinator.

        Args:
            hass: Home Assistant instance
            ollama_client: Ollama API client
            router_model: Name of the router model
            chat_model: Name of the chat model
            keep_alive_persistent: If True, use keep_alive: -1

        """
        self._hass = hass
        self._ollama_client = ollama_client
        self._router_model = router_model
        self._chat_model = chat_model
        self._keep_alive_persistent = keep_alive_persistent

        # Separate context for each model
        self._router_context = ModelContext(
            model=router_model,
            context_limit=ROUTER_CONTEXT_LIMIT,
        )
        self._chat_context = ModelContext(
            model=chat_model,
            context_limit=CHAT_CONTEXT_LIMIT,
        )

        # Device list cache
        self._device_list: str = ""
        self._device_list_built_at: datetime | None = None

        # Conversation history per conversation_id (for chat model only)
        self._conversations: dict[str, list[dict[str, str]]] = {}

        # Disk storage
        self._store: Store[dict[str, Any]] = Store(
            hass, STORAGE_VERSION, STORAGE_KEY
        )

        # Cache validity
        self._cache_valid: bool = False

    @property
    def keep_alive(self) -> int | str:
        """Return the keep_alive value based on configuration."""
        return -1 if self._keep_alive_persistent else "60m"

    @property
    def is_cache_valid(self) -> bool:
        """Return whether the cache is currently valid."""
        return self._cache_valid

    @property
    def device_list(self) -> str:
        """Return the cached device list."""
        return self._device_list

    @property
    def router_model(self) -> str:
        """Return the router model name."""
        return self._router_model

    @property
    def chat_model(self) -> str:
        """Return the chat model name."""
        return self._chat_model

    async def async_build_device_list(self) -> str:
        """
        Build a compact device list for the router model.

        Returns:
            Formatted string of devices suitable for router context

        """
        area_registry = ar.async_get(self._hass)
        device_registry = dr.async_get(self._hass)
        entity_registry = er.async_get(self._hass)

        # Group entities by area for compact representation
        areas_entities: dict[str, list[str]] = {}
        unassigned: list[str] = []

        # Priority domains for device control
        controllable_domains = {
            "light", "switch", "climate", "cover", "fan",
            "lock", "scene", "script", "media_player",
            "alarm_control_panel",
        }

        for entity in entity_registry.entities.values():
            if entity.domain not in controllable_domains:
                continue

            # Get area name
            area_name = "Unassigned"
            if entity.area_id:
                area = area_registry.async_get_area(entity.area_id)
                area_name = area.name if area else "Unassigned"
            elif entity.device_id:
                device = device_registry.async_get(entity.device_id)
                if device and device.area_id:
                    area = area_registry.async_get_area(device.area_id)
                    area_name = area.name if area else "Unassigned"

            # Get friendly name
            friendly_name = entity.name or entity.original_name or entity.entity_id

            # Format: "entity_id (Friendly Name)"
            entry = f"{entity.entity_id} ({friendly_name})"

            if area_name == "Unassigned":
                unassigned.append(entry)
            else:
                if area_name not in areas_entities:
                    areas_entities[area_name] = []
                areas_entities[area_name].append(entry)

        # Build compact list
        lines = []
        for area_name in sorted(areas_entities.keys()):
            entities = areas_entities[area_name]
            lines.append(f"\n{area_name}:")
            for entity in sorted(entities):
                lines.append(f"  - {entity}")

        if unassigned:
            lines.append("\nUnassigned:")
            for entity in sorted(unassigned):
                lines.append(f"  - {entity}")

        self._device_list = "\n".join(lines) if lines else "No controllable devices found."
        self._device_list_built_at = datetime.now()

        LOGGER.debug(
            "Built device list: %d areas, %d entities",
            len(areas_entities),
            sum(len(e) for e in areas_entities.values()) + len(unassigned),
        )

        return self._device_list

    async def async_warm_both_models(self) -> None:
        """
        Pre-warm both models with their respective contexts.

        This should be called on startup when keep_alive_persistent is enabled.
        """
        LOGGER.info("Warming both models with persistent loading")

        # Build device list first
        if not self._device_list:
            await self.async_build_device_list()

        # Pre-load both models
        results = await self._ollama_client.async_preload_models(
            router_model=self._router_model,
            chat_model=self._chat_model,
            keep_alive=-1 if self._keep_alive_persistent else "60m",
        )

        # Update context states
        if results.get(self._router_model):
            self._router_context.warmed_at = datetime.now()

        if results.get(self._chat_model):
            self._chat_context.warmed_at = datetime.now()

        self._cache_valid = all(results.values())

        if self._cache_valid:
            LOGGER.info("Both models warmed successfully")
            await self.async_save_to_disk()
        else:
            LOGGER.warning("Some models failed to warm: %s", results)

    async def async_invalidate_cache(self) -> None:
        """
        Invalidate the cache and trigger re-warming.

        Called when device registry changes or at scheduled refresh time.
        """
        LOGGER.info("Invalidating cache - device definitions may have changed")
        self._cache_valid = False
        self._device_list = ""

        # Clear conversation histories (they may reference old entities)
        self._conversations.clear()

        # Rebuild and re-warm
        await self.async_build_device_list()
        await self.async_warm_both_models()

    @callback
    def async_handle_registry_update(self, event: Event) -> None:
        """
        Handle device or area registry update events.

        Args:
            event: The registry update event

        """
        LOGGER.debug("Registry update detected: %s", event.event_type)

        # Schedule cache invalidation (don't await in callback)
        self._hass.async_create_task(
            self._async_handle_registry_update_task()
        )

    async def _async_handle_registry_update_task(self) -> None:
        """Async task to handle registry updates with debounce."""
        # Add a small delay to batch rapid changes
        await self._hass.async_add_executor_job(
            __import__("time").sleep, 2
        )
        await self.async_invalidate_cache()

    def get_conversation_messages(
        self, conversation_id: str | None
    ) -> list[dict[str, str]]:
        """
        Get messages for a specific conversation.

        Args:
            conversation_id: Unique conversation identifier

        Returns:
            List of messages for the conversation

        """
        if conversation_id is None:
            return []
        return self._conversations.get(conversation_id, []).copy()

    def add_conversation_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> None:
        """
        Add a message to a conversation's history.

        Args:
            conversation_id: Unique conversation identifier
            role: Message role ('user' or 'assistant')
            content: Message content

        """
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = []

        self._conversations[conversation_id].append({
            "role": role,
            "content": content,
        })

        # Limit conversation history to prevent token overflow
        # Estimate: keep ~20 messages or stay under token limit
        max_history = 20
        if len(self._conversations[conversation_id]) > max_history:
            self._conversations[conversation_id] = self._conversations[
                conversation_id
            ][-max_history:]

    def clear_conversation(self, conversation_id: str) -> None:
        """
        Clear a conversation's history.

        Args:
            conversation_id: Unique conversation identifier

        """
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]

    async def async_save_to_disk(self) -> None:
        """
        Serialize state to disk storage.

        Saves to /config/.storage/optimal_agent_state.json
        """
        data = {
            "router": {
                "model": self._router_model,
                "warmed_at": (
                    self._router_context.warmed_at.isoformat()
                    if self._router_context.warmed_at
                    else None
                ),
            },
            "chat": {
                "model": self._chat_model,
                "warmed_at": (
                    self._chat_context.warmed_at.isoformat()
                    if self._chat_context.warmed_at
                    else None
                ),
            },
            "device_list": self._device_list,
            "device_list_built_at": (
                self._device_list_built_at.isoformat()
                if self._device_list_built_at
                else None
            ),
            "conversations": self._conversations,
        }

        await self._store.async_save(data)
        LOGGER.debug("Saved state to disk")

    async def async_load_from_disk(self) -> bool:
        """
        Load cached state from disk if available.

        Returns:
            True if state was loaded successfully, False otherwise

        """
        data = await self._store.async_load()

        if data is None:
            LOGGER.debug("No cached state found on disk")
            return False

        # Verify models match
        router_data = data.get("router", {})
        chat_data = data.get("chat", {})

        if router_data.get("model") != self._router_model:
            LOGGER.info(
                "Cached router model mismatch (%s vs %s), ignoring",
                router_data.get("model"),
                self._router_model,
            )
            return False

        if chat_data.get("model") != self._chat_model:
            LOGGER.info(
                "Cached chat model mismatch (%s vs %s), ignoring",
                chat_data.get("model"),
                self._chat_model,
            )
            return False

        # Restore state
        self._device_list = data.get("device_list", "")
        self._conversations = data.get("conversations", {})

        if data.get("device_list_built_at"):
            self._device_list_built_at = datetime.fromisoformat(
                data["device_list_built_at"]
            )

        if router_data.get("warmed_at"):
            self._router_context.warmed_at = datetime.fromisoformat(
                router_data["warmed_at"]
            )

        if chat_data.get("warmed_at"):
            self._chat_context.warmed_at = datetime.fromisoformat(
                chat_data["warmed_at"]
            )

        self._cache_valid = bool(self._device_list)
        LOGGER.info("Loaded state from disk")

        return self._cache_valid

    async def async_clear_disk_cache(self) -> None:
        """Clear the disk cache."""
        await self._store.async_remove()
        LOGGER.debug("Cleared disk cache")
