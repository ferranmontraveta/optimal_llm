"""
Cache Coordinator for optimal_llm_haos.

Manages the context cache for Ollama conversations, including:
- Building and maintaining the base prompt (system + devices)
- Persisting cache to disk for HA restarts
- Handling cache invalidation on device/area registry changes
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.helpers.storage import Store

from .const import (
    LOGGER,
    STORAGE_KEY,
    STORAGE_VERSION,
)

if TYPE_CHECKING:
    from .ollama_client import OllamaClient
    from .prompt_builder import PromptBuilder


class OllamaCacheCoordinator:
    """Coordinator for managing Ollama context cache."""

    def __init__(
        self,
        hass: HomeAssistant,
        client: OllamaClient,
        prompt_builder: PromptBuilder,
        model: str,
        aggressive_caching: bool = True,
    ) -> None:
        """
        Initialize the cache coordinator.

        Args:
            hass: Home Assistant instance
            client: Ollama API client
            prompt_builder: Prompt builder for constructing prompts
            model: Name of the Ollama model to use
            aggressive_caching: If True, persist cache to disk

        """
        self._hass = hass
        self._client = client
        self._prompt_builder = prompt_builder
        self._model = model
        self._aggressive_caching = aggressive_caching

        # Cache storage
        self._base_messages: list[dict[str, str]] = []
        self._cache_valid: bool = False
        self._cache_warmed_at: datetime | None = None
        self._store: Store[dict[str, Any]] = Store(
            hass, STORAGE_VERSION, STORAGE_KEY
        )

        # Conversation history per conversation_id
        self._conversations: dict[str, list[dict[str, str]]] = {}

    @property
    def is_cache_valid(self) -> bool:
        """Return whether the cache is currently valid."""
        return self._cache_valid

    @property
    def base_messages(self) -> list[dict[str, str]]:
        """Return the cached base messages."""
        return self._base_messages.copy()

    @property
    def model(self) -> str:
        """Return the model name."""
        return self._model

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

    async def async_warm_cache(self) -> None:
        """
        Build base prompt and pre-warm the Ollama cache.

        This sends a minimal request to Ollama with the base prompt,
        causing it to embed and cache the context for future requests.
        """
        LOGGER.info("Warming Ollama context cache for model %s", self._model)

        # Build the base system prompt
        system_prompt = await self._prompt_builder.async_build_base_prompt()

        self._base_messages = [
            {"role": "system", "content": system_prompt},
        ]

        # Send a pre-warming request to populate the KV cache
        try:
            warm_messages = self._base_messages + [
                {"role": "user", "content": "Hello"},
            ]

            await self._client.async_chat(
                model=self._model,
                messages=warm_messages,
                keep_alive="60m",  # Keep model loaded for a while
            )

            self._cache_valid = True
            self._cache_warmed_at = datetime.now()
            LOGGER.info("Ollama context cache warmed successfully")

            # Persist to disk if aggressive caching is enabled
            if self._aggressive_caching:
                await self.async_save_to_disk()

        except Exception as exc:
            LOGGER.error("Failed to warm cache: %s", exc)
            self._cache_valid = False
            raise

    async def async_invalidate_cache(self) -> None:
        """Invalidate the current cache and trigger re-warming."""
        LOGGER.info("Invalidating Ollama context cache")
        self._cache_valid = False
        self._base_messages = []

        # Clear all conversation histories
        self._conversations.clear()

        # Re-warm the cache
        await self.async_warm_cache()

    @callback
    def async_handle_registry_update(self, event: Event) -> None:
        """
        Handle device or area registry update events.

        This callback is triggered when devices or areas are added,
        removed, or modified, requiring a cache refresh.

        Args:
            event: The registry update event

        """
        LOGGER.debug("Registry update detected: %s", event.event_type)

        # Schedule cache invalidation (don't await in callback)
        self._hass.async_create_task(
            self._async_handle_registry_update_task()
        )

    async def _async_handle_registry_update_task(self) -> None:
        """Async task to handle registry updates."""
        # Add a small delay to batch rapid changes
        await self._hass.async_add_executor_job(
            __import__("time").sleep, 2
        )
        await self.async_invalidate_cache()

    async def async_save_to_disk(self) -> None:
        """
        Serialize context cache to disk storage.

        Saves to /config/.storage/ollama_context_cache.json
        """
        if not self._aggressive_caching:
            return

        data = {
            "model": self._model,
            "base_messages": self._base_messages,
            "cache_warmed_at": (
                self._cache_warmed_at.isoformat()
                if self._cache_warmed_at
                else None
            ),
            "conversations": self._conversations,
        }

        await self._store.async_save(data)
        LOGGER.debug("Saved context cache to disk")

    async def async_load_from_disk(self) -> bool:
        """
        Load cached context from disk if available.

        Returns:
            True if cache was loaded successfully, False otherwise

        """
        if not self._aggressive_caching:
            return False

        data = await self._store.async_load()

        if data is None:
            LOGGER.debug("No cached context found on disk")
            return False

        # Verify the cache is for the correct model
        if data.get("model") != self._model:
            LOGGER.info(
                "Cached context is for different model (%s vs %s), ignoring",
                data.get("model"),
                self._model,
            )
            return False

        self._base_messages = data.get("base_messages", [])
        self._conversations = data.get("conversations", {})

        if data.get("cache_warmed_at"):
            self._cache_warmed_at = datetime.fromisoformat(
                data["cache_warmed_at"]
            )

        self._cache_valid = bool(self._base_messages)
        LOGGER.info("Loaded context cache from disk")

        return self._cache_valid

    async def async_clear_disk_cache(self) -> None:
        """Clear the disk cache."""
        await self._store.async_remove()
        LOGGER.debug("Cleared disk cache")

    def build_chat_messages(
        self,
        conversation_id: str | None,
        user_message: str,
        volatile_context: str | None = None,
    ) -> list[dict[str, str]]:
        """
        Build the complete message array for a chat request.

        Combines:
        1. Base messages (cached system prompt + device list)
        2. Volatile context (current entity states)
        3. Conversation history
        4. New user message

        Args:
            conversation_id: Unique conversation identifier
            user_message: The user's current message
            volatile_context: Current entity states (optional)

        Returns:
            Complete list of messages for Ollama API

        """
        messages: list[dict[str, str]] = []

        # Add base messages (system prompt with device list)
        messages.extend(self._base_messages)

        # Add volatile context as a system message if provided
        if volatile_context:
            messages.append({
                "role": "system",
                "content": f"Current entity states:\n{volatile_context}",
            })

        # Add conversation history
        if conversation_id:
            messages.extend(self.get_conversation_messages(conversation_id))

        # Add the new user message
        messages.append({
            "role": "user",
            "content": user_message,
        })

        return messages
