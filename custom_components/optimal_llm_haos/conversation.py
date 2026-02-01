"""
Conversation entity for optimal_llm_haos.

Implements a ConversationEntity that uses Ollama for natural language
processing with persistent KV caching for efficient operation.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Literal

from homeassistant.components.conversation import (
    ConversationEntity,
    ConversationInput,
    ConversationResult,
)
from homeassistant.components.conversation.chat_log import ChatLog
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN, LOGGER

if TYPE_CHECKING:
    from .cache_coordinator import OllamaCacheCoordinator
    from .data import OllamaConfigEntryData
    from .ollama_client import OllamaClient
    from .prompt_builder import PromptBuilder


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """
    Set up the conversation entity from a config entry.

    Args:
        hass: Home Assistant instance
        config_entry: The config entry
        async_add_entities: Callback to add entities

    """
    data: OllamaConfigEntryData = config_entry.runtime_data
    async_add_entities([
        OllamaConversationEntity(
            config_entry=config_entry,
            client=data.client,
            coordinator=data.coordinator,
            prompt_builder=data.prompt_builder,
        )
    ])


class OllamaConversationEntity(ConversationEntity):
    """Conversation entity using Ollama with context caching."""

    _attr_has_entity_name = True
    _attr_name = None  # Use device name

    def __init__(
        self,
        config_entry: ConfigEntry,
        client: OllamaClient,
        coordinator: OllamaCacheCoordinator,
        prompt_builder: PromptBuilder,
    ) -> None:
        """
        Initialize the conversation entity.

        Args:
            config_entry: The config entry
            client: Ollama API client
            coordinator: Cache coordinator
            prompt_builder: Prompt builder

        """
        self._config_entry = config_entry
        self._client = client
        self._coordinator = coordinator
        self._prompt_builder = prompt_builder

        # Entity attributes
        self._attr_unique_id = config_entry.entry_id
        self._attr_device_info = {
            "identifiers": {(DOMAIN, config_entry.entry_id)},
            "name": f"Ollama ({coordinator.model})",
            "manufacturer": "Ollama",
            "model": coordinator.model,
        }

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """
        Return supported languages.

        Returns '*' to indicate all languages are supported.
        """
        return "*"

    async def async_added_to_hass(self) -> None:
        """Called when entity is added to hass."""
        await super().async_added_to_hass()

        # Ensure cache is warmed when entity is added
        if not self._coordinator.is_cache_valid:
            try:
                # Try to load from disk first
                loaded = await self._coordinator.async_load_from_disk()
                if not loaded:
                    # No disk cache, warm fresh
                    await self._coordinator.async_warm_cache()
            except Exception as exc:
                LOGGER.error("Failed to initialize cache: %s", exc)

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: ChatLog,
    ) -> ConversationResult:
        """
        Handle an incoming conversation message.

        Args:
            user_input: The user's input containing text and context
            chat_log: Chat log for conversation history and LLM API access

        Returns:
            ConversationResult with the assistant's response

        """
        conversation_id = user_input.conversation_id or str(uuid.uuid4())

        try:
            # Get exposed entities from the LLM API if available
            exposed_entities = await self._get_exposed_entities(chat_log)

            # Build volatile context with current entity states
            volatile_context = self._prompt_builder.build_volatile_context(
                exposed_entities
            )

            # Build complete message array
            messages = self._coordinator.build_chat_messages(
                conversation_id=conversation_id,
                user_message=user_input.text,
                volatile_context=volatile_context,
            )

            # Send to Ollama with cache recovery
            response = await self._client.async_chat_with_recovery(
                model=self._coordinator.model,
                messages=messages,
                warm_cache_callback=self._coordinator.async_warm_cache,
            )

            # Extract assistant message
            assistant_message = response.get("message", {})
            response_text = assistant_message.get("content", "")

            if not response_text:
                response_text = "I'm sorry, I couldn't generate a response."

            # Store messages in conversation history
            self._coordinator.add_conversation_message(
                conversation_id=conversation_id,
                role="user",
                content=user_input.text,
            )
            self._coordinator.add_conversation_message(
                conversation_id=conversation_id,
                role="assistant",
                content=response_text,
            )

            # Build the response
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_speech(response_text)

            return ConversationResult(
                conversation_id=conversation_id,
                response=intent_response,
            )

        except Exception as exc:
            LOGGER.error("Error processing conversation: %s", exc)

            # Return error response
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Error communicating with Ollama: {exc}",
            )

            return ConversationResult(
                conversation_id=conversation_id,
                response=intent_response,
            )

    async def _get_exposed_entities(
        self,
        chat_log: ChatLog,
    ) -> list[dict[str, Any]]:
        """
        Get list of exposed entities from the LLM API.

        Args:
            chat_log: Chat log with LLM API access

        Returns:
            List of exposed entity dictionaries

        """
        exposed_entities = []

        # Try to get entities from the LLM API
        try:
            if hasattr(chat_log, "llm_api") and chat_log.llm_api:
                # Get tools which represent exposed entities/services
                tools = getattr(chat_log.llm_api, "tools", [])

                for tool in tools:
                    # Extract entity info from tool if available
                    if hasattr(tool, "entity_id"):
                        exposed_entities.append({
                            "entity_id": tool.entity_id,
                        })

        except Exception as exc:
            LOGGER.debug("Could not get exposed entities from LLM API: %s", exc)

        return exposed_entities

    async def async_prepare(self, language: str | None = None) -> None:
        """
        Prepare the conversation entity for use.

        This is called before processing requests and can be used
        to load models or other resources.

        Args:
            language: The language to prepare for (optional)

        """
        # Ensure cache is valid
        if not self._coordinator.is_cache_valid:
            LOGGER.info("Cache not valid, warming...")
            await self._coordinator.async_warm_cache()
