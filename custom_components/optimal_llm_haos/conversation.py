"""Conversation entity for optimal_llm_haos with streaming support."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterator
from typing import TYPE_CHECKING, Any, Literal

import ollama

from homeassistant.components import conversation
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.util.ssl import get_default_context

from .const import DOMAIN, LOGGER

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

    from .cache_coordinator import OllamaCacheCoordinator
    from .data import OllamaConfigEntryData
    from .prompt_builder import PromptBuilder


async def async_setup_entry(
    hass: HomeAssistant,  # noqa: ARG001
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the conversation entity from a config entry."""
    data: OllamaConfigEntryData = config_entry.runtime_data
    async_add_entities([
        OllamaConversationEntity(
            config_entry=config_entry,
            coordinator=data.coordinator,
            prompt_builder=data.prompt_builder,
        )
    ])


async def _transform_stream(
    result: AsyncIterator[ollama.ChatResponse],
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Transform Ollama streaming response into Home Assistant format.

    This matches the official HA Ollama integration's transform function exactly.
    """
    new_msg = True
    
    async for response in result:
        LOGGER.debug("Received response: %s", response)
        response_message = response["message"]
        chunk: conversation.AssistantContentDeltaDict = {}
        
        if new_msg:
            new_msg = False
            chunk["role"] = "assistant"
        
        if (tool_calls := response_message.get("tool_calls")) is not None:
            chunk["tool_calls"] = [
                llm.ToolInput(
                    tool_name=tool_call["function"]["name"],
                    tool_args=tool_call["function"].get("arguments", {}),
                )
                for tool_call in tool_calls
            ]
        
        if (content := response_message.get("content")) is not None:
            chunk["content"] = content
        
        if response.get("done"):
            new_msg = True
        
        yield chunk


def _convert_chat_content_to_ollama(
    content: conversation.Content,
) -> ollama.Message:
    """Convert Home Assistant chat content to Ollama message format."""
    if isinstance(content, conversation.ToolResultContent):
        return ollama.Message(
            role="tool",
            content=json.dumps(content.tool_result),
        )

    if isinstance(content, conversation.AssistantContent):
        tool_calls = None
        if content.tool_calls:
            tool_calls = [
                ollama.Message.ToolCall(
                    function=ollama.Message.ToolCall.Function(
                        name=tc.tool_name,
                        arguments=tc.tool_args,
                    )
                )
                for tc in content.tool_calls
            ]
        return ollama.Message(
            role="assistant",
            content=content.content or "",
            tool_calls=tool_calls,
        )

    if isinstance(content, conversation.UserContent):
        return ollama.Message(
            role="user",
            content=content.content,
        )

    if isinstance(content, conversation.SystemContent):
        return ollama.Message(
            role="system",
            content=content.content,
        )

    raise TypeError(f"Unexpected content type: {type(content)}")


class OllamaConversationEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
):
    """Conversation entity using Ollama with context caching and streaming."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supports_streaming = True  # Enable streaming support

    def __init__(
        self,
        config_entry: ConfigEntry,
        coordinator: OllamaCacheCoordinator,
        prompt_builder: PromptBuilder,
    ) -> None:
        """Initialize the conversation entity."""
        self._config_entry = config_entry
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

        # Enable device control features
        self._attr_supported_features = conversation.ConversationEntityFeature.CONTROL

        # Ollama client (using ollama library)
        self._ollama_client: ollama.AsyncClient | None = None

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return supported languages."""
        return "*"

    async def async_added_to_hass(self) -> None:
        """Handle entity added to hass."""
        await super().async_added_to_hass()

        # Register as a conversation agent
        conversation.async_set_agent(self.hass, self._config_entry, self)

        # Initialize ollama client (with SSL verification like official integration)
        ollama_url = self._config_entry.data.get("ollama_url", "http://localhost:11434")
        self._ollama_client = ollama.AsyncClient(host=ollama_url, verify=get_default_context())

        # Ensure cache is warmed when entity is added
        if not self._coordinator.is_cache_valid:
            try:
                loaded = await self._coordinator.async_load_from_disk()
                if not loaded:
                    await self._coordinator.async_warm_cache()
            except Exception:  # noqa: BLE001
                LOGGER.exception("Failed to initialize cache")

    async def async_will_remove_from_hass(self) -> None:
        """Handle entity being removed from hass."""
        conversation.async_unset_agent(self.hass, self._config_entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Handle a conversation message with streaming support."""
        LOGGER.info("_async_handle_message called - using streaming API")
        
        # Debug: Check if delta_listener is set (needed for real-time streaming)
        has_delta_listener = hasattr(chat_log, 'delta_listener') and chat_log.delta_listener is not None
        LOGGER.info("chat_log.delta_listener set: %s", has_delta_listener)

        if self._ollama_client is None:
            raise HomeAssistantError("Ollama client not initialized")

        # Ensure cache is valid
        if not self._coordinator.is_cache_valid:
            LOGGER.info("Cache not valid, warming...")
            await self._coordinator.async_warm_cache()

        # Build volatile context with current entity states
        volatile_context = self._prompt_builder.build_volatile_context(
            exposed_entities=None
        )

        # Prepare system prompt from our cache
        system_prompt = ""
        if self._coordinator.base_messages:
            # Get the cached system prompt
            for msg in self._coordinator.base_messages:
                if msg.get("role") == "system":
                    system_prompt = msg.get("content", "")
                    break

        # Add volatile context to system prompt if available
        if volatile_context:
            system_prompt += f"\n\n## Current Entity States\n{volatile_context}"

        # Provide LLM data to chat_log (like official integration does)
        # This sets up the conversation context properly
        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                user_llm_prompt=system_prompt,
                user_extra_system_prompt=user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        # Build messages from chat_log (which now has proper context)
        messages: list[ollama.Message] = []

        # Add messages from chat_log (conversation history)
        for content in chat_log.content:
            try:
                messages.append(_convert_chat_content_to_ollama(content))
            except TypeError:
                LOGGER.debug("Skipping unsupported content type: %s", type(content))

        try:
            LOGGER.info("Starting streaming response from Ollama")
            
            # Stream response from Ollama
            response_generator = await self._ollama_client.chat(
                model=self._coordinator.model,
                messages=messages,
                stream=True,
                keep_alive="60m",
            )

            # Stream content to Home Assistant UI using HA's streaming API
            async for content in chat_log.async_add_delta_content_stream(
                self.entity_id,
                _transform_stream(response_generator),
            ):
                pass  # Content is streamed internally by HA

            LOGGER.info("Streaming complete")
            # Return result from chat_log
            return conversation.async_get_result_from_chat_log(user_input, chat_log)

        except ollama.RequestError as err:
            LOGGER.error("Ollama request error: %s", err)
            raise HomeAssistantError(
                f"Error communicating with Ollama: {err}"
            ) from err
        except ollama.ResponseError as err:
            LOGGER.error("Ollama response error: %s", err)
            raise HomeAssistantError(
                f"Ollama returned an error: {err}"
            ) from err

    async def async_prepare(self, language: str | None = None) -> None:
        """Prepare the conversation entity for use."""
        _ = language
        if not self._coordinator.is_cache_valid:
            LOGGER.info("Cache not valid, warming...")
            await self._coordinator.async_warm_cache()
