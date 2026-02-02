"""Conversation entity for optimal_agent with dual-model routing."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import TYPE_CHECKING, Any, Literal

import ollama
from homeassistant.components import conversation
from homeassistant.components.conversation import trace
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import intent

from .const import CHAT_CONTEXT_LIMIT, DOMAIN, LOGGER
from .response_templates import render_error, render_response

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

    from .cache_coordinator import DualModelCacheCoordinator
    from .data import OptimalAgentConfigEntryData
    from .router import IntentRouter


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the conversation entity from a config entry."""
    data: OptimalAgentConfigEntryData = config_entry.runtime_data
    async_add_entities([
        OptimalAgentConversationEntity(
            config_entry=config_entry,
            coordinator=data.coordinator,
            router=data.router,
        )
    ])


async def _transform_stream(
    result: AsyncIterator[ollama.ChatResponse],
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Transform Ollama streaming response into Home Assistant format."""
    new_msg = True

    async for response in result:
        response_message = response.get("message", {})
        chunk: conversation.AssistantContentDeltaDict = {}

        if new_msg:
            new_msg = False
            chunk["role"] = "assistant"

        if (content := response_message.get("content")) is not None:
            chunk["content"] = content

        if response.get("done"):
            new_msg = True

        yield chunk


class OptimalAgentConversationEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
):
    """Conversation entity using dual-model routing with Ollama."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supports_streaming = True

    def __init__(
        self,
        config_entry: ConfigEntry,
        coordinator: DualModelCacheCoordinator,
        router: IntentRouter,
    ) -> None:
        """Initialize the conversation entity."""
        self._config_entry = config_entry
        self._coordinator = coordinator
        self._router = router

        # Entity attributes
        self._attr_unique_id = config_entry.entry_id
        self._attr_device_info = {
            "identifiers": {(DOMAIN, config_entry.entry_id)},
            "name": f"Optimal Agent ({coordinator.router_model}/{coordinator.chat_model})",
            "manufacturer": "Optimal Agent",
            "model": f"Router: {coordinator.router_model}, Chat: {coordinator.chat_model}",
        }

        # Enable device control features
        self._attr_supported_features = conversation.ConversationEntityFeature.CONTROL

        # Ollama client for chat model
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

        # Initialize Ollama client for chat model
        ollama_url = self._config_entry.data.get("ollama_url", "http://localhost:11434")
        self._ollama_client = ollama.AsyncClient(host=ollama_url)

        # Ensure cache is warmed when entity is added
        if not self._coordinator.is_cache_valid:
            try:
                loaded = await self._coordinator.async_load_from_disk()
                if not loaded:
                    await self._coordinator.async_warm_both_models()
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
        """Handle a conversation message with dual-model routing.

        Flow:
        1. Send user message to Router Model for intent classification
        2. If tool call detected (JSON response):
           - Execute the Home Assistant service
           - Return response using Jinja2 template (no LLM needed)
        3. If NULL response (conversation):
           - Forward to Chat Model for natural language response
        """
        import time

        request_start = time.monotonic()

        LOGGER.info(
            "[Request] User message: '%s' (conversation_id=%s)",
            user_input.text[:80],
            user_input.conversation_id,
        )

        # Step 1: Route the message using Router Model
        LOGGER.info(
            "[Router] Sending to router model: %s",
            self._router.model,
        )
        route_start = time.monotonic()
        router_result = await self._router.async_route(user_input.text)
        route_time = (time.monotonic() - route_start) * 1000

        LOGGER.info(
            "[Router] Decision in %.0fms: %s | action=%s, entity=%s, params=%s",
            route_time,
            "TOOL_CALL" if router_result.is_tool_call else "CONVERSATION",
            router_result.action,
            router_result.entity,
            router_result.params,
        )

        trace.async_conversation_trace_append(
            trace.ConversationTraceEventType.AGENT_DETAIL,
            {"router_result": {
                "is_tool_call": router_result.is_tool_call,
                "action": router_result.action,
                "entity": router_result.entity,
                "params": router_result.params,
                "routing_time_ms": round(route_time),
            }},
        )

        if router_result.is_tool_call:
            # Path A: Execute tool and respond with template
            result = await self._handle_tool_call(user_input, chat_log, router_result)
            total_time = (time.monotonic() - request_start) * 1000
            LOGGER.info(
                "[Complete] Path A (Tool) finished in %.0fms total",
                total_time,
            )
            return result

        # Path B: Forward to Chat Model
        result = await self._handle_chat(user_input, chat_log)
        total_time = (time.monotonic() - request_start) * 1000
        LOGGER.info(
            "[Complete] Path B (Chat) finished in %.0fms total",
            total_time,
        )
        return result

    async def _handle_tool_call(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
        router_result: Any,
    ) -> conversation.ConversationResult:
        """
        Handle a tool call from the router.

        Args:
            user_input: The user's input
            chat_log: The conversation log
            router_result: Parsed result from the router

        Returns:
            Conversation result with template-based response

        """
        import time

        LOGGER.info(
            "[Tool] Executing: %s on %s with params=%s",
            router_result.action,
            router_result.entity,
            router_result.params,
        )

        # Get service call data
        service_data = self._router.get_service_call_data(router_result)
        if not service_data:
            LOGGER.warning("[Tool] Invalid action format: %s", router_result.action)
            response_text = render_error(
                "service_not_found",
                action=router_result.action,
            )
            return self._build_response(user_input, chat_log, response_text)

        domain, service, data = service_data
        LOGGER.info("[Tool] Service call: %s.%s with data=%s", domain, service, data)

        # Check if entity exists
        entity_id = data.get("entity_id")
        if entity_id and not entity_id.endswith(".all"):
            state = self.hass.states.get(entity_id)
            if not state:
                LOGGER.warning("[Tool] Entity not found: %s", entity_id)
                response_text = render_error(
                    "entity_not_found",
                    entity_id=entity_id,
                )
                return self._build_response(user_input, chat_log, response_text)

            if state.state == "unavailable":
                response_text = render_error(
                    "entity_unavailable",
                    friendly_name=state.attributes.get("friendly_name", entity_id),
                )
                return self._build_response(user_input, chat_log, response_text)

        # Execute the service
        try:
            service_start = time.monotonic()
            await self.hass.services.async_call(
                domain=domain,
                service=service,
                service_data=data,
                blocking=True,
            )
            service_time = (time.monotonic() - service_start) * 1000

            # Generate response using Jinja2 template
            response_text = render_response(
                action=router_result.action,
                entity_id=entity_id,
                params=router_result.params,
                hass=self.hass,
            )

            LOGGER.info(
                "[Tool] Service executed in %.0fms | Response: '%s'",
                service_time,
                response_text,
            )

        except Exception as exc:  # noqa: BLE001
            LOGGER.error("[Tool] Service call FAILED: %s", exc)
            response_text = render_error(
                "service_failed",
                action=router_result.action,
                error=str(exc),
            )

        return self._build_response(user_input, chat_log, response_text)

    async def _handle_chat(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Handle a conversation message using the Chat Model.

        Args:
            user_input: The user's input
            chat_log: The conversation log

        Returns:
            Conversation result with streamed response

        """
        import time

        LOGGER.info(
            "[Chat] Forwarding to chat model: %s",
            self._coordinator.chat_model,
        )

        if self._ollama_client is None:
            raise HomeAssistantError("Ollama client not initialized")

        # Build messages for chat model
        messages: list[ollama.Message] = []

        # System prompt with device context
        system_content = self._build_chat_system_prompt()
        messages.append(ollama.Message(role="system", content=system_content))

        # Add conversation history
        history = self._coordinator.get_conversation_messages(
            user_input.conversation_id
        )
        for msg in history:
            messages.append(ollama.Message(
                role=msg["role"],
                content=msg["content"],
            ))

        # Add current user message
        messages.append(ollama.Message(role="user", content=user_input.text))

        LOGGER.info(
            "[Chat] Sending %d messages (system + %d history + user) to %s",
            len(messages),
            len(history),
            self._coordinator.chat_model,
        )

        try:
            chat_start = time.monotonic()

            # Stream response from Chat Model
            response_generator = await self._ollama_client.chat(
                model=self._coordinator.chat_model,
                messages=messages,
                stream=True,
                keep_alive=self._coordinator.keep_alive,
                options={"num_ctx": CHAT_CONTEXT_LIMIT},
            )

            first_token_time = None
            # Collect full response for history
            full_response = ""

            # Stream to Home Assistant
            async for content in chat_log.async_add_delta_content_stream(
                self.entity_id,
                self._collect_and_stream(response_generator),
            ):
                if first_token_time is None:
                    first_token_time = time.monotonic()
                if hasattr(content, "content") and content.content:
                    full_response += content.content

            chat_time = (time.monotonic() - chat_start) * 1000
            ttft = ((first_token_time - chat_start) * 1000) if first_token_time else 0

            # Save to conversation history
            self._coordinator.add_conversation_message(
                user_input.conversation_id,
                "user",
                user_input.text,
            )
            if full_response:
                self._coordinator.add_conversation_message(
                    user_input.conversation_id,
                    "assistant",
                    full_response,
                )

            LOGGER.info(
                "[Chat] Response complete: %d chars in %.0fms (TTFT: %.0fms) | Preview: '%s'",
                len(full_response),
                chat_time,
                ttft,
                full_response[:100].replace("\n", " ") + ("..." if len(full_response) > 100 else ""),
            )
            return conversation.async_get_result_from_chat_log(user_input, chat_log)

        except ollama.RequestError as exc:
            LOGGER.error("Chat model request error: %s", exc)
            raise HomeAssistantError(
                f"Error communicating with Ollama: {exc}"
            ) from exc
        except ollama.ResponseError as exc:
            LOGGER.error("Chat model response error: %s", exc)
            raise HomeAssistantError(
                f"Ollama returned an error: {exc}"
            ) from exc

    async def _collect_and_stream(
        self,
        response_generator: AsyncIterator[ollama.ChatResponse],
    ) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
        """Collect response while streaming for history tracking."""
        async for chunk in _transform_stream(response_generator):
            yield chunk

    def _build_chat_system_prompt(self) -> str:
        """Build system prompt for the chat model."""
        device_list = self._coordinator.device_list

        return f"""You are a helpful smart home assistant for Home Assistant.
You can help users with questions about their home and provide information.

For device control commands, the routing system handles them automatically.
Focus on providing helpful, conversational responses.

Available devices in the home:
{device_list}

Be concise, helpful, and friendly. If you're not sure about something, ask for clarification."""

    def _build_response(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
        response_text: str,
    ) -> conversation.ConversationResult:
        """
        Build a conversation result from a text response.

        Args:
            user_input: The user's input
            chat_log: The conversation log
            response_text: The response text

        Returns:
            ConversationResult

        """
        # Add user message to chat log
        chat_log.async_add_user_content(
            conversation.UserContent(content=user_input.text)
        )

        # Add assistant response to chat log
        chat_log.async_add_assistant_content_without_tools(
            conversation.AssistantContent(
                agent_id=self.entity_id,
                content=response_text,
            )
        )

        # Create intent response with speech set (required for frontend)
        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(response_text)

        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=user_input.conversation_id,
        )

    async def async_prepare(self, language: str | None = None) -> None:
        """Prepare the conversation entity for use."""
        _ = language
        if not self._coordinator.is_cache_valid:
            LOGGER.info("Cache not valid, warming models...")
            await self._coordinator.async_warm_both_models()
