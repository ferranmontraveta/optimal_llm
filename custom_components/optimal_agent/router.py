"""Intent Router for optimal_agent - Routes user messages to tools or chat model."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import ollama

from .const import LOGGER, ROUTER_CONTEXT_LIMIT, ROUTER_SYSTEM_PROMPT

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


@dataclass
class RouterResult:
    """Result from the intent router."""

    is_tool_call: bool
    action: str | None = None  # e.g., "light.turn_on"
    entity: str | None = None  # e.g., "light.living_room"
    params: dict[str, Any] = field(default_factory=dict)  # brightness, temperature, etc.
    raw_response: str = ""  # Original model output for debugging


def parse_router_response(response: str) -> RouterResult:
    """
    Parse flat JSON from router model response.

    Expected formats:
    - Tool call: {"action": "light.turn_on", "entity": "light.living_room", "brightness": 128}
    - No tool: NULL

    Args:
        response: Raw text response from the router model

    Returns:
        RouterResult with parsed data or is_tool_call=False for conversation

    """
    text = response.strip()

    # Check for NULL response (conversation mode)
    if text.upper() == "NULL" or not text:
        return RouterResult(is_tool_call=False, raw_response=text)

    # Try to extract JSON from response (model might add extra text)
    json_start = text.find("{")
    json_end = text.rfind("}") + 1

    if json_start == -1 or json_end == 0:
        LOGGER.debug("No JSON found in router response: %s", text[:100])
        return RouterResult(is_tool_call=False, raw_response=text)

    json_str = text[json_start:json_end]

    try:
        data = json.loads(json_str)

        # Extract required fields
        action = data.pop("action", None)
        entity = data.pop("entity", None)

        if not action:
            LOGGER.debug("No action in router JSON: %s", json_str)
            return RouterResult(is_tool_call=False, raw_response=text)

        # Remaining keys are service parameters (brightness, temperature, etc.)
        return RouterResult(
            is_tool_call=True,
            action=action,
            entity=entity,
            params=data,
            raw_response=text,
        )

    except json.JSONDecodeError as exc:
        LOGGER.debug("Failed to parse router JSON: %s - %s", json_str[:100], exc)
        return RouterResult(is_tool_call=False, raw_response=text)


class IntentRouter:
    """Routes user intents to tools or chat model using a small classifier model."""

    def __init__(
        self,
        hass: HomeAssistant,
        ollama_url: str,
        model: str,
        keep_alive: int | str = "5m",
    ) -> None:
        """
        Initialize the intent router.

        Args:
            hass: Home Assistant instance
            ollama_url: URL of the Ollama server
            model: Name of the router model (e.g., gemma3:1b)
            keep_alive: How long to keep model loaded (-1 for indefinite)

        """
        self._hass = hass
        self._ollama_url = ollama_url
        self._model = model
        self._keep_alive = keep_alive
        self._ollama_client = ollama.AsyncClient(host=ollama_url)

        # Context tracking for router (separate from chat model)
        self._context_messages: list[ollama.Message] = []
        self._system_prompt: str = ""

    @property
    def model(self) -> str:
        """Return the router model name."""
        return self._model

    async def async_initialize(self, device_list: str) -> None:
        """
        Initialize the router with device list.

        Args:
            device_list: Formatted list of available devices

        """
        self._system_prompt = ROUTER_SYSTEM_PROMPT.format(device_list=device_list)
        LOGGER.debug(
            "Router initialized with %d char system prompt",
            len(self._system_prompt),
        )

    async def async_warm_model(self) -> None:
        """Pre-warm the router model by sending an initial request."""
        if not self._system_prompt:
            LOGGER.warning("Router not initialized, skipping warm-up")
            return

        LOGGER.info("Warming router model: %s", self._model)

        try:
            messages = [
                ollama.Message(role="system", content=self._system_prompt),
                ollama.Message(role="user", content="Hello"),
            ]

            await self._ollama_client.chat(
                model=self._model,
                messages=messages,
                keep_alive=self._keep_alive,
                options={"num_ctx": ROUTER_CONTEXT_LIMIT},
            )

            LOGGER.info("Router model warmed successfully")

        except (ollama.RequestError, ollama.ResponseError) as exc:
            LOGGER.error("Failed to warm router model: %s", exc)
            raise

    async def async_route(self, message: str) -> RouterResult:
        """
        Route a user message to determine if it's a tool call or conversation.

        Args:
            message: The user's message text

        Returns:
            RouterResult indicating whether to execute a tool or use chat model

        """
        if not self._system_prompt:
            LOGGER.warning("Router not initialized, defaulting to chat")
            return RouterResult(is_tool_call=False)

        try:
            messages = [
                ollama.Message(role="system", content=self._system_prompt),
                ollama.Message(role="user", content=message),
            ]

            response = await self._ollama_client.chat(
                model=self._model,
                messages=messages,
                keep_alive=self._keep_alive,
                options={"num_ctx": ROUTER_CONTEXT_LIMIT},
            )

            response_text = response.get("message", {}).get("content", "")
            LOGGER.debug("Router response: %s", response_text[:200])

            return parse_router_response(response_text)

        except (ollama.RequestError, ollama.ResponseError) as exc:
            LOGGER.error("Router request failed: %s", exc)
            # Fall back to chat model on error
            return RouterResult(is_tool_call=False)

    def get_service_call_data(
        self, result: RouterResult
    ) -> tuple[str, str, dict[str, Any]] | None:
        """
        Convert RouterResult to Home Assistant service call parameters.

        Args:
            result: The parsed router result

        Returns:
            Tuple of (domain, service, service_data) or None if invalid

        """
        if not result.is_tool_call or not result.action:
            return None

        # Parse action: "light.turn_on" -> domain="light", service="turn_on"
        try:
            domain, service = result.action.split(".", 1)
        except ValueError:
            LOGGER.warning("Invalid action format: %s", result.action)
            return None

        # Build service data
        service_data: dict[str, Any] = {}

        if result.entity:
            # Handle "all" entity for domain-wide operations
            if result.entity.lower() == "all":
                service_data["entity_id"] = f"{domain}.all"
            else:
                service_data["entity_id"] = result.entity

        # Add any additional parameters (brightness, temperature, etc.)
        service_data.update(result.params)

        return domain, service, service_data
