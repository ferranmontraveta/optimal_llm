"""Intent Router for optimal_agent - Routes user messages to tools or chat model.

Uses Ollama's native tool calling API with FunctionGemma or similar models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import ollama

from .const import LOGGER, ROUTER_CONTEXT_LIMIT, ROUTER_SYSTEM_PROMPT

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


# ============================================================================
# Tool Definitions for Ollama Native Tool Calling
# ============================================================================
# These are passed to ollama.chat(tools=[...]) and the model returns
# tool_calls with function name and arguments.

SMART_HOME_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "light_turn_on",
            "description": "Turn on a light or set its brightness",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "The light entity ID (e.g., light.living_room, light.bedroom). Use 'all' for all lights.",
                    },
                    "brightness": {
                        "type": "integer",
                        "description": "Brightness level from 0-255. 128 = 50%, 255 = 100%",
                    },
                },
                "required": ["entity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "light_turn_off",
            "description": "Turn off a light",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "The light entity ID (e.g., light.living_room). Use 'all' for all lights.",
                    },
                },
                "required": ["entity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "switch_turn_on",
            "description": "Turn on a switch",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "The switch entity ID (e.g., switch.coffee_maker)",
                    },
                },
                "required": ["entity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "switch_turn_off",
            "description": "Turn off a switch",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "The switch entity ID (e.g., switch.coffee_maker)",
                    },
                },
                "required": ["entity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "climate_set_temperature",
            "description": "Set the temperature on a thermostat or climate device",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "The climate entity ID (e.g., climate.thermostat)",
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Target temperature in the system's unit (Fahrenheit or Celsius)",
                    },
                },
                "required": ["entity_id", "temperature"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cover_open",
            "description": "Open a cover, blind, or garage door",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "The cover entity ID (e.g., cover.blinds, cover.garage_door)",
                    },
                },
                "required": ["entity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cover_close",
            "description": "Close a cover, blind, or garage door",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "The cover entity ID (e.g., cover.blinds, cover.garage_door)",
                    },
                },
                "required": ["entity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fan_turn_on",
            "description": "Turn on a fan",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "The fan entity ID (e.g., fan.bedroom)",
                    },
                    "percentage": {
                        "type": "integer",
                        "description": "Fan speed percentage from 0-100",
                    },
                },
                "required": ["entity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fan_turn_off",
            "description": "Turn off a fan",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "The fan entity ID (e.g., fan.bedroom)",
                    },
                },
                "required": ["entity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lock_lock",
            "description": "Lock a door lock",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "The lock entity ID (e.g., lock.front_door)",
                    },
                },
                "required": ["entity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lock_unlock",
            "description": "Unlock a door lock",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "The lock entity ID (e.g., lock.front_door)",
                    },
                },
                "required": ["entity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scene_activate",
            "description": "Activate a scene",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "The scene entity ID (e.g., scene.movie_time)",
                    },
                },
                "required": ["entity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "script_run",
            "description": "Run a script",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "The script entity ID (e.g., script.good_morning)",
                    },
                },
                "required": ["entity_id"],
            },
        },
    },
]

# Mapping from tool function names to Home Assistant service calls
TOOL_TO_SERVICE: dict[str, tuple[str, str]] = {
    "light_turn_on": ("light", "turn_on"),
    "light_turn_off": ("light", "turn_off"),
    "switch_turn_on": ("switch", "turn_on"),
    "switch_turn_off": ("switch", "turn_off"),
    "climate_set_temperature": ("climate", "set_temperature"),
    "cover_open": ("cover", "open_cover"),
    "cover_close": ("cover", "close_cover"),
    "fan_turn_on": ("fan", "turn_on"),
    "fan_turn_off": ("fan", "turn_off"),
    "lock_lock": ("lock", "lock"),
    "lock_unlock": ("lock", "unlock"),
    "scene_activate": ("scene", "turn_on"),
    "script_run": ("script", "turn_on"),
}


@dataclass
class RouterResult:
    """Result from the intent router."""

    is_tool_call: bool
    action: str | None = None  # e.g., "light.turn_on"
    entity: str | None = None  # e.g., "light.living_room"
    params: dict[str, Any] = field(default_factory=dict)  # brightness, temperature, etc.
    raw_response: str = ""  # Original model output for debugging


def parse_tool_calls(response: ollama.ChatResponse) -> RouterResult:
    """
    Parse tool calls from Ollama's native tool calling response.

    Args:
        response: The ChatResponse from Ollama with potential tool_calls

    Returns:
        RouterResult with parsed tool call or is_tool_call=False for conversation

    """
    message = response.get("message", {})
    tool_calls = message.get("tool_calls")

    # No tool calls means conversation mode
    if not tool_calls:
        content = message.get("content", "")
        return RouterResult(is_tool_call=False, raw_response=content)

    # Take the first tool call (most relevant)
    tool_call = tool_calls[0]
    function_info = tool_call.get("function", {})
    function_name = function_info.get("name", "")
    arguments = function_info.get("arguments", {})

    # Map tool function name to Home Assistant service
    service_mapping = TOOL_TO_SERVICE.get(function_name)
    if not service_mapping:
        LOGGER.warning("Unknown tool function: %s", function_name)
        return RouterResult(is_tool_call=False, raw_response=str(tool_call))

    domain, service = service_mapping
    action = f"{domain}.{service}"

    # Extract entity_id from arguments
    entity_id = arguments.pop("entity_id", None)

    # Handle "all" entity
    if entity_id and entity_id.lower() == "all":
        entity_id = f"{domain}.all"
    elif entity_id and not entity_id.startswith(f"{domain}."):
        # If entity doesn't have domain prefix, add it
        entity_id = f"{domain}.{entity_id}"

    return RouterResult(
        is_tool_call=True,
        action=action,
        entity=entity_id,
        params=arguments,  # Remaining args are service params (brightness, temperature, etc.)
        raw_response=str(tool_call),
    )


class IntentRouter:
    """Routes user intents to tools or chat model using Ollama's native tool calling."""

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
            model: Name of the router model (e.g., functiongemma:latest)
            keep_alive: How long to keep model loaded (-1 for indefinite)

        """
        self._hass = hass
        self._ollama_url = ollama_url
        self._model = model
        self._keep_alive = keep_alive
        # Lazy initialization to avoid blocking SSL call in event loop
        self._ollama_client: ollama.AsyncClient | None = None

        # System prompt for tool-calling context
        self._system_prompt: str = ""
        self._device_list: str = ""

    def _get_client(self) -> ollama.AsyncClient:
        """Get or create the Ollama client (lazy initialization)."""
        if self._ollama_client is None:
            self._ollama_client = ollama.AsyncClient(host=self._ollama_url)
        return self._ollama_client

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
        self._device_list = device_list
        self._system_prompt = ROUTER_SYSTEM_PROMPT.format(device_list=device_list)
        LOGGER.debug(
            "Router initialized with %d char system prompt, %d tools",
            len(self._system_prompt),
            len(SMART_HOME_TOOLS),
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

            await self._get_client().chat(
                model=self._model,
                messages=messages,
                tools=SMART_HOME_TOOLS,
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

        Uses Ollama's native tool calling - if the model returns tool_calls,
        we execute them. If it returns plain text, we forward to chat model.

        Args:
            message: The user's message text

        Returns:
            RouterResult indicating whether to execute a tool or use chat model

        """
        import time

        if not self._system_prompt:
            LOGGER.warning("[Router] Not initialized, defaulting to chat")
            return RouterResult(is_tool_call=False)

        try:
            messages = [
                ollama.Message(role="system", content=self._system_prompt),
                ollama.Message(role="user", content=message),
            ]

            start_time = time.monotonic()
            response = await self._get_client().chat(
                model=self._model,
                messages=messages,
                tools=SMART_HOME_TOOLS,
                keep_alive=self._keep_alive,
                options={"num_ctx": ROUTER_CONTEXT_LIMIT},
            )
            elapsed_ms = (time.monotonic() - start_time) * 1000

            # Log router metrics
            prompt_tokens = response.get("prompt_eval_count", 0)
            eval_tokens = response.get("eval_count", 0)

            # Check for tool calls in the response
            message_obj = response.get("message", {})
            tool_calls = message_obj.get("tool_calls")
            content = message_obj.get("content", "")

            if tool_calls:
                LOGGER.debug(
                    "[Router] Tool call detected in %.0fms (prompt: %d tokens, eval: %d tokens): %s",
                    elapsed_ms,
                    prompt_tokens,
                    eval_tokens,
                    str(tool_calls)[:150],
                )
            else:
                LOGGER.debug(
                    "[Router] No tool call in %.0fms (prompt: %d tokens, eval: %d tokens): %s",
                    elapsed_ms,
                    prompt_tokens,
                    eval_tokens,
                    content[:150].replace("\n", " ") if content else "(empty)",
                )

            return parse_tool_calls(response)

        except (ollama.RequestError, ollama.ResponseError) as exc:
            LOGGER.error("[Router] Request FAILED: %s", exc)
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
