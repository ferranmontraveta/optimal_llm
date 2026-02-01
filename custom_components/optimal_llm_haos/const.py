"""Constants for optimal_llm_haos."""

from logging import Logger, getLogger

LOGGER: Logger = getLogger(__package__)

DOMAIN = "optimal_llm_haos"

# Configuration keys
CONF_OLLAMA_URL = "ollama_url"
CONF_MODEL = "model"
CONF_AGGRESSIVE_CACHING = "aggressive_caching"
CONF_LLM_HASS_API = "llm_hass_api"
CONF_SYSTEM_PROMPT = "system_prompt"
CONF_CONTEXT_PROMPT = "context_prompt"

# Default values
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
DEFAULT_TIMEOUT = 120  # seconds - models can be slow on low-powered hardware

# Storage
STORAGE_KEY = "ollama_context_cache"
STORAGE_VERSION = 1

# Cache refresh schedule
CACHE_REFRESH_HOUR = 3  # 03:00 AM
CACHE_REFRESH_MINUTE = 0

# Default System Prompt Template
# Use {device_summary} as a placeholder for the auto-generated device list
DEFAULT_SYSTEM_PROMPT = """You are a helpful smart home assistant for Home Assistant.
You can help users control their smart home devices and answer questions about their home.

## Available Devices and Areas

{device_summary}

## Available Services

You can control devices by calling Home Assistant services. Common services include:
- light.turn_on / light.turn_off - Control lights
- switch.turn_on / switch.turn_off - Control switches
- climate.set_temperature - Set thermostat temperature
- cover.open_cover / cover.close_cover - Control blinds/covers
- media_player.play_media / media_player.pause - Control media players
- scene.turn_on - Activate scenes
- script.turn_on - Run scripts

When the user asks to control a device, identify the appropriate entity and service.

## Guidelines

1. Be concise and helpful
2. If you're not sure which device the user means, ask for clarification
3. Confirm actions before executing them when appropriate
4. If a device is unavailable or in an error state, inform the user
5. Provide current state information when relevant"""

# Default Context Prompt Template
# This is added as a system message with current entity states
DEFAULT_CONTEXT_PROMPT = """Current state of relevant entities:
{entity_states}"""
