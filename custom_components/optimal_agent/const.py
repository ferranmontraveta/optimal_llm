"""Constants for optimal_agent - Dual-Model Routing Integration."""

from logging import Logger, getLogger

LOGGER: Logger = getLogger(__package__)

DOMAIN = "optimal_agent"

# Configuration keys
CONF_OLLAMA_URL = "ollama_url"
CONF_ROUTER_MODEL = "router_model"
CONF_CHAT_MODEL = "chat_model"
CONF_KEEP_ALIVE_PERSISTENT = "keep_alive_persistent"
CONF_LLM_HASS_API = "llm_hass_api"

# Default values
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_ROUTER_MODEL = "functiongemma:latest"
DEFAULT_CHAT_MODEL = "gemma3:4b"
DEFAULT_TIMEOUT = 120  # seconds - models can be slow on low-powered hardware

# Context token limits for memory management
ROUTER_CONTEXT_LIMIT = 1024  # tokens - keep router context small for speed
CHAT_CONTEXT_LIMIT = 4096  # tokens - allow longer conversations

# Storage
STORAGE_KEY = "optimal_agent_state"
STORAGE_VERSION = 1

# Cache refresh schedule
CACHE_REFRESH_HOUR = 3  # 03:00 AM
CACHE_REFRESH_MINUTE = 0

# Router system prompt - optimized for small models with flat JSON output
ROUTER_SYSTEM_PROMPT = """You are an intent classifier for a smart home. Output ONLY valid JSON or NULL.

If the user wants to control a device, output a FLAT JSON object:
{"action": "domain.service", "entity": "entity_id"}

With optional parameters as top-level keys:
{"action": "light.turn_on", "entity": "light.living_room", "brightness": 128}
{"action": "climate.set_temperature", "entity": "climate.main", "temperature": 72}

Examples:
- "Turn on the living room light" -> {"action": "light.turn_on", "entity": "light.living_room"}
- "Set bedroom to 50% brightness" -> {"action": "light.turn_on", "entity": "light.bedroom", "brightness": 128}
- "Turn off all lights" -> {"action": "light.turn_off", "entity": "all"}
- "Set thermostat to 72" -> {"action": "climate.set_temperature", "entity": "climate.thermostat", "temperature": 72}
- "Open the blinds" -> {"action": "cover.open_cover", "entity": "cover.blinds"}
- "Lock the front door" -> {"action": "lock.lock", "entity": "lock.front_door"}

If the user wants general conversation (not device control), output exactly: NULL

Valid actions: light.turn_on, light.turn_off, switch.turn_on, switch.turn_off,
climate.set_temperature, cover.open_cover, cover.close_cover, fan.turn_on,
fan.turn_off, lock.lock, lock.unlock, scene.turn_on, script.turn_on

Available devices:
{device_list}
"""
