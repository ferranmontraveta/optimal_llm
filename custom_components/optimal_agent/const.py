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
# Note: Router needs more context now due to tool definitions (~1500-2000 tokens)
ROUTER_CONTEXT_LIMIT = 4096  # tokens - includes tool schemas + device list
CHAT_CONTEXT_LIMIT = 8192  # tokens - allow longer conversations

# Storage
STORAGE_KEY = "optimal_agent_state"
STORAGE_VERSION = 1

# Cache refresh schedule
CACHE_REFRESH_HOUR = 3  # 03:00 AM
CACHE_REFRESH_MINUTE = 0

# Router system prompt - optimized for native tool calling models like FunctionGemma
# The model uses Ollama's tools parameter, so this prompt provides context only.
# NOTE: {device_list} is the only placeholder
ROUTER_SYSTEM_PROMPT = """You are a smart home assistant that controls devices. You have access to tools for controlling lights, switches, climate, covers, fans, locks, scenes, and scripts.

When the user asks to control a device, use the appropriate tool with the correct entity_id from the available devices list.

For general conversation, greetings, or questions that don't involve device control, respond naturally without using any tools.

Available devices in this home:
{device_list}

Match user requests to the closest entity_id from the list above. If the user says a room name or device name, find the matching entity."""
