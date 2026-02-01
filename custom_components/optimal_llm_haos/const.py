"""Constants for optimal_llm_haos."""

from logging import Logger, getLogger

LOGGER: Logger = getLogger(__package__)

DOMAIN = "optimal_llm_haos"

# Configuration keys
CONF_OLLAMA_URL = "ollama_url"
CONF_MODEL = "model"
CONF_AGGRESSIVE_CACHING = "aggressive_caching"
CONF_LLM_HASS_API = "llm_hass_api"

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
