"""Config flow for Optimal Agent integration."""

from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.helpers import selector

from .const import (
    CONF_CHAT_MODEL,
    CONF_KEEP_ALIVE_PERSISTENT,
    CONF_OLLAMA_URL,
    CONF_ROUTER_MODEL,
    DEFAULT_CHAT_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_ROUTER_MODEL,
    DOMAIN,
    LOGGER,
)
from .ollama_client import (
    OllamaClient,
    OllamaClientCommunicationError,
    OllamaClientError,
)


class OptimalAgentFlowHandler(config_entries.ConfigFlow, domain=DOMAIN):
    """Config flow for Optimal Agent."""

    VERSION = 1

    def __init__(self) -> None:
        """Initialize the config flow."""
        self._ollama_url: str = DEFAULT_OLLAMA_URL
        self._available_models: list[str] = []

    async def async_step_user(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> config_entries.ConfigFlowResult:
        """Handle the initial step - Ollama URL configuration."""
        errors: dict[str, str] = {}

        if user_input is not None:
            self._ollama_url = user_input[CONF_OLLAMA_URL]

            # Test connection and fetch available models
            try:
                client = OllamaClient(base_url=self._ollama_url)
                self._available_models = await client.async_get_model_names()

                if not self._available_models:
                    errors["base"] = "no_models"
                else:
                    # Connection successful, proceed to model selection
                    return await self.async_step_models()

            except OllamaClientCommunicationError as exc:
                LOGGER.warning("Connection to Ollama failed: %s", exc)
                errors["base"] = "cannot_connect"
            except OllamaClientError as exc:
                LOGGER.exception("Unexpected error: %s", exc)
                errors["base"] = "unknown"

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_OLLAMA_URL,
                        default=self._ollama_url,
                    ): selector.TextSelector(
                        selector.TextSelectorConfig(
                            type=selector.TextSelectorType.URL,
                        ),
                    ),
                },
            ),
            errors=errors,
        )

    async def async_step_models(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> config_entries.ConfigFlowResult:
        """Handle the model selection step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            router_model = user_input[CONF_ROUTER_MODEL]
            chat_model = user_input[CONF_CHAT_MODEL]
            keep_alive_persistent = user_input.get(CONF_KEEP_ALIVE_PERSISTENT, True)

            # Set unique ID based on URL and models
            await self.async_set_unique_id(
                f"{self._ollama_url}_{router_model}_{chat_model}"
            )
            self._abort_if_unique_id_configured()

            return self.async_create_entry(
                title=f"Optimal Agent ({router_model})",
                data={
                    CONF_OLLAMA_URL: self._ollama_url,
                    CONF_ROUTER_MODEL: router_model,
                    CONF_CHAT_MODEL: chat_model,
                    CONF_KEEP_ALIVE_PERSISTENT: keep_alive_persistent,
                },
            )

        # Build model selector options
        model_options = [
            selector.SelectOptionDict(value=model, label=model)
            for model in self._available_models
        ]

        # Default router model
        default_router = DEFAULT_ROUTER_MODEL
        if DEFAULT_ROUTER_MODEL in self._available_models:
            default_router = DEFAULT_ROUTER_MODEL
        elif self._available_models:
            default_router = self._available_models[0]

        # Default chat model
        default_chat = DEFAULT_CHAT_MODEL
        if DEFAULT_CHAT_MODEL in self._available_models:
            default_chat = DEFAULT_CHAT_MODEL
        elif self._available_models:
            default_chat = self._available_models[0]

        return self.async_show_form(
            step_id="models",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_ROUTER_MODEL,
                        default=default_router,
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=model_options,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        ),
                    ),
                    vol.Required(
                        CONF_CHAT_MODEL,
                        default=default_chat,
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=model_options,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        ),
                    ),
                    vol.Required(
                        CONF_KEEP_ALIVE_PERSISTENT,
                        default=True,
                    ): selector.BooleanSelector(),
                },
            ),
            errors=errors,
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Get the options flow for this handler."""
        return OptimalAgentOptionsFlowHandler(config_entry)


class OptimalAgentOptionsFlowHandler(config_entries.OptionsFlow):
    """Options flow for Optimal Agent."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self._config_entry = config_entry

    async def async_step_init(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> config_entries.ConfigFlowResult:
        """Handle options flow."""
        errors: dict[str, str] = {}

        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        # Get current values
        current_data = self._config_entry.data
        current_options = self._config_entry.options

        # Fetch available models
        available_models = []
        try:
            client = OllamaClient(base_url=current_data[CONF_OLLAMA_URL])
            available_models = await client.async_get_model_names()
        except OllamaClientError as exc:
            LOGGER.warning("Could not fetch models: %s", exc)
            # Use current models as fallback
            available_models = [
                current_options.get(
                    CONF_ROUTER_MODEL,
                    current_data.get(CONF_ROUTER_MODEL, DEFAULT_ROUTER_MODEL),
                ),
                current_options.get(
                    CONF_CHAT_MODEL,
                    current_data.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL),
                ),
            ]
            # Remove duplicates
            available_models = list(set(available_models))

        model_options = [
            selector.SelectOptionDict(value=model, label=model)
            for model in available_models
        ]

        current_router = current_options.get(
            CONF_ROUTER_MODEL,
            current_data.get(CONF_ROUTER_MODEL, DEFAULT_ROUTER_MODEL),
        )
        current_chat = current_options.get(
            CONF_CHAT_MODEL,
            current_data.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL),
        )
        current_keep_alive = current_options.get(
            CONF_KEEP_ALIVE_PERSISTENT,
            current_data.get(CONF_KEEP_ALIVE_PERSISTENT, True),
        )

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_ROUTER_MODEL,
                        default=current_router,
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=model_options,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        ),
                    ),
                    vol.Required(
                        CONF_CHAT_MODEL,
                        default=current_chat,
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=model_options,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        ),
                    ),
                    vol.Required(
                        CONF_KEEP_ALIVE_PERSISTENT,
                        default=current_keep_alive,
                    ): selector.BooleanSelector(),
                },
            ),
            errors=errors,
        )
