"""Config flow for Optimal LLM HAOS integration."""

from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.helpers import selector
from homeassistant.helpers.aiohttp_client import async_create_clientsession

from .const import (
    CONF_AGGRESSIVE_CACHING,
    CONF_MODEL,
    CONF_OLLAMA_URL,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    DOMAIN,
    LOGGER,
)
from .ollama_client import (
    OllamaClient,
    OllamaClientCommunicationError,
    OllamaClientError,
)


class OllamaFlowHandler(config_entries.ConfigFlow, domain=DOMAIN):
    """Config flow for Optimal LLM HAOS."""

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
                client = OllamaClient(
                    base_url=self._ollama_url,
                    session=async_create_clientsession(self.hass),
                )
                self._available_models = await client.async_get_model_names()

                if not self._available_models:
                    errors["base"] = "no_models"
                else:
                    # Connection successful, proceed to model selection
                    return await self.async_step_model()

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

    async def async_step_model(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> config_entries.ConfigFlowResult:
        """Handle the model selection step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Create the config entry
            model = user_input[CONF_MODEL]
            aggressive_caching = user_input.get(CONF_AGGRESSIVE_CACHING, True)
            llm_hass_api = user_input.get(CONF_LLM_HASS_API, "assist")

            # Set unique ID based on URL and model
            await self.async_set_unique_id(f"{self._ollama_url}_{model}")
            self._abort_if_unique_id_configured()

            return self.async_create_entry(
                title=f"Ollama ({model})",
                data={
                    CONF_OLLAMA_URL: self._ollama_url,
                    CONF_MODEL: model,
                    CONF_AGGRESSIVE_CACHING: aggressive_caching,
                    CONF_LLM_HASS_API: llm_hass_api,
                },
            )

        # Build model selector options
        model_options = [
            selector.SelectOptionDict(value=model, label=model)
            for model in self._available_models
        ]

        # Default to first model or DEFAULT_MODEL if available
        default_model = DEFAULT_MODEL
        if DEFAULT_MODEL in self._available_models:
            default_model = DEFAULT_MODEL
        elif self._available_models:
            default_model = self._available_models[0]

        return self.async_show_form(
            step_id="model",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_MODEL,
                        default=default_model,
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=model_options,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        ),
                    ),
                    vol.Required(
                        CONF_AGGRESSIVE_CACHING,
                        default=True,
                    ): selector.BooleanSelector(),
                    vol.Optional(
                        CONF_LLM_HASS_API,
                        default="assist",
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[
                                selector.SelectOptionDict(
                                    value="assist",
                                    label="Assist (Recommended)",
                                ),
                                selector.SelectOptionDict(
                                    value="none",
                                    label="None",
                                ),
                            ],
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        ),
                    ),
                },
            ),
            errors=errors,
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Get the options flow for this handler."""
        return OllamaOptionsFlowHandler(config_entry)


class OllamaOptionsFlowHandler(config_entries.OptionsFlow):
    """Options flow for Optimal LLM HAOS."""

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
            # Update the config entry options
            return self.async_create_entry(title="", data=user_input)

        # Get current values
        current_data = self._config_entry.data

        # Fetch available models for dropdown
        available_models = []
        try:
            client = OllamaClient(
                base_url=current_data[CONF_OLLAMA_URL],
                session=async_create_clientsession(self.hass),
            )
            available_models = await client.async_get_model_names()
        except OllamaClientError as exc:
            LOGGER.warning("Could not fetch models: %s", exc)
            # Use current model as fallback
            available_models = [current_data.get(CONF_MODEL, DEFAULT_MODEL)]

        model_options = [
            selector.SelectOptionDict(value=model, label=model)
            for model in available_models
        ]

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_MODEL,
                        default=current_data.get(CONF_MODEL, DEFAULT_MODEL),
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=model_options,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        ),
                    ),
                    vol.Required(
                        CONF_AGGRESSIVE_CACHING,
                        default=current_data.get(CONF_AGGRESSIVE_CACHING, True),
                    ): selector.BooleanSelector(),
                },
            ),
            errors=errors,
        )
