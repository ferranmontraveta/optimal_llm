"""Ollama API Client for optimal_agent with dual-model support."""

from __future__ import annotations

import asyncio
from typing import Any

import ollama

from .const import CHAT_CONTEXT_LIMIT, DEFAULT_TIMEOUT, LOGGER, ROUTER_CONTEXT_LIMIT


class OllamaClientError(Exception):
    """Base exception for Ollama client errors."""


class OllamaClientCommunicationError(OllamaClientError):
    """Exception for communication errors with Ollama server."""


class OllamaClient:
    """Async client for Ollama API with dual-model support."""

    def __init__(
        self,
        base_url: str,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        """
        Initialize the Ollama client.

        Args:
            base_url: Base URL for Ollama server (e.g., http://localhost:11434)
            timeout: Request timeout in seconds

        """
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = ollama.AsyncClient(host=base_url)

    @property
    def base_url(self) -> str:
        """Return the base URL."""
        return self._base_url

    async def async_get_models(self) -> list[dict[str, Any]]:
        """
        Fetch available models from Ollama server.

        Returns:
            List of model dictionaries with name, size, etc.

        Raises:
            OllamaClientCommunicationError: If connection fails

        """
        try:
            async with asyncio.timeout(30):
                response = await self._client.list()
                return response.get("models", [])
        except TimeoutError as exc:
            msg = f"Timeout connecting to Ollama at {self._base_url}"
            raise OllamaClientCommunicationError(msg) from exc
        except ollama.RequestError as exc:
            msg = f"Error connecting to Ollama at {self._base_url}: {exc}"
            raise OllamaClientCommunicationError(msg) from exc

    async def async_get_model_names(self) -> list[str]:
        """
        Get list of available model names.

        Returns:
            List of model name strings

        """
        models = await self.async_get_models()
        return [model.get("name", model.get("model", "")) for model in models]

    async def async_preload_models(
        self,
        router_model: str,
        chat_model: str,
        keep_alive: int = -1,
    ) -> dict[str, bool]:
        """
        Pre-warm both router and chat models with persistent loading.

        Sends empty requests to both models with keep_alive: -1 to ensure
        they are pre-loaded before the first user interaction.

        Args:
            router_model: Name of the router model
            chat_model: Name of the chat model
            keep_alive: How long to keep models loaded (-1 for indefinite)

        Returns:
            Dict with model names as keys and success status as values

        """
        LOGGER.info(
            "Pre-loading models: router=%s, chat=%s (keep_alive=%s)",
            router_model,
            chat_model,
            keep_alive,
        )

        results = {}

        # Warm router model
        try:
            await self._warm_model(
                model=router_model,
                keep_alive=keep_alive,
                context_limit=ROUTER_CONTEXT_LIMIT,
            )
            results[router_model] = True
            LOGGER.info("Router model %s pre-loaded successfully", router_model)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to pre-load router model %s: %s", router_model, exc)
            results[router_model] = False

        # Warm chat model (skip if same as router)
        if chat_model != router_model:
            try:
                await self._warm_model(
                    model=chat_model,
                    keep_alive=keep_alive,
                    context_limit=CHAT_CONTEXT_LIMIT,
                )
                results[chat_model] = True
                LOGGER.info("Chat model %s pre-loaded successfully", chat_model)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Failed to pre-load chat model %s: %s", chat_model, exc)
                results[chat_model] = False
        else:
            results[chat_model] = results.get(router_model, False)

        return results

    async def _warm_model(
        self,
        model: str,
        keep_alive: int | str,
        context_limit: int,
    ) -> None:
        """
        Send a warm-up request to pre-load a model.

        Args:
            model: Model name
            keep_alive: Keep alive duration
            context_limit: Context window size

        """
        messages = [
            ollama.Message(role="user", content="Hello"),
        ]

        async with asyncio.timeout(self._timeout):
            await self._client.chat(
                model=model,
                messages=messages,
                keep_alive=keep_alive,
                options={"num_ctx": context_limit},
            )

    async def async_chat(
        self,
        model: str,
        messages: list[ollama.Message],
        keep_alive: int | str = "5m",
        context_limit: int | None = None,
        stream: bool = False,
    ) -> dict[str, Any] | ollama.ChatResponse:
        """
        Send a chat request to Ollama.

        Args:
            model: Name of the model to use
            messages: List of message dicts
            keep_alive: How long to keep model loaded
            context_limit: Optional context window size
            stream: Whether to stream the response

        Returns:
            Response dict or async generator if streaming

        Raises:
            OllamaClientCommunicationError: If communication fails

        """
        options = {}
        if context_limit:
            options["num_ctx"] = context_limit

        try:
            async with asyncio.timeout(self._timeout):
                return await self._client.chat(
                    model=model,
                    messages=messages,
                    keep_alive=keep_alive,
                    options=options if options else None,
                    stream=stream,
                )
        except TimeoutError as exc:
            msg = f"Timeout waiting for Ollama response (>{self._timeout}s)"
            raise OllamaClientCommunicationError(msg) from exc
        except ollama.RequestError as exc:
            msg = f"Error communicating with Ollama: {exc}"
            raise OllamaClientCommunicationError(msg) from exc

    async def async_chat_stream(
        self,
        model: str,
        messages: list[ollama.Message],
        keep_alive: int | str = "5m",
        context_limit: int | None = None,
    ) -> Any:
        """
        Send a streaming chat request to Ollama.

        Args:
            model: Name of the model to use
            messages: List of message dicts
            keep_alive: How long to keep model loaded
            context_limit: Optional context window size

        Returns:
            Async generator yielding response chunks

        """
        options = {}
        if context_limit:
            options["num_ctx"] = context_limit

        try:
            return await self._client.chat(
                model=model,
                messages=messages,
                keep_alive=keep_alive,
                options=options if options else None,
                stream=True,
            )
        except ollama.RequestError as exc:
            msg = f"Error communicating with Ollama: {exc}"
            raise OllamaClientCommunicationError(msg) from exc

    async def async_test_connection(self) -> bool:
        """
        Test if Ollama server is reachable.

        Returns:
            True if connection successful, False otherwise

        """
        try:
            await self.async_get_models()
            return True
        except OllamaClientError:
            return False
