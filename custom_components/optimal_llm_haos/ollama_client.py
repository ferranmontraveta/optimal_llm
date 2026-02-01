"""Ollama API Client for optimal_llm_haos."""

from __future__ import annotations

import asyncio
import socket
from typing import Any

import aiohttp

from .const import DEFAULT_TIMEOUT, LOGGER


class OllamaClientError(Exception):
    """Base exception for Ollama client errors."""


class OllamaClientCommunicationError(OllamaClientError):
    """Exception for communication errors with Ollama server."""


class OllamaServerRestartedError(OllamaClientError):
    """Exception when Ollama server has been restarted (KV cache cleared)."""


class OllamaClient:
    """Async client for Ollama API."""

    def __init__(
        self,
        base_url: str,
        session: aiohttp.ClientSession,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        """
        Initialize the Ollama client.

        Args:
            base_url: Base URL for Ollama server (e.g., http://localhost:11434)
            session: aiohttp ClientSession for making requests
            timeout: Request timeout in seconds

        """
        self._base_url = base_url.rstrip("/")
        self._session = session
        self._timeout = timeout
        self._last_response_time: float | None = None

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
                response = await self._session.get(
                    f"{self._base_url}/api/tags",
                )
                response.raise_for_status()
                data = await response.json()
                return data.get("models", [])
        except TimeoutError as exc:
            msg = f"Timeout connecting to Ollama at {self._base_url}"
            raise OllamaClientCommunicationError(msg) from exc
        except aiohttp.ClientError as exc:
            msg = f"Error connecting to Ollama at {self._base_url}: {exc}"
            raise OllamaClientCommunicationError(msg) from exc
        except socket.gaierror as exc:
            msg = f"Cannot resolve Ollama host {self._base_url}: {exc}"
            raise OllamaClientCommunicationError(msg) from exc

    async def async_get_model_names(self) -> list[str]:
        """
        Get list of available model names.

        Returns:
            List of model name strings

        """
        models = await self.async_get_models()
        return [model["name"] for model in models]

    async def async_chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        keep_alive: str = "5m",
        stream: bool = False,
    ) -> dict[str, Any]:
        """
        Send chat request to Ollama.

        Args:
            model: Name of the model to use
            messages: List of message dicts with 'role' and 'content' keys
            keep_alive: How long to keep model loaded (e.g., "5m", "1h")
            stream: Whether to stream response (not implemented)

        Returns:
            Response dict containing 'message' with assistant response

        Raises:
            OllamaClientCommunicationError: If connection fails
            OllamaServerRestartedError: If server was restarted (cache cleared)

        """
        import time

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "keep_alive": keep_alive,
        }

        try:
            start_time = time.monotonic()
            async with asyncio.timeout(self._timeout):
                response = await self._session.post(
                    f"{self._base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                result = await response.json()

            elapsed = time.monotonic() - start_time
            LOGGER.debug(
                "Ollama chat completed in %.2fs for model %s",
                elapsed,
                model,
            )

            # Track response time for cache miss detection
            self._last_response_time = elapsed

            return result

        except TimeoutError as exc:
            msg = f"Timeout waiting for Ollama response (>{self._timeout}s)"
            raise OllamaClientCommunicationError(msg) from exc
        except aiohttp.ClientError as exc:
            # Check for server restart indicators
            if "context" in str(exc).lower():
                raise OllamaServerRestartedError(
                    "Ollama server appears to have been restarted"
                ) from exc
            msg = f"Error communicating with Ollama: {exc}"
            raise OllamaClientCommunicationError(msg) from exc

    async def async_chat_with_recovery(
        self,
        model: str,
        messages: list[dict[str, str]],
        warm_cache_callback: Any = None,
        keep_alive: str = "5m",
    ) -> dict[str, Any]:
        """
        Send chat request with automatic cache recovery.

        If the Ollama server has been restarted (clearing KV cache),
        this method will attempt to re-warm the cache and retry.

        Args:
            model: Name of the model to use
            messages: List of message dicts
            warm_cache_callback: Async callback to warm cache if needed
            keep_alive: How long to keep model loaded

        Returns:
            Response dict containing 'message' with assistant response

        """
        try:
            response = await self.async_chat(model, messages, keep_alive)

            # Check for sudden slowdown indicating cache miss
            if self._detect_cache_miss(response):
                LOGGER.warning("Cache miss detected, re-warming cache")
                if warm_cache_callback:
                    await warm_cache_callback()

            return response

        except OllamaServerRestartedError:
            LOGGER.warning("Ollama server restarted, re-warming cache")
            if warm_cache_callback:
                await warm_cache_callback()
            # Retry the request
            return await self.async_chat(model, messages, keep_alive)

    def _detect_cache_miss(self, response: dict[str, Any]) -> bool:
        """
        Detect if response indicates a cache miss.

        A sudden increase in response time may indicate the KV cache
        was cleared and the base prompt had to be re-embedded.

        Args:
            response: Response from Ollama API

        Returns:
            True if cache miss is suspected

        """
        # Check eval_count and eval_duration for unusual patterns
        if "eval_count" in response and "eval_duration" in response:
            eval_count = response.get("eval_count", 0)
            prompt_eval_count = response.get("prompt_eval_count", 0)

            # If prompt_eval_count is very high, the cache wasn't used
            if prompt_eval_count > 1000:
                LOGGER.debug(
                    "High prompt_eval_count (%d) suggests cache miss",
                    prompt_eval_count,
                )
                return True

        return False

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
