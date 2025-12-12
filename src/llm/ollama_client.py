"""Async Ollama API client wrapper for inference."""
import httpx
from typing import Optional
import logging

from src.config import settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """Async client for Ollama API with DeepSeek-R1 support."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 300.0  # 5 minutes for reasoning tasks
    ):
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.ollama_model
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate a response from the model.

        Args:
            prompt: The user prompt to generate a response for.
            system: Optional system prompt to set context.
            temperature: Sampling temperature (default from settings).
            max_tokens: Maximum tokens to generate.

        Returns:
            The generated text response.
        """
        if self._client is None:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or settings.temperature,
            }
        }

        if system:
            payload["system"] = system

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        logger.debug(f"Sending request to Ollama: model={self.model}, prompt_len={len(prompt)}")

        response = await self._client.post("/api/generate", json=payload)
        response.raise_for_status()

        data = response.json()
        return data.get("response", "")

    async def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None
    ) -> str:
        """Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Sampling temperature.

        Returns:
            The assistant's response content.
        """
        if self._client is None:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or settings.temperature,
            }
        }

        response = await self._client.post("/api/chat", json=payload)
        response.raise_for_status()

        data = response.json()
        return data.get("message", {}).get("content", "")

    async def health_check(self) -> bool:
        """Check if Ollama service is available.

        Returns:
            True if service is healthy, False otherwise.
        """
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=5.0) as client:
                response = await client.get("/")
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False
