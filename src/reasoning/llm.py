import httpx
import logging
import json
from src.config import settings

logger = logging.getLogger(__name__)

SEARCH_TOOL_ENABLED = True  # Verified capable of web search

class OllamaClient:
    def __init__(self, base_url: str = settings.ollama_base_url, model: str = settings.ollama_model):
        self.base_url = base_url
        self.model = model
    
    async def generate(self, prompt: str, system: str = None, temperature: float = settings.temperature, model: str = None, api_key: str = None) -> str:
        """Generate text using Ollama API.

        Args:
            prompt: The prompt to generate from.
            system: Optional system prompt.
            temperature: Sampling temperature.
            model: Optional model override (uses instance model if not specified).
            api_key: Optional API key for authentication.
        """
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": settings.max_context_tokens
            }
        }

        if system:
            payload["system"] = system

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient(headers=headers) as client:
            try:
                response = await client.post(url, json=payload, timeout=60.0)
                response.raise_for_status()
                return response.json().get("response", "")
            except Exception as e:
                logger.error(f"Ollama generation failed: {e}")
                raise

    async def chat(self, messages: list, temperature: float = settings.temperature, model: str = None, api_key: str = None) -> str:
        """Chat using Ollama API.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Sampling temperature.
            model: Optional model override (uses instance model if not specified).
            api_key: Optional API key for authentication.
        """
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": settings.max_context_tokens
            }
        }

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient(headers=headers) as client:
            try:
                response = await client.post(url, json=payload, timeout=60.0)
                response.raise_for_status()
                # Ollama chat response structure
                return response.json().get("message", {}).get("content", "")
            except Exception as e:
                logger.error(f"Ollama chat failed: {e}")
                raise

    async def generate_stream(self, prompt: str, system: str = None, temperature: float = settings.temperature, model: str = None, api_key: str = None):
        """Stream text generation using Ollama API.

        Args:
            prompt: The prompt to generate from.
            system: Optional system prompt.
            temperature: Sampling temperature.
            model: Optional model override (uses instance model if not specified).
            api_key: Optional API key for authentication.
        """
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_ctx": settings.max_context_tokens
            }
        }

        if system:
            payload["system"] = system

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient(headers=headers) as client:
            try:
                async with client.stream("POST", url, json=payload, timeout=60.0) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            data = json.loads(line)
                            token = data.get("response", "")
                            if token:
                                yield token
                            if data.get("done"):
                                break
            except Exception as e:
                logger.error(f"Ollama streaming generation failed: {e}")
                raise

    async def chat_stream(self, messages: list, temperature: float = settings.temperature, model: str = None, api_key: str = None):
        """Stream chat using Ollama API.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Sampling temperature.
            model: Optional model override (uses instance model if not specified).
            api_key: Optional API key for authentication.
        """
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_ctx": settings.max_context_tokens
            }
        }

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient(headers=headers) as client:
            try:
                async with client.stream("POST", url, json=payload, timeout=60.0) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            data = json.loads(line)
                            token = data.get("message", {}).get("content", "")
                            if token:
                                yield token
                            if data.get("done"):
                                break
            except Exception as e:
                logger.error(f"Ollama streaming chat failed: {e}")
                raise


def get_model_from_config(config: dict) -> str:
    """Extract model from LangGraph config, falling back to default."""
    return config.get("configurable", {}).get("model") or settings.ollama_model


# Global instance
llm = OllamaClient()
