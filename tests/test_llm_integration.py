import pytest
from src.llm import OllamaClient

@pytest.mark.asyncio
async def test_ollama_connectivity():
    """Verify Ollama is reachable."""
    async with OllamaClient() as client:
        is_healthy = await client.health_check()
    assert is_healthy, "Ollama service is not healthy/reachable"

@pytest.mark.asyncio
async def test_ollama_generation():
    """Verify Ollama can generate text."""
    async with OllamaClient() as client:
        response = await client.generate("Say 'test' and nothing else.", temperature=0.0)
    
    assert response, "Ollama returned empty response"
    assert "test" in response.lower(), f"Unexpected response: {response}"
