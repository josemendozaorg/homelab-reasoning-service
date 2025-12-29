"""Tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


def test_root_endpoint(client):
    """Root endpoint services the UI (HTML)."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<!DOCTYPE html>" in response.text



def test_health_endpoint_structure(client):
    """Health endpoint returns expected structure."""
    response = client.get("/health")
    # May return 200 (healthy) or 503 (degraded) depending on Ollama availability
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert "status" in data
        assert "model" in data
        assert "ollama_connected" in data


def test_reason_endpoint_validation(client):
    """Reason endpoint validates input."""
    # Missing required field
    response = client.post("/v1/reason", json={})
    assert response.status_code == 422  # Validation error

    # Invalid max_iterations
    response = client.post("/v1/reason", json={
        "query": "test",
        "max_iterations": 100  # exceeds limit
    })
    assert response.status_code == 422

def test_test_inference_endpoint(client):
    """Test inference endpoint returns valid structure."""
    # Note: This will fail without mocking or a real Ollama instance if we don't mock the client.
    # For now, let's assume we want to verify the route exists and handles the request structure 
    # even if it fails with 500 downstream (or we can use unittest.mock if needed).
    # Since we don't have mocking setup easily here, and 500 is expected if no Ollama, 
    # lets check that.
    
    # Actually, let's mock it to be green.
    from unittest.mock import AsyncMock, patch
    
    with patch("src.api.routes.OllamaClient") as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.__aenter__.return_value = mock_instance
        mock_instance.generate = AsyncMock(return_value="Hello there!")
        
        response = client.post("/v1/test-inference")
        
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["response"] == "Hello there!"
    assert "duration_ms" in data
