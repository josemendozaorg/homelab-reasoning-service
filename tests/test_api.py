"""Tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


def test_root_endpoint(client):
    """Root endpoint returns service info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "LangGraph Reasoning Service"
    assert "version" in data
    assert "model" in data


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
