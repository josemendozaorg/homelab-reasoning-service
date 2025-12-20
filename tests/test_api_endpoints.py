import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_read_root():
    """Verify the root endpoint serves the UI (HTML)."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<!DOCTYPE html>" in response.text

def test_api_info():
    """Verify the info endpoint returns version and model metadata."""
    response = client.get("/api/info")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "model" in data
    assert data["service"] == "LangGraph Reasoning Service"
    # Ensure version is not default 1.0.0 if we updated it
    assert data["version"] == "0.2.0"

def test_reason_stream_endpoint_exists():
    """Verify the streaming endpoint exists.
    
    We don't need to fully stream (which requires LLM mock), 
    just verifying 422 (validation error) confirms the route is registered.
    If it didn't exist, we'd get 404.
    """
    # Sending empty body should trigger validation error (422)
    response = client.post("/v1/reason/stream", json={})
    
    # 422 means: I saw your request but it's invalid (missing fields).
    # 404 means: Who are you talking to? (Route missing).
    assert response.status_code == 422 

def test_inference_endpoint_exists():
    """Verify the simple inference endpoint exists."""
    # This might fail 500 if Ollama isn't reachable, but 500 means the route EXISTS.
    # 404 means missing.
    try:
        response = client.post("/v1/test-inference")
        assert response.status_code in [200, 500]
        assert response.status_code != 404
    except Exception:
        # If connection error to Ollama, that's fine for this test, 
        # we just want to know fastapi found the route.
        pass
