"""Pydantic models for API request/response schemas."""
from typing import Optional
from pydantic import BaseModel, Field


class ReasoningRequest(BaseModel):
    """Request model for submitting a reasoning task."""

    query: str = Field(
        ...,
        description="The question or problem to reason about",
        examples=["What are the implications of Godel's incompleteness theorems for AGI?"]
    )
    model: Optional[str] = Field(
        default=None,
        description="Model to use for reasoning (defaults to server configuration)"
    )
    fast_model: Optional[str] = Field(
        default=None,
        description="Model to use for fast execution (defaults to server configuration)"
    )
    max_iterations: Optional[int] = Field(
        default=None,
        description="Maximum reasoning iterations (overrides default)",
        ge=1,
        le=20
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Sampling temperature for generation",
        ge=0.0,
        le=2.0
    )
    history: list[dict] = Field(
        default=[],
        description="Previous conversation history (list of messages)"
    )
    search_provider: str = Field(
        default="ddg",
        description="Search provider to use (tavily, google, brave, ddg)"
    )
    search_api_key: Optional[str] = Field(
        default=None,
        description="API key for the selected search provider"
    )
    search_cse_id: Optional[str] = Field(
        default=None,
        description="Custom Search Engine ID (required for Google)"
    )
    search_api_keys: Optional[dict[str, str]] = Field(
        default={},
        description="Dictionary of API keys for specific providers (e.g. {'exa': '...', 'tavily': '...'})"
    )


class ReasoningResponse(BaseModel):
    """Response model for a completed reasoning task."""

    query: str = Field(description="The original query")
    reasoning_trace: list[str] = Field(
        description="List of reasoning steps with <think> output"
    )
    final_answer: str = Field(description="The final approved answer")
    iterations: int = Field(description="Number of reasoning iterations performed")
    is_approved: bool = Field(
        description="Whether the answer was explicitly approved by critique"
    )


class TestInferenceResponse(BaseModel):
    """Response model for the fast inference test."""
    
    status: str = "ok"
    response: str
    duration_ms: float
    model: str = Field(
        description="The LLM model used for inference"
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(description="Service health status")
    model: str = Field(description="Configured LLM model")
    ollama_connected: bool = Field(description="Whether Ollama is reachable")


class ModelInfo(BaseModel):
    """Information about an available model."""

    name: str = Field(description="Model name (e.g., 'deepseek-r1:14b')")
    size: int = Field(default=0, description="Model size in bytes")
    modified_at: str = Field(default="", description="Last modified timestamp")


class ModelsResponse(BaseModel):
    """Response model for listing available models."""

    models: list[ModelInfo] = Field(description="List of available models")
    default: str = Field(description="Default model from configuration")
    default_fast: str = Field(description="Default fast model from configuration")
