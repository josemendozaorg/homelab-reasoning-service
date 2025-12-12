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


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(description="Service health status")
    model: str = Field(description="Configured LLM model")
    ollama_connected: bool = Field(description="Whether Ollama is reachable")
