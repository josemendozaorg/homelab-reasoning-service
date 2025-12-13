"""API route definitions for the reasoning service."""
import logging
import time
from fastapi import APIRouter, HTTPException

from src.config import settings
from src.llm import OllamaClient
from src.reasoning import create_reasoning_graph
from src.reasoning.state import create_initial_state
from .models import ReasoningRequest, ReasoningResponse, HealthResponse, TestInferenceResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Create the reasoning graph once at module load
_reasoning_graph = None


def get_reasoning_graph():
    """Get or create the reasoning graph singleton."""
    global _reasoning_graph
    if _reasoning_graph is None:
        _reasoning_graph = create_reasoning_graph()
    return _reasoning_graph


@router.post("/v1/reason", response_model=ReasoningResponse)
async def reason(request: ReasoningRequest) -> ReasoningResponse:
    """Submit a reasoning task with self-correction loop.

    The service will:
    1. Generate initial reasoning using DeepSeek-R1's <think> tokens
    2. Critique the answer for logical errors
    3. Refine and iterate until approved or max iterations reached

    Args:
        request: The reasoning request with query and optional parameters.

    Returns:
        The reasoning response with trace, final answer, and metadata.
    """
    logger.info(f"Received reasoning request: {request.query[:100]}...")

    # Override settings if provided
    if request.max_iterations:
        # Note: This is a simplified override. In production, pass through state.
        pass

    # Create initial state
    initial_state = create_initial_state(request.query)

    # Run the reasoning graph
    graph = get_reasoning_graph()

    try:
        result = await graph.ainvoke(initial_state)
    except Exception as e:
        logger.error(f"Reasoning failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Reasoning process failed: {str(e)}"
        )

    # Determine if answer was approved
    critique = result.get("critique", "")
    is_approved = "APPROVED" in critique.upper()

    return ReasoningResponse(
        query=request.query,
        reasoning_trace=result.get("reasoning_trace", []),
        final_answer=result.get("final_answer", result.get("current_answer", "")),
        iterations=result.get("iteration", 0),
        is_approved=is_approved
    )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check service health and Ollama connectivity.

    Returns:
        Health status including model configuration and Ollama connection state.
    """
    # Check Ollama connectivity
    client = OllamaClient()
    ollama_connected = await client.health_check()

    return HealthResponse(
        status="healthy" if ollama_connected else "degraded",
        model=settings.ollama_model,
        ollama_connected=ollama_connected
    )

@router.post("/v1/test-inference", response_model=TestInferenceResponse)
async def test_inference() -> TestInferenceResponse:
    """Run a fast inference check (max 10 tokens)."""
    start_time = time.time()
    
    try:
        async with OllamaClient() as client:
            response = await client.generate(
                prompt="Say hello!",
                max_tokens=10,
                temperature=0.7
            )
            
        duration = (time.time() - start_time) * 1000
        
        return TestInferenceResponse(
            status="ok",
            response=response.strip(),
            duration_ms=duration,
            model=settings.ollama_model
        )
            
    except Exception as e:
        logger.error(f"Inference check failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Inference check failed: {str(e)}"
        )
