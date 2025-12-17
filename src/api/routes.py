from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse
import json
import logging
import time
import asyncio


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
        raise HTTPException(
            status_code=500, 
            detail=f"Inference check failed: {str(e)}"
        )


@router.post("/v1/reason/stream")
async def reason_stream(request: ReasoningRequest, req: Request):
    """Stream reasoning progress using Server-Sent Events (SSE)."""
    
    async def event_generator():
        # Create initial state
        initial_state = create_initial_state(request.query, request.history)
        graph = get_reasoning_graph()
        
        queue = asyncio.Queue()
        
        async def producer():
            try:
                async for event in graph.astream_events(initial_state, version="v1"):
                    await queue.put({"type": "event", "payload": event})
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                await queue.put({"type": "error", "payload": str(e)})
            finally:
                await queue.put({"type": "done"})

        # Start producer task
        producer_task = asyncio.create_task(producer())
        
        while True:
            try:
                # Wait for event with timeout for keep-alive
                item = await asyncio.wait_for(queue.get(), timeout=15.0)
                
                msg_type = item.get("type")
                
                if msg_type == "done":
                    yield {"event": "done", "data": "Reasoning complete"}
                    break
                
                elif msg_type == "error":
                    yield {"event": "error", "data": item["payload"]}
                    break
                
                elif msg_type == "event":
                    event = item["payload"]
                    # We are interested in chat model stream events (if using LangChain/native streaming)
                    # BUT since we use DSPy (non-streaming node), we rely on node output
                    if event["event"] == "on_chain_end":
                         # Check for tool node output
                        node_name = event.get("metadata", {}).get("langgraph_node")
                        if node_name == "tool":
                            output = event["data"]["output"]
                            if output and "reasoning_trace" in output:
                                trace = output["reasoning_trace"]
                                if trace:
                                    last_item = trace[-1]
                                    data = {
                                        "token": "\n" + last_item + "\n",
                                        "node": "tool"
                                    }
                                    yield {"data": json.dumps(data)}
                        elif node_name == "reason" or node_name == "critique":
                             # For DSPy nodes (non-streaming), we get the chunk at the end
                             output = event["data"]["output"]
                             # We need to extract the "delta" manually by comparing traces?
                             # Or just send the latest trace item?
                             # Logic: reason_node appends to reasoning_trace.
                             # If we can get update from output:
                             if output and "reasoning_trace" in output:
                                trace = output["reasoning_trace"]
                                if trace:
                                   last_item = trace[-1]
                                   # HEURISTIC: Only emit if it looks like a new item (contains iteration header or similar)
                                   # Actually, just emit it. The UI appends.
                                   data = {
                                       "token": "\n" + last_item + "\n",
                                       "node": node_name
                                   }
                                   yield {"data": json.dumps(data)}

            except asyncio.TimeoutError:
                # Send keep-alive ping
                yield {"event": "ping", "data": ""}

        # Clean up
        if not producer_task.done():
            producer_task.cancel()
            try:
                await producer_task
            except asyncio.CancelledError:
                pass

    return EventSourceResponse(event_generator())
