"""Node implementations for the reasoning graph."""
import re
import logging
from typing import Any
import dspy

from src.config import settings
from .state import ReasoningState
from .signatures import ReasonSignature, CritiqueSignature, RefineSignature, CritiqueSearchSignature
from .tools import perform_web_search
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

# Configure DSPy with Ollama
# Using dspy.LM for generic provider support (DSPy 3.x)
lm = dspy.LM(
    f"ollama/{settings.ollama_model}",
    api_base=settings.ollama_base_url,
    temperature=settings.temperature,
    max_tokens=settings.max_context_tokens,
    api_key="nomatter" # dummy key often needed
)
dspy.configure(lm=lm)


import asyncio
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential, retry_if_exception_type

# ... imports ...

# Helper to wrap DSPy calls with retry
async def predict_with_retry(predictor, **kwargs):
    """Invoke DSPy predictor with retry logic in a thread."""
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    ):
        with attempt:
            try:
                # Run synchronous DSPy call in a separate thread to avoid blocking event loop
                return await asyncio.to_thread(predictor, **kwargs)
            except Exception as e:
                logger.error(f"DSPy invocation failed: {e}")
                raise


def parse_reasoning_response(response: str) -> tuple[str, str]:
    """Parse a response to extract reasoning and answer.

    Args:
        response: The full model response.

    Returns:
        Tuple of (reasoning_trace, answer).
    """
    # Extract content within <think> tags
    think_pattern = r"<think>(.*?)</think>"
    think_matches = re.findall(think_pattern, response, re.DOTALL)
    reasoning = "\n".join(think_matches) if think_matches else ""

    # Everything after </think> is the answer
    answer_pattern = r"</think>\s*(.*?)$"
    answer_match = re.search(answer_pattern, response, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else response.strip()

    return reasoning, answer


def parse_search_request(response: str) -> str | None:
    """Parse a response to check for search requests.
    
    Args:
        response: The full model response.
        
    Returns:
        The search query if found, else None.
    """
    search_pattern = r"<search>(.*?)</search>"
    search_match = re.search(search_pattern, response, re.DOTALL)
    return search_match.group(1).strip() if search_match else None


def format_history(history: list) -> str:
    """Format chat history for context."""
    formatted = []
    for msg in history:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        formatted.append(f"{role.upper()}: {content}")
    return "\n".join(formatted)


async def reason_node(state: ReasoningState) -> dict[str, Any]:
    """Generate reasoning using DSPy and DeepSeek-R1's native think tokens.

    Args:
        state: Current reasoning state.

    Returns:
        Updated state fields with new reasoning trace and answer.
    """
    response_text = ""
    
    # Get current time context
    from datetime import datetime
    current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    time_context = f"Current Date/Time: {current_time}\n"

    if state["critique"]:
        # We're refining based on critique
        if state["current_answer"]:
            # Refine answer
            refine = dspy.Predict(RefineSignature)
            pred = await predict_with_retry(
                refine,
                question=state["query"],
                previous_answer=state["current_answer"],
                critique=f"{time_context}{state['critique']}"
            )
            response_text = pred.improved_response
        else:
            # Critique of search results -> formulate answer or new search
            # We treat this as a "reasoning" step but with search context potentially
            # Actually, the prompt in previous implementation was complex.
            # Let's map it to ReasonSignature but with context.
            reason = dspy.Predict(ReasonSignature)
            context = f"{time_context}Search Results Critique: {state['critique']}"
            pred = await predict_with_retry(
                reason,
                question=state["query"],
                context=context
            )
            response_text = pred.response
    else:
        # Initial reasoning
        reason = dspy.Predict(ReasonSignature)
        history_str = format_history(state.get("chat_history", []))
        full_context = f"{time_context}{history_str if history_str else 'No previous context.'}"
        pred = await predict_with_retry(
            reason,
            question=state["query"],
            context=full_context
        )
        response_text = pred.response

    # Parse output
    reasoning, answer = parse_reasoning_response(response_text)

    new_trace = state["reasoning_trace"].copy()
    if reasoning:
        new_trace.append(f"[Iteration {state['iteration'] + 1}]\n{reasoning}")

    search_query = parse_search_request(response_text)
    
    if search_query:
        logger.info(f"Reason node iteration {state['iteration'] + 1}: requested search for '{search_query}'")
        new_trace.append(f"[Search Requested: {search_query}]")
        return {
            "reasoning_trace": new_trace,
            "current_answer": None,
            "pending_search_query": search_query,
            "iteration": state["iteration"] # Keep iteration same or increment? kept same in logic before comment
        }

    logger.info(f"Reason node iteration {state['iteration'] + 1}: generated {len(answer)} char answer")

    return {
        "reasoning_trace": new_trace,
        "current_answer": answer,
        "iteration": state["iteration"] + 1,
        "pending_search_query": None
    }


async def tool_node(state: ReasoningState) -> dict[str, Any]:
    """Execute pending tool calls.
    
    Args:
        state: Current reasoning state.
        
    Returns:
        Updated state with tool results.
    """
    query = state.get("pending_search_query")
    if not query:
        return {}
        
    results = await perform_web_search(query)
    
    new_trace = state["reasoning_trace"].copy()
    new_trace.append(f"[Search Results]\n{results}")
    
    return {
        "reasoning_trace": new_trace,
        "pending_search_query": None
    }


async def critique_node(state: ReasoningState) -> dict[str, Any]:
    """Evaluate the current answer for errors or improvements.

    Args:
        state: Current reasoning state.

    Returns:
        Updated state with critique.
    """
    args_answer = state.get("current_answer")
    critique_text = ""
    
    if args_answer:
        # Standard critique of an answer
        critique_module = dspy.Predict(CritiqueSignature)
        trace_context = "\n".join(state["reasoning_trace"][-3:])
        pred = await predict_with_retry(
            critique_module,
            question=state["query"],
            reasoning_trace=trace_context,
            answer=args_answer
        )
        critique_text = pred.critique
    else:
        # Critique of search results
        critique_search = dspy.Predict(CritiqueSearchSignature)
        last_trace = state["reasoning_trace"][-1] if state["reasoning_trace"] else "No trace"
        pred = await predict_with_retry(
            critique_search,
            question=state["query"],
            search_results=last_trace
        )
        critique_text = pred.critique

    logger.info(f"Critique node: {'APPROVED' if 'APPROVED' in critique_text.upper() else 'needs improvement'}")

    return {"critique": critique_text.strip()}


def decide_node(state: ReasoningState) -> dict[str, Any]:
    """Decide whether to continue reasoning or finalize.

    Args:
        state: Current reasoning state.

    Returns:
        Updated state with completion status.
    """
    critique = state.get("critique", "")
    iteration = state.get("iteration", 0)
    current_answer = state.get("current_answer")

    # Only approve if we actually have an answer
    is_approved = "APPROVED" in critique.upper() and current_answer is not None
    max_reached = iteration >= settings.max_reasoning_iterations

    if is_approved:
        logger.info(f"Decide node: Answer approved after {iteration} iterations")
        return {
            "is_complete": True,
            "final_answer": state["current_answer"]
        }
    elif max_reached:
        logger.warning(f"Decide node: Max iterations ({settings.max_reasoning_iterations}) reached")
        return {
            "is_complete": True,
            "final_answer": state["current_answer"]
        }
    else:
        logger.info(f"Decide node: Continuing refinement (iteration {iteration})")
        return {"is_complete": False}


def should_continue(state: ReasoningState) -> str:
    """Router function to decide next node.

    Args:
        state: Current reasoning state.

    Returns:
        "end" if complete, "reason" to continue refining.
    """
    if state.get("pending_search_query"):
        return "tool"
    if state.get("is_complete", False):
        return "end"
    return "reason"


def route_reason_output(state: ReasoningState) -> str:
    """Router for reason node output.
    
    Args:
        state: Current reasoning state.
        
    Returns:
        "tool" if search requested, else "critique".
    """
    if state.get("pending_search_query"):
        return "tool"
    return "critique"
