"""Node implementations for the reasoning graph."""
import re
import logging
from typing import Any
import asyncio
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config import settings
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from .state import ReasoningState
from .tools import perform_web_search
from .llm import llm

logger = logging.getLogger(__name__)

# Helper to wrap LLM calls with retry (now wraps coroutines)
async def predict_with_retry(coro_func, *args, **kwargs):
    """Invoke LLM coroutine with retry logic."""
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    ):
        with attempt:
            try:
                # If it's a coroutine object, we can't await it multiple times surely?
                # Better to pass a lambda that returns a coroutine
                return await coro_func(*args, **kwargs)
            except Exception as e:
                logger.error(f"LLM invocation failed: {e}")
                raise


def parse_reasoning_response(response: str) -> tuple[str, str]:
    """Parse a response to extract reasoning and answer.

    Args:
        response: The full model response.

    Returns:
        Tuple of (reasoning_trace, answer).
    """
    # Robust extraction: look for tags but handle cases where opening tag might be stripped by Ollama
    if "</think>" in response:
        if "<think>" in response:
            think_pattern = r"<think>(.*?)</think>"
            think_matches = re.findall(think_pattern, response, re.DOTALL)
            reasoning = "\n".join(think_matches) if think_matches else ""
        else:
            # Missing <think> but has </think>
            reasoning = response.split("</think>")[0].strip()
    else:
        reasoning = ""

    # Everything after </think> is the answer
    if "</think>" in response:
        answer_match = re.search(r"</think>\s*(.*?)$", response, re.DOTALL)
        raw_answer = answer_match.group(1).strip() if answer_match else response.strip()
    else:
        raw_answer = response.strip()

    # If "Final Answer:" is present, extract only what follows
    final_marker = "Final Answer:"
    if final_marker in raw_answer:
        answer = raw_answer.split(final_marker, 1)[1].strip()
    elif final_marker.lower() in raw_answer.lower():
         # Case insensitive fallback
         pattern = re.compile(re.escape(final_marker), re.IGNORECASE)
         parts = pattern.split(raw_answer, 1)
         answer = parts[1].strip() if len(parts) > 1 else raw_answer
    else:
        answer = raw_answer

    # Cleanup: Remove potential JSON/Dictionary artifacts from bad model output
    # e.g. {'# reasoning#': ...}
    if "{'#" in answer or '{"#' in answer:
        answer = re.sub(r"\{['\"]#.*?\}", "", answer, flags=re.DOTALL).strip()
    
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


async def reason_node(state: ReasoningState, config: RunnableConfig) -> dict[str, Any]:
    """Generate reasoning using direct prompts."""
    response_text = ""
    
    # Get current time context
    from datetime import datetime
    now = datetime.now() # Use local time for better user alignment
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    today_date = now.strftime("%Y-%m-%d")
    time_context = f"TODAY'S DATE: {today_date}\nCURRENT TIME: {current_time}\n"

    history_str = format_history(state.get("chat_history", []))

    # Shared Search Instructions
    search_instructions = f"""
    TODAY'S DATE: {today_date}
    KNOWLEDGE CUTOFF: OUT OF DATE.
    
    SEARCH TRIGGER: If the question requires REAL-TIME data (prices, weather, news, current events), you MUST use the <search>Query</search> tool immediately.
    
    EXCEPTION: If you only need today's date or time (already provided below), do NOT search for it. Use the provided context."""

    if state["critique"] and state["current_answer"]:
        # REFINEMENT MODE
        system_prompt = f"""{search_instructions}
        
        You are a helpful assistant. Use <think> tags to reason, then refine the previous answer based on Feedback. Start Final Answer with 'Final Answer:'."""
        
        user_content = f"""{time_context}
Previous Question: {state['query']}
Previous Answer: {state['current_answer']}
Feedback: {state['critique']}

Please provide the corrected answer."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    elif state["critique"]:
         # CRITIQUE SEARCH RESULTS MODE (or just general critique retry)
        system_prompt = f"""{search_instructions}
        
        You are a helpful assistant. Use <think> tags for internal reasoning. TRUST THE Context and search results provided. Start Final Answer with 'Final Answer:'."""
        
        context_block = f"{time_context}Search Results Critique/Feedback: {state['critique']}"
        user_content = f"{context_block}\n\nQuestion: {state['query']}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    else:
        # INITIAL REASONING MODE
        system_prompt = f"""{search_instructions}
        
        You are a helpful assistant. Use <think> tags for internal reasoning. TRUST THE Context as the absolute source of truth. Start Final Answer with 'Final Answer:'."""
        
        user_content = f"{time_context}{history_str if history_str else 'No previous context.'}\n\nQuestion: {state['query']}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

    try:
        # Stream the response and dispatch tokens
        async for token in llm.chat_stream(messages):
            response_text += token
            # Dispatch token as custom event
            dispatch_custom_event(
                "token", 
                {"token": token, "node": "reason"},
                config=config
            )
    except Exception as e:
        logger.error(f"Reasoning iteration failed: {e}")
        raise

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
            "iteration": state["iteration"] # Keep iteration same
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


async def critique_node(state: ReasoningState, config: RunnableConfig) -> dict[str, Any]:
    """Evaluate the current answer for errors or improvements."""
    args_answer = state.get("current_answer")
    critique_text = ""
    
    if args_answer:
        # Standard critique of an answer
        system_prompt = """You are a rigorous critic.
Instructions:
1. Review the Question, Reasoning, and Answer.
2. Check for logical errors, factual inaccuracies, or missing information.
3. If the answer is satisfactory, simply output "Critique: APPROVED".
4. If there are issues, describe them concisely."""
        
        trace_context = "\n".join(state["reasoning_trace"][-3:])
        user_content = f"""Question: {state['query']}
Reasoning History:
{trace_context}

Proposed Answer: {args_answer}

Critique this answer."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

    else:
        # Critique of search results
        system_prompt = """You are a rigorous critic.
Instructions:
1. Review the Question and Search Results.
2. Are the results sufficient to answer the question?
3. If yes, output "Critique: APPROVED".
4. If no, explain what is missing."""
        
        last_trace = state["reasoning_trace"][-1] if state["reasoning_trace"] else "No trace"
        user_content = f"""Question: {state['query']}
Search Results:
{last_trace}

Critique these results."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

    try:
        async for token in llm.chat_stream(messages):
            critique_text += token
            # Dispatch token as custom event
            dispatch_custom_event(
                "token", 
                {"token": token, "node": "critique"},
                config=config
            )
    except Exception as e:
        logger.error(f"Critique failed: {e}")
        critique_text = "Critique failed, assuming no critical errors to keep moving."

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
    """Router function to decide next node."""
    if state.get("pending_search_query"):
        return "tool"
    if state.get("is_complete", False):
        return "end"
    return "reason"


def route_reason_output(state: ReasoningState) -> str:
    """Router for reason node output."""
    if state.get("pending_search_query"):
        return "tool"
    return "critique"
