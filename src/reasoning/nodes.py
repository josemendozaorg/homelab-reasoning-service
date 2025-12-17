"""Node implementations for the reasoning graph."""
import re
import logging
from typing import Any

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import settings
from .state import ReasoningState
from .prompts import REASON_SYSTEM_PROMPT, CRITIQUE_SYSTEM_PROMPT, REFINE_SYSTEM_PROMPT
from .tools import perform_web_search
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
async def invoke_llm_with_retry(llm: ChatOllama, messages: list) -> str:
    """Invoke LLM with retry logic for transient errors."""
    try:
        response_msg = await llm.ainvoke(messages)
        return response_msg.content
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


async def reason_node(state: ReasoningState) -> dict[str, Any]:
    """Generate reasoning using DeepSeek-R1's native think tokens.

    This node prompts the model to reason through the problem step by step,
    utilizing the <think> token format for internal deliberation.

    Args:
        state: Current reasoning state.

    Returns:
        Updated state fields with new reasoning trace and answer.
    """
    # Build the prompt
    if state["critique"]:
        # We're refining based on critique (either of answer or search results)
        if state["current_answer"]:
            prompt = f"""Original question: {state["query"]}

Previous answer: {state["current_answer"]}

Critique: {state["critique"]}

Please provide an improved answer addressing the critique."""
        else:
            # Critique of search results
            prompt = f"""Original question: {state["query"]}

Search Results Critique: {state["critique"]}

Based on this critique of the search results, please either:
1. Formulate an answer if the results are sufficient.
2. Request a new search if the critique indicates missing information.

Recall you can use <search>query</search> to search."""
        system_content = REFINE_SYSTEM_PROMPT
    else:
        # Initial reasoning
        prompt = state["query"]
        system_content = REASON_SYSTEM_PROMPT

    # Use LangChain ChatOllama for standard integration (supports streaming)
    llm = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=settings.temperature
    )

    messages = [SystemMessage(content=system_content)]
    
    # Inject history if available (and not refining)
    if not (state["critique"] and state["current_answer"]):
        for msg in state.get("chat_history", []):
            role = msg.get("role")
            content = msg.get("content")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                # For assistant history, prevent deepthink leakage if possible, 
                # but standard ChatOllama handles "AIMessage".
                # To be safe, we reconstruct roughly.
                from langchain_core.messages import AIMessage
                messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=prompt))

    response_text = await invoke_llm_with_retry(llm, messages)

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
            "current_answer": None, # No answer yet if searching
            "pending_search_query": search_query,
            "iteration": state["iteration"] # Don't increment iteration for tool use steps, or maybe do? Let's keep it same or increment. 
            # If we increment, we hit max iterations faster. Let's increment to prevent infinite loops.
            # actually, let's NOT increment iteration for the tool step itself, but the reason node did work.
            # Let's increment.
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
        
    results = perform_web_search(query)
    
    new_trace = state["reasoning_trace"].copy()
    new_trace.append(f"[Search Results]\n{results}")
    
    return {
        "reasoning_trace": new_trace,
        "pending_search_query": None
    }


async def critique_node(state: ReasoningState) -> dict[str, Any]:
    """Evaluate the current answer for errors or improvements.

    This node acts as a harsh critic, identifying logical flaws,
    missing considerations, and areas for improvement.

    Args:
        state: Current reasoning state.

    Returns:
        Updated state with critique.
    """
    args_answer = state.get("current_answer")
    
    if args_answer:
        # Standard critique of an answer
        prompt = f"""Question: {state["query"]}

Reasoning trace:
{chr(10).join(state["reasoning_trace"][-3:])}

Current answer: {args_answer}

Evaluate this answer critically. If it's fully satisfactory, respond with APPROVED.
Otherwise, provide specific feedback on what needs improvement."""
    else:
        # Critique of search results
        # We look at the last item in reasoning trace which should be search results
        last_trace = state["reasoning_trace"][-1] if state["reasoning_trace"] else "No trace"
        prompt = f"""Question: {state["query"]}

Recent Activity:
{last_trace}

Evaluate the search results above. 
- Do they directly answer the user's question?
- Are they relevant and sufficient?
- If they are sufficient, respond with "Search results are sufficient, please formulate an answer."
- If they are irrelevant or missing info, explain what is missing to guide the next search."""

    llm = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=0.3
    )
    
    messages = [
        SystemMessage(content=CRITIQUE_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]

    critique = await invoke_llm_with_retry(llm, messages)

    logger.info(f"Critique node: {'APPROVED' if 'APPROVED' in critique.upper() else 'needs improvement'}")

    return {"critique": critique.strip()}


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
