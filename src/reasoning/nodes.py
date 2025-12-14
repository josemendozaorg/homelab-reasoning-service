"""Node implementations for the reasoning graph."""
import re
import logging
from typing import Any

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import settings
from .state import ReasoningState
from .prompts import REASON_SYSTEM_PROMPT, CRITIQUE_SYSTEM_PROMPT, REFINE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


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
    if state["critique"] and state["current_answer"]:
        # We're refining based on critique
        prompt = f"""Original question: {state["query"]}

Previous answer: {state["current_answer"]}

Critique: {state["critique"]}

Please provide an improved answer addressing the critique."""
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

    response_msg = await llm.ainvoke(messages)
    response_text = response_msg.content

    reasoning, answer = parse_reasoning_response(response_text)

    new_trace = state["reasoning_trace"].copy()
    if reasoning:
        new_trace.append(f"[Iteration {state['iteration'] + 1}]\n{reasoning}")

    logger.info(f"Reason node iteration {state['iteration'] + 1}: generated {len(answer)} char answer")

    return {
        "reasoning_trace": new_trace,
        "current_answer": answer,
        "iteration": state["iteration"] + 1
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
    prompt = f"""Question: {state["query"]}

Reasoning trace:
{chr(10).join(state["reasoning_trace"][-3:])}

Current answer: {state["current_answer"]}

Evaluate this answer critically. If it's fully satisfactory, respond with APPROVED.
Otherwise, provide specific feedback on what needs improvement."""

    llm = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=0.3
    )
    
    messages = [
        SystemMessage(content=CRITIQUE_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]

    response_msg = await llm.ainvoke(messages)
    critique = response_msg.content

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

    is_approved = "APPROVED" in critique.upper()
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
    if state.get("is_complete", False):
        return "end"
    return "reason"
