"""State management for the reasoning workflow."""
from typing import TypedDict, Optional


class ReasoningState(TypedDict):
    """State schema for the reasoning graph.

    Attributes:
        query: The original question or problem to reason about.
        reasoning_trace: List of reasoning steps (including <think> outputs).
        current_answer: The current proposed answer.
        critique: The most recent critique of the answer.
        iteration: Current iteration number.
        is_complete: Whether reasoning is complete.
        final_answer: The final approved answer.
    """
    query: str
    reasoning_trace: list[str]
    current_answer: Optional[str]
    critique: Optional[str]
    iteration: int
    is_complete: bool
    final_answer: Optional[str]
    chat_history: list[dict]


def create_initial_state(query: str, history: list[dict] = []) -> ReasoningState:
    """Create the initial state for a reasoning task.

    Args:
        query: The question or problem to reason about.
        history: Previous conversation history.

    Returns:
        Initial ReasoningState with empty trace and no answer.
    """
    return ReasoningState(
        query=query,
        reasoning_trace=[],
        current_answer=None,
        critique=None,
        iteration=0,
        is_complete=False,
        final_answer=None,
        chat_history=history
    )
