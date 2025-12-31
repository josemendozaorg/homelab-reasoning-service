"""State management for the reasoning workflow."""
from typing import TypedDict, Optional, Any


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
        model: The model to use for this reasoning task.
    """
    query: str
    reasoning_trace: list[str]
    current_answer: Optional[str]
    critique: Optional[str]
    iteration: int
    is_complete: bool
    final_answer: Optional[str]
    chat_history: list[dict]
    pending_search_query: Optional[str]
    model: Optional[str]  # Model to use for this reasoning task
    # Phase 2: Best-of-N Support
    candidates: list[dict]
    verification_scores: list[dict]
    best_candidate: Optional[dict]
    # Phase 3: MCTS Support
    initial_plan: Optional[str]  # Planning phase output
    tree_state: dict[str, Any]  # Serialized Dict[str, MCTSNode]
    root_id: Optional[str]
    selected_node_id: Optional[str]  # For the current MCTS step
    search_budget: int
    # Phase 3b: LATS Improvements (per research)
    current_children_ids: list[str]  # IDs of children from current expansion
    reflected_ids: list[str]  # IDs of children that have been reflected upon
    evaluated_ids: list[str]  # IDs of children that have been evaluated
    best_terminal_id: Optional[str]  # ID of best terminal node (for early exit)
    # Query Classification
    query_complexity: Optional[str]  # "simple" or "complex" - determines fast/deep path


def create_initial_state(query: str, history: list[dict] = [], model: str = None) -> ReasoningState:
    """Create the initial state for a reasoning task.

    Args:
        query: The question or problem to reason about.
        history: Previous conversation history.
        model: The model to use for this reasoning task.

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
        chat_history=history,
        pending_search_query=None,
        model=model,
        candidates=[],
        verification_scores=[],
        best_candidate=None,
        initial_plan=None,
        tree_state={},
        root_id=None,
        selected_node_id=None,
        search_budget=10,  # Default budget
        # LATS improvements
        current_children_ids=[],
        reflected_ids=[],
        evaluated_ids=[],
        best_terminal_id=None,
        # Query classification
        query_complexity=None
    )
