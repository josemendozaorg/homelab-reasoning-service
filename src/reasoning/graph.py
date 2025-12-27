"""LangGraph workflow for self-correcting reasoning."""
import logging
from langgraph.graph import StateGraph, END

from .state import ReasoningState
from .nodes import (
    reason_node,
    critique_node,
    decide_node,
    should_continue,
    tool_node,
    route_reason_output,
    generate_candidates_node,
    evaluate_candidates_node,
    select_best_node,
    initialize_tree_node,
    plan_node,
    mcts_select_node,
    mcts_expand_node,
    mcts_reflect_node,  # NEW: Reflection before evaluation (per LATS paper)
    mcts_evaluate_node,
    mcts_backprop_node,
    mcts_finalize_node,
    should_continue_mcts
)

logger = logging.getLogger(__name__)


def create_reasoning_graph() -> StateGraph:
    """Create the reasoning workflow graph.

    The graph implements System 2 Reasoning.
    
    Mode 1: Best-of-N (Active)
    ENTRY -> GENERATE -> EVALUATE -> SELECT -> END
    
    Mode 2: MCTS (Active if configured)
    ENTRY -> INIT_TREE -> ITERATE [Loop] -> FINALIZE -> END
    """
    # Create the graph with our state schema
    workflow = StateGraph(ReasoningState)

    # --- Best-of-N Nodes ---
    workflow.add_node("generate", generate_candidates_node)
    workflow.add_node("evaluate", evaluate_candidates_node)
    workflow.add_node("select", select_best_node)
    
    # --- MCTS Nodes ---
    workflow.add_node("plan", plan_node)
    workflow.add_node("init_tree", initialize_tree_node)
    workflow.add_node("mcts_select", mcts_select_node)
    workflow.add_node("mcts_expand", mcts_expand_node)
    workflow.add_node("mcts_reflect", mcts_reflect_node)  # NEW: Reflection node (LATS)
    workflow.add_node("mcts_evaluate", mcts_evaluate_node)
    workflow.add_node("mcts_backprop", mcts_backprop_node)
    workflow.add_node("tool_node", tool_node)  # Explicit tool node
    workflow.add_node("mcts_final", mcts_finalize_node)

    # Set entry point
    workflow.set_entry_point("plan")

    # --- Best-of-N Edges ---
    # ... (Kept existing)

    # --- MCTS Edges ---
    # Flow: plan → init → select → expand → (tool | reflect) → evaluate → backprop → (loop | final)
    workflow.add_edge("plan", "init_tree")
    workflow.add_edge("init_tree", "mcts_select")
    workflow.add_edge("mcts_select", "mcts_expand")

    # Router after expand: Search? Or Reflect?
    def route_mcts_expand(state: ReasoningState):
        if state.get("pending_search_query"):
            return "tool_node"
        return "mcts_reflect"  # Go to reflection instead of directly to evaluate

    workflow.add_conditional_edges(
        "mcts_expand",
        route_mcts_expand,
        {
            "tool_node": "tool_node",
            "mcts_reflect": "mcts_reflect"  # NEW: Route to reflection
        }
    )

    # Tool -> Expand (Loop back to use search results)
    workflow.add_edge("tool_node", "mcts_expand")

    # Reflect -> Evaluate (reflection informs scoring)
    workflow.add_edge("mcts_reflect", "mcts_evaluate")

    # Evaluate -> Backprop
    workflow.add_edge("mcts_evaluate", "mcts_backprop")
    
    workflow.add_conditional_edges(
        "mcts_backprop",
        should_continue_mcts,
        {
            "mcts_loop": "mcts_select",
            "finalize": "mcts_final"
        }
    )
    workflow.add_edge("mcts_final", END)

    # Compile the graph
    app = workflow.compile()

    logger.info("Reasoning graph compiled successfully (MCTS Mode Active)")

    return app
