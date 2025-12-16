"""LangGraph workflow for self-correcting reasoning."""
import logging
from langgraph.graph import StateGraph, END

from .state import ReasoningState
from .nodes import reason_node, critique_node, decide_node, should_continue, tool_node, route_reason_output

logger = logging.getLogger(__name__)


def create_reasoning_graph() -> StateGraph:
    """Create the reasoning workflow graph.

    The graph implements a Self-Correction loop:

    ENTRY → REASON ─┬─→ CRITIQUE → DECIDE ─┬─→ END
                    │                      │
                    └─→ TOOL ──────────────┘
    
    Returns:


    Returns:
        Compiled LangGraph StateGraph ready for invocation.
    """
    # Create the graph with our state schema
    workflow = StateGraph(ReasoningState)

    # Add nodes
    workflow.add_node("reason", reason_node)
    workflow.add_node("critique", critique_node)
    workflow.add_node("decide", decide_node)
    workflow.add_node("tool", tool_node)

    # Set entry point
    workflow.set_entry_point("reason")

    # Add edges
    # Remove fixed edge from reason
    # workflow.add_edge("reason", "critique") # Changed to conditional
    workflow.add_edge("critique", "decide")
    workflow.add_edge("tool", "reason")

    # Add conditional edge from reason
    workflow.add_conditional_edges(
        "reason",
        route_reason_output,
        {
            "tool": "tool",
            "critique": "critique"
        }
    )


    # Add conditional edge from decide
    workflow.add_conditional_edges(
        "decide",
        should_continue,
        {
            "end": END,
            "reason": "reason",
            "tool": "tool"
        }
    )

    # Compile the graph
    app = workflow.compile()

    logger.info("Reasoning graph compiled successfully")

    return app
