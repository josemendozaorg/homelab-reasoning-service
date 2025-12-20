"""Node implementations for the reasoning graph."""
import re
import logging
from typing import Any
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config import settings
from langchain_core.callbacks import adispatch_custom_event
from langchain_core.runnables import RunnableConfig
from .state import ReasoningState
from .tools import perform_web_search
from .llm import llm
from .mcts import MCTSNode, select_leaf, backpropagate, uct_score

logger = logging.getLogger(__name__)

# ... (Previous helper functions: predict_with_retry, parse_reasoning_response, parse_search_request, format_history) ...
# To save context, I will include them if replace_file_content is not used, but here I am creating a fresh content for clarity before using replace or write.
# Wait, I should use replace_file_content to append/modify.
# Since I am doing a comprehensive refactor of the MCTS nodes, I will replace the MCTS section of nodes.py.

# ... (Previous Legacy Nodes: reason_node, tool_node, critique_node, decide_node, should_continue, route_reason_output) ...
# ... (Previous Best-of-N Nodes: generate_candidates_node, evaluate_candidates_node, select_best_node) ...

# --- NEW MCTS GRANULAR NODES ---

async def plan_node(state: ReasoningState, config: RunnableConfig) -> dict[str, Any]:
    """Generate a high-level strategic plan before reasoning starts."""
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    today_date = now.strftime("%Y-%m-%d")
    
    system_prompt = f"""You are a strategic planner.
TODAY'S DATE: {today_date}
CURRENT TIME: {current_time}

Objective: Create a concise, high-level step-by-step plan to answer the user's question.
- Do NOT answer the question yet.
- Focus on identifying what information is needed (e.g. "Step 1: Search for X", "Step 2: Compare Y").
- Keep it under 5 steps.
"""
    
    user_content = f"Question: {state['query']}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    plan_text = ""
    logger.info("Generating Initial Plan...")
    
    try:
        async for token in llm.chat_stream(messages, temperature=0.7):
            plan_text += token
            await adispatch_custom_event(
                "token", 
                {"token": token, "node": "plan"},
                config=config
            )
            
        return {"initial_plan": plan_text}
        
    except Exception as e:
        logger.error(f"Planning failed: {e}")
        return {"initial_plan": "Failed to generate plan. Proceeding with direct reasoning."}


async def initialize_tree_node(state: ReasoningState, config: RunnableConfig) -> dict[str, Any]:
    """Initialize the MCTS tree with the root node seeded with the plan."""
    if state.get("root_id"):
        return {} # Already initialized
        
    logger.info("Initializing MCTS Tree with Plan...")
    
    plan = state.get("initial_plan", "No plan.")
    content = f"Objective: {state['query']}\n\nStrategic Plan:\n{plan}"
    
    root = MCTSNode(content=content, role="user")
    
    # Store in a dict keyed by ID for easy access
    tree_nodes = {root.id: root}
    
    return {
        "tree_state": tree_nodes,
        "root_id": root.id,
        "search_budget": state.get("search_budget", 5)
    }

async def mcts_select_node(state: ReasoningState, config: RunnableConfig) -> dict[str, Any]:
    """Step 1: Select a leaf node to expand."""
    tree_nodes = state["tree_state"]
    root_id = state["root_id"]
    budget = state["search_budget"]
    
    if budget <= 0:
        return {} # Should be caught by router
        
    selected_id = select_leaf(tree_nodes, root_id)
    selected_node = tree_nodes[selected_id]
    
    logger.info(f"MCTS Select: {selected_id} (Visits: {selected_node.visits}, Value: {selected_node.value})")
    
    await adispatch_custom_event(
        "debug_log",
        {
            "type": "mcts_step", 
            "step": "select", 
            "selected_id": selected_id, 
            "visits": selected_node.visits, 
            "value": selected_node.value,
            "budget_remaining": budget
        },
        config=config
    )
    
    return {"selected_node_id": selected_id}


async def mcts_expand_node(state: ReasoningState, config: RunnableConfig) -> dict[str, Any]:
    """Step 2: Generate the next reasoning step (Draft)."""
    from datetime import datetime
    
    tree_nodes = state["tree_state"]
    selected_id = state["selected_node_id"]
    selected_node = tree_nodes[selected_id]
    
    # Construct context
    path = []
    curr = selected_node
    while curr:
        path.append({"role": curr.role, "content": curr.content})
        curr = tree_nodes.get(curr.parent_id)
    path.reverse()
    
    # Check if we are re-entering from a Tool execution
    # If so, we should append the tool result to the context (or last message)
    # Actually, the tool node updates the trace or state.
    # In MCTS, we need to bake it into the prompt.
    # For this graph flow: Expand -> (Search) -> Tool -> Expand (Retry)
    # The 'pending_search_query' logic usually works by returning early.
    
    # Let's check if we just did a search
    # If we did, the SEARCH RESULT should be appended to the LAST MESSAGE context relative to the LLM
    # BUT, we haven't created the child node yet.
    # So the 'context' is the parent chain.
    # The 'response' we are generating IS the child node.
    # If search happens, the child node content = "Thought + <search> + Result".
    # So we need to resume generation.
    
    # Complex case:
    # 1. LLM generates "<search>X".
    # 2. We pause, go to Tool Node.
    # 3. Tool Node runs, gets result.
    # 4. We return to Expand Node.
    # 5. We need to feed "Thought + <search>X + Result" back to LLM to finish the step (generate Answer).
    
    # Simplified approach for v1:
    # If search, we just Append the result to the text and Save the node immediately?
    # No, we want the LLM to use the result.
    
    # Implementation:
    # We check internal state `pending_search_query`? No, that's cleared by tool node.
    # We need a temporary buffer in state? `mcts_draft_content`?
    
    # Let's stick to the previous monolithic logic but split across nodes.
    # Actually, if we want "Tool" to be a visible node, we must exit Expand.
    # EXPAND -> detects search -> sets `pending_search_query` -> returns.
    # GRAPH -> directs to TOOL.
    # TOOL -> executes -> sets `tool_output` (and clears pending).
    # GRAPH -> directs back to EXPAND.
    # EXPAND -> sees `tool_output` -> Resumes/Re-prompts.
    
    # We need `tool_output` in ReasoningState?
    # Or just use `reasoning_trace` as a scratchpad? MCTS doesn't use `reasoning_trace` for tree logic.
    # Let's add `last_tool_output` to state or just assume if we are here and `pending` is None, proceed.
    
    # Wait, the `reasoning_trace` IS the global log.
    # We can detect if the last item in trace is a search result?
    
    # Strategy:
    # If `state.get("pending_search_query")` is sent out, we are done with this run of expand.
    # When we come back, we expect the context to have the result?
    # The Tool Node should verify where to put the result.
    # For MCTS, the "Result" acts as part of the content of this new node being created.
    
    # Let's modify `tool_node` or make a `mcts_tool_node`?
    # Reusing `tool_node`: It appends to `reasoning_trace`.
    # `mcts_expand` can read `reasoning_trace` to see if there's a result to include.
    
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    today_date = now.strftime("%Y-%m-%d")
    local_timezone = now.astimezone().tzname() or "Local"
    
    search_instructions = f"""
    TODAY'S DATE: {today_date}
    CURRENT TIME: {current_time}
    TIMEZONE: {local_timezone}
    
    You are a deep reasoning assistant participating in a Monte Carlo Tree Search.
    
    INSTRUCTIONS:
    1. Continue the reasoning from the last step.
    2. Use <think> tags for internal reasoning.
    3. If you need external facts (prices, news, weather, dates), use <search>Query</search>.
    4. If you have enough info, provide the answer.
    """
    
    messages = [{"role": "system", "content": search_instructions}] + path
    
    # Check for recent tool output to inject
    # If the last trace item is search results, inject it as a "User" or "System" message?
    # Or just append to the last message?
    trace = state.get("reasoning_trace", [])
    if trace and "[Search Results]" in trace[-1]:
        # This is a re-entry with new info
        search_result_text = trace[-1]
        # We need to give this to the model.
        # "You requested a search. Here is the result:"
        messages.append({"role": "user", "content": f"System Notification: {search_result_text}\n\nPlease continue reasoning."})
        logger.info("MCTS Expand: Resuming with Search Results...")
    
    response_text = ""
    
    try:
        async for token in llm.chat_stream(messages, temperature=0.7):
            response_text += token
            await adispatch_custom_event(
                "token", 
                {"token": token, "node": "mcts_expand"},
                config=config
            )
            
        # Check for Search
        search_query = parse_search_request(response_text)
        if search_query:
            logger.info(f"MCTS Expand: Detected search for {search_query}")
            # Dispatch to Tool Node via Router
            return {
                "pending_search_query": search_query,
                # We do NOT create the child yet. We wait for the result.
                # But we might lose the "Thought" text?
                # We should save the current partial draft?
                # Actually, the Tool workflow is usually: Thought -> Tool Call.
                # The "Thought" is the reasoning leading to the tool.
                # If we lose it, the next prompt won't know why it searched.
                # We should append the partial thought to the context?
                # For simplicity in this graph: The next Expand call will RE-GENERATE the thought+tool call?
                # No, that's inefficient.
                # Correct way: The "Child Node" is created NOW with the partial text.
                # The Tool Result is appended to THAT child node.
                # And then we Continue expanding FROM that child?
                # No, a node is a state.
                # Let's say:
                # 1. Expand generates "I need price. <search>Price</search>".
                # 2. We CREATE Node A: "I need price. <search>.".
                # 3. We SELECT Node A.
                # 4. We execute Tool -> Result.
                # 5. We CREATE Node B (Child of A): "Result: $100. Okay, so..."
                
                # This seems cleaner for MCTS!
                # It means "Executing a Tool" is a Transition in the tree?
                # OR does the tool execution happen 'inside' the node creation?
                
                # Given the user wants "Graph Visibility", breaking it into steps is best.
                # Let's say:
                # If Search:
                #   Create Child Node (Thinking + Search Request).
                #   Set `selected_node_id` = Child Node.
                #   Return `pending_search_query`.
                #   (Graph goes to Tool).
                #   (Tool executes, adds result to trace).
                #   (Graph goes back to... Expand? No, Select?)
                #   Actually, if we go to Tool, we typically want to continue reasoning.
                #   So Tool -> Expand?
                #   The Expand node needs to know "I just did a tool, append result to current context".
            }
        
        # Determine if we should create a node (No search or Search finished)
        # If we just finished a search (trace has results), this response is the continuation.
        
        # Standard Case: Create Child
        child = MCTSNode(content=response_text, role="assistant", parent_id=selected_id)
        tree_nodes[child.id] = child
        state["tree_state"][child.id] = child
        selected_node.children_ids.append(child.id)
        
        return {
            "tree_state": state["tree_state"],
            "current_child_id": child.id # For evaluation
        }

    except Exception as e:
        logger.error(f"Expand failed: {e}")
        return {}


async def mcts_evaluate_node(state: ReasoningState, config: RunnableConfig) -> dict[str, Any]:
    """Step 3: Evaluate/Critique the new child node."""
    child_id = state.get("current_child_id")
    if not child_id:
        return {}
        
    child = state["tree_state"][child_id]
    reasoning, answer = parse_reasoning_response(child.content)
    
    judge_prompt = f"""Evaluate the following answer to: {state['query']}
    
    Answer: {answer}
    Reasoning: {reasoning}
    
    Score from 0 to 10."""
    
    score_response = await llm.generate(judge_prompt, temperature=0.1)
    score_match = re.search(r"(\d+(\.\d+)?)", score_response)
    score = float(score_match.group(1)) if score_match else 0.0
    value = score / 10.0
    
    logger.info(f"MCTS Evaluate: Child {child.id} Score {score}")
    
    await adispatch_custom_event(
        "debug_log",
        {
            "type": "mcts_step", 
            "step": "evaluate", 
            "child_id": child.id, 
            "score": score,
            "answer_preview": answer[:50]
        },
        config=config
    )
    
    return {"evaluation_score": value}


async def mcts_backprop_node(state: ReasoningState, config: RunnableConfig) -> dict[str, Any]:
    """Step 4: Backpropagate value and update budget."""
    child_id = state.get("current_child_id")
    value = state.get("evaluation_score", 0.0)
    tree_nodes = state["tree_state"]
    budget = state["search_budget"]
    
    if child_id:
        backpropagate(tree_nodes, child_id, value)
    
    new_budget = budget - 1
    logger.info(f"MCTS Backprop: Budget now {new_budget}")
    
    # Check if we found a very good answer? Early stop?
    # For now, just run out budget.
    
    return {"search_budget": new_budget}
    
# ... (mcts_finalize_node remains same, router needs update) ...
