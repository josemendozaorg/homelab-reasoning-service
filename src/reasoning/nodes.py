"""Node implementations for the reasoning graph."""
import re
import logging
import asyncio # Added for parallelization
from typing import Any, List # Added List
import json # Added for JSON reflection
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config import settings
from langchain_core.callbacks import adispatch_custom_event
from langchain_core.runnables import RunnableConfig
from .state import ReasoningState
from .tools import perform_web_search
from .llm import llm
from .mcts import (
    MCTSNode,
    select_leaf,
    backpropagate,
    uct_score,
    get_adaptive_branching_factor,
    is_terminal_answer
)  # MCTS imports

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
    reasoning = ""
    raw_answer = response

    if "<think>" in response:
        if "</think>" in response:
            think_pattern = r"<think>(.*?)</think>"
            think_matches = re.findall(think_pattern, response, re.DOTALL)
            reasoning = "\n".join(think_matches) if think_matches else ""
            # Everything after </think> is the answer candidate
            answer_match = re.search(r"</think>\s*(.*?)$", response, re.DOTALL)
            raw_answer = answer_match.group(1).strip() if answer_match else response.strip()
        else:
            # Has <think> but no closing tag - assume everything until "Answer:" or end is reasoning
            parts = response.split("<think>", 1)
            # Check if there is a clear "Answer:" marker later
            temp_content = parts[1]
            params = ["Final Answer:", "Answer:"]
            split_marker = None
            for p in params:
                if p in temp_content:
                    split_marker = p
                    break
                elif p.lower() in temp_content.lower():
                    split_marker = p # Case insensitive match logic below handle this
                    break
            
            if split_marker:
                 pass 
                 reasoning = temp_content 
            else:
                reasoning = temp_content

    elif "</think>" in response:
        # Missing <think> but has </think>
        reasoning = response.split("</think>")[0].strip()
        raw_answer = response.split("</think>")[1].strip()

    # If we haven't extracted reasoning cleanly yet, and no tags:
    if not reasoning and not "<think>" in response and not "</think>" in response:
         pass 

    # Now extract Final Answer from raw_answer
    markers = ["Final Answer:", "Answer:"]
    answer = raw_answer 
    
    for marker in markers:
        if marker in raw_answer:
            answer = raw_answer.split(marker, 1)[1].strip()
            # If we found reasoning earlier, great. If not, maybe the part BEFORE Answer: is reasoning?
            if not reasoning:
                candidate_reasoning = raw_answer.split(marker, 1)[0].strip()
                if len(candidate_reasoning) > 10: # arbitrary filter to avoid noise
                    reasoning = candidate_reasoning
            break
        elif marker.lower() in raw_answer.lower():
             pattern = re.compile(re.escape(marker), re.IGNORECASE)
             parts = pattern.split(raw_answer, 1)
             if len(parts) > 1:
                 answer = parts[1].strip()
                 if not reasoning:
                     candidate_reasoning = parts[0].strip()
                     if len(candidate_reasoning) > 10:
                        reasoning = candidate_reasoning
                 break

    # Cleanup: Remove potential JSON/Dictionary artifacts from bad model output
    # e.g. {'# reasoning#': ...}
    if "{'#" in answer or '{"#' in answer:
        answer = re.sub(r"\{['\"]#.*?\}", "", answer, flags=re.DOTALL).strip()
    
    return reasoning.strip(), answer.strip()


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
    local_timezone = now.astimezone().tzname() or "Local"
    
    time_context = f"TODAY'S DATE: {today_date}\nCURRENT TIME: {current_time}\nTIMEZONE: {local_timezone}\n"

    history_str = format_history(state.get("chat_history", []))

    # Shared Search Instructions
    search_instructions = f"""
    TODAY'S DATE: {today_date}
    CURRENT TIME: {current_time}
    TIMEZONE: {local_timezone}
    KNOWLEDGE CUTOFF: OUT OF DATE.
    
    SEARCH TRIGGER: If the question requires REAL-TIME data (prices, weather, news, current events), you MUST use the <search>Query</search> tool immediately.
    
    QUERY CONSTRUCTION: 
    - The search query is built by YOU.
    - For time-sensitive queries, you MUST include the current date or year in the query string itself (e.g. "Bitcoin price {today_date}", "Weather Tokyo {current_time}").
    - Use the provided TIMEZONE to interpret "morning", "tonight", etc. correctly.
    
    EXCEPTION: If you only need today's date or time (already provided below), do NOT search for it. Use the provided context."""

    if state["critique"] and state["current_answer"]:
        # REFINEMENT MODE
        system_prompt = f"""{search_instructions}
        
        You are a helpful assistant. 
        CRITICAL INSTRUCTION: Analyze the Feedback below. 
        - If the Feedback says the previous answer is HALLUCINATED, OUTDATED, or MISSING INFORMATION, you MUST use the <search>Query</search> tool to find the correct info.
        - Do NOT try to fix it by just rewriting if you don't have the data. SEARCH for it.
        
        Use <think> tags to reason, then refine the previous answer based on Feedback. Start Final Answer with 'Final Answer:'."""
        
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
        # FEW-SHOT INJECTION
        few_shot_examples = """
<example>
User: What is 2 + 2?
Assistant: <think>
This is a basic arithmetic question.
I need to add the number 2 to itself.
2 + 2 = 4.
</think>
Final Answer: 4
</example>

<example>
User: Which is larger, 9.11 or 9.9?
Assistant: <think>
I need to compare two numbers: 9.11 and 9.9.
First, compare the integer parts: both are 9.
Next, compare the tenths digit:
- 9.11 has 1 in the tenths place.
- 9.9 has 9 in the tenths place.
See 9 > 1, 9.9 is larger than 9.11.
</think>
Final Answer: 9.9
</example>
"""
        system_prompt = f"""{search_instructions}
        
        You are a helpful assistant. You are capable of complex reasoning.
        
        FORMAT INSTRUCTIONS:
        1. You MUST enclose your internal reasoning trace within <think> and </think> tags.
        2. You MUST provide your final answer starting with "Final Answer:".
        
        {few_shot_examples}
        
        Use <think> tags for internal reasoning. TRUST THE Context as the absolute source of truth."""
        
        user_content = f"{time_context}{history_str if history_str else 'No previous context.'}\n\nQuestion: {state['query']}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        # DEBUG: Snapshot the Prompt
        await adispatch_custom_event(
            "prompt_snapshot",
            {"node": "reason", "iteration": state.get("iteration", 0), "messages": messages},
            config=config
        )

    try:
        # Stream the response and dispatch tokens
        async for token in llm.chat_stream(messages):
            response_text += token
            # Dispatch token as custom event
            await adispatch_custom_event(
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


async def tool_node(state: ReasoningState, config: RunnableConfig) -> dict[str, Any]:
    """Execute pending tool calls with tiered search.

    Uses the new tiered search approach:
    - First iteration: "selective" (snippets + top 3 URLs)
    - Later iterations: "snippets" (fast, just snippets)

    Automatically appends current year to time-sensitive queries.

    Args:
        state: Current reasoning state.

    Returns:
        Updated state with tool results.
    """
    from datetime import datetime
    import re

    query = state.get("pending_search_query")
    selected_id = state.get("selected_node_id")  # Which node requested this
    iteration = state.get("iteration", 0)

    if not query:
        return {}

    # Get current date for time-sensitive searches
    now = datetime.now()
    current_year = now.strftime("%Y")
    today_date = now.strftime("%Y-%m-%d")

    # Auto-append year for time-sensitive queries if not already present
    # Detect time-sensitive keywords
    time_sensitive_keywords = [
        "current", "latest", "now", "today", "recent", "new", "update",
        "price", "news", "weather", "stock", "rate", "score", "live"
    ]
    query_lower = query.lower()

    # Check if query is time-sensitive and doesn't already have a year
    has_year = bool(re.search(r'\b20[0-9]{2}\b', query))
    is_time_sensitive = any(kw in query_lower for kw in time_sensitive_keywords)

    if is_time_sensitive and not has_year:
        query = f"{query} {current_year}"
        logger.info(f"Tool Node: Auto-appended year to query: '{query}'")

    # Determine search depth based on iteration
    # First search: more thorough. Later: faster snippets-only
    if iteration == 0:
        depth = "selective"  # ~5-10 sec, snippets + top 3 scraped
    else:
        depth = "snippets"   # ~2 sec, just snippets

    results = await perform_web_search(query, depth=depth, config=config)

    # Emit tokens for UI visibility
    formatted_results = f"[Search Results (depth={depth})]\n{results}\n"
    await adispatch_custom_event(
        "token",
        {"token": formatted_results, "node": "tool"},
        config=config
    )

    # CRITICAL FIX: Persist results to the Tree Node so Expand sees it next time
    if selected_id and selected_id in state["tree_state"]:
        node = state["tree_state"][selected_id]
        # Append results to the node content
        node.content += f"\n\n[System Notification: Search Results]\n{results}"
        # Store search results for external verification in reflection phase
        node.search_results = results
        logger.info(f"Tool Node: Persisted search results to Node {selected_id}")

    new_trace = state["reasoning_trace"].copy()
    new_trace.append(formatted_results)

    return {
        "reasoning_trace": new_trace,
        "pending_search_query": None,
        "tree_state": state["tree_state"]  # Return updated tree
    }


async def critique_node(state: ReasoningState, config: RunnableConfig) -> dict[str, Any]:
    """Evaluate the current answer for errors or improvements."""
    from datetime import datetime

    args_answer = state.get("current_answer")
    critique_text = ""

    # Get current date for context
    now = datetime.now()
    today_date = now.strftime("%Y-%m-%d")

    if args_answer:
        # Standard critique of an answer
        system_prompt = f"""TODAY'S DATE: {today_date}

You are a rigorous critic.
Instructions:
1. Review the Question, Reasoning, and Answer.
2. Check for logical errors, factual inaccuracies, or missing information.
3. For time-sensitive questions, verify the information is current (as of {today_date}).
4. If the answer is satisfactory, simply output "Critique: APPROVED".
5. If there are issues, describe them concisely."""

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
        system_prompt = f"""TODAY'S DATE: {today_date}

You are a rigorous critic.
Instructions:
1. Review the Question and Search Results.
2. Are the results sufficient to answer the question?
3. For time-sensitive questions, verify the search results are current (as of {today_date}).
4. If yes, output "Critique: APPROVED".
5. If no, explain what is missing."""

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
            await adispatch_custom_event(
                "token", 
                {"token": token, "node": "critique"},
                config=config
            )
    except Exception as e:
        logger.error(f"Critique failed: {e}")
        critique_text = "Critique failed, assuming no critical errors to keep moving."

    logger.info(f"Critique node: {'APPROVED' in critique_text.upper() if critique_text else 'failed'}")

    return {"critique": critique_text.strip()}


async def decide_node(state: ReasoningState, config: RunnableConfig) -> dict[str, Any]:
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
        
        # Log approval
        await adispatch_custom_event(
            "debug_log",
            {"type": "decision", "value": "APPROVED", "iteration": iteration},
            config=config
        )

        return {
            "is_complete": True,
            "final_answer": state["current_answer"]
        }
    elif max_reached:
        logger.warning(f"Decide node: Max iterations ({settings.max_reasoning_iterations}) reached")
        
        # Log max reached
        await adispatch_custom_event(
            "debug_log",
            {"type": "decision", "value": "MAX_ITERATIONS", "iteration": iteration},
            config=config
        )

        return {
            "is_complete": True,
            "final_answer": state["current_answer"]
        }
    else:
        logger.info(f"Decide node: Continuing refinement (iteration {iteration})")
        
        # Log continue
        await adispatch_custom_event(
            "debug_log",
            {"type": "decision", "value": "CONTINUE", "critique": critique},
            config=config
        )

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

async def generate_candidates_node(state: ReasoningState, config: RunnableConfig) -> dict[str, Any]:
    """Generate multiple reasoning candidates (Sequential Best-of-N)."""
    candidates = []
    # N=3 for single GPU constraints
    num_candidates = 3
    
    logger.info(f"Generating {num_candidates} candidates sequentially...")
    
    # Use higher temperature for diversity
    # We need to construct the prompt similar to reason_node but repeat it
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    today_date = now.strftime("%Y-%m-%d")
    local_timezone = now.astimezone().tzname() or "Local"
    
    time_context = f"TODAY'S DATE: {today_date}\nCURRENT TIME: {current_time}\nTIMEZONE: {local_timezone}\n"
    history_str = format_history(state.get("chat_history", []))
    
    # Reuse the same system prompt structure
    search_instructions = f"""
    TODAY'S DATE: {today_date}
    CURRENT TIME: {current_time}
    TIMEZONE: {local_timezone}
    
    You are a helpful assistant. You are capable of complex reasoning.
    
    FORMAT INSTRUCTIONS:
    1. You MUST enclose your internal reasoning trace within <think> and </think> tags.
    2. You MUST provide your final answer starting with "Final Answer:".
    
    Use <think> tags for internal reasoning. TRUST THE Context as the absolute source of truth.
    """
    
    user_content = f"{time_context}{history_str if history_str else 'No previous context.'}\n\nQuestion: {state['query']}"
    messages = [
        {"role": "system", "content": search_instructions},
        {"role": "user", "content": user_content}
    ]

    for i in range(num_candidates):
        try:
            logger.info(f"Generating candidate {i+1}/{num_candidates}...")
            # Use higher temperature (0.7) for distinct paths
            response_text = await llm.chat(messages, temperature=0.7)
            
            reasoning, answer = parse_reasoning_response(response_text)
            
            candidates.append({
                "id": i,
                "reasoning": reasoning,
                "answer": answer,
                "full_text": response_text
            })
            
            await adispatch_custom_event(
                "candidate_generated",
                {"id": i, "answer_preview": answer[:50] + "..."},
                config=config
            )
            
        except Exception as e:
            logger.error(f"Candidate generation {i} failed: {e}")
            
    return {"candidates": candidates}


async def evaluate_candidates_node(state: ReasoningState, config: RunnableConfig) -> dict[str, Any]:
    """Evaluate each candidate and assign a score."""
    candidates = state.get("candidates", [])
    scores = []
    
    logger.info(f"Evaluating {len(candidates)} candidates...")
    
    for cand in candidates:
        try:
            # Judge prompt
            system_prompt = """You are a critical judge.
            Evaluate the following reasoning and answer.
            Check for:
            1. Logical steps validity.
            2. Factuality.
            3. Adherence to constraints.
            
            Output a score from 0 to 10 and a brief critique.
            Format: Score: X\nCritique: ..."""
            
            user_content = f"""Question: {state['query']}
            
            Candidate Reasoning:
            {cand['reasoning']}
            
            Candidate Answer:
            {cand['answer']}
            """
            
            response = await llm.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ], temperature=0.1) # Low temp for judging
            
            # Parse score
            score_match = re.search(r"Score:\s*(\d+(\.\d+)?)", response, re.IGNORECASE)
            score = float(score_match.group(1)) if score_match else 0.0
            
            scores.append({
                "candidate_id": cand["id"],
                "score": score,
                "critique": response
            })
            
            await adispatch_custom_event(
                "candidate_scored",
                {"id": cand["id"], "score": score},
                config=config
            )
            
        except Exception as e:
            logger.error(f"Evaluation failed for candidate {cand['id']}: {e}")
            scores.append({"candidate_id": cand["id"], "score": 0.0, "critique": "Evaluation failed"})
            
    return {"verification_scores": scores}


async def select_best_node(state: ReasoningState, config: RunnableConfig) -> dict[str, Any]:
    """Select the best candidate based on scores."""
    scores = state.get("verification_scores", [])
    candidates = state.get("candidates", [])
    
    if not scores or not candidates:
        return {"is_complete": True, "final_answer": "Failed to generate valid candidates."}
        
    # Sort by score desc
    best_score = max(scores, key=lambda x: x["score"])
    best_cand = next((c for c in candidates if c["id"] == best_score["candidate_id"]), None)
    
    logger.info(f"Selected candidate {best_cand['id']} with score {best_score['score']}")
    
    # Update state to look like a normal completion
    return {
        "best_candidate": best_cand,
        "current_answer": best_cand["answer"],
        "reasoning_trace": [f"[Selected Candidate {best_cand['id']} Score: {best_score['score']}]\n{best_cand['reasoning']}"],
        "is_complete": True,
        "final_answer": best_cand["answer"]
    }


# --- PLANS & MCTS NODES ---

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
    
    # Emit token for UI visibility
    selection_text = f"[MCTS Select] Selected Node: {selected_id} (Visits: {selected_node.visits}, Value: {selected_node.value:.2f})\n"
    await adispatch_custom_event(
        "token", 
        {"token": selection_text, "node": "mcts_select"},
        config=config
    )
    
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
    
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    today_date = now.strftime("%Y-%m-%d")
    local_timezone = now.astimezone().tzname() or "Local"
    
    today_date = now.strftime("%Y-%m-%d")
    local_timezone = now.astimezone().tzname() or "Local"
    
    search_instructions = f"""
    TODAY'S DATE: {today_date}
    CURRENT TIME: {current_time}
    TIMEZONE: {local_timezone}
    
    You are a deep reasoning assistant participating in a Monte Carlo Tree Search.
    
    ### WHEN TO SEARCH:
    - **Verify Facts**: If unsure about specific names, dates, or recent events.
    - **Real-Time Info**: For news, weather, stock prices, or events AFTER 2023.
    - **Specifics**: When the user asks for "latest", "current", or specific details you don't know.
    - **DO NOT SEARCH**: For common knowledge, simple math, or if you already have the info in context.

    ### HOW TO SEARCH:
    - **Keywords Only**: Use precise keywords, not chatty sentences. 
      - GOOD: "current time Paris", "population Tokyo 2024", "latest events Bytow Poland 2025"
      - BAD: "What is the time in Paris?", "I need to look up..."
    - **Location Aware**: Always include the country/state to avoid ambiguity (e.g. "Bytow, Poland" not just "Bytow").
    
    ### FORMAT:
    - Output <search>Your Query Here</search> to trigger a search.
    - If no search is needed, just continue reasoning with <think>.
    """
    
    messages = [{"role": "system", "content": search_instructions}] + path

    # PARALLEL EXPANSION (LATS) with ADAPTIVE BRANCHING
    # Research: AB-MCTS dynamically decides branching based on uncertainty
    num_candidates = get_adaptive_branching_factor(tree_nodes, selected_node)
    logger.info(f"MCTS Expand: Generating {num_candidates} parallel candidates (adaptive)...")
    
    # We use llm.chat (non-streaming) for parallel, as mixing streaming with gather is complex
    # Create N tasks
    tasks = [llm.chat(messages, temperature=0.7) for _ in range(num_candidates)]
    
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        new_children_ids = []
        
        for i, response_text in enumerate(results):
            if isinstance(response_text, Exception):
                logger.error(f"Candidate {i} generation failed: {response_text}")
                continue
                
            if not response_text:
                continue

            # Check for Search
            search_query = parse_search_request(response_text)
            
            if search_query:
                # Truncation logic
                end_tag = "</search>"
                if end_tag in response_text:
                    end_idx = response_text.find(end_tag) + len(end_tag)
                    if len(response_text) - end_idx > 20: 
                        logger.warning(f"Truncated hallucinated content after search tag: {len(response_text) - end_idx} chars")
                    response_text = response_text[:end_idx]
                    
                logger.info(f"MCTS Expand: Candidate {i} requests search for '{search_query[:50]}...'")
                
            else:
                logger.info(f"MCTS Expand: Candidate {i} generated thought.")

            # Create Child Node
            child = MCTSNode(content=response_text, role="assistant", parent_id=selected_id)

            # Check for terminal state (complete answer found)
            if is_terminal_answer(response_text, state['query']):
                child.is_terminal = True
                logger.info(f"MCTS Expand: Candidate {i} contains terminal answer!")

            tree_nodes[child.id] = child
            state["tree_state"][child.id] = child
            selected_node.children_ids.append(child.id)
            new_children_ids.append(child.id)

            # Emit token just for visibility that something happened
            terminal_marker = " [TERMINAL]" if child.is_terminal else ""
            await adispatch_custom_event(
                "token",
                {"token": f"\n[Candidate {i+1} Generated]{terminal_marker}\n" + (f"[Search: {search_query}]" if search_query else ""), "node": "mcts_expand"},
                config=config
            )

        if not new_children_ids:
            logger.error("All candidates failed to generate.")
            return {}

        # If any candidate requested search, we need to handle it.
        # For V1 LATS Simplification:
        # If ANY child requests a search, we prioritize searching.
        # We pick the FIRST child that requested search to drive the 'tool_node'.
        # Or, we just pass the list of new children to Evaluate, and let it decide?
        # But `tool_node` expects `pending_search_query`.
        
        # Policy: Pick the first search request if any.
        first_search_child = next((tree_nodes[cid] for cid in new_children_ids if parse_search_request(tree_nodes[cid].content)), None)
        
        if first_search_child:
            query = parse_search_request(first_search_child.content)
            # We must set this child as 'selected' for the tool node to append results to it
            state["selected_node_id"] = first_search_child.id
            return {
                "pending_search_query": query,
                "tree_state": state["tree_state"],
                "selected_node_id": first_search_child.id,
                "current_children_ids": new_children_ids # Pass all for potential eval later?
            }
        
        # Else, pass all children to next step (Evaluate)
        return {
            "tree_state": state["tree_state"],
            "current_children_ids": new_children_ids 
        }

    except Exception as e:
        logger.error(f"Expand failed: {e}")
        return {}


async def mcts_reflect_node(state: ReasoningState, config: RunnableConfig) -> dict[str, Any]:
    """Step 3a: Generate reflection/critique for each candidate (LATS requirement).

    Research: LATS requires reflection BEFORE scoring. The reflection provides
    reasoning about the reasoning, identifies failure modes, and enables
    learning across rollouts.

    From LATS paper: "The reflection generator produces verbal self-reflection
    that summarizes what the agent did, what went wrong, and what insight
    can be gleaned for future trials."
    """
    from datetime import datetime

    child_ids = state.get("current_children_ids", [])
    if not child_ids:
        return {}

    tree_nodes = state["tree_state"]
    logger.info(f"MCTS Reflect: Generating reflections for {len(child_ids)} candidates...")

    # Get current date for context
    now = datetime.now()
    today_date = now.strftime("%Y-%m-%d")

    async def reflect_single(cid):
        child = tree_nodes[cid]

        # Build trajectory context
        trajectory = []
        curr = child
        while curr:
            trajectory.append(curr.content[:500])  # Truncate for context window
            curr = tree_nodes.get(curr.parent_id)
        trajectory.reverse()
        trajectory_text = "\n---\n".join(trajectory[-3:])  # Last 3 steps

        reflection_prompt = f"""TODAY'S DATE: {today_date}

You are analyzing a reasoning trajectory for the task: "{state['query']}"

Trajectory (recent steps):
{trajectory_text}

Current Step Being Analyzed:
{child.content}

Analyze this reasoning step:
1. CORRECTNESS: Are there any factual errors or logical flaws?
2. COMPLETENESS: Is anything missing to answer the question?
3. PROGRESS: Does this step advance toward the goal?
4. EXTERNAL INFO: Does this step use search results effectively (if any)?

Provide a structured reflection in 2-3 sentences.
End with: QUALITY: [HIGH/MEDIUM/LOW]
"""

        try:
            reflection = await llm.generate(reflection_prompt, temperature=0.3)
            child.reflection = reflection

            # Extract quality signal for scoring
            if "QUALITY: HIGH" in reflection.upper():
                child.reflection_score = 0.9
            elif "QUALITY: MEDIUM" in reflection.upper():
                child.reflection_score = 0.6
            else:
                child.reflection_score = 0.3

            # Check for external feedback integration
            if child.search_results:
                # Verify claims against search results
                verification_prompt = f"""Given these search results:
{child.search_results[:1000]}

And this claim/answer:
{child.content[:500]}

Are the claims supported by the sources? Score 0.0-1.0.
Output only a number."""
                try:
                    ext_score = await llm.generate(verification_prompt, temperature=0.1)
                    child.external_score = min(1.0, max(0.0, float(ext_score.strip())))
                except:
                    child.external_score = 0.5  # Neutral if parsing fails

            logger.info(f"MCTS Reflect: Node {cid} Reflection={child.reflection_score:.2f} External={child.external_score:.2f}")

            # Emit for UI
            await adispatch_custom_event(
                "token",
                {"token": f"[MCTS Reflect] {cid[:8]}... | Quality: {child.reflection_score:.1f} | {reflection[:100]}...\n", "node": "mcts_reflect"},
                config=config
            )

            return cid
        except Exception as e:
            logger.error(f"Reflection failed for {cid}: {e}")
            child.reflection = "Reflection failed"
            child.reflection_score = 0.5
            return cid

    tasks = [reflect_single(cid) for cid in child_ids]
    await asyncio.gather(*tasks)

    return {"tree_state": tree_nodes, "reflected_ids": child_ids}


async def mcts_evaluate_node(state: ReasoningState, config: RunnableConfig) -> dict[str, Any]:
    """Step 3b: Score candidates using multi-signal value function.

    Research: LLM self-scoring alone is unreliable. We combine:
    1. Reflection score (from mcts_reflect_node)
    2. External verification score (from search results)
    3. Terminal bonus (if complete answer found)

    Per ReST-MCTS* paper: "Since the LLM itself is an unreliable reward model,
    it's crucial to use Process Reward Models and external feedback."
    """
    # Use reflected_ids if available, fallback to current_children_ids
    child_ids = state.get("reflected_ids") or state.get("current_children_ids", [])
    if not child_ids:
        single_child = state.get("current_child_id")
        if single_child:
            child_ids = [single_child]
        else:
            return {}

    tree_nodes = state["tree_state"]
    logger.info(f"MCTS Evaluate: Scoring {len(child_ids)} candidates with multi-signal value function...")

    evaluated_ids = []

    for cid in child_ids:
        child = tree_nodes[cid]

        # MULTI-SIGNAL VALUE FUNCTION (per research recommendations)
        # Signal 1: Reflection score (already computed in reflect node)
        reflection_score = getattr(child, 'reflection_score', 0.5)

        # Signal 2: External verification (already computed if search results exist)
        external_score = getattr(child, 'external_score', 0.0)

        # Signal 3: Terminal bonus (reward complete answers)
        terminal_bonus = 0.2 if child.is_terminal else 0.0

        # Combine scores - weight external feedback higher per research
        if external_score > 0:
            # External feedback available - weight it 60%
            combined_score = (reflection_score * 0.3) + (external_score * 0.5) + terminal_bonus
        else:
            # No external feedback - rely on reflection + terminal
            combined_score = (reflection_score * 0.8) + terminal_bonus

        # Clamp to 0-1
        combined_score = min(1.0, max(0.0, combined_score))

        child.value = combined_score
        child.visits = 1

        logger.info(
            f"MCTS Evaluate: Node {cid} | Reflection={reflection_score:.2f} | "
            f"External={external_score:.2f} | Terminal={terminal_bonus:.2f} | "
            f"Combined={combined_score:.2f}"
        )

        # Emit token for UI visibility
        eval_text = (
            f"[MCTS Evaluate] {child.id[:8]}... | "
            f"Reflection: {reflection_score:.2f} | External: {external_score:.2f} | "
            f"Terminal: {terminal_bonus:.2f} | Final: {combined_score:.2f}\n"
        )
        await adispatch_custom_event(
            "token",
            {"token": eval_text, "node": "mcts_evaluate"},
            config=config
        )

        evaluated_ids.append(cid)

    return {"tree_state": tree_nodes, "evaluated_ids": evaluated_ids}


async def mcts_backprop_node(state: ReasoningState, config: RunnableConfig) -> dict[str, Any]:
    """Step 4: Backpropagate value with decay and check for early termination."""
    # LATS: Backpropagate for ALL evaluated children
    child_ids = state.get("evaluated_ids")
    if not child_ids:
        # Fallback
        single_id = state.get("current_child_id")
        if single_id:
            child_ids = [single_id]
        else:
            return {"search_budget": state["search_budget"] - 1}

    tree_nodes = state["tree_state"]
    budget = state["search_budget"]

    max_score = 0.0
    found_terminal = False
    best_terminal_id = None

    for cid in child_ids:
        node = tree_nodes[cid]
        # Backpropagate with decay (gamma=0.95) per research
        backpropagate(tree_nodes, cid, node.value, gamma=0.95)

        if node.value > max_score:
            max_score = node.value

        # Check for terminal nodes with high scores
        if node.is_terminal and node.value > 0.7:
            found_terminal = True
            best_terminal_id = cid
            logger.info(f"MCTS Backprop: Found high-quality terminal node {cid} with score {node.value}")

    # Decrement budget once per expansion step
    new_budget = budget - 1

    # Early Exit Conditions:
    # 1. Found a terminal node with good score
    # 2. Found any node with very high score (>0.95)
    early_exit = False
    if found_terminal:
        logger.info(f"MCTS Backprop: Terminal answer found with good score. Stopping early.")
        new_budget = 0
        early_exit = True
    elif max_score > 0.95:
        logger.info(f"MCTS Backprop: High score ({max_score}) detected. Stopping early.")
        new_budget = 0
        early_exit = True

    logger.info(f"MCTS Backprop: Budget now {new_budget}")

    # Emit token for UI visibility
    early_marker = " [EARLY EXIT]" if early_exit else ""
    backprop_text = f"[MCTS Backprop] Updated {len(child_ids)} paths (γ=0.95). Budget: {new_budget}{early_marker}\n"
    await adispatch_custom_event(
        "token",
        {"token": backprop_text, "node": "mcts_backprop"},
        config=config
    )

    result = {"search_budget": new_budget, "tree_state": tree_nodes}
    if best_terminal_id:
        result["best_terminal_id"] = best_terminal_id

    return result


async def mcts_finalize_node(state: ReasoningState, config: RunnableConfig) -> dict[str, Any]:
    """Select the best path from the tree after search is exhausted.

    Uses either:
    1. A pre-identified terminal node (from backprop early exit)
    2. The most-visited path (standard MCTS selection)
    """
    tree_nodes = state["tree_state"]
    root_id = state["root_id"]

    root = tree_nodes[root_id]
    if not root.children_ids:
        return {"is_complete": True, "final_answer": "MCTS failed to expand root."}

    # Check if we have a pre-identified best terminal node
    best_terminal_id = state.get("best_terminal_id")
    if best_terminal_id and best_terminal_id in tree_nodes:
        best_leaf = tree_nodes[best_terminal_id]
        logger.info(f"MCTS Complete. Using pre-identified terminal node {best_leaf.id} with score {best_leaf.value:.2f}")
    else:
        # Traverse to find the best leaf (most visited path) - standard MCTS
        current_node = root
        path_nodes = []

        while current_node.children_ids:
            if not current_node.children_ids:
                break

            # Select best child by visit count (standard MCTS selection)
            best_child_id = max(current_node.children_ids, key=lambda cid: tree_nodes[cid].visits)
            best_child = tree_nodes[best_child_id]
            path_nodes.append(best_child)
            current_node = best_child

        best_leaf = current_node
        logger.info(f"MCTS Complete. Traversed to leaf {best_leaf.id} with {best_leaf.visits} visits.")

    # Parse the content to get the answer
    _, answer = parse_reasoning_response(best_leaf.content)

    # If no answer found in leaf, try to find any terminal node with an answer
    if not answer or len(answer) < 10:
        logger.warning("No answer in best leaf, searching for terminal nodes...")
        for node_id, node in tree_nodes.items():
            if node.is_terminal:
                _, potential_answer = parse_reasoning_response(node.content)
                if potential_answer and len(potential_answer) > 10:
                    answer = potential_answer
                    best_leaf = node
                    logger.info(f"Found answer in terminal node {node_id}")
                    break

    # Build reflection summary for the trace
    reflection_summary = ""
    if hasattr(best_leaf, 'reflection') and best_leaf.reflection:
        reflection_summary = f"\n[Reflection: {best_leaf.reflection[:200]}...]"

    logger.info(f"MCTS Complete. Final answer from {best_leaf.id} (score: {best_leaf.value:.2f})")

    return {
        "is_complete": True,
        "final_answer": answer,
        "reasoning_trace": [f"[MCTS Selection | Score: {best_leaf.value:.2f}]{reflection_summary}\n{best_leaf.content}"]
    }


def should_continue_mcts(state: ReasoningState) -> str:
    """Router for MCTS loop."""
    if state["search_budget"] > 0:
        return "mcts_loop"
    return "finalize"
