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
from .mcts import MCTSNode, select_leaf, backpropagate, uct_score # Added MCTS imports

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
    """Execute pending tool calls.
    
    Args:
        state: Current reasoning state.
        
    Returns:
        Updated state with tool results.
    """
    query = state.get("pending_search_query")
    selected_id = state.get("selected_node_id") # We need to know WHICH node requested this
    
    if not query:
        return {}
        
    results = await perform_web_search(query, config=config)
    
    # Emit tokens for UI visibility
    formatted_results = f"[Search Results]\n{results}\n"
    await adispatch_custom_event(
        "token", 
        {"token": formatted_results, "node": "tool"},
        config=config
    )
    
    # CRITICAL FIX: Persist results to the Tree Node so Expand sees it next time
    if selected_id and selected_id in state["tree_state"]:
        node = state["tree_state"][selected_id]
        # Append results to the node content effectively "simulating" the user providing the info
        # We append it as a System Notification to the content history
        node.content += f"\n\n[System Notification: Search Results]\n{results}"
        logger.info(f"Tool Node: Persisted search results to Node {selected_id}")
    
    new_trace = state["reasoning_trace"].copy()
    new_trace.append(formatted_results)
    
    return {
        "reasoning_trace": new_trace,
        "pending_search_query": None,
        "tree_state": state["tree_state"] # Return updated tree
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
    
    search_instructions = f"""
    TODAY'S DATE: {today_date}
    CURRENT TIME: {current_time}
    TIMEZONE: {local_timezone}
    
    You are a deep reasoning assistant participating in a Monte Carlo Tree Search.
    
    CRITICAL REAL-TIME DATA RULE:
    - You generally do NOT know current prices, stock values, weather, or news events after your training cut-off.
    - If the user asks for ANY real-time info (e.g. "price of X", "weather in Y", "latest news"), you MUST use <search>Query</search> IMMEDIATELY.
    - DO NOT HALLUCINATE or guess values.
    - LOCATION AWARENESS: Be very careful with location names. Ensure you are searching for the correct city/country (e.g. "Bytow, Poland" is NOT "Baytown, Texas").
    
    INSTRUCTIONS:
    1. Continue the reasoning from the last step.
    2. Use <think> tags for internal reasoning.
    3. If you lack information, output <search>Your Query Here</search>.
    4. If you have enough info, provide the final answer.
    """
    
    messages = [{"role": "system", "content": search_instructions}] + path
    
    # We no longer need to manually inject "trace[-1]" because tool_node now
    # explicitly appends the search results to the 'content' of the selected_node.
    # So 'path' above ALREADY includes the search results if they exist.
    
    response_text = ""
    
    try:
        async for token in llm.chat_stream(messages, temperature=0.7):
            response_text += token
            await adispatch_custom_event(
                "token", 
                {"token": token, "node": "mcts_expand"},
                config=config
            )
            
        # Check for Search (and truncate if found to prevent hallucination loops)
        search_query = parse_search_request(response_text)
        if search_query:
            # TRUNCATION FIX:
            # If the model outputs "<search>...</search> Then I hallucinate result...", 
            # we must cut it off after the search tag.
            # This forces the system to actually run the tool.
            end_tag = "</search>"
            if end_tag in response_text:
                end_idx = response_text.find(end_tag) + len(end_tag)
                # FORCE TRUNCATION: Ignore anything after </search>
                # Log if we cut off a significant amount
                if len(response_text) - end_idx > 20: 
                     logger.warning(f"Truncated hallucinated content after search tag: {len(response_text) - end_idx} chars")
                response_text = response_text[:end_idx]
                
            logger.info(f"MCTS Expand: Detected search for {search_query} (Truncated)")
            
            # Create the node representing this Thought + Search Request
            child = MCTSNode(content=response_text, role="assistant", parent_id=selected_id)
            tree_nodes[child.id] = child
            state["tree_state"][child.id] = child
            selected_node.children_ids.append(child.id)
            
            # Update selected node to this new child, so next expansion continues from here
            state["selected_node_id"] = child.id
            
            return {
                "pending_search_query": search_query,
                "tree_state": state["tree_state"],
                "selected_node_id": child.id 
            }
        
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
    
    # Emit token for UI visibility
    eval_text = f"[MCTS Evaluate] Child: {child.id} | Score: {score}/10\n"
    await adispatch_custom_event(
        "token", 
        {"token": eval_text, "node": "mcts_evaluate"},
        config=config
    )
    
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
    
    # Early Exit Optimization:
    # If we found a very high-quality answer (Score > 0.9), stop searching.
    if value > 0.9:
        logger.info(f"MCTS Backprop: High score ({value}) detected. Stopping early.")
        new_budget = 0
        
    logger.info(f"MCTS Backprop: Budget now {new_budget}")
    
    # Emit token for UI visibility
    backprop_text = f"[MCTS Backprop] Updated Value. Budget Remaining: {new_budget}\n"
    await adispatch_custom_event(
        "token", 
        {"token": backprop_text, "node": "mcts_backprop"},
        config=config
    )
    
    return {"search_budget": new_budget}


async def mcts_finalize_node(state: ReasoningState, config: RunnableConfig) -> dict[str, Any]:
    """Select the best path from the tree after search is exhausted."""
    tree_nodes = state["tree_state"]
    root_id = state["root_id"]
    
    root = tree_nodes[root_id]
    if not root.children_ids:
        return {"is_complete": True, "final_answer": "MCTS failed to expand root."}
        
    # Traverse to find the best leaf (most visited path)
    current_node = root
    path_nodes = []
    
    while current_node.children_ids:
        # Select best child by visit count
        if not current_node.children_ids:
            break
            
        best_child_id = max(current_node.children_ids, key=lambda cid: tree_nodes[cid].visits)
        best_child = tree_nodes[best_child_id]
        path_nodes.append(best_child)
        current_node = best_child
        
    # Now current_node is the leaf
    best_leaf = current_node
    logger.info(f"MCTS Complete. Traversed to leaf {best_leaf.id} with {best_leaf.visits} visits.")
    
    # Parse the content to get the answer from the LEAF (or the accumulated path if needed, but usually leaf has it)
    # Actually, MCTS expansion adds history. So leaf content is just the latest delta.
    # But usually the model mimics "thinking -> answer" in the final step.
    # If the leaf is just "Response", we might need to look at the whole path?
    # Our prompt says: "Continue reasoning... provide answer".
    # So the Answer should be in the leaf.
    _, answer = parse_reasoning_response(best_leaf.content)
    
    logger.info(f"MCTS Complete. Selected leaf {best_leaf.id} with {best_leaf.visits} visits.")
    
    # Emit the FINAL ANSWER tokens so the UI renders them in the Answer Bubble
    # (Since all previous tokens were hidden in the Trace)
    await adispatch_custom_event(
        "token", 
        {"token": answer, "node": "mcts_final"}, # Node name matches app.js allow-list
        config=config
    )
    
    return {
        "is_complete": True,
        "final_answer": answer,
        "reasoning_trace": [f"[MCTS Selection]\n{best_leaf.content}"]
    }


def should_continue_mcts(state: ReasoningState) -> str:
    """Router for MCTS loop."""
    if state["search_budget"] > 0:
        return "mcts_loop"
    return "finalize"
