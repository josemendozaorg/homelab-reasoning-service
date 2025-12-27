# MCTS/LATS Implementation Review

## Branch: `fix-mcts-issues`

This document reviews the MCTS/LATS implementation against academic research and best practices.

---

## Executive Summary

The implementation shows a solid foundation for LATS (Language Agent Tree Search), but has several gaps compared to the state-of-the-art research. Key issues:

| Issue | Severity | Research Basis |
|-------|----------|----------------|
| No reflection/self-critique in evaluation | HIGH | Core LATS component missing |
| Value function is unreliable (LLM self-score) | HIGH | Research shows LLM self-scoring is weak |
| No external feedback integration | HIGH | External feedback is what makes MCTS effective |
| Fixed branching factor (3) | MEDIUM | Should be adaptive |
| No termination detection | MEDIUM | Can't detect when answer is found |
| Backpropagation doesn't decay | LOW | Should discount distant ancestors |

---

## Research Background

### What LATS Should Do (Per [ICML 2024 Paper](https://arxiv.org/abs/2310.04406))

```
1. SELECT    → UCT-based node selection ✓ (implemented)
2. EXPAND    → Generate N candidate actions ✓ (implemented)
3. SIMULATE  → Execute actions, get environment feedback ⚠️ (partial)
4. REFLECT   → Generate self-critique of outcomes ✗ (missing)
5. EVALUATE  → Score based on reflection + external feedback ⚠️ (weak)
6. BACKPROP  → Update ancestor values ✓ (implemented)
```

### Key Research Findings

From [Scaling LLM Test-Time Compute (Google)](https://arxiv.org/abs/2408.03314):
> "The effectiveness of different approaches to scaling test-time compute **critically varies depending on the difficulty of the prompt**."

From [When Can LLMs Self-Correct? (MIT)](https://arxiv.org/abs/2406.01297):
> "LLM self-evaluation without external feedback generally **does not improve** and can make performance **worse**."

From [ReST-MCTS* (Process Reward Models)](https://arxiv.org/abs/2406.03816):
> "Since the LLM itself is an **unreliable reward model**, especially in weaker models, it's crucial to adjust MCTS action selection with **PRM and partial order rules**."

---

## Detailed Analysis

### 1. Evaluation Node - CRITICAL ISSUE

**Current Implementation:**
```python
async def mcts_evaluate_node(state, config):
    judge_prompt = f"""Evaluate the reasoning step below...
    Rate the logical soundness and relevance from 0.0 to 1.0.
    OUTPUT MUST BE VALID JSON: {{"score": 0.5, "critique": "reasoning..."}}
    """
    response_json = await llm.generate(judge_prompt, temperature=0.1)
    score = float(data.get("score", 0.0))
    child.value = score
```

**Problems:**
1. **LLM self-scoring is unreliable** - Research shows LLMs can't accurately judge their own outputs
2. **No external feedback** - The score is based purely on self-assessment
3. **No reflection chain** - LATS requires a dedicated reflection step before scoring
4. **Single-shot evaluation** - Should use multiple evaluations or self-consistency

**What LATS Paper Recommends:**
```python
# Value = (1) LM-generated reflection score + (2) Self-consistency score
value = reflection_score * 0.5 + consistency_score * 0.5

# Where consistency_score = agreement across multiple rollouts
```

**Recommended Fix:**
```python
async def mcts_evaluate_node(state, config):
    child = tree_nodes[cid]

    # Step 1: REFLECTION (Generate critique, not just score)
    reflection_prompt = f"""
    Task: {state['query']}
    Reasoning Path: {child.content}

    Reflect on this reasoning:
    1. What was done well?
    2. What could be improved?
    3. Are there any errors or gaps?
    4. Is external information needed?

    Provide a detailed reflection.
    """
    reflection = await llm.generate(reflection_prompt)

    # Step 2: EXTERNAL FEEDBACK (if available)
    external_score = 0.0
    if has_search_results(child):
        # Check if claims are supported by sources
        external_score = verify_against_sources(child.content, search_results)

    # Step 3: SCORE based on reflection + external feedback
    score_prompt = f"""
    Based on this reflection: {reflection}
    And external verification score: {external_score}

    Rate the reasoning from 0.0 to 1.0.
    """
    score = await llm.generate(score_prompt)

    # Combine scores (external feedback weighted higher per research)
    final_score = score * 0.4 + external_score * 0.6
```

---

### 2. Missing Reflection Chain - CRITICAL ISSUE

**Current:** No dedicated reflection step. Evaluation asks for score directly.

**LATS Paper Requirement:**
> "The reflection generator produces verbal self-reflection that summarizes what the agent did, what went wrong, and what insight can be gleaned for future trials."

**Why This Matters:**
- Reflection provides **reasoning about the reasoning**
- It identifies **specific failure modes**
- It enables **learning across rollouts** (even without training)

**Recommended Addition:**
```python
async def mcts_reflect_node(state, config):
    """Generate reflection before evaluation."""
    child = tree_nodes[state["current_child_id"]]

    # Get the full trajectory
    trajectory = get_trajectory(tree_nodes, child.id)

    reflection_prompt = f"""
    You are analyzing a reasoning trajectory for the task: {state['query']}

    Trajectory:
    {format_trajectory(trajectory)}

    Analyze:
    1. CORRECTNESS: Are there any factual errors?
    2. COMPLETENESS: Is anything missing?
    3. EFFICIENCY: Were unnecessary steps taken?
    4. INSIGHT: What can be learned for future attempts?

    Provide structured reflection.
    """

    reflection = await llm.generate(reflection_prompt)
    child.reflection = reflection

    return {"current_reflection": reflection}
```

---

### 3. Value Function Quality - HIGH PRIORITY

**Current:** Simple 0-1 score from LLM self-assessment.

**Research Recommendation ([ReST-MCTS*](https://arxiv.org/abs/2406.03816)):**
> "Use a Process Reward Model (PRM) which evaluates any partial solution's quality."

**Options for Improvement:**

#### Option A: Multi-Signal Value Function
```python
def compute_value(node, state):
    signals = []

    # Signal 1: Self-consistency (run multiple times, check agreement)
    consistency = compute_consistency(node.content, num_samples=3)
    signals.append(("consistency", consistency, 0.3))

    # Signal 2: External verification (search results support claims?)
    if node.has_search_results:
        verification = verify_claims(node.content, node.search_results)
        signals.append(("verification", verification, 0.4))

    # Signal 3: Progress toward goal (does this advance the task?)
    progress = evaluate_progress(node.content, state['query'])
    signals.append(("progress", progress, 0.3))

    return weighted_average(signals)
```

#### Option B: Contrastive Evaluation
From [SC-MCTS*](https://arxiv.org/abs/2410.01707):
```python
def contrastive_value(candidate, alternatives):
    """Score candidate relative to alternatives."""
    # Compare this candidate against others
    # Higher score if it's distinctly better
    scores = []
    for alt in alternatives:
        comparison = llm.compare(candidate, alt, task)
        scores.append(comparison)
    return aggregate(scores)
```

---

### 4. Branching Factor - MEDIUM PRIORITY

**Current:** Fixed `num_candidates = 3`

**Research Finding ([Wider or Deeper?](https://arxiv.org/abs/2503.04412)):**
> "AB-MCTS dynamically decides whether to branch out or refine deeper in each iteration."

**Problem:**
- Easy questions don't need 3 candidates (wasteful)
- Hard questions might need more exploration

**Recommended Fix:**
```python
def get_branching_factor(state, node):
    """Adaptive branching based on uncertainty."""

    # If previous attempts had high variance → explore more
    if node.children_ids:
        child_scores = [tree_nodes[c].value for c in node.children_ids]
        variance = statistics.variance(child_scores) if len(child_scores) > 1 else 0
        if variance > 0.3:
            return 5  # High uncertainty → more exploration

    # If we're deep in the tree → focus (fewer candidates)
    depth = get_depth(node)
    if depth > 3:
        return 2  # Deep → exploit

    return 3  # Default
```

---

### 5. No Terminal State Detection - MEDIUM PRIORITY

**Current:** Only stops when `search_budget` reaches 0 or score > 0.95.

**Problem:** Can't detect when a valid answer has been found mid-search.

**LATS Paper:**
> "It stops searching either when it has generated a **valid solution** OR when it has reached the maximum number of rollouts."

**Recommended Fix:**
```python
def is_terminal(node, state):
    """Check if node contains a valid final answer."""
    content = node.content

    # Check for answer markers
    has_answer = "Final Answer:" in content or "Answer:" in content

    # Check for completeness (does it address the query?)
    if has_answer:
        _, answer = parse_reasoning_response(content)
        completeness = evaluate_completeness(answer, state['query'])
        if completeness > 0.8:
            return True

    return False

# In mcts_expand_node:
if is_terminal(child, state):
    child.is_terminal = True
    # Consider early termination
```

---

### 6. Backpropagation Decay - LOW PRIORITY

**Current:**
```python
def backpropagate(tree_nodes, node_id, value):
    current_id = node_id
    while current_id:
        node = tree_nodes[current_id]
        node.visits += 1
        node.value += value  # Full value to all ancestors
        current_id = node.parent_id
```

**Standard MCTS Practice:** Apply discount factor for distant ancestors.

```python
def backpropagate(tree_nodes, node_id, value, gamma=0.9):
    current_id = node_id
    current_value = value
    while current_id:
        node = tree_nodes[current_id]
        node.visits += 1
        node.value += current_value
        current_value *= gamma  # Decay for ancestors
        current_id = node.parent_id
```

---

### 7. Search Integration - GOOD BUT IMPROVABLE

**Current:** Search is triggered by `<search>` tags, results appended to tree.

**What's Good:**
- ✓ Search integrated in the MCTS loop
- ✓ Results become part of the tree context

**What Could Be Better:**
- Should verify search results against claims (external feedback)
- Should track which sources contributed to which conclusions
- Should use search results in evaluation scoring

---

## Recommended Architecture Changes

### Current Flow:
```
PLAN → INIT → SELECT → EXPAND → EVALUATE → BACKPROP → [loop]
                          ↓
                        TOOL
```

### Recommended Flow:
```
PLAN → INIT → SELECT → EXPAND → REFLECT → EVALUATE → BACKPROP → [loop]
                          ↓                    ↑
                        TOOL ─────────────────┘
                                    (external feedback)
```

---

## Implementation Priority

| Change | Effort | Impact | Priority |
|--------|--------|--------|----------|
| Add reflection node | 3 hours | HIGH | 1 |
| Multi-signal value function | 4 hours | HIGH | 2 |
| External feedback in evaluation | 3 hours | HIGH | 3 |
| Terminal state detection | 2 hours | MEDIUM | 4 |
| Adaptive branching | 2 hours | MEDIUM | 5 |
| Backprop decay | 30 min | LOW | 6 |

---

## Summary

The current implementation has the **structure** of LATS but is missing key **quality signals**:

1. **Reflection is absent** - Core LATS component
2. **Evaluation is weak** - LLM self-scoring without external feedback
3. **No source verification** - Search results aren't used to validate claims

The single highest-impact change would be adding a **reflection node** and incorporating **external feedback from search results** into the evaluation score.

---

## References

- [LATS Paper (ICML 2024)](https://arxiv.org/abs/2310.04406)
- [Scaling Test-Time Compute (Google)](https://arxiv.org/abs/2408.03314)
- [ReST-MCTS* (Process Reward)](https://arxiv.org/abs/2406.03816)
- [SC-MCTS* (Contrastive)](https://arxiv.org/abs/2410.01707)
- [LangGraph LATS Tutorial](https://langchain-ai.github.io/langgraph/tutorials/lats/lats/)
- [When Can LLMs Self-Correct?](https://arxiv.org/abs/2406.01297)
