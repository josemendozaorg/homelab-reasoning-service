# Deep Research Agent: Architecture Revision Plan

## Executive Summary

After deep research into inference-time scaling, self-correction effectiveness, and state-of-the-art deep research agents, this plan revises the architecture based on **what actually works according to research**.

**Key Finding:** The current critique loop is implementing "intrinsic self-correction" which research shows **does not improve performance** and can make it worse. However, the iterative search and synthesis pattern IS valuable. The architecture needs surgical changes, not wholesale replacement.

---

## Research Findings

### 1. Inference-Time Scaling IS Valuable

From [Scaling LLM Test-Time Compute (Google/Berkeley)](https://arxiv.org/abs/2408.03314):
> "Scaling test-time compute can be MORE effective than scaling model parameters"

From [Inference Scaling Laws](https://arxiv.org/abs/2408.00724):
> "Smaller models + advanced inference algorithms offer Pareto-optimal trade-offs"

**DeepSeek-R1 specifics:**
- Uses `<think>` tokens for chain-of-thought reasoning (12K-23K tokens per complex question)
- This IS inference-time scaling - the model "thinks longer" for harder problems
- Maximum generation: 32,768 tokens

**Implication:** Keep supporting deep reasoning with `<think>` tokens. Let the model use as many tokens as it needs.

### 2. Self-Correction: Nuanced Reality

From [When Can LLMs Actually Correct Their Own Mistakes? (MIT)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00713):

| Self-Correction Type | Does it Help? |
|---------------------|---------------|
| Intrinsic (LLM evaluates own reasoning) | **NO** - Often makes performance WORSE |
| With external tools (search, code) | **YES** - New information enables correction |
| With verification against sources | **YES** - Can check claims against retrieved data |
| Fine-tuned for self-correction | **YES** - But requires specific training |

> "The quality of self-generated feedback is bounded by the model's existing knowledge. Internal feedback may not offer any advantage; it might even steer the model away from the correct answer."

**Implication:** Remove the critique node that evaluates reasoning. Keep the loop that checks if MORE INFORMATION is needed.

### 3. Deep Research Agent Architecture (State of the Art)

From [OpenAI Deep Research](https://openai.com/index/introducing-deep-research/) and [Deep Research Agents Survey](https://arxiv.org/abs/2506.18096):

**OpenAI's Architecture:**
```
CLARIFY → PLAN → ITERATIVE SEARCH → SYNTHESIZE → VERIFY → OUTPUT
```

**Key patterns:**
1. **Clarification step** - Understand user intent before searching
2. **Query decomposition** - Break complex questions into sub-questions
3. **Iterative search** - Search → read → decide next search (not one-shot)
4. **Multi-format handling** - HTML, PDF, images
5. **Source-based verification** - Check claims against retrieved sources
6. **Citations** - Every claim linked to source

> "A single Deep Research query can involve dozens of search queries... one query led the agent to consult 21 different sources across nearly 28 minutes of processing."

---

## Current Architecture Analysis

### What's Actually Good ✓

| Component | Why It's Good |
|-----------|---------------|
| `<think>` token parsing | DeepSeek-R1's native inference-time scaling |
| Web search with scraping | External feedback - research shows this helps |
| LLM summarization | Synthesizing sources (like OpenAI's approach) |
| Iteration loop | Allows spending more compute when needed |
| Streaming support | Good UX for long-running research |

### What's Problematic ✗

| Component | Problem | Research Evidence |
|-----------|---------|-------------------|
| `critique_node` on reasoning | Intrinsic self-correction - doesn't help | [MIT Survey](https://arxiv.org/abs/2406.01297) |
| `decide_node` | Just parses "APPROVED" - pure overhead | N/A |
| Fixed 10 URL scraping | Not adaptive to query complexity | OpenAI uses 1-21+ sources |
| No planning phase | Jumps straight to answering | OpenAI decomposes queries first |
| No source verification | Doesn't verify claims against sources | Key pattern in deep research |
| Custom Ollama client | Redundant with langchain_ollama | N/A |

### Current Flow (Problematic)
```
REASON → CRITIQUE (intrinsic, doesn't help) → DECIDE → [loop or end]
    ↓
  TOOL ────────────────────────────────────────────────┘
```

---

## Proposed Architecture

### New Flow (Research-Backed)
```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   PLAN ──→ REASON ──→ SEARCH ──→ SYNTHESIZE ──→ VERIFY ──→ END │
│     ↑         │           ↑           │            │            │
│     │         │           │           │            │            │
│     │         └─ need info? ──────────┘            │            │
│     │                                              │            │
│     └────────────── gaps found? ───────────────────┘            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Node Definitions

#### 1. PLAN Node (NEW)
```python
"""Decompose complex queries into sub-questions."""
# Input: User query
# Output: List of sub-questions to research
# Only activates for complex queries (optional for simple ones)
```

**Prompt:**
```
You are a research planner. Given a query, identify:
1. What specific information is needed?
2. What sub-questions should be researched?
3. What sources might be relevant?

Query: {query}
Output a research plan.
```

#### 2. REASON Node (Modified)
```python
"""Generate answer using DeepSeek-R1's <think> tokens."""
# Key change: Don't ask "is my reasoning correct?"
# Instead ask: "Do I have enough information to answer?"
```

**Prompt:**
```
You are a research assistant with access to web search.

{time_context}
{research_plan if exists}
{gathered_information if exists}

Question: {query}

Instructions:
1. Use <think> tags for your reasoning process
2. If you need real-time information, use <search>query</search>
3. If you have enough information, provide your answer
4. End with: INFORMATION_STATUS: [SUFFICIENT | NEED_MORE: what's missing]
```

**Key difference:** No "critique your reasoning" - just "do you have enough info?"

#### 3. SEARCH Node (Enhanced)
```python
"""Iterative search with adaptive depth."""
# Change 1: Adaptive result count (3 for simple, 10 for complex)
# Change 2: Track which sources contributed what
# Change 3: Allow multiple search rounds
```

#### 4. SYNTHESIZE Node (NEW - replaces critique of search results)
```python
"""Combine information from multiple sources."""
# Input: Search results, previous findings
# Output: Synthesized information with source attribution
```

**Prompt:**
```
You have gathered the following information:
{search_results}

Synthesize this into a coherent understanding. For each key point, note which source it came from.
Identify any gaps or contradictions between sources.
```

#### 5. VERIFY Node (NEW - replaces critique of reasoning)
```python
"""Verify claims against sources - NOT intrinsic self-correction."""
# This is source-based verification, not "am I right?"
# Checks: Do the sources actually support the claims?
```

**Prompt:**
```
Final Answer: {answer}
Sources Used: {sources}

Verify: Does each claim in the answer have support from the sources?
- If YES: VERIFIED
- If NO: List unsupported claims and what additional research is needed
```

### Simplified Alternative (Minimal Changes)

If the full architecture is too ambitious, here's a minimal fix:

```
REASON ──→ [SEARCH ──→ ASSESS_INFORMATION] ──→ END
              ↑              │
              └── need more ─┘
```

**Just change the critique prompt from:**
```
❌ "Check for logical errors, factual inaccuracies..."
```

**To:**
```
✓ "Do you have sufficient information from the sources to answer confidently?"
```

This single change aligns with the research: evaluate INFORMATION SUFFICIENCY, not REASONING QUALITY.

---

## Implementation Plan

### Phase 1: Fix the Critique (High Impact, Low Risk)

**Change critique_node prompt from intrinsic self-correction to information assessment:**

```python
# OLD (doesn't help per research):
system_prompt = """You are a rigorous critic.
Check for logical errors, factual inaccuracies..."""

# NEW (helps per research):
system_prompt = """You are a research completeness assessor.
Given the question and gathered information:
1. Is there sufficient evidence to answer confidently?
2. Are there gaps that require additional search?
3. Do sources contradict each other?

If sufficient: "STATUS: COMPLETE"
If gaps exist: "STATUS: NEED_MORE - [specific gaps]"
"""
```

### Phase 2: Add Source Tracking

```python
class ReasoningState(TypedDict):
    query: str
    reasoning_trace: list[str]
    sources: list[dict]  # NEW: {url, title, summary, claims_supported}
    current_answer: Optional[str]
    information_gaps: list[str]  # NEW: What's still needed
    iteration: int
    ...
```

### Phase 3: Adaptive Search Depth

```python
async def perform_web_search(query: str, complexity: str = "medium") -> str:
    """Adaptive search based on query complexity."""
    max_results = {
        "simple": 3,
        "medium": 5,
        "complex": 10
    }.get(complexity, 5)
    ...
```

### Phase 4: Add Verification Node

```python
async def verify_node(state: ReasoningState) -> dict:
    """Verify claims against sources (NOT intrinsic self-correction)."""
    prompt = f"""
    Answer: {state['current_answer']}
    Sources: {format_sources(state['sources'])}

    For each claim in the answer, check if it's supported by the sources.
    List any unsupported claims.
    """
    # This is source-based verification, which research shows DOES help
```

### Phase 5: Simplify Infrastructure

- Replace custom Ollama client with `ChatOllama`
- Use `DuckDuckGoSearchRun` as base, add scraping on top
- Clean up unused code

---

## What NOT to Do

Based on research, avoid these changes:

| Don't Do This | Why |
|---------------|-----|
| Remove iteration loop entirely | Inference-time scaling is valuable |
| Replace with simple ReAct agent | Loses the deep research capability |
| Keep intrinsic critique unchanged | Research shows it hurts performance |
| Remove web scraping | External feedback is what makes correction work |
| Simplify to single LLM call | Defeats the purpose of inference-time scaling |

---

## Metrics to Track

| Metric | Why It Matters |
|--------|----------------|
| Answer accuracy (eval framework) | Core quality metric |
| Sources cited per answer | Deep research should use multiple sources |
| Search iterations per query | Are we doing iterative research? |
| Time to first token | UX metric |
| Total response time | Acceptable for deep research (5-30 min is normal) |
| Information gap detection rate | Is the new critique working? |

---

## Summary: Surgical Changes, Not Replacement

| Change | Effort | Impact |
|--------|--------|--------|
| Fix critique prompt (info assessment vs reasoning critique) | 1 hour | HIGH |
| Add source tracking | 2 hours | MEDIUM |
| Adaptive search depth | 1 hour | MEDIUM |
| Add verification node | 3 hours | HIGH |
| Replace Ollama client | 2 hours | LOW (cleanup) |
| Add planning node | 4 hours | MEDIUM |

**Recommended order:** 1 → 4 → 2 → 3 → 5 → 6

---

## Appendix: Key Research Sources

### Inference-Time Scaling
- [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314) - Google/Berkeley
- [Inference Scaling Laws](https://arxiv.org/abs/2408.00724) - Compute-optimal inference
- [State of LLM Reasoning Model Inference](https://magazine.sebastianraschka.com/p/state-of-llm-reasoning-and-inference-scaling)

### Self-Correction Research
- [When Can LLMs Actually Correct Their Own Mistakes?](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00713) - MIT Critical Survey
- [Self-Reflection in LLM Agents](https://arxiv.org/abs/2405.06682) - Effects on problem-solving
- [LLMs Cannot Self-Correct Reasoning Yet](https://arxiv.org/pdf/2310.01798) - Key limitations

### Deep Research Agents
- [OpenAI Deep Research](https://openai.com/index/introducing-deep-research/) - Architecture overview
- [Deep Research Agents Survey](https://arxiv.org/abs/2506.18096) - Systematic examination
- [From Web Search to Agentic Deep Research](https://arxiv.org/abs/2506.18959) - Reasoning agents
- [LangChain Open Deep Research](https://github.com/langchain-ai/open_deep_research) - Open implementation

### DeepSeek-R1
- [DeepSeek-R1 Paper](https://arxiv.org/pdf/2501.12948) - Training and architecture
- [How Reasoning Works in DeepSeek-R1](https://mccormickml.com/2025/02/07/how-reasoning-works-in-deepseek-r1/)
