# Research Agent Simplification Plan

## Executive Summary

**The current implementation is massively overengineered.** After deep research into existing agent frameworks (LangGraph prebuilt components, Smolagents, OpenAI Agents SDK), it's clear that what currently takes ~600 lines of custom code across 5 files could be reduced to ~80 lines using LangGraph's prebuilt components.

**Key Finding:** The entire custom graph (reason → critique → decide → tool) can be replaced with `create_react_agent()` from `langgraph.prebuilt`, which implements the industry-standard ReAct pattern in a single function call.

---

## Research Summary

### Frameworks Evaluated

| Framework | Pros | Cons | Verdict |
|-----------|------|------|---------|
| **LangGraph prebuilt** | Already using LangGraph; `create_react_agent` handles everything; native tool calling | Need to upgrade langgraph version | **RECOMMENDED** |
| **Smolagents** (HuggingFace) | Minimalist (~1000 lines); code-first agents | Less mature; requires good coding LLM | Good for prototypes |
| **OpenAI Agents SDK** | Production-ready; great tracing | OpenAI-focused; less flexible for local LLMs | Not ideal for Ollama |

### Industry Best Practice: ReAct Pattern

The ReAct (Reasoning + Acting) pattern is now the industry standard:

```
USER QUERY → REASON → ACTION (tool call) → OBSERVE → REASON → ... → FINAL ANSWER
```

**What LangGraph provides out of the box:**
- `create_react_agent()` - Creates entire agent loop
- `ChatOllama` - Native Ollama integration with tool calling
- `DuckDuckGoSearchRun` - Pre-built search tool
- `ToolNode` + `tools_condition` - Pre-built tool execution and routing

**Sources:**
- [LangGraph ReAct Agent from Scratch](https://langchain-ai.github.io/langgraph/how-tos/react-agent-from-scratch/)
- [LangGraph Prebuilt Components](https://reference.langchain.com/python/langgraph/agents/)
- [Smolagents vs LangGraph Comparison](https://www.analyticsvidhya.com/blog/2025/01/smolagents-vs-langgraph/)

---

## Current vs Proposed Architecture

### Current Implementation (~600 lines)

```
src/reasoning/
├── graph.py      # 72 lines - Custom 4-node graph
├── nodes.py      # 351 lines - reason_node, critique_node, decide_node, tool_node
├── state.py      # 49 lines - ReasoningState (9 fields)
├── tools.py      # 139 lines - Custom web search with scraping + summarization
└── llm.py        # 130 lines - Custom Ollama client

src/llm/
└── ollama_client.py  # Duplicate Ollama client
```

**Flow:**
```
ENTRY → REASON ─┬─→ CRITIQUE → DECIDE ─┬─→ END
                │                      │
                └─→ TOOL ──────────────┘
```

**Problems:**
1. Custom `<search>` tag parsing instead of native tool calling
2. Separate critique LLM call doubles latency
3. Custom Ollama client instead of `langchain_ollama.ChatOllama`
4. Heavy web search: 10 URLs × (HTTP + LLM summarization) = 30-60 seconds
5. 3 different prompt modes with duplicated logic

### Proposed Implementation (~80 lines)

```
src/reasoning/
├── agent.py      # ~60 lines - Uses create_react_agent
└── tools.py      # ~20 lines - DuckDuckGoSearchRun wrapper
```

**Flow (Standard ReAct):**
```
ENTRY → AGENT ─┬─→ TOOLS ──┐
               │           │
               └───────────┴─→ END
```

---

## Proposed Implementation Code

### Option A: Use `create_react_agent` (Simplest)

```python
# src/reasoning/agent.py (~60 lines total)
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from src.config import settings

# Initialize LLM with tool calling support
llm = ChatOllama(
    model=settings.ollama_model,
    base_url=settings.ollama_base_url,
    temperature=settings.temperature,
)

# Define tools
search = DuckDuckGoSearchRun(max_results=3)

@tool
def get_current_date() -> str:
    """Get today's date and current time."""
    now = datetime.now()
    return f"Today is {now.strftime('%Y-%m-%d')}, current time is {now.strftime('%H:%M:%S')}"

tools = [search, get_current_date]

# System prompt
SYSTEM_PROMPT = """You are a helpful research assistant with access to web search.

Instructions:
1. Use the search tool for real-time information (prices, weather, news, current events)
2. Use get_current_date if you need today's date
3. Reason step by step before answering
4. Be concise and accurate
"""

# Create the agent - this single line replaces 400+ lines of custom code
agent = create_react_agent(
    llm,
    tools,
    prompt=SYSTEM_PROMPT,
)

async def run_agent(query: str, history: list = None):
    """Run the research agent on a query."""
    messages = history or []
    messages.append({"role": "user", "content": query})

    result = await agent.ainvoke({"messages": messages})
    return result["messages"][-1].content
```

### Option B: Custom Graph with Prebuilt Components (More Control)

```python
# src/reasoning/agent.py (~80 lines total)
from typing import Annotated, Sequence
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from src.config import settings

# State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# LLM
llm = ChatOllama(
    model=settings.ollama_model,
    base_url=settings.ollama_base_url,
    temperature=settings.temperature,
)

# Tools
tools = [DuckDuckGoSearchRun(max_results=3)]
llm_with_tools = llm.bind_tools(tools)

# Nodes
async def agent_node(state: AgentState):
    response = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

# Graph - just 2 nodes instead of 4!
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,  # Built-in routing: tools or END
    {"tools": "tools", END: END}
)
workflow.add_edge("tools", "agent")

agent = workflow.compile()
```

---

## What This Eliminates

| Current Code | Lines | Replacement |
|-------------|-------|-------------|
| `nodes.py` - reason_node | 104 | `create_react_agent` handles this |
| `nodes.py` - critique_node | 63 | **Eliminated** - unnecessary with good LLM |
| `nodes.py` - decide_node | 32 | `tools_condition` handles routing |
| `nodes.py` - tool_node | 22 | `ToolNode` prebuilt |
| `nodes.py` - routing functions | 20 | `tools_condition` prebuilt |
| `nodes.py` - parsing functions | 60 | Native tool calling, no parsing needed |
| `tools.py` - web scraping | 90 | `DuckDuckGoSearchRun` returns snippets |
| `tools.py` - LLM summarization | 49 | **Eliminated** - search snippets are enough |
| `llm.py` - custom Ollama client | 130 | `ChatOllama` from langchain_ollama |
| `state.py` - complex state | 49 | Simple `messages` list |
| `graph.py` - custom graph | 72 | `create_react_agent` or 10-line graph |

**Total reduction: ~600 lines → ~80 lines (87% reduction)**

---

## Why the Self-Correction Loop is Unnecessary

The current implementation has a critique → decide → refine loop. Research shows this is often counterproductive:

1. **Modern LLMs are good enough:** DeepSeek-R1 with `<think>` tokens already does internal reasoning
2. **Doubles latency:** Every answer requires 2 LLM calls minimum
3. **Reflection research findings:** "Self-reflection can help but the increase in performance was modest compared to the cost" ([arxiv.org/abs/2405.06682](https://arxiv.org/abs/2405.06682))
4. **ReAct is the standard:** Industry has converged on simple Reason→Act→Observe loops

**Recommendation:** Remove the critique loop entirely. If quality issues arise, add it back as an optional mode.

---

## Migration Plan

### Phase 1: Dependencies Update
```bash
# Update requirements.txt
langgraph>=0.2.0  # Was >=0.0.20
langchain-ollama>=0.1.0  # New - native Ollama integration
langchain-community>=0.2.0  # For DuckDuckGoSearchRun
```

### Phase 2: Implement New Agent
1. Create `src/reasoning/agent.py` with ~80 lines (see code above)
2. Update `src/api/routes.py` to use new agent
3. Keep old code temporarily for A/B testing

### Phase 3: Update Streaming
The new agent supports streaming via:
```python
async for event in agent.astream_events({"messages": messages}, version="v2"):
    if event["event"] == "on_chat_model_stream":
        yield event["data"]["chunk"].content
```

### Phase 4: Remove Old Code
- Delete `src/reasoning/nodes.py`
- Delete `src/reasoning/tools.py` (or keep minimal wrapper)
- Delete `src/llm/ollama_client.py`
- Simplify `src/reasoning/state.py`
- Update tests

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Ollama doesn't support tool calling well | Test with DeepSeek-R1; fallback to text-based tool detection |
| Search quality drops | Keep DuckDuckGoSearchRun with max_results=5 |
| Streaming breaks | Test SSE thoroughly; `astream_events` is well-documented |
| Quality regression | Run eval framework before/after; keep old code for rollback |

---

## Decision Matrix

| Approach | Effort | Risk | Benefit |
|----------|--------|------|---------|
| **Option A: create_react_agent** | Low (1-2 hours) | Low | 87% code reduction, standard pattern |
| Option B: Custom prebuilt graph | Medium (3-4 hours) | Low | More control, still 80% reduction |
| Keep current + minor fixes | Low (1 hour) | None | Only 10-20% improvement |

**Recommendation:** Start with **Option A**. If more control is needed, migrate to Option B.

---

## Appendix: Key Research Sources

- [LangGraph ReAct Agent Documentation](https://langchain-ai.github.io/langgraph/how-tos/react-agent-from-scratch/)
- [LangGraph Prebuilt Components Reference](https://reference.langchain.com/python/langgraph/agents/)
- [Building Local AI Agents with LangGraph and Ollama](https://www.digitalocean.com/community/tutorials/local-ai-agents-with-langgraph-and-ollama)
- [Smolagents vs LangGraph Comparison](https://www.analyticsvidhya.com/blog/2025/01/smolagents-vs-langgraph/)
- [OpenAI Agents SDK Review](https://mem0.ai/blog/openai-agents-sdk-review)
- [Self-Reflection in LLM Agents Research](https://arxiv.org/abs/2405.06682)
- [LangChain DuckDuckGo Integration](https://python.langchain.com/docs/integrations/tools/ddg)
