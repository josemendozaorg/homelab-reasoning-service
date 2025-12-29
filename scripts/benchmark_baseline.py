import asyncio
import logging
from unittest.mock import patch, MagicMock
from src.reasoning.nodes import reason_node, ReasoningState
from langchain_core.runnables import RunnableConfig

# Configure logging
logging.basicConfig(level=logging.INFO)

async def mock_dispatch(*args, **kwargs):
    pass

@patch("src.reasoning.nodes.adispatch_custom_event", side_effect=mock_dispatch)
async def main(mock_event):
    state = ReasoningState(
        query="How many r's are in the word strawberry?",
        chat_history=[],
        iteration=0,
        reasoning_trace=[],
        current_answer=None,
        pending_search_query=None,
        critique=None,
        is_complete=False
    )
    
    config = RunnableConfig(configurable={"thread_id": "test_1"})
    
    print("Running reason_node...")
    result = await reason_node(state, config)
    
    print("\n--- Result ---")
    print(f"Reasoning Trace length: {len(result.get('reasoning_trace', []))}")
    if result.get('reasoning_trace'):
        print(f"Last Trace:\n{result['reasoning_trace'][-1]}")
    
    print(f"Current Answer: {result.get('current_answer')}")
    
    if result.get('current_answer') and result.get('reasoning_trace'):
        print("\nSUCCESS: Generated both reasoning and answer.")
    else:
        print("\nFAILURE: Missing reasoning or answer.")

if __name__ == "__main__":
    asyncio.run(main())
