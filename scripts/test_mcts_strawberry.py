import asyncio
import logging
from unittest.mock import patch
from src.reasoning.graph import create_reasoning_graph
from src.reasoning.state import create_initial_state
from langchain_core.runnables import RunnableConfig

# Configure logging
logging.basicConfig(level=logging.INFO)

async def mock_dispatch(*args, **kwargs):
    pass

@patch("src.reasoning.nodes.adispatch_custom_event", side_effect=mock_dispatch)
async def main(mock_event):
    state = create_initial_state(
        query="How many r's are in the word strawberry?",
        history=[]
    )
    # Set MCTS budget
    state["search_budget"] = 5
    
    app = create_reasoning_graph()
    config = RunnableConfig(configurable={"thread_id": "test_mcts"})
    
    print("Running MCTS Graph...")
    result = await app.ainvoke(state, config)
    
    print("\n--- Result Summary ---")
    tree = result.get("tree_state", {})
    root_id = result.get("root_id")
    final_answer = result.get("final_answer")
    
    print(f"Tree Size: {len(tree)} nodes")
    print(f"Root Visits: {tree[root_id].visits}")
    print(f"Final Answer: {final_answer}")
    
    # Analyze leaves
    root = tree[root_id]
    for child_id in root.children_ids:
        child = tree[child_id]
        print(f"Child {child_id[:8]} -> Visits: {child.visits}, Value: {child.value:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
