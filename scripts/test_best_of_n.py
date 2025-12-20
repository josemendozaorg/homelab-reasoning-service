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
    
    app = create_reasoning_graph()
    config = RunnableConfig(configurable={"thread_id": "test_best_of_n"})
    
    print("Running Best-of-N Graph...")
    result = await app.ainvoke(state, config)
    
    print("\n--- Result Summary ---")
    candidates = result.get("candidates", [])
    scores = result.get("verification_scores", [])
    best = result.get("best_candidate")
    
    print(f"Candidates Generated: {len(candidates)}")
    for i, cand in enumerate(candidates):
        score_data = next((s for s in scores if s["candidate_id"] == cand["id"]), {})
        print(f"\nCandidate {i} (Score: {score_data.get('score')}):")
        print(f"Answer: {cand['answer'][:100]}...")
        
    print(f"\nSelected Best Candidate ID: {best['id'] if best else 'None'}")
    print(f"Final Answer: {result.get('final_answer')}")

if __name__ == "__main__":
    asyncio.run(main())
