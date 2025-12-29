import unittest
from unittest.mock import AsyncMock, patch, MagicMock
from langchain_core.runnables import RunnableConfig

from src.reasoning.state import ReasoningState, create_initial_state
from src.reasoning.nodes import generate_candidates_node, mcts_iteration_node
from src.reasoning.mcts import MCTSNode

class TestReasoningNodes(unittest.IsolatedAsyncioTestCase):
    
    def setUp(self):
        self.config = RunnableConfig(configurable={"thread_id": "test"})
        
    @patch("src.reasoning.nodes.llm")
    @patch("src.reasoning.nodes.adispatch_custom_event")
    async def test_generate_candidates_node(self, mock_dispatch, mock_llm):
        # Configure AsyncMock for chat
        mock_llm.chat = AsyncMock(return_value="<think>Step 1</think>Answer: 42")
        
        state = create_initial_state("Q")
        
        # Test
        result = await generate_candidates_node(state, self.config)
        
        self.assertIn("candidates", result)
        candidates = result["candidates"]
        self.assertEqual(len(candidates), 3) # N=3 hardcoded for now
        self.assertEqual(candidates[0]["answer"], "42")
        self.assertEqual(candidates[0]["reasoning"], "Step 1")
        
    @patch("src.reasoning.nodes.llm")
    @patch("src.reasoning.nodes.adispatch_custom_event")
    async def test_mcts_iteration_node(self, mock_dispatch, mock_llm):
        # Setup Tree
        # Root -> Child1 (Unvisited)
        # Iteration should select Child1, Expand (create Child1-1), Verify, Backprop
        
        root = MCTSNode("root")
        root.visits = 1
        
        # Configure AsyncMocks
        mock_llm.chat = AsyncMock(return_value="<think>Reasoning</think>Answer: NewChild")
        mock_llm.generate = AsyncMock(return_value="Score: 9.0")
        
        state = create_initial_state("Q")
        state["tree_state"] = {root.id: root}
        state["root_id"] = root.id
        state["search_budget"] = 5
        
        # Test
        result = await mcts_iteration_node(state, self.config)
        
        self.assertIn("search_budget", result)
        self.assertEqual(result["search_budget"], 4)
        
        tree = result["tree_state"]
        # Root should now have a child
        self.assertEqual(len(root.children_ids), 1)
        child_id = root.children_ids[0]
        
        # Child should be in tree
        self.assertIn(child_id, tree)
        child = tree[child_id]
        
        # Child should have stats (1 visit, 0.9 value)
        self.assertEqual(child.visits, 1)
        self.assertAlmostEqual(child.value, 0.9)
        
        # Root backprop (initial visits was 1, now 2)
        self.assertEqual(root.visits, 2)

if __name__ == '__main__':
    unittest.main()
