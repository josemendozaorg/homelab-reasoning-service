import unittest
from unittest.mock import AsyncMock, patch, MagicMock
from langchain_core.runnables import RunnableConfig

from src.reasoning.state import ReasoningState, create_initial_state
from src.reasoning.nodes import generate_candidates_node, mcts_expand_node
from src.reasoning.mcts import MCTSNode

class TestReasoningNodes(unittest.IsolatedAsyncioTestCase):
    
    def setUp(self):
        # Config needs fast_model for expand node
        self.config = RunnableConfig(configurable={
            "thread_id": "test",
            "fast_model": "test-fast-model"
        })
        
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
        
        # Verify it used the fast model (via get_fast_model_from_config -> config)
        # We need to verify the call arguments to chat
        args, kwargs = mock_llm.chat.call_args
        self.assertEqual(kwargs.get("model"), "test-fast-model")

    @patch("src.reasoning.nodes.llm")
    @patch("src.reasoning.nodes.adispatch_custom_event")
    async def test_mcts_expand_node(self, mock_dispatch, mock_llm):
        # Setup Tree with a selected node
        root = MCTSNode(content="Root", role="user")
        
        state = create_initial_state("Q")
        state["tree_state"] = {root.id: root}
        state["selected_node_id"] = root.id
        
        # Mock LLM response for expansion
        mock_llm.chat = AsyncMock(return_value="<think>Reasoning</think>Answer: NewChild")
        
        # Test
        result = await mcts_expand_node(state, self.config)
        
        # Check that children were generated
        self.assertIn("current_children_ids", result)
        children_ids = result["current_children_ids"]
        self.assertTrue(len(children_ids) > 0)
        
        # Verify the child is in the tree
        child_id = children_ids[0]
        self.assertIn(child_id, state["tree_state"])
        child = state["tree_state"][child_id]
        self.assertEqual(child.parent_id, root.id)
        
        # Verify it used the fast model
        args, kwargs = mock_llm.chat.call_args
        self.assertEqual(kwargs.get("model"), "test-fast-model")

if __name__ == '__main__':
    unittest.main()
