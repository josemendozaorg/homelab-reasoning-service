import unittest
import math
from src.reasoning.mcts import MCTSNode, uct_score, select_leaf, backpropagate

class TestMCTSLogic(unittest.TestCase):
    
    def test_node_initialization(self):
        node = MCTSNode(content="root", role="user")
        self.assertEqual(node.visits, 0)
        self.assertEqual(node.value, 0.0)
        self.assertEqual(node.q_value, 0.0)
        self.assertFalse(node.children_ids)
        
    def test_uct_score_unvisited_priority(self):
        node = MCTSNode("child")
        # Unvisited nodes should have infinite score to be prioritized
        score = uct_score(node, parent_visits=10)
        self.assertEqual(score, float('inf'))
        
    def test_uct_score_calculation(self):
        node = MCTSNode("child")
        node.visits = 5
        node.value = 4.0 # 0.8 mean
        
        parent_visits = 100
        
        expected_exploitation = 0.8
        expected_exploration = 1.41 * math.sqrt(math.log(100) / 5)
        expected = expected_exploitation + expected_exploration
        
        score = uct_score(node, parent_visits)
        self.assertAlmostEqual(score, expected)
        
    def test_select_leaf(self):
        # Create a tiny tree
        root = MCTSNode("root")
        child1 = MCTSNode("c1", parent_id=root.id)
        child2 = MCTSNode("c2", parent_id=root.id)
        
        tree_nodes = {root.id: root, child1.id: child1, child2.id: child2}
        root.children_ids = [child1.id, child2.id]
        
        # c1 visited once, low value
        child1.visits = 1
        child1.value = 0.0
        
        # c2 unvisited -> Should be selected due to infinite UCT
        child2.visits = 0
        
        root.visits = 1
        
        selected_id = select_leaf(tree_nodes, root.id)
        self.assertEqual(selected_id, child2.id)
        
    def test_backpropagate(self):
        root = MCTSNode("root")
        child = MCTSNode("child", parent_id=root.id)
        tree_nodes = {root.id: root, child.id: child}
        
        backpropagate(tree_nodes, child.id, 0.5)
        
        self.assertEqual(child.visits, 1)
        self.assertEqual(child.value, 0.5)
        
        self.assertEqual(root.visits, 1)
        self.assertEqual(root.value, 0.5)

if __name__ == '__main__':
    unittest.main()
