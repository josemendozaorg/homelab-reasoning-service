"""Monte Carlo Tree Search (MCTS) implementation."""
import math
import uuid
from typing import Optional, List, Dict, Any

class MCTSNode:
    """Represents a node in the MCTS tree."""
    
    def __init__(self, content: str, role: str = "assistant", parent_id: Optional[str] = None):
        self.id = str(uuid.uuid4())
        self.content = content  # The actual text generated (thought/answer)
        self.role = role        # "user", "assistant", "system"
        self.parent_id = parent_id
        self.children_ids: List[str] = []
        
        # MCTS Statistics
        self.visits = 0
        self.value = 0.0  # Cumulative value
        
        # State metadata
        self.is_terminal = False
        self.is_fully_expanded = False
        
    @property
    def q_value(self) -> float:
        """Mean value of the node."""
        if self.visits == 0:
            return 0.0
        return self.value / self.visits

def uct_score(node: MCTSNode, parent_visits: int, exploration_weight: float = 1.41) -> float:
    """Calculate the Upper Confidence Bound for Trees (UCT) score."""
    if node.visits == 0:
        return float('inf')  # Prioritize unvisited nodes
        
    exploitation = node.q_value
    exploration = exploration_weight * math.sqrt(math.log(parent_visits) / node.visits)
    return exploitation + exploration

def select_leaf(tree_nodes: Dict[str, MCTSNode], root_id: str) -> str:
    """Traverse the tree from root to a leaf using UCT."""
    current_id = root_id
    
    while True:
        current_node = tree_nodes[current_id]
        
        if not current_node.children_ids:
            return current_id  # Found a leaf
            
        if not current_node.is_fully_expanded:
             # If strictly following MCTS, we expand if not fully expanded.
             # But here we might rely on the LLM to generate all children at once?
             # For simpler "Growth", we assume if it has children, we pick one.
             pass
             
        # UCT Selection among children
        best_score = float('-inf')
        best_child_id = None
        
        for child_id in current_node.children_ids:
            child = tree_nodes[child_id]
            score = uct_score(child, current_node.visits)
            if score > best_score:
                best_score = score
                best_child_id = child_id
                
        if best_child_id:
            current_id = best_child_id
        else:
            return current_id # Should not happen if children exist

def backpropagate(tree_nodes: Dict[str, MCTSNode], node_id: str, value: float):
    """Update stats for the lineage of the node."""
    current_id = node_id
    while current_id:
        node = tree_nodes[current_id]
        node.visits += 1
        node.value += value
        current_id = node.parent_id
