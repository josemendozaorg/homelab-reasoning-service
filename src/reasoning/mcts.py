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

        # NEW: Reflection and external feedback (per LATS paper)
        self.reflection: Optional[str] = None  # Self-critique text
        self.search_results: Optional[str] = None  # External feedback from web search
        self.external_score: float = 0.0  # Score from external verification
        self.reflection_score: float = 0.0  # Score from self-reflection

    @property
    def q_value(self) -> float:
        """Mean value of the node."""
        if self.visits == 0:
            return 0.0
        return self.value / self.visits

    @property
    def combined_score(self) -> float:
        """Combined score weighting external feedback higher (per research)."""
        if self.external_score > 0:
            # External feedback is more reliable - weight it higher
            return self.reflection_score * 0.4 + self.external_score * 0.6
        return self.reflection_score

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

def backpropagate(tree_nodes: Dict[str, MCTSNode], node_id: str, value: float, gamma: float = 0.95):
    """Update stats for the lineage of the node with decay.

    Args:
        tree_nodes: Dictionary of all nodes
        node_id: Starting node ID
        value: Value to propagate
        gamma: Decay factor for ancestors (default 0.95)
    """
    current_id = node_id
    current_value = value
    while current_id:
        node = tree_nodes[current_id]
        node.visits += 1
        node.value += current_value
        current_value *= gamma  # Decay for ancestors
        current_id = node.parent_id


def get_depth(tree_nodes: Dict[str, MCTSNode], node_id: str) -> int:
    """Get the depth of a node in the tree."""
    depth = 0
    current_id = node_id
    while current_id:
        node = tree_nodes.get(current_id)
        if not node:
            break
        current_id = node.parent_id
        depth += 1
    return depth


def get_adaptive_branching_factor(tree_nodes: Dict[str, MCTSNode], node: MCTSNode) -> int:
    """Determine branching factor based on uncertainty and depth.

    Research: AB-MCTS dynamically decides whether to branch out or refine deeper.

    Args:
        tree_nodes: Dictionary of all nodes
        node: Current node to expand from

    Returns:
        Number of candidates to generate (2-5)
    """
    # If previous attempts had high variance → explore more
    if node.children_ids:
        child_scores = [tree_nodes[c].value for c in node.children_ids if c in tree_nodes]
        if len(child_scores) > 1:
            import statistics
            variance = statistics.variance(child_scores)
            if variance > 0.3:
                return 5  # High uncertainty → more exploration

    # If we're deep in the tree → focus (fewer candidates)
    depth = get_depth(tree_nodes, node.id)
    if depth > 3:
        return 2  # Deep → exploit

    return 3  # Default


def is_terminal_answer(content: str, query: str) -> bool:
    """Check if node contains a complete, valid final answer.

    Args:
        content: Node content to check
        query: Original query for context

    Returns:
        True if this appears to be a complete answer
    """
    # Check for answer markers
    has_final_answer = "Final Answer:" in content or "Answer:" in content

    if not has_final_answer:
        return False

    # Extract the answer portion
    answer = ""
    for marker in ["Final Answer:", "Answer:"]:
        if marker in content:
            answer = content.split(marker, 1)[1].strip()
            break

    # Basic completeness checks
    if len(answer) < 10:
        return False  # Too short to be a real answer

    # Check it's not just a placeholder or continuation
    incomplete_markers = [
        "I need to search",
        "Let me find",
        "I should look up",
        "<search>",
        "...",
        "to be continued"
    ]

    for marker in incomplete_markers:
        if marker.lower() in answer.lower():
            return False

    return True
