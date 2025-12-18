import torch
from typing import Optional, List, Set, Any

class EntangledState:
    """
    The Hive Mind shared by all branches.
    Stores lessons learned from failures so other branches don't repeat them.
    """
    def __init__(self):
        self.failed_patterns: List[str] = []   # e.g., "Don't use x.run(), it failed in Branch A"
        self.verified_imports: Set[str] = set() # e.g., "from crewai import Agent works"
        self.rag_context: str = ""       # The framework docs loaded at start
        self.artifacts: dict = {}        # Shared storage for generated code snippets

class Node:
    def __init__(
        self,
        token_id: torch.Tensor,
        text: str,
        score: float,
        parent: Optional["Node"] = None,
        state: Optional[EntangledState] = None
    ):
        self.token_id = token_id
        self.text = text
        self.score = score  # Accumulated log-probability
        self.parent = parent
        
        # --- NEW LOGIC: Entanglement ---
        # If a specific state is provided (usually at root), use it.
        # Otherwise, inherit the reference to the same state object from the parent.
        if state is not None:
            self.state = state
        elif parent is not None:
            self.state = parent.state
        else:
            # Fallback for isolated nodes (should rarely happen in Entangle)
            self.state = EntangledState()
            
        self.children: List["Node"] = []
        
        # You might want to store local critiques here later
        self.local_critique: Optional[str] = None

    def add_child(self, child: "Node") -> None:
        self.children.append(child)

    def get_path_ids(self) -> torch.Tensor:
        """Reconstruct full token sequence from root to this node."""
        path_ids = []
        curr = self
        while curr is not None:
            path_ids.append(curr.token_id)
            curr = curr.parent
        path_ids.reverse()
        return torch.cat(path_ids) if path_ids else torch.tensor([])

    def get_full_text(self) -> str:
        """Get concatenated text from root to this node."""
        texts = []
        curr = self
        while curr is not None:
            texts.append(curr.text)
            curr = curr.parent
        texts.reverse()
        return "".join(texts)


def print_tree(node: Node, level: int = 0, limit_width: int = 3) -> None:
    """Recursively print tree structure to console."""
    indent = "    " * level
    display_text = node.text.replace("\n", "\\n")
    print(f"{indent}|-- [{node.score:.2f}] '{display_text}'")

    sorted_children = sorted(node.children, key=lambda x: x.score, reverse=True)[
        :limit_width
    ]
    for child in sorted_children:
        print_tree(child, level + 1, limit_width)


def get_leaf_nodes(root: Node) -> List[Node]:
    """Get all leaf nodes (endpoints) from the tree."""
    leaves = []

    def traverse(node: Node):
        if not node.children:
            leaves.append(node)
        else:
            for child in node.children:
                traverse(child)

    traverse(root)
    return leaves