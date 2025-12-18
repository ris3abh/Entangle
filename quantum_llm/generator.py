import torch
import torch.nn.functional as F
from typing import List, Optional
from .tree import Node, EntangledState

class QuantumGenerator:
    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(
        self,
        prompt: str,
        rag_context: str = "",
        steps: int = 5,
        branches: int = 3,
        max_leaves: int = 10,
        temperature: float = 1.0,
    ) -> Node:
        """
        Generate a tree of architectural decisions (Entangled Multiverse).

        Args:
            prompt: The user's goal (e.g. "Build a stock analyst agent")
            rag_context: Retrieved framework documentation (LangChain/CrewAI docs)
            steps: Number of 'thought steps' (tokens for now) to generate
            branches: How many parallel universes to explore per step
            max_leaves: Pruning limit to focus resources
            temperature: Creativity (higher = more experimental architectures)
        """
        # 1. Initialize the Hive Mind (Global State)
        state = EntangledState()
        state.rag_context = rag_context
        
        # 2. Construct the Architect System Prompt
        # This formats the input to look like an instruction for Phi-3/Llama 3
        full_system_prompt = self._construct_system_prompt(prompt, state)
        
        input_ids = self.tokenizer.encode(full_system_prompt, return_tensors="pt").to(self.device)
        
        # 3. Create Root with Entangled State
        # The state object is now embedded in the root and will be inherited by all children
        root = Node(input_ids[0], full_system_prompt, 0.0, state=state)
        active_leaves = [root]

        print(f"\n--- Starting Entangled Simulation: '{prompt}' ---")

        for step in range(steps):
            if not active_leaves:
                break

            # Core Matrix Math (Unchanged - it's the engine!)
            batch_sequences = self._prepare_batch(active_leaves, root)
            next_token_log_probs = self._forward_pass(batch_sequences, temperature)
            new_leaves = self._branch(active_leaves, next_token_log_probs, branches)

            # Pruning: Keep only the most logical architectural paths
            new_leaves = sorted(new_leaves, key=lambda x: x.score, reverse=True)[
                :max_leaves
            ]
            active_leaves = new_leaves

            print(f"Step {step + 1}: {len(active_leaves)} active timelines")

        return root

    def _construct_system_prompt(self, user_prompt: str, state: EntangledState) -> str:
        """
        Wraps the user request in an Architect Persona.
        Dynamically injects 'Known Failures' from the Hive Mind.
        """
        # Note: Phi-3 and other instruction models expect specific tagging
        # We can adjust this template based on the specific model we loaded
        
        system_instruction = f"""You are an Expert AI Architect specializing in Agentic Frameworks.
        
        CONTEXT & DOCUMENTATION (RAG):
        {state.rag_context}
        
        GLOBAL LESSONS (DO NOT REPEAT MISTAKES):
        {state.failed_patterns if state.failed_patterns else "None so far."}
        
        TASK:
        {user_prompt}
        
        Generate the next logical Python code block to achieve this task. 
        Focus on robust error handling and correct imports."""

        # Format for Phi-3 / Llama-3 Instruct
        return f"<|user|>\n{system_instruction}<|end|>\n<|assistant|>\n"

    def _prepare_batch(self, leaves: List[Node], root: Node) -> torch.Tensor:
        """Prepare batch of sequences from all active leaves."""
        batch_sequences = []

        for leaf in leaves:
            # Reconstruct the path from leaf to root
            curr = leaf
            path_ids = []
            
            # Walk up the tree
            while curr.parent is not None:
                path_ids.append(curr.token_id)
                curr = curr.parent
            
            # Add root (which now contains the full System Prompt)
            path_ids.append(root.token_id)

            # Flip to get correct order (Root -> Child -> Leaf)
            full_seq = (
                torch.cat(list(reversed(path_ids)))
                if isinstance(path_ids[0], torch.Tensor)
                else torch.tensor(path_ids).to(self.device)
            )
            
            if full_seq.dim() == 1:
                full_seq = full_seq.unsqueeze(0)
            batch_sequences.append(full_seq)

        # Padding logic for the GPU batch
        max_len = max([s.size(1) for s in batch_sequences])
        padded_batch = (
            torch.zeros((len(batch_sequences), max_len), dtype=torch.long, device=self.device)
            + self.tokenizer.eos_token_id
        )
        for i, seq in enumerate(batch_sequences):
            padded_batch[i, : seq.size(1)] = seq

        return padded_batch

    def _forward_pass(
        self, batch: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        """Run model forward pass and return log probabilities."""
        with torch.no_grad():
            # We explicitly disable caching to avoid the DynamicCache error
            outputs = self.model(input_ids=batch, use_cache=False) 
            next_token_logits = outputs.logits[:, -1, :]

            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)

        return next_token_log_probs

    def _branch(
        self,
        leaves: List[Node],
        log_probs: torch.Tensor,
        branches: int,
    ) -> List[Node]:
        """Create new nodes by branching from each leaf."""
        new_leaves = []

        for i, leaf in enumerate(leaves):
            probs = log_probs[i]
            top_probs, top_indices = torch.topk(probs, branches)

            for score, idx in zip(top_probs, top_indices):
                token_text = self.tokenizer.decode(idx)
                new_score = leaf.score + score.item()
                
                # New Node inherits the state automatically via the Node class
                new_node = Node(idx.unsqueeze(0), token_text, new_score, parent=leaf)
                leaf.add_child(new_node)
                new_leaves.append(new_node)

        return new_leaves

    def print_results(self, root: Node, limit: int = 3, rich: bool = True) -> None:
        """Print the generated tree."""
        if rich:
            from .visualize import print_rich_tree, print_analysis
            print_rich_tree(root, limit_width=limit)
            print_analysis(root)
        else:
            from .tree import print_tree
            print("\n--- Simulation Results (Tree of Thoughts) ---")
            print_tree(root, limit_width=limit)