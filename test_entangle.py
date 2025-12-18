# FILE: test_entangle.py
import sys
import torch
from quantum_llm.model import load_model
from quantum_llm.generator import QuantumGenerator

def run_test():
    print("🔋 Powering up the Entangle Engine...")
    
    # 1. Load the "Brain" (Phi-3)
    # This might take a minute to download the model (~4GB) on first run.
    model, tokenizer, device = load_model()
    
    generator = QuantumGenerator(model, tokenizer, device)

    # 2. Simulate RAG Context (The "Knowledge")
    # In a real app, this comes from a Vector DB. Here, we mock it.
    fake_rag_docs = """
    [DOCS: CrewAI Framework]
    To create an agent, use the Agent class.
    REQUIRED ARGUMENTS:
    - role: str
    - goal: str
    - backstory: str
    - verbose: bool (Default: True)
    - allow_delegation: bool (Default: False)
    
    Example:
    researcher = Agent(
        role='Researcher',
        goal='Analyze data',
        backstory='Expert analyst',
        verbose=True
    )
    """

    # 3. Define the Prompt
    user_goal = "Create a CrewAI agent designed to summarize news articles."

    # 4. Run the Multiverse Simulation
    print(f"\n🧠 Initializing Architect Persona with Goal: '{user_goal}'")
    
    root = generator.generate(
        prompt=user_goal,
        rag_context=fake_rag_docs,
        steps=5,        # Keep steps low for a quick test
        branches=3,     # Test 3 parallel timelines
        temperature=0.7 # Slight creativity
    )

    # 5. Visualize Results
    print("\n--- 🌌 ENTANGLED TIMELINES GENERATED ---")
    generator.print_results(root, limit=3)

if __name__ == "__main__":
    run_test()