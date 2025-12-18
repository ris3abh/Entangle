# Entangle 
### The Self-Correcting Agent Architect

**Entangle** is a "Quantum" Inference Engine that treats LLM reasoning not as a single stream of tokens, but as a multiverse of possibilities entangled by a shared state.

Unlike standard generation which hallucinates in isolation, Entangle explores multiple timelines simultaneously. If one timeline discovers a logic error or contradicts the provided documentation (RAG), it updates a shared "Hive Mind," instantly preventing all other parallel timelines from making the same mistake.

## Core Concepts

### 1. Quantum Parallelism (Inference)
Instead of generating one sequence, Entangle utilizes the GPU's matrix multiplication capabilities to generate $N$ parallel "universes" (branches) simultaneously. This allows the model to explore diverse architectural strategies in real-time.

### 2. The Hive Mind (Entangled State)
Standard Beam Search branches are isolated. In Entangle, all nodes share a `GlobalState`.
- **Registry of Failures:** If Branch A tries a deprecated import and fails, Branch B immediately "knows" not to try it.
- **RAG Enforcement:** Documentation is injected into the Hive Mind, ensuring all branches adhere to framework rules.

## 🗺️ Project Roadmap (The Awesome Phases)

### Phase 1: The Entangled Engine (Current)
* **Brain Transplant:** Replaced GPT-2 with **Microsoft Phi-3 Mini** (Instruction Tuned) for logic and reasoning capabilities.
* **System 2 Prompting:** Wraps simple user goals into complex "Architect" system instructions.
* **State Injection:** Implements the `EntangledState` class to pass RAG context and learned errors down the tree.

### Phase 2: The Self-Correcting Loop (Next)
* **The Critic:** A secondary "Juror" agent that runs at every generation step.
* **The Immune System:**
    1.  **Generate:** The Architect proposes a step.
    2.  **Critique:** The Critic checks against RAG docs (e.g., "Did you use the correct arguments for `Agent` class?").
    3.  **Kill & Learn:** If the check fails, the branch is pruned, and the specific error pattern is added to the Hive Mind.

## Installation

```bash
pip install -r requirements.txt
```

Requirements: torch, transformers, accelerate, rich. Note: This project requires a model download (~4GB) on the first run.

## Usage
Run the test suite to simulate an Architect designing a CrewAI agent:

```bash
python test_entangle.py
```

## Structure

```bash
entangle/
├── quantum_llm/
│   ├── generator.py   # The Engine: Manages the multiverse loop
│   ├── model.py       # The Brain: Loads Phi-3/Llama-3
│   ├── tree.py        # The Memory: Nodes & EntangledState
│   ├── critic.py      # The Judge: Phase 2 logic validation
│   └── visualize.py   # The UI: Renders the Tree of Thoughts
├── test_entangle.py   # Main entry point
└── README.md
```



