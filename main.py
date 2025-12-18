#!/usr/bin/env python3
"""
Quantum LLM Inference Prototype

Explores multiple reasoning paths simultaneously by maintaining
probability vectors instead of collapsing to single tokens.
"""

from quantum_llm.cli import parse_args
from quantum_llm.model import load_model
from quantum_llm.generator import QuantumGenerator


def main():
    args = parse_args()

    model, tokenizer, device = load_model(args.model, args.device)

    generator = QuantumGenerator(model, tokenizer, device)

    root = generator.generate(
        prompt=args.prompt,
        steps=args.steps,
        branches=args.branches,
        max_leaves=args.max_leaves,
        temperature=args.temperature,
    )

    generator.print_results(root, limit=args.branches)


if __name__ == "__main__":
    main()
