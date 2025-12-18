import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantum LLM - Explore multiple reasoning paths simultaneously"
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        required=True,
        help="Initial prompt text",
    )
    parser.add_argument(
        "--steps",
        "-s",
        type=int,
        default=5,
        help="Number of tokens to generate (default: 5)",
    )
    parser.add_argument(
        "--branches",
        "-b",
        type=int,
        default=3,
        help="Number of branches per node (default: 3)",
    )
    parser.add_argument(
        "--max-leaves",
        "-m",
        type=int,
        default=10,
        help="Maximum active leaves for pruning (default: 10)",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=1.0,
        help="Sampling temperature - higher = more diverse (default: 1.0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name (default: gpt2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect)",
    )

    return parser.parse_args()
