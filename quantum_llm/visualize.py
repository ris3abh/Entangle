import math
from typing import List, Tuple
from rich.console import Console
from rich.tree import Tree as RichTree
from rich.text import Text
from rich.panel import Panel
from .tree import Node, get_leaf_nodes


def score_to_color(score: float) -> str:
    """Map log-probability score to color. Higher (closer to 0) = greener."""
    if score > -2:
        return "bright_green"
    elif score > -4:
        return "green"
    elif score > -6:
        return "yellow"
    elif score > -8:
        return "orange1"
    else:
        return "red"


def score_to_bar(score: float, max_width: int = 10) -> str:
    """Create visual probability bar."""
    prob = math.exp(score)
    filled = int(prob * max_width * 10)
    filled = min(filled, max_width)
    return "█" * filled + "░" * (max_width - filled)


def print_rich_tree(root: Node, limit_width: int = 3) -> None:
    """Print tree with rich colors and formatting."""
    console = Console()

    def build_rich_tree(node: Node, rich_node: RichTree, depth: int = 0):
        sorted_children = sorted(node.children, key=lambda x: x.score, reverse=True)[
            :limit_width
        ]
        for child in sorted_children:
            color = score_to_color(child.score)
            bar = score_to_bar(child.score)
            display_text = child.text.replace("\n", "\\n")

            label = Text()
            label.append(f"[{child.score:+.2f}] ", style="dim")
            label.append(bar + " ", style=color)
            label.append(f"'{display_text}'", style=f"bold {color}")

            child_tree = rich_node.add(label)
            build_rich_tree(child, child_tree, depth + 1)

    display_text = root.text.replace("\n", "\\n")
    tree = RichTree(
        Text(f"🌌 '{display_text}'", style="bold bright_white"),
        guide_style="dim",
    )
    build_rich_tree(root, tree)

    console.print()
    console.print(
        Panel(tree, title="[bold cyan]Multiverse Tree[/]", border_style="cyan")
    )
    console.print()


def generate_html_tree(root: Node, limit_width: int = 3) -> str:
    """Generate interactive HTML visualization."""
    html_parts = [
        """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Multiverse - LLM Thought Tree</title>
    <style>
        body {
            font-family: 'SF Mono', 'Fira Code', monospace;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            padding: 40px;
            min-height: 100vh;
        }
        .tree-container {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 30px;
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #00d4ff;
            margin-bottom: 30px;
        }
        .node {
            margin-left: 30px;
            padding: 8px 0;
            border-left: 2px solid #333;
            padding-left: 20px;
        }
        .node-content {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 8px 16px;
            background: rgba(255,255,255,0.03);
            border-radius: 8px;
            margin-bottom: 4px;
            transition: background 0.2s;
        }
        .node-content:hover {
            background: rgba(255,255,255,0.08);
        }
        .score {
            font-size: 12px;
            color: #888;
            min-width: 60px;
        }
        .bar {
            width: 100px;
            height: 8px;
            background: #333;
            border-radius: 4px;
            overflow: hidden;
        }
        .bar-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s;
        }
        .text {
            font-weight: 600;
        }
        .root-node {
            font-size: 1.3em;
            color: #00d4ff;
            padding: 16px;
            background: rgba(0,212,255,0.1);
            border-radius: 12px;
            margin-bottom: 20px;
            text-align: center;
        }
        .high { background: linear-gradient(90deg, #00ff88, #00cc6a); }
        .medium-high { background: linear-gradient(90deg, #88ff00, #66cc00); }
        .medium { background: linear-gradient(90deg, #ffcc00, #cc9900); }
        .medium-low { background: linear-gradient(90deg, #ff8800, #cc6600); }
        .low { background: linear-gradient(90deg, #ff4444, #cc3333); }
        .legend {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            color: #888;
        }
        .legend-bar {
            width: 40px;
            height: 8px;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="tree-container">
        <h1>🌌 Multiverse - LLM Thought Tree</h1>
        <div class="legend">
            <div class="legend-item"><div class="legend-bar high"></div> High confidence</div>
            <div class="legend-item"><div class="legend-bar medium"></div> Medium</div>
            <div class="legend-item"><div class="legend-bar low"></div> Low confidence</div>
        </div>
"""
    ]

    def get_color_class(score: float) -> str:
        if score > -2:
            return "high"
        elif score > -4:
            return "medium-high"
        elif score > -6:
            return "medium"
        elif score > -8:
            return "medium-low"
        return "low"

    def get_bar_width(score: float) -> int:
        prob = math.exp(score)
        return min(100, int(prob * 1000))

    def render_node(node: Node, is_root: bool = False) -> str:
        if is_root:
            return f'<div class="root-node">🎯 "{node.text}"</div>'

        display_text = node.text.replace("<", "&lt;").replace(">", "&gt;")
        color_class = get_color_class(node.score)
        bar_width = get_bar_width(node.score)

        children_html = ""
        sorted_children = sorted(node.children, key=lambda x: x.score, reverse=True)[
            :limit_width
        ]
        for child in sorted_children:
            children_html += render_node(child)

        return f"""
        <div class="node">
            <div class="node-content">
                <span class="score">[{node.score:+.2f}]</span>
                <div class="bar"><div class="bar-fill {color_class}" style="width: {bar_width}%"></div></div>
                <span class="text">'{display_text}'</span>
            </div>
            {children_html}
        </div>
        """

    html_parts.append(render_node(root, is_root=True))

    sorted_children = sorted(root.children, key=lambda x: x.score, reverse=True)[
        :limit_width
    ]
    for child in sorted_children:
        html_parts.append(render_node(child))

    html_parts.append(
        """
    </div>
</body>
</html>
"""
    )

    return "".join(html_parts)


def analyze_tree(root: Node) -> dict:
    """Analyze tree for interesting metrics."""
    leaves = get_leaf_nodes(root)

    if not leaves:
        return {}

    scores = [leaf.score for leaf in leaves]
    max_score = max(scores)
    min_score = min(scores)

    probs = [math.exp(s) for s in scores]
    total_prob = sum(probs)
    normalized_probs = [p / total_prob for p in probs] if total_prob > 0 else probs

    entropy = -sum(p * math.log2(p) if p > 0 else 0 for p in normalized_probs)
    max_entropy = math.log2(len(leaves)) if len(leaves) > 1 else 1

    dominant_prob = max(normalized_probs) if normalized_probs else 0

    bifurcation_points = []

    def find_bifurcations(node: Node, path: str = ""):
        if len(node.children) >= 2:
            sorted_children = sorted(
                node.children, key=lambda x: x.score, reverse=True
            )[:2]
            score_diff = abs(sorted_children[0].score - sorted_children[1].score)
            if score_diff < 1.0:
                bifurcation_points.append(
                    {
                        "location": path or "root",
                        "options": [c.text for c in sorted_children],
                        "score_diff": score_diff,
                    }
                )
        for i, child in enumerate(node.children):
            find_bifurcations(child, f"{path}/{child.text.strip()}")

    find_bifurcations(root)

    return {
        "total_paths": len(leaves),
        "entropy": entropy,
        "normalized_entropy": entropy / max_entropy if max_entropy > 0 else 0,
        "dominant_path_probability": dominant_prob * 100,
        "convergence_score": (1 - entropy / max_entropy) * 100 if max_entropy > 0 else 100,
        "bifurcation_points": bifurcation_points[:5],
        "score_range": (min_score, max_score),
    }


def print_analysis(root: Node) -> None:
    """Print analysis with rich formatting."""
    console = Console()
    analysis = analyze_tree(root)

    if not analysis:
        console.print("[yellow]No analysis available (empty tree)[/]")
        return

    console.print()
    console.print(Panel("[bold]Tree Analysis[/]", border_style="magenta"))

    conv = analysis["convergence_score"]
    conv_color = "green" if conv > 70 else "yellow" if conv > 40 else "red"
    console.print(f"  📊 Convergence Score: [{conv_color}]{conv:.1f}%[/]")
    console.print(f"  🎯 Dominant Path Probability: {analysis['dominant_path_probability']:.1f}%")
    console.print(f"  🌿 Total Paths Explored: {analysis['total_paths']}")
    console.print(f"  📈 Entropy: {analysis['entropy']:.2f} bits")

    if analysis["bifurcation_points"]:
        console.print("\n  🔀 [bold]Key Bifurcation Points[/] (where model hesitated):")
        for bp in analysis["bifurcation_points"][:3]:
            opts = " vs ".join([f"'{o.strip()}'" for o in bp["options"]])
            console.print(f"     • {opts} (Δ={bp['score_diff']:.2f})")

    console.print()
