#!/usr/bin/env python3
"""
Multiverse - Interactive LLM Thought Tree Explorer
Gradio web interface for exploring parallel reasoning paths.
"""

import gradio as gr
import torch
from quantum_llm.model import load_model
from quantum_llm.generator import QuantumGenerator
from quantum_llm.visualize import generate_html_tree, analyze_tree
from quantum_llm.tree import get_leaf_nodes

model, tokenizer, device = None, None, None
generator = None


def initialize_model():
    global model, tokenizer, device, generator
    if model is None:
        model, tokenizer, device = load_model("gpt2")
        generator = QuantumGenerator(model, tokenizer, device)
    return generator


def format_analysis(analysis: dict) -> str:
    """Format analysis as markdown."""
    if not analysis:
        return "No analysis available"

    conv = analysis["convergence_score"]
    conv_emoji = "🟢" if conv > 70 else "🟡" if conv > 40 else "🔴"

    md = f"""
### Analysis Results

| Metric | Value |
|--------|-------|
| {conv_emoji} Convergence | **{conv:.1f}%** |
| 🎯 Dominant Path | {analysis['dominant_path_probability']:.1f}% |
| 🌿 Paths Explored | {analysis['total_paths']} |
| 📈 Entropy | {analysis['entropy']:.2f} bits |

"""

    if analysis.get("bifurcation_points"):
        md += "\n### 🔀 Key Hesitation Points\n\n"
        for bp in analysis["bifurcation_points"][:3]:
            opts = " **vs** ".join([f"`{o.strip()}`" for o in bp["options"]])
            md += f"- {opts}\n"

    interpretation = ""
    if conv > 70:
        interpretation = "**Interpretation:** The model is quite confident - most paths converge to similar conclusions."
    elif conv > 40:
        interpretation = "**Interpretation:** Moderate uncertainty - the model sees multiple viable directions."
    else:
        interpretation = "**Interpretation:** High divergence! The model is exploring very different possibilities."

    md += f"\n{interpretation}"

    return md


def get_top_paths(root, limit: int = 5) -> str:
    """Get top completed paths as markdown."""
    leaves = get_leaf_nodes(root)
    if not leaves:
        return "No paths generated"

    sorted_leaves = sorted(leaves, key=lambda x: x.score, reverse=True)[:limit]

    md = "### 🏆 Top Reasoning Paths\n\n"
    for i, leaf in enumerate(sorted_leaves, 1):
        full_text = leaf.get_full_text()
        prob_pct = torch.exp(torch.tensor(leaf.score)).item() * 100
        md += f"**{i}.** `{full_text}` (score: {leaf.score:.2f})\n\n"

    return md


def explore_multiverse(
    prompt: str,
    steps: int,
    branches: int,
    temperature: float,
    max_leaves: int,
):
    """Main generation function."""
    if not prompt.strip():
        return "<p>Please enter a prompt</p>", "Enter a prompt to begin", ""

    gen = initialize_model()

    root = gen.generate(
        prompt=prompt,
        steps=int(steps),
        branches=int(branches),
        max_leaves=int(max_leaves),
        temperature=temperature,
    )

    html = generate_html_tree(root, limit_width=int(branches))
    analysis = analyze_tree(root)
    analysis_md = format_analysis(analysis)
    paths_md = get_top_paths(root, limit=5)

    return html, analysis_md, paths_md


EXAMPLES = [
    ["The meaning of life is", 5, 3, 1.0, 12],
    ["AI will", 5, 3, 1.0, 12],
    ["The best way to learn is", 5, 3, 1.0, 12],
    ["Love is", 4, 3, 1.2, 12],
    ["In the future, humans will", 5, 3, 1.0, 12],
    ["The secret to happiness is", 5, 3, 1.0, 12],
]

CSS = """
.container { max-width: 1400px; margin: auto; }
.title { text-align: center; margin-bottom: 20px; }
.subtitle { text-align: center; color: #666; margin-bottom: 30px; }
"""

with gr.Blocks(css=CSS, title="Multiverse - LLM Thought Explorer") as demo:
    gr.HTML(
        """
        <div class="title">
            <h1>🌌 Multiverse</h1>
            <p class="subtitle">Explore the parallel universes of LLM reasoning</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here...",
                value="The meaning of life is",
                lines=2,
            )

            with gr.Row():
                steps_slider = gr.Slider(
                    minimum=2,
                    maximum=8,
                    value=5,
                    step=1,
                    label="Depth (tokens)",
                    info="How many tokens to generate",
                )
                branches_slider = gr.Slider(
                    minimum=2,
                    maximum=5,
                    value=3,
                    step=1,
                    label="Branches",
                    info="Paths per node",
                )

            with gr.Row():
                temp_slider = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Temperature",
                    info="Higher = more diverse",
                )
                max_leaves_slider = gr.Slider(
                    minimum=6,
                    maximum=20,
                    value=12,
                    step=2,
                    label="Max Paths",
                    info="Limit total active paths",
                )

            explore_btn = gr.Button("🚀 Explore Multiverse", variant="primary", size="lg")

            gr.Examples(
                examples=EXAMPLES,
                inputs=[prompt_input, steps_slider, branches_slider, temp_slider, max_leaves_slider],
                label="Try these prompts",
            )

        with gr.Column(scale=2):
            tree_output = gr.HTML(label="Thought Tree")

    with gr.Row():
        with gr.Column():
            analysis_output = gr.Markdown(label="Analysis")
        with gr.Column():
            paths_output = gr.Markdown(label="Top Paths")

    explore_btn.click(
        fn=explore_multiverse,
        inputs=[prompt_input, steps_slider, branches_slider, temp_slider, max_leaves_slider],
        outputs=[tree_output, analysis_output, paths_output],
    )

    gr.HTML(
        """
        <div style="text-align: center; margin-top: 40px; color: #666; font-size: 14px;">
            <p>Built with GPT-2 | Instead of sampling one token, we explore ALL probable paths</p>
            <p>🔗 <a href="https://github.com/your-repo/multiverse" target="_blank">GitHub</a></p>
        </div>
        """
    )


if __name__ == "__main__":
    demo.launch(share=True)
