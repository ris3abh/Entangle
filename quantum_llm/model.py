# FILE: quantum_llm/model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name: str = "microsoft/Phi-3-mini-4k-instruct", device: str = None):
    """
    Load the Architect Model (Phi-3) and tokenizer.
    Uses float16 for memory efficiency.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Architect Model {model_name} on {device}...")
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load Model with optimizations
    # float16 reduces VRAM usage by ~50%
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True
    ).to(device)

    model.eval()

    # Ensure we have a pad token (Phi-3 usually has one, but safety first)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device