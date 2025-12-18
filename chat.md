# Research Log: Batch Processing & Quantum Simulation

## Theory: Vector vs. Matrix Inference

**Q: Consider using not a numerical but vector type for LLMs to multiply matrices - will I be able to process multiple prompts at once this way during inferencing?**

**A:** Yes, you are absolutely on the right track. What you are describing—moving from processing single numerical inputs (scalars) or single vectors to processing groups of vectors combined into a matrix—is the fundamental mechanical principle behind **Batch Processing**.

To answer your specific question: Yes, treating the input as a "stack" of vectors (a matrix) allows you to process multiple prompts simultaneously during inference.

### 1. The Math: From Vector to Matrix
In a standard Large Language Model, the core operation is Matrix Multiplication (MatMul).

**Single Prompt Inference (Vector-Matrix)**
Input ($x$): A vector of shape $(1,D)$.
Weights ($W$): A matrix of shape $(D, D_{out})$.
Operation: $y = xW$

**Multiple Prompt Inference (Matrix-Matrix)**
Input ($X$): A matrix of shape $(B,D)$, where $B$ is Batch Size.
Weights ($W$): Stays the same.
Operation: $Y = XW$

By doing this, the GPU performs one large operation rather than many small ones, which is significantly more efficient.

### 2. Why this enables "Simultaneous" Processing
Hardware accelerators (GPUs/TPUs) are designed for SIMT (Single Instruction, Multiple Threads).
* **Single Prompt:** Much of the compute capacity sits idle.
* **Batch:** The GPU loads weights once and applies them to all prompts in parallel. The cost for 10 prompts is often nearly the same as 1.

### 3. The Engineering Challenge: "Ragged" Batches
Real-world text creates a problem: Prompts have different lengths.
* **Naive Solution (Padding):** Add zeros to match the longest prompt. Wastes compute.
* **Modern Solution (FlashAttention & Continuous Batching):** Engines like vLLM do not physically stack vectors into a rigid rectangle. They use **PagedAttention** to manage memory in non-contiguous blocks, allowing fluid processing of ragged batches.

---

## Practice: Is this standard?

**Q: Yes, is this widely used practice or novel idea?**

**A:** It is a widely used, industry-standard practice. It is the primary way commercial LLMs serve millions of users.

* **Static Batching (Old):** Wait for 4 requests, pad them, run. High latency for short queries trapped behind long ones.
* **Continuous Batching (New ~2023):** Treat the batch as a fluid list. As soon as Request A finishes, its slot is freed for Request C, even if Request B is still generating.

---

## Sampling & Temperature (Russian)

**Q: Скажи, а вообще это э правильно ли, что говорит Карпаты в своём видео, что когда э LLM работает, то на каждом шаге делается random sampling?**

**A:** Да, Карпаты прав. LLM выбирают токены не всегда самый частый, а используют случайную выборку из распределения вероятностей.

**Q: Какой параметр для этого используется?**

**A:** Основной параметр — **Temperature**.
* **Высокая температура:** Модель более склонна выбирать токены с меньшей вероятностью (творчество/хаос).
* **Низкая температура:** Модель выбирает токены с высокой вероятностью (предсказуемость).
* **Top-p (Nucleus Sampling):** Ограничивает выборку только суммарной вероятностью $P$.

**Q: При каком значении температуры выборка не случайная становится?**

**A:** При `Temperature = 0` выборка становится детерминированной (всегда выбирается токен с максимальной вероятностью).

---

## The "Quantum" Simulation Experiment

**Q: Давай предположим, что мы моделируем квантовый компьютер с помощью LLM... На каждом шаге генерации LLM будет хранить вектор вероятностей следующих токенов...**

**A:** Отлично, давайте представим такой сценарий! В этом случае LLM будет как бы находиться в состоянии **суперпозиции мыслей**.

**Как это будет работать:**
1.  **На входе:** Вместо одного токена подаем матрицу состояний.
2.  **Обработка:** На каждом шаге пересчитываем вероятности для всех путей параллельно.
3.  **Измерение:** Чтобы получить ответ, мы производим выборку (Collapse the wave function).

**Q: Design such prototype based on any LLM and inference engine.**

**A:** Это отличная инженерная задача. Мы можем реализовать это на Python используя Beam Search логику, но с сохранением всего спектра вероятностей.

### Python Prototype (GPT-2)

```python
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 1. Setup Quantum Machine
model_name = "gpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
tokenizer.pad_token = tokenizer.eos_token

def quantum_step_generation(prompt_text, steps=5, branches=3):
    initial_ids = tokenizer.encode(prompt_text, return_tensors='pt').to(device)
    paths = [(initial_ids, 0.0)] 

    for step in range(steps):
        # PHASE 1: Parallel Batch Processing
        batch_inputs = torch.cat([p[0] for p in paths], dim=0)
        
        with torch.no_grad():
            outputs = model(batch_inputs)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)

        # PHASE 2: Branching (Superposition)
        all_candidates = []
        for i, (current_ids, current_score) in enumerate(paths):
            probs = next_token_log_probs[i]
            top_probs, top_indices = torch.topk(probs, branches)
            
            for log_prob, token_index in zip(top_probs, top_indices):
                new_score = current_score + log_prob.item()
                new_ids = torch.cat([current_ids, token_index.unsqueeze(0).unsqueeze(0)], dim=1)
                all_candidates.append((new_ids, new_score))

        # PHASE 3: Pruning (Interference)
        paths = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:10]
```

Interpretation
Matrix Approach: torch.cat allows us to process all "universes" in one forward pass.


Measurement:

Maximum Likelihood: The path with the highest score (logical).

Entropy Analysis: If all paths diverge, the model is uncertain. If they converge, it is confident.

Visualization: The Thought Tree
The final step is visualizing these paths not just as a list, but as a branching tree to see exactly where the model "hesitated" (Bifurcation points).

Plaintext

|-- [0.00] 'The meaning of life is'
    |-- [-2.45] ' to'
    |    |-- [-4.10] ' find'
    |-- [-3.10] ' not'
[-2.45]: Log-Probability.


Bifurcation: If two children have very similar scores (e.g., -2.45 vs -2.50), the model is in a high-entropy state (Quantum Uncertainty).
