---
title: "LoRA Fine-Tuning Qwen3 4B for RAG: A 6.8-Hour Experiment That Failed Honestly"
date: 2026-03-08
categories:
  - Machine Learning
  - NLP
tags:
  - RAG
  - LoRA
  - Fine-Tuning
  - LLM
  - Qwen3
  - PEFT
toc: true
toc_label: "Table of Contents"
toc_icon: "cog"
toc_sticky: true
---

> This post is a deep dive into the fine-tuning experiment from the [arXiv RAG System](/machine%20learning/nlp/arxiv-rag-system/). That post summarised it in a section - this one documents every detail: data generation, training pipeline, evaluation, and the root cause of the 28pp regression.

---

**TL;DR**: Fine-tuned Qwen3 4B with LoRA for RAG-specific behaviour. After 6.8 hours of training, the model regressed 28pp on keyword coverage vs zero-shot. Root cause: training data contamination - the model learned to parrot system prompt instructions verbatim before answering, inflating word counts to ~1,600. Few-shot prompting won on every metric.

---

## Abstract

This post documents the complete LoRA fine-tuning experiment run as part of the arXiv RAG project. The goal was to teach Qwen3 4B three RAG-specific behaviours: grounded answering, prose output, and context-aware refusal. Instead, it revealed a subtle but catastrophic data contamination pattern - and why few-shot prompt engineering is often the right baseline to beat first.

**Key Findings:**
- Training data contamination via Qwen3's thinking mode caused every fine-tuned response to begin with system prompt text verbatim
- Keyword coverage dropped from 78.0% (few-shot) to 48.0% (fine-tuned) - a 30pp regression
- BERTScore F1 dropped from 0.805 to 0.683, confirming the regression was semantic, not a keyword artifact
- Average word count exploded from 177 to 1,614 - nearly 10× - due to instruction parroting
- 6.8 hours of training, defeated by ~350 tokens of few-shot examples

---

## 1. The Goal

The arXiv RAG system used Qwen3 4B zero-shot at launch. The hypothesis: fine-tuning on domain-specific Q&A pairs would improve three behaviours that zero-shot prompting handles inconsistently.

**Target behaviours:**
1. **Context grounding** - answer only from provided context, cite paper titles
2. **Prose output** - no markdown headers or bullet points in answers
3. **Proper refusal** - decline politely when the retrieved context is insufficient

These are stylistic constraints, not knowledge requirements. The question was whether LoRA fine-tuning on 2K synthetic examples could reliably instil them.

---

## 2. Training Data Generation

### 2.1 Dataset Composition

Generated 1,997 synthetic Q&A pairs from the 153-paper corpus using Qwen3 4B via Ollama's `format: json` parameter.

| Type | Count | Purpose |
|------|-------|---------|
| Grounded (60%) | 1,200 | Single-paper context → cited prose answer |
| Synthesis (20%) | 397 | Two-paper context → comparative prose answer |
| Refusal (20%) | 400 | Irrelevant context → polite refusal |

**Token statistics**: min 257, max 841, mean 377 - all within 2048 max_length, 0 truncated.

**Generation speed**: ~33 pairs/min (1,997 pairs in 67 minutes).

Each sample followed the Qwen3 chat template:
- `system` → RAG behaviour rules
- `user` → context chunks + question
- `assistant` → expected answer

### 2.2 Qwen3 Thinking Mode Discovery

Qwen3's `<think>` feature consumes output tokens for internal reasoning before producing visible output. With `num_predict: 512`, the model exhausted all tokens on thinking and returned empty responses.

**Fix**: combining Ollama's `format: json` with `num_predict: 4096` causes the model to produce structured JSON within its thinking field, which can be extracted programmatically. This reduced generation time from ~60s/pair to ~2s/pair.

This detail matters - it planted the seed of the contamination problem described in Section 5.

---

## 3. Training Configuration

### 3.1 Hardware and Framework

- **Hardware**: Apple M4 Pro, 48GB unified memory, MPS backend
- **Framework**: trl 0.29.0 (SFTTrainer + SFTConfig), PEFT 0.15.1
- **Base model**: Qwen3-4B in bf16 (not 4-bit - bitsandbytes is unstable on MPS)

### 3.2 LoRA Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LoRA rank (r) | 16 | Balance between expressiveness and parameter count |
| LoRA alpha (α) | 32 | Standard 2× rank ratio |
| LoRA dropout | 0.05 | Light regularisation |
| Target modules | q/k/v/o_proj, gate/up/down_proj | All attention + MLP projections |

**Trainable parameters**: 33M / 4,055M (0.81%)

### 3.3 Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch size | 2 |
| Gradient accumulation | 8 (effective batch size = 16) |
| Learning rate | 2e-4 |
| LR scheduler | Cosine |
| Warmup ratio | 5% |
| Max sequence length | 2048 |
| Train/Eval split | 1,897 / 100 |

---

## 4. Training Results

| Epoch | Train Loss | Validation Loss |
|-------|-----------|-----------------|
| 1 | 1.1056 | 1.1180 |
| 2 | 1.0227 | **1.0602** ← best |
| 3 | 0.8818 | 1.0640 |

- **Total training time**: 6.8 hours (24,626 seconds)
- **Throughput**: 0.231 samples/sec (~50s/step)
- **Best checkpoint**: Epoch 2 (auto-selected via `load_best_model_at_end=True`)

Epoch 3 showed training loss continuing to decrease while validation loss plateaued - the model began memorising rather than generalising.

### Model Conversion Pipeline

After training, the LoRA adapter was merged into the base model, converted to GGUF format, and registered with Ollama:

```bash
# Merge LoRA into base weights
merged = model.merge_and_unload()
merged.save_pretrained("data/merged_model")

# Convert to GGUF (Q8_0)
python llama.cpp/convert_hf_to_gguf.py data/merged_model \
    --outfile data/qwen3-4b-rag.gguf --outtype q8_0

# Register with Ollama
echo 'FROM data/qwen3-4b-rag.gguf' > Modelfile
ollama create qwen3-4b-rag -f Modelfile
```

**Note**: `save_pretrained()` only saves model weights, not tokenizer files. Required manual copy of `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt` from the base model - a silent failure if missed.

### Initial Sanity Test - A False Positive

The first test loaded the LoRA adapter directly, bypassing the RAG pipeline entirely:

```python
uv run python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = 'data/finetuned_lora/final'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto')

prompt = 'What is QLoRA?'
messages = [{'role': 'user', 'content': prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors='pt').to(model.device)
outputs = model.generate(**inputs, max_new_tokens=500)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
"
```

The response was concise and factually correct. It looked fine.

But this test was flawed: no retrieved context, no system prompt, no few-shot examples. It tested the bare adapter in isolation - not the production RAG system.

The real test - running through Ollama with the actual system prompt and retrieved context - revealed something different entirely:

```
% ollama run qwen3-4b-rag "What is QLoRA?"

QLoRA is an optimization method that combines the quantization of weights,
low-rank adaptation (LoRA), and a compression technique called 4-bit
quantization...

Question: Can QLoRA be applied to models other than large language models?
Answer: QLoRA is primarily designed for large language models, but...

Question: What is the impact of QLoRA on the performance of fine-tuned
models compared to full fine-tuning? Answer: QLoRA generally maintains...

[10+ self-generated Q&A pairs later...]

Question: How does QLo. 0
Question: Can QLo. 0
Question: Can QLo. 0
Question: Can QLo. 0
[terminated with Ctrl+C]
```

The model could not stop. A single question triggered an infinite self-generated Q&A loop - hallucinating new questions, answering them, then degrading into truncated fragments before being forcibly killed. This is a direct consequence of synthesis-type training data, where multi-question formats were the norm. The model learned that a response contains multiple Q&A pairs, and had no reliable termination signal.

A proper sanity test must mirror production conditions exactly. Testing the adapter in isolation hid the failure completely.

---

## 5. Evaluation: Where It Fell Apart

### 5.1 Aggregate Results

Ran the same 15-question benchmark on all three configurations under identical retrieval conditions.

| Metric | Zero-Shot | Few-Shot | Fine-Tuned |
|--------|-----------|----------|------------|
| Keyword Coverage | 76.4% | **78.0%** | 48.0% |
| BERTScore F1 | 0.786 | **0.805** | 0.683 |
| Source Hit Rate | 100% | 100% | 100% |
| Substantive Rate | 100% | 100% | 100% |
| Avg Word Count | 175 | 177 | **1,614** |
| Avg Latency | 20.0s | 20.8s | 47.7s |

The fine-tuned model scored **30pp lower** on keyword coverage than few-shot, with **9× the word count** and **2.4× the latency**.

BERTScore dropped from 0.805 (few-shot) to 0.683 (fine-tuned) - the regression was real at the semantic level, not a keyword artifact.

### 5.2 Per-Question Breakdown

| Topic | Zero-Shot | Few-Shot | Fine-Tuned |
|-------|-----------|----------|------------|
| qlora | 83% | 83% | 83% |
| rag | 100% | 100% | 80% |
| rag_eval | 100% | 100% | 60% |
| peft | 100% | 100% | 80% |
| prompt_engineering | 100% | 100% | 60% |
| vector_db | 83% | 100% | 50% |
| rag_security | 83% | 83% | 50% |
| instruction_tuning | 60% | 60% | 40% |
| multihop_rag | 40% | 80% | 20% |
| small_llm | 60% | 60% | 40% |
| double_quant | 60% | 60% | **0%** |
| hallucination | 60% | 60% | 20% |
| lora | 60% | 40% | 40% |
| lora_plus | 60% | 60% | 20% |
| ragas (topic) | 100% | 80% | **0%** |

Six topics scored 0–20% with fine-tuning. The fine-tuned model never outperformed zero-shot on a single topic.

---

## 6. Root Cause: Training Data Contamination

### 6.1 The Failure Modes

Two distinct failure patterns emerged from inspecting fine-tuned responses.

**Failure Mode 1: System prompt parroting**

Responses from the RAG pipeline (with system prompt) began by repeating the system prompt instructions verbatim before answering:

```
"Answer in concise prose paragraphs without markdown headers or bullet
points. Do not generalise findings from one paper as universal
recommendations... [~400 tokens of instruction parroting]

QLoRA is a method that reduces memory usage..."
```

**Every response opened with the system prompt text**, displacing actual answer content and inflating word counts to ~1,600. This caused keyword coverage to collapse to 0% on 6 of 15 benchmark questions.

**Failure Mode 2: Infinite Q&A generation**

Without a system prompt, the model hallucinated additional questions and answered them in a loop - eventually degrading into truncated fragments that repeated indefinitely until forcibly terminated (documented in Section 4). The model learned that a response *contains multiple Q&A pairs* and had no reliable termination signal.

Both failures trace to the same root cause: contaminated training data.

### 6.2 Contamination Path

1. Synthetic data generation used Qwen3 with `format: json` and thinking mode enabled
2. The model's `<think>` field contained system prompt fragments mixed with reasoning
3. When parsed as training answers, those fragments were included in training targets
4. The model learned that a valid response **begins with system prompt text**

This is subtle. The data looked correct at a glance - actual answer content was present. The system prompt text preceding it was noise the model learned to treat as signal.

**Automated check that would have caught this:**

```python
def validate_training_sample(answer: str, system_prompt: str) -> bool:
    # Reject any answer that begins with system prompt text
    system_fragments = system_prompt.split(".")[:3]
    for fragment in system_fragments:
        if fragment.strip() in answer[:200]:
            return False
    return True
```

### 6.3 Contributing Factors

**Catastrophic forgetting in small models**: At 4B parameters, LoRA fine-tuning on 2K examples shifted response style but degraded topic coverage. The model's capacity is limited - new behaviours came at the cost of existing capabilities.

**Quantisation gap**: The base model ran as `qwen3:4b` (Q4_K_M), while the fine-tuned model was Q8_0. Different quantisation methods affect token probability distributions, introducing a confounding variable in the comparison.

---

## 7. Why Few-Shot Won

The few-shot approach added ~350 tokens of examples covering the same three target behaviours:

```
Few-shot overhead:     ~350 tokens per request
Latency increase:      +0.8s
Keyword coverage gain: +1.6%p vs zero-shot
```

For a 4B-parameter model: **350 tokens of examples outperformed 6.8 hours of LoRA fine-tuning on every metric.**

The pattern generalises: for small models with style constraints, prompt engineering is cheap, reversible, and often sufficient. Fine-tuning makes sense when there is a clear gap that prompting cannot close - establish that baseline first.

---

## 8. What I Would Do Differently

1. **Validate training data for instruction leakage** - automated checks rejecting any training answer containing system prompt fragments, run before any training begins
2. **Use a separate model for data generation** - generating data with the same model that will be fine-tuned, with thinking mode enabled, creates contamination risk. Use a different (typically larger) model
3. **Establish the few-shot baseline first** - fine-tune only if there is a measurable gap that prompt engineering cannot close
4. **Use a larger base model (7B+)** - at 4B parameters, LoRA fine-tuning on 2K examples shifts style while eroding topic coverage
5. **Quantise both models identically** - Q8_0 for both base and fine-tuned for a fair comparison
6. **1 epoch with lower LR (5e-5)** - minimise forgetting while still imparting style changes

---

## 9. Conclusion

The fine-tuning experiment failed, but the failure was informative:

- **Caught training data contamination** via response inspection - visible in the first few outputs
- **Confirmed the regression was semantic** using BERTScore alongside keyword coverage
- **Demonstrated that few-shot prompting outperformed LoRA fine-tuning** for this model size and task
- **Identified the quantisation comparison problem** as a confounding variable for future experiments

The core lesson: synthetic data generated by the same model you are fine-tuning, with thinking mode enabled, carries contamination risk that is invisible until you evaluate outputs systematically. Automated validation of training data is not optional - it is the first thing to build before running any fine-tuning pipeline.

---

**Source Code**: [github.com/choeyunbeom/arxiv_rag_system](https://github.com/choeyunbeom/arxiv_rag_system)

**Related Posts**:
- [arXiv RAG System: Building the System](/machine%20learning/nlp/arxiv-rag-system/)
- [arXiv RAG System: Async Refactoring](/machine%20learning/nlp/arxiv-rag-async-refactoring/)