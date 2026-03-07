---
title: "About"
permalink: /about/
author_profile: true
---

## Hi, I'm Yunbeom Choe

MSc Data Science & AI student at the University of Liverpool. I build end-to-end ML systems — from reinforcement learning agents to production-ready RAG pipelines — and write honestly about what breaks along the way.

---

## Projects

### arXiv RAG System

End-to-end Retrieval-Augmented Generation system for querying academic papers from arXiv.

- **Stack**: FastAPI, ChromaDB, Qwen3 4B (via Ollama), Streamlit, Docker
- **7-day build**: broken embedding pipeline on Day 1, systematic retrieval optimisation that hit 100% hit rate by Day 5
- Hybrid retrieval (dense + sparse) + cross-encoder reranking over 153 arXiv papers
- LoRA fine-tuning experiment — documented a 28pp regression caused by training data contamination
- Async refactoring of the entire I/O pipeline (FastAPI + httpx), fixing 7 bugs in the process
- Fully local inference on Apple M4 Pro via Ollama — no external API calls

[Part 1: Building the system](/machine%20learning/nlp/arxiv-rag-system/) · [Part 2: Async refactoring](/machine%20learning/nlp/arxiv-rag-async-refactoring/)

---

### TORCS Corkscrew RL Racing Agent

Autonomous racing agent trained on the TORCS Corkscrew track using deep reinforcement learning.

- **Stack**: Python, Stable-Baselines3, SAC, PPO
- 9.7M training steps across 4,349 episodes — 37 track completions (0.85%)
- Systematic failure mode analysis: 52.59% early crashes, 32.33% S-curve failures
- Reward shaping, hyperparameter sensitivity analysis, and catastrophic forgetting investigation

[Read the post](/reinforcement%20learning/autonomous%20driving/torcs-rl-journey/)

---

## Skills

**Languages**: Python, SQL  
**ML/AI**: PyTorch, Hugging Face, LangChain, RAG, RL (SAC, PPO), LoRA fine-tuning  
**MLOps**: FastAPI, Docker, ChromaDB, pytest
**Tools**: Git, Ollama, Streamlit

---

## Contact

- **Email**: [yunbeom.choe.dev@gmail.com](mailto:yunbeom.choe.dev@gmail.com)
- **GitHub**: [github.com/choeyunbeom](https://github.com/choeyunbeom)
- **LinkedIn**: [linkedin.com/in/yunbeom-choe-52a348370](https://www.linkedin.com/in/yunbeom-choe-52a348370)
