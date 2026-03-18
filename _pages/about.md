---
title: "About"
permalink: /about/
author_profile: true
---

## Hi, I'm Yunbeom Choe

MSc Data Science & AI student at the University of Liverpool, with a background in naval architecture and ocean engineering (Inha University). The shift from modelling physical systems to building ML pipelines was less of a leap than it sounds — both come down to systematic experimentation and honest failure analysis. I wrote more about the transition [here](/data%20science/naval-architecture-to-ai/).

---

## Projects

### DefectVision - Manufacturing Defect Detector

Real-time manufacturing defect detection using unsupervised anomaly detection — trained on normal images only, no labeled defects required.

- **Stack**: Anomalib, PatchCore, PyTorch, OpenVINO, FastAPI, Streamlit, OpenCV, Docker
- 100% Image AUROC on MVTec AD bottle; 8–20pp drop on the harder MVTec AD 2 benchmark
- Lighting augmentation experiment: discovered augmentation hurts memory-bank methods by widening the normal distribution in feature space
- FastAPI inference API with `/calibrate` endpoint for on-site threshold tuning
- Real-time webcam streaming with queue-based non-blocking inference pipeline
- 14 tests, CI/CD with GitHub Actions, Docker deployment

[Read the post](/machine%20learning/computer%20vision/defectvision-anomaly-detection/)

---

### FinScope - Multi-Agent Financial Report Analyst

Multi-agent RAG system that analyses SEC EDGAR and Companies House filings through a 3-agent pipeline.

- **Stack**: LangGraph, Groq (llama-3.3-70b), ChromaDB, FastAPI, Streamlit, Docker
- Retriever → Analyzer → Critic pipeline with conditional retry loop
- Parallel Risk / Growth / Competitor analysis via `asyncio.gather`
- Hybrid retrieval (dense + BM25 + RRF + cross-encoder rerank)
- Critic agent: LLM-as-judge hallucination check with fail-open design and retry guard
- Extended from arXiv RAG - same retrieval core, new multi-agent orchestration layer

[Part 1: Building the System](/machine%20learning/nlp/finscope-multi-agent-financial-analyst/) · [Part 2: Critic Eval](/machine%20learning/nlp/critic-agent-hallucination-eval/)

---

### arXiv RAG System

End-to-end Retrieval-Augmented Generation system for querying academic papers from arXiv.

- **Stack**: FastAPI, ChromaDB, Qwen3 4B (via Ollama), Streamlit, Docker
- **7-day build**: broken embedding pipeline on Day 1, systematic retrieval optimisation that hit 100% hit rate by Day 5
- Hybrid retrieval (dense + sparse) + cross-encoder reranking over 153 arXiv papers
- LoRA fine-tuning experiment — documented a 28pp regression caused by training data contamination
- Async refactoring of the entire I/O pipeline (FastAPI + httpx), fixing 7 bugs in the process
- Fully local inference on Apple M4 Pro via Ollama — no external API calls

[Part 1: Building the system](/machine%20learning/nlp/arxiv-rag-system/) · [Part 2: Async refactoring](/machine%20learning/nlp/arxiv-rag-async-refactoring/) · [Part 3: LoRA Fine-Tuning](/machine%20learning/nlp/lora-finetuning-experiment/)

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
**ML/AI**: PyTorch, Hugging Face, LangChain, LangGraph, RAG, Multi-Agent Systems, RL (SAC, PPO), LoRA fine-tuning, Anomaly Detection (PatchCore), OpenVINO
**MLOps**: FastAPI, Docker, ChromaDB, pytest, GitHub Actions
**Tools**: Git, Ollama, Streamlit

---

## Contact

- **Email**: [yunbeom.choe.dev@gmail.com](mailto:yunbeom.choe.dev@gmail.com)
- **GitHub**: [github.com/choeyunbeom](https://github.com/choeyunbeom)
- **LinkedIn**: [linkedin.com/in/yunbeom-choe-52a348370](https://www.linkedin.com/in/yunbeom-choe-52a348370)
