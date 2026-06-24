---
title: "Portfolio"
permalink: /portfolio/
layout: single
author_profile: true
classes: wide
---

**AI/ML Engineer** building production-minded systems — RAG, multi-agent LLM
orchestration, computer-vision anomaly detection, and reinforcement learning.
MSc Data Science & AI, University of Liverpool.

**🏆 Reply AI Challenge 2026 — 137 / 1,971 (Top 7%)**

<a href="https://github.com/choeyunbeom" class="btn btn--primary">GitHub</a>
<a href="https://www.linkedin.com/in/yunbeom-choe-52a348370" class="btn btn--info">LinkedIn</a>
<a href="mailto:yunbeom.choe.dev@gmail.com" class="btn btn--success">Email</a>
<a href="/assets/cv.pdf" class="btn btn--warning">CV (PDF)</a>

---

## arXiv RAG System

End-to-end Retrieval-Augmented Generation system for querying academic papers,
built and optimised from scratch.

- **Hit Rate 60 → 100%, MRR 0.51 → 0.82** through systematic retrieval optimisation
- FastAPI `lifespan` startup + fully async I/O pipeline (FastAPI + httpx)
- Traced a **28pp** fine-tuning regression to training-data contamination
- **Stack**: FastAPI · ChromaDB · LLM (Qwen3 4B / Ollama) · Docker

<a href="https://github.com/choeyunbeom/arxiv_rag_system" class="btn btn--primary">Code</a>
<a href="/machine%20learning/nlp/arxiv-rag-system/" class="btn btn--inverse">Write-up</a>

> Started with arXiv RAG to optimise retrieval (Hit Rate 60→100%, MRR 0.51→0.82).
> Hit the limits of a single-LLM approach, so architected FinScope — a LangGraph
> multi-agent system with a Critic Agent for hallucination checking.

---

## FinScope — Multi-Agent Financial Analyst

Multi-agent RAG system that analyses SEC EDGAR and Companies House filings.

- **LangGraph** orchestration: Retriever → Analyzer → Critic pipeline
- Parallel Risk / Growth / Competitor analysis via `asyncio.gather` (~15s end-to-end)
- **Critic Agent**: LLM-as-judge hallucination check with a conditional retry loop
- Langfuse monitoring across the full agent graph
- **Stack**: LangGraph · Groq (llama-3.3-70b) · ChromaDB · FastAPI · Langfuse · Docker

<a href="https://github.com/choeyunbeom/finscope" class="btn btn--primary">Code</a>
<a href="/machine%20learning/nlp/finscope-multi-agent-financial-analyst/" class="btn btn--inverse">Write-up</a>

---

## DefectVision — Manufacturing Defect Detector

Real-time manufacturing defect detection using unsupervised anomaly detection.

- **PatchCore**, trained on normal images only — no labelled defects required
- FastAPI inference API + Streamlit dashboard with live webcam streaming
- 100% Image AUROC on MVTec AD bottle; honest failure analysis on harder benchmarks
- **Stack**: Anomalib · PatchCore · PyTorch · OpenVINO · FastAPI · Streamlit · Docker

<a href="https://github.com/choeyunbeom/defectvision" class="btn btn--primary">Code</a>
<a href="/machine%20learning/computer%20vision/defectvision-anomaly-detection/" class="btn btn--inverse">Write-up</a>

---

## TORCS RL Racing Agent

Autonomous racing agent trained on the TORCS Corkscrew track with deep RL.

- **SAC, 9.7M training steps** with reward shaping and curriculum learning
- Investigated catastrophic forgetting and perverse reward incentives
- Self-implemented SAC (auto-entropy, twin-Q) in the second iteration
- **Stack**: Python · PyTorch · SAC · PPO

<a href="https://github.com/choeyunbeom/ibm_ai_race" class="btn btn--primary">Code</a>
<a href="/reinforcement%20learning/autonomous%20driving/torcs-rl-journey/" class="btn btn--inverse">Write-up</a>
