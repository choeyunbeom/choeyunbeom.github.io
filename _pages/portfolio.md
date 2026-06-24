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

<p>
<a href="https://github.com/choeyunbeom" class="btn btn--primary">GitHub</a>
<a href="https://www.linkedin.com/in/yunbeom-choe-52a348370" class="btn btn--info">LinkedIn</a>
<a href="mailto:yunbeom.choe.dev@gmail.com" class="btn btn--success">Email</a>
<a href="/assets/cv.pdf" class="btn btn--warning">CV (PDF)</a>
</p>

## Projects

Started with **arXiv RAG** to optimise retrieval (Hit Rate 60→100%, MRR 0.51→0.82).
Hit the limits of a single-LLM approach, so architected **FinScope** — a LangGraph
multi-agent system with a Critic Agent for hallucination checking. Below, in order
of progression:

{% include project-cards.html %}

## Competition

<div class="pcard-grid" style="grid-template-columns: 1fr;">
  <div class="pcard">
    <div class="pcard-head pcard-reply">
      <i class="fas fa-trophy"></i><span>Reply AI Challenge 2026 — Multi-Agent Fraud Detection</span>
    </div>
    <div class="pcard-body">
      <div class="pcard-metrics">
        <span class="pcard-metric">137 / 1,971 teams (Top 7%)</span>
        <span class="pcard-metric">6-hour sprint</span>
        <span class="pcard-metric">5-agent system</span>
      </div>
      <div class="pcard-desc">
        Built a multi-agent fraud-detection system in a 6-hour timed challenge:
        a LangChain orchestrator routing transactions through Isolation Forest,
        an LLM investigator, and Critic / Memory agents — with Langfuse tracing
        and a fixed LLM budget. Placed 137th of 1,971 teams.
      </div>
      <div class="pcard-stack">LangChain · Langfuse · IsolationForest · Whisper · OpenRouter</div>
      <div class="pcard-actions">
        <a href="https://github.com/choeyunbeom/reply_ai_chal" class="btn btn--primary btn--small">Code</a>
        <a href="/machine%20learning/competition/reply-mirror-ai-challenge-2026/" class="btn btn--inverse btn--small">Write-up</a>
      </div>
    </div>
  </div>
</div>
