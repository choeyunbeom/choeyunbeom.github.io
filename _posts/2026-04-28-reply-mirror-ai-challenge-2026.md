---
title: "Reply Mirror AI Challenge 2026: Multi-Agent Fraud Detection Under a 6-Hour Clock"
date: 2026-04-28
categories:
  - Machine Learning
  - Competition
tags:
  - Fraud Detection
  - Multi-Agent
  - Anomaly Detection
  - IsolationForest
  - LangChain
  - Langfuse
  - Whisper
  - OpenRouter
toc: true
toc_label: "Table of Contents"
toc_icon: "cog"
toc_sticky: true
---

# Reply Mirror AI Challenge 2026: Multi-Agent Fraud Detection Under a 6-Hour Clock

**TL;DR**: Built a multi-agent fraud detection system for the Reply Mirror AI Challenge 2026 — 6 hours, 5 levels, one shot at each submission. The system combines an Isolation Forest scorer with an LLM-based Investigator (GPT-4o via OpenRouter), coordinated by an Orchestrator that routes only ambiguous transactions through the expensive LLM path. Finished **134th out of 1,971 teams (top 6.8%)**. The dominant lesson: gray zone routing is everything — on the large-scale levels, Scorer-only ran in 0.3 seconds while the LLM path took 40+ minutes.

---

## Abstract

This post documents a 6-hour competition build: a multi-agent system for unsupervised fraud detection in financial transaction data. The challenge had no fraud labels in training data — the only signal was anomaly. Across 5 levels (each a different fictional-world dataset with escalating scale), the system had to submit one fraud ID list per level within the total time budget. The core architecture: Isolation Forest for anomaly scoring, a gray-zone router that sends borderline cases to an LLM Investigator, and a Critic agent (designed for L4–L5) that audits the Investigator's reasoning. Level 3 added audio fraud signals via STT (Whisper). Levels 4–5 had 7,400 and 14,672 transactions respectively — which forced the gray zone to collapse to zero, making the LLM path irrelevant on the hardest levels.

**Key Contributions:**
- Dynamic gray-zone routing whose optimal width is **scale-dependent**, not just score-distribution-dependent
- Unsupervised fraud detection: IsolationForest trained on normal-only data, with eval-set statistics recomputed at runtime
- Langfuse v3 tracing pattern via LangChain CallbackHandler — the only one that worked reliably after the v2→v3 API break

---

## 1. Challenge Overview

- **Competition**: Reply Mirror AI Challenge 2026
- **Date**: April 16, 2026, 6 hours
- **Team**: yunbeom-choe-3845
- **Goal**: Detect fraudulent transactions in financial data — unsupervised, no fraud labels in training set
- **Levels**: 5 levels with increasing difficulty, each based on a fictional-world theme (The Truman Show, Brave New World, Deus Ex, Blade Runner, 1984)
- **Budget**: Per-level LLM API cost limits ($8–$60)
- **Submission**: Langfuse session ID + list of fraud transaction IDs (`.txt`) per level — only the first submission per level counted

The key constraint was **one shot per level**. There was no iterative tuning on a public leaderboard — you submitted once and moved on. This made robust engineering (validation before submission, cost guards) more valuable than aggressive model tuning.

---

## 2. Architecture

<pre class="mermaid">
flowchart TD
    TX([Transaction stream]):::gray --> ORC

    ORC["**Orchestrator agent**\nWider gray zone on L4-L5"]:::purple

    ORC --> SCO
    ORC --> CTX
    ORC --> INV

    SCO["**Scorer agent**\nIsolationForest + drift features"]:::green
    CTX["**Context agent**\nMemory-enriched evidence"]:::green
    INV["**Investigator agent**\nGPT-4o"]:::brown

    INV --> CRI["**Critic agent**\nChallenges the verdict"]:::maroon

    SCO --> DF
    CTX --> DF
    INV --> DF
    CRI --> DF

    DF["**Decision fusion**\nInvestigator + Critic consensus"]:::purple

    DF --> OUT([Fraud IDs output]):::gray

    OUT --> MEM["**Memory agent**\nL1-L3 patterns carried forward"]:::gold
    MEM -.->|known fraud merchants| SCO

    classDef gray   fill:#4a4a4a,color:#fff,stroke:none
    classDef purple fill:#4a3f8f,color:#fff,stroke:none
    classDef green  fill:#1a6b5a,color:#fff,stroke:none
    classDef brown  fill:#7a3b2e,color:#fff,stroke:none
    classDef maroon fill:#6b2d4a,color:#fff,stroke:none
    classDef gold   fill:#8a6a1a,color:#fff,stroke:none
</pre>

All levels used GPT-4o via OpenRouter as the LLM backend.

### Role Split

The team divided into three roles with agreed interface contracts:

| Role | Components |
|------|-----------|
| Yunbeom Choe | Orchestrator, Decision Fusion, CostTracker, Validator |
| Katalin Pazmany | Scorer, Context |
| Alex Yeung | Investigator, Memory, Critic, STT |

Interface signatures were a team contract — any change had to be flagged to all roles before committing. In a 6-hour window, silent interface breakage is a competition-ender.

### Routing Logic

```
score = scorer.predict(tx)

if score < gray_low:    → legit  (no LLM call)
elif score >= gray_high: → fraud  (no LLM call)
else:                   → build context → Investigator judges
                          → (L4-5) Critic verifies
```

The routing condition is the single most important parameter in the system. More on this in §5.

---

## 3. Core Components

### Scorer (Isolation Forest)

The Scorer is the foundation. With no fraud labels in training data, the only option was unsupervised anomaly detection — Isolation Forest via scikit-learn.

Features used:
- Amount z-score (relative to user's own history)
- `is_night` (transaction outside normal hours)
- `is_new_merchant` (merchant not seen in training)
- Payment method risk score
- Transaction type risk score
- Balance ratio
- Days since last transaction
- `is_known_fraud_merchant` (injected from Memory agent)

The model is retrained from scratch at each level. No transfer from prior levels — the fictional-world themes mean user populations shift completely.

### Context Agent

Builds a per-transaction evidence bundle: user profile, GPS location match, SMS/email phishing signals. Level 3 added audio signals from STT-transcribed call recordings (Whisper tiny model). The bundle is consumed by the Investigator prompt.

### Investigator (LLM Judge)

The Investigator generates competing fraud hypotheses (H0 = legitimate, H1–H4 = fraud types), scores them via Bayesian update against the Context evidence bundle, and returns the highest-posterior verdict. Falls back to rule-based when the cost tracker blocks LLM calls.

### Critic Agent

A Critic agent was designed to audit the Investigator's verdict on L4–L5 — checking logical validity, evidential sufficiency, and ignored high-diagnostic signals. It never ran in production. See §5 for why.

### Memory Agent

Two components carried real weight in the system:
- **FraudMerchantTracker**: set of merchants from confirmed fraud, O(1) lookup, feeds back into Scorer as the `is_known_fraud_merchant` feature
- **DriftMonitor**: rolling Wasserstein distance on hour distribution + log-amount median shift + new location fraction → single `drift_score`

(An AccountGraph and a HypothesisGenerator were also implemented but their contribution to final scoring was not isolated.)

---

## 4. Dataset by Level

| Level | Train | Validation | Notable |
|-------|-------|------------|---------|
| 1 | The Truman Show - train | The Truman Show - validation | Baseline |
| 2 | Brave New World - train | Brave New World - validation | Baseline |
| 3 | Deus Ex - train | Deus Ex - validation | MP3 audio files → STT |
| 4 | Blade Runner - train | Blade Runner - validation | ~7,400 transactions |
| 5 | 1984 - train | 1984 - validation | ~14,672 transactions |

Levels 4 and 5 are where the architecture had to adapt. The LLM path that worked cleanly on L1–L3 simply doesn't scale to 14,672 transactions within a 6-hour window.

---

## 5. Troubleshooting Log

### Bug 1: Train/Eval User Distribution Mismatch → 100% Flagged as Fraud

**Problem**: Fitting IsolationForest on training data user statistics, then running it on validation data with a completely different user population, caused every transaction to score as an anomaly. Fraud IDs: 56 (nearly 100% of the validation set).

**Root cause**: User-level statistics (mean amount, standard deviation, transaction count, etc.) were initialized from the training set distribution. On validation data, every user looked like a statistical outlier by that baseline.

**Fix**: After loading the validation data, recompute user statistics against the eval population before scoring:

```python
# After context_agent.reload()
scorer._stats.build(eval_df)
scorer._home_cities = build_home_cities(eval_df, eval_users)
```

This is a subtle but critical point: in a competition with domain shift between train and eval, you cannot use training-set statistics as your scoring baseline. You need to build the "normal" baseline from whichever population you're actually scoring.

---

### Bug 2: Langfuse Session ID Not Registering

**Problem**: Using `langfuse.openai.OpenAI` wrapper, `session_id` was not appearing in the Langfuse platform — traces existed but were unlinked.

**Attempted approaches that failed**:
- `langfuse.openai` wrapper → failed
- `langfuse.trace()` → API changed in v3, no longer works this way
- `TraceContext` → failed

**Fix**: Official tutorial pattern — `langfuse.langchain.CallbackHandler` + `langchain_openai.ChatOpenAI`:

```python
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
from langchain_openai import ChatOpenAI

session_id = f"{team_name}-{ulid.new().str}"
llm_client = ChatOpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    model="gpt-4o",
    temperature=0.2,
)
llm_client._langfuse_handler = LangfuseCallbackHandler()
llm_client._langfuse_session_id = session_id
```

The lesson: when a library's API changes between major versions (v2 → v3), go straight to the official tutorial rather than adapting patterns from Stack Overflow or prior code.

---

### Bug 3: Gray Zone Boundary Condition → Missed LLM Calls

**Problem**: With `gray_low == gray_high == 0.55`, transactions with `score == 0.55` fell into the gray zone and triggered LLM calls unexpectedly.

**Root cause**: The fraud fast-path condition was `score > gray_high` (strict greater-than). A score exactly equal to `gray_high` fell through to the gray zone.

**Fix**: Change to `>=`:

```python
# Before
if score > self.gray_high:
    return {"fraud": True, ...}

# After
if score >= self.gray_high:
    return {"fraud": True, ...}
```

Small bug, but in a competition where every LLM call costs money and time, this could drain budget on false gray-zone cases.

---

### Bug 4: Large-Scale Levels → LLM Path Too Slow

**Problem**: L4 (7,400 transactions) and L5 (14,672 transactions) with gray zone routing + LLM calls for borderline cases was taking 40+ minutes — incompatible with a 6-hour total competition window covering 5 levels.

| Approach | L5 runtime |
|----------|-----------|
| Gray zone + LLM | 40+ minutes |
| Scorer only | ~0.3 seconds |

**Fix**: Collapse the gray zone entirely by setting `gray_low = gray_high = 0.55`. With this setting, all transactions are classified by the Scorer alone — anything at or above 0.55 is fraud, below is legit. Zero LLM calls. This sacrifices the Investigator's ability to handle borderline cases, but on a dataset of 14,672 transactions within a 6-hour budget, there is no other viable option.

The broader insight: gray-zone routing is a cost-accuracy tradeoff that is **scale-dependent**. It's the right call on L1–L3 (hundreds of transactions), wrong on L4–L5. The Critic agent designed to audit borderline LLM verdicts becomes irrelevant the moment you decide there are no borderline LLM verdicts.

---

### Bug 5: Phishing Boost Backfire

**Problem**: The Orchestrator detected phishing signals and artificially inflated the anomaly score to push transactions into the gray zone — ensuring they'd get LLM review. But this caused legitimate transactions with phishing context (users who received phishing attempts but didn't fall for them) to be routed through the Investigator, which then classified them as fraud.

**Fix**: Remove the phishing boost block entirely. Phishing signals are still available to the Investigator as part of the Context evidence bundle — they inform the reasoning without mechanically pushing scores upward.

The lesson generalizes: when you have an LLM judge and a mechanical heuristic, don't let the heuristic preemptively bias what the judge sees. Pass evidence, not pre-baked conclusions.

---

## 6. Final Configuration

After all fixes, the level configuration converged to:

```python
LEVEL_CONFIG = {
    1: {"gray_low": 0.55, "gray_high": 0.55, "critic": False, "llm_model": "gpt-4o"},
    2: {"gray_low": 0.55, "gray_high": 0.55, "critic": False, "llm_model": "gpt-4o"},
    3: {"gray_low": 0.55, "gray_high": 0.55, "critic": False, "llm_model": "gpt-4o", "audio_folder": "audio"},
    4: {"gray_low": 0.55, "gray_high": 0.55, "critic": False, "llm_model": "gpt-4o"},
    5: {"gray_low": 0.55, "gray_high": 0.55, "critic": False, "llm_model": "gpt-4o"},
}
```

This is the punchline. The Critic ended up unused across all levels. On L4–L5 the LLM path was off entirely. On L1–L3 the collapsed gray zone meant nothing reached the Investigator, so nothing reached the Critic either.

---

## 7. Results

**Final ranking: 134th / 1,971 teams — top 6.8%**

Level 3 was run in parallel on a teammate's machine to handle the STT audio processing while other levels ran concurrently. Final L5 submission flagged 938 transactions out of ~14,672.

---

## 8. Lessons Learned

**1. Train/Eval domain shift requires re-anchoring your baseline.**
If users in the eval set have a completely different statistical profile than training users, your "normality" baseline is wrong. Always rebuild per-user statistics from the population you're actually scoring, not the population you trained on.

**2. LLM cost vs. accuracy tradeoff is scale-dependent.**
Gray-zone LLM routing improves accuracy on small datasets. On datasets with tens of thousands of transactions, it's economically and temporally infeasible within a fixed competition window. The same architecture parameter (`gray_low`, `gray_high`) needs to be tuned per-level based on dataset size, not just anomaly score distribution.

**3. Follow the official tutorial for tracing libraries.**
Langfuse changed its core API between v2 and v3. Custom wrappers and older Stack Overflow patterns didn't work. The official tutorial approach (LangChain CallbackHandler) worked on the first try. When a library is evolving quickly, the official documentation is more reliable than community patterns.

**4. One-shot submission changes the engineering calculus.**
With only one submission per level and no public leaderboard feedback, confidence in correctness matters more than marginal accuracy improvements. The Submission Validator (checking for empty submissions, all-fraud flags, invalid IDs) was not optional infrastructure — it was the safety net that prevented submitting a corrupted result under time pressure.

---

## Appendix: Critical Safeguards

Two modules were marked "do not remove or weaken" in the codebase:

**Cost Tracker** (`utils/cost_tracker.py`): Every LLM call goes through this. At 90% of the level budget, the Orchestrator narrows the gray zone by ±0.10. At 100%, LLM calls are blocked and gray-zone cases fall back to the Scorer threshold. Without this, a gray-zone spike could exhaust the budget mid-level.

**Submission Validator** (`utils/validator.py`): Called before writing any `.txt` output. Rejects empty submissions, all-transactions-flagged outputs, IDs not present in the evaluation set, and duplicate IDs. In a competition where the first submission is the only submission, a corrupted output file is equivalent to not submitting.

Code: [github.com/choeyunbeom/reply_ai_chal](https://github.com/choeyunbeom/reply_ai_chal)