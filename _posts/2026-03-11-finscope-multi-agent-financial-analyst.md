---
title: "From arXiv to SEC: Building a Multi-Agent Financial Report Analyst with LangGraph"
date: 2026-03-11
categories:
  - Machine Learning
  - NLP
tags:
  - RAG
  - LLM
  - LangGraph
  - Multi-Agent
  - SEC EDGAR
  - ChromaDB
  - Groq
toc: true
toc_label: "Table of Contents"
toc_icon: "cog"
toc_sticky: true
---

# From arXiv to SEC: Building a Multi-Agent Financial Report Analyst with LangGraph

**TL;DR**: Extended my arXiv RAG system to financial filings (SEC EDGAR + Companies House) in 3 days. The key upgrade: single-agent Q&A → 3-agent pipeline (Retriever → Analyzer → Critic) with parallel analysis and a hallucination check loop. 24/24 unit tests passing. Several bugs caught and fixed along the way.

---

## Abstract

This post documents a 3-day build extending an academic RAG system to financial filings. The core engineering challenge was moving from single-agent Q&A to a multi-agent pipeline where three specialised agents (Retriever, Analyzer, Critic) coordinate through LangGraph's StateGraph. The Analyzer runs risk, growth, and competitor analyses in parallel via `asyncio.gather`; the Critic checks citation coverage and triggers retries when >30% of claims are unsupported. Along the way: EDGAR API quirks, a company name resolution bug that returned a hotel REIT instead of Apple Inc., and a code review that exposed the hybrid retriever was never actually connected. The follow-up post evaluates whether the Critic's hallucination detection actually works.

**Key Contributions:**
- Multi-agent architecture with conditional retry edges in LangGraph
- Parallel analysis with concurrent Groq API calls (risk, growth, competitor)
- LLM-as-Judge Critic agent with citation-based hallucination threshold
- SEC EDGAR + Companies House dual-source ingestion pipeline
- 24/24 unit tests covering agents, ingestion, and error handling

---

## 1. Why Extend to Financial Domain?

My [arXiv RAG system](https://choeyunbeom.github.io/machine%20learning/nlp/2026/03/04/arxiv-rag-system.html) got retrieval to 100% hit rate, but the use case was narrow — academic Q&A. Financial filings are a more demanding domain:

- Documents are **dense with numbers** — revenue figures, risk disclosures, forward-looking statements
- **Hallucinations are costly** — a wrong number in a financial analysis is meaningfully worse than a vague answer about a research paper
- **Multiple angles matter** — you don't want one answer, you want risk, growth, and competitive context simultaneously

This pushed me toward a multi-agent architecture instead of a single-agent Q&A loop.

---

## 2. Architecture

<pre class="mermaid">
graph TD
    A["User Query\n(e.g. 'AAPL' or 'Apple')"] --> B["Input Resolver\nticker/name → CIK → latest 10-K"]
    B --> C["Retriever Agent\nChromaDB dense + BM25 hybrid + RRF + cross-encoder rerank"]
    C --> D["Analyzer Agent"]
    D --> D1["Risk"]
    D --> D2["Growth"]
    D --> D3["Competitor"]
    D1 & D2 & D3 --> E["Critic Agent\ncitation check → retry if >30% uncited (max 2x)"]
    E -->|"pass"| F["Final Report with source citations"]
    E -->|"fail (retry)"| C
</pre>

The key design decision: **why 3 agents instead of 1?**

A single agent doing risk + growth + competitor analysis sequentially has two problems. First, the context window gets polluted — each analysis bleeds into the next. Second, it's slow — 3 sequential LLM calls vs 3 parallel. `asyncio.gather` with `asyncio.to_thread` (since Groq's SDK is sync) cuts wall time to roughly `max(t_risk, t_growth, t_competitor)` instead of their sum.

The Critic exists because financial analysis is high-stakes. An uncited claim about "revenue growth of 15%" that doesn't appear anywhere in the retrieved chunks is exactly the kind of hallucination that looks plausible but is wrong. The Critic checks citation coverage and triggers a retry if >30% of claims are unsupported.

---

## 3. Retrieval Foundation

### SEC EDGAR Integration

EDGAR's API is well-documented but quirky. The ticker→CIK mapping requires fetching the full company tickers JSON (several MB) and doing a linear scan. The CIK then unlocks the filings endpoint:

```python
# Ticker → CIK
GET https://www.sec.gov/files/company_tickers.json

# CIK → latest 10-K metadata
GET https://data.sec.gov/submissions/CIK{cik:010d}.json

# Download filing
GET https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{filename}
```

**Bug 1: ticker regex was too loose.** `re.fullmatch(r"[A-Za-z]{1,5}", input)` matched "Apple" (5 chars, all alpha) as a ticker, then failed when "APPLE" wasn't in the ticker map. Fixed to `[A-Z]{1,5}` — uppercase only.

**Bug 2: EDGAR search API field rename.** `hits[0]["_source"]["entity_id"]` had been renamed to `ciks` (a list). Silent KeyError that only surfaced when testing company name search.

**Bug 3: HTML entities in retrieved chunks.** EDGAR's HTML filings are full of `&#160;` (non-breaking space), `&#8217;` (right single quotation mark), etc. Regex-only HTML stripping left these raw in the chunks, polluting retrieved context. Fixed with `html.unescape()` after tag removal.

### ChromaDB Deduplication

Re-ingesting the same company caused `DuplicateIDError`. The original chunk ID was `sha256(content[:100])` — too short, leading to collisions between chunks with identical openings (e.g. multiple pages starting with "APPLE INC.").

Fixed to `sha256(f"{source}:{filing_date}:{global_index}:{content[:80]}")`. The global index ensures uniqueness even for near-identical chunks.

### Smoke Test Result

```
AAPL 10-K (2025-10-31, HTML) → 575 chunks → 575/575 indexed ✓
Q: "What are Apple's main risk factors?"
→ 5 relevant chunks retrieved, Groq answered with citations ✓
```

---

## 4. Multi-Agent Graph

### LangGraph StateGraph

LangGraph's `StateGraph` is the right tool here — it manages state transitions explicitly and handles conditional retry edges cleanly:

```python
graph.add_conditional_edges(
    "critic",
    should_retry,
    {"retry": "retriever", "done": END},
)
```

`should_retry` checks `state["critique"] == "insufficient" and state["retry_count"] < 2`. Simple, but the `retry_count` guard is critical — without it, a consistently low-quality retrieval result loops forever.

### Parallel Analyzer

```python
async def analyzer_node(state: AgentState) -> dict:
    risk, growth, competitors = await asyncio.gather(
        analyze_risk(state["documents"]),
        analyze_growth(state["documents"]),
        analyze_competitors(state["documents"]),
    )
    ...
```

Each `analyze_*` function calls `asyncio.to_thread(_call_groq, prompt)`. This is concurrent threading, not true parallelism — worth being precise about in interviews. But for I/O-bound LLM API calls, it behaves like parallelism: all three HTTP requests are in-flight simultaneously.

### Critic Agent

The Critic uses LLM-as-judge with a structured output format:

```
CITED_COUNT: <n>
UNCITED_COUNT: <n>
VERDICT: sufficient | insufficient
FEEDBACK: <one sentence>
```

One design tradeoff: `_parse_verdict` defaults to `"sufficient"` on malformed responses. This is fail-open — if the LLM produces garbled output, the analysis passes through. The alternative (fail-closed, return `"insufficient"`) would cause unnecessary retries and potentially hit the max retry limit on transient formatting failures. For a prototype, fail-open is the right call; production would want structured outputs (JSON mode) to eliminate the ambiguity.

---

## 5. Production Layer

FastAPI `/analyze` endpoint, Streamlit UI, Langfuse tracing. The UI connects to FastAPI via `httpx` — agents stay server-side, UI stays thin.

Langfuse tracing is optional: if `LANGFUSE_PUBLIC_KEY` isn't set, `trace_graph()` becomes a no-op context manager. The reason this matters: during development you're running without Langfuse keys constantly, and a hard import error or missing-credential exception on every request is the kind of friction that makes you rip monitoring out entirely. No-op by default means tracing is always ready to switch on, never a blocker.

The Streamlit UI uses a 180-second `httpx` timeout. That's not arbitrary — the full pipeline (ingest if cold, embed, hybrid retrieve, 3 parallel Groq calls, Critic) can take 60-90 seconds on first run for a company not yet in ChromaDB. 30 seconds would time out on cold starts; 180 covers it with room.

### Companies House Integration

Companies House was noticeably smoother than SEC EDGAR. HTTP Basic Auth with the API key as username and empty password, then two endpoints:

```python
GET /search/companies?q={name}          # → company_number
GET /company/{number}/filing-history    # → document_id per filing
GET document-api.../document/{id}/content  # → PDF (requires Accept: application/pdf header)
```

The one gotcha: the document download endpoint requires an explicit `Accept: application/pdf` header — without it you get a metadata JSON response instead of the binary PDF. That's it. No HTML filing fallback, no entity decoding edge cases, no field renames. After the EDGAR integration, it felt almost too easy.

---

## 6. External Code Review

After shipping, I got a code review that caught several real issues:

**1. HybridRetriever was never connected to the agent pipeline.**

`hybrid_retriever.py` had the full dense + BM25 + RRF + cross-encoder implementation, but `retriever_node` was doing plain ChromaDB dense search only. README said "hybrid retrieval" — it wasn't. Fixed by calling `HybridRetriever.retrieve()` from `retriever_node`, loading the full ChromaDB collection for BM25 corpus construction.

Performance note: `_load_all_documents()` fetches the entire collection on every query to build BM25 and load the cross-encoder. For 575 chunks this is fine (~1-2s overhead). At production scale, the right fix is to cache the `HybridRetriever` instance per collection and invalidate on new ingestion.

**2. `resp.json()` called outside `with` block.**

`_ticker_to_cik` had `data = resp.json()` after the `with httpx.Client(...) as client:` block closed. httpx buffers responses so it works, but it violates the context manager's intent. Moved inside the `with` block.

**3. Test coverage gap.**

12 agent-level tests, zero ingestion tests. The ingestion layer is the most likely to break (external API changes, HTML format changes, etc.). Added 8 SEC EDGAR tests and 4 Companies House tests covering resolve, fetch, error handling, and HTML stripping.

---

## 7. Bug Fix: "apple" Returned a Hotel REIT

After the code review fixes, I ran the system end-to-end with `company: apple`. The report came back about cybersecurity threats, REIT qualification risks, and hotel management companies. Apple Inc. does not own 217 hotels.

**Root cause 1: case-sensitive ticker regex.**

`resolve_to_cik` checked if input matched `[A-Z]{1,5}` before attempting ticker lookup. Lowercase `"apple"` failed that check and fell through to company name search — which is where things went wrong.

**Root cause 2: wrong company name search API.**

`_search_company` was using EDGAR's full-text search endpoint (`/LATEST/search-index`), which ranks results by filing recency and keyword frequency — not company size or relevance. "Apple Hospitality REIT" had more recent filings and its name literally starts with "apple", so it came back first. Apple Inc. (CIK 320193) was somewhere further down the list.

The fix was to switch to EDGAR's `/browse-edgar` company search endpoint with `output=atom`. This returns companies ranked by relevance and market presence — Apple Inc. comes back first for `"apple"`, ahead of the REIT. The `resolve_to_cik` input handling was also fixed to uppercase before the ticker regex check, with a try/except fallback to name search if the ticker isn't found:

```python
def resolve_to_cik(self, user_input: str) -> str:
    stripped = user_input.strip()
    upper = stripped.upper()
    if re.fullmatch(r"[A-Z]{1,5}", upper):
        try:
            return self._ticker_to_cik(upper)
        except ValueError:
            pass
    return self._search_company(stripped)
```

Verification:
```
apple  → CIK 320193  (Apple Inc.)   ✓
AAPL   → CIK 320193  (Apple Inc.)   ✓
Tesla  → CIK 1318605 (Tesla Inc.)   ✓
```

**Root cause 3: retriever fallback leaked cross-company data.**

The retriever was designed to filter ChromaDB by `company` metadata. But if the filter returned zero results, it fell back to loading the full collection — meaning a query for "apple" with no Apple data in ChromaDB would return whatever happened to be there (in this case, Tesla and the REIT). The fallback was removed entirely. If the company filter returns nothing, the pipeline returns empty documents and the Critic marks it insufficient. The user gets a clear "no data" outcome rather than silently wrong data.

---

## 8. What's Different from arXiv RAG

| | arXiv RAG | finscope |
|---|---|---|
| Domain | Academic papers | Financial filings (10-K, annual reports) |
| Agent architecture | Single-agent | Multi-agent (Retriever → Analyzer → Critic) |
| Analysis | Single Q&A | Parallel Risk / Growth / Competitor |
| Hallucination check | None | Critic agent with citation check + retry loop |
| Data sources | arXiv API | SEC EDGAR + Companies House |
| Chunking | Default | 512-token with financial metadata |

---

## 9. Results

Tested on Apple (AAPL) 10-K filing (2025-10-31):

| Metric | Result |
|---|---|
| Filing ingested | 575 chunks from HTML 10-K |
| Retrieval (hybrid) | 8 chunks retrieved per query |
| Critic verdict (typical) | `sufficient` on second pass |
| Critic retry triggered | Confirmed in production — retry_count: 1 on Apple risk query |
| End-to-end latency | ~15s (Groq llama-3.3-70b, 3 parallel analyses) |
| Unit tests | 24/24 passing |

---

## 10. Lessons Learned

1. **Fail-open vs fail-closed is a design decision, not a default.** The Critic defaults to `"sufficient"` on malformed LLM output. For a prototype this avoids unnecessary retries, but production needs structured outputs (JSON mode) to eliminate the ambiguity entirely.

2. **The obvious API is not always the right API.** EDGAR's full-text search ranked Apple Hospitality REIT above Apple Inc. because it optimises for filing recency, not company relevance. Switching to `/browse-edgar` with `output=atom` fixed it — but only after a user-facing bug shipped.

3. **Integration tests catch what unit tests miss.** 24 unit tests passed while the hybrid retriever was never connected to the pipeline. The code review caught it. End-to-end tests against real data would have caught it sooner.

4. **Observability is not optional for multi-agent systems.** The initial Langfuse integration showed the entire pipeline as a single span. Adding `@observe` per node turned debugging from guesswork into reading a trace tree.

---

## 11. What's Next

**Update (March 11):** After publishing, a live Apple 10-K query returned `retry_count: 1` in the Langfuse trace — the Critic flagged the first-pass analysis as insufficient and sent it back to the retriever. The second pass returned `critique: "sufficient"` with feedback: "the majority of claims being directly cited." This confirms the retry loop is working in production. The likely cause: retrieved chunks were heavy on financial risk disclosures, leaving growth and competitor analyses with lower citation coverage on the first pass. The Tesla query (87.5% cited) didn't trigger a retry because the retrieved chunks covered all three analysis dimensions more evenly.

**Update (March 13):** Two follow-up fixes shipped. First, Langfuse v3 broke the tracing layer — `lf.trace()` was removed in favour of the `@observe` decorator pattern. Migrated `monitoring/langfuse_config.py` and `src/api/main.py` accordingly. Second, the original trace had no visibility into individual LangGraph node execution — the entire pipeline appeared as a single span. Added `@observe` to each node function (`retriever_node`, `analyzer_node`, `critic_node`) and each parallel sub-task (`analyze_risk`, `analyze_growth`, `analyze_competitors`). The Langfuse trace now shows the full tree:

```
financial-analysis
└── retriever-node
└── analyzer-node
    ├── analyze-risk
    ├── analyze-growth
    └── analyze-competitors
└── critic-node
```

Also added `critique_feedback` to `AgentState` — the Critic's FEEDBACK string is now persisted in state, so the first-pass rejection reason is visible in the trace output rather than being discarded.

**Update**: The Critic evaluation is now done — built a synthetic eval suite, tested 70B vs 8B as the judge model, and measured hallucination detection accuracy. Full results in the follow-up post: [LLM-as-Judge for Hallucination Detection: Does the Critic Agent Actually Work?](/machine%20learning/nlp/critic-agent-hallucination-eval/)

Code: [github.com/choeyunbeom/finscope](https://github.com/choeyunbeom/finscope)
