---
title: "arXiv RAG System: Async Refactoring and Bug Fixes"
date: 2026-03-06
categories:
  - Machine Learning
  - NLP
tags:
  - RAG
  - FastAPI
  - async
  - httpx
  - pytest
  - refactoring
toc: true
toc_label: "Table of Contents"
toc_icon: "cog"
toc_sticky: true
---

> This is a follow-up to [arXiv RAG System: Engineering an Academic Paper Q&A System from Scratch](https://choeyunbeom.github.io/machine%20learning/nlp/arxiv-rag-system/). The system was functionally complete after 7 days, but a code review revealed the entire I/O pipeline was synchronous. This post documents the async refactoring and 7 bug fixes that followed.

---

## 1. The Problem: A Blocking Pipeline Inside an Async Framework

FastAPI is built on Starlette and runs on an async event loop (uvicorn). When a request comes in, FastAPI expects the handler to `await` I/O - yielding control back to the event loop while waiting for network responses. If the handler blocks instead, the entire event loop stalls.

The original codebase had this structure:

```python
# routers/query.py - original
def query(req: Request, body: QueryRequest):
    rag_chain = req.app.state.rag_chain
    result = rag_chain.query(body.question, top_k=body.top_k)
    return result
```

And inside `rag_chain.py`:

```python
# rag_chain.py - original
def query(self, question: str, top_k: int = 5):
    chunks, _ = self.retriever.search(question, top_k=top_k)  # blocking HTTP call
    answer = self.llm.generate(prompt=..., system=...)         # blocking HTTP call
    return RAGResponse(...)
```

Both `retriever.search()` and `llm.generate()` made synchronous `httpx.Client` calls to Ollama. Under concurrent load, each request would block a thread for 15-20 seconds (the LLM generation time), making the system effectively single-threaded.

---

## 2. The Refactoring

The fix required changes at every layer of the stack.

### 2.1 HTTP Clients

Both `HybridRetriever` and `LLMClient` used `httpx.Client`. Replaced with `httpx.AsyncClient`:

```python
# Before
self._http_client = httpx.Client(timeout=60.0)

# After
self._http_client = httpx.AsyncClient(timeout=60.0)
```

### 2.2 Embedding and Generation

```python
# hybrid_retriever.py - before
def _embed_query(self, text: str) -> list[float]:
    response = self._http_client.post(url, json=payload)
    return response.json()["embedding"]

# hybrid_retriever.py - after
async def _embed_query(self, text: str) -> list[float]:
    response = await self._http_client.post(url, json=payload)
    return response.json()["embedding"]
```

Same pattern applied to `search()` in `HybridRetriever` and `generate()` in `LLMClient`.

### 2.3 RAG Chain and Router

```python
# rag_chain.py - after
async def query(self, question: str, top_k: int = 5, include_vis: bool = False):
    chunks, embeddings = await self.retriever.search(question, top_k=top_k)
    answer = await self.llm.generate(prompt=..., system=...)
    return RAGResponse(...)

# routers/query.py - after
async def query(req: Request, body: QueryRequest):
    result = await req.app.state.rag_chain.query(
        question=body.question, top_k=body.top_k, include_vis=body.include_vis
    )
    return result
```

### 2.4 Evaluation Pipeline

The evaluation script drives the full RAG pipeline, so it also needed async:

```python
# evaluation/evaluate.py - after
async def evaluate_retrieval(retriever, dataset):
    ...

async def run_evaluation(system_prompt=None, query_template=None):
    ...

if __name__ == "__main__":
    asyncio.run(run_evaluation())
```

### 2.5 HTTP Client Lifecycle

`httpx.AsyncClient` holds open connections that need to be closed. Added `__del__()` to both classes as a safety net:

```python
def __del__(self):
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(self._http_client.aclose())
        else:
            loop.run_until_complete(self._http_client.aclose())
    except Exception:
        pass
```

---

## 3. Bug Fixes

The async refactoring triggered a full code review that surfaced 7 bugs.

### Bug 1: UMAP Dead Code

`rag_chain.py` had the UMAP projection block written twice - copy-paste from an earlier iteration. The second block silently overwrote the first:

```python
# rag_chain.py - two identical blocks
if include_vis and embeddings:
    points = self._project_umap(embeddings, query_embedding)  # result discarded
    ...

if include_vis and embeddings:
    points = self._project_umap(embeddings, query_embedding)  # this one used
    rag_response.vis_data = VisData(points=points)
```

Removed the first block. The second was the live one.

### Bug 2: Tuple Unpacking in evaluate.py

`retriever.search()` returns `(chunks, embeddings)`. The evaluation script was using the return value as if it were a plain list:

```python
# Before - TypeError at runtime
results = retriever.search(question, top_k=top_k)
for chunk in results:  # iterating over a tuple, not a list of chunks
    ...

# After
results, _ = await retriever.search(question, top_k=top_k)
for chunk in results:
    ...
```

This bug would have caused `evaluate.py` to fail silently whenever it ran.

### Bug 3: Chunker Condition Logic

```python
# Before - merges two distinct cases
if not sections or "full_text" in sections:
    full = paper.get("full_text", "") or sections.get("full_text", "")
    sections = {"full_text": strip_references_from_text(full)}
```

The `or` condition caused incorrect behaviour when `sections` was a non-empty dict that happened to include a `"full_text"` key alongside other section keys. All the other sections would be discarded.

```python
# After - explicit branches
if not sections:
    full = paper.get("full_text", "")
    sections = {"full_text": strip_references_from_text(full)}
elif "full_text" in sections and len(sections) == 1:
    sections = {"full_text": strip_references_from_text(sections["full_text"])}
# else: sections has real section keys - use them as-is
```

### Bug 4: MD5 on FIPS Systems

```python
# Before
return hashlib.md5(key.encode()).hexdigest()[:12]

# After
return hashlib.sha256(key.encode()).hexdigest()[:12]
```

`hashlib.md5` raises `ValueError: [digital envelope routines] unsupported` on FIPS-compliant systems (certain Linux distributions in enterprise/government environments). `sha256` works everywhere and produces the same 12-char hex interface.

### Bug 5: `/no_think` Duplication

Qwen3's thinking mode is suppressed by prefixing `/no_think` to the system prompt. The original code also injected it into the user prompt, which was redundant and polluted the actual query:

```python
# Before
payload = {
    "prompt": "/no_think\n\n" + prompt,   # unnecessary
    ...
}
if system:
    payload["system"] = "/no_think\n\n" + system  # correct placement

# After
payload = {
    "prompt": prompt,  # clean
    ...
}
if system:
    payload["system"] = "/no_think\n\n" + system
```

### Bug 6: Import Ordering (E402)

```python
# Before - logger declared between imports, triggering ruff E402
import logging
logger = logging.getLogger(__name__)  # ← here
import json
import pathlib
from src.api.core.config import settings

# After - all imports first, then module-level declarations
import json
import logging
import pathlib
from src.api.core.config import settings

logger = logging.getLogger(__name__)
```

---

## 4. Test Fixes

The async refactoring required corresponding changes across all test files. The core pattern: replace `MagicMock` with `AsyncMock` for any method that is now `async`, and decorate test functions with `@pytest.mark.asyncio`.

### LLMClient Tests

<details>
<summary>Before/after diff (click to expand)</summary>

```python
# Before
@pytest.fixture
def client():
    with patch("src.api.core.llm_client.httpx.Client") as MockClient:
        mock_http = MagicMock()
        MockClient.return_value = mock_http
        llm = LLMClient(model="qwen3:4b")
        llm._mock_http = mock_http
        yield llm

def test_default_temperature(self, client):
    client._mock_http.post = MagicMock(return_value=_make_mock_response())
    client.generate(prompt="test")
    ...

# After
@pytest.fixture
def client():
    with patch("src.api.core.llm_client.httpx.AsyncClient") as MockClient:
        mock_http = MagicMock()
        MockClient.return_value = mock_http
        llm = LLMClient(model="qwen3:4b")
        llm._mock_http = mock_http
        yield llm

@pytest.mark.asyncio
async def test_default_temperature(self, client):
    client._mock_http.post = AsyncMock(return_value=_make_mock_response())
    await client.generate(prompt="test")
    ...
```

</details>

### RAG Chain Tests

```python
# Before
def test_returns_rag_response(self, mock_rag_chain, sample_chunks):
    mock_rag_chain._mock_retriever.search = MagicMock(return_value=(sample_chunks, None))
    mock_rag_chain._mock_llm.generate = MagicMock(return_value="RAG is a technique.")
    response = mock_rag_chain.query("What is RAG?")
    ...

# After
@pytest.mark.asyncio
async def test_returns_rag_response(self, mock_rag_chain, sample_chunks):
    mock_rag_chain._mock_retriever.search = AsyncMock(return_value=(sample_chunks, None))
    mock_rag_chain._mock_llm.generate = AsyncMock(return_value="RAG is a technique.")
    response = await mock_rag_chain.query("What is RAG?")
    ...
```

### Integration Tests

```python
# Before
chain.query = MagicMock(return_value=rag_response)

# After
chain.query = AsyncMock(return_value=rag_response)
```

**Final result: 104 tests, 0 failures.**

---

## 5. What Changed and Why It Matters

| Change | Before | After |
|--------|--------|-------|
| HTTP client | `httpx.Client` (blocking) | `httpx.AsyncClient` (non-blocking) |
| Query handler | `def query()` | `async def query()` |
| LLM call | `response = client.post(...)` | `response = await client.post(...)` |
| Embedding call | `response = client.post(...)` | `response = await client.post(...)` |
| Concurrent requests | Threads blocked for 15-20s each | Event loop yields during I/O |

For a single-user demo the difference is invisible - requests are sequential anyway. But for any real serving scenario (multiple users, automated evaluation runs, load tests), the synchronous version would saturate the thread pool quickly. The async version allows the event loop to handle other requests while waiting on Ollama.

The bug fixes - particularly the tuple unpacking error in `evaluate.py` and the chunker condition logic - would have caused silent failures or incorrect results in production evaluation runs.

---

## 6. Takeaways

**1. Async all the way down.** Using `httpx.AsyncClient` in a sync function defeats the purpose. The async keyword needs to propagate from the I/O call all the way to the FastAPI route handler.

**2. `MagicMock` won't catch async bugs.** Tests that mock an async method with `MagicMock` instead of `AsyncMock` pass even when the production code is wrong - the mock returns a coroutine object rather than raising a `TypeError`. Always use `AsyncMock` for `async def` methods.

**3. Boolean conditions that merge distinct cases cause subtle bugs.** `if not sections or "full_text" in sections` looks concise but collapses two different states into one code path. When the two cases have different semantics, write them as separate branches.

**4. Read the ruff output.** E402 (module-level import ordering) is a lint warning, but the underlying pattern - declaring module state between imports - indicates code that grew without discipline. Fixing it made the structure clearer.

---

## Code

The full implementation is available at [github.com/choeyunbeom/arxiv_rag_system](https://github.com/choeyunbeom/arxiv_rag_system).
