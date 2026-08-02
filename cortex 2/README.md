# Cortex

Secure, multimodal RAG agent with an optimized small-language-model (SLM) core,
fine-tuned for application security Q&A. Built to run entirely on CPU/local
compute — no GPU or cloud dependency required for inference.

## Why this exists

Most RAG demos wire together LangChain + OpenAI and call it done. This project
instead demonstrates the full lifecycle a Forward Deployed Engineer actually
owns in production:

- **Ingestion**: multimodal parsing (PDF, DOCX, images, audio) with PII/secret
  scrubbing *before* anything touches storage.
- **Retrieval**: hybrid dense + keyword search over a local vector store.
- **Agentic orchestration**: an explicit, hand-rolled state machine (not a
  black-box framework) that routes queries between a local SLM and escalates
  to a larger model only when needed — a real cost/latency tradeoff, not a toy.
- **SLM optimization**: a quantized 1.5B–3B model, LoRA fine-tuned on an
  appsec-domain dataset, benchmarked against its own base and a teacher model.
- **Security & privacy**: prompt-injection detection, output validation against
  OWASP LLM Top 10 categories, and an immutable audit log — extending the
  scanner logic from Sentry API and the red-team payloads from Warlock into a
  live defensive layer instead of a standalone tool.

## Architecture

```
raw files -> ingestion (parse + scrub PII) -> chunk + embed -> Chroma (local)
                                                                    |
user query -> security.input_guard -> agent.router -----------------
                                          |            \
                                    local SLM        escalate to API model
                                          |            /
                                    agent.critic <-----
                                          |
                                security.output_guard -> audit log -> response
```

## Repo layout

| Dir | Purpose |
|---|---|
| `ingestion/` | multimodal parsers + PII/secret scrubbing |
| `retrieval/` | embedding, chunking, Chroma vector store, hybrid search |
| `agent/` | router, tool-calling, critic — the orchestration state machine |
| `slm/` | model serving (llama.cpp/GGUF), LoRA fine-tuning scripts, benchmarks |
| `security/` | input/output guards, injection detection, audit logging |
| `api/` | FastAPI app wiring everything together |
| `data/` | raw docs, processed chunks, vector store persistence (gitignored) |
| `scripts/` | one-off setup/eval scripts |
| `notebooks/` | benchmark + fine-tune exploration |

## Status

Early scaffold — ingestion + PII scrubbing + vector store + FastAPI skeleton
are functional. Agent orchestration, SLM fine-tuning, and security guards are
stubbed with clear interfaces. See TODOs at the bottom of each module and the
project-level TODO list from Claude below.

### Retrieval tuning & reranking (new)

`retrieval/hybrid.py`'s RRF fusion is now fully parameterized (`rrf_k`,
`dense_weight`, `sparse_weight`, `over_fetch`) instead of hardcoded, and
supports an optional second-pass cross-encoder reranker over the fused
candidate pool.

- `scripts/tune_rrf.py` — eyeball A/B across a handful of (k, weight) configs
  against a small labeled query set.
- `scripts/rrf_sensitivity.py` — quantitative sweep: hit_rate, MRR, and a
  **rank-overlap sensitivity score** (Jaccard overlap between adjacent
  configs in the k-sweep — low overlap means results reshuffle a lot for
  small parameter changes, i.e. the system is *sensitive* in that region).
  Run with `--with-reranker` to see whether cross-encoder reranking flattens
  that sensitivity (a reranker that fixes bad RRF ordering shows higher
  hit_rate/MRR and more stability across the k-sweep — concrete evidence for
  whether the extra reranking cost is worth it on your corpus).
- `retrieval/reranker.py` — `CrossEncoderReranker`, using
  `cross-encoder/ms-marco-MiniLM-L-6-v2` (CPU-friendly, no new dependency —
  ships inside `sentence-transformers`, already in requirements.txt). Wired
  into `HybridRetriever` as an optional constructor arg; opt-in at the API
  layer via `CORTEX_ENABLE_RERANKER=true` (off by default — it adds model
  load time you don't want to pay while iterating on ingestion basics).
- `scripts/eval_queries.py` — single source of truth for the labeled query
  set both scripts consume. **Only 4 starter queries** — extend to 15-20+
  before trusting the sensitivity numbers for anything real.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload
```

## Local model

Default inference target: `Qwen2.5-1.5B-Instruct` quantized to GGUF (Q4_K_M),
served via `llama-cpp-python`. Small enough to fine-tune and run on CPU in
reasonable time. Swap in `slm/config.py`.
