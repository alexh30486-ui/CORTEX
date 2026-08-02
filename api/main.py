"""
FastAPI entrypoint. Wires:
  ingestion (via /ingest) -> retrieval.store + retrieval.hybrid -> agent.orchestrator -> response

Run with: uvicorn api.main:app --reload
"""
from __future__ import annotations

import json
import os

from fastapi import FastAPI
from pydantic import BaseModel

from agent.orchestrator import Orchestrator
from ingestion.pipeline import IngestionPipeline
from retrieval.hybrid import HybridRetriever
from retrieval.store import VectorStore
from security.audit import AuditLog

app = FastAPI(title="Cortex", description="Secure multimodal RAG agent with local SLM optimization")

# --- shared singletons ---
_vector_store = VectorStore()
_audit_log = AuditLog()


def _placeholder_generate(query: str, context: str) -> str:
    """Stand-in until slm/serve.py's model is wired in + an API escalation
    target is chosen. Keeps the whole pipeline runnable end-to-end today."""
    return f"[stub response] Would answer '{query}' using {len(context)} chars of retrieved context."


def _build_retriever() -> HybridRetriever:
    # BM25 needs the raw corpus text; pull it from Chroma's underlying collection.
    # For small corpora this is fine; for large corpora cache this instead of
    # rebuilding per-request (TODO once corpus size grows).
    raw = _vector_store.collection.get()
    corpus_texts = raw.get("documents", []) or [""]

    reranker = None
    if os.environ.get("CORTEX_ENABLE_RERANKER", "false").lower() == "true":
        # Opt-in via env var: loading the cross-encoder adds ~80MB and a
        # few hundred ms of startup cost, not worth paying by default while
        # you're still iterating on ingestion/retrieval basics.
        from retrieval.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()

    return HybridRetriever(_vector_store, corpus_texts, reranker=reranker)


_orchestrator = Orchestrator(
    retriever=_build_retriever(),
    local_generate_fn=_placeholder_generate,
    escalate_generate_fn=_placeholder_generate,
    audit_log=_audit_log,
)


class QueryRequest(BaseModel):
    query: str
    role_clearance: str = "public"
    force_deep_mode: bool = False


class IngestRequest(BaseModel):
    directory: str = "data/raw"


@app.post("/ingest")
def ingest(req: IngestRequest):
    pipeline = IngestionPipeline(use_pii_scrubbing=False)  # flip True once spaCy model is downloaded
    chunks = pipeline.ingest_directory(req.directory, out_path="data/processed/chunks.json")

    with open("data/processed/chunks.json") as f:
        chunk_dicts = json.load(f)
    _vector_store.add_chunks(chunk_dicts)

    global _orchestrator
    _orchestrator.retriever = _build_retriever()

    return {"ingested_chunks": len(chunks), "total_in_store": _vector_store.count()}


@app.post("/query")
def query(req: QueryRequest):
    result = _orchestrator.handle(
        query=req.query,
        role_clearance=req.role_clearance,
        force_deep_mode=req.force_deep_mode,
    )
    return {
        "response": result.response,
        "blocked": result.blocked,
        "block_reason": result.block_reason,
        "trace": result.trace,
    }


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": _vector_store.count()}
