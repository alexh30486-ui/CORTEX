"""
Local vector store on top of Chroma (persisted to disk, no server process
required — this is why we chose it over Qdrant/pgvector for the local-first
version of this project).

Sensitivity tagging from ingestion carries through as metadata, so retrieval
can be scoped by caller role (see `query` role_clearance param). This is the
mechanism that turns "confidential" tags from ingestion/pipeline.py into an
actual access control boundary instead of just a label.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import chromadb
from chromadb.utils import embedding_functions

SensitivityLevel = Literal["public", "internal", "confidential"]

# Ordered so we can do "at or below this clearance level" comparisons.
_CLEARANCE_RANK = {"public": 0, "internal": 1, "confidential": 2}

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast, CPU-friendly


class VectorStore:
    def __init__(
        self,
        persist_dir: str | Path = "data/vectorstore",
        collection_name: str = "cortex_docs",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embed_fn
        )

    def add_chunks(self, chunks: list[dict]) -> None:
        """
        chunks: list of dicts matching ingestion.pipeline.IngestedChunk fields
        (as produced by `asdict()`), plus a unique 'id' per chunk.
        """
        ids = [c.get("id") or f"{c['source_path']}::{c.get('page_or_segment')}::{i}"
               for i, c in enumerate(chunks)]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {
                "source_path": c["source_path"],
                "modality": c["modality"],
                "sensitivity": c["sensitivity"],
                "page_or_segment": c.get("page_or_segment") or -1,
            }
            for c in chunks
        ]
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        role_clearance: SensitivityLevel = "public",
    ) -> list[dict]:
        """
        Returns only chunks whose sensitivity <= the caller's clearance.
        Chroma doesn't support inequality filters on arbitrary fields
        natively in all versions, so we over-fetch and filter client-side
        for portability.
        """
        max_rank = _CLEARANCE_RANK[role_clearance]
        allowed = [lvl for lvl, rank in _CLEARANCE_RANK.items() if rank <= max_rank]

        raw = self.collection.query(
            query_texts=[query_text],
            n_results=max(n_results * 3, 10),  # over-fetch, then filter
            where={"sensitivity": {"$in": allowed}},
        )

        results = []
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            results.append({"text": doc, "metadata": meta, "distance": dist})

        return results[:n_results]

    def count(self) -> int:
        return self.collection.count()


if __name__ == "__main__":
    import json

    store = VectorStore()

    with open("data/processed/chunks.json") as f:
        chunks = json.load(f)

    store.add_chunks(chunks)
    print(f"Indexed {store.count()} chunks total.")

    results = store.query("example query", role_clearance="internal")
    for r in results:
        print(f"[{r['metadata']['sensitivity']}] {r['text'][:80]!r} (dist={r['distance']:.3f})")
