"""
Cross-encoder reranking.

RRF fusion (retrieval/hybrid.py) is cheap but scores each candidate using
independent representations — dense embedding similarity and BM25 term
overlap, fused by rank position, with no direct query-document interaction.
A cross-encoder instead runs the (query, document) pair jointly through a
single transformer, letting attention directly compare the two texts. This
is strictly more accurate but far more expensive per pair, which is why it
only runs over the small candidate pool RRF has already narrowed down —
never over the full corpus.

Model choice: cross-encoder/ms-marco-MiniLM-L-6-v2. ~80MB, CPU inference
in the tens-of-milliseconds-per-pair range, trained on MS MARCO passage
ranking — a reasonable general-purpose default before you have appsec-
specific reranking training data. Swappable via `model_name`.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass
class RerankConfig:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 16


@lru_cache(maxsize=4)
def _load_model(model_name: str):
    """Cached so repeated CrossEncoderReranker instances in the same
    process (e.g. across requests) don't reload weights from disk."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


class CrossEncoderReranker:
    def __init__(self, config: RerankConfig | None = None):
        self.config = config or RerankConfig()
        self._model = None  # lazy-loaded on first use, not at construction

    @property
    def model(self):
        if self._model is None:
            self._model = _load_model(self.config.model_name)
        return self._model

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """
        candidates: list of dicts with a "text" key (as produced by
        HybridRetriever's RRF fusion step).
        Returns the same dicts, each with an added "rerank_score" key,
        sorted descending by that score.
        """
        if not candidates:
            return candidates

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs, batch_size=self.config.batch_size)

        for c, score in zip(candidates, scores):
            c["rerank_score"] = float(score)

        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)


if __name__ == "__main__":
    reranker = CrossEncoderReranker()
    query = "How do I fix an IDOR vulnerability?"
    candidates = [
        {"text": "IDOR occurs when object references are exposed without authorization checks."},
        {"text": "Cross-site scripting lets attackers inject client-side scripts."},
        {"text": "To prevent IDOR, enforce access control checks on every object reference server-side."},
    ]
    ranked = reranker.rerank(query, candidates)
    for r in ranked:
        print(f"{r['rerank_score']:.3f}  {r['text']}")
