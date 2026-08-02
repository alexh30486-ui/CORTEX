"""
Hybrid retrieval: combine dense (Chroma/embeddings) with sparse (BM25)
scoring, since appsec/technical corpora often have exact-term queries
(CVE IDs, function names, error strings) where pure embedding search
under-performs.

Reciprocal rank fusion (RRF) combines the two ranked lists. Parameters
are exposed so you can tune for corpus size and query mix:

  - rrf_k          : smoothing constant (lower -> more weight on top ranks)
  - dense_weight   : multiplier for the dense contribution
  - sparse_weight  : multiplier for the BM25 contribution
  - over_fetch     : how many candidates to pull from each side before fusion

Default rrf_k=20 is intentionally lower than the classic 60 because the
expected local appsec corpus is small-to-medium; lower k gives sharper
top-rank discrimination. Raise it toward 60 if the corpus grows large.

Optional reranking: pass a `reranker` (see retrieval.reranker.CrossEncoderReranker)
to re-score the fused top candidates with a cross-encoder before returning.
This is a second, more expensive but more accurate pass over a small
candidate set — RRF narrows N results down cheaply, the cross-encoder then
re-orders just those few candidates using actual query-document attention
rather than independent embedding similarity.
"""
from __future__ import annotations

from typing import Optional

from rank_bm25 import BM25Okapi

from retrieval.store import VectorStore


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore,
        corpus_texts: list[str],
        rrf_k: int = 20,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        over_fetch: int = 3,
        reranker: Optional["object"] = None,
    ):
        self.store = store
        self.corpus_texts = corpus_texts
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.over_fetch = over_fetch
        self.reranker = reranker  # retrieval.reranker.CrossEncoderReranker instance, or None

        tokenized = [t.lower().split() for t in corpus_texts]
        self.bm25 = BM25Okapi(tokenized)

    def _bm25_rank(self, query: str, top_k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        role_clearance: str = "public",
        rrf_k: int | None = None,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
        over_fetch: int | None = None,
        rerank: bool | None = None,
        rerank_candidate_pool: int = 20,
    ) -> list[dict]:
        """
        Returns list of dicts with keys:
          text, metadata, distance (original dense distance or None),
          rrf_score (fused score - higher is better),
          rerank_score (cross-encoder score, only present if reranked)

        rerank: None -> uses reranker if one was passed to the constructor.
                True  -> force reranking (raises if no reranker configured).
                False -> skip reranking even if a reranker is configured.
        rerank_candidate_pool: how many RRF-fused candidates to hand to the
                cross-encoder before truncating to n_results. Keep this
                modest (10-30) since cross-encoder scoring is O(candidates),
                unlike the cheap RRF fusion step.
        """
        k = rrf_k if rrf_k is not None else self.rrf_k
        dw = dense_weight if dense_weight is not None else self.dense_weight
        sw = sparse_weight if sparse_weight is not None else self.sparse_weight
        of = over_fetch if over_fetch is not None else self.over_fetch

        should_rerank = rerank if rerank is not None else (self.reranker is not None)
        if should_rerank and self.reranker is None:
            raise ValueError("rerank=True but no reranker was configured on this HybridRetriever")

        # Widen the fetch when reranking so the cross-encoder has a real
        # candidate pool to work with, not just the final n_results.
        target_n = max(n_results, rerank_candidate_pool) if should_rerank else n_results
        fetch_n = max(target_n * of, target_n)

        dense_results = self.store.query(
            query, n_results=fetch_n, role_clearance=role_clearance
        )
        sparse_ranked = self._bm25_rank(query, top_k=fetch_n)

        # RRF fusion keyed on text content (simplest portable join key
        # given we don't have a shared stable ID across both indices here)
        rrf_scores: dict[str, float] = {}
        text_lookup: dict[str, dict] = {}

        for rank, r in enumerate(dense_results):
            key = r["text"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + dw * (1.0 / (k + rank + 1))
            text_lookup[key] = dict(r)  # shallow copy so we can add scores

        for rank, (idx, _) in enumerate(sparse_ranked):
            key = self.corpus_texts[idx]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + sw * (1.0 / (k + rank + 1))
            if key not in text_lookup:
                text_lookup[key] = {"text": key, "metadata": {}, "distance": None}

        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        candidate_n = rerank_candidate_pool if should_rerank else n_results
        results = []
        for key, score in fused[:candidate_n]:
            item = text_lookup[key]
            item["rrf_score"] = score
            results.append(item)

        if should_rerank:
            results = self.reranker.rerank(query, results)

        return results[:n_results]
