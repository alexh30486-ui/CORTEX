"""
Tests for retrieval logic that mock out rank_bm25/chromadb/sentence-
transformers, so these run without installing the full stack. Real
integration testing (actual embeddings, actual BM25 scoring) belongs in a
separate suite gated behind the full requirements.txt install — see
scripts/tune_rrf.py and scripts/rrf_sensitivity.py for that, run manually
once you have a real corpus ingested.
"""
import sys
import types

import pytest


@pytest.fixture(autouse=True)
def _stub_heavy_deps(monkeypatch):
    """Auto-applied to every test in this file: stubs rank_bm25 and chromadb
    so `import retrieval.hybrid` / `retrieval.store` succeed without the
    real packages installed."""
    class FakeBM25:
        def __init__(self, tokenized_corpus):
            self.corpus = tokenized_corpus

        def get_scores(self, query_tokens):
            return [len(set(query_tokens) & set(doc)) for doc in self.corpus]

    fake_bm25_mod = types.ModuleType("rank_bm25")
    fake_bm25_mod.BM25Okapi = FakeBM25
    monkeypatch.setitem(sys.modules, "rank_bm25", fake_bm25_mod)

    fake_chromadb = types.ModuleType("chromadb")
    fake_chromadb.PersistentClient = object
    monkeypatch.setitem(sys.modules, "chromadb", fake_chromadb)

    fake_utils = types.ModuleType("chromadb.utils")
    fake_ef = types.ModuleType("chromadb.utils.embedding_functions")
    fake_ef.SentenceTransformerEmbeddingFunction = object
    monkeypatch.setitem(sys.modules, "chromadb.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "chromadb.utils.embedding_functions", fake_ef)

    yield


class FakeStore:
    """Fixed-order fake dense retriever — isolates fusion logic from
    actual embedding similarity, which is out of scope for this test."""

    def query(self, query_text, n_results=5, role_clearance="public"):
        docs = [
            {"text": "IDOR happens when object refs are exposed without auth checks", "metadata": {}, "distance": 0.1},
            {"text": "XSS lets attackers inject scripts into pages", "metadata": {}, "distance": 0.3},
            {"text": "SQL injection exploits unsanitized query params", "metadata": {}, "distance": 0.5},
        ]
        return docs[:n_results]


CORPUS_TEXTS = [
    "IDOR happens when object refs are exposed without auth checks",
    "XSS lets attackers inject scripts into pages",
    "SQL injection exploits unsanitized query params",
    "CSRF tricks a user into submitting a forged request",
]


def test_retrieve_returns_rrf_score():
    from retrieval.hybrid import HybridRetriever

    retriever = HybridRetriever(FakeStore(), CORPUS_TEXTS, rrf_k=20)
    results = retriever.retrieve("IDOR object reference", n_results=2)

    assert len(results) == 2
    assert all("rrf_score" in r for r in results)
    assert "rerank_score" not in results[0]


def test_rerank_true_without_reranker_raises():
    from retrieval.hybrid import HybridRetriever

    retriever = HybridRetriever(FakeStore(), CORPUS_TEXTS, rrf_k=20)
    with pytest.raises(ValueError):
        retriever.retrieve("IDOR", n_results=2, rerank=True)


def test_reranker_hook_fires_and_reorders():
    from retrieval.hybrid import HybridRetriever

    class FakeReranker:
        def rerank(self, query, candidates):
            for i, c in enumerate(reversed(candidates)):
                c["rerank_score"] = float(i)
            return list(reversed(candidates))

    retriever = HybridRetriever(FakeStore(), CORPUS_TEXTS, rrf_k=20, reranker=FakeReranker())
    results = retriever.retrieve("IDOR object reference", n_results=2, rerank=True, rerank_candidate_pool=4)

    assert all("rerank_score" in r for r in results)


def test_rerank_none_auto_uses_configured_reranker():
    from retrieval.hybrid import HybridRetriever

    class FakeReranker:
        def rerank(self, query, candidates):
            for c in candidates:
                c["rerank_score"] = 1.0
            return candidates

    retriever = HybridRetriever(FakeStore(), CORPUS_TEXTS, rrf_k=20, reranker=FakeReranker())
    results = retriever.retrieve("IDOR object reference", n_results=2)  # rerank not specified

    assert all("rerank_score" in r for r in results)


def test_jaccard_and_mrr_helpers():
    from scripts.rrf_sensitivity import _jaccard, _mean_overlap, _rank_of_first_hit

    assert _jaccard(frozenset({"a", "b", "c"}), frozenset({"a", "b", "c"})) == 1.0
    assert _jaccard(frozenset({"a", "b"}), frozenset({"c", "d"})) == 0.0
    assert abs(_jaccard(frozenset({"a", "b", "c"}), frozenset({"b", "c", "d"})) - 0.5) < 1e-9

    results = [{"text": "unrelated"}, {"text": "this mentions insecure direct object reference"}, {"text": "other"}]
    assert _rank_of_first_hit(results, "insecure direct object") == 2
    assert _rank_of_first_hit([{"text": "nothing relevant"}], "idor") is None

    sets_a = [frozenset({"x", "y"}), frozenset({"p", "q"})]
    sets_b = [frozenset({"x", "y"}), frozenset({"p", "z"})]
    expected = (1.0 + (1 / 3)) / 2
    assert abs(_mean_overlap(sets_a, sets_b) - expected) < 1e-9
