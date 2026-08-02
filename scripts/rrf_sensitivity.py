#!/usr/bin/env python3
"""
Quantitative RRF sensitivity analysis.

tune_rrf.py lets you eyeball a handful of configs. This script instead
sweeps a grid of (rrf_k, dense_weight/sparse_weight ratio) and reports:

  - hit_rate@k   : fraction of labeled queries whose expected substring
                   appears anywhere in the top-k results
  - mrr          : mean reciprocal rank of the first chunk containing the
                   expected substring (0 if it never appears in the pool)
  - rank_overlap : Jaccard overlap of the top-k result sets between this
                   config and its neighbor in the sweep — this is the
                   actual "sensitivity" number: how much does a small
                   parameter change move the results? Low overlap between
                   adjacent configs means the system is sensitive (fragile)
                   to that parameter in this region; high overlap means
                   it's robust and you have room to tune for other reasons
                   (e.g. corpus growth) without breaking current behavior.

Also reports the same grid WITH cross-encoder reranking applied on top,
if retrieval/reranker.py's dependencies are installed, so you can see
whether reranking flattens sensitivity to RRF parameter choice (a
reranker that fixes bad RRF ordering would show much higher hit_rate/mrr
and much less variance across the k sweep — a concrete argument for
whether the extra reranking cost is worth it for your corpus).

Usage:
    python scripts/rrf_sensitivity.py
    python scripts/rrf_sensitivity.py --with-reranker
"""
from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval.hybrid import HybridRetriever
from retrieval.store import VectorStore
from scripts.eval_queries import LABELED_QUERIES

# Sweep grid — extend as needed. Kept modest so a run finishes in seconds
# even on a laptop CPU; BM25 + Chroma queries dominate cost here, not math.
RRF_K_VALUES = [5, 10, 20, 30, 60, 100]
WEIGHT_CONFIGS = [
    {"name": "equal", "dense_weight": 1.0, "sparse_weight": 1.0},
    {"name": "dense-heavy", "dense_weight": 0.7, "sparse_weight": 0.3},
    {"name": "sparse-heavy", "dense_weight": 0.4, "sparse_weight": 0.6},
]

N_RESULTS = 5


def _rank_of_first_hit(results: list[dict], expected: str) -> int | None:
    for i, r in enumerate(results, start=1):
        if expected.lower() in r["text"].lower():
            return i
    return None


def _evaluate_config(retriever: HybridRetriever, rrf_k: int, dense_weight: float,
                      sparse_weight: float, use_reranker: bool) -> dict:
    hits = 0
    reciprocal_ranks = []
    top_k_sets = []  # for rank-overlap computation across the sweep

    for q in LABELED_QUERIES:
        results = retriever.retrieve(
            q["query"],
            n_results=N_RESULTS,
            rrf_k=rrf_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            rerank=use_reranker,
        )
        rank = _rank_of_first_hit(results, q["expected"])
        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        top_k_sets.append(frozenset(r["text"] for r in results))

    n = len(LABELED_QUERIES)
    return {
        "rrf_k": rrf_k,
        "dense_weight": dense_weight,
        "sparse_weight": sparse_weight,
        "hit_rate": hits / n,
        "mrr": sum(reciprocal_ranks) / n,
        "top_k_sets": top_k_sets,  # consumed by the overlap calc, not printed directly
    }


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 1.0


def _mean_overlap(sets_a: list[frozenset], sets_b: list[frozenset]) -> float:
    """Average per-query Jaccard overlap between two configs' result sets."""
    return sum(_jaccard(a, b) for a, b in zip(sets_a, sets_b)) / len(sets_a)


def run_sweep(retriever: HybridRetriever, use_reranker: bool) -> list[dict]:
    all_results = []
    for weight_cfg in WEIGHT_CONFIGS:
        prev = None
        for rrf_k in RRF_K_VALUES:
            result = _evaluate_config(
                retriever, rrf_k, weight_cfg["dense_weight"], weight_cfg["sparse_weight"], use_reranker
            )
            result["weight_name"] = weight_cfg["name"]
            result["rank_overlap_vs_prev_k"] = (
                _mean_overlap(prev["top_k_sets"], result["top_k_sets"]) if prev else None
            )
            all_results.append(result)
            prev = result
    return all_results


def print_table(results: list[dict], title: str) -> None:
    print(f"\n{'=' * 90}\n{title}\n{'=' * 90}")
    print(f"{'weights':<14}{'rrf_k':<8}{'hit_rate':<10}{'mrr':<8}{'overlap_vs_prev_k':<20}")
    for r in results:
        overlap = f"{r['rank_overlap_vs_prev_k']:.2f}" if r["rank_overlap_vs_prev_k"] is not None else "  (first in sweep)"
        print(f"{r['weight_name']:<14}{r['rrf_k']:<8}{r['hit_rate']:<10.2f}{r['mrr']:<8.3f}{overlap:<20}")

    # Headline sensitivity summary: average overlap across the whole k-sweep,
    # per weight config. Low = results reshuffle a lot as k changes (sensitive).
    print("\n--- sensitivity summary (mean rank_overlap across k-sweep, per weight config) ---")
    by_weight: dict[str, list[float]] = {}
    for r in results:
        if r["rank_overlap_vs_prev_k"] is not None:
            by_weight.setdefault(r["weight_name"], []).append(r["rank_overlap_vs_prev_k"])
    for name, overlaps in by_weight.items():
        avg = sum(overlaps) / len(overlaps)
        verdict = "robust to k" if avg > 0.6 else "sensitive to k" if avg < 0.35 else "moderately sensitive to k"
        print(f"  {name:<14} mean_overlap={avg:.2f}  -> {verdict}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-reranker", action="store_true",
                         help="Also run the sweep with cross-encoder reranking applied")
    args = parser.parse_args()

    store = VectorStore()
    if store.count() == 0:
        print("Vector store is empty. Ingest documents first (POST /ingest or run pipeline).")
        return

    raw = store.collection.get()
    corpus_texts = raw.get("documents", []) or [""]
    print(f"Corpus size: {len(corpus_texts)} chunks | labeled queries: {len(LABELED_QUERIES)}")
    if len(LABELED_QUERIES) < 10:
        print("NOTE: fewer than 10 labeled queries — treat these numbers as a smoke test, "
              "not a real sensitivity conclusion. Extend scripts/eval_queries.py first.")

    retriever = HybridRetriever(store, corpus_texts)

    results_no_rerank = run_sweep(retriever, use_reranker=False)
    print_table(results_no_rerank, "RRF-only (no reranking)")

    if args.with_reranker:
        from retrieval.reranker import CrossEncoderReranker
        retriever.reranker = CrossEncoderReranker()
        results_with_rerank = run_sweep(retriever, use_reranker=True)
        print_table(results_with_rerank, "RRF + cross-encoder reranking")

        print(f"\n{'=' * 90}\nDelta summary (reranking effect on hit_rate/mrr, averaged over full sweep)\n{'=' * 90}")
        avg_hit_no = sum(r["hit_rate"] for r in results_no_rerank) / len(results_no_rerank)
        avg_hit_re = sum(r["hit_rate"] for r in results_with_rerank) / len(results_with_rerank)
        avg_mrr_no = sum(r["mrr"] for r in results_no_rerank) / len(results_no_rerank)
        avg_mrr_re = sum(r["mrr"] for r in results_with_rerank) / len(results_with_rerank)
        print(f"  hit_rate: {avg_hit_no:.3f} -> {avg_hit_re:.3f}  (delta {avg_hit_re - avg_hit_no:+.3f})")
        print(f"  mrr:      {avg_mrr_no:.3f} -> {avg_mrr_re:.3f}  (delta {avg_mrr_re - avg_mrr_no:+.3f})")


if __name__ == "__main__":
    main()
