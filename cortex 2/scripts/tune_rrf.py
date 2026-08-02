#!/usr/bin/env python3
"""
Tiny offline helper for A/B-ing RRF parameters.

Usage (once you have documents ingested and a small labeled set):

    python scripts/tune_rrf.py

It prints, for each query, the top-k results under several (k, dense_w, sparse_w)
configurations so you can eyeball which setting surfaces the expected chunks.

For a quantitative (not eyeball) comparison, see scripts/rrf_sensitivity.py,
which sweeps a grid of parameters and reports hit-rate/MRR per config.

Replace LABELED_QUERIES with real appsec queries + optional expected substrings
once you have them. No external dependencies beyond the rest of the project.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval.hybrid import HybridRetriever
from retrieval.store import VectorStore
from scripts.eval_queries import LABELED_QUERIES

# Configurations to compare. First one matches the project defaults.
CONFIGS = [
    {"name": "default (k=20, equal)", "rrf_k": 20, "dense_weight": 1.0, "sparse_weight": 1.0},
    {"name": "classic (k=60, equal)", "rrf_k": 60, "dense_weight": 1.0, "sparse_weight": 1.0},
    {"name": "sparse-heavy (k=20)", "rrf_k": 20, "dense_weight": 0.4, "sparse_weight": 0.6},
    {"name": "dense-heavy (k=20)", "rrf_k": 20, "dense_weight": 0.7, "sparse_weight": 0.3},
    {"name": "sharp (k=10, equal)", "rrf_k": 10, "dense_weight": 1.0, "sparse_weight": 1.0},
]


def main() -> None:
    store = VectorStore()
    if store.count() == 0:
        print("Vector store is empty. Ingest documents first (POST /ingest or run pipeline).")
        return

    raw = store.collection.get()
    corpus_texts = raw.get("documents", []) or [""]
    print(f"Corpus size: {len(corpus_texts)} chunks\n")

    base = HybridRetriever(store, corpus_texts)  # uses constructor defaults

    for q in LABELED_QUERIES:
        print("=" * 72)
        print(f"QUERY: {q['query']}")
        if q.get("expected"):
            print(f"  (looking for substring: {q['expected']!r})")
        print()

        for cfg in CONFIGS:
            results = base.retrieve(
                q["query"],
                n_results=3,
                rrf_k=cfg["rrf_k"],
                dense_weight=cfg["dense_weight"],
                sparse_weight=cfg["sparse_weight"],
            )
            print(f"  [{cfg['name']}]")
            for i, r in enumerate(results, 1):
                preview = r["text"][:90].replace("\n", " ")
                hit = ""
                if q.get("expected") and q["expected"].lower() in r["text"].lower():
                    hit = "  <- expected hit"
                print(f"    {i}. rrf={r['rrf_score']:.4f}  dist={r.get('distance')}  {preview!r}{hit}")
            print()


if __name__ == "__main__":
    main()
