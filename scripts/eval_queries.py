"""
Single source of truth for the labeled appsec query set used across
retrieval evaluation scripts (tune_rrf.py, rrf_sensitivity.py). Keeping
this in one place means updating your eval set once updates every script
that consumes it, rather than hunting down copies.

"expected" is a substring (case-insensitive) that should appear in a
genuinely relevant chunk. Extend this as your corpus grows — aim for at
least 15-20 labeled queries before trusting the sensitivity numbers much;
four queries (the starter set below) is enough to smoke-test the scripts
but too few to draw real conclusions from.
"""
from __future__ import annotations

LABELED_QUERIES: list[dict] = [
    {"query": "How do I fix IDOR?", "expected": "insecure direct object"},
    {"query": "What does OWASP say about broken access control?", "expected": "broken access control"},
    {"query": "CVE-2021-44228 Log4Shell mitigation", "expected": "log4j"},
    {"query": "prompt injection defense for LLM apps", "expected": "injection"},
]
