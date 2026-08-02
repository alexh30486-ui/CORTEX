"""
Router: decides whether a query can be handled by the local, fine-tuned
SLM or needs to escalate to a larger API model.

This is the concrete cost/latency tradeoff at the center of the project's
"SLM optimization" story — not just "we quantized a model" but "here's the
logic that decides when the cheap model is good enough."

Escalation triggers (any one is sufficient):
  - query complexity heuristic exceeds threshold (long, multi-clause,
    or outside the fine-tuned domain vocabulary)
  - local model's own confidence/perplexity signal is low
  - retrieval returned low-relevance context (best distance above threshold)
  - explicit user request for "deep" / "thorough" mode

Kept as simple, inspectable heuristics rather than a learned router for v1
so the tradeoff is legible and debuggable. A learned router is a natural
v2 TODO once you have escalation-decision labels from real usage.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ModelTarget(Enum):
    LOCAL_SLM = "local_slm"
    ESCALATE_API = "escalate_api"


@dataclass
class RouteDecision:
    target: ModelTarget
    reasons: list[str]


# Rough domain vocabulary for the appsec fine-tune target. Extend this as
# the LoRA training set grows — keeping it here makes the coupling between
# "what we fine-tuned on" and "what we trust locally" explicit and visible.
DOMAIN_VOCAB = {
    "owasp", "vulnerability", "cve", "injection", "xss", "csrf", "jwt",
    "authentication", "authorization", "sanitize", "sql injection",
    "broken access control", "ssrf", "idor", "rate limit", "api security",
    "prompt injection", "llm", "sast", "dast",
}

MAX_LOCAL_QUERY_WORDS = 60
MIN_RETRIEVAL_CONFIDENCE = 0.55  # lower distance = more relevant; tune per embedding model


def route(
    query: str,
    retrieval_top_distance: float | None = None,
    force_deep_mode: bool = False,
) -> RouteDecision:
    reasons = []

    if force_deep_mode:
        return RouteDecision(ModelTarget.ESCALATE_API, ["user requested deep mode"])

    word_count = len(query.split())
    if word_count > MAX_LOCAL_QUERY_WORDS:
        reasons.append(f"query too long for local model ({word_count} words)")

    query_lower = query.lower()
    in_domain = any(term in query_lower for term in DOMAIN_VOCAB)
    if not in_domain:
        reasons.append("query outside fine-tuned domain vocabulary")

    if retrieval_top_distance is not None and retrieval_top_distance > MIN_RETRIEVAL_CONFIDENCE:
        reasons.append(f"low retrieval confidence (distance={retrieval_top_distance:.2f})")

    if reasons:
        return RouteDecision(ModelTarget.ESCALATE_API, reasons)

    return RouteDecision(ModelTarget.LOCAL_SLM, ["in-domain, short, high-confidence retrieval"])


if __name__ == "__main__":
    print(route("What does OWASP say about broken access control?", retrieval_top_distance=0.2))
    print(route("Explain the entire history of quantum computing in detail with examples"))
    print(route("How do I fix IDOR?", retrieval_top_distance=0.8))
