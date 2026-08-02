"""
PII and secret scrubbing.

This runs BEFORE any text is chunked, embedded, or written to the vector
store. That ordering is the whole privacy story of this project: nothing
enters long-term storage without having passed through here first.

Two layers:
  1. Regex-based secret detection (API keys, AWS creds, private keys, JWTs).
     Fast, deterministic, no model dependency — always runs.
  2. Presidio-based PII detection (names, emails, phone numbers, SSNs, etc.)
     Model-backed, catches things regex can't.

Both layers redact in place and return a report of what was found + where,
so the audit log (security/audit.py) has a record of what was scrubbed
without ever storing the sensitive value itself.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# --- Layer 1: regex secret patterns -----------------------------------------
# Keep these anchored and specific enough to avoid false-positive-heavy noise.
SECRET_PATTERNS: dict[str, re.Pattern] = {
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "aws_secret_key": re.compile(r"(?i)aws(.{0,20})?(secret|access)?[_-]?key(.{0,20})?['\"]?\s*[:=]\s*['\"]?[A-Za-z0-9/+=]{40}"),
    "generic_api_key": re.compile(r"(?i)(api[_-]?key|apikey|secret[_-]?key)['\"]?\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,64}"),
    "private_key_block": re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----[\s\S]+?-----END (RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
    "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,255}\b"),
}


@dataclass
class ScrubReport:
    original_length: int
    redacted_length: int
    findings: list[dict] = field(default_factory=list)  # [{type, category, count}]

    @property
    def had_findings(self) -> bool:
        return len(self.findings) > 0


def _scrub_secrets(text: str) -> tuple[str, list[dict]]:
    findings = []
    for label, pattern in SECRET_PATTERNS.items():
        matches = list(pattern.finditer(text))
        if matches:
            findings.append({"type": label, "category": "secret", "count": len(matches)})
            text = pattern.sub(f"[REDACTED_SECRET:{label.upper()}]", text)
    return text, findings


def _scrub_pii(text: str, analyzer=None) -> tuple[str, list[dict]]:
    """
    Presidio-based PII scrubbing. Lazily accepts an analyzer instance so
    callers can reuse one across many documents (loading the spaCy model
    per-call is expensive).
    """
    if analyzer is None:
        return text, []

    results = analyzer.analyze(
        text=text,
        entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN",
                  "CREDIT_CARD", "IP_ADDRESS", "LOCATION"],
        language="en",
    )

    findings = []
    # Sort in reverse so we can splice by offset without shifting later indices
    for r in sorted(results, key=lambda x: x.start, reverse=True):
        text = text[: r.start] + f"[REDACTED_PII:{r.entity_type}]" + text[r.end :]
    if results:
        by_type: dict[str, int] = {}
        for r in results:
            by_type[r.entity_type] = by_type.get(r.entity_type, 0) + 1
        findings = [{"type": t, "category": "pii", "count": c} for t, c in by_type.items()]

    return text, findings


def scrub(text: str, analyzer=None) -> tuple[str, ScrubReport]:
    """
    Main entry point. Runs secret scrubbing then PII scrubbing.

    `analyzer` is an optional presidio_analyzer.AnalyzerEngine instance.
    Pass None to skip PII scrubbing (e.g. in environments where the spaCy
    model isn't installed yet) — secret scrubbing always runs regardless.
    """
    original_length = len(text)
    text, secret_findings = _scrub_secrets(text)
    text, pii_findings = _scrub_pii(text, analyzer=analyzer)

    report = ScrubReport(
        original_length=original_length,
        redacted_length=len(text),
        findings=secret_findings + pii_findings,
    )
    return text, report


def get_default_analyzer():
    """
    Lazy factory for a presidio AnalyzerEngine. Import is deferred so the
    rest of the ingestion pipeline works even if presidio/spacy aren't
    installed yet (useful for the first local run before `pip install -r
    requirements.txt` + `python -m spacy download en_core_web_lg`).
    """
    from presidio_analyzer import AnalyzerEngine

    return AnalyzerEngine()


if __name__ == "__main__":
    sample = """
    Contact John Smith at john.smith@example.com or 555-123-4567.
    AWS key: AKIAIOSFODNN7EXAMPLE
    api_key: "sk_live_51H8xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    """
    cleaned, report = scrub(sample)  # analyzer=None -> secrets only, for a quick smoke test
    print("--- cleaned text ---")
    print(cleaned)
    print("--- report ---")
    print(report)
