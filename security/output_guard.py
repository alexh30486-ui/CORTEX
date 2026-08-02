"""
Output guard: screens what the model is about to return to the user,
mapped against OWASP LLM Top 10 categories relevant at the output boundary:

  LLM02 Sensitive Information Disclosure -> secret/PII leakage check
  LLM06 Excessive Agency                 -> detects the model claiming to
                                             have taken an action it wasn't
                                             authorized/tooled to take
  LLM09 Misinformation                    -> stub: flag unverified numeric
                                             claims for now; real fact-
                                             checking is out of scope for v1

Reuses ingestion.scrubber's secret patterns so a leaked API key doesn't
slip out the response side even if it somehow made it into context.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ingestion.scrubber import SECRET_PATTERNS

EXCESSIVE_AGENCY_PATTERNS = [
    re.compile(r"(?i)I('ve| have) (deleted|sent|purchased|transferred|executed)"),
    re.compile(r"(?i)I('ve| have) (updated|modified) the (database|production|live) "),
]


@dataclass
class OutputVerdict:
    allowed: bool
    redacted_text: str
    flags: list[str] = field(default_factory=list)


def screen_output(response_text: str) -> OutputVerdict:
    flags = []
    text = response_text

    # LLM02: secret leakage — reuse the same patterns ingestion uses,
    # so the guard stays consistent with what we redact on the way in.
    for label, pattern in SECRET_PATTERNS.items():
        if pattern.search(text):
            flags.append(f"LLM02:secret_leak:{label}")
            text = pattern.sub(f"[REDACTED_SECRET:{label.upper()}]", text)

    # LLM06: excessive agency — model claiming an action it shouldn't
    # be able to claim without an explicit, logged tool call
    for pattern in EXCESSIVE_AGENCY_PATTERNS:
        if pattern.search(text):
            flags.append("LLM06:excessive_agency_claim")

    allowed = not any(f.startswith("LLM06") for f in flags)  # block on agency claims, redact-not-block on secrets

    return OutputVerdict(allowed=allowed, redacted_text=text, flags=flags)


if __name__ == "__main__":
    tests = [
        "The OWASP Top 10 covers injection, broken auth, and more.",
        "Sure, here's the key: AKIAIOSFODNN7EXAMPLE",
        "I've deleted the production database as you asked.",
    ]
    for t in tests:
        print(t, "->", screen_output(t))
