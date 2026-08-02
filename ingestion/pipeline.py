"""
End-to-end ingestion pipeline: parse -> scrub -> chunk -> tag -> emit.

This is the single choke point every document passes through before
reaching retrieval/embedder.py. If you need to add a new modality or
change scrubbing behavior, this is the file that changes, not the
downstream retrieval code.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from ingestion.parsers import ParsedChunk, parse_file
from ingestion.scrubber import ScrubReport, get_default_analyzer, scrub

SensitivityLevel = Literal["public", "internal", "confidential"]


@dataclass
class IngestedChunk:
    text: str
    modality: str
    source_path: str
    sensitivity: SensitivityLevel
    scrub_report: dict
    page_or_segment: int | None = None
    metadata: dict = field(default_factory=dict)


def _infer_sensitivity(scrub_report: ScrubReport) -> SensitivityLevel:
    """
    Simple heuristic: any secret finding -> confidential. Any PII -> at
    least internal. Otherwise public. This is intentionally conservative;
    tune per your actual data governance policy.
    """
    if any(f["category"] == "secret" for f in scrub_report.findings):
        return "confidential"
    if any(f["category"] == "pii" for f in scrub_report.findings):
        return "internal"
    return "public"


class IngestionPipeline:
    def __init__(self, use_pii_scrubbing: bool = True):
        """
        use_pii_scrubbing=False skips the presidio/spaCy layer (useful for
        a fast first run before those models are downloaded). Secret
        scrubbing always runs regardless.
        """
        self.analyzer = get_default_analyzer() if use_pii_scrubbing else None

    def ingest_file(self, path: str | Path) -> list[IngestedChunk]:
        raw_chunks: list[ParsedChunk] = parse_file(path)
        ingested = []

        for chunk in raw_chunks:
            clean_text, report = scrub(chunk.text, analyzer=self.analyzer)
            sensitivity = _infer_sensitivity(report)

            ingested.append(
                IngestedChunk(
                    text=clean_text,
                    modality=chunk.modality,
                    source_path=chunk.source_path,
                    sensitivity=sensitivity,
                    scrub_report=asdict(report),
                    page_or_segment=chunk.page_or_segment,
                    metadata=chunk.metadata,
                )
            )
        return ingested

    def ingest_directory(self, directory: str | Path, out_path: str | Path | None = None) -> list[IngestedChunk]:
        directory = Path(directory)
        all_chunks: list[IngestedChunk] = []

        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in {
                ".pdf", ".docx", ".png", ".jpg", ".jpeg", ".mp3", ".wav", ".m4a",
            }:
                try:
                    all_chunks.extend(self.ingest_file(path))
                except Exception as e:  # noqa: BLE001 - log and continue on a bad file
                    print(f"[ingestion] failed on {path}: {e}")

        if out_path:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump([asdict(c) for c in all_chunks], f, indent=2)

        return all_chunks


if __name__ == "__main__":
    # quick manual smoke test: point at data/raw and dump to data/processed
    pipeline = IngestionPipeline(use_pii_scrubbing=False)  # flip True once spaCy model is downloaded
    chunks = pipeline.ingest_directory("data/raw", out_path="data/processed/chunks.json")
    print(f"Ingested {len(chunks)} chunks.")
    for c in chunks[:3]:
        print(f"  [{c.sensitivity}] ({c.modality}) {c.text[:80]!r}")
