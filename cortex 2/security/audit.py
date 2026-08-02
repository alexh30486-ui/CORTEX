"""
Append-only audit log. Every input guard verdict, retrieval call, model
route decision, tool call, and output guard verdict gets a record here.

Local version uses a JSONL file with hash-chaining (each record includes
the hash of the previous record) so tampering is at least *detectable*
even without a database. Swap `AuditLog` for a Postgres-backed version
later without changing the call sites — that's the point of the thin
interface.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class AuditRecord:
    timestamp: float
    event_type: str  # e.g. "input_guard", "retrieval", "route_decision", "tool_call", "output_guard"
    payload: dict
    prev_hash: str
    record_hash: str = field(init=False)

    def __post_init__(self):
        digest_input = json.dumps(
            {"timestamp": self.timestamp, "event_type": self.event_type,
             "payload": self.payload, "prev_hash": self.prev_hash},
            sort_keys=True,
        ).encode()
        self.record_hash = hashlib.sha256(digest_input).hexdigest()


class AuditLog:
    def __init__(self, path: str | Path = "data/processed/audit_log.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def _last_hash(self) -> str:
        """
        Reads the last line of the log to get the previous record's hash.
        Uses a simple full-scan rather than byte-seeking; at the log
        volumes a local single-user deployment produces this is fast
        enough and far less error-prone than seek-based tricks. Swap for
        a proper backend (Postgres, SQLite) if log volume ever grows large
        enough for this to matter.
        """
        last_line = None
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    last_line = line
        if last_line is None:
            return "genesis"
        return json.loads(last_line)["record_hash"]

    def log(self, event_type: str, payload: dict) -> AuditRecord:
        record = AuditRecord(
            timestamp=time.time(),
            event_type=event_type,
            payload=payload,
            prev_hash=self._last_hash(),
        )
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")
        return record

    def verify_chain(self) -> bool:
        """Walk the log and confirm no record's prev_hash has been altered."""
        prev = "genesis"
        with open(self.path) as f:
            for line in f:
                rec = json.loads(line)
                if rec["prev_hash"] != prev:
                    return False
                prev = rec["record_hash"]
        return True


if __name__ == "__main__":
    log = AuditLog(path="data/processed/audit_log_smoketest.jsonl")
    log.log("input_guard", {"query": "example", "allowed": True, "risk_score": 0.0})
    log.log("output_guard", {"flags": [], "allowed": True})
    print("Chain valid:", log.verify_chain())
