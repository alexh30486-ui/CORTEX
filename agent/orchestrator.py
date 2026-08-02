"""
Explicit state machine: input_guard -> retrieve -> route -> generate ->
critic -> output_guard -> audit.

Hand-rolled rather than a framework (LangGraph etc.) so every transition
is inspectable and the control flow doesn't hide behind a DSL — the point
of this file is to be the thing you walk an interviewer through line by
line.

TODO (slm/): wire in the actual local model call once slm/serve.py is
implemented. TODO (agent/): wire in the escalation API call once you pick
a provider. Both are stubbed with clear call sites below.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from agent.router import ModelTarget, route
from security.audit import AuditLog
from security.input_guard import screen_input
from security.output_guard import screen_output


class Stage(Enum):
    INPUT_GUARD = auto()
    RETRIEVE = auto()
    ROUTE = auto()
    GENERATE = auto()
    CRITIC = auto()
    OUTPUT_GUARD = auto()
    DONE = auto()
    BLOCKED = auto()


@dataclass
class OrchestratorResult:
    response: str | None
    blocked: bool
    block_reason: str | None = None
    trace: list[str] = field(default_factory=list)


class Orchestrator:
    def __init__(self, retriever, local_generate_fn, escalate_generate_fn, audit_log: AuditLog | None = None):
        """
        retriever: object exposing .retrieve(query, n_results, role_clearance) -> list[dict]
                   (retrieval.hybrid.HybridRetriever satisfies this)
        local_generate_fn: callable(query, context) -> str          (slm/serve.py)
        escalate_generate_fn: callable(query, context) -> str        (your chosen API model)
        """
        self.retriever = retriever
        self.local_generate_fn = local_generate_fn
        self.escalate_generate_fn = escalate_generate_fn
        self.audit = audit_log or AuditLog()

    def handle(self, query: str, role_clearance: str = "public", force_deep_mode: bool = False) -> OrchestratorResult:
        trace = []

        # --- Stage 1: input guard ---
        verdict = screen_input(query)
        self.audit.log("input_guard", {"query": query, "allowed": verdict.allowed,
                                        "risk_score": verdict.risk_score,
                                        "matched": verdict.matched_categories})
        trace.append(f"input_guard: {verdict.reason}")
        if not verdict.allowed:
            return OrchestratorResult(response=None, blocked=True,
                                       block_reason=f"input blocked: {verdict.reason}", trace=trace)

        # --- Stage 2: retrieve ---
        results = self.retriever.retrieve(query, n_results=5, role_clearance=role_clearance)
        top_distance = results[0]["distance"] if results and results[0].get("distance") is not None else None
        context = "\n\n".join(r["text"] for r in results)
        self.audit.log("retrieval", {"query": query, "n_results": len(results),
                                      "top_distance": top_distance})
        trace.append(f"retrieved {len(results)} chunks (top_distance={top_distance})")

        # --- Stage 3: route ---
        decision = route(query, retrieval_top_distance=top_distance, force_deep_mode=force_deep_mode)
        self.audit.log("route_decision", {"target": decision.target.value, "reasons": decision.reasons})
        trace.append(f"route: {decision.target.value} ({'; '.join(decision.reasons)})")

        # --- Stage 4: generate ---
        if decision.target == ModelTarget.LOCAL_SLM:
            raw_response = self.local_generate_fn(query, context)
        else:
            raw_response = self.escalate_generate_fn(query, context)
        trace.append(f"generated {len(raw_response)} chars via {decision.target.value}")

        # --- Stage 5: critic (placeholder — TODO: self-consistency or
        #     verifier-model pass before returning to the user) ---
        # For v1 this is a no-op pass-through so the pipeline is complete
        # end-to-end; extend here once you want a second model pass.
        critiqued_response = raw_response

        # --- Stage 6: output guard ---
        out_verdict = screen_output(critiqued_response)
        self.audit.log("output_guard", {"flags": out_verdict.flags, "allowed": out_verdict.allowed})
        trace.append(f"output_guard: flags={out_verdict.flags}")

        if not out_verdict.allowed:
            return OrchestratorResult(response=None, blocked=True,
                                       block_reason=f"output blocked: {out_verdict.flags}", trace=trace)

        return OrchestratorResult(response=out_verdict.redacted_text, blocked=False, trace=trace)
