"""
Benchmark: base model vs LoRA fine-tuned model vs (optionally) a teacher
API model, on held-out appsec Q&A. Produces the evidence table for the
"SLM optimization" claim — latency, memory, and a rough accuracy proxy.

Accuracy proxy for v1: exact-match / ROUGE-L against expected answers on
a held-out set (slm/data/appsec_eval.jsonl, same schema as training data
but NOT overlapping with it). Swap in an LLM-as-judge scorer later for a
more nuanced score — noted as a TODO below.
"""
from __future__ import annotations

import json
import time
import tracemalloc
from dataclasses import asdict, dataclass

EVAL_PATH = "slm/data/appsec_eval.jsonl"


@dataclass
class BenchResult:
    model_name: str
    avg_latency_sec: float
    peak_memory_mb: float
    rouge_l_f1: float
    n_examples: int


def _rouge_l_f1(pred: str, ref: str) -> float:
    """Minimal ROUGE-L implementation (LCS-based) to avoid an extra dependency."""
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    m, n = len(pred_tokens), len(ref_tokens)
    if m == 0 or n == 0:
        return 0.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]

    precision = lcs / m
    recall = lcs / n
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def run_benchmark(model_name: str, generate_fn, eval_path: str = EVAL_PATH) -> BenchResult:
    with open(eval_path) as f:
        examples = [json.loads(line) for line in f if line.strip()]

    tracemalloc.start()
    latencies = []
    scores = []

    for ex in examples:
        query = f"{ex['instruction']}\n{ex.get('input', '')}"
        start = time.perf_counter()
        prediction = generate_fn(query, context="")
        latencies.append(time.perf_counter() - start)
        scores.append(_rouge_l_f1(prediction, ex["output"]))

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return BenchResult(
        model_name=model_name,
        avg_latency_sec=sum(latencies) / len(latencies),
        peak_memory_mb=peak / (1024 * 1024),
        rouge_l_f1=sum(scores) / len(scores),
        n_examples=len(examples),
    )


def print_table(results: list[BenchResult]) -> None:
    print(f"{'Model':<30}{'Avg Latency (s)':<18}{'Peak Mem (MB)':<16}{'ROUGE-L F1':<12}")
    for r in results:
        print(f"{r.model_name:<30}{r.avg_latency_sec:<18.3f}{r.peak_memory_mb:<16.1f}{r.rouge_l_f1:<12.3f}")


if __name__ == "__main__":
    # TODO once fine-tuning is done: import and register each candidate
    # generate_fn here, e.g.:
    #
    #   from slm.serve import generate as base_generate
    #   results = [run_benchmark("qwen2.5-1.5b-base", base_generate)]
    #
    #   # after merging LoRA adapter into a new GGUF:
    #   from slm.serve_finetuned import generate as finetuned_generate
    #   results.append(run_benchmark("qwen2.5-1.5b-appsec-lora", finetuned_generate))
    #
    #   print_table(results)
    print("Register model generate_fn callables in __main__ once fine-tuning is complete. See TODO comments.")
