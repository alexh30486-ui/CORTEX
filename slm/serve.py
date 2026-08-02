"""
Local CPU inference via llama-cpp-python + GGUF quantized weights.

This is the function the orchestrator calls for `local_generate_fn`.
Kept as a thin wrapper so swapping the underlying runtime (e.g. to
Ollama, or to a merged LoRA GGUF once fine-tuning is done) doesn't
touch orchestrator.py.
"""
from __future__ import annotations

from functools import lru_cache

from slm.config import GGUF_FILENAME, GGUF_REPO_ID, MAX_NEW_TOKENS, N_CTX, N_THREADS, TEMPERATURE

SYSTEM_PROMPT = (
    "You are a focused application security assistant. Answer using only "
    "the provided context when relevant. If the context doesn't contain "
    "the answer, say so rather than guessing. Be concise and technical."
)


@lru_cache(maxsize=1)
def _get_model():
    """
    Lazily download (first run only, via huggingface_hub) + load the GGUF
    model. Cached so repeated calls in the same process don't reload
    weights from disk each time.
    """
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    model_path = hf_hub_download(repo_id=GGUF_REPO_ID, filename=GGUF_FILENAME)
    return Llama(model_path=model_path, n_ctx=N_CTX, n_threads=N_THREADS, verbose=False)


def generate(query: str, context: str) -> str:
    llm = _get_model()

    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nContext:\n{context}\n\nQuestion: {query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    output = llm(
        prompt,
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        stop=["<|im_end|>"],
    )
    return output["choices"][0]["text"].strip()


if __name__ == "__main__":
    print(generate(
        "What is IDOR and how do you prevent it?",
        context="IDOR (Insecure Direct Object Reference) occurs when an application "
                "exposes a reference to an internal object without proper authorization checks.",
    ))
