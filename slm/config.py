"""
Central config for model choice. Change here, not scattered across files.

Model choice rationale for CPU-only local training:
  - Qwen2.5-1.5B-Instruct: best accuracy/size tradeoff we've found for
    CPU LoRA fine-tuning in a reasonable timeframe (hours, not days, for
    a few thousand examples on a modern laptop CPU).
  - If your machine really struggles, drop to Qwen2.5-0.5B-Instruct —
    same code path, just swap BASE_MODEL_ID.
  - GGUF quantized version (Q4_K_M) is what actually serves inference;
    the full-precision HF version is only needed during LoRA training.
"""

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"          # HF hub id, used for LoRA training
GGUF_REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"       # pre-quantized, used for serving
GGUF_FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

LORA_OUTPUT_DIR = "slm/lora_adapters/appsec_v1"
MERGED_MODEL_DIR = "slm/merged_models/appsec_v1"

# llama.cpp serving params — conservative defaults for CPU
N_CTX = 4096
N_THREADS = 8          # set to your actual core count
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.2      # low temp: this is a technical Q&A assistant, not creative writing
