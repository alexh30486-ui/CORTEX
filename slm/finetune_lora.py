"""
LoRA fine-tuning on CPU.

Honest expectation-setting: CPU LoRA training on a 1.5B model with a few
thousand examples will take HOURS, not minutes. This is fine for a
portfolio project (kick it off overnight) but is the one place in this
repo where "local" has a real cost. If it's ever too slow to iterate on,
the fallback is a single Colab session just for this script — everything
else (serving, ingestion, retrieval, agent, security) stays fully local
either way.

Expects a JSONL dataset at slm/data/appsec_train.jsonl with the shape:
    {"instruction": "...", "input": "...", "output": "..."}
one record per line. See slm/data/README.md for how to build this from
the Sentry API / Warlock docs + OWASP references.
"""
from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from slm.config import BASE_MODEL_ID, LORA_OUTPUT_DIR

DATA_PATH = "slm/data/appsec_train.jsonl"


def load_dataset(path: str = DATA_PATH) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def format_example(example: dict, tokenizer) -> dict:
    prompt = (
        f"<|im_start|>system\nYou are a focused application security assistant.<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction']}\n{example.get('input', '')}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['output']}<|im_end|>"
    )
    tokenized = tokenizer(prompt, truncation=True, max_length=1024, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype="auto")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # standard for Qwen2 arch
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # sanity check — should be a small % of total params

    dataset = load_dataset()
    tokenized_dataset = dataset.map(lambda ex: format_example(ex, tokenizer), remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=LORA_OUTPUT_DIR,
        per_device_train_batch_size=1,       # keep small — CPU memory is the constraint
        gradient_accumulation_steps=8,       # simulate a larger effective batch size
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        no_cuda=True,                        # explicit: force CPU path
        fp16=False,                          # fp16 needs GPU; stay in fp32 on CPU
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=collator,
    )

    trainer.train()
    model.save_pretrained(LORA_OUTPUT_DIR)
    tokenizer.save_pretrained(LORA_OUTPUT_DIR)
    print(f"LoRA adapter saved to {LORA_OUTPUT_DIR}")


if __name__ == "__main__":
    Path(LORA_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    main()
