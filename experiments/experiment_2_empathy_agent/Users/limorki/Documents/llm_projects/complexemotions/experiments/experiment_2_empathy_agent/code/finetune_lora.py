import os, json
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType


@dataclass
class Config:
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    # derive relative paths
    here: str = os.path.abspath(os.path.dirname(__file__))
    base: str = os.path.abspath(os.path.join(here, os.pardir))
    data_path: str = os.path.join(base, "data", "sft_empathyagent.jsonl")
    output_dir: str = os.path.join(base, "output", "llama3_lora")
    lr: float = 2e-5
    epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    cutoff_len: int = 1024
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def format_examples(records: List[Dict], tokenizer: AutoTokenizer, cutoff: int):
    # Build plain text prompts for SFT (chat-style flattened)
    prompts = []
    for rec in records:
        msgs = rec.get("messages", [])
        parts = []
        for m in msgs:
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                parts.append(f"<|user|>\n{content}\n")
            else:
                parts.append(f"<|assistant|>\n{content}\n")
        text = "".join(parts) + "<|end|>\n"
        enc = tokenizer(text, truncation=True, max_length=cutoff)
        prompts.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
    return prompts


def main(cfg=Config()):
    os.makedirs(cfg.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    peft_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base, peft_cfg)

    # Load jsonl to memory (small subset to start)
    records = list(load_jsonl(cfg.data_path))
    ds = {
        "train": format_examples(records, tokenizer, cfg.cutoff_len)
    }

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=not torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print("Saved LoRA adapter to:", cfg.output_dir)


if __name__ == "__main__":
    main()


