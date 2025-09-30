"""
Minimal LoRA SFT on CPU (TinyLlama) using your local paths.
- Reads a small JSONL SFT file (tries common key names)
- Trains a tiny LoRA for a few steps
- Saves adapter under OUTPUT_DIR/<output_name>
Run:
  python training_code/train_lora_min.py --max_steps 30 --output_name lora_tinyllama_min \
         --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
         --train_file ./sft_empathyagent_mini.jsonl
"""
from pathlib import Path
import argparse, json

from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model, TaskType

# Make the parent "Code" folder importable so we can find cfg_paths.py
from pathlib import Path
import sys
THIS_DIR = Path(__file__).resolve().parent
CODE_DIR = THIS_DIR.parent          # .../EmpathyAgentReplication_ex1/Code
sys.path.insert(0, str(CODE_DIR))

import cfg_paths as P

def guess_text(example):
    # Try common SFT schemas
    if "messages" in example and isinstance(example["messages"], list):
        # join chat into a single training text
        parts = []
        for m in example["messages"]:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"{role.upper()}: {content}")
        return "\n".join(parts) + "\nASSISTANT:"
    instr = example.get("instruction") or example.get("prompt") or example.get("source") or ""
    inp   = example.get("input") or ""
    out   = example.get("output") or example.get("response") or example.get("target") or ""
    if inp and instr:
        return f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
    elif instr:
        return f"### Instruction:\n{instr}\n\n### Response:\n{out}"
    # fallback: if a single 'text' key exists
    if "text" in example:
        return str(example["text"])
    # last resort: dump the row
    return json.dumps(example)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--train_file", default=str((P.CODE_DIR/"sft_empathyagent_mini.jsonl").resolve()))
    ap.add_argument("--output_name", default="lora_tinyllama_min")
    ap.add_argument("--max_steps", type=int, default=30)
    ap.add_argument("--max_length", type=int, default=768)
    args = ap.parse_args()

    # Paths
    P.ensure_dirs()
    out_dir = (P.OUTPUT_DIR / args.output_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Model & tokenizer (CPU)
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.base_model)

    # LoRA config — tiny & fast
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], # works for most Llama-like models
        bias="none"
    )
    model = get_peft_model(model, lora_cfg)

    # Data
    train_path = Path(args.train_file)
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    ds = load_dataset("json", data_files=str(train_path), split="train")

    def to_text(ex):
        txt = guess_text(ex)
        return {"text": txt}

    def tokenize(ex):
        enc = tok(
            ex["text"], truncation=True, max_length=args.max_length,
            padding="max_length", return_tensors=None
        )
        # causal LM: labels are input_ids
        enc["labels"] = enc["input_ids"].copy()
        return enc

    ds = ds.map(to_text, remove_columns=ds.column_names)
    ds = ds.map(tokenize, batched=False)

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # Training
    targs = TrainingArguments(
        output_dir=str(out_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        logging_steps=5,
        save_steps=args.max_steps,
        bf16=False, fp16=False,  # CPU
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
        tokenizer=tok,
    )

    trainer.train()

    # Save LoRA adapter
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"✅ Saved LoRA adapter to: {out_dir}")

if __name__ == "__main__":
    main()
