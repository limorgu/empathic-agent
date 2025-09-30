# Minimal LoRA inference on CPU (Python 3.9 compatible) with --greedy support
from pathlib import Path
from typing import Optional
import sys, argparse, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# make Code/ importable (so cfg_paths.py is found)
THIS_DIR = Path(__file__).resolve().parent
CODE_DIR = THIS_DIR.parent
sys.path.insert(0, str(CODE_DIR))
import cfg_paths as P  # uses OUTPUT_DIR where the adapter was saved


def load_model(base_model: str, adapter_dir: Path):
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(base_model)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    return tok, model


def fmt(instr: str, inp: Optional[str] = None):
    if inp:
        return f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n"
    return f"### Instruction:\n{instr}\n\n### Response:\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--adapter_name", default="lora_tinyllama_min")
    ap.add_argument("--prompt", default=None)        # single prompt string
    ap.add_argument("--prompts_file", default=None)  # optional JSONL with {instruction,input}
    ap.add_argument("--max_new_tokens", type=int, default=150)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--greedy", action="store_true",
                    help="Deterministic decoding (do_sample=False). Use this instead of temperature=0.")
    args = ap.parse_args()

    # Paths & model
    adapter_dir = P.OUTPUT_DIR / args.adapter_name
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_dir}")
    tok, model = load_model(args.base_model, adapter_dir)

    # Local helper (captures tok/model/args)
    def generate_one(prompt: str) -> str:
        inputs = tok(prompt, return_tensors="pt")

        # Greedy if --greedy or temperature <= 0; else sampling
        use_sampling = (not args.greedy) and (args.temperature is not None and args.temperature > 0)

        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=use_sampling,
            pad_token_id=tok.eos_token_id,
        )
        if use_sampling:
            gen_kwargs.update(
                temperature=args.temperature,
                top_p=0.95,
                repetition_penalty=1.1,
            )
        else:
            # deterministic greedy
            gen_kwargs.update(num_beams=1)

        with torch.no_grad():
            out_ids = model.generate(**inputs, **gen_kwargs)
        return tok.decode(out_ids[0], skip_special_tokens=True)

    # Single prompt mode
    if args.prompt:
        prompt = args.prompt if "### Instruction" in args.prompt else fmt(args.prompt)
        print(generate_one(prompt))
        return

    # Batch mode from JSONL
    if args.prompts_file:
        out_path = adapter_dir / "inference.jsonl"

        # count total lines for nicer logs
        with open(args.prompts_file, "r", encoding="utf-8", errors="ignore") as f:
            total = sum(1 for line in f if line.strip())

        n = 0
        with open(args.prompts_file, "r", encoding="utf-8", errors="ignore") as f, \
             open(out_path, "w", encoding="utf-8") as wf:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt = fmt(row.get("instruction") or row.get("prompt") or "Respond empathetically.",
                             row.get("input"))
                gen = generate_one(prompt)
                json.dump({"prompt": prompt, "output": gen}, wf, ensure_ascii=False)
                wf.write("\n")
                wf.flush()  # so you can watch file grow

                n += 1
                if n % 5 == 0 or n == total:
                    print(f"[infer] {n}/{total} done → {out_path}")

        print(f"✅ wrote {n} generations to {out_path}")
        return

    # Default demo
    demo = fmt("You are an empathetic assistant. A teen is overwhelmed after a fight with a sibling. "
               "Offer a brief plan with 2 concrete, kind actions the caregiver can take right now.")
    print(generate_one(demo))


if __name__ == "__main__":
    main()
