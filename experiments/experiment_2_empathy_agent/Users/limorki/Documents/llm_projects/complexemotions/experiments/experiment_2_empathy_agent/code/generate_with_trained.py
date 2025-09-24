import os, csv, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

HERE = os.path.abspath(os.path.dirname(__file__))
BASE = os.path.abspath(os.path.join(HERE, os.pardir))
OUT_DIR = os.path.join(BASE, "output", "trained", "empathetic_action")
TEST_JSON = os.path.join(BASE, "code", "EmpathyAgent", "dataset", "testset_100.json")
ADAPTER_DIR = os.path.join(BASE, "output", "llama3_lora")


def build_prompt(scenario: str, dialogue: str) -> str:
    return (
        f"Character context: {scenario}\n"
        f"User dialogue: {dialogue}\n"
        f"As an empathetic assistant, propose a concise empathetic action plan and a supportive sentence."
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(ADAPTER_DIR, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(ADAPTER_DIR, torch_dtype=torch.float16, device_map="auto")
    model = base  # adapter is merged if saved from Trainer.save_model()

    with open(TEST_JSON, "r", encoding="utf-8") as f:
        test = json.load(f)

    out_csv = os.path.join(OUT_DIR, "trained_llama3_lora_inference.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["data_idx", "response"])
        for idx, item in enumerate(test):
            scenario = item.get("scenario", "")
            dialogue = item.get("dialogue", "")
            prompt = build_prompt(scenario, dialogue)

            input_ids = tok(prompt, return_tensors="pt").to(model.device)
            gen = model.generate(**input_ids, max_new_tokens=200, do_sample=False)
            text = tok.decode(gen[0], skip_special_tokens=True)
            # naive split to keep only assistant tail
            response = text.split(dialogue)[-1].strip() if dialogue else text
            w.writerow([idx, response])

    print("Wrote:", out_csv)


if __name__ == "__main__":
    main()


