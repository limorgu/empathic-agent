# EmpathyAgent – **Empathetic Action** Task: End‑to‑End Replication (LoRA → Inference → Scoring)

This README captures the exact, repeatable pipeline you just ran for the **empathetic_action** task using a small **LoRA** adapter on **TinyLlama**. It assumes your project layout on Mac and mirrors Colab when needed.

> **Project root (Mac):** `~/Documents/to_git/EmpathyAgentReplication/EmpathyAgentReplication_ex1/`
>
> ├─ **Code/** (scripts live here)
> ├─ **data/**
> └─ **output/**

---Phases:

## 0) One‑shot cheatsheet (copy/paste from `Code/`)

```bash
# (first time) create venv & install deps
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.3.1 transformers==4.44.2 peft==0.13.2 datasets==2.20.0 \
            accelerate==0.33.0 evaluate==0.4.2 tqdm jsonlines scikit-learn

# verify paths
python cfg_paths.py

# ensure mini SFT file is visible for quick sanity runs
ln -sf OriginalPaperEmpathyAgent/dataset/sft_empathyagent_mini.jsonl ./sft_empathyagent_mini.jsonl

# train tiny LoRA (CPU OK)
python training_code/train_lora_min.py \
  --max_steps 20 \
  --output_name lora_tinyllama_min \
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --train_file ./sft_empathyagent_mini.jsonl

# make *paper* test prompts + refs (robust UTF‑8 handling)
python training_code/make_test_prompts.py \
  --src OriginalPaperEmpathyAgent/dataset/testset_100.json \
  --out_dir OriginalPaperEmpathyAgent/dataset
ln -sf OriginalPaperEmpathyAgent/dataset/test_prompts.jsonl ./test_prompts.jsonl
ln -sf OriginalPaperEmpathyAgent/dataset/test_refs.jsonl     ./test_refs.jsonl

# inference on *paper prompts*
python training_code/infer_lora_min.py \
  --adapter_name lora_tinyllama_min \
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompts_file ./test_prompts.jsonl \
  --max_new_tokens 160 --temperature 0.7

# export predictions for scoring
python training_code/export_for_scoring.py --adapter_name lora_tinyllama_min

# score (reference‑based: Overlap/LCS/TF‑IDF)
python training_code/score_actions_min.py \
  --adapter_name lora_tinyllama_min \
  --references ./test_refs.jsonl

# (optional) score against a single gold key (empathy_goal_nl)
python - <<'PY'
from pathlib import Path
import json
src = Path("OriginalPaperEmpathyAgent/dataset/testset_100.json")
dst = Path("OriginalPaperEmpathyAgent/dataset/test_refs_empathy_goal_nl.jsonl")
data = json.loads(src.read_text(encoding="utf-8", errors="ignore"))
if isinstance(data, dict):
    for v in data.values():
        if isinstance(v, list):
            data = v; break
with dst.open("w", encoding="utf-8") as f:
    for i, ex in enumerate(data):
        ref = ex.get("empathy_goal_nl", "")
        if isinstance(ref, (list, dict)):
            ref = json.dumps(ref, ensure_ascii=False)
        f.write(json.dumps({"id": i, "reference": (ref or "")}, ensure_ascii=False) + "\n")
print(f"✅ wrote {dst}")
PY

python training_code/score_actions_min.py \
  --adapter_name lora_tinyllama_min \
  --references OriginalPaperEmpathyAgent/dataset/test_refs_empathy_goal_nl.jsonl

# (optional) log your run to output/run_log.csv
python training_code/log_run.py \
  --adapter_name lora_tinyllama_min \
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_tag testset_100 \
  --tokens 160 --temp 0.7 \
  --notes "empathetic_action full pass"
```

---

## 1) Files this pipeline uses

**In `Code/`:**

* `cfg_paths.py` – path resolver; ensures `output/` exists.
* `training_code/train_lora_min.py` – minimal LoRA SFT trainer (TinyLlama).
* `training_code/infer_lora_min.py` – inference for single prompt or JSONL (progress prints when enabled).
* `training_code/export_for_scoring.py` – `inference.jsonl` → `predictions.csv`.
* `training_code/score_actions_min.py` – reference‑based scorer (Overlap/LCS/TF‑IDF).
* `training_code/make_test_prompts.py` – converts `testset_100.json` → `test_prompts.jsonl` + `test_refs.jsonl` (robust decoding + schema extraction incl. `empathy_goal_nl`, `high_level_plan`).
* `training_code/log_run.py` – appends summary row to `output/run_log.csv`.

**Paper files (kept under `Code/OriginalPaperEmpathyAgent/dataset/`):**

* `testset_100.json` (provided by paper)
* `sft_empathyagent_mini.jsonl` (your tiny SFT sanity file)
* `test_prompts.jsonl` + `test_refs.jsonl` (generated)

**Outputs (under `output/lora_tinyllama_min/`):**

* LoRA adapter (PEFT) + tokenizer copy
* `inference.jsonl` → `predictions.csv` → `scores.csv`

---

## 2) Step‑by‑step (explainers)

### A) Environment

```bash
cd ~/Documents/to_git/EmpathyAgentReplication/EmpathyAgentReplication_ex1/Code
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.3.1 transformers==4.44.2 peft==0.13.2 datasets==2.20.0 \
            accelerate==0.33.0 evaluate==0.4.2 tqdm jsonlines scikit-learn
```

### B) Paths & data links

```bash
python cfg_paths.py
ln -sf OriginalPaperEmpathyAgent/dataset/sft_empathyagent_mini.jsonl ./sft_empathyagent_mini.jsonl
```

### C) Train LoRA

* CPU‑friendly TinyLlama + LoRA (r=8) for a few steps just to validate training end‑to‑end.

```bash
python training_code/train_lora_min.py \
  --max_steps 20 \
  --output_name lora_tinyllama_min \
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --train_file ./sft_empathyagent_mini.jsonl
```

### D) Build paper prompts + refs

* Converts `testset_100.json` to `.jsonl` prompts for our inference runner.

```bash
python training_code/make_test_prompts.py \
  --src OriginalPaperEmpathyAgent/dataset/testset_100.json \
  --out_dir OriginalPaperEmpathyAgent/dataset
ln -sf OriginalPaperEmpathyAgent/dataset/test_prompts.jsonl ./test_prompts.jsonl
ln -sf OriginalPaperEmpathyAgent/dataset/test_refs.jsonl     ./test_refs.jsonl
```

### E) Inference → Export → Score

```bash
python training_code/infer_lora_min.py \
  --adapter_name lora_tinyllama_min \
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompts_file ./test_prompts.jsonl \
  --max_new_tokens 160 --temperature 0.7

python training_code/export_for_scoring.py --adapter_name lora_tinyllama_min

python training_code/score_actions_min.py \
  --adapter_name lora_tinyllama_min \
  --references ./test_refs.jsonl
```

### F) (Optional) Single‑key gold (apples‑to‑apples)

* Score specifically against `empathy_goal_nl` extracted from the paper JSON.

```bash
python training_code/score_actions_min.py \
  --adapter_name lora_tinyllama_min \
  --references OriginalPaperEmpathyAgent/dataset/test_refs_empathy_goal_nl.jsonl
```

### G) (Optional) Log the run

```bash
python training_code/log_run.py \
  --adapter_name lora_tinyllama_min \
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_tag testset_100 \
  --tokens 160 --temp 0.7 \
  --notes "empathetic_action full pass"
```

---

## 3) Troubleshooting

* **`ModuleNotFoundError: cfg_paths`** – ensure `cfg_paths.py` is in `Code/` and that our scripts insert `CODE_DIR` to `sys.path`.
* **`Training file not found`** – re‑link `sft_empathyagent_mini.jsonl` or point `--train_file` to the correct path.
* **Prompts look empty** – converter now falls back to include **all non‑reference** fields as `input`.
* **Refs look empty** – converter now extracts nested keys including `empathy_goal_nl` and `high_level_plan`.
* **Run seems idle** – second terminal: `wc -l output/lora_tinyllama_min/inference.jsonl` in a loop to confirm progress; TinyLlama HF cache may download on first use.
* **LibreSSL/urllib3 warning** – harmless for this workflow.

---

## 4) What’s *not* needed for this task

* **Videos/frames** from the larger embodied setup are **not required** for `empathetic_action` reference‑based scoring.

---

## 5) Next steps

* Temperature/token sweeps (`--temperature`, `--max_new_tokens`) and alternative base models (e.g., `distil`‑class) to explore score sensitivity.
* Switch to the paper’s **scenario_understanding** task using **BERTScore** (see separate README we can add).
* Swap in OpenAI or other API backends if desired by adapting the inference step (kept modular by design here).

---

