# EmpathyAgentReplication_ex1 – Minimal LoRA Training • Inference • Scoring

This README captures the exact steps we used so you can rerun the tiny LoRA pipeline anytime (Mac or Colab). It trains a small LoRA adapter (TinyLlama), runs inference, exports predictions, and scores them (Overlap/LCS/TF‑IDF) against references.

> **Your layout (Mac):**
>
> `~/Documents/to_git/EmpathyAgentReplication/EmpathyAgentReplication_ex1/`
>
> ├─ **Code/**  ← scripts live here
> ├─ **data/**
> └─ **output/**

---

## 0 Quick one‑liners (cheatsheet)

From **Code/** with venv active:

```bash
# verify paths
python cfg_paths.py

# install deps (first run only)
pip install --upgrade pip
pip install torch==2.3.1 transformers==4.44.2 peft==0.13.2 datasets==2.20.0 \
            accelerate==0.33.0 evaluate==0.4.2 tqdm jsonlines scikit-learn

# ensure the mini train file is visible at ./sft_empathyagent_mini.jsonl
ln -sf OriginalPaperEmpathyAgent/dataset/sft_empathyagent_mini.jsonl ./sft_empathyagent_mini.jsonl

# train tiny LoRA (CPU ok)
python training_code/train_lora_min.py \
  --max_steps 20 \
  --output_name lora_tinyllama_min \
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --train_file ./sft_empathyagent_mini.jsonl

# single‑prompt inference
python training_code/infer_lora_min.py \
  --adapter_name lora_tinyllama_min \
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "Parent asks for help: 'My 12-year-old is yelling at his brother and I'm exhausted.' Give 2 kind, concrete steps."

# batch inference (JSONL → inference.jsonl)
python training_code/infer_lora_min.py \
  --adapter_name lora_tinyllama_min \
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompts_file OriginalPaperEmpathyAgent/dataset/sft_empathyagent_mini.jsonl

# export predictions CSV for scoring
python training_code/export_for_scoring.py --adapter_name lora_tinyllama_min

# score vs. references → prints averages, writes scores.csv
python training_code/score_actions_min.py \
  --adapter_name lora_tinyllama_min \
  --references OriginalPaperEmpathyAgent/dataset/sft_empathyagent_mini.jsonl
```

Outputs land in: `output/lora_tinyllama_min/` (adapter, inference.jsonl, predictions.csv, scores.csv).

---

## 1) Environment

### Mac (first time)

1. Open Terminal and go to **Code/**:

   ```bash
   cd ~/Documents/to_git/EmpathyAgentReplication/EmpathyAgentReplication_ex1/Code
   pwd && ls
   ```
2. Create & activate a Python venv:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -V
   ```

### Install required packages (inside venv)

```bash
python -m pip install --upgrade pip
pip install torch==2.3.1 transformers==4.44.2 peft==0.13.2 datasets==2.20.0 \
            accelerate==0.33.0 evaluate==0.4.2 tqdm jsonlines scikit-learn
```

> *LibreSSL warning from urllib3 is harmless on macOS; you can ignore it for this CPU demo.*

---

## 2) Paths (checked into repo)

`Code/cfg_paths.py` — already created. Run to verify:

```bash
python cfg_paths.py
# expects: "✅ Paths ready" and a summary of ROOT/CODE/DATA/OUTPUT
```

---

## 3) Data (mini SFT file)

Your mini file lives at: `Code/OriginalPaperEmpathyAgent/dataset/sft_empathyagent_mini.jsonl`.
Create a local link so scripts can read `./sft_empathyagent_mini.jsonl`:

```bash
ln -sf OriginalPaperEmpathyAgent/dataset/sft_empathyagent_mini.jsonl ./sft_empathyagent_mini.jsonl
ls -l sft_empathyagent_mini.jsonl
```

---

## 4) Train tiny LoRA (CPU)

Script: `Code/training_code/train_lora_min.py`

Run:

```bash
python training_code/train_lora_min.py \
  --max_steps 20 \
  --output_name lora_tinyllama_min \
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --train_file ./sft_empathyagent_mini.jsonl
```

Success message:

```
✅ Saved LoRA adapter to: output/lora_tinyllama_min
```

---

## 5 Inference

Script: `Code/training_code/infer_lora_min.py` (Python 3.9 compatible)

**A Single prompt demo**

```bash
python training_code/infer_lora_min.py \
  --adapter_name lora_tinyllama_min \
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "Parent asks for help: 'My 12-year-old is yelling at his brother and I'm exhausted.' Give 2 kind, concrete steps."
```

**B Batch (JSONL → inference.jsonl)**

```bash
python training_code/infer_lora_min.py \
  --adapter_name lora_tinyllama_min \
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompts_file OriginalPaperEmpathyAgent/dataset/sft_empathyagent_mini.jsonl \
  --max_new_tokens 160 --temperature 0.7
```

Creates: `output/lora_tinyllama_min/inference.jsonl`.

---

## 6) Export predictions for scoring (CSV)

Script: `Code/training_code/export_for_scoring.py`

```bash
python training_code/export_for_scoring.py --adapter_name lora_tinyllama_min
```

Creates: `output/lora_tinyllama_min/predictions.csv` with `id,prediction`.

---

## 7) Score predictions (reference‑based)

Script: `Code/training_code/score_actions_min.py`

```bash
python training_code/score_actions_min.py \
  --adapter_name lora_tinyllama_min \
  --references OriginalPaperEmpathyAgent/dataset/sft_empathyagent_mini.jsonl
```

Prints averages (Overlap/LCS/TF‑IDF) and writes `output/lora_tinyllama_min/scores.csv`.

---

## 8 Colab quick start (mirrors Mac)

1. Mount Drive & `cd` into your project:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/Colab\ Notebooks/empathic-agent/experiments/EmpathyAgentReplication/Code
!python cfg_paths.py
```

2. Install deps and run the **same** commands as on Mac (training/inference/export/score). Output goes to your `output/` under the project.

---

## 9 Common issues & fixes

* **`ModuleNotFoundError: cfg_paths`**
  Ensure `cfg_paths.py` is in **Code/**. In training/inference scripts we add `sys.path.insert(0, str(CODE_DIR))` so importing works.

* **`Training file not found`**
  Create/refresh the link:
  `ln -sf OriginalPaperEmpathyAgent/dataset/sft_empathyagent_mini.jsonl ./sft_empathyagent_mini.jsonl`

* **Python 3.9 typing error (`str | None`)**
  We use a 3.9‑safe version (`Optional[str]`). If you copy code from elsewhere, replace `str | None` with `Optional[str]` and `from typing import Optional`.

* **LibreSSL warning**
  Safe to ignore for this pipeline.

* **Check your venv**
  `echo $VIRTUAL_ENV` and `which python` should point inside `Code/.venv`. If not, activate again: `source .venv/bin/activate`.

---

## 10 Next: “True replication” with paper test sets

When you’re ready:

1. Build prompts from the paper’s `testset_100.json` → `test_prompts.jsonl`.
2. Run batch inference with your adapter on that file.
3. Score against the testset references.

> Ask me for **“make test prompts”** and I’ll drop in the two tiny commands + script to convert the paper’s test file into prompts and run the full scoring.

---

### Files we created (under **Code/**)

* `cfg_paths.py`
* `training_code/train_lora_min.py`
* `training_code/infer_lora_min.py`
* `training_code/export_for_scoring.py`
* `training_code/score_actions_min.py`

### Outputs

* `output/lora_tinyllama_min/` → LoRA adapter + `inference.jsonl` + `predictions.csv` + `scores.csv`

---

**You’re set.** Re‑run the cheatsheet block at the top whenever you want to retrain or evaluate quickly.
