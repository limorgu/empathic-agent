# EmpathyAgent – Replication Results (Action & Scenario)

This README summarizes **what you ran**, **your results**, and **how to re‑run** quickly. It covers the two paper tasks you replicated with our portable LoRA pipeline.

> **System:** macOS (Python 3.9 venv), CPU only
> **Base model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
> **Adapter (LoRA) folder:** `output/lora_tinyllama_min/`
> **Test file:** `Code/OriginalPaperEmpathyAgent/dataset/testset_100.json`

---

## ✅ Results Summary

### A) Empathetic Action (reference‑based)

**Scorer:** Overlap (Jaccard), LCS (normalized), TF‑IDF cosine
**Prompts:** `test_prompts.jsonl`  →  **Predictions:** `predictions.csv`  →  **Refs:** `test_refs.jsonl`

| Variant                | Samples |    Overlap |        LCS |     TF‑IDF |
| ---------------------- | ------: | ---------: | ---------: | ---------: |
| All reference fields   |     100 | **0.3886** | **0.3202** | **0.5912** |
| `empathy_goal_nl` only |     100 | **0.3886** | **0.3202** | **0.5914** |

**Interpretation (short):**

* **Overlap (Jaccard)** — unigram set overlap (0–1). Higher = more shared words.
* **LCS (normalized)** — longest common subsequence relative to length; rewards in‑order matches.
* **TF‑IDF cosine** — cosine similarity in TF‑IDF space; balances shared rare words vs. stopwords.
  Values are sensitive to prompt format and response length; use them for **relative comparisons** across your runs.

### B) Scenario Understanding (semantic)

**Scorer:** BERTScore (`distilroberta-base`) on CPU
**Prompts:** `scenario_prompts.jsonl`  →  **Predictions:** `predictions.csv`  →  **Refs:** `scenario_refs.jsonl`

| Samples | BERTScore P | BERTScore R | BERTScore F1 | Model              |
| ------: | ----------: | ----------: | -----------: | ------------------ |
|     100 |  **0.8084** |  **0.8868** |   **0.8458** | distilroberta-base |

**Interpretation (short):**
BERTScore uses contextual embeddings (no exact match needed). **P** measures how much of the *candidate* is supported by the reference, **R** how much of the *reference* is covered by the candidate, and **F1** balances both.

---

## Re‑run Cheat‑Sheet (from `Code/`, venv active)

```bash
# 0) Activate venv
source .venv/bin/activate

# 1) Verify paths
python cfg_paths.py

# 2) (If needed) Rebuild paper prompts/refs with robust decoder
python training_code/make_test_prompts.py \
  --src OriginalPaperEmpathyAgent/dataset/testset_100.json \
  --out_dir OriginalPaperEmpathyAgent/dataset
ln -sf OriginalPaperEmpathyAgent/dataset/test_prompts.jsonl ./test_prompts.jsonl
ln -sf OriginalPaperEmpathyAgent/dataset/test_refs.jsonl     ./test_refs.jsonl

# 3) Action task — inference → export → score
python training_code/infer_lora_min.py \
  --adapter_name lora_tinyllama_min \
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompts_file ./test_prompts.jsonl \
  --max_new_tokens 160 --temperature 0.7
python training_code/export_for_scoring.py --adapter_name lora_tinyllama_min
python training_code/score_actions_min.py \
  --adapter_name lora_tinyllama_min \
  --references ./test_refs.jsonl

# 4) (Optional) Action task — single gold key
python training_code/score_actions_min.py \
  --adapter_name lora_tinyllama_min \
  --references OriginalPaperEmpathyAgent/dataset/test_refs_empathy_goal_nl.jsonl

# 5) Scenario task — build prompts/refs → inference → export → score (BERTScore)
python - <<'PY'
from pathlib import Path, json
src = Path('OriginalPaperEmpathyAgent/dataset/test_prompts.jsonl')
dst = Path('OriginalPaperEmpathyAgent/dataset/scenario_prompts.jsonl')
with src.open('r', encoding='utf-8', errors='ignore') as f, dst.open('w', encoding='utf-8') as wf:
  for line in f:
    if not line.strip(): continue
    row = json.loads(line); row['instruction'] = 'Summarize the situation in 1–2 sentences focusing on the main goal and constraints.'
    wf.write(json.dumps(row, ensure_ascii=False)+'\n')
print('✅ wrote', dst)
PY
python - <<'PY'
from pathlib import Path, json
src = Path('OriginalPaperEmpathyAgent/dataset/testset_100.json')
dst = Path('OriginalPaperEmpathyAgent/dataset/scenario_refs.jsonl')
data = json.loads(src.read_text(encoding='utf-8', errors='ignore'))
if isinstance(data, dict):
  for v in data.values():
    if isinstance(v, list): data = v; break
with dst.open('w', encoding='utf-8') as f:
  for i, ex in enumerate(data):
    ref = ex.get('high_level_plan','')
    if isinstance(ref, (list, dict)):
      ref = json.dumps(ref, ensure_ascii=False)
    f.write(json.dumps({'id': i, 'reference': ref or ''}, ensure_ascii=False)+'\n')
print('✅ wrote', dst)
PY
python training_code/infer_lora_min.py \
  --adapter_name lora_tinyllama_min \
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompts_file OriginalPaperEmpathyAgent/dataset/scenario_prompts.jsonl \
  --max_new_tokens 120 --temperature 0.6
python training_code/export_for_scoring.py --adapter_name lora_tinyllama_min
python training_code/score_scenario_bertscore.py \
  --adapter_name lora_tinyllama_min \
  --references OriginalPaperEmpathyAgent/dataset/scenario_refs.jsonl \
  --model_type distilroberta-base
```

---

## Where files live

```
output/lora_tinyllama_min/
  ├─ adapter_config.json, adapter_model.safetensors, tokenizer files
  ├─ inference.jsonl
  ├─ predictions.csv
  ├─ scores.csv                    # action task metrics
  └─ scenario_bertscore.csv        # scenario task metrics
```

---

## Tips & Troubleshooting

* **LibreSSL/urllib3 warning** — harmless on macOS; OK to ignore.
* **Silent inference** — use the progress‑print version of `infer_lora_min.py` (flushes every 5 rows). Monitor with:
  `watch -n 5 wc -l output/lora_tinyllama_min/inference.jsonl`
* **zsh: number expected** — happens if you paste printed lines (with `|` pipes) into the shell. Only paste **commands**, not printed logs.

---

## Interpreting the numbers

* Use **Action** metrics (Overlap/LCS/TF‑IDF) for *lexical* similarity checks.
* Use **Scenario** BERTScore F1 for *semantic* similarity; it’s more tolerant of paraphrase.
* When comparing runs, keep **prompts** and **max_new_tokens** consistent; vary **temperature** or **base model** and observe deltas.

---

## Next experiments (optional)

* **Temperature/token sweep**: run a small grid (e.g., `temp ∈ {0.3,0.7,1.0}`, tokens `120/160/200`) and chart TF‑IDF/BERTScore.
* **LoRA depth**: try `r=16` on a small subset to see if scores move.
* **Base model**: swap TinyLlama ↔︎ `Qwen/` or `Phi-2` class models (CPU‑friendly) and re‑run.

---

**Done.** This doc plus the earlier pipeline README gives you a full, repeatable record of your empathetic_action + scenario_understanding replication and results.
That
