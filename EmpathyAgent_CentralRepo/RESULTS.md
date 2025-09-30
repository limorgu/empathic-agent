# Results — EmpathyAgent Replication

System: macOS (Python 3.9 venv), CPU-only  
Base model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`  
Adapter: `output/lora_tinyllama_min/` (LoRA)

---

## A) Empathetic Action (reference-based)

**Scorer:** Overlap (Jaccard), LCS (normalized), TF‑IDF cosine  
**Prompts:** `test_prompts.jsonl` → **Predictions:** `predictions.csv` → **Refs:** `test_refs.jsonl`

| Variant | Samples | Overlap | LCS | TF‑IDF |
|---|---:|---:|---:|---:|
| All reference fields | 100 | **0.3886** | **0.3202** | **0.5912** |
| `empathy_goal_nl` only | 100 | **0.3886** | **0.3202** | **0.5914** |

**Notes:** These are lexical similarity metrics; treat them as relative indicators across your runs. Decoding with `--greedy` typically increases Overlap/LCS vs sampling.

---

## B) Scenario Understanding (semantic)

**Scorer:** BERTScore (`distilroberta-base`)  
**Prompts:** `scenario_prompts.jsonl` → **Predictions:** `predictions.csv` → **Refs:** `scenario_refs.jsonl`

| Samples | BERTScore P | BERTScore R | BERTScore F1 | Model |
|---:|---:|---:|---:|---|
| 100 | **0.8084** | **0.8868** | **0.8458** | distilroberta-base |

**Interpretation:** BERTScore uses contextual embeddings. **P**: how much of the *candidate* is supported by the reference; **R**: how much of the *reference* is covered; **F1** balances both.

---

## Re-running

- See `docs/ACTION_PIPELINE.md` and `docs/SCENARIO_PIPELINE.md` for copy‑pasteable commands.
- To log runs, keep appending to your local `output/run_log.csv` (from the experiment folder).
