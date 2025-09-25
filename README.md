# Empathic Agent 
*A living page tracking replication progress, metrics, and takeaways.*

# Empathic Agent — Replication & Extensions

Small, reproducible experiments for **EmpathyAgent** replication and multimodal extensions (e.g., **CASE**). Public code & docs — private data and large artifacts are kept out of Git.

## Why this repo?
- **Quick start** for anyone to run the baseline and see results.
- **Transparent pipeline**: from data → inference → evaluation → results.
- **Learning-in-public**: we post small updates as we go.


## Experiment 2 · EmpatheticDialogues (baseline)
**Status:** running  
**Model:** GPT-4o (API)  
**Task:** Empathetic Action  
**Metrics:** Overlap · LCS · TF-IDF

**Early notes (numbers coming):**
- Text-similarity metrics are modest and clarifying.
- Baseline before multimodal (CASE) comparison.




## Experiment 3 · CASE comparison (planned)
Compare ED vs. CASE (ECG, GSR, respiration).  
**Hypothesis:** physiological signals will shift evaluation.

## Methods (quick)
- Code: EmpathyAgent/baseline/inference.py
- Run: python inference.py --model_name gpt-4o --task empathetic_action --reference_free_eval
- Artifacts: output/empathetic_action/*csv, cache/*json

- Experiment 2 baseline code (EmpathyAgent) + training scripts.
- How to run (Colab):
  1) Upload experiment2_essentials.zip and run_colab.py
  2) Install deps, HF login, then: %run /content/run_colab.py
Outputs: experiments/experiment_2_empathy_agent/output/
