# Empathic Agent 
*A living page tracking replication progress, metrics, and takeaways.*

## Experiment 2 · EmpatheticDialogues (baseline)
**Status:** running  
**Model:** GPT-4o (API)  
**Task:** Empathetic Action  
**Metrics:** Overlap · LCS · TF-IDF

**Early notes (numbers coming):**
- Text-similarity metrics are modest and clarifying.
- Baseline before multimodal (CASE) comparison.

**Figures (add when ready):**
![Exp2 Chart](assets/exp2_chart.png)
![Exp2 Table](assets/exp2_table.png)

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
