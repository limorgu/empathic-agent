# EmpathyAgent — Replication & Comparative Evaluation

> **Goal:** Reproduce key results from *EmpathyAgent: Can Embodied Agents Conduct Empathetic Actions?* and compare them with my own runs and cross-dataset experiments (e.g., CASE).  
> **Paper:** Chen et al., 2025 (arXiv:2503.16545). :contentReference[oaicite:0]{index=0}

---

## Navigation

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Environment & Setup](#environment--setup)
- [Data](#data)
- [Reproduction Pipeline](#reproduction-pipeline)
- [Evaluation & Metrics](#evaluation--metrics)
- [Results: Paper vs. Mine](#results-paper-vs-mine)
- [CASE Cross-Dataset Experiment](#case-crossdataset-experiment)
- [Replication Checklist](#replication-checklist)
- [Roadmap](#roadmap)
- [References](#references)

---

## Overview

**What is EmpathyAgent?**  
A benchmark of **10,000** multimodal simulated home scenarios (VirtualHome) with reference empathetic task plans, designed to test whether embodied agents can **understand needs** and **conduct empathetic actions**. Authors also fine-tune **Llama-3-8B** on this data and report gains vs. baselines. :contentReference[oaicite:1]{index=1}

**My aim:**  
1) Reproduce the baseline/FT evaluation reported in the paper;  
2) Log a transparent pipeline and metrics;  
3) Extend to **CASE** to test generalization and build a comparable evaluation table. :contentReference[oaicite:2]{index=2}

---

## Project Structure

.
├─ empathyagent/ # Fork of authors' repo (code to run baseline/eval)
│ ├─ baseline/ # Inference & scoring scripts
│ ├─ docs/ # Paper notes & run logs
│ └─ ...
├─ complexemotions-eval/ # My evaluation hub (results, tables, notebooks)
│ ├─ experiments/
│ ├─ results/
│ └─ notebooks/
└─ datasets/
├─ empathyagent/ # Data or links as required by the code
└─ case_raw/ # CASE dataset (as submodule or local folder)

markdown
Copy code

- **Repo (mine):** `limorgu/complexemotions-eval` — central place for results, CSVs, comparison tables, and notebooks.  
- **Repo (fork):** `limorgu/empathic-agent` — runnable code adapted from the paper’s repo. (Original: authors’ GitHub linked in the paper.) :contentReference[oaicite:3]{index=3}

---

## Environment & Setup

- **Python**: use a fresh venv/conda.  
- **APIs**: if using hosted LLMs, export `OPENAI_API_KEY` (and `OPENAI_API_BASE` only if you use a non-default endpoint).  
- **GPU**: optional for API-based inference; required if you fine-tune locally.

```bash
# create and activate venv (example)
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r empathyagent/requirements/base.txt
Data
EmpathyAgent data: see the authors’ repo/paper for obtaining data/assets aligned with VirtualHome. 
arXiv
+2
GitHub
+2

CASE dataset (cross-dataset): Springer Nature Figshare “CASE_Dataset-full” (30 participants). Place raw files under datasets/case_raw/ or add as a submodule. 
Figshare

Note: VirtualHome is the simulation environment underlying many embodied agent datasets/tools referenced by EmpathyAgent. 
GitHub
+1

Reproduction Pipeline
Inference

For each task (e.g., scenario_understanding, empathetic_action), run the provided inference.py with the model flag you intend (e.g., gpt-4o or local FT model).

Reference-based scoring

Compute Overlap, LCS, TF-IDF against reference plans.

Reference-free scoring

Use the empathy-process checks described by the authors for appropriateness/alignment (paper’s evaluation suite).

Logging

Save per-item outputs to CSV/JSON and aggregate metrics to a single results_summary.json.

Example (illustrative):

bash
Copy code
cd empathyagent/baseline
python inference.py --model_name gpt-4o --task scenario_understanding --reference_free_eval
python evaluate.py --task scenario_understanding --reference_based
Evaluation & Metrics
Reference-based:

Overlap, LCS, TF-IDF similarity between generated and reference plans.
Reference-free:

Psychology-informed checks of empathetic process/appropriateness as defined by the authors’ suite. 
arXiv

Store outputs into complexemotions-eval/results/<run_id>/ with:

*_inference.csv

*_reference_based_score.csv

*_reference_free_metrics.json

results_summary.json

Results: Paper vs. Mine
Use this training stages comparison template (kept for all future papers):

1) Training setup

Datasets, pipeline steps, LLM/vision model, fine-tuning (if any).

2) Evaluation after training

Task-wise metrics (Overlap, LCS, TF-IDF, ref-free scores), highlight deltas.

3) User replication pipeline

Exactly what I ran; what was missing/changed vs. paper.

4) Comparison table

Task	Paper Baseline	Paper Post-Train	My Replication
Empathetic Action	…	…	…
Scenario Understanding	…	…	…

5) Summary of gap

What matches, what differs, and what’s needed to close the gap.

CASE Cross-Dataset Experiment
Goal: Test how EmpathyAgent-trained or prompted models generalize to CASE data.
Approach:

Map CASE text/labels to closest EmpathyAgent task(s) (focus on planning/understanding).

Run the same inference/evaluation suite (reference-based where applicable; ref-free notes otherwise).

Compare distributional differences and performance deltas.

Reference: CASE dataset landing page. 
Figshare

Replication Checklist
 OPENAI_API_KEY (and OPENAI_API_BASE if custom) exported

 VirtualHome/EmpathyAgent assets available (as required by code) 
GitHub
+1

 inference.py runs for each task with logs

 Reference-based metrics computed and saved

 Reference-free metrics computed and saved

 results_summary.json generated

 Comparison table updated in experiments/*/README.md

Roadmap
 Tighten prompt templates & decoding params for stability

 Add ablations (vision-only / text-only / plan-only)

 Run FT vs. zero-shot comparisons across tasks

 Complete CASE mapping + joint table

 Prepare camera-ready figures & reproducibility bundle

References
EmpathyAgent (paper): arXiv:2503.16545 (authors’ GitHub linked within). 
arXiv
+1

EmpathyAgent (HTML/PDF): for figures and evaluation details. 
arXiv
+1

VirtualHome (env): simulator used widely in embodied AI. 
GitHub
+1

CASE dataset: Springer Nature Figshare page. 
Figshare
