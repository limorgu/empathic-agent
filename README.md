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

# Project Structure


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

### Setup / Environment

```bash
# create and activate venv (example)
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r empathyagent/requirements/base.txt
'''


epo Structure
empathyagent-replication/
│
├── README.md
├── requirements.txt
├── .env.example
│
├── data/
│   ├── testset_100.json           # EmpathyAgent 100-sample subset (small)
│   └── case_test_subset.csv       # (optional) CASE mini slice for transfer
│
├── results/                       # Curated outputs (100-sample runs)
│   ├── gpt-4o_inference.csv
│   ├── gpt-4o_reference_based_score.csv
│   └── gpt-4o_reference_free_score.csv
│
├── scripts/                       # Clean subset of runnable code
│   ├── inference.py
│   ├── overlap.py                 # Overlap / LCS / TF-IDF
│   ├── NLG_metric.py              # BLEU / ROUGE / CIDEr / BERTScore
│   └── ea_runner.py               # (optional) one-button runner
│
├── notebooks/
│   └── analysis.ipynb             # Plots & comparison tables
│
└── docs/
    ├── discussion.md              # What worked / didn’t (short notes)
    └── insights.md                # Key takeaways & next questions


✅ Include small sample CSVs (100 items), code you actually ran, and an analysis notebook.
🚫 Do not include big datasets, raw API dumps, or your real .env.

🔄 Experiment Pipeline
flowchart TD
    A[Inputs] --> B[Tasks]
    B --> C[Metrics]
    C --> D[Results]

    A -->|Background + Scenario Video + Dialogue| B

    B -->|1. Scenario Understanding| C
    B -->|2. Empathetic Planning| C
    B -->|3. Empathetic Actions| C

    C -->|NLG: BLEU, ROUGE, CIDEr, SPICE, BERTScore| D
    C -->|Actions: Overlap, LCS, TF-IDF| D
    C -->|Ref-free Empathy Dims| D

    D -->|Paper Baseline (GPT-4o)| E1[~27% Overlap; BERTScore ~0.62]
    D -->|Fine-tuned (Llama-3-8B LoRA)| E2[~56% Overlap; modest NLG gains]
    D -->|My Replication (GPT-4)| E3[Matched baseline]


Key insight (TL;DR): Fine-tuning with LoRA doubled action alignment by learning the dataset’s structured action style; generalization beyond that narrow format is uncertain.

🧩 Paper Setup (what they did)

Dataset: EmpathyAgent 10k multimodal (EmpatheticDialogues text + VirtualHome scenarios/videos + annotated empathetic plans).

Tasks: (1) Scenario Understanding → (2) Empathetic Planning → (3) Empathetic Actions.

Models: Baselines (GPT-4/GPT-4o, others), plus Llama-3-8B supervised fine-tuning (LoRA) on ~9k train samples.

Metrics:

Scenario/Planning → BLEU, ROUGE-L, CIDEr, SPICE, BERTScore

Actions → Overlap, LCS, TF-IDF

Reference-free empathy dimensions (psych-based scoring)

📊 Paper Results (baseline → post-training)
Task	Baseline (GPT-4o)	Fine-tuned (Llama-3-8B LoRA)	Gain
Scenario Understanding (BERTScore)	~0.62	~0.65	modest
Empathetic Actions – Overlap	~27%	~56%	≈2×
Empathetic Actions – LCS	~24%	~52%	≈2×
Empathetic Actions – TF-IDF	~21%	~47%	≈2×

Biggest jump: Action alignment metrics (Overlap/LCS/TF-IDF).

📊 My Replication — Phase 1 (Baseline)

Scenario Understanding: BERTScore = 0.619 (matches paper baseline)

Empathetic Actions:

Overlap = 27.7%, LCS = 24%, TF-IDF = 21% (matches paper GPT-4 baseline)

✅ Conclusion: Baseline replication is faithful; pipeline and scoring are correct.

📝 What the fine-tuning changed (why scores jumped)

Method: Supervised instruction tuning with LoRA adapters (small trainable matrices inside attention/FFN).

Effect: Biases the model to produce structured empathetic plans in the dataset’s style → big gains in format-sensitive metrics (Overlap/LCS/TF-IDF).

Caveat: Since training targets one gold plan per scenario, the learned behavior can be rigid and may not capture the multi-path nature of real empathy.

#How to Run (Baseline, 100-sample)
1) activate env
source .venv/bin/activate

 2) run baseline
cd scripts
python inference.py --model_name gpt-4o --task empathetic_action --reference_free_eval

 3) see outputs
ls ../results
gpt-4o_inference.csv
gpt-4o_reference_based_score.csv
gpt-4o_reference_free_score.csv

# Navigation

Code: scripts/

Mini datasets: data/

Results (CSV): results/

Notebook (tables/plots): notebooks/analysis.ipynb

Notes: docs/discussion.md
 • docs/insights.md

Tip: Keep CSVs small (100 items). Link out to large datasets instead of committing them.

#Phase Roadmap

Phase 1 — Baseline (done): Reproduced GPT-4 baseline; verified metrics and pipeline.
Phase 2 — Cross-domain transfer: Compare fine-tuned vs GPT-4 on CASE slice; add Distinct-n, Self-BLEU to measure rigidity.
Phase 3 — Multi-reference stress test: 3 gold plans per scenario; report Coverage@k, best-of-k BERTScore/LCS/TF-IDF + human spot-checks.
Phase 4 — Analysis & write-up: Consolidate results into plots/tables and a short report.

# License & Attribution

Code in this repo: your chosen license (e.g., MIT).

Underlying datasets belong to their respective owners; do not redistribute large copies here.
