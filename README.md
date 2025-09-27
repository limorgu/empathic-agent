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

### Setup / Environment

```bash
# create and activate venv (example)
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r empathyagent/requirements/base.txt
'''
