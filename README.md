# EmpathyAgent Replication — Phase 1.
*Replicating EmpathyAgent (ACL 2024) — testing empathetic actions in embodied agents, reproducing baselines, and probing generalization.*
Paper's url: https://arxiv.org/abs/2503.16545 

#Quick Start
clone + setup
git clone https://github.com/<yourname>/empathyagent-replication.git
cd empathyagent-replication
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

add your API key
cp .env.example .env
edit .env and set: OPENAI_API_KEY=...

#Navigation
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


## Experiment Pipeline

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

## EmpathyAgent Paper - details about what they did:
Dataset: EmpathyAgent 10k multimodal (EmpatheticDialogues text + VirtualHome scenarios/videos + annotated empathetic plans).

Tasks: (1) Scenario Understanding → (2) Empathetic Planning → (3) Empathetic Actions.

Models: Baselines (GPT-4/GPT-4o, others), plus Llama-3-8B supervised fine-tuning (LoRA) on ~9k train samples.

Metrics:

Scenario/Planning → BLEU, ROUGE-L, CIDEr, SPICE, BERTScore

Actions → Overlap, LCS, TF-IDF

Reference-free empathy dimensions (psych-based scoring)

## Oirignal Paper Results:
| Task                                   | Baseline (GPT-4o) | Fine-tuned (Llama-3-8B LoRA) | Gain    |
| -------------------------------------- | ----------------: | ---------------------------: | ------- |
| **Scenario Understanding** (BERTScore) |             ~0.62 |                        ~0.65 | modest  |
| **Empathetic Actions – Overlap**       |              ~27% |                         ~56% | **≈2×** |
| **Empathetic Actions – LCS**           |              ~24% |                         ~52% | ≈2×     |
| **Empathetic Actions – TF-IDF**        |              ~21% |                         ~47% | ≈2×     |

My Replication — Phase 1 (Baseline)

Scenario Understanding: BERTScore = 0.619 (matches paper baseline)

Empathetic Actions:

Overlap = 27.7%, LCS = 24%, TF-IDF = 21% (matches paper GPT-4 baseline)

Conclusion: Baseline replication is faithful; pipeline and scoring are correct.

## How to run baseline sampele=100
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

##Navigation

Code: scripts/

Mini datasets: data/

Results (CSV): results/

Notebook (tables/plots): notebooks/analysis.ipynb

Notes: docs/discussion.md
 • docs/insights.md

 ##Phase Roadmap

Phase 1 — Baseline (done): Reproduced GPT-4 baseline; verified metrics and pipeline.
Phase 2 — Cross-domain transfer: Compare fine-tuned vs GPT-4 on CASE slice; add Distinct-n, Self-BLEU to measure rigidity.
Phase 3 — Multi-reference stress test: 3 gold plans per scenario; report Coverage@k, best-of-k BERTScore/LCS/TF-IDF + human spot-checks.
Phase 4 — Analysis & write-up: Consolidate results into plots/tables and a short report.

##License & Attribution

Underlying datasets belong to their respective owners; do not redistribute large copies here.



