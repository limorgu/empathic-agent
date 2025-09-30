# Results — Experiment 2 (EmpatheticDialogues)

**Baseline replication** of EmpathyAgent on *EmpatheticDialogues*.

## Models & Settings
| Model       | Provider | Modality                  | Inference | Notes                                       |
|-------------|----------|---------------------------|-----------|---------------------------------------------|
| GPT-4o      | OpenAI   | text+vision (used as text)| API       | Baseline replication on EmpatheticDialogues |
| GPT-4o-mini | OpenAI   | text                      | API       | Sanity / lower-cost ablation (optional)     |

- **Tasks:** Scenario Understanding (BERTScore), Empathetic Actions (Overlap, LCS, TF-IDF)  
- **Dataset:** EmpatheticDialogues-derived test subset (paper testmini=100)  
- **Command:**  
  ```bash
  # from EmpathyAgent/baseline
  python inference.py --model_name gpt-4o --task empathetic_action --reference_free_eval
  ```

---

## Scenario Understanding (BERTScore)
**BERTScore (GPT-4o):** `0.619`

---

## Empathetic Actions — Reference-Based (Our Replication)

| Metric   |   Score |
|:---------|--------:|
| Overlap  |   0.277 |
| LCS      |   0.24  |
| TF-IDF   |   0.209 |

**Figure**  
![Our Actions Metrics](./assets/exp2_actions_bar.png)

---

## Authors vs Replication (Actions)

| Metric   |   Authors (GPT-4o) |   Replication (GPT-4o) |   Δ (abs) |
|:---------|-------------------:|-----------------------:|----------:|
| Overlap  |             0.276  |                 0.277  |     0.001 |
| LCS      |             0.2517 |                 0.24   |    -0.012 |
| TF-IDF   |             0.2103 |                 0.2095 |    -0.001 |

**Figure**  
![Authors vs Replication](./assets/exp2_vs_paper_bar.png)

> Scores are **very close** to the authors’ GPT‑4o baseline; small deltas are expected from subset sampling and prompt/version drift.

---

## Inference Stats (Run Summary)

| task                   |   total_rows |   non_empty_responses |   avg_response_chars |   min_response_chars |   max_response_chars |   median_response_chars |   p95_response_chars |
|:-----------------------|-------------:|----------------------:|---------------------:|---------------------:|---------------------:|------------------------:|---------------------:|
| empathetic_action      |          100 |                   100 |                196   |                  105 |                  292 |                   193.5 |               255.05 |
| scenario_understanding |          100 |                   100 |                163.8 |                   34 |                  278 |                   165   |               215.35 |

---

## One-Line Insight
Text-similarity metrics are **modest** and useful as a **baseline**; they don’t capture embodied nuance — we’ll probe that with multimodal signals next.

## Reproduce
```bash
# from EmpathyAgent/baseline
python inference.py --model_name gpt-4o --task empathetic_action --reference_free_eval
```

Artifacts are written under `EmpathyAgent/output/empathetic_action/`.  
Only small shareable files are copied to `results/` for publication.

## Next
Extend to **CASE** (ECG/GSR/respiration) and compare against this baseline.

_Last updated: 2025-09-28_