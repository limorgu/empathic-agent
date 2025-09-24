import os
import csv
import json
from collections import Counter
from typing import Dict, List

try:
    import pandas as pd
except Exception:
    pd = None  # graceful fallback if pandas missing

try:
    import numpy as np
except Exception:
    np = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

BASE = "/Users/limorki/Documents/llm_projects/complexemotions/experiments/experiment_2_empathy_agent"

EA_OUT = os.path.join(BASE, "output", "empathetic_action")
EA_OUT_TRAINED = os.path.join(BASE, "output", "trained", "empathetic_action")
SU_OUT = os.path.join(BASE, "output", "scenario_understanding")


def read_csv(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def summarize_inference(rows):
    n = len(rows)
    non_empty = sum(1 for r in rows if (r.get("response") or "").strip())
    lengths = [len((r.get("response") or "")) for r in rows]
    avg_len = round(sum(lengths) / n, 1) if n else 0
    stats = {
        "total_rows": n,
        "non_empty_responses": non_empty,
        "avg_response_chars": avg_len,
        "min_response_chars": min(lengths) if lengths else 0,
        "max_response_chars": max(lengths) if lengths else 0,
    }
    if np is not None and lengths:
        stats["median_response_chars"] = float(np.median(lengths))
        stats["p95_response_chars"] = float(np.percentile(lengths, 95))
    return stats


def read_ref_based_score(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            # may be CSV with two columns [Metric, Score]
            reader = csv.reader(f)
            rows = list(reader)
            if rows and rows[0] and rows[0][0].lower().startswith("metric"):
                return {r[0]: r[1] for r in rows[1:] if len(r) >= 2}
    except Exception:
        pass
    return {}


def main():
    report = {}

    # Empathetic Action (baseline)
    ea_inf = read_csv(os.path.join(EA_OUT, "gpt-4o_inference.csv")) or \
             read_csv(os.path.join(EA_OUT, "gpt-4o-mini_inference.csv"))
    report["empathetic_action_inference"] = summarize_inference(ea_inf)
    report["empathetic_action_ref_based"] = read_ref_based_score(
        os.path.join(EA_OUT, "gpt-4o_reference_based_score.csv")
    ) or read_ref_based_score(
        os.path.join(EA_OUT, "gpt-4o-mini_reference_based_score.csv")
    )

    # Empathetic Action (trained)
    ea_inf_tr = read_csv(os.path.join(EA_OUT_TRAINED, "trained_llama3_lora_inference.csv"))
    if ea_inf_tr:
        report["empathetic_action_inference_trained"] = summarize_inference(ea_inf_tr)

    # Scenario Understanding
    su_inf = read_csv(os.path.join(SU_OUT, "gpt-4o_inference.csv"))
    report["scenario_understanding_inference"] = summarize_inference(su_inf)
    report["scenario_understanding_ref_based"] = read_ref_based_score(
        os.path.join(SU_OUT, "gpt-4o_reference_based_score.csv")
    )

    out_dir = os.path.join(BASE, "output")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "summary_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("Wrote:", out_json)

    # Quick console view
    for k, v in report.items():
        print("\n==", k)
        if isinstance(v, dict):
            for kk, vv in v.items():
                print(f"{kk}: {vv}")
        else:
            print(v)

    # ---- Optional: pandas summary tables -----------------------------------
    if pd is not None:
        tables_dir = os.path.join(out_dir, "tables")
        os.makedirs(tables_dir, exist_ok=True)

        # Inference stats table
        inf_rows: List[Dict] = []
        inf_rows.append({"task": "empathetic_action_baseline", **report["empathetic_action_inference"]})
        inf_rows.append({"task": "scenario_understanding", **report["scenario_understanding_inference"]})
        if report.get("empathetic_action_inference_trained"):
            inf_rows.append({"task": "empathetic_action_trained", **report["empathetic_action_inference_trained"]})
        df_inf = pd.DataFrame(inf_rows)
        df_inf.to_csv(os.path.join(tables_dir, "inference_stats.csv"), index=False)

        # Reference-based metrics table (wide → long)
        ref_rows: List[Dict] = []
        for task_key, task_name in (
            ("empathetic_action_ref_based", "empathetic_action_baseline"),
            ("scenario_understanding_ref_based", "scenario_understanding"),
        ):
            for metric, score in (report.get(task_key) or {}).items():
                ref_rows.append({"task": task_name, "metric": metric, "score": float(score)})
        if ref_rows:
            df_ref = pd.DataFrame(ref_rows)
            df_ref.to_csv(os.path.join(tables_dir, "reference_based_metrics.csv"), index=False)

    # ---- Optional: quick visuals (PNG) -------------------------------------
    figs_dir = os.path.join(out_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    if plt is not None and pd is not None:
        # Bar chart for empathetic_action reference metrics
        ea_ref = report.get("empathetic_action_ref_based") or {}
        if ea_ref:
            m_names = list(ea_ref.keys())
            m_vals = [float(ea_ref[m]) for m in m_names]
            plt.figure(figsize=(6, 4))
            plt.bar(m_names, m_vals, color="#4C78A8")
            plt.title("Empathetic Action — Reference-based Metrics")
            plt.ylabel("Score")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(figs_dir, "empathetic_action_ref_metrics.png"))
            plt.close()

        # Bar for scenario_understanding BERTScore (if present)
        su_ref = report.get("scenario_understanding_ref_based") or {}
        if su_ref:
            names = list(su_ref.keys())
            vals = [float(su_ref[k]) for k in names]
            plt.figure(figsize=(5, 4))
            plt.bar(names, vals, color="#F58518")
            plt.title("Scenario Understanding — Ref-based Metrics")
            plt.ylabel("Score")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(figs_dir, "scenario_understanding_ref_metrics.png"))
            plt.close()

    # Histogram of response lengths for each task (if matplotlib available)
    if plt is not None:
        # Re-load rows to get lengths
        ea_inf_rows = read_csv(os.path.join(EA_OUT, "gpt-4o_inference.csv")) or \
                      read_csv(os.path.join(EA_OUT, "gpt-4o-mini_inference.csv"))
        su_inf_rows = read_csv(os.path.join(SU_OUT, "gpt-4o_inference.csv"))

        for name, rows in (("empathetic_action", ea_inf_rows), ("scenario_understanding", su_inf_rows)):
            if not rows:
                continue
            lens = [len((r.get("response") or "")) for r in rows]
            plt.figure(figsize=(6, 4))
            plt.hist(lens, bins=20, color="#72B7B2", edgecolor="#333")
            plt.title(f"{name.replace('_', ' ').title()} — Response Length Distribution")
            plt.xlabel("Response length (chars)")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(figs_dir, f"{name}_response_length_hist.png"))
            plt.close()


if __name__ == "__main__":
    main()


