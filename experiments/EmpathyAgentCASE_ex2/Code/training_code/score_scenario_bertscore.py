# Direct BERTScore scorer (no evaluate.load), Python 3.9-compatible
from pathlib import Path
import sys, argparse, json, csv
from statistics import mean

# If you hit ModuleNotFoundError: bert_score → pip install bert-score
from bert_score import score as bertscore

# Make Code/ importable so we can use cfg_paths.py
THIS_DIR = Path(__file__).resolve().parent
CODE_DIR = THIS_DIR.parent
sys.path.insert(0, str(CODE_DIR))
import cfg_paths as CFG  # <-- renamed to avoid shadowing

def read_predictions(csv_path: Path):
    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
    try: rows.sort(key=lambda r: int(r["id"]))
    except Exception: rows.sort(key=lambda r: r["id"])
    return [r.get("prediction","") for r in rows]

def read_refs(jsonl_path: Path):
    refs = []
    with open(jsonl_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip(): continue
            ex = json.loads(line)
            refs.append(ex.get("reference",""))
    return refs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_name", default="lora_tinyllama_min")
    ap.add_argument("--references", required=True)            # path to scenario_refs.jsonl
    ap.add_argument("--model_type", default="distilroberta-base")  # CPU-friendly
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--rescale_with_baseline", action="store_true")
    args = ap.parse_args()

    adir = CFG.OUTPUT_DIR / args.adapter_name
    preds_csv = adir / "predictions.csv"
    if not preds_csv.exists():
        raise FileNotFoundError(f"Missing predictions: {preds_csv}")

    refs_path = (CODE_DIR / args.references) if not Path(args.references).exists() else Path(args.references)
    if not refs_path.exists():
        raise FileNotFoundError(f"Missing references: {refs_path}")

    preds = read_predictions(preds_csv)
    refs  = read_refs(refs_path)
    n = min(len(preds), len(refs))
    preds, refs = preds[:n], refs[:n]
    if n == 0:
        raise RuntimeError("No comparable rows to score.")

    P_list, R_list, F1_list = bertscore(
        cands=preds, refs=refs,
        model_type=args.model_type,
        lang=args.lang,
        rescale_with_baseline=args.rescale_with_baseline,
        device="cpu",
        batch_size=args.batch_size,
        verbose=True,
    )

    # Convert to Python floats
    P_vals  = [float(p) for p in P_list]
    R_vals  = [float(r) for r in R_list]
    F1_vals = [float(f) for f in F1_list]

    P_avg, R_avg, F_avg = mean(P_vals), mean(R_vals), mean(F1_vals)

    out_csv = adir / "scenario_bertscore.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as wf:
        w = csv.writer(wf)
        w.writerow(["id","P","R","F1"])
        for i, (p,r,f) in enumerate(zip(P_vals, R_vals, F1_vals)):
            w.writerow([i, f"{p:.6f}", f"{r:.6f}", f"{f:.6f}"])

    print("==================================================")
    print(f"samples: {n} | model_type: {args.model_type}")
    print(f"BERTScore P: {P_avg:.4f} | R: {R_avg:.4f} | F1: {F_avg:.4f}")
    print(f"✅ wrote per-row: {out_csv}")

if __name__ == "__main__":
    main()
