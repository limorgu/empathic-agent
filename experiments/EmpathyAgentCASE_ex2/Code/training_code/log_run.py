from pathlib import Path
import sys, csv, json, argparse
THIS_DIR = Path(__file__).resolve().parent
CODE_DIR = THIS_DIR.parent
sys.path.insert(0, str(CODE_DIR))
import cfg_paths as P

ap = argparse.ArgumentParser()
ap.add_argument("--adapter_name", required=True)
ap.add_argument("--base_model", required=True)
ap.add_argument("--dataset_tag", default="testset_100")
ap.add_argument("--tokens", type=int, default=160)
ap.add_argument("--temp", type=float, default=0.7)
ap.add_argument("--notes", default="")
args = ap.parse_args()

adir = P.OUTPUT_DIR / args.adapter_name
scores = adir / "scores.csv"
if not scores.exists():
    raise FileNotFoundError(f"Missing scores: {scores}")

rows = list(csv.DictReader(open(scores, newline="", encoding="utf-8")))
Overlap = sum(float(r["Overlap"]) for r in rows)/len(rows)
LCS     = sum(float(r["LCS"])     for r in rows)/len(rows)
TFIDF   = sum(float(r["TF-IDF"])  for r in rows)/len(rows)

log = P.OUTPUT_DIR / "run_log.csv"
header = ["adapter","base_model","dataset","max_new_tokens","temperature","avg_overlap","avg_lcs","avg_tfidf","notes"]
newrow = [args.adapter_name, args.base_model, args.dataset_tag, args.tokens, args.temp, f"{Overlap:.4f}", f"{LCS:.4f}", f"{TFIDF:.4f}", args.notes]

exists = log.exists()
with open(log, "a", newline="", encoding="utf-8") as wf:
    w = csv.writer(wf)
    if not exists: w.writerow(header)
    w.writerow(newrow)

print(f"✅ appended to {log}")
