# Convert inference.jsonl -> predictions.csv (id,prediction)
from pathlib import Path
import sys, json, argparse, csv

THIS_DIR = Path(__file__).resolve().parent
CODE_DIR = THIS_DIR.parent
sys.path.insert(0, str(CODE_DIR))
import cfg_paths as P

ap = argparse.ArgumentParser()
ap.add_argument("--adapter_name", default="lora_tinyllama_min")
args = ap.parse_args()

adir = P.OUTPUT_DIR / args.adapter_name
src = adir / "inference.jsonl"
dst = adir / "predictions.csv"

if not src.exists():
    raise FileNotFoundError(f"Missing: {src}")

with open(src, "r", encoding="utf-8") as f, open(dst, "w", newline="", encoding="utf-8") as wf:
    w = csv.writer(wf)
    w.writerow(["id", "prediction"])  # simple schema; scorer mapper can read this
    for i, line in enumerate(f):
        if not line.strip(): 
            continue
        row = json.loads(line)
        pred = row.get("output") or ""
        w.writerow([i, pred])

print(f"✅ wrote {dst}")
