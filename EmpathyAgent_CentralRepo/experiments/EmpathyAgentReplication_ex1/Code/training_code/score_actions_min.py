# Minimal reference-based scorer: Overlap (Jaccard), LCS, TF-IDF cosine
from pathlib import Path
from typing import List, Iterable
import sys, argparse, json, csv

# import cfg_paths
THIS_DIR = Path(__file__).resolve().parent
CODE_DIR = THIS_DIR.parent
sys.path.insert(0, str(CODE_DIR))
import cfg_paths as P  # noqa

def _tok(s: str) -> List[str]:
    return [t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in s).split() if t]

def jaccard(a: str, b: str) -> float:
    A, B = set(_tok(a)), set(_tok(b))
    if not A and not B: return 1.0
    if not A or not B:  return 0.0
    return len(A & B) / len(A | B)

def lcs_norm(a: str, b: str) -> float:
    x, y = _tok(a), _tok(b)
    if not x and not y: return 1.0
    if not x or not y:  return 0.0
    m, n = len(x), len(y)
    dp = [0]*(n+1)
    for _ in range(1, m+1):
        prev = 0
        for j in range(1, n+1):
            tmp, prev = dp[j], dp[j-1]+1 if x[_-1]==y[j-1] else max(dp[j], dp[j-1])
            dp[j] = prev
            prev = tmp
    l = dp[n]
    return l / max(m, n)

def tfidf_cosine_batch(preds: List[str], refs: List[str]) -> List[float]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = refs + preds
    vec = TfidfVectorizer(min_df=1).fit(texts)
    V = vec.transform(texts)
    R, Pm = V[:len(refs)], V[len(refs):]
    sims = []
    for i in range(len(refs)):
        r, p = R[i], Pm[i]
        denom = (r.power(2).sum()**0.5) * (p.power(2).sum()**0.5)
        sims.append(float((r @ p.T)[0,0] / denom) if denom else 0.0)
    return sims

def read_predictions(csv_path: Path) -> List[str]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    try: rows.sort(key=lambda r: int(r["id"]))
    except Exception: rows.sort(key=lambda r: r["id"])
    return [r.get("prediction","") for r in rows]

def _first_nonempty(d: dict, keys: Iterable[str]) -> str:
    for k in keys:
        v = d.get(k)
        if v is None: continue
        if isinstance(v, list): v = " ".join(str(x) for x in v)
        v = str(v).strip()
        if v: return v
    return ""

def read_references(path: Path) -> List[str]:
    refs = []
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                ex = json.loads(line)
                txt = _first_nonempty(ex, ["reference","target","output","answer","plan","response"])
                if not txt and "messages" in ex:
                    for m in reversed(ex["messages"]):
                        if m.get("role") in ("assistant","system"):
                            txt = str(m.get("content","")).strip()
                            if txt: break
                refs.append(txt)
    elif path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    data = v; break
        if not isinstance(data, list):
            raise ValueError("Unsupported JSON structure for references.")
        for ex in data:
            refs.append(_first_nonempty(ex, ["reference","target","output","answer","plan","response"]))
    elif path.suffix == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            refs.append(r.get("reference") or r.get("output") or r.get("target") or "")
    else:
        raise ValueError(f"Unsupported reference file type: {path.suffix}")
    return refs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_name", default="lora_tinyllama_min")
    ap.add_argument("--references", required=True, help="Path to gold refs (json/jsonl/csv)")
    args = ap.parse_args()

    adir = P.OUTPUT_DIR / args.adapter_name
    preds_csv = adir / "predictions.csv"
    if not preds_csv.exists():
        raise FileNotFoundError(f"Missing predictions: {preds_csv}")

    refs_path = (CODE_DIR / args.references) if not Path(args.references).exists() else Path(args.references)
    if not refs_path.exists():
        raise FileNotFoundError(f"Missing references: {refs_path}")

    preds = read_predictions(preds_csv)
    refs  = read_references(refs_path)
    n = min(len(preds), len(refs))
    preds, refs = preds[:n], refs[:n]
    if n == 0:
        raise RuntimeError("No comparable rows. Ensure predictions and references align by index.")

    jac = [jaccard(p, r) for p, r in zip(preds, refs)]
    lcs = [lcs_norm(p, r) for p, r in zip(preds, refs)]
    tfc = tfidf_cosine_batch(preds, refs)

    out_csv = adir / "scores.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as wf:
        w = csv.writer(wf)
        w.writerow(["id", "Overlap", "LCS", "TF-IDF"])
        for i, (a,b,c) in enumerate(zip(jac, lcs, tfc)):
            w.writerow([i, f"{a:.6f}", f"{b:.6f}", f"{c:.6f}"])

    avg = lambda xs: sum(xs)/len(xs)
    print("==================================================")
    print(f"Samples scored: {n}")
    print(f"Average Overlap: {avg(jac):.4f}")
    print(f"Average LCS    : {avg(lcs):.4f}")
    print(f"Average TF-IDF : {avg(tfc):.4f}")
    print(f"✅ Wrote per-row scores: {out_csv}")

if __name__ == "__main__":
    main()
