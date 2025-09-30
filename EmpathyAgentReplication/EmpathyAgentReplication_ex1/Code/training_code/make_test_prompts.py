"""
Convert the paper's test set (e.g., testset_100.json) into:
  - test_prompts.jsonl  (rows with {id, instruction, input?})
  - test_refs.jsonl     (rows with {id, reference})
The script is schema-robust and aligns rows by index.

Usage (from Code/):
  python training_code/make_test_prompts.py \
    --src OriginalPaperEmpathyAgent/dataset/testset_100.json \
    --out_dir OriginalPaperEmpathyAgent/dataset
"""
from pathlib import Path
from typing import Iterable
import json, argparse, csv, sys

THIS_DIR = Path(__file__).resolve().parent
CODE_DIR = THIS_DIR.parent
sys.path.insert(0, str(CODE_DIR))
import cfg_paths as P  # noqa

KEYS_PROMPT  = ["instruction","prompt","query","question","task","situation","context","observation","goal","input"]
# add these keys the paper uses for gold text
KEYS_REF = ["reference","target","gold","answer","plan","expected","output","response",
            "empathy_goal_nl","high_level_plan","ground_truth","expected_plan","gold_plan"]

def _value_to_str(v):
    import json as _json
    if v is None:
        return ""
    if isinstance(v, dict):
        # join values in key order
        parts = []
        for k in sorted(v.keys()):
            x = v[k]
            if isinstance(x, (list, dict)):
                parts.append(_json.dumps(x, ensure_ascii=False))
            else:
                s = str(x).strip()
                if s: parts.append(s)
        return " ".join(parts)
    if isinstance(v, list):
        parts = []
        for x in v:
            if isinstance(x, (list, dict)):
                parts.append(_json.dumps(x, ensure_ascii=False))
            else:
                s = str(x).strip()
                if s: parts.append(s)
        return " ".join(parts)
    return str(v).strip()

def _first_nonempty(d: dict, keys) -> str:
    for k in keys:
        if k in d:
            s = _value_to_str(d[k])
            if s:
                return s
    return ""


def _first_nonempty(d: dict, keys: Iterable[str]) -> str:
    for k in keys:
        v = d.get(k)
        if v is None: 
            continue
        if isinstance(v, list): v = " ".join(str(x) for x in v)
        v = str(v).strip()
        if v: 
            return v
    return ""


def load_any(path: Path):
    # Read bytes first, then try multiple decoders before giving up
    data_bytes = Path(path).read_bytes()

    def _decode(b: bytes) -> str:
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return b.decode(enc)
            except UnicodeDecodeError:
                continue
        try:
            import chardet  # optional
            enc = (chardet.detect(b)["encoding"] or "utf-8")
            return b.decode(enc, errors="replace")
        except Exception:
            return b.decode("utf-8", errors="replace")

    if path.suffix == ".json":
        txt = _decode(data_bytes)
        return json.loads(txt)
    if path.suffix == ".jsonl":
        txt = _decode(data_bytes)
        return [json.loads(l) for l in txt.splitlines() if l.strip()]
    if path.suffix == ".csv":
        import io, csv as _csv
        txt = _decode(data_bytes)
        return list(_csv.DictReader(io.StringIO(txt)))
    raise ValueError(f"Unsupported file type: {path.suffix}")
def _join_known_fields(d: dict, keys: Iterable[str]) -> str:
    # keep for partial use; we’ll now also have a full fallback below
    parts = []
    for k in keys:
        v = d.get(k)
        if v is None or (isinstance(v, str) and not v.strip()):
            continue
        if isinstance(v, list) or isinstance(v, dict):
            import json as _json
            v = _json.dumps(v, ensure_ascii=False)
        parts.append(f"{k}: {str(v).strip()}")
    return "\n".join(parts)

def to_prompt_row(ex: dict, idx: int):
    # preferred single instruction if present
    instr = _first_nonempty(ex, ["instruction","prompt","task","question"])
    if not instr:
        instr = "Given the following scenario details, propose an empathetic, concrete action plan."

    # try known context keys first
    context = _join_known_fields(ex, ["situation","context","observation","goal","input","details","notes"])

    # HARD FALLBACK: include **all non-reference** fields as context
    if not context:
        ref_keys = set(KEYS_REF) | {"instruction","prompt","task","question"}
        parts = []
        import json as _json
        for k, v in ex.items():
            if k in ref_keys: 
                continue
            if isinstance(v, (list, dict)):
                v = _json.dumps(v, ensure_ascii=False)
            s = str(v).strip()
            if s:
                parts.append(f"{k}: {s}")
        context = "\n".join(parts)

    row = {"id": idx, "instruction": instr}
    if context:
        row["input"] = context
    return row


def to_ref_row(ex: dict, idx: int):
    ref = _first_nonempty(ex, KEYS_REF)
    return {"id": idx, "reference": ref}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to testset_100.json (or jsonl/csv)")
    ap.add_argument("--out_dir", default=str(P.CODE_DIR / "OriginalPaperEmpathyAgent" / "dataset"))
    args = ap.parse_args()

    src = (P.CODE_DIR / args.src) if not Path(args.src).exists() else Path(args.src)
    if not src.exists():
        raise FileNotFoundError(f"Missing: {src}")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = out_dir / "test_prompts.jsonl"
    refs_path    = out_dir / "test_refs.jsonl"

    data = load_any(src)
    # unwrap dict containers like {'data': [...]} or {'test': [...]}
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                data = v; break

    if not isinstance(data, list):
        raise ValueError("Unsupported JSON structure for test set; expected a list of examples.")

    with open(prompts_path, "w", encoding="utf-8") as fp, open(refs_path, "w", encoding="utf-8") as fr:
        for i, ex in enumerate(data):
            json.dump(to_prompt_row(ex, i), fp, ensure_ascii=False); fp.write("\n")
            json.dump(to_ref_row(ex, i),    fr, ensure_ascii=False); fr.write("\n")

    print(f"✅ Wrote:\n- {prompts_path}\n- {refs_path}")

if __name__ == "__main__":
    main()
