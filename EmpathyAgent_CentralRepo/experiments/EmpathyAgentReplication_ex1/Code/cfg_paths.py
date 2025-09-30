# cfg_paths.py — minimal, robust, and aware of your "Code" folder
from pathlib import Path

# Resolve paths relative to this file (works on your Mac layout)
THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent                 # .../EmpathyAgentReplication_ex1
CODE_DIR = THIS_DIR                    # .../EmpathyAgentReplication_ex1/Code
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

# Original paper copy you keep under Code/
OP_DIR       = CODE_DIR / "OriginalPaperEmpathyAgent"
BASELINE_DIR = OP_DIR / "baseline"
DATASET_DIR  = OP_DIR / "dataset"

def ensure_dirs():
    """Create common output/data dirs if missing (training uses OUTPUT_DIR)."""
    for p in [DATA_DIR, OUTPUT_DIR, OP_DIR, BASELINE_DIR, DATASET_DIR]:
        p.mkdir(parents=True, exist_ok=True)

def summary():
    return "\n".join([
        f"ROOT         : {ROOT}",
        f"CODE_DIR     : {CODE_DIR}",
        f"DATA_DIR     : {DATA_DIR}",
        f"OUTPUT_DIR   : {OUTPUT_DIR}",
        f"OP_DIR       : {OP_DIR}",
        f"BASELINE_DIR : {BASELINE_DIR}",
        f"DATASET_DIR  : {DATASET_DIR}",
    ])

if __name__ == "__main__":
    ensure_dirs()
    for p in [ROOT, CODE_DIR]:
        assert p.exists(), f"Missing expected path: {p}"
    print("✅ Paths ready\n" + summary())
