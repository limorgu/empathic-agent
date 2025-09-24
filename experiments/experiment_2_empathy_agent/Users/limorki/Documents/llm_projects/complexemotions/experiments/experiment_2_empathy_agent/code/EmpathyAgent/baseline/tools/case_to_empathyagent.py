import os
import json
import argparse
from typing import List, Dict


def read_case_sequences(seq_order_path: str) -> Dict[str, List[str]]:
    """
    Read CASE `seqs_order.txt` (tab-delimited, quoted strings) and build a mapping:
      key: subject column header, value: ordered list of video labels for that subject
    """
    rows: List[List[str]] = []
    with open(seq_order_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip().strip('"') for p in line.split("\t")]
            rows.append(parts)

    if not rows or len(rows) < 3:
        raise ValueError("Unexpected format in seqs_order.txt (need headers + rows)")

    headers = rows[0]
    # Subsequent rows contain alternating blue screen rows; keep only non-bluVid entries
    per_subject: Dict[str, List[str]] = {h: [] for h in headers}
    for r in rows[1:]:
        for col_idx, cell in enumerate(r):
            if col_idx >= len(headers):
                continue
            if not cell:
                continue
            if cell == "bluVid":
                continue
            per_subject[headers[col_idx]].append(cell)

    return per_subject


def build_minimal_samples(per_subject: Dict[str, List[str]],
                          max_per_subject: int = 4) -> List[Dict]:
    """
    Create minimal EmpathyAgent-like samples from CASE ordering:
      - character_id: derived from subject index
      - action_id: derived from an index cycling 0..19 (EmpathyAgent has 20 action folders)
      - scenario/dialogue: templated from video label
    """
    samples: List[Dict] = []
    subject_names = sorted(per_subject.keys())
    for s_idx, subj in enumerate(subject_names):
        vids = [v for v in per_subject[subj] if v not in {"startVid", "endVid"}]
        if not vids:
            continue
        for k, vid in enumerate(vids[:max_per_subject]):
            action_id = (k % 20)
            item = {
                "character_id": 1 + (s_idx % 100),
                "action_id": action_id,
                "scenario_id": 0,
                "scenario": f"Participant {subj} watching CASE clip '{vid}'.",
                "dialogue": f"This clip is labeled {vid}.",
                # Minimal fields for baseline; reference-based metrics will ignore empathy_goal
                "empathy_goal": {"0": "", "1": ""},
                "rank": [-1, 1],
                "empathy_goal_nl": {"0": [""], "1": [""]},
                "high_level_plan": {"0": "", "1": ""}
            }
            samples.append(item)
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqs_order", required=True,
                        help="Path to CASE metadata seqs_order.txt")
    parser.add_argument("--out", required=True,
                        help="Output JSON path (e.g., ../dataset/case_test.json)")
    parser.add_argument("--max_per_subject", type=int, default=4,
                        help="Max samples per subject")
    args = parser.parse_args()

    per_subject = read_case_sequences(args.seqs_order)
    samples = build_minimal_samples(per_subject, max_per_subject=args.max_per_subject)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(samples)} samples to {args.out}")


if __name__ == "__main__":
    main()



