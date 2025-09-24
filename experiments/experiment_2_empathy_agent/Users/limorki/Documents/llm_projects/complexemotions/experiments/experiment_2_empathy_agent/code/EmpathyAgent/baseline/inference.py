import os
import sys
import csv
import json
import time
import random
import argparse

from tqdm import tqdm
from gpt import GPT
from overlap import Overlap, TF_IDF, LCS
from NLG_metric import BERTScore  # used for L1/L2 tasks
from reference_free_metrics.api_eval import EmpathyEvaluator
from reference_free_metrics.legality import LegalityChecker
from reference_free_metrics.scorer import EmpathyScorer


# -----------------------------
# Utilities
# -----------------------------
def load_json_smart(path: str):
    """Load JSON with a couple of fallback encodings."""
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    # last resort: ignore bad bytes (shouldn't be needed)
    with open(path, "rb") as f:
        text = f.read().decode("utf-8", errors="ignore")
    return json.loads(text)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-mini",
        help="Default is gpt-4o-mini (stable/cheaper). Options: 'gpt-4o','gpt-4-turbo','gpt-4-vision-preview'",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="empathetic_action",
        help="Choose between 'scenario_understanding','empathetic_planning','empathetic_action'",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="../dataset/testset_100.json",  # run from baseline/
        help="Path to the test JSON file",
    )
    parser.add_argument(
        "--script_path",
        type=str,
        default="../dataset/scripts",
        help="Path to frames/scripts root (folders named by action_id). For text-only, point to empty folders 0..19",
    )
    parser.add_argument(
        "--reference_free_eval",
        action="store_true",
        help="Enable reference-free evaluation",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start with a fresh CSV (delete previous inference file).",
    )
    return parser.parse_args()


def load_existing_indices(csv_file_path: str):
    """Read existing indices from an inference CSV in a header-safe way."""
    existing = set()
    if os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0:
        with open(csv_file_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                val = (row.get("data_idx") or "").strip()
                if val.isdigit():
                    existing.add(int(val))
    return existing


def generate_with_retry(
    model, video_or_script_input, input_prompt, attempts=6, base_delay=1.5
):
    """
    Generic retry wrapper around model.generate(...).
    Works with your existing GPT class; adds exponential backoff + jitter.
    """
    delay = base_delay
    for k in range(attempts):
        try:
            return model.generate(video_or_script_input, input_prompt)
        except Exception as e:
            msg = str(e).lower()
            # hard stops
            if "quota exceeded" in msg or "insufficient_quota" in msg:
                raise
            if k == attempts - 1:
                raise
            # transient/network/rate issues → backoff and retry
            if any(
                t in msg
                for t in ("connection", "timeout", "read timed out", "rate limit")
            ):
                time.sleep(delay + random.random())  # jitter
                delay *= 1.8
            else:
                # Unknown error: don't spin forever
                raise


# -----------------------------
# Inference
# -----------------------------
def inference(
    model,
    task,
    model_name: str = "",
    empathy_scenario_data_path: str = "",
    character_data_path: str = "",
    video_path: str = "",
    script_path: str = "",
):
    # --- load data ---
    empathy_scenario_data = load_json_smart(empathy_scenario_data_path)
    character_data = load_json_smart(character_data_path)

    # --- outputs: robust append (header only once) ---
    out_dir = os.path.join("output", task)
    os.makedirs(out_dir, exist_ok=True)
    csv_file_path = os.path.join(out_dir, f"{model_name}_inference.csv")

    file_is_new = (not os.path.exists(csv_file_path)) or (
        os.path.getsize(csv_file_path) == 0
    )
    existing_indices = load_existing_indices(csv_file_path)

    # --- prompt selection ---
    prompt_file = {
        "scenario_understanding": "./prompt/prompt_video_l1.txt",
        "empathetic_planning": "./prompt/prompt_video_l2.txt",
        "empathetic_action": "./prompt/prompt_video_l3.txt",
    }[task]
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read().strip()

    # --- main loop ---
    with open(csv_file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if file_is_new:
            writer.writerow(["data_idx", "response"])

        for idx, data in tqdm(
            enumerate(empathy_scenario_data), total=len(empathy_scenario_data)
        ):
            if idx in existing_indices:
                continue

            character_id = data["character_id"]
            dialogue = data["dialogue"]
            action_id = data["action_id"]

            # Choose video or frames folder
            video_or_script_input = (
                os.path.join(video_path, f"{action_id}.mp4")
                if video_path
                else os.path.join(script_path, f"{action_id}")
            )

            if not os.path.exists(video_or_script_input):
                print(f"[WARN] Missing input for idx {idx}: {video_or_script_input}")
                continue

            character_info = character_data[str(character_id)]
            input_prompt = prompt.format(
                character_info=character_info, dialogue=dialogue
            )

            try:
                response = generate_with_retry(
                    model,
                    video_or_script_input,
                    input_prompt,
                    attempts=6,
                    base_delay=1.5,
                )
                writer.writerow([idx, response])
                time.sleep(0.08)  # gentle pacing between calls
            except Exception as e:
                msg = str(e).lower()
                if "quota exceeded" in msg:
                    print("Out of quota. Exiting now.")
                    raise
                if any(
                    t in msg
                    for t in ("connection", "timeout", "read timed out", "rate limit")
                ):
                    print(f"Error processing index {idx}: {e}. Skipping.")
                    continue
                raise

    print(f"Inference Done for {model_name}!")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    args = parse_args()

    # Fresh start if requested
    out_dir = os.path.join("output", args.task)
    if args.fresh:
        try:
            os.remove(os.path.join(out_dir, f"{args.model_name}_inference.csv"))
        except FileNotFoundError:
            pass
        os.makedirs(os.path.join(out_dir, "cache"), exist_ok=True)

    # --- inference ---
    if args.model_name in [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4-vision-preview",
    ]:
        gpt = GPT(model_name=args.model_name)
        inference(
            gpt,
            args.task,
            args.model_name,
            empathy_scenario_data_path=args.test_file,
            character_data_path="../dataset/character.json",
            script_path=args.script_path,  # frames folders live here
            # video_path="../dataset/video",   # uncomment if you use videos instead
        )
    else:
        print("Model name is wrong!")
        sys.exit(1)

    # --- evaluation ---
    csv_file = f"./output/{args.task}/{args.model_name}_inference.csv"

    # Build response dict
    response_dict = {}
    with open(csv_file, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            resp = (row.get("response") or "").strip()
            idx = (row.get("data_idx") or "").strip()
            if resp and idx.isdigit():
                response_dict[idx] = resp

    if args.task == "empathetic_action":
        if args.reference_free_eval:
            os.makedirs(f"output/{args.task}/cache", exist_ok=True)
            out_json = f"output/{args.task}/cache/reference_free_metrics_{args.model_name}.json"

            evaluator = EmpathyEvaluator(test_json_file=args.test_file, level=3)
            evaluator.evaluate(csv_file=csv_file, output_file=out_json)

            if os.path.exists(out_json):
                checker = LegalityChecker(csv_file=csv_file, output_file=out_json)
                checker.process(verbose=True)

                scorer = EmpathyScorer(
                    result_path=f"output/{args.task}/cache/reference_free_metrics_{args.model_name}_legality.json",
                    level=3,
                )
                print("\n", "=" * 50)
                print("Results of reference free metrics:")
                scorer.save_results(
                    filename=f"output/{args.task}/{args.model_name}reference_free_score.csv"
                )
            else:
                print(
                    f"[WARN] Expected file not found: {out_json}. Skipping legality/scorer."
                )

        print("\n", "=" * 50)
        print("Results of reference based metrics:")
        overlap = Overlap()
        lcs = LCS()
        tf_idf = TF_IDF()
        results = [
            ["Metric", "Score"],
            ["Overlap", overlap.score(response_dict, args.test_file)],
            ["LCS", lcs.score(response_dict, args.test_file)],
            ["TF-IDF", tf_idf.score(response_dict, args.test_file)],
        ]
        with open(
            f"output/{args.task}/{args.model_name}_reference_based_score.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            csv.writer(f).writerows(results)

    elif args.task in {"empathetic_planning", "scenario_understanding"}:
        if args.reference_free_eval:
            os.makedirs(f"output/{args.task}/cache", exist_ok=True)
            out_csv = (
                f"output/{args.task}/cache/reference_free_metrics_{args.model_name}.csv"
            )
            level = 2 if args.task == "empathetic_planning" else 1

            evaluator = EmpathyEvaluator(test_json_file=args.test_file, level=level)
            evaluator.evaluate(csv_file=csv_file, output_file=out_csv)

            scorer = EmpathyScorer(result_path=out_csv, level=level)
            print("\n", "=" * 50)
            print("Results of reference free metrics:")
            scorer.save_results(
                filename=f"output/{args.task}/{args.model_name}reference_free_score.csv"
            )

        print("\n", "=" * 50)
        print("Results of reference based metrics:")
        bert_score = BERTScore(model_dir="google-bert/bert-base-uncased")
        score = bert_score.score(response_dict, args.test_file, test_level=args.task)
        with open(
            f"output/{args.task}/{args.model_name}_reference_based_score.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            csv.writer(f).writerows([["Metric", "Score"], ["Bert_score", score]])
# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    args = parse_args()

    # ✅ NEW: show model and task at startup
    print(f"[INFO] Running with model: {args.model_name}, task: {args.task}")

    # Fresh start if requested
    out_dir = os.path.join("output", args.task)
    if args.fresh:
        try:
            os.remove(os.path.join(out_dir, f"{args.model_name}_inference.csv"))
        except FileNotFoundError:
            pass
        os.makedirs(os.path.join(out_dir, "cache"), exist_ok=True)

    # --- inference ---
    if args.model_name in [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4-vision-preview",
    ]:
        gpt = GPT(model_name=args.model_name)
        inference(
            gpt,
            args.task,
            args.model_name,
            empathy_scenario_data_path=args.test_file,
            character_data_path="../dataset/character.json",
            script_path="../dataset/scripts",  # frames folders live here
            # video_path="../dataset/video",   # uncomment if you use videos instead
        )
    else:
        print("Model name is wrong!")
        sys.exit(1)
