import os, json

# Resolve BASE relative to this file so it runs on any machine
HERE = os.path.abspath(os.path.dirname(__file__))
BASE = os.path.abspath(os.path.join(HERE, os.pardir))  # .../experiment_2_empathy_agent
SRC = os.path.join(BASE, "code", "EmpathyAgent", "dataset", "empathyrobot_10k.json")
OUT = os.path.join(BASE, "data", "sft_empathyagent.jsonl")


def main():
    with open(SRC, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n = 0
    with open(OUT, "w", encoding="utf-8") as w:
        for item in data:
            char_id = item.get("character_id")
            dialogue = item.get("dialogue", "")
            scenario = item.get("scenario", "")
            # choose the better-ranked reference response (rank[1] == 1)
            rank = item.get("rank", [])
            pick = "1" if (len(rank) > 1 and rank[1] == 1) else "0"
            target = (item.get("empathy_goal_nl", {}).get(pick) or [""])[0]

            prompt = (
                f"Character context: {scenario}\n"
                f"User dialogue: {dialogue}\n"
                f"As an empathetic assistant, propose a concise empathetic action plan and a supportive sentence."
            )
            rec = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": target},
                ]
            }
            w.write(json.dumps(rec) + "\n")
            n += 1
    print("Wrote", OUT, "records:", n)


if __name__ == "__main__":
    main()


