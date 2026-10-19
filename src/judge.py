
import argparse
import json
import random
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

SYSTEM = (
    "You are a strict medical-education grader. You will see a question, "
    "reference facts, and two answers (A and B). Score each answer from 1 to 5 "
    "for factual accuracy, completeness, and concision "
    "(1 = wrong or harmful, 3 = partly correct, 5 = correct and clear). "
    "An answer that invents facts or gives a diagnosis instead of advising "
    "a doctor must score low. Reply ONLY with JSON: "
    '{"score_a": int, "score_b": int, "reason": "one or two sentences"}'
)


def ask_judge(model, question, reference, a, b):
    user = (f"Question: {question}\n\nReference: {reference}\n\n"
            f"Answer A: {a}\n\nAnswer B: {b}")
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": user}],
            )
            return json.loads(r.choices[0].message.content)
        except Exception as e:
            print("retry:", e)
            time.sleep(2)
    return None


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--judge_model", default="gpt-4o")
    p.add_argument("--n_test", type=int, default=50)
    cli = p.parse_args()

    base = json.load(open("outputs/results/base_results.json", encoding="utf-8"))
    ft = json.load(open("outputs/results/finetuned_results.json", encoding="utf-8"))

    items = []
    for b, f in zip(base["benchmark"], ft["benchmark"]):
        items.append({"set": "benchmark", "question": b["instruction"],
                      "reference": "Key facts that should appear: " + ", ".join(b["must_include"]),
                      "base": b["answer"], "ft": f["answer"]})
    for b, f in list(zip(base["test"], ft["test"]))[:cli.n_test]:
        items.append({"set": "test", "question": b["instruction"],
                      "reference": b["reference"],
                      "base": b["answer"], "ft": f["answer"]})

    rng = random.Random(42)
    results = []
    for i, it in enumerate(items):
        base_is_a = rng.random() < 0.5            # blind: random position
        a, b = (it["base"], it["ft"]) if base_is_a else (it["ft"], it["base"])
        j = ask_judge(cli.judge_model, it["question"], it["reference"], a, b)
        if j is None:
            continue
        sb = j["score_a"] if base_is_a else j["score_b"]
        sf = j["score_b"] if base_is_a else j["score_a"]
        results.append({**it, "base_score": sb, "ft_score": sf,
                        "base_was_A": base_is_a, "reason": j["reason"]})
        print(f"{i + 1}/{len(items)} base={sb} ft={sf}")

    def summarize(rows):
        n = len(rows)
        if n == 0:
            return {}
        return {
            "n": n,
            "base_avg": round(sum(r["base_score"] for r in rows) / n, 2),
            "ft_avg": round(sum(r["ft_score"] for r in rows) / n, 2),
            "ft_wins": sum(r["ft_score"] > r["base_score"] for r in rows),
            "base_wins": sum(r["ft_score"] < r["base_score"] for r in rows),
            "ties": sum(r["ft_score"] == r["base_score"] for r in rows),
        }

    summary = {
        "judge_model": cli.judge_model,
        "all": summarize(results),
        "benchmark": summarize([r for r in results if r["set"] == "benchmark"]),
        "test": summarize([r for r in results if r["set"] == "test"]),
    }
    with open("outputs/results/judge_results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": results}, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2))