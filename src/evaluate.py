
import argparse
import json
import os
import time

import torch
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import yaml


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(tokenizer, question):
    # Same chat format as training: the user asks, the assistant answers.
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate(model, tokenizer, questions, batch_size, max_new_tokens):
    tokenizer.padding_side = "left"   # needed for batched generation
    answers = []
    for i in range(0, len(questions), batch_size):
        prompts = [build_prompt(tokenizer, q) for q in questions[i:i + batch_size]]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,                      # greedy = reproducible
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = out[:, enc["input_ids"].shape[1]:]   # drop the prompt
        answers += tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        print(f"  generated {min(i + batch_size, len(questions))}/{len(questions)}")
    return [a.strip() for a in answers]


def keyword_score(answer, must_include):
    a = answer.lower()
    hits = [k for k in must_include if k.lower() in a]
    return len(hits) / len(must_include)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--name", default="base")
    parser.add_argument("--adapter", default=None, help="path to a LoRA adapter")
    parser.add_argument("--test_limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    cli = parser.parse_args()

    with open(cli.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=torch.float16, device_map={"": 0}
    )
    if cli.adapter:
        model = PeftModel.from_pretrained(model, cli.adapter)
    model.eval()

    start = time.time()

    # ---- 1. Handcrafted benchmark (keyword score) ----
    bench = read_jsonl("data/benchmark.jsonl")
    print("Benchmark:", len(bench), "questions")
    bench_answers = generate(model, tokenizer, [b["instruction"] for b in bench],
                             cli.batch_size, cli.max_new_tokens)
    bench_rows = []
    for b, ans in zip(bench, bench_answers):
        bench_rows.append({
            "instruction": b["instruction"],
            "category": b["category"],
            "must_include": b["must_include"],
            "answer": ans,
            "score": keyword_score(ans, b["must_include"]),
        })

    # ---- 2. Test split (ROUGE-L against the reference answer) ----
    test = read_jsonl("data/test.jsonl")
    if cli.test_limit:
        test = test[:cli.test_limit]
    print("Test split:", len(test), "questions")
    test_answers = generate(model, tokenizer, [t["instruction"] for t in test],
                            cli.batch_size, cli.max_new_tokens)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    test_rows = []
    for t, ans in zip(test, test_answers):
        r = scorer.score(t["output"], ans)["rougeL"].fmeasure
        test_rows.append({
            "instruction": t["instruction"],
            "reference": t["output"],
            "answer": ans,
            "rougeL": r,
        })

    # ---- 3. Summary ----
    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    by_cat = {}
    for r in bench_rows:
        by_cat.setdefault(r["category"], []).append(r["score"])

    summary = {
        "name": cli.name,
        "adapter": cli.adapter,
        "benchmark_keyword_score": round(mean([r["score"] for r in bench_rows]), 4),
        "benchmark_by_category": {k: round(mean(v), 4) for k, v in by_cat.items()},
        "test_rougeL": round(mean([r["rougeL"] for r in test_rows]), 4),
        "avg_answer_words_benchmark": round(mean([len(r["answer"].split()) for r in bench_rows]), 1),
        "avg_answer_words_test": round(mean([len(r["answer"].split()) for r in test_rows]), 1),
        "avg_reference_words_test": round(mean([len(r["reference"].split()) for r in test_rows]), 1),
        "n_benchmark": len(bench_rows),
        "n_test": len(test_rows),
        "minutes": round((time.time() - start) / 60, 1),
    }

    os.makedirs("outputs/results", exist_ok=True)
    path = f"outputs/results/{cli.name}_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "benchmark": bench_rows, "test": test_rows},
                  f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2))
    print("Saved to", path)