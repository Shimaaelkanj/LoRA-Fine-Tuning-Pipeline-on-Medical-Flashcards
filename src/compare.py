
import json


def load(name):
    with open(f"outputs/results/{name}_results.json", encoding="utf-8") as f:
        return json.load(f)


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


base, ft = load("base"), load("finetuned")
bs, fs = base["summary"], ft["summary"]
lines = []

lines.append("# Base vs fine-tuned\n")
lines.append("| Metric | Base | Fine-tuned | Change |")
lines.append("|---|---|---|---|")
for key in ["benchmark_keyword_score", "test_rougeL",
            "avg_answer_words_test"]:
    b, f = bs[key], fs[key]
    lines.append(f"| {key} | {b} | {f} | {f - b:+.4f} |")

# Per-category breakdown (benchmark)
lines.append("\n## Benchmark by category\n")
lines.append("| Category | Base | Fine-tuned | Change |")
lines.append("|---|---|---|---|")
for cat in sorted(bs["benchmark_by_category"]):
    b = bs["benchmark_by_category"][cat]
    f = fs["benchmark_by_category"].get(cat, 0)
    lines.append(f"| {cat} | {b} | {f} | {f - b:+.4f} |")

# Head-to-head on every benchmark question (same order in both files)
helped, hurt, same = [], [], []
for b, f in zip(base["benchmark"], ft["benchmark"]):
    assert b["instruction"] == f["instruction"]
    if f["score"] > b["score"]:
        helped.append((b, f))
    elif f["score"] < b["score"]:
        hurt.append((b, f))
    else:
        same.append((b, f))

lines.append(f"\n## Benchmark head-to-head\n")
lines.append(f"Helped: {len(helped)}  |  Hurt: {len(hurt)}  |  Same: {len(same)}\n")


def show(title, pairs):
    lines.append(f"### {title}\n")
    if not pairs:
        lines.append("None.\n")
    for b, f in pairs:
        lines.append(f"**{b['instruction']}** ({b['category']}, needs {b['must_include']})")
        lines.append(f"- Base ({b['score']:.2f}): {b['answer'][:300]}")
        lines.append(f"- Fine-tuned ({f['score']:.2f}): {f['answer'][:300]}\n")


show("Where fine-tuning helped", helped)
show("Regressions (where it hurt)", hurt)

# Test split: ROUGE-L per example
diffs = []
for b, f in zip(base["test"], ft["test"]):
    diffs.append((f["rougeL"] - b["rougeL"], b, f))
diffs.sort(key=lambda x: x[0])
up = sum(1 for d in diffs if d[0] > 0)
down = sum(1 for d in diffs if d[0] < 0)
lines.append(f"\n## Test split (ROUGE-L)\n")
lines.append(f"Improved on {up} of {len(diffs)} examples, got worse on {down}.\n")

lines.append("### 5 biggest regressions\n")
for d, b, f in diffs[:5]:
    lines.append(f"**{b['instruction']}** (change {d:+.3f})")
    lines.append(f"- Reference: {b['reference'][:250]}")
    lines.append(f"- Base: {b['answer'][:250]}")
    lines.append(f"- Fine-tuned: {f['answer'][:250]}\n")

with open("outputs/results/comparison.md", "w", encoding="utf-8") as out:
    out.write("\n".join(lines))
print("\n".join(lines[:25]))
print("\nSaved outputs/results/comparison.md")