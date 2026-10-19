
import argparse
import glob
import json
import subprocess
import sys

import yaml

PREFERRED = ["acc_norm,none", "acc,none", "prompt_level_strict_acc,none"]


def run(name, model_name, adapter, tasks, limit):
    model_args = f"pretrained={model_name},dtype=float16"
    if adapter:
        model_args += f",peft={adapter}"
    out = f"outputs/results/forgetting_{name}"
    cmd = [sys.executable, "-m", "lm_eval", "--model", "hf",
           "--model_args", model_args, "--tasks", tasks,
           "--limit", str(limit), "--batch_size", "8",
           "--output_path", out]
    subprocess.run(cmd, check=True)
    files = sorted(glob.glob(f"{out}/**/results*.json", recursive=True))
    return json.load(open(files[-1]))["results"]


def pick(metrics):
    for k in PREFERRED:
        if k in metrics:
            return k, metrics[k]
    return None, None


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="outputs/runs/rank_32/best_adapter")
    p.add_argument("--tasks", default="arc_easy,hellaswag")
    p.add_argument("--limit", type=int, default=200)
    cli = p.parse_args()

    cfg = yaml.safe_load(open("configs/train_config.yaml"))
    base = run("base", cfg["model_name"], None, cli.tasks, cli.limit)
    ft = run("finetuned", cfg["model_name"], cli.adapter, cli.tasks, cli.limit)

    rows = []
    print("\n| Task | Metric | Base | Fine-tuned | Change |")
    print("|---|---|---|---|---|")
    for task in base:
        metric, b = pick(base[task])
        f = ft[task].get(metric)
        if b is None or f is None:
            continue
        rows.append({"task": task, "metric": metric, "base": round(b, 4),
                     "finetuned": round(f, 4), "change": round(f - b, 4)})
        print(f"| {task} | {metric} | {b:.4f} | {f:.4f} | {f - b:+.4f} |")

    json.dump(rows, open("outputs/results/forgetting_summary.json", "w"), indent=2)