
import copy
import csv
import json
import os

import yaml

from train import run_training

with open("configs/train_config.yaml") as f:
    base = yaml.safe_load(f)
with open("configs/sweep_config.yaml") as f:
    sweep = yaml.safe_load(f)

results_dir = os.path.join(base["output_root"], "results")
os.makedirs(results_dir, exist_ok=True)

results = []
for run in sweep["runs"]:
    name = run["name"]
    result_file = os.path.join(base["output_root"], "runs", name, "result.json")

    # Skip runs that already finished (useful if Colab disconnects)
    if os.path.exists(result_file):
        print(f"Skipping {name} (already done)")
        with open(result_file) as f:
            results.append(json.load(f))
        continue

    cfg = copy.deepcopy(base)
    cfg["lora"]["r"] = run["r"]
    cfg["training"]["learning_rate"] = run["learning_rate"]
    cfg["training"]["epochs"] = run["epochs"]

    print(f"\n=== Running {name} ===")
    try:
        results.append(run_training(cfg, run_name=name))
    except Exception as e:
        print(f"Run {name} failed: {e}")

# Best first (lowest validation loss)
results.sort(key=lambda r: r["eval_loss"])

cols = ["name", "r", "learning_rate", "epochs", "eval_loss", "perplexity",
        "best_step", "stopped_step", "minutes", "peak_gpu_gb"]

with open(os.path.join(results_dir, "sweep_results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(results)

with open(os.path.join(results_dir, "sweep_comparison.md"), "w") as f:
    f.write("| " + " | ".join(cols) + " |\n")
    f.write("|" + "---|" * len(cols) + "\n")
    for r in results:
        f.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")

print("\nBest run:", results[0]["name"] if results else "none")
print(open(os.path.join(results_dir, "sweep_comparison.md")).read())