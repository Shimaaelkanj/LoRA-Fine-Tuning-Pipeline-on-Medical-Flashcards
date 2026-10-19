
import argparse
import subprocess
import sys

STEPS = {
    "data": [
        [sys.executable, "src/prepare_data.py"],
        [sys.executable, "src/make_benchmark.py"],
        [sys.executable, "src/dataset_stats.py"],
    ],
    "train": [[sys.executable, "src/train.py"]],
    "sweep": [[sys.executable, "src/sweep.py"]],
    "eval-base": [[sys.executable, "src/evaluate.py", "--name", "base"]],
    "eval-finetuned": [[sys.executable, "src/evaluate.py", "--name", "finetuned",
                        "--adapter", "deploy/adapters/medical-rank32"]],
    "compare": [[sys.executable, "src/compare.py"]],
}
STEPS["all"] = (STEPS["data"] + STEPS["train"] + STEPS["eval-base"]
                + STEPS["eval-finetuned"] + STEPS["compare"])

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run one stage of the pipeline")
    p.add_argument("stage", choices=STEPS)
    cli = p.parse_args()
    for cmd in STEPS[cli.stage]:
        print("\n>>>", " ".join(cmd))
        subprocess.run(cmd, check=True)