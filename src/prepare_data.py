
import re
import yaml
from dotenv import load_dotenv
from datasets import load_dataset


load_dotenv()

with open("configs/train_config.yaml") as f:
    cfg = yaml.safe_load(f)

# 1. Load the dataset
ds = load_dataset(cfg["dataset_name"])["train"]
print("Columns:", ds.column_names)
print("Original size:", len(ds))
print("Example:", ds[0])

# 2. Clean text (remove extra spaces and line breaks)
def clean(example):
    for key in ["instruction", "input", "output"]:
        example[key] = re.sub(r"\s+", " ", example[key] or "").strip()
    return example

ds = ds.map(clean)

# 3. Remove empty questions and too-short answers
ds = ds.filter(lambda x: len(x["input"]) > 0
                         and len(x["output"]) >= cfg["min_answer_chars"])
print("After filtering short/empty:", len(ds))

# 4. Remove duplicate questions (prevents leakage between splits)
seen = set()
keep_indices = []
for i, q in enumerate(ds["input"]):
    key = q.lower()
    if key not in seen:
        seen.add(key)
        keep_indices.append(i)

ds = ds.select(keep_indices)
print("After removing duplicate questions:", len(ds))

# 4b. Put the real question in "instruction" and leave "input" empty
def fix_format(example):
    example["instruction"] = example["input"]
    example["input"] = ""
    return example

ds = ds.map(fix_format)

# 5. Take a subset
n = min(cfg["num_examples"], len(ds))
ds = ds.shuffle(seed=cfg["seed"]).select(range(n))

# 6. Split 80/10/10
first = ds.train_test_split(test_size=0.2, seed=cfg["seed"])
second = first["test"].train_test_split(test_size=0.5, seed=cfg["seed"])

train, val, test = first["train"], second["train"], second["test"]

# 7. Save
train.to_json("data/train.jsonl")
val.to_json("data/validation.jsonl")
test.to_json("data/test.jsonl")

print(f"Train: {len(train)}  Validation: {len(val)}  Test: {len(test)}")