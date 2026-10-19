
import json
import os
import shutil

SRC = "outputs/runs/rank_32/best_adapter"
DST = "deploy/adapters/medical-rank32"
KEEP = ["adapter_config.json", "adapter_model.safetensors"]

os.makedirs(DST, exist_ok=True)
for name in KEEP:
    shutil.copy(os.path.join(SRC, name), os.path.join(DST, name))

cfg = json.load(open(os.path.join(DST, "adapter_config.json")))
size_mb = os.path.getsize(os.path.join(DST, "adapter_model.safetensors")) / 1e6

info = {
    "name": "medical-rank32",
    "base_model": cfg["base_model_name_or_path"],
    "rank": cfg["r"],
    "alpha": cfg["lora_alpha"],
    "dropout": cfg["lora_dropout"],
    "target_modules": sorted(cfg["target_modules"]),
    "dataset": "medalpaca/medical_meadow_medical_flashcards",
    "adapter_size_mb": round(size_mb, 1),
    "note": "Learning project. Not a source of medical advice.",
}
with open(os.path.join(DST, "adapter_info.json"), "w") as f:
    json.dump(info, f, indent=2)

print(json.dumps(info, indent=2))