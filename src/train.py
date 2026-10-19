
import argparse
import gc
import inspect
import json
import math
import os
import time

import mlflow
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


def make_run_name(cfg):
    return f"r{cfg['lora']['r']}_lr{cfg['training']['learning_rate']}_ep{cfg['training']['epochs']}"


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = str(v)
    return out


def load_split(path, tokenizer):
    """Turn each example into one chat-formatted text string."""
    ds = load_dataset("json", data_files=path, split="train")

    def to_text(ex):
        user = ex["instruction"]
        if ex.get("input"):
            user += "\n\n" + ex["input"]
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": ex["output"]},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    return ds.map(to_text, remove_columns=ds.column_names)


class GpuMemoryCallback(TrainerCallback):
    """Logs peak GPU memory to MLflow at every evaluation."""

    def on_evaluate(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            gb = torch.cuda.max_memory_allocated() / 1e9
            mlflow.log_metric("gpu_mem_gb", gb, step=state.global_step)


def run_training(cfg, run_name=None):
    run_name = run_name or make_run_name(cfg)
    run_dir = os.path.join(cfg["output_root"], "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    set_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # bf16 needs compute capability 8.0+ (A100 and newer). A T4 is 7.5, so use fp16.
    use_bf16 = (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 8
    )
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    model = trainer = None

    with mlflow.start_run(run_name=run_name):
        try:
            mlflow.log_params(flatten(cfg))
            mlflow.log_dict(cfg, "config.json")

            # --- tokenizer and data ---
            tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            train_ds = load_split(cfg["data"]["train_file"], tokenizer)
            val_ds = load_split(cfg["data"]["validation_file"], tokenizer)

            # --- model (QLoRA = 4-bit base model) ---
            kwargs = {}
            if cfg["quantization"]["use_4bit"]:
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                )
            kwargs["torch_dtype"] = compute_dtype
            model = AutoModelForCausalLM.from_pretrained(
                cfg["model_name"], device_map={"": 0}, **kwargs
            )
            model.config.use_cache = False
            if cfg["quantization"]["use_4bit"]:
                model = prepare_model_for_kbit_training(model)

            # --- LoRA ---
            r = cfg["lora"]["r"]
            lora_config = LoraConfig(
                r=r,
                lora_alpha=int(r * cfg["lora"]["alpha_multiplier"]),
                lora_dropout=cfg["lora"]["dropout"],
                target_modules=cfg["lora"]["target_modules"],
                bias="none",
                task_type="CAUSAL_LM",
            )

            # --- training arguments (names differ slightly between library versions) ---
            t = cfg["training"]
            sig = inspect.signature(SFTConfig.__init__).parameters
            args = dict(
                output_dir=os.path.join(run_dir, "checkpoints"),
                num_train_epochs=t["epochs"],
                max_steps=t.get("max_steps", -1),
                per_device_train_batch_size=t["batch_size"],
                per_device_eval_batch_size=t["batch_size"],
                gradient_accumulation_steps=t["grad_accum_steps"],
                learning_rate=t["learning_rate"],
                lr_scheduler_type="cosine",
                logging_steps=10,
                eval_steps=t["eval_steps"],
                save_steps=t["eval_steps"],
                save_strategy="steps",
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                report_to="mlflow",
                run_name=run_name,
                seed=cfg["seed"],
                bf16=use_bf16,
                fp16=not use_bf16,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                dataset_text_field="text",
            )
            args["eval_strategy" if "eval_strategy" in sig else "evaluation_strategy"] = "steps"
            args["max_length" if "max_length" in sig else "max_seq_length"] = t["max_seq_length"]

            # warmup: the setting name differs between library versions
            if "warmup_ratio" in sig:
                args["warmup_ratio"] = 0.03
            else:
                args["warmup_steps"] = 5

            # drop any setting this library version doesn't know, and say so
            unknown = [k for k in args if k not in sig]
            if unknown:
                print("Dropping unsupported args:", unknown)
            args = {k: v for k, v in args.items() if k in sig}

            tok_key = (
                "processing_class"
                if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters
                else "tokenizer"
            )
            trainer = SFTTrainer(
                model=model,
                args=SFTConfig(**args),
                train_dataset=train_ds,
                eval_dataset=val_ds,
                peft_config=lora_config,
                callbacks=[
                    EarlyStoppingCallback(early_stopping_patience=t["early_stopping_patience"]),
                    GpuMemoryCallback(),
                ],
                **{tok_key: tokenizer},
            )

            # fp16 + grad scaler needs trainable weights in float32 (T4 fix)
            if not use_bf16:
                for p in trainer.model.parameters():
                    if p.requires_grad and p.dtype != torch.float32:
                        p.data = p.data.to(torch.float32)

            # --- train ---
            start = time.time()
            trainer.train()
            minutes = (time.time() - start) / 60

            # --- best checkpoint and why ---
            state = trainer.state
            evals = [h for h in state.log_history if "eval_loss" in h]
            best_ckpt = state.best_model_checkpoint
            best_loss = state.best_metric
            best_step = int(best_ckpt.split("-")[-1]) if best_ckpt else None
            stopped_step = state.global_step
            reason = (
                f"Selected {best_ckpt} (step {best_step}) because it had the lowest "
                f"validation loss ({best_loss}) among {len(evals)} evaluations. "
                f"Training ended at step {stopped_step}."
            )
            print(reason)
            mlflow.log_text(reason, "checkpoint_selection.txt")
            mlflow.set_tag("best_checkpoint", str(best_ckpt))

            # --- final validation with the best model, then save the adapter ---
            final = trainer.evaluate()
            final_loss = final["eval_loss"]
            adapter_dir = os.path.join(run_dir, "best_adapter")
            trainer.save_model(adapter_dir)
            tokenizer.save_pretrained(adapter_dir)

            peak_gb = (
                torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            )
            mlflow.log_metrics(
                {
                    "final_eval_loss": final_loss,
                    "final_perplexity": math.exp(final_loss),
                    "train_minutes": minutes,
                    "peak_gpu_gb": peak_gb,
                    "best_step": best_step or 0,
                    "stopped_step": stopped_step,
                }
            )

            result = {
                "name": run_name,
                "r": r,
                "learning_rate": t["learning_rate"],
                "epochs": t["epochs"],
                "eval_loss": round(final_loss, 4),
                "perplexity": round(math.exp(final_loss), 3),
                "best_step": best_step,
                "stopped_step": stopped_step,
                "minutes": round(minutes, 1),
                "peak_gpu_gb": round(peak_gb, 2),
            }
            with open(os.path.join(run_dir, "result.json"), "w") as f:
                json.dump(result, f, indent=2)
            return result

        finally:
            del trainer, model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--max_steps", type=int, default=None, help="quick smoke test")
    cli = parser.parse_args()

    with open(cli.config) as f:
        cfg = yaml.safe_load(f)
    if cli.max_steps:
        cfg["training"]["max_steps"] = cli.max_steps

    print(run_training(cfg, cli.run_name))