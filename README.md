
## Task
Answer short factual medical-education questions accurately and concisely.


## Phase 1: Data
- Source: medalpaca/medical_meadow_medical_flashcards
- Cleaning: whitespace normalization, removal of short/empty answers, removal of duplicate questions
- Split (seed 42): 1,600 train / 200 validation / 200 test
- Benchmark: 30 handcrafted questions (easy, hard, edge cases, failure modes), kept separate from training data

## Phase 2: Training Pipeline

I built a reproducible QLoRA training pipeline with Hugging Face PEFT and TRL, designed to run on a free Colab T4 GPU. The base model (Qwen2.5-1.5B-Instruct) is loaded in 4-bit and kept frozen, and only small LoRA adapters are trained on the attention layers `q_proj` and `v_proj`. Every setting lives in a YAML config file, so any run can be reproduced from the config and the fixed seed (42). Each run is tracked with MLflow (hyperparameters, loss curves, peak GPU memory, training time). Validation loss is checked every 25 steps with early stopping (patience 3), and the checkpoint with the lowest validation loss is kept.

### Design choices

| Setting | Value | Why |
|---|---|---|
| Method | QLoRA (4-bit NF4 base model) | Fits on a 15 GB T4 GPU |
| LoRA rank (r) | 8, 16, 32 compared | Controls adapter capacity |
| LoRA alpha | 2 x rank | Standard scaling rule, follows rank automatically |
| LoRA dropout | 0.05 | Light regularization for a small dataset |
| Target modules | `q_proj`, `v_proj` | Cheap attention-only baseline |
| Learning rate | 2e-4, cosine schedule | Typical LoRA starting point |
| Epochs | 3 | Baseline setting |
| Batch size | 4 x 4 gradient accumulation (effective 16) | Fits in T4 memory |
| Max sequence length | 512 | Flashcard answers are short |
| Precision | fp16, LoRA weights in float32 | The T4 does not support bf16 kernels |

### Sweep results

Three of the seven planned runs completed: the baseline and the two rank variants. The learning-rate runs (`lr_1e-4`, `lr_5e-4`) were interrupted and the epochs runs never started, because the free Colab GPU quota ran out. Learning rate and epochs were therefore not compared, which is a limitation. Runs were compared on validation loss only, so the test set is untouched.

| Run | Rank | LR | Epochs | Val loss | Perplexity | Best step | Minutes | Peak GPU (GB) |
|---|---|---|---|---|---|---|---|---|
| rank_32 | 32 | 2e-4 | 3 | 0.8359 | 2.307 | 300 | 15.6 | 2.31 |
| baseline | 16 | 2e-4 | 3 | 0.8407 | 2.318 | 300 | 15.5 | 2.27 |
| rank_8 | 8 | 2e-4 | 3 | 0.8508 | 2.342 | 300 | 15.5 | 2.25 |

**Selected configuration:** rank 32, learning rate 2e-4, 3 epochs, chosen for the lowest validation loss at nearly identical time and memory cost.

### Findings
- Validation loss fell from 1.23 (30-step smoke test) to about 0.84 after 300 steps.
- Rank has a small effect: the three runs differ by under 2%, and each setting was run once, so the gap may not be significant.
- In every completed run the best checkpoint was the final step, so the model was probably still improving and early stopping never triggered. More epochs may help, but this is untested.
- Lower validation loss does not guarantee better answers. Phase 3 checks this on the test set and the benchmark.

### Challenges solved
- Colab resets deleted my files, so I wrote results straight to Google Drive and made the setup repeatable with a zip upload and file-size checks.
- A newer `transformers` version renamed `warmup_ratio`, so the script now checks which argument names exist.
- The T4 does not support bf16 kernels, so the script uses fp16 on GPUs below compute capability 8.0 and casts the LoRA weights to float32 for the gradient scaler.
- W&B was blocked on my network, so I used MLflow for tracking.

### How to run (Colab)
1. On the laptop, zip `configs/`, `data/`, `src/`, and `requirements.txt`.
2. In Colab, select the T4 GPU, upload and unzip the zip, then install:
```bash
   pip install peft trl bitsandbytes accelerate datasets pyyaml mlflow
```
3. Link `outputs/` to Google Drive so finished runs survive a disconnect.
4. Smoke test, then the sweep:
```bash
   python src/train.py --max_steps 30 --run_name smoke_test
   python src/sweep.py
```


## Phase 3: Evaluation and Comparison

I compared the base model (Qwen2.5-1.5B-Instruct) with the fine-tuned model (rank 32 adapter) using the same prompt template, greedy decoding, and identical scoring. The test split was used for the first time in this phase and not used to tune anything.

| Metric | Base | Fine-tuned | Change |
|---|---|---|---|
| Benchmark keyword score (30 questions) | 0.833 | 0.800 | -0.033 |
| Test ROUGE-L (200 questions) | 0.186 | 0.307 | +0.121 |
| Avg answer length, test (words; reference 49.6) | 122.4 | 59.4 | -63.0 |
| Avg answer length, benchmark (words) | 89.2 | 25.0 | -64.2 |

| Benchmark category | Questions | Base | Fine-tuned |
|---|---|---|---|
| easy | 7 | 1.000 | 0.857 |
| hard | 12 | 1.000 | 0.917 |
| edge_case | 6 | 0.833 | 1.000 |
| failure_mode | 5 | 0.200 | 0.200 |

### Findings
- Fine-tuning mainly changed **style**: answers went from about 120 words to about 60, close to the flashcard references, and ROUGE-L rose from 0.186 to 0.307. Part of that gain comes from the length match.
- Benchmark accuracy did not improve. The 0.033 drop equals one question out of 30, so it is not meaningful.
- The fine-tuned model **hallucinates on fake topics**. In a demo, it invented a treatment for the made-up "Zorbitis syndrome", while the base model said the disease does not exist. The flashcards always contain an answer, so the model learned to always answer.
- The fine-tuned model also made a factual error on the heparin antidote question (it described warfarin's mechanism), although the final answer, protamine, was correct. The keyword score gave both models full marks.

### Not completed
- LLM-as-judge was skipped (needs a paid API key).
- The catastrophic-forgetting test was skipped because of the free GPU quota, so I cannot say whether general ability dropped.

### Limitations
- Keyword scoring is crude. For failure-mode questions it misses refusals such as "I can't answer", so the equal 0.20 scores should not be read as equal behavior.
- With 30 benchmark questions, one question moves the score by 3 points, and each model was trained and evaluated once.
- ROUGE-L measures word overlap, not medical correctness.
- This is a learning project, not a source of medical advice.

## Phase 4: Inference Pipeline

- **Adapter package:** `deploy/adapters/medical-rank32` (adapter weights 17.4 MB, rank 32, targets `q_proj` and `v_proj`), small enough to commit to git. The base model is downloaded separately.
- **Serving:** vLLM 0.31 on a Colab T4 loads the base model and attaches the adapter as `medical`. It exposes an OpenAI-style `/v1/chat/completions` API for both models.
- **A/B endpoint:** a FastAPI app (`src/ab_server.py`) sends one prompt to both models in parallel and returns both answers with latency, token counts, and eval scores when available.

### Serving benchmark (T4, max_tokens 128, short prompts)

| Concurrency | Base P50 / P95 (s) | Base tokens/s | Adapter P50 / P95 (s) | Adapter tokens/s |
|---|---|---|---|---|
| 1 | 3.43 / 4.27 | 33 | 0.77 / 5.37 | 21 |
| 8 | 2.00 / 2.32 | 388 | 1.02 / 5.86 | 117 |
| 32 | 2.78 / 3.45 | 1069 | 1.31 / 7.18 | 351 |
| 64 | 3.30 / 4.33 | 1732 | 1.94 / 10.53 | 523 |

Highest tested concurrency with 0 errors and P95 ≤ 10 s: adapter 32, base at least 64 (ceiling not reached).

### Findings and limits
- The adapter has lower typical latency (shorter answers) but a heavier tail and about 3x lower token throughput, probably because LoRA is applied unmerged. Merging the adapter before serving is untested.
- Tokens/s is not directly comparable, because the fine-tuned model generates far fewer tokens per answer.
- Numbers apply to a Colab T4 and short prompts. Each level ran once.
- The A/B demo ran without the stored eval scores (they came back `null`), because the result files were not yet available when it started.