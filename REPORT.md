
# Experiment Report: LoRA Fine-Tuning for Concise Medical Q&A

## 1. Problem and approach

**Goal:** make a small open-source model (Qwen2.5-1.5B-Instruct) answer short medical-education questions the way a flashcard does: brief and factual.

**Observed problem:** the base model answers correctly but verbosely (about 122 words per answer on the test split, against 50 words in the reference answers).

**Why fine-tuning, and not the alternatives:**
- **Few-shot prompting** can show the desired format, but it uses prompt tokens on every request and the format can drift. I did **not** test a few-shot baseline, so I cannot claim fine-tuning beat it. This is a weakness of the study.
- **RAG** (retrieval) adds knowledge the model lacks. The problem here is style and length, not missing knowledge: the base model already scored 0.83 on my benchmark. RAG would not fix verbosity.
- **Fine-tuning** changes behavior permanently and shortens every answer without extra prompt tokens. The cost is training, plus the risks measured below (hallucination, forgetting).

## 2. Dataset

- Source: `medalpaca/medical_meadow_medical_flashcards` (33,955 rows). License: ___ (check the dataset page).
- Cleaning: whitespace normalization; removed empty questions and answers under 20 characters (33,955 → 33,467); removed duplicate questions (→ ___); the dataset stores the question in `input`, so I moved it to `instruction`.
- Sample: 2,000 random examples (seed 42), split 80/10/10: 1,600 train / 200 validation / 200 test.
- Statistics (from `data/dataset_stats.json`): ___ (average question and answer length per split).
- Benchmark: 30 handcrafted questions: easy 7, hard 12, edge_case 6, failure_mode 5. No exact-match overlap with training data (checked).
- Limitation: I only deduplicated exact question text, so near-duplicates may remain across splits.

## 3. Training and hyperparameter sweep

QLoRA (4-bit), attention layers `q_proj` and `v_proj`, alpha = 2 × rank, dropout 0.05, learning rate 2e-4 with a cosine schedule, 3 epochs, effective batch size 16, on a Colab T4 (fp16). Tracked with MLflow. Validation loss every 25 steps with early stopping (patience 3).

| Run | Rank | LR | Epochs | Val loss | Perplexity | Minutes |
|---|---|---|---|---|---|---|
| rank_32 (selected) | 32 | 2e-4 | 3 | 0.8359 | 2.307 | 15.6 |
| baseline | 16 | 2e-4 | 3 | 0.8407 | 2.318 | 15.5 |
| rank_8 | 8 | 2e-4 | 3 | 0.8508 | 2.342 | 15.5 |

Only 3 of 7 planned runs finished. The learning-rate and epochs runs were interrupted by the free Colab GPU quota. Rank had a small effect (under 2%), each setting was run once, and the best checkpoint was always the last step, so training may not have converged. Rank 32 was selected on validation loss alone, which is a weak choice.

## 4. Head-to-head: base vs fine-tuned

| Metric | Base | Fine-tuned |
|---|---|---|
| Benchmark keyword score (30) | 0.833 | 0.800 |
| Test ROUGE-L (200) | 0.186 | 0.307 |
| Avg answer words, test (reference 49.6) | 122.4 | 59.4 |

**Main result:** fine-tuning changed style, not knowledge. Answers became about half as long and much closer to the references. Benchmark accuracy did not improve (a one-question difference). Part of the ROUGE-L gain comes from the length match.

**Example 1: beriberi (success).** Both are correct. The base model wrote 132 tokens, the fine-tuned model 18 ("Beriberi is caused by thiamine (vitamin B1) deficiency."), about 4× faster.

**Example 2: "Zorbitis syndrome", a fake disease (regression).**
- Base: says the term does not exist in the medical literature.
- Fine-tuned: "includes corticosteroids and immunosuppressive agents…", an invented treatment.

**Example 3: "Fakemycin", a fake drug (regression).** The base model refused, though for the wrong reason (it called the question "sensitive"). The fine-tuned model invented a dose: "10 mg/kg/day".

**Example 4: heparin antidote (hidden error).** Both answered "protamine sulfate". The fine-tuned model also described heparin as inhibiting vitamin K-dependent factors (II, VII, IX, X), which is warfarin's mechanism. The keyword score gave both full marks.

**Failure modes (manual review of 5 answers):** on the 2 real advice questions (stop medication, diagnose me) both models were safe. On the 2 fake entities only the base model refused or denied. The flashcards always contain an answer, so the model learned to always answer. The keyword metric scored both models 0.20, which is invalid for this category.

**Serving (Colab T4, 128 max tokens):** the adapter has a lower median latency (0.77 s vs 3.43 s at concurrency 1) but a heavier tail (P95 10.5 s vs 4.3 s at concurrency 64) and about 3× lower token throughput, probably because the adapter is not merged (unverified).


## 5. What I would do next
- Add "I don't know" examples about nonexistent entities to the training data, then re-measure the fake-entity questions.
- Test a few-shot baseline against the fine-tuned model.
- Finish the learning-rate and epochs sweep, and repeat each run with several seeds.
- Replace the keyword score with an LLM judge or manual grading.
- Merge the adapter and re-run the serving benchmark.

## 6. Limitations
Small benchmark (30 questions), single runs, crude keyword and ROUGE metrics, LLM judge not run, Docker GPU path untested. This is a learning project and not a source of medical advice.