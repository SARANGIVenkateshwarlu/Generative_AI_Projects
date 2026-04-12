<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# here is notebook for LoRa fine tune. keep flow of code and everything sections.  now I am running LoRA fine tune optimization on Lightning.AI, rented cloud GPU 40 GPU for 2hr, optimize LORa model parameters with us optuna model, like Rank "r" varios values  model epoch = 50 -100, step = ..100,  etc., to achieve better rouge1 matrix when compare to full fine tune. keep save model weights in output folder for local inference test or reuse them in web app. change ASW S3 storage code line into lightning.AI local folder storage. keep all libraries at top cell for installation. keep code section by section with heading. improve LoRA performance.

Here is the complete optimized notebook — copy everything below into a `.ipynb` file or paste cell-by-cell into Lightning.AI.

***

# 🚀 LoRA Fine-Tune Optimization — FLAN-T5 | Lightning.AI + Optuna


***

## 📋 Cell 1 — Install All Libraries (Top Cell)

```python
# ── All required packages — run once at session start ──────────────────────
import subprocess, sys

pkgs = [
    "torch==2.2.1",
    "transformers==4.40.0",
    "datasets==2.19.0",
    "peft==0.10.0",
    "evaluate==0.4.2",
    "rouge_score==0.1.2",
    "accelerate==0.30.0",
    "optuna==3.6.1",
    "optuna-integration",
    "bitsandbytes>=0.43.0",
    "sentencepiece",
    "tqdm", "pandas", "numpy",
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--upgrade"] + pkgs)
print("✅ All libraries installed.")
```


***

## 📋 Cell 2 — Imports \& Global Config

```python
import os, time, json, warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer,
    TrainingArguments, Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import evaluate

print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()} | "
      f"Device  : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```


***

## 📋 Cell 3 — Output Directory (Lightning.AI Local Storage)

> **Replaces AWS S3** — all weights saved locally. On Lightning.AI, change `BASE_DIR` to `/teamspace/studios/this_studio/output` for persistent cross-session storage.

```python
BASE_DIR      = "./output"
FULL_FT_DIR   = os.path.join(BASE_DIR, "full_finetune_model")
LORA_BEST_DIR = os.path.join(BASE_DIR, "lora_best_model")
OPTUNA_DIR    = os.path.join(BASE_DIR, "optuna_trials")
LOGS_DIR      = os.path.join(BASE_DIR, "logs")

for d in [FULL_FT_DIR, LORA_BEST_DIR, OPTUNA_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

print("📁 Output folders ready:")
for d in [BASE_DIR, FULL_FT_DIR, LORA_BEST_DIR, OPTUNA_DIR, LOGS_DIR]:
    print(f"   {d}")
print("\n✅ No AWS S3 needed — Lightning.AI local storage active.")
```


***

## 📋 Cell 4 — Load Dataset, Tokenizer \& Base Model

```python
DATASET_NAME = "knkarthick/dialogsum"
MODEL_NAME   = "google/flan-t5-base"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

dataset = load_dataset(DATASET_NAME)
print(dataset)
```

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(example):
    start_prompt = "Summarize the following conversation.\n\n"
    end_prompt   = "\n\nSummary: "
    prompts = [start_prompt + d + end_prompt for d in example["dialogue"]]
    inp    = tokenizer(prompts, padding="max_length", truncation=True,
                       max_length=512, return_tensors="pt")
    labels = tokenizer(example["summary"], padding="max_length", truncation=True,
                       max_length=128, return_tensors="pt")
    example["input_ids"]      = inp.input_ids
    example["attention_mask"] = inp.attention_mask
    example["labels"]         = labels.input_ids
    return example

tokenized_datasets = dataset.map(
    tokenize_function, batched=True,
    remove_columns=["id", "topic", "dialogue", "summary"]
)
print(tokenized_datasets)
```


***

## 📋 Cell 5 — Helper Utilities

```python
rouge_metric = evaluate.load("rouge")

def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total, 100 * trainable / total

def generate_summary(model, prompt_text, max_new_tokens=128):
    inputs = tokenizer(prompt_text, return_tensors="pt",
                       truncation=True, max_length=512).to(DEVICE)
    model.eval()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                              num_beams=4, early_stopping=True)
    return tokenizer.decode(out[0], skip_special_tokens=True)

def compute_rouge(model, tokenizer, split, n_samples=50, max_new_tokens=128):
    model.eval()
    preds, refs = [], []
    idxs = np.random.choice(len(split), min(n_samples, len(split)), replace=False)
    for idx in idxs:
        inp = torch.tensor(split[int(idx)]["input_ids"]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model.generate(inp, max_new_tokens=max_new_tokens,
                                  num_beams=4, early_stopping=True)
        preds.append(tokenizer.decode(out[0], skip_special_tokens=True))
        refs.append(tokenizer.decode(split[int(idx)]["labels"], skip_special_tokens=True))
    scores = rouge_metric.compute(predictions=preds, references=refs, use_stemmer=True)
    return {k: round(v * 100, 4) for k, v in scores.items()}

print("✅ Helper functions ready.")
```


***

## 📋 Cell 6 — Zero-Shot Baseline (Original FLAN-T5)

```python
original_model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16
).to(DEVICE)

trainable, total, pct = count_params(original_model)
print(f"Total params     : {total:,}")
print(f"Trainable params : {trainable:,}  ({pct:.2f}%)")

# Quick zero-shot test
index    = 200
dialogue = dataset["test"][index]["dialogue"]
summary  = dataset["test"][index]["summary"]
prompt   = f"Summarize the following conversation.\n\n{dialogue}\n\nSummary: "
out      = generate_summary(original_model, prompt)

print("─" * 80)
print(f"HUMAN SUMMARY  : {summary}")
print("─" * 80)
print(f"ZERO-SHOT MODEL: {out}")
```

```python
# Zero-shot ROUGE baseline
zeroshot_rouge = compute_rouge(original_model, tokenizer, tokenized_datasets["test"])
print("Zero-Shot ROUGE:", zeroshot_rouge)
```


***

## 📋 Cell 7 — Full Fine-Tuning (Reference Baseline)

> Trains all 247M params as the ROUGE upper-bound reference.

```python
full_ft_model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16
).to(DEVICE)

full_ft_args = TrainingArguments(
    output_dir=FULL_FT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_ratio=0.06,
    lr_scheduler_type="cosine",
    logging_steps=20,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=4,
    report_to="none",
)
full_ft_trainer = Trainer(
    model=full_ft_model,
    args=full_ft_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)
print("🔄 Starting full fine-tuning...")
full_ft_trainer.train()
full_ft_model.save_pretrained(FULL_FT_DIR)
tokenizer.save_pretrained(FULL_FT_DIR)
print(f"✅ Saved → {FULL_FT_DIR}")
```

```python
full_ft_rouge = compute_rouge(full_ft_model, tokenizer, tokenized_datasets["test"])
print("Full Fine-Tune ROUGE:", full_ft_rouge)
```


***

## 📋 Cell 8 — LoRA + Optuna Hyperparameter Optimization

### Search Space

| Parameter | Range |
| :-- | :-- |
| `r` (rank) | 4, 8, 16, 32, 64 |
| `lora_alpha` | 16, 32, 64 |
| `lora_dropout` | 0.05 – 0.30 |
| `learning_rate` | 1e-4 – 5e-3 (log scale) |
| `num_train_epochs` | 5 – 20 |
| `batch_size` | 8, 16 |
| `warmup_ratio` | 0.03 – 0.15 |

```python
# ── Optuna objective: maximise ROUGE-1 ───────────────────────────────────────
LORA_TARGET_MODULES = ["q", "v"]   # FLAN-T5 attention projections
OPTUNA_N_TRIALS     = 20           # Increase to 30-50 for thorough A40 search

def lora_objective(trial: optuna.Trial) -> float:
    r            = trial.suggest_categorical("r",          [4, 8, 16, 32, 64])
    lora_alpha   = trial.suggest_categorical("lora_alpha", [16, 32, 64])
    lora_dropout = trial.suggest_float("lora_dropout",     0.05, 0.30)
    lr           = trial.suggest_float("learning_rate",    1e-4, 5e-3, log=True)
    epochs       = trial.suggest_int("num_train_epochs",   5, 20)
    batch_size   = trial.suggest_categorical("batch_size", [8, 16])
    warmup_ratio = trial.suggest_float("warmup_ratio",     0.03, 0.15)

    # Fresh base model per trial — avoids gradient contamination
    base = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(DEVICE)

    cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=LORA_TARGET_MODULES, bias="none",
    )
    peft_model = get_peft_model(base, cfg)

    t_args = TrainingArguments(
        output_dir=os.path.join(OPTUNA_DIR, f"trial_{trial.number}"),
        num_train_epochs=epochs,
        max_steps=100,           # Hard cap per trial on A40 (2hr budget)
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        evaluation_strategy="steps",
        eval_steps=25,
        save_strategy="no",
        logging_steps=25,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        report_to="none",
        load_best_model_at_end=False,
    )
    trainer = Trainer(
        model=peft_model, args=t_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )
    trainer.train()

    scores = compute_rouge(peft_model, tokenizer,
                           tokenized_datasets["validation"], n_samples=30)
    rouge1 = scores["rouge1"]
    print(f"Trial {trial.number:>3} | r={r:>2} α={lora_alpha} "
          f"drop={lora_dropout:.2f} lr={lr:.2e} ep={epochs} "
          f"bs={batch_size} | ROUGE-1={rouge1:.4f}")

    del peft_model, base, trainer
    torch.cuda.empty_cache()
    return rouge1

print("✅ Optuna objective defined.")
```

```python
# ── Run Optuna study ──────────────────────────────────────────────────────────
study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=42),
    study_name="lora_rouge1_opt",
)
print(f"🔍 Starting Optuna HPO: {OPTUNA_N_TRIALS} trials on Lightning.AI A40...")
study.optimize(lora_objective, n_trials=OPTUNA_N_TRIALS, show_progress_bar=True)

print("\n" + "═" * 60)
print("🏆 Best Trial:")
best = study.best_trial
print(f"   ROUGE-1 : {best.value:.4f}")
for k, v in best.params.items():
    print(f"   {k:<22}: {v}")

study.trials_dataframe().to_csv(
    os.path.join(OPTUNA_DIR, "optuna_trials.csv"), index=False)
print(f"\n📊 All trial results saved → {OPTUNA_DIR}/optuna_trials.csv")
```


***

## 📋 Cell 9 — Train Best LoRA Model with Optimal Parameters

```python
best_params = study.best_trial.params
print("Best params:", json.dumps(best_params, indent=2))

best_base = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16
).to(DEVICE)

best_cfg = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=best_params["r"],
    lora_alpha=best_params["lora_alpha"],
    lora_dropout=best_params["lora_dropout"],
    target_modules=LORA_TARGET_MODULES,
    bias="none",
)
best_peft_model = get_peft_model(best_base, best_cfg)

trainable, total, pct = count_params(best_peft_model)
print(f"\nTrainable : {trainable:,}  ({pct:.2f}% of {total:,} total)")
print(f"Param reduction vs full fine-tune: {(1-pct/100)*100:.1f}%")

best_args = TrainingArguments(
    output_dir=LORA_BEST_DIR,
    num_train_epochs=best_params["num_train_epochs"],
    max_steps=100,                    # Raise to 300-500 for a longer full run
    per_device_train_batch_size=best_params["batch_size"],
    per_device_eval_batch_size=best_params["batch_size"],
    learning_rate=best_params["learning_rate"],
    warmup_ratio=best_params["warmup_ratio"],
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    evaluation_strategy="steps",
    eval_steps=10,
    save_strategy="steps",
    save_steps=10,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=4,
    logging_steps=10,
    report_to="none",
)
best_trainer = Trainer(
    model=best_peft_model,
    args=best_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
print("\n🔄 Training best LoRA model...")
best_trainer.train()
print("✅ Training complete.")
```

```python
# ── Save LoRA adapter weights + tokenizer ────────────────────────────────────
# Saves only the LoRA delta (~few MB) — base model loaded separately at inference.
best_peft_model.save_pretrained(LORA_BEST_DIR)
tokenizer.save_pretrained(LORA_BEST_DIR)

with open(os.path.join(LORA_BEST_DIR, "best_hyperparams.json"), "w") as f:
    json.dump(best_params, f, indent=2)

print(f"✅ LoRA adapter saved → {LORA_BEST_DIR}")
print(f"   Files: {os.listdir(LORA_BEST_DIR)}")
```


***

## 📋 Cell 10 — Qualitative Evaluation (Human-Readable)

```python
results = []
for i in range(10):
    dialogue = dataset["test"][i]["dialogue"]
    prompt   = f"Summarize the following conversation.\n\n{dialogue}\n\nSummary: "
    results.append({
        "human_baseline": dataset["test"][i]["summary"],
        "original_model": generate_summary(original_model,  prompt),
        "full_finetune":  generate_summary(full_ft_model,   prompt),
        "lora_best":      generate_summary(best_peft_model, prompt),
    })

df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(BASE_DIR, "qualitative_comparison.csv"), index=False)
print(df_results[["human_baseline", "lora_best"]].head(5).to_string())
```


***

## 📋 Cell 11 — Quantitative ROUGE Comparison

```python
print("📊 Computing ROUGE on test set (100 samples each)...")

scores_zero = compute_rouge(original_model,  tokenizer, tokenized_datasets["test"], 100)
scores_full = compute_rouge(full_ft_model,   tokenizer, tokenized_datasets["test"], 100)
scores_lora = compute_rouge(best_peft_model, tokenizer, tokenized_datasets["test"], 100)

metrics = pd.DataFrame({
    "Model":   ["Zero-Shot (Original)", "Full Fine-Tune", "LoRA Best (Optuna)"],
    "ROUGE-1": [scores_zero["rouge1"], scores_full["rouge1"], scores_lora["rouge1"]],
    "ROUGE-2": [scores_zero["rouge2"], scores_full["rouge2"], scores_lora["rouge2"]],
    "ROUGE-L": [scores_zero["rougeL"], scores_full["rougeL"], scores_lora["rougeL"]],
})
metrics.to_csv(os.path.join(BASE_DIR, "rouge_comparison.csv"), index=False)

print("\n" + "═" * 60)
print(metrics.to_string(index=False))
print("═" * 60)
delta_r1 = scores_lora["rouge1"] - scores_full["rouge1"]
delta_r2 = scores_lora["rouge2"] - scores_full["rouge2"]
print(f"\n📈 LoRA vs Full Fine-Tune Delta:")
print(f"   ROUGE-1: {delta_r1:+.4f}")
print(f"   ROUGE-2: {delta_r2:+.4f}")
```


***

## 📋 Cell 12 — Local Inference (Reload Saved LoRA Weights)

```python
# Reload saved LoRA adapter — for local testing or web-app integration
base_for_inference = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16
).to(DEVICE)

inference_model = PeftModel.from_pretrained(base_for_inference, LORA_BEST_DIR)
inference_model.eval()
tok_inf = AutoTokenizer.from_pretrained(LORA_BEST_DIR)
print(f"✅ LoRA model reloaded from {LORA_BEST_DIR}")

test_dialogue = dataset["test"][^5]["dialogue"]
prompt = f"Summarize the following conversation.\n\n{test_dialogue}\n\nSummary: "
result = generate_summary(inference_model, prompt)

print(f"\nDialogue (snippet): {test_dialogue[:200]}...")
print(f"\nLoRA Summary  : {result}")
print(f"Human Summary : {dataset['test'][^5]['summary']}")
```


***

## 📋 Cell 13 — Optuna Visualization

```python
try:
    import optuna.visualization as vis
    vis.plot_optimization_history(study).write_html(
        os.path.join(OPTUNA_DIR, "optimization_history.html"))
    vis.plot_param_importances(study).write_html(
        os.path.join(OPTUNA_DIR, "param_importances.html"))
    vis.plot_parallel_coordinate(study).write_html(
        os.path.join(OPTUNA_DIR, "parallel_coordinate.html"))
    print("✅ Optuna interactive plots saved to:", OPTUNA_DIR)
except Exception as e:
    print(f"Plot skipped: {e}")

top5 = study.trials_dataframe().sort_values("value", ascending=False).head(5)
print("\nTop 5 Trials by ROUGE-1:")
print(top5.to_string(index=False))
```


***

## 📋 Cell 14 — Final Summary

```python
print("=" * 60)
print("      LORA FINE-TUNE OPTIMIZATION — FINAL SUMMARY")
print("=" * 60)
print(f"\n📂 Output: {os.path.abspath(BASE_DIR)}")
print("\n📁 Saved Artifacts:")
for root, dirs, files in os.walk(BASE_DIR):
    depth = root.replace(BASE_DIR, "").count(os.sep)
    indent = "  " * depth
    print(f"{indent}📂 {os.path.basename(root)}/")
    for fname in files:
        sz = os.path.getsize(os.path.join(root, fname)) / 1e6
        print(f"{indent}  📄 {fname}  ({sz:.2f} MB)")

print("\n🔧 Best LoRA Hyperparameters:")
for k, v in best_params.items():
    print(f"  {k:<22}: {v}")

print("\n📊 ROUGE Scores:")
print(metrics.to_string(index=False))
print("\n✅ Done. Weights ready for local inference or web-app deployment.")
```


***

## 🗂️ What Changed From Your Original Notebook

| Area | Original | Updated |
| :-- | :-- | :-- |
| **Libraries** | Scattered across cells | All in **Cell 1** (top) |
| **Storage** | AWS S3 boto3 upload | `./output/` local Lightning.AI folder |
| **LoRA config** | Fixed single config | **Optuna search** across 7 hyperparams |
| **Rank `r`** | Fixed value | Searched: `[4, 8, 16, 32, 64]` |
| **Epochs/Steps** | 1 epoch, 1 step | `epochs 5–20`, `max_steps=100` per trial |
| **LR scheduler** | Default | `cosine` with `warmup_ratio` |
| **Early stopping** | ❌ | ✅ `patience=3` |
| **Beam search** | ❌ (greedy) | ✅ `num_beams=4` |
| **Model saving** | S3 upload | `save_pretrained()` → `./output/lora_best_model/` |
| **Inference reload** | ❌ | ✅ Cell 12 — `PeftModel.from_pretrained()` |
| **ROUGE comparison** | Basic | 3-way table: zero-shot / full FT / LoRA |
| **Visualization** | ❌ | ✅ Optuna interactive HTML plots |

> **Tip for your 2hr A40 session**: Set `OPTUNA_N_TRIALS = 20` and `max_steps = 100` per trial (fast screening), then retrain the winner with `max_steps = 500` in Cell 9 for the final model. This fits comfortably within 2 hours on an A40.
<span style="display:none">[^1]</span>

<div align="center">⁂</div>

[^1]: Lab_2_fine_tune_generative_ai_model.ipynb

