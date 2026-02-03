import os
import math
import json
import torch
import numpy as np
import pandas as pd
import optuna

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, DataCollatorWithPadding, Trainer


from utils import ensure_dir, NUM_CLASSES, NUM_STORY_CLASSES, compute_soft_label
from helpers import make_splitter, subset_df, balanced_class_weights_from_counts, compute_metrics_fn, compute_metrics_fn_story, compute_metrics_binary, rmse_expected_likert, mae_expected_likert, brier_multiclass, wasserstein_distance
from trainer import SoftLabelTrainer
from hparams import build_objective
from data import DataCollatorWithSoftLabels, SoftLabelDataset, StoryDataset

def run_model_nested_cv(model_name, df, cfg, out_root):
    print(f"\n===== Model: {model_name} =====")
    results_dir = os.path.join(out_root, model_name.replace("/", "_"))
    ensure_dir(results_dir)
    outer_k = cfg["cv"]["outer_folds"]
    y = df["hard_label"].values
    splitter = make_splitter(outer_k, y)

    outer_metrics = []
    fold_preds = []

    is_story = 'story' in cfg["data"]["label_col"]
    training_type = cfg["training"]["type"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    collator = DataCollatorWithSoftLabels(tokenizer) if training_type == "soft" else DataCollatorWithPadding(tokenizer)
    study = optuna.create_study(direction=cfg["optuna"]["direction"])
    objective = None

    for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(np.arange(len(df)), y)):
        print(f"\n--- Outer fold {fold_idx+1}/{outer_k} ---")
        df_tr = subset_df(df, tr_idx)
        df_va = subset_df(df, va_idx)

        objective = build_objective(model_name, df_tr, cfg, results_dir)
        study.optimize(objective, n_trials=cfg["optuna"]["n_trials"], show_progress_bar=False)
        best_params = study.best_trial.params.copy()
        print("Best inner-CV params:", best_params)

        # Merge best params with training defaults
        hp = {
            "learning_rate": best_params.get("learning_rate", 2e-5),
            "weight_decay": best_params.get("weight_decay", 0.0),
            "num_train_epochs": best_params.get("num_epochs", cfg["training"]["num_epochs"]),
            "per_device_train_batch_size": best_params.get("per_device_train_batch_size",
                                                          cfg["training"]["per_device_train_batch_size"]),
            "gradient_accumulation_steps": best_params.get("gradient_accumulation_steps",
                                                           cfg["training"]["gradient_accumulation_steps"]),
            "max_length": best_params.get("max_length", cfg["training"]["max_length"]),
            "warmup_ratio": best_params.get("warmup_ratio", cfg["training"]["warmup_ratio"]),
            "max_grad_norm": best_params.get("max_grad_norm", cfg["training"]["max_grad_norm"]),
        }

        train_ds = SoftLabelDataset(df_tr, tokenizer, cfg["data"]["text_col"], hp["max_length"]) if training_type == "soft" else StoryDataset(df_tr, tokenizer, cfg["data"]["text_col"], hp["max_length"], cfg["data"]["label_col"])
        val_ds   = SoftLabelDataset(df_va, tokenizer, cfg["data"]["text_col"], hp["max_length"]) if training_type == "soft" else StoryDataset(df_va, tokenizer, cfg["data"]["text_col"], hp["max_length"], cfg["data"]["label_col"])

        fold_dir = os.path.join(results_dir, f"outer_fold_{fold_idx+1}")
        ensure_dir(fold_dir)

        n_classes = NUM_STORY_CLASSES if is_story else NUM_CLASSES
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=n_classes
        )

        args = TrainingArguments(
            output_dir=fold_dir,
            num_train_epochs=hp["num_train_epochs"],
            per_device_train_batch_size=hp["per_device_train_batch_size"],
            per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
            gradient_accumulation_steps=hp["gradient_accumulation_steps"],
            learning_rate=hp["learning_rate"],
            weight_decay=hp["weight_decay"],
            warmup_ratio=hp["warmup_ratio"],
            max_grad_norm=hp["max_grad_norm"],
            eval_strategy=cfg["training"]["eval_strategy"],
            save_strategy=cfg["training"]["save_strategy"],
            load_best_model_at_end=True,
            metric_for_best_model=cfg["training"]["metric_for_best_model"],
            logging_steps=cfg["training"]["logging_steps"],
            fp16=cfg["training"]["fp16"],
            report_to=cfg["training"]["report_to"],
            seed=cfg["training"]["seed"],
            data_seed=cfg["training"]["seed"],
            remove_unused_columns=False,
        )

        if training_type == "soft":
            Y_soft_tr = np.stack([train_ds[i]["labels"] for i in range(len(train_ds))])   
            eff_counts = Y_soft_tr.sum(axis=0)                                     
            class_weights = balanced_class_weights_from_counts(eff_counts).tolist()

        if is_story:
            if training_type == "soft":
                compute_metrics_fun = compute_metrics_fn_story
            else:
                compute_metrics_fun = compute_metrics_binary
        else:
            compute_metrics_fun = compute_metrics_fn
            
        trainer = SoftLabelTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_fun,
            class_weights=class_weights
        ) if training_type == "soft" else Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_fun
        )

        trainer.train()
        pre = trainer.evaluate()
        pred_val = trainer.predict(val_ds)
        logits_val = pred_val.predictions
       
        fold_out = {
            "fold": fold_idx + 1,
            "metrics": {f"{k}": float(v) for k, v in pre.items()},
            "best_params": best_params,
        }

       
        labels_range = np.array([0, 1]) if is_story else np.array([1, 2, 3, 4, 5], dtype=float)
        if training_type == "hard":
            probs = torch.softmax(torch.tensor(logits_val), dim=-1).cpu().numpy()
            true_probs = np.array([compute_soft_label(df_va.iloc[i], cfg["data"]["annotator_cols"], story=is_story) for i in range(len(df_va))])
            fold_out['metrics'].update({"eval_brier": brier_multiclass(probs, true_probs),
                                        "eval_wasserstein": float(np.mean([wasserstein_distance(labels_range, labels_range, t, p) for t, p in zip(true_probs, probs)])),
                                        "eval_rmse_exp": rmse_expected_likert(true_probs, probs, story=is_story),
                                        "eval_mae_exp": mae_expected_likert(true_probs, probs, story=is_story)})
            
        with open(os.path.join(fold_dir, "fold_results.json"), "w") as f:
            json.dump(fold_out, f, indent=2)

        outer_metrics.append(fold_out["metrics"])
        fold_preds.append(fold_out)

        del trainer, model
        torch.cuda.empty_cache()

        study = optuna.create_study(direction=cfg["optuna"]["direction"])


    keys = [
        "eval_loss", "eval_accuracy", "eval_f1", "eval_macro_f1", "eval_micro_f1", "eval_weighted_f1",
        "eval_brier", "eval_wasserstein", "eval_mae_exp", "eval_rmse_exp",
    ]
    agg = {}
    for k in keys:
        vals = [m.get(k, np.nan) for m in outer_metrics]
        vals = [v for v in vals if not (v is None or (isinstance(v, float) and math.isnan(v)))]
        if len(vals) > 0:
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump({"outer_metrics": outer_metrics, "aggregate": agg}, f, indent=2)

    print("\nAggregate across outer folds:")
    for k, v in agg.items():
        print(f"  {k}: {v:.4f}")

    return {"model_name": model_name, "aggregate": agg, "folds": fold_preds}
