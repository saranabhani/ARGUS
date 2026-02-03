import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, DataCollatorWithPadding, Trainer

from utils import NUM_CLASSES, NUM_STORY_CLASSES
from helpers import make_splitter, subset_df, balanced_class_weights_from_counts, compute_metrics_fn, compute_metrics_fn_story, compute_metrics_binary
from data import DataCollatorWithSoftLabels, SoftLabelDataset, StoryDataset
from trainer import SoftLabelTrainer

def build_objective(model_name, df_train_outer, cfg, out_dir):

    def objective(trial):

        search = cfg["optuna"]["shared_search_spaces"]
        model_search = cfg["optuna"]["model_search_spaces"]

        def sample_lr():
            s = search.get("learning_rate", None)
            if isinstance(s, dict):
                return trial.suggest_float("learning_rate", s["low"], s["high"], log=s.get("log", False))
            return trial.suggest_categorical("learning_rate", s)

        def sample_wd():
            s = search.get("weight_decay", 0.0)
            if isinstance(s, dict):
                return trial.suggest_float("weight_decay", s["low"], s["high"], log=s.get("log", False))
            return trial.suggest_categorical("weight_decay", s) if isinstance(s, list) else s

        def sample_wr():
            s = search.get("warmup_ratio", None)
            if isinstance(s, dict):
                return trial.suggest_float("warmup_ratio", s["low"], s["high"], log=s.get("log", False))
            return trial.suggest_categorical("warmup_ratio", s)

        def sample_mgn():
            s= search.get("max_grad_norm", None)
            if isinstance(s, dict):
                return trial.suggest_float("max_grad_norm", s["low"], s["high"], log=s.get("log", False))
            return trial.suggest_categorical("max_grad_norm", s) if isinstance(s, list) else s

        def sample_epochs():
            s = search.get("num_epochs", cfg["training"]["num_epochs"])
            if isinstance(s, dict):
                return trial.suggest_int("num_epochs", s["low"], s["high"], step=s.get("step", 1))
            return trial.suggest_categorical("num_epochs", s) if isinstance(s, list) else s

        def sample_bsz():
            s = search.get("per_device_train_batch_size", cfg["training"]["per_device_train_batch_size"])
            if isinstance(s, list):
                return trial.suggest_categorical("per_device_train_batch_size", s)
            return s

        def sample_gas():
            s = search.get("gradient_accumulation_steps", cfg["training"]["gradient_accumulation_steps"])
            if isinstance(s, list):
                return trial.suggest_categorical("gradient_accumulation_steps", s)
            return s

        def sample_max_length():
            s = model_search[model_name].get("max_length", cfg["training"]["max_length"])
            if isinstance(s, list):
                return trial.suggest_categorical("max_length", s)
            return s

        lr = sample_lr()
        wd = sample_wd()
        wr = sample_wr()
        mgn = sample_mgn()
        num_epochs = sample_epochs()
        bsz = sample_bsz()
        gas = sample_gas()
        max_length = sample_max_length()
        
        is_story = 'story' in cfg["data"]["label_col"]
        training_type = cfg["training"]["type"]

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        collator = DataCollatorWithSoftLabels(tokenizer) if training_type == "soft" else DataCollatorWithPadding(tokenizer)
        n_classes = NUM_STORY_CLASSES if is_story else NUM_CLASSES
        inner_folds = cfg["cv"]["inner_folds"]
        y = df_train_outer["hard_label"].values
        splitter = make_splitter(inner_folds, y)
       
        val_losses = []
        for in_idx, val_idx in splitter.split(np.arange(len(df_train_outer)), y):
            df_tr = subset_df(df_train_outer, in_idx)
            df_va = subset_df(df_train_outer, val_idx)

            train_ds = SoftLabelDataset(df_tr, tokenizer, cfg["data"]["text_col"], max_length) if training_type == "soft" else StoryDataset(df_tr, tokenizer, cfg["data"]["text_col"], max_length, cfg["data"]["label_col"])
            val_ds   = SoftLabelDataset(df_va, tokenizer, cfg["data"]["text_col"], max_length) if training_type == "soft" else StoryDataset(df_va, tokenizer, cfg["data"]["text_col"], max_length, cfg["data"]["label_col"])

            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=n_classes
            )

            args = TrainingArguments(
                output_dir=os.path.join(f"{cfg['output']['root_dir']}/.tmp", f"trial_{trial.number}"),
                num_train_epochs=num_epochs,
                per_device_train_batch_size=bsz,
                per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
                gradient_accumulation_steps=gas,
                learning_rate=lr,
                max_grad_norm=mgn,
                weight_decay=wd,
                warmup_ratio=wr,
                eval_strategy="epoch",
                save_strategy="no",
                logging_steps=cfg["training"]["logging_steps"],
                fp16=cfg["training"]["fp16"],
                report_to=cfg["training"]["report_to"],
                disable_tqdm=True,
                remove_unused_columns=False,
                data_seed=cfg["training"]["seed"],
                seed=cfg["training"]["seed"],
            )
            # class weights for soft labels
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
                class_weights=class_weights,
            ) if training_type == "soft" else Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                data_collator=collator,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics_fun,
            )
            
            eval_res = trainer.train()
            metrics = trainer.evaluate()
            val_losses.append(float(metrics["eval_loss"]))

            del trainer, model
            torch.cuda.empty_cache()

        return float(np.mean(val_losses))

    return objective