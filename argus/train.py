import os
import time
import yaml
import json
import torch
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer

from utils import load_config, set_all_seeds, ensure_dir, NUM_CLASSES, NUM_STORY_CLASSES, compute_soft_label
from data import load_dataframe, DataCollatorWithSoftLabels, SoftLabelDataset, StoryDataset
from trainer import SoftLabelTrainer
from helpers import compute_all_metrics, expected_likert, binarize_expected
from temperature_scaler import temperature_scaling
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_all_seeds(cfg["training"]["seed"])   

    best_params_path = cfg["best_params"]
    with open(best_params_path, 'r') as f:
        best_params = json.load(f)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(cfg["output"]["root_dir"], f"{cfg['output']['run_name']}_{cfg['training']['type']}_{ts}")
    ensure_dir(out_root)
    with open(os.path.join(out_root, "used_config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    
    is_story = 'story' in cfg["data"]["label_col"]
    training_type = cfg["training"]["type"]

    df_train = load_dataframe(cfg["data"], split="train")
    df_test = load_dataframe(cfg["data"], split="test")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=cfg["training"]["seed"])
    y_train_strat = df_train[cfg["data"]["label_col"]].to_numpy()
    train_idx, cal_idx = next(splitter.split(df_train, y_train_strat))

    df_train_fit = df_train.iloc[train_idx].reset_index(drop=True).copy()
    df_cal = df_train.iloc[cal_idx].reset_index(drop=True).copy()

    model_name = cfg["training"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    collator = DataCollatorWithSoftLabels(tokenizer=tokenizer) if training_type == "soft_label" else DataCollatorWithPadding(tokenizer=tokenizer)  

    train_ds = SoftLabelDataset(
        df_train_fit,
        tokenizer,
        text_col=cfg["data"]["text_col"],
        max_length=best_params["max_length"]
    ) if training_type == "soft" else StoryDataset(
        df_train_fit,
        tokenizer,
        text_col=cfg["data"]["text_col"],
        max_length=best_params["max_length"],
        label_col=cfg["data"]["label_col"]
    )

    cal_ds = SoftLabelDataset(
        df_cal,
        tokenizer,
        text_col=cfg["data"]["text_col"],
        max_length=best_params["max_length"]
    ) if training_type == "soft" else StoryDataset(
        df_cal,
        tokenizer,
        text_col=cfg["data"]["text_col"],
        max_length=best_params["max_length"],
        label_col=cfg["data"]["label_col"]
    )
    test_ds = SoftLabelDataset(
        df_test,
        tokenizer,
        text_col=cfg["data"]["text_col"],
        max_length=best_params["max_length"]
    ) if training_type == "soft" else StoryDataset(
        df_test,
        tokenizer,
        text_col=cfg["data"]["text_col"],
        max_length=best_params["max_length"],
        label_col=cfg["data"]["label_col"]
    )

    n_classes =  NUM_STORY_CLASSES if is_story else NUM_CLASSES

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_classes)

    model_dir = os.path.join(out_root, model_name.replace("/", "_"))
    
    args = TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=best_params["num_epochs"],
        per_device_train_batch_size=best_params["per_device_train_batch_size"],
        gradient_accumulation_steps=best_params["gradient_accumulation_steps"],
        learning_rate=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
        warmup_ratio=best_params["warmup_ratio"],
        max_grad_norm=best_params["max_grad_norm"],
        eval_strategy="no",
        save_strategy="no",
        fp16=cfg["training"]["fp16"],
        seed=cfg["training"]["seed"],
        data_seed=cfg["training"]["seed"],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=None,
        data_collator=collator,
    ) if training_type == "hard" else SoftLabelTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=None,
        data_collator=collator,
    )

    trainer.train()
    model.eval()

    pred_test = trainer.predict(test_ds)
    logits_test_before = torch.tensor(pred_test.predictions, dtype=torch.float32)
    probs_test_before = torch.softmax(logits_test_before, dim=-1).cpu().numpy()

    y_test_hard = df_test["hard_label"].to_numpy(dtype=np.int64)
    y_test_soft = np.stack(df_test["soft_label"].values).astype(np.float32) if training_type == "soft" else np.array([compute_soft_label(df_test.iloc[i], cfg["data"]["annotator_cols"], story=is_story) for i in range(len(df_test))])

    metrics_before = compute_all_metrics(
        probs=probs_test_before,
        y_soft=y_test_soft,
        y_hard=y_test_hard,
        story=is_story)
    
    expected_before = expected_likert(probs_test_before, story=is_story).tolist()
    binarized_before = [binarize_expected(score, story=is_story) for score in expected_before]
    
    # predictions before df
    preds_before_df = pd.DataFrame({
        "text": df_test[cfg["data"]["text_col"]].astype(str).values,
        "hard_label": y_test_hard,
        "soft_label": [json.dumps(list(map(float, s))) for s in y_test_soft],
        "predicted_scalar": expected_before,
        "predicted_binary": binarized_before,
        **{f"p_{i+1}": probs_test_before[:, i] for i in range(probs_test_before.shape[1])}
    })
    preds_before_path = os.path.join(out_root, "preds_before.csv")
    preds_before_df.to_csv(preds_before_path, index=False)

    # calibration with temperature scaling
    pred_cal = trainer.predict(cal_ds)
    logits_cal = torch.tensor(pred_cal.predictions, dtype=torch.float32)
    y = torch.tensor(df_cal["hard_label"].astype(int).to_numpy(), dtype=torch.long) if training_type == "hard" else torch.tensor(np.stack(df_cal["soft_label"].values).astype(np.float32))

    t_value, temp = temperature_scaling(logits_cal, y, training_type=training_type)
    
    with open(os.path.join(out_root, "temperature.json"), "w") as f:
        json.dump({"temperature": t_value}, f, indent=2)

    model.config.temperature = t_value
    save_dir = os.path.join(model_dir, "calibrated_model")
    ensure_dir(save_dir)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    with torch.no_grad():
        logits_test_after = temp(logits_test_before)
    probs_test_after = torch.softmax(logits_test_after, dim=-1).cpu().numpy()

    metrics_after = compute_all_metrics(
        probs=probs_test_after,
        y_soft=y_test_soft,
        y_hard=y_test_hard,
        story=is_story)
    
    expected_after = expected_likert(probs_test_after, story=is_story).tolist()
    binarized_after = [binarize_expected(score, story=is_story) for score in expected_after]
    
    preds_after_df = pd.DataFrame({
        "text": df_test[cfg["data"]["text_col"]].astype(str).values,
        "hard_label": y_test_hard,
        "soft_label": [json.dumps(list(map(float, s))) for s in y_test_soft],
        "predicted_scalar": expected_after,
        "predicted_binary": binarized_after,
        **{f"p_{i+1}": probs_test_after[:, i] for i in range(probs_test_after.shape[1])}
    })
    preds_after_path = os.path.join(out_root, "preds_after.csv")
    preds_after_df.to_csv(preds_after_path, index=False)

    # save metrics summary
    metrics = {
        "before": metrics_before,
        "after": metrics_after,
        "temperature": t_value
    }
    with open(os.path.join(out_root, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    def fmt(m):
        return (
            f"RMSE={m['rmse_exp']:.4f}  MAE={m['mae_exp']:.4f}  Brier={m['brier']:.4f}  Wasserstein={m['wasserstein']:.4f}  "
            f"Acc={m['accuracy']:.4f}  F1(bin)={m['f1']:.4f}  "
            f"F1 micro={m['micro_f1']:.4f}  F1 macro={m['macro_f1']:.4f}  F1 weighted={m['weighted_f1']:.4f}"
        )
    
    print("\n=== METRICS (Before calibration) ===")
    print(fmt(metrics_before))
    print("\n=== METRICS (After calibration) ===")
    print(fmt(metrics_after))
    print(f"\nSaved:\n- {preds_before_path}\n- {preds_after_path}\n- {os.path.join(out_root, 'metrics.json')}\n- {os.path.join(out_root, 'temperature.json')}\n- Calibrated model in {save_dir}")


if __name__ == "__main__":
    main()