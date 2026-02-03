import os
import time
import json
import yaml
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils import load_config, set_all_seeds, ensure_dir, NUM_CLASSES, NUM_STORY_CLASSES
from helpers import expected_likert

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
   
    cfg = load_config(args.config)
    set_all_seeds(cfg["seed"])

    is_story = "story" in cfg["feature"]
    dataset = cfg["dataset"]

    n_classes = NUM_STORY_CLASSES if is_story else NUM_CLASSES

    ts= time.strftime("%Y%m%d_%H%M%S")
    print(f"Running inference with config: {args.config} at {ts}")
    out_root = os.path.join(cfg["output"]["root_dir"], f"{cfg['output']['run_name']}_{ts}")
    ensure_dir(out_root)
    with open(os.path.join(out_root, "used_config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    df = pd.read_csv(cfg["data"]["csv_path"])
    texts = df[cfg["data"]["text_col"]].tolist()

    model_path = cfg["model"]["model_path"]

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=n_classes)
    model.to(device)
    model.eval()

    batch_size = cfg["inference"].get("batch_size", 32)
    max_length = cfg["inference"].get("max_length", 512)
    fp16 = cfg["inference"].get("fp16", False)

    probs_all,  exp_all = [], []
    probs_all_cal,  exp_all_cal = [], []
    binary_all, binary_all_cal = [], []
    dtype = torch.float16 if (fp16 and device == "cuda") else None

    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), ncols=100, desc="Infer"):
            batch_texts = texts[start:start+batch_size]
            enc = tokenizer(
                batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            if dtype is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(**enc).logits
            else:
                logits = model(**enc).logits
    
    
            T = getattr(model.config, "temperature")
            logits_cal = logits / T 
            probs = torch.softmax(logits, dim=-1)
            probs = probs.detach().cpu().numpy()
            probs_cal = torch.softmax(logits_cal, dim=-1)
            probs_cal = probs_cal.detach().cpu().numpy()
            probs_all_cal.extend(probs_cal.tolist())
            probs_all.extend(probs.tolist())

            exp_vals = expected_likert(probs, is_story)
            exp_vals_cal = expected_likert(probs_cal, is_story)
            exp_all.extend(exp_vals)
            exp_all_cal.extend(exp_vals_cal)

            binary = exp_vals >= 0.5 if is_story else exp_vals >= 2.5
            binary_cal = exp_vals_cal >= 0.5 if is_story else exp_vals_cal >= 2.5
            binary_all.extend(binary)
            binary_all_cal.extend(binary_cal)
    
    rows = []
    for i, text in enumerate(texts):
        p = probs_all[i]
        p_cal = probs_all_cal[i]
        if dataset == "cmv":
            rows.append({
                "name":df.iloc[i]["name"], 
                "text": text, 
                **{f"p{k+1}": p[k] for k in range(len(p))},
                "expected_likert": exp_all[i],
                "binary": binary_all[i],
                 **{f"p{k+1}_cal": p_cal[k] for k in range(len(p_cal))}, 
                 "expected_likert_cal": exp_all_cal[i], 
                 "binary_cal": binary_all_cal[i]
            })
        else:
            rows.append({ 
                "text": text, 
                **{f"p{k+1}": p[k] for k in range(len(p))},
                "expected_likert": exp_all[i],
                "binary": binary_all[i],
                 **{f"p{k+1}_cal": p_cal[k] for k in range(len(p_cal))}, 
                 "expected_likert_cal": exp_all_cal[i], 
                 "binary_cal": binary_all_cal[i]
            })

    out_csv = os.path.join(out_root, "inference_results.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved predictions to: {out_csv}")


if __name__ == "__main__":
    main()