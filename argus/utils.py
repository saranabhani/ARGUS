import os
import yaml
import random
import json
import numpy as np
import torch
import pandas as pd
from collections import Counter

from transformers import set_seed

NUM_CLASSES = 5
NUM_STORY_CLASSES = 2

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def set_all_seeds(seed):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def likert_to_index(x):
    return int(x) - 1

def compute_soft_label(row, annotator_cols, story):
    #TODO: check if this is needed
    if 'p1' in row.index and row['p1'] >= 0:
        return np.array([row['p1'], row['p2'], row['p3'], row['p4'], row['p5']], dtype=np.float32)
    
    votes = []
    for c in annotator_cols:
        val = row.get(c, np.nan)
        if pd.notna(val):
            index_val = int(val) if story else likert_to_index(int(val))
            votes.append(index_val)
    n_classes = NUM_STORY_CLASSES if story else NUM_CLASSES
    counts = np.bincount(votes, minlength=n_classes).astype(np.float32)
    probs = counts / counts.sum()
    return probs

def select_best_params(out_dir, cfg):
    models = cfg['models']
    for model in models:
        best_params = []
        for fold in range(1,6):
            with open(f'{out_dir}/{model.replace("/", "_")}/outer_fold_{fold}/fold_results.json', 'r') as f:
                params = json.load(f)
            best_params.append(params['best_params'])

        avg_learning_rate = sum([params['learning_rate'] for params in best_params]) / len(best_params)
        avg_weight_decay = sum([params['weight_decay'] for params in best_params]) / len(best_params)
        avg_warmup_ratio = sum([params['warmup_ratio'] for params in best_params]) / len(best_params)
        avg_max_grad_norm = sum([params['max_grad_norm'] for params in best_params]) / len(best_params)
        most_common_epochs = Counter([params['num_epochs'] for params in best_params]).most_common(1)[0][0]
        most_common_train_batch_size = Counter([params['per_device_train_batch_size'] for params in best_params]).most_common(1)[0][0]
        most_common_gradient_accumulation_steps = Counter([params['gradient_accumulation_steps'] for params in best_params]).most_common(1)[0][0]
        most_common_max_length = Counter([params['max_length'] for params in best_params]).most_common(1)[0][0]

        # save the averaged params as json
        best_params = {
            'learning_rate': avg_learning_rate,
            'weight_decay': avg_weight_decay,
            'num_epochs': most_common_epochs,
            'per_device_train_batch_size': most_common_train_batch_size,
            'gradient_accumulation_steps': most_common_gradient_accumulation_steps,
            'warmup_ratio': avg_warmup_ratio,
            'max_grad_norm': avg_max_grad_norm,
            'max_length': most_common_max_length,
            'model_name': model
        }
        with open(f'{out_dir}/{model.replace("/", "_")}/best_params.json', 'w') as f:
            json.dump(best_params, f, indent=4)