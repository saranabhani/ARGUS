import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import wasserstein_distance

_VALUES_1TO5 = np.arange(1, 6, dtype=np.float32)
_VALUES_0TO1 = np.arange(0, 2, dtype=np.float32)

def make_splitter(n_splits, y):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7)

def subset_df(df, idx):
    return df.iloc[idx].reset_index(drop=True)

def balanced_class_weights_from_counts(counts, eps=1e-8, normalize_mean_one=True):
    counts = np.asarray(counts, dtype=np.float64) + eps
    N, K = counts.sum(), len(counts)
    w = N / (K * counts)          
    if normalize_mean_one:
        w *= K / w.sum()   
    return w.astype(np.float32)

def classes_metrics(true, pred):
    acc = accuracy_score(true, pred)
    macro_f1 = f1_score(true, pred, average="macro")
    micro_f1 = f1_score(true, pred, average="micro")
    f1 = f1_score(true, pred, average="binary")
    weighted_f1 = f1_score(true, pred, average="weighted")
    return {
        "accuracy": acc,
        "f1": f1,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
    }

def binarize_expected(y_exp, story=False):
    threshold = 0.5 if story else 2.5
    return int(y_exp >= threshold)

def expected_likert(probs, story= False):
    values = _VALUES_0TO1 if story else _VALUES_1TO5
    return probs @ values

def rmse_expected_likert(p_targets, q_preds, story):
    y_true = expected_likert(p_targets, story=story)
    y_pred = expected_likert(q_preds, story=story)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

def mae_expected_likert(p_targets, q_preds, story=False):
    y_true = expected_likert(p_targets, story=story)
    y_pred = expected_likert(q_preds, story=story)
    return float(np.mean(np.abs(y_pred - y_true)))

def brier_multiclass(probs, targets):
    return float(np.mean(np.sum((probs - targets) ** 2, axis=1)))

def scalar_metrics(targets, probs, story= False):
    brier = brier_multiclass(targets, probs)
    rmse_exp = rmse_expected_likert(targets, probs, story=story)
    mae_exp = mae_expected_likert(targets, probs, story=story)

    labels = np.array([0, 1]) if story else np.array([1, 2, 3, 4, 5], dtype=float)
    w_per_row = [
        wasserstein_distance(labels, labels, u_weights=t, v_weights=p)
        for t, p in zip(targets, probs)
    ]
    wasserstein = float(np.mean(w_per_row))
    return {
        "brier": brier,
        "mae_exp": mae_exp,
        "rmse_exp": rmse_exp,
        "wasserstein": wasserstein,
    }

def metrics_from_logits(logits, soft_targets, story= False):
    probs = torch.softmax(torch.tensor(logits), dim=-1).cpu().numpy()
    if story:
        hard_true = soft_targets.argmax(axis=-1)
        hard_pred = probs.argmax(axis=-1)
    else:
        hard_true = [binarize_expected(score, story) for score in expected_likert(soft_targets, story)]
        hard_pred = [binarize_expected(score, story) for score in expected_likert(probs, story)]
    cl_metrics = classes_metrics(hard_true, hard_pred)
    sc_metrics = scalar_metrics(soft_targets, probs, story=story)
    return {**cl_metrics, **sc_metrics}

def compute_metrics_binary(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = probs.argmax(axis=-1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")
    macro_f1 = f1_score(labels, preds, average="macro")
    micro_f1 = f1_score(labels, preds, average="micro")
    weighted_f1 = f1_score(labels, preds, average="weighted")
    precision = precision_score(labels, preds, average="binary", pos_label=1)
    recall = recall_score(labels, preds, average="binary", pos_label=1)

    return {
        "accuracy": acc,
        "f1": f1,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "precision": precision,
        "recall": recall,
    }

def compute_metrics_fn(eval_pred):
    if hasattr(eval_pred, "predictions"):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        logits, labels = eval_pred

    def _np(x): return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    logits = _np(logits)
    if isinstance(labels, (list, tuple)):
        soft_targets, hard_true = None, None
        for arr in labels:
            a = _np(arr)
            if a.ndim == 2: soft_targets = a
            elif a.ndim == 1: hard_true = a.astype(int)
    else:
        soft_targets = _np(labels)

    m = metrics_from_logits(logits, soft_targets, story=False)
    return {f"{k}": v for k, v in {
        "accuracy": m["accuracy"],
        "f1": m["f1"],
        "weighted_f1": m["weighted_f1"],
        "macro_f1": m["macro_f1"],
        "micro_f1": m["micro_f1"],
        "brier": m["brier"],
        "mae_exp": m["mae_exp"],
        "rmse_exp": m["rmse_exp"],
        "wasserstein": m["wasserstein"],
    }.items()}

def compute_metrics_fn_story(eval_pred):
    if hasattr(eval_pred, "predictions"):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        logits, labels = eval_pred

    def _np(x): return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    logits = _np(logits)
    if isinstance(labels, (list, tuple)):
        soft_targets, hard_true = None, None
        for arr in labels:
            a = _np(arr)
            if a.ndim == 2: soft_targets = a
            elif a.ndim == 1: hard_true = a.astype(int)
        if soft_targets is None: raise ValueError("Expected 2D soft targets in labels.")
    else:
        soft_targets = _np(labels)

    m = metrics_from_logits(logits, soft_targets, story=True)
    return {f"{k}": v for k, v in {
        "accuracy": m["accuracy"],
        "f1": m["f1"],
        "weighted_f1": m["weighted_f1"],
        "macro_f1": m["macro_f1"],
        "micro_f1": m["micro_f1"],
        "brier": m["brier"],
        "mae_exp": m["mae_exp"],
        "rmse_exp": m["rmse_exp"],
        "wasserstein": m["wasserstein"],
    }.items()}

def compute_all_metrics(probs, y_soft, y_hard, binary_threshold=0.5, story=False):
    hard_true = y_hard
    hard_pred = probs.argmax(axis=-1)
    if not story:
        hard_true = [binarize_expected(score, story) for score in expected_likert(y_soft, story)]
        hard_pred = [binarize_expected(score, story) for score in expected_likert(probs, story)]
    cl_metrics = classes_metrics(hard_true, hard_pred)
    sc_metrics = scalar_metrics(y_soft, probs, story=story)
    return {**cl_metrics, **sc_metrics}