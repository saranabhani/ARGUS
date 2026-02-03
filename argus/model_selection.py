import os
import json
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


def holm_adjust(pvals, alpha=0.05):

    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    p_sorted = [pvals[i] for i in order]

    reject_sorted = [False] * m
    for k, p in enumerate(p_sorted):
        thresh = alpha / (m - k)
        if p <= thresh:
            reject_sorted[k] = True
        else:
            break

    if any(reject_sorted):
        last_true = max(i for i, r in enumerate(reject_sorted) if r)
        for i in range(last_true + 1):
            reject_sorted[i] = True

    p_adj_sorted = [0.0] * m
    for k, p in enumerate(p_sorted):
        p_adj_sorted[k] = (m - k) * p

    for k in range(1, m):
        if p_adj_sorted[k] < p_adj_sorted[k - 1]:
            p_adj_sorted[k] = p_adj_sorted[k - 1]
    p_adj_sorted = [min(1.0, x) for x in p_adj_sorted]

    p_adj = [None] * m
    reject = [False] * m
    for sorted_idx, orig_idx in enumerate(order):
        p_adj[orig_idx] = p_adj_sorted[sorted_idx]
        reject[orig_idx] = reject_sorted[sorted_idx]

    return p_adj, reject

def model_sig_test(data_dir, cfg):
    with open(os.path.join(data_dir, "sig_test_report.txt"), 'w') as freport:
        folds_results = []
        metrics = [
            "eval_loss", "eval_accuracy", "eval_f1", "eval_macro_f1", "eval_micro_f1", "eval_weighted_f1",
            "eval_brier", "eval_wasserstein", "eval_mae_exp", "eval_rmse_exp",
        ]
        models = cfg["models"]
        for model in models:
            fold_file = os.path.join(data_dir, model.replace("/", "_"), "summary.json")
            with open(fold_file) as f:
                fold_data = json.load(f)
                outer_metrics = fold_data["outer_metrics"]
                for i, mets in enumerate(outer_metrics):
                    for met in mets:
                        folds_results.append({
                                "fold": i+1,
                                "model": model,
                                "metric": met,
                                "value": mets[met]
                            })
        df = pd.DataFrame(folds_results)
        p_vals = {}
        for metric in metrics:
            stat, p = friedmanchisquare(
                *[df[(df["metric"]==metric) & (df["model"]==m)]["value"].to_numpy() for m in models]
            )
            freport.write(f"\nMetric: {metric}\n")
            freport.write(f"Friedman statistic: {stat}, p-value: {p}\n")
            print(f"\nMetric: {metric}")
            print("Friedman statistic:", stat, "p-value:", p)
            if p < 0.05:
                freport.write(f"Significant differences found between models for metric {metric}: {p:.4f}\n")
                print(f"Significant differences found between models for metric {metric}: {p:.4f}")
            p_vals[metric] = p

        if any(p < 0.05 for p in p_vals.values()):
            freport.write("\nPost-hoc analysis (Wilcoxon) is needed for metrics with significant differences.\n")
            print("\nPost-hoc analysis (Wilcoxon) is needed for metrics with significant differences.")
            models_pairs = [(m1, m2) for i, m1 in enumerate(models) for j, m2 in enumerate(models) if i < j]
            lower_is_better = {"eval_loss","eval_brier","eval_wasserstein","eval_mae_exp","eval_rmse_exp"}
            for metric in metrics:
                alt = "less" if metric in lower_is_better else "greater"
                if p_vals[metric] < 0.05:
                    freport.write(f"\nPost-hoc Wilcoxon tests for metric: {metric}\n")
                    print(f"\nPost-hoc Wilcoxon tests for metric: {metric}")
                    stats = []
                    raw_ps = []
                    for m1, m2 in models_pairs:
                        stat, p_wilcoxon = wilcoxon(
                            df[(df["metric"]==metric) & (df["model"]==m1)]["value"].to_numpy(),
                            df[(df["metric"]==metric) & (df["model"]==m2)]["value"].to_numpy(),
                            alternative=alt
                        )
                        freport.write(f"  {m1} vs {m2}: statistic={stat}, p-value={p_wilcoxon:.4f}\n")
                        print(f"  {m1} vs {m2}: statistic={stat}, p-value={p_wilcoxon:.4f}")
                        if p_wilcoxon < 0.05:
                            freport.write(f"  Significant difference found between {m1} and {m2} for metric {metric}: {p_wilcoxon:.4f}\n")
                            print(f"  Significant difference found between {m1} and {m2} for metric {metric}: {p_wilcoxon:.4f}")
                        stats.append((m1, m2, stat, p_wilcoxon))
                        raw_ps.append(p_wilcoxon)
                    p_adj, reject = holm_adjust(raw_ps, alpha=0.05)
                    freport.write(f"\nHolm-Bonferroni adjusted p-values for metric: {metric}\n")
                    print(f"\nHolm-Bonferroni adjusted p-values for metric: {metric}")
                    for i, (m1, m2, stat, p_wilcoxon) in enumerate(stats):
                        freport.write(f"  {m1} vs {m2}: statistic={stat}, raw p-value={p_wilcoxon:.4f}, adjusted p-value={p_adj[i]:.4f}, reject H0: {reject[i]}\n")
                        print(f"  {m1} vs {m2}: statistic={stat}, raw p-value={p_wilcoxon:.4f}, adjusted p-value={p_adj[i]:.4f}, reject H0: {reject[i]}")


