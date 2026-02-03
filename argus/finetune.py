import os
import time
import argparse
import pandas as pd
import yaml
import torch
from utils import load_config, set_all_seeds, ensure_dir, select_best_params
from data import load_dataframe
from runner import run_model_nested_cv
from model_selection import model_sig_test

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    #TODO: add rest of the args

    cfg = load_config(args.config)

    set_all_seeds(cfg["training"]["seed"])

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(cfg["output"]["root_dir"], f"{cfg['output']['run_name']}_{cfg['training']['type']}_{ts}")
    ensure_dir(out_root)
    # save used config
    with open(os.path.join(out_root, "used_config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    # load dataframe
    df = load_dataframe(cfg["data"], split="train")

    all_results = []
    for model_name in cfg["models"]:
        res = run_model_nested_cv(model_name, df, cfg, out_root)
        all_results.append(res)

    rows = []
    for r in all_results:
        agg = r["aggregate"]
        row = {"model": r["model_name"], **{f"agg.{k}": v for k, v in agg.items()}}
        rows.append(row)

    df_sum = pd.DataFrame(rows)
    df_sum.to_csv(os.path.join(out_root, "summary_results.csv"), index=False)
    print(f"\nSummary results saved to {os.path.join(out_root, 'summary_results.csv')}")
    print("\n=== FINAL SUMMARY ===")
    print(df_sum.to_string(index=False))

    # run significance tests
    model_sig_test(out_root, cfg)

    # save best parameters
    select_best_params(out_root, cfg)

if __name__ == "__main__":
    main()
