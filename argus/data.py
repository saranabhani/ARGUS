import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from transformers import AutoTokenizer, DataCollatorWithPadding

from utils import compute_soft_label

def load_dataframe(cfg_data, split='train'):
    df = pd.read_csv(cfg_data["csv_path"]) if split == "train" else pd.read_csv(cfg_data["csv_path"].replace("train", "test"))
    ann_cols = cfg_data["annotator_cols"]
    label_col = cfg_data["label_col"]
    is_story = 'story' in label_col
    soft = df.apply(lambda r: compute_soft_label(r, ann_cols, story=is_story), axis=1)
    df["soft_label"] = soft
    df["hard_label"] = df[label_col]
    return df

@dataclass
class DataCollatorWithSoftLabels:
    tokenizer: AutoTokenizer

    def __call__(self, features):
        labels = torch.stack([torch.tensor(f["labels"], dtype=torch.float32) for f in features])
        hard = torch.tensor([int(f["hard_label"]) for f in features], dtype=torch.long)

        stripped = []
        for f in features:
            f = dict(f)
            f.pop("labels", None)
            f.pop("hard_label", None)
            stripped.append(f)

        pad_collator = DataCollatorWithPadding(self.tokenizer)
        batch = pad_collator(stripped)
        batch["labels"] = labels
        batch["hard_label"] = hard
        return batch
    
class SoftLabelDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, text_col, max_length):
        assert "soft_label" in df.columns, "missing soft_label column"
        assert "hard_label" in df.columns, "missing hard_label column"
        self.df = df.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row[self.text_col])
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )
        labels = np.array(row["soft_label"], dtype=np.float32)
        enc["labels"] = labels
        enc["hard_label"] = int(row["hard_label"])
        return {k: torch.tensor(v) if isinstance(v, list) else v for k, v in enc.items()}

class StoryDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, text_col, max_length, label_col):
        self.texts = df[text_col].tolist()
        self.labels = df[label_col].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False, 
        )
        enc["labels"] = int(self.labels[idx])
        return enc