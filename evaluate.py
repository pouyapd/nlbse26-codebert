"""
NLBSE'26 Code Comment Classification — Evaluation Script
Loads saved models and prints per-category metrics + competition score.
"""

import time
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

MODEL_NAME = "microsoft/codebert-base"
MAX_LEN    = 128
OUTPUT_DIR = "models"

LANGS = ["java", "python", "pharo"]
LABEL_NAMES = {
    "java":   ["summary", "Ownership", "Expand", "usage", "Pointer", "deprecation", "rational"],
    "python": ["Usage", "Parameters", "DevelopmentNotes", "Expand", "Summary"],
    "pharo":  ["Keyimplementationpoints", "Example", "Responsibilities",
               "Intent", "Keymessages", "Collaborators"],
}

BASELINE_F1 = {
    "java":   {"summary": 0.879, "Ownership": 1.0, "Expand": 0.374, "usage": 0.867,
               "Pointer": 0.861, "deprecation": 0.778, "rational": 0.356},
    "python": {"Usage": 0.674, "Parameters": 0.719, "DevelopmentNotes": 0.305,
               "Expand": 0.540, "Summary": 0.672},
    "pharo":  {"Keyimplementationpoints": 0.600, "Example": 0.881, "Responsibilities": 0.681,
               "Intent": 0.783, "Keymessages": 0.579, "Collaborators": 0.167},
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CommentDataset(TorchDataset):
    def __init__(self, hf_split, tokenizer, max_len):
        self.texts   = [str(t) for t in hf_split["combo"]]
        self.labels  = hf_split["labels"]
        self.tok     = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.float),
        }


class CodeBERTClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        hidden          = self.encoder.config.hidden_size
        self.dropout    = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls))


def evaluate_lang(lang, model, ds, tokenizer):
    loader = DataLoader(
        CommentDataset(ds[f"{lang}_test"], tokenizer, MAX_LEN),
        batch_size=64, shuffle=False,
    )
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            preds  = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int)
            all_preds.append(preds)
            all_labels.append(batch["labels"].numpy().astype(int))

    y_pred = np.vstack(all_preds).T
    y_true = np.vstack(all_labels).T
    rows = []
    for i, cat in enumerate(LABEL_NAMES[lang]):
        tp = int(np.sum((y_true[i] == 1) & (y_pred[i] == 1)))
        fp = int(np.sum((y_true[i] == 0) & (y_pred[i] == 1)))
        fn = int(np.sum((y_true[i] == 1) & (y_pred[i] == 0)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2*tp) / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0.0
        rows.append({"lang": lang, "category": cat,
                     "precision": round(precision, 4),
                     "recall":    round(recall, 4),
                     "f1":        round(f1, 4)})
    return rows


if __name__ == "__main__":
    ds        = load_dataset("NLBSE/nlbse26-code-comment-classification")
    tokenizer = AutoTokenizer.from_pretrained(f"{OUTPUT_DIR}/tokenizer")

    all_scores, total_time, total_flops = [], 0, 0

    for lang in LANGS:
        model = CodeBERTClassifier(MODEL_NAME, len(LABEL_NAMES[lang])).to(device)
        model.load_state_dict(torch.load(f"{OUTPUT_DIR}/codebert_{lang}.pt", map_location=device))
        model.eval()

        x_texts = [str(t) for t in ds[f"{lang}_test"]["combo"]]
        enc = tokenizer(x_texts, max_length=MAX_LEN, padding="max_length",
                        truncation=True, return_tensors="pt")
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.profiler.profile(with_flops=True) as p:
            t0 = time.time()
            with torch.no_grad():
                for _ in range(10):
                    model(input_ids, attention_mask)
            total_time += time.time() - t0
        total_flops += sum(k.flops for k in p.key_averages()) / 1e9

        all_scores.extend(evaluate_lang(lang, model, ds, tokenizer))

    scores_df = pd.DataFrame(all_scores)
    avg_f1    = scores_df["f1"].mean()
    avg_rt    = total_time / 10
    avg_fl    = total_flops / 10
    comp_sc   = round(0.6*avg_f1 + 0.2*max((5-avg_rt)/5, 0) + 0.2*max((5000-avg_fl)/5000, 0), 4)

    rows = []
    for _, row in scores_df.iterrows():
        bl = BASELINE_F1[row["lang"]][row["category"]]
        rows.append({"lang": row["lang"], "category": row["category"],
                     "baseline_f1": bl, "ours_f1": row["f1"],
                     "delta": round(row["f1"] - bl, 4)})
    cmp = pd.DataFrame(rows)

    print(scores_df.to_string(index=False))
    print("\n" + "="*50)
    print(cmp.to_string(index=False))
    print(f"\nOverall avg F1 — Baseline: {cmp['baseline_f1'].mean():.4f}  |  Ours: {avg_f1:.4f}")
    print(f"Competition score: {comp_sc}")

    scores_df.to_csv("results.csv", index=False)
    print("\nSaved: results.csv")
