"""
NLBSE'26 Code Comment Classification — Training Script
Model: microsoft/codebert-base fine-tuned with weighted multi-label BCE loss
"""

import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from tqdm.auto import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME         = "microsoft/codebert-base"
MAX_LEN            = 128
EPOCHS             = 7
BATCH_SIZE         = 16
LR                 = 2e-5
CLASS_WEIGHT_SCALE = 10.0
OUTPUT_DIR         = "models"

LANGS = ["java", "python", "pharo"]
LABEL_NAMES = {
    "java":   ["summary", "Ownership", "Expand", "usage", "Pointer", "deprecation", "rational"],
    "python": ["Usage", "Parameters", "DevelopmentNotes", "Expand", "Summary"],
    "pharo":  ["Keyimplementationpoints", "Example", "Responsibilities",
               "Intent", "Keymessages", "Collaborators"],
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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


def train_model(lang, ds, tokenizer):
    num_labels   = len(LABEL_NAMES[lang])
    train_loader = DataLoader(
        CommentDataset(ds[f"{lang}_train"], tokenizer, MAX_LEN),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    model     = CodeBERTClassifier(MODEL_NAME, num_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    all_labels = torch.tensor(ds[f"{lang}_train"]["labels"], dtype=torch.float)
    pos_count  = all_labels.sum(dim=0).clamp(min=1)
    neg_count  = len(all_labels) - pos_count
    pos_weight = (neg_count / pos_count).clamp(max=CLASS_WEIGHT_SCALE).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"{lang} epoch {epoch+1}/{EPOCHS}"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(input_ids, attention_mask), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  [{lang}] epoch {epoch+1} loss: {total_loss/len(train_loader):.4f}")

    return model


if __name__ == "__main__":
    ds        = load_dataset("NLBSE/nlbse26-code-comment-classification")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for lang in LANGS:
        print(f"\n=== Training {lang} ===")
        model = train_model(lang, ds, tokenizer)
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/codebert_{lang}.pt")
        print(f"Saved: {OUTPUT_DIR}/codebert_{lang}.pt")

    tokenizer.save_pretrained(f"{OUTPUT_DIR}/tokenizer")
    print("\nAll models trained and saved.")
