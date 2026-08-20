"""
[Colab] Fine-tune VietAI/envit5-translation with Construction Terminology
========================================================================
Fine-tunes VietAI/envit5-translation on 8,064 bi-directional construction
utterances to ensure domain accuracy for industrial speech translation.

Target Model: VietAI/envit5-translation (~600MB)
"""

import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────
MODEL_NAME  = "VietAI/envit5-translation"
DATA_FILE   = "data/onevoice_construction_v2/test.csv"
OUTPUT_DIR  = "envit5_finetuned_construction"
EPOCHS      = 3
BATCH_SIZE  = 8
LR          = 3e-5
MAX_LENGTH  = 128
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# ── Dataset ───────────────────────────────────────────────────────────────
class ConstructionTranslationDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=128):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vi = row.get("vi_text", "").strip()
                    en = row.get("en_text", "").strip()
                    if vi and en:
                        # Add EnViT5 prefixes
                        self.samples.append(("vi: " + vi, en))
                        self.samples.append(("en: " + en, vi))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, tgt = self.samples[idx]
        enc = self.tokenizer(
            src, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        tgt_enc = self.tokenizer(
            tgt, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        labels = tgt_enc["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": labels,
        }

# ── Load Model & Tokenizer ────────────────────────────────────────────────
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)

if os.path.exists(DATA_FILE):
    dataset = ConstructionTranslationDataset(DATA_FILE, tokenizer, MAX_LENGTH)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=LR)

    print(f"Dataset size: {len(dataset)} sentence pairs")

    # ── Training Loop ─────────────────────────────────────────────────────────
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} — Loss: {avg_loss:.4f}")

    # Save Fine-tuned Model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n✅ Fine-tuned model saved to: {OUTPUT_DIR}/")
