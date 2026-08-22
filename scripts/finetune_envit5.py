"""Resumable, direction-scoped EnViT5 fine-tuning for OneVoice V2.

All durable state belongs under a caller-provided output directory (normally
Google Drive).  A Colab restart can therefore resume from the last completed
epoch without relying on /content.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.reporting import create_run_manifest


class ConstructionTranslationDataset(Dataset):
    def __init__(self, filepath: str | Path, tokenizer, direction: str, max_length: int):
        self.samples: list[tuple[str, str]] = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                vi, en = row.get("vi", "").strip(), row.get("en", "").strip()
                if vi and en:
                    source, target = (vi, en) if direction == "vi2en" else (en, vi)
                    self.samples.append((f"{direction[:2]}: {source}", target))
        if not self.samples:
            raise ValueError(f"No valid {direction} examples found in {path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source, target = self.samples[index]
        encoded = self.tokenizer(
            source,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_encoded = self.tokenizer(
            target,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        labels = target_encoded["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": labels,
        }


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _save_state(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def evaluate_loss(model, loader, device: str) -> float:
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            total += float(model(**batch).loss.item())
    model.train()
    return total / max(len(loader), 1)


def _load_resume_state(output: Path, resume: str) -> dict | None:
    if resume == "never":
        return None
    state_path = Path(resume) if resume != "auto" else output / "training_state.pt"
    if state_path.is_file():
        return torch.load(state_path, map_location="cpu", weights_only=False)
    if resume != "auto":
        raise FileNotFoundError(f"Resume state not found: {state_path}")
    if (output / "checkpoints").exists():
        raise RuntimeError(
            "Checkpoint directory exists but training_state.pt is missing. "
            "Use a new --output directory or explicitly choose --resume never; "
            "do not silently overwrite a possibly interrupted training run."
        )
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="VietAI/envit5-translation")
    parser.add_argument("--train", default="data/onevoice_construction_v2/train.csv")
    parser.add_argument("--dev", default="data/onevoice_construction_v2/dev.csv")
    parser.add_argument("--output", required=True, help="Persistent Drive directory for checkpoints")
    parser.add_argument("--direction", choices=["vi2en", "en2vi"], default="vi2en")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--resume",
        default="auto",
        help="auto (default), never, or explicit training_state.pt path",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "training_state.pt"
    state = _load_resume_state(output, args.resume)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    completed_epoch = int(state.get("completed_epoch", 0)) if state else 0
    if completed_epoch > args.epochs:
        raise ValueError(
            f"Checkpoint already completed epoch {completed_epoch}, above requested --epochs {args.epochs}"
        )
    if state:
        if state.get("direction") != args.direction:
            raise ValueError("Resume direction differs from checkpoint direction")
        checkpoint_dir = output / state["checkpoint_dir"]
        if not (checkpoint_dir / "config.json").is_file():
            raise FileNotFoundError(f"Resume checkpoint is incomplete: {checkpoint_dir}")
        source = str(checkpoint_dir)
        local_only = True
        print(f"Resuming after epoch {completed_epoch} from {checkpoint_dir}")
    else:
        source = args.model
        local_only = False
        print(f"Starting {args.direction} fine-tune from {source}")

    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=local_only)
    model = AutoModelForSeq2SeqLM.from_pretrained(source, local_files_only=local_only).to(device)
    train_data = ConstructionTranslationDataset(args.train, tokenizer, args.direction, args.max_length)
    dev_data = ConstructionTranslationDataset(args.dev, tokenizer, args.direction, args.max_length)
    dev_loader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=False)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    if state:
        optimizer.load_state_dict(state["optimizer_state"])
    best_dev = float(state.get("best_dev_loss", float("inf"))) if state else float("inf")
    history = list(state.get("history", [])) if state else []

    for epoch in range(completed_epoch, args.epochs):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        train_loader = DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, generator=generator
        )
        model.train()
        total = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        dev_loss = evaluate_loss(model, dev_loader, device)
        row = {
            "epoch": epoch + 1,
            "train_loss": total / len(train_loader),
            "dev_loss": dev_loss,
        }
        history.append(row)
        checkpoint_dir = output / "checkpoints" / f"epoch-{epoch + 1:03d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        _write_json_atomic(checkpoint_dir / "checkpoint_manifest.json", row)
        if dev_loss < best_dev:
            best_dev = dev_loss
            best_dir = output / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            _write_json_atomic(best_dir / "checkpoint_manifest.json", row)
        _write_json_atomic(output / "training_history.json", history)
        _save_state(
            state_path,
            {
                "completed_epoch": epoch + 1,
                "checkpoint_dir": checkpoint_dir.relative_to(output).as_posix(),
                "optimizer_state": optimizer.state_dict(),
                "best_dev_loss": best_dev,
                "history": history,
                "direction": args.direction,
                "model": args.model,
                "seed": args.seed,
            },
        )
        print(
            f"Epoch {epoch + 1}: train_loss={row['train_loss']:.4f}, dev_loss={dev_loss:.4f}; "
            f"checkpoint={checkpoint_dir}"
        )

    create_run_manifest(
        output / "run_manifest.json",
        command="finetune_envit5",
        inputs=[args.train, args.dev, output / "best"],
        metadata={
            **vars(args),
            "device": device,
            "best_dev_loss": best_dev,
            "completed_epoch": args.epochs,
            "resumed": bool(state),
        },
    )
    print(f"Best dev loss: {best_dev:.4f}; best checkpoint: {output / 'best'}")


if __name__ == "__main__":
    main()
