"""Controlled GIPFormer RNN-T construction-domain adaptation.

This trainer deliberately uses the model factory and RNN-T loss implementation
from the same Icefall Zipformer source loaded by official GIPFormer inference.
It trains only prepared train/dev JSONL inputs, never a test split.  Checkpoints
are epoch-resumable and the current best dev-loss model is exported as a local
inference bundle (``best/model.pt``, ``bpe.model``, ``tokens.txt``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SAMPLE_RATE = 16000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Record:
    audio_path: Path
    text: str
    variant: str


def read_records(path: Path) -> list[Record]:
    records: list[Record] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}:{number}") from exc
        audio_path = Path(str(row.get("audio_path", "")))
        text = str(row.get("text", "")).strip()
        if not audio_path.is_file() or not text:
            raise ValueError(f"Invalid prepared record at {path}:{number}")
        records.append(Record(audio_path.resolve(), text, str(row.get("variant", "unknown"))))
    if not records:
        raise ValueError(f"No records in {path}")
    return records


def import_upstream(icefall_dir: Path) -> dict[str, Any]:
    recipe_dir = icefall_dir / "egs" / "librispeech" / "ASR" / "zipformer"
    for directory in (icefall_dir, icefall_dir / "egs" / "librispeech" / "ASR", recipe_dir):
        value = str(directory)
        if value not in sys.path:
            sys.path.insert(0, value)
    try:
        import k2
        import kaldifeat
        import sentencepiece as spm
        import torch
        import torchaudio
        from torch.nn.utils.rnn import pad_sequence
        from train import add_model_arguments, get_model, get_params
    except ImportError as exc:
        raise ImportError(
            "Run this script with the official GIPFormer uv PyTorch environment "
            "and pass its Icefall checkout via --icefall-dir. "
            f"Original import error: {exc}"
        ) from exc
    return locals()


def load_model(stack: dict[str, Any], model_dir: Path, device: Any):
    torch = stack["torch"]
    k2 = stack["k2"]
    get_model = stack["get_model"]
    get_params = stack["get_params"]
    add_model_arguments = stack["add_model_arguments"]
    tokens = model_dir / "tokens.txt"
    checkpoint_path = model_dir / "model.pt"
    if not tokens.is_file() or not checkpoint_path.is_file() or not (model_dir / "bpe.model").is_file():
        raise FileNotFoundError("model-dir must contain model.pt, bpe.model, and tokens.txt")

    token_table = k2.SymbolTable.from_file(tokens)
    params = get_params()
    # GIPFormer's official inference script obtains these architecture defaults
    # through train.add_model_arguments(parser) before calling get_model().
    # This trainer must construct the identical parameter set, not rely on the
    # smaller training-state dictionary returned by get_params().
    defaults_parser = argparse.ArgumentParser(add_help=False)
    add_model_arguments(defaults_parser)
    params.update(vars(defaults_parser.parse_args([])))
    params.blank_id = token_table["<blk>"]
    count = sum(1 for symbol in token_table.symbols if not symbol.startswith("#"))
    if params.blank_id == 0:
        count -= 1
    params.vocab_size = count + 1
    model = get_model(params)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Official checkpoint is not architecture-compatible: "
            f"missing={incompatible.missing_keys[:8]}, unexpected={incompatible.unexpected_keys[:8]}"
        )
    model.to(device)
    return model, params


def make_fbank(stack: dict[str, Any], device: Any):
    kaldifeat = stack["kaldifeat"]
    opts = kaldifeat.FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = SAMPLE_RATE
    opts.mel_opts.num_bins = 80
    opts.mel_opts.high_freq = -400
    return kaldifeat.Fbank(opts)


def make_batch(stack: dict[str, Any], records: list[Record], fbank: Any, device: Any, max_duration: float):
    torch = stack["torch"]
    torchaudio = stack["torchaudio"]
    pad_sequence = stack["pad_sequence"]
    waves = []
    texts = []
    for record in records:
        wave, sample_rate = torchaudio.load(record.audio_path)
        if sample_rate != SAMPLE_RATE:
            wave = torchaudio.functional.resample(wave, sample_rate, SAMPLE_RATE)
        wave = wave[0].contiguous()
        duration = wave.numel() / SAMPLE_RATE
        if duration <= 0 or duration > max_duration:
            raise ValueError(f"Unsupported audio duration ({duration:.2f}s): {record.audio_path}")
        waves.append(wave.to(device))
        texts.append(record.text)
    features_list = fbank(waves)
    lengths = torch.tensor([feature.size(0) for feature in features_list], device=device)
    features = pad_sequence(features_list, batch_first=True, padding_value=math.log(1e-10))
    return features, lengths, texts


def rnnt_loss(stack: dict[str, Any], model: Any, params: Any, features: Any, lengths: Any, texts: list[str], sp: Any):
    k2 = stack["k2"]
    targets = k2.RaggedTensor(sp.encode(texts, out_type=int))
    simple_loss, pruned_loss, _ = model(
        x=features,
        x_lens=lengths,
        y=targets,
        prune_range=params.prune_range,
        am_scale=params.am_scale,
        lm_scale=params.lm_scale,
    )[:3]
    return params.simple_loss_scale * simple_loss + pruned_loss


def batches(records: list[Record], batch_size: int, seed: int):
    order = list(records)
    random.Random(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def atomic_torch_save(torch: Any, payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_best_bundle(torch: Any, payload: dict[str, Any], model_dir: Path, output_dir: Path) -> None:
    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(torch, payload, best_dir / "model.pt")
    for name in ("bpe.model", "tokens.txt"):
        shutil.copy2(model_dir / name, best_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--icefall-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--save-every-steps", type=int, default=500)
    parser.add_argument("--resume", choices=("auto", "never"), default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch-size, and learning-rate must be positive")
    if args.device != "cuda":
        parser.error("Fine-tuning is GPU-only; use the benchmark notebook for CPU evaluation")

    stack = import_upstream(args.icefall_dir)
    torch = stack["torch"]
    spm = stack["spm"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for GIPFormer fine-tuning")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train = read_records(args.train)
    dev = read_records(args.dev)
    if {record.audio_path for record in train} & {record.audio_path for record in dev}:
        raise ValueError("Prepared train/dev audio leakage")
    args.output.mkdir(parents=True, exist_ok=True)
    model, params = load_model(stack, args.model_dir, device)
    params.prune_range = 5
    params.am_scale = 0.0
    params.lm_scale = 0.25
    params.simple_loss_scale = 0.5
    sp = spm.SentencePieceProcessor(model_file=str(args.model_dir / "bpe.model"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=not args.no_fp16)
    fbank = make_fbank(stack, device)

    last_path = args.output / "last.pt"
    best_loss = float("inf")
    start_epoch = 1
    global_step = 0
    if args.resume == "auto" and last_path.is_file():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state.get("train_sha256") != sha256(args.train) or state.get("dev_sha256") != sha256(args.dev):
            raise RuntimeError("Refusing resume: prepared train/dev manifests changed")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch_completed"]) + 1
        global_step = int(state["global_step"])
        best_loss = float(state["best_dev_loss"])
        print(f"Resuming after completed epoch {start_epoch - 1}; interrupted epochs are intentionally repeated.")

    run = {
        "source_model": str(args.model_dir.resolve()),
        "source_model_sha256": sha256(args.model_dir / "model.pt"),
        "icefall_dir": str(args.icefall_dir.resolve()),
        "train_sha256": sha256(args.train),
        "dev_sha256": sha256(args.dev),
        "train_records": len(train),
        "dev_records": len(dev),
        "test_split_included": False,
        "args": vars(args),
        "history": [],
    }
    print(json.dumps({key: run[key] for key in ("train_records", "dev_records", "test_split_included", "args")}, default=str, indent=2))

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_frames = 0
        for batch in batches(train, args.batch_size, args.seed + epoch):
            features, lengths, texts = make_batch(stack, batch, fbank, device, args.max_duration)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=not args.no_fp16):
                loss = rnnt_loss(stack, model, params, features, lengths, texts, sp)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            train_loss += float(loss.detach().cpu())
            train_frames += int(lengths.sum().detach().cpu())
            if global_step % 50 == 0:
                print(f"epoch={epoch} step={global_step} train_loss_per_frame={train_loss / max(train_frames, 1):.6f}", flush=True)
            if global_step % args.save_every_steps == 0:
                atomic_torch_save(torch, {"model": model.state_dict(), "epoch_completed": epoch - 1, "global_step": global_step, "best_dev_loss": best_loss, "train_sha256": sha256(args.train), "dev_sha256": sha256(args.dev)}, args.output / f"checkpoint-step-{global_step}.pt")

        model.eval()
        dev_loss = 0.0
        dev_frames = 0
        with torch.no_grad():
            for batch in batches(dev, args.batch_size, args.seed):
                features, lengths, texts = make_batch(stack, batch, fbank, device, args.max_duration)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=not args.no_fp16):
                    loss = rnnt_loss(stack, model, params, features, lengths, texts, sp)
                dev_loss += float(loss.detach().cpu())
                dev_frames += int(lengths.sum().detach().cpu())
        normalized_dev = dev_loss / max(dev_frames, 1)
        normalized_train = train_loss / max(train_frames, 1)
        epoch_record = {"epoch": epoch, "global_step": global_step, "train_loss_per_frame": normalized_train, "dev_loss_per_frame": normalized_dev}
        run["history"].append(epoch_record)
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "epoch_completed": epoch, "global_step": global_step, "best_dev_loss": min(best_loss, normalized_dev), "train_sha256": sha256(args.train), "dev_sha256": sha256(args.dev), "run": run}
        atomic_torch_save(torch, payload, args.output / f"epoch-{epoch:03d}.pt")
        atomic_torch_save(torch, payload, last_path)
        if normalized_dev < best_loss:
            best_loss = normalized_dev
            payload["best_dev_loss"] = best_loss
            save_best_bundle(torch, payload, args.model_dir, args.output)
            print(f"epoch={epoch}: new best dev loss {best_loss:.6f}", flush=True)
        else:
            print(f"epoch={epoch}: dev loss {normalized_dev:.6f}; best remains {best_loss:.6f}", flush=True)
        (args.output / "training_summary.json").write_text(json.dumps(run, default=str, indent=2), encoding="utf-8")

    print(json.dumps({"best_bundle": str(args.output / "best"), "best_dev_loss": best_loss, "epochs_completed": args.epochs}, indent=2))


if __name__ == "__main__":
    main()
