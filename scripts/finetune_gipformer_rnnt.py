"""Production-oriented GIPFormer RNN-T construction-domain adaptation.

The model factory and RNN-T loss come from the same pinned Icefall Zipformer
source used to prove GIPFormer PyTorch/ONNX compatibility. Fine-tuning keeps
the original model architecture intact so a later ONNX export remains valid.
The default notebook recipe updates encoder, decoder and joiner with Icefall's
ScaledAdam/Eden schedule; a decoder/joiner-only experiment is retained as a
separate opt-in diagnostic and must not be promoted without the dev gate.

Only prepared train/dev JSONL inputs are accepted; test is never loaded. The
single ``last.pt`` checkpoint includes model, optimizer, scaler and exact batch
position, so an interrupted Colab job resumes inside the epoch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor


SAMPLE_RATE = 16000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Record:
    audio_path: Path
    text: str
    variant: str
    duration_s: float


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
        # Do not open every WAV here: on Drive/FUSE that metadata scan can take
        # tens of minutes before training even starts. Prepared manifests may
        # carry duration_s; otherwise use a neutral value and let torchaudio
        # validate/decode the real file in make_batch.
        duration = row.get("duration_s", row.get("duration_seconds", 1.0))
        duration = float(duration)
        if duration <= 0:
            raise ValueError(f"Invalid duration at {path}:{number}")
        records.append(Record(audio_path.resolve(), text, str(row.get("variant", "unknown")), duration))
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
        from train import get_model, get_params, get_parser
        # GIPFormer's pinned Icefall checkout keeps the optimizer in the
        # recipe directory (``optim.py``), while parameter grouping lives in
        # ``icefall.utils``.  Do not assume a newer ``icefall.optim`` module.
        from optim import Eden, ScaledAdam
        from icefall.utils import get_parameter_groups_with_lrs
    except ImportError as exc:
        raise ImportError(
            "Run this script with the official GIPFormer uv PyTorch environment "
            "and pass its Icefall checkout via --icefall-dir. "
            f"Original import error: {exc}"
        ) from exc
    return locals()


def make_optimizer(stack: dict[str, Any], model: Any, params: Any, learning_rate: float):
    """Use Icefall's production optimizer/schedule for Zipformer adaptation."""
    ScaledAdam = stack["ScaledAdam"]
    Eden = stack["Eden"]
    get_parameter_groups_with_lrs = stack["get_parameter_groups_with_lrs"]
    groups = get_parameter_groups_with_lrs(
        model, lr=learning_rate, include_names=True
    )
    optimizer = ScaledAdam(groups, lr=learning_rate, clipping_scale=2.0)
    scheduler = Eden(optimizer, params.lr_batches, params.lr_epochs)
    return optimizer, scheduler


def load_model(stack: dict[str, Any], model_dir: Path, device: Any):
    torch = stack["torch"]
    k2 = stack["k2"]
    get_model = stack["get_model"]
    get_params = stack["get_params"]
    get_parser = stack["get_parser"]
    tokens = model_dir / "tokens.txt"
    checkpoint_path = model_dir / "model.pt"
    if not tokens.is_file() or not checkpoint_path.is_file() or not (model_dir / "bpe.model").is_file():
        raise FileNotFoundError("model-dir must contain model.pt, bpe.model, and tokens.txt")

    token_table = k2.SymbolTable.from_file(tokens)
    params = get_params()
    # The upstream model factory depends on defaults spread across both its
    # model and trainer parser (e.g. context_size, use_transducer).  Start
    # from that complete parser so the model shape exactly matches inference.
    defaults_parser = get_parser()
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
    # The published checkpoint also carries an auxiliary CTC head.  This
    # recipe intentionally adapts the RNN-T path only, so those two extra
    # tensors are expected and safely ignored.  Any other mismatch is fatal.
    unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("ctc_output.")]
    if incompatible.missing_keys or unexpected:
        raise RuntimeError(
            "Official checkpoint is not architecture-compatible: "
            f"missing={incompatible.missing_keys[:8]}, unexpected={unexpected[:8]}"
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


def _cached_path(path: Path, cache_dir: Path) -> Path:
    key = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{key}_{path.name}"


def _load_wave(torchaudio: Any, record: Record, sample_rate: int, cache_dir: Path | None):
    source = record.audio_path
    if cache_dir is not None:
        source = _cached_path(source, cache_dir)
        source.parent.mkdir(parents=True, exist_ok=True)
        if not source.is_file():
            shutil.copy2(record.audio_path, source)
    wave_data, loaded_rate = torchaudio.load(source)
    if loaded_rate != sample_rate:
        wave_data = torchaudio.functional.resample(wave_data, loaded_rate, sample_rate)
    return wave_data[0].contiguous()


def make_batch(
    stack: dict[str, Any],
    records: list[Record],
    fbank: Any,
    device: Any,
    max_duration: float,
    load_workers: int = 1,
    cache_dir: Path | None = None,
):
    torch = stack["torch"]
    torchaudio = stack["torchaudio"]
    pad_sequence = stack["pad_sequence"]
    if any(record.duration_s > max_duration for record in records):
        record = next(record for record in records if record.duration_s > max_duration)
        raise ValueError(f"Unsupported audio duration ({record.duration_s:.2f}s): {record.audio_path}")
    worker_count = max(1, min(int(load_workers), len(records)))
    loader = lambda record: _load_wave(torchaudio, record, SAMPLE_RATE, cache_dir)
    if worker_count == 1:
        waves = [loader(record) for record in records]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            waves = list(executor.map(loader, records))
    texts = [record.text for record in records]
    waves = [wave.to(device) for wave in waves]
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


def configure_trainable_parameters(model: Any, prefixes: list[str]) -> tuple[list[Any], dict[str, int]]:
    """Freeze all parameters except complete modules matched by ``prefixes``.

    Prefix matching is deliberately exact at a module boundary. For example,
    ``decoder`` accepts ``decoder.*`` but not an accidental ``decoder_extra``.
    """
    normalized = [prefix.strip().rstrip(".") for prefix in prefixes if prefix.strip()]
    if not normalized:
        raise ValueError("At least one --trainable-prefix is required")
    matched = {prefix: 0 for prefix in normalized}
    trainable: list[Any] = []
    trainable_count = 0
    total_count = 0
    for name, parameter in model.named_parameters():
        total_count += parameter.numel()
        selected = False
        for prefix in normalized:
            if name == prefix or name.startswith(prefix + "."):
                matched[prefix] += parameter.numel()
                selected = True
        parameter.requires_grad_(selected)
        if selected:
            trainable.append(parameter)
            trainable_count += parameter.numel()
    missing = [prefix for prefix, count in matched.items() if count == 0]
    if missing:
        raise ValueError(f"No model parameters match --trainable-prefix: {missing}")
    return trainable, {
        "total": total_count,
        "trainable": trainable_count,
        "frozen": total_count - trainable_count,
    }


def batches(records: list[Record], batch_size: int, seed: int, bucket_size: int = 1):
    order = list(records)
    rng = random.Random(seed)
    rng.shuffle(order)
    if bucket_size <= 1:
        for start in range(0, len(order), batch_size):
            yield order[start : start + batch_size]
        return
    # Sort only small shuffled pools. This keeps stochasticity while making
    # each batch contain similarly long utterances, greatly reducing padding
    # and RNN-T compute on heterogeneous construction speech.
    pool_width = max(batch_size, batch_size * bucket_size)
    grouped: list[list[Record]] = []
    for pool_start in range(0, len(order), pool_width):
        pool = order[pool_start : pool_start + pool_width]
        pool.sort(key=lambda record: record.duration_s)
        grouped.extend(pool[start : start + batch_size] for start in range(0, len(pool), batch_size))
    rng.shuffle(grouped)
    yield from grouped


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


def make_resume_payload(
    model: Any,
    optimizer: Any,
    scaler: Any,
    scheduler: Any,
    *,
    epoch: int,
    next_batch_index: int,
    global_step: int,
    train_loss: float,
    train_frames: int,
    best_dev_loss: float,
    train_sha256: str,
    dev_sha256: str,
    trainable_prefixes: list[str],
    run: dict[str, Any],
) -> dict[str, Any]:
    """A complete, atomic Colab-resume state; no model-only step snapshots."""
    return {
        "recipe": "gipformer_icefall_ft_v2",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "next_batch_index": next_batch_index,
        "global_step": global_step,
        "train_loss": train_loss,
        "train_frames": train_frames,
        "best_dev_loss": best_dev_loss,
        "train_sha256": train_sha256,
        "dev_sha256": dev_sha256,
        "trainable_prefixes": trainable_prefixes,
        "run": run,
    }


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
    parser.add_argument(
        "--trainable-prefix",
        action="append",
        default=None,
        help="Complete module prefix to train; repeatable. Defaults to decoder + joiner.",
    )
    parser.add_argument(
        "--bucket-size",
        type=int,
        default=32,
        help="Approximate duration bucketing factor; 1 disables bucketing",
    )
    parser.add_argument(
        "--load-workers",
        type=int,
        default=2,
        help="Parallel WAV readers (use 1 if Drive throttles concurrent reads)",
    )
    parser.add_argument(
        "--cache-audio-dir",
        type=Path,
        default=None,
        help="Optional local SSD cache (e.g. /content/gipformer_audio_cache)",
    )
    parser.add_argument("--resume", choices=("auto", "never"), default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--optimizer", choices=("icefall", "adamw"), default="icefall")
    parser.add_argument("--lr-batches", type=float, default=5000.0)
    parser.add_argument("--lr-epochs", type=float, default=3.0)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch-size, and learning-rate must be positive")
    if args.bucket_size < 1 or args.load_workers < 1:
        parser.error("bucket-size and load-workers must be positive")
    if args.device != "cuda":
        parser.error("Fine-tuning is GPU-only; use the benchmark notebook for CPU evaluation")

    stack = import_upstream(args.icefall_dir)
    torch = stack["torch"]
    spm = stack["spm"]
    warnings.filterwarnings("ignore", category=UserWarning, module=r"torchaudio")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for GIPFormer fine-tuning")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    trainable_prefixes = args.trainable_prefix or ["encoder", "decoder", "joiner"]

    train = read_records(args.train)
    dev = read_records(args.dev)
    if {record.audio_path for record in train} & {record.audio_path for record in dev}:
        raise ValueError("Prepared train/dev audio leakage")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.cache_audio_dir is not None:
        args.cache_audio_dir.mkdir(parents=True, exist_ok=True)
        print(f"Audio cache enabled: {args.cache_audio_dir.resolve()}", flush=True)
    model, params = load_model(stack, args.model_dir, device)
    trainable_parameters, parameter_counts = configure_trainable_parameters(
        model, trainable_prefixes
    )
    print(
        "Trainable modules=" + ",".join(trainable_prefixes)
        + f"; parameters={parameter_counts['trainable']:,}/{parameter_counts['total']:,} "
        + f"({parameter_counts['trainable'] / max(parameter_counts['total'], 1):.2%})",
        flush=True,
    )
    params.prune_range = 5
    params.am_scale = 0.0
    params.lm_scale = 0.25
    params.simple_loss_scale = 0.5
    params.lr_batches = args.lr_batches
    params.lr_epochs = args.lr_epochs
    sp = spm.SentencePieceProcessor(model_file=str(args.model_dir / "bpe.model"))
    if args.optimizer == "icefall":
        optimizer, scheduler = make_optimizer(stack, model, params, args.learning_rate)
    else:
        optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=0.01)
        scheduler = None
    scaler = torch.amp.GradScaler("cuda", enabled=not args.no_fp16)
    fbank = make_fbank(stack, device)

    last_path = args.output / "last.pt"
    best_loss = float("inf")
    start_epoch = 1
    start_batch_index = 0
    global_step = 0
    resume_train_loss = 0.0
    resume_train_frames = 0
    if args.resume == "auto" and last_path.is_file():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state.get("train_sha256") != sha256(args.train) or state.get("dev_sha256") != sha256(args.dev):
            raise RuntimeError("Refusing resume: prepared train/dev manifests changed")
        if state.get("recipe") not in (None, "gipformer_head_ft_v1", "gipformer_icefall_ft_v2"):
            raise RuntimeError(f"Refusing resume from a different recipe: {state.get('recipe')}")
        saved_prefixes = state.get("trainable_prefixes")
        if saved_prefixes is not None and list(saved_prefixes) != list(trainable_prefixes):
            raise RuntimeError("Refusing resume: --trainable-prefix changed")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        if scheduler is not None and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        if "next_batch_index" in state:
            start_epoch = int(state["epoch"])
            start_batch_index = int(state["next_batch_index"])
            resume_train_loss = float(state.get("train_loss", 0.0))
            resume_train_frames = int(state.get("train_frames", 0))
        else:
            # Legacy epoch-only checkpoint: resume at the following epoch.
            start_epoch = int(state["epoch_completed"]) + 1
        global_step = int(state["global_step"])
        best_loss = float(state["best_dev_loss"])
        print(
            f"Resuming epoch {start_epoch} at batch {start_batch_index}; global step {global_step}.",
            flush=True,
        )

    run = {
        "source_model": str(args.model_dir.resolve()),
        "source_model_sha256": sha256(args.model_dir / "model.pt"),
        "icefall_dir": str(args.icefall_dir.resolve()),
        "train_sha256": sha256(args.train),
        "dev_sha256": sha256(args.dev),
        "train_records": len(train),
        "dev_records": len(dev),
        "test_split_included": False,
        "recipe": "gipformer_icefall_ft_v2",
        "optimizer": args.optimizer,
        "trainable_prefixes": trainable_prefixes,
        "parameter_counts": parameter_counts,
        "args": vars(args),
        "history": [],
    }
    print(json.dumps({key: run[key] for key in ("train_records", "dev_records", "test_split_included", "args")}, default=str, indent=2))

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = resume_train_loss if epoch == start_epoch else 0.0
        train_frames = resume_train_frames if epoch == start_epoch else 0
        epoch_batches = list(batches(train, args.batch_size, args.seed + epoch, args.bucket_size))
        if start_batch_index > len(epoch_batches):
            raise RuntimeError(
                f"Resume batch {start_batch_index} is outside epoch {epoch} ({len(epoch_batches)} batches)"
            )
        for batch_index, batch in enumerate(epoch_batches):
            if epoch == start_epoch and batch_index < start_batch_index:
                continue
            step_started = time.perf_counter()
            features, lengths, texts = make_batch(
                stack, batch, fbank, device, args.max_duration, args.load_workers, args.cache_audio_dir
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=not args.no_fp16):
                loss = rnnt_loss(stack, model, params, features, lengths, texts, sp)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 5.0)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step_batch(global_step)
            global_step += 1
            train_loss += float(loss.detach().cpu())
            train_frames += int(lengths.sum().detach().cpu())
            if global_step <= 5 or global_step % 10 == 0:
                elapsed = time.perf_counter() - step_started
                print(
                    f"epoch={epoch} step={global_step} train_loss_per_frame="
                    f"{train_loss / max(train_frames, 1):.6f} batch_seconds={elapsed:.2f}",
                    flush=True,
                )
            if global_step % args.save_every_steps == 0:
                atomic_torch_save(
                    torch,
                    make_resume_payload(
                        model,
                        optimizer,
                        scaler,
                        scheduler,
                        epoch=epoch,
                        next_batch_index=batch_index + 1,
                        global_step=global_step,
                        train_loss=train_loss,
                        train_frames=train_frames,
                        best_dev_loss=best_loss,
                        train_sha256=sha256(args.train),
                        dev_sha256=sha256(args.dev),
                        trainable_prefixes=trainable_prefixes,
                        run=run,
                    ),
                    last_path,
                )

        model.eval()
        dev_loss = 0.0
        dev_frames = 0
        with torch.no_grad():
            for batch in batches(dev, args.batch_size, args.seed, args.bucket_size):
                features, lengths, texts = make_batch(
                    stack, batch, fbank, device, args.max_duration, args.load_workers, args.cache_audio_dir
                )
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=not args.no_fp16):
                    loss = rnnt_loss(stack, model, params, features, lengths, texts, sp)
                dev_loss += float(loss.detach().cpu())
                dev_frames += int(lengths.sum().detach().cpu())
        if scheduler is not None:
            scheduler.step_epoch(epoch - 1)
        normalized_dev = dev_loss / max(dev_frames, 1)
        normalized_train = train_loss / max(train_frames, 1)
        epoch_record = {"epoch": epoch, "global_step": global_step, "train_loss_per_frame": normalized_train, "dev_loss_per_frame": normalized_dev}
        run["history"].append(epoch_record)
        payload = make_resume_payload(
            model,
            optimizer,
            scaler,
            scheduler,
            epoch=epoch + 1,
            next_batch_index=0,
            global_step=global_step,
            train_loss=0.0,
            train_frames=0,
            best_dev_loss=min(best_loss, normalized_dev),
            train_sha256=sha256(args.train),
            dev_sha256=sha256(args.dev),
            trainable_prefixes=trainable_prefixes,
            run=run,
        )
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
