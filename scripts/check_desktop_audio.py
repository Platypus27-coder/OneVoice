"""Check repeated TTS, optionally online and/or playing to a selected PC output.

This does not record a microphone or load ASR/MT models. Device playback
completion is not proof that a listener heard the correct speech.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tts.tts_engine import TTSEngine
from audio.devices import parse_output_device


PROMPTS = {
    "en2vi": (
        "Kiểm tra mũ bảo hộ trước khi tiếp tục.",
        "Có vấn đề với vùng nguy hiểm.",
        "Hệ thống đang hoạt động bình thường.",
    ),
    "vi2en": (
        "Check the safety helmet before continuing.",
        "There is a problem with the hazard zone.",
        "The system is operating normally.",
    ),
}


def run_check(config: dict, direction: str, repeats: int, play: bool,
              backend: str = "auto", allow_online_tts: bool = False) -> dict:
    engine = TTSEngine(
        config,
        profile="edge",
        offline=not allow_online_tts,
        backend=backend,
        allow_online_tts=allow_online_tts,
    )
    report = {
        "schema_version": 1,
        "direction": direction,
        "offline": not allow_online_tts,
        "tts_backend_requested": backend,
        "online_tts_enabled": allow_online_tts,
        "requested_calls": repeats,
        "playback_requested": play,
        "output_device": config["audio"].get("output_device"),
        "listener_confirmation": "not_recorded",
        "intelligibility_passed": None,
        "scope": "TTS/output-device smoke only; not microphone/ASR/MT or audible-listener proof",
        "calls": [],
        "passed": False,
        "error": None,
    }
    try:
        engine.load(direction=direction)
        report["engine"] = engine.engine_name(direction)
        for index in range(repeats):
            text = PROMPTS[direction][index % len(PROMPTS[direction])]
            started = time.perf_counter()
            audio, sample_rate = engine.synthesize(text, direction=direction)
            if engine.is_silence(audio) or not np.isfinite(audio).all():
                raise RuntimeError("TTS produced silence or non-finite samples")
            row = {
                "call": index + 1,
                "text": text,
                "samples": len(audio),
                "sample_rate": sample_rate,
                "synthesis_ms": round((time.perf_counter() - started) * 1000, 3),
                "device_playback_completed": False,
            }
            if play:
                row.update(engine.play(audio, sample_rate=sample_rate))
            report["calls"].append(row)
            print(f"[PASS technical] {direction} call={index + 1}/{repeats} playback={row['device_playback_completed']}", flush=True)
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[FAIL] {report['error']}", flush=True)
    return report


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--direction", choices=("vi2en", "en2vi"), required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--play", action="store_true", help="Actually play each synthesized sentence; no microphone recording")
    parser.add_argument("--output-device", type=parse_output_device)
    parser.add_argument("--tts-backend", choices=("auto", "sapi", "espeak", "gtts"), default="auto")
    parser.add_argument(
        "--allow-online-tts",
        action="store_true",
        help="Allow gTTS to send each prompt to Google Translate TTS; pair with --tts-backend gtts",
    )
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.tts_backend == "gtts" and not args.allow_online_tts:
        parser.error("gTTS sends prompt text online; also pass --allow-online-tts")
    if args.allow_online_tts and args.tts_backend != "gtts":
        parser.error("--allow-online-tts requires --tts-backend gtts")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.output_device is not None:
        config["audio"]["output_device"] = args.output_device
    report = run_check(
        config,
        args.direction,
        args.repeats,
        args.play,
        backend=args.tts_backend,
        allow_online_tts=args.allow_online_tts,
    )
    args.report_dir.mkdir(parents=True, exist_ok=True)
    destination = args.report_dir / "desktop_audio_check.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {destination.resolve()}")
    if not report["passed"]:
        raise SystemExit(1)
    if args.play:
        print("Technical playback completed; intelligibility has NOT been assessed. A listener must confirm that all sentences were clear and in the correct language.")


if __name__ == "__main__":
    main()
