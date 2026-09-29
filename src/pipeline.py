"""OneVoice V2 speech-to-speech runtime (VI ↔ EN).

The public CLI remains compatible with V1 while adding explicit runtime
profiles, offline preflight, construction context and deterministic safety
handling. Microphone capture is still endpoint-driven; rolling ASR hooks live
behind the shared V2 contracts and can be enabled without changing later stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__))

from context.engine import ConstructionContextEngine
from audio.devices import parse_input_device, parse_output_device
from context.site_pack import SitePackLoader
from contracts import ASRHypothesis, AudioFrame, CommitKind, SynthesizedChunk
from runtime.preflight import verify_artifacts
from safety.audio_store import SafetyAudioStore
from streaming.semantic_commit import (
    RollingHypothesisAssembler,
    SemanticCommitController,
    StablePrefixAligner,
)
from streaming.session import RollingUtteranceSession
from utils.srt_generator import SRTGenerator
from utils.text_normalizer import normalize_asr


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class OneVoicePipeline:
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        direction: str = "vi2en",
        profile: str | None = None,
        site_pack_path: str | None = None,
        offline: bool = False,
        report_dir: str | None = None,
        vad_energy_threshold: float | None = None,
        output_device: int | str | None = None,
        input_device: int | str | None = None,
        tts_backend: str | None = None,
        allow_online_tts: bool = False,
        windows_en_voice: str | None = None,
        windows_en_rate: int | None = None,
        event_callback=None,
        pause_mic_during_playback: bool = False,
    ):
        if direction not in {"vi2en", "en2vi"}:
            raise ValueError("direction must be 'vi2en' or 'en2vi'")
        self.cfg = load_config(config_path)
        if windows_en_voice is not None:
            if not windows_en_voice.strip():
                raise ValueError("windows_en_voice cannot be empty")
            self.cfg["tts"]["windows_en_voice"] = windows_en_voice.strip()
        if windows_en_rate is not None:
            if not -10 <= windows_en_rate <= 10:
                raise ValueError("windows_en_rate must be between -10 and 10")
            self.cfg["tts"]["windows_en_rate"] = windows_en_rate
        if vad_energy_threshold is not None:
            if not 0 < vad_energy_threshold <= 1:
                raise ValueError("vad_energy_threshold must be greater than 0 and at most 1")
            self.cfg["audio"]["vad_energy_threshold"] = float(vad_energy_threshold)
            print(f"[VAD] Energy threshold override: {vad_energy_threshold:g}")
        if output_device is not None:
            self.cfg["audio"]["output_device"] = parse_output_device(str(output_device))
        if input_device is not None:
            self.cfg["audio"]["input_device"] = parse_input_device(str(input_device))
        self.direction = direction
        self.profile = profile or self.cfg["pipeline"].get("profile", "development")
        if self.profile not in self.cfg.get("profiles", {}):
            raise ValueError(f"Unknown runtime profile: {self.profile}")
        self.offline = bool(offline or self.cfg["pipeline"].get("offline") or self.profile == "edge")
        self.tts_backend = tts_backend
        self.allow_online_tts = bool(allow_online_tts)
        if self.tts_backend == "gtts" and not self.allow_online_tts:
            raise ValueError("gTTS sends translated text to Google; add --allow-online-tts explicitly")
        if self.allow_online_tts and self.tts_backend != "gtts":
            raise ValueError("--allow-online-tts is only valid together with --tts-backend gtts")
        self.report_dir = Path(report_dir) if report_dir else None
        self.stop_event = threading.Event()
        self._fatal_error: BaseException | None = None
        self.event_callback = event_callback
        self.pause_mic_during_playback = bool(pause_mic_during_playback)

        # GIPFormer loads sherpa-onnx before the edge translator is initialized.
        # On some Windows Conda installations, importing ONNX Runtime afterwards
        # (through Optimum) can terminate Python with 0xC0000005 while loading
        # onnxruntime's native extension. Prime the shared ONNX Runtime first.
        if self.profile == "edge" and self.direction == "vi2en":
            try:
                import onnxruntime  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "The vi2en edge runtime requires the onnxruntime package"
                ) from exc

        # Heavy model/audio dependencies are imported only when a pipeline is
        # instantiated, so CLI help and static tooling work in minimal envs.
        from asr.asr_manager import ASRManager
        from audio.capture import AudioCapture
        from audio.denoise import Denoiser
        from translation.mt_engine import Translator
        from tts.tts_engine import TTSEngine

        q_size = int(self.cfg["pipeline"]["queue_maxsize"])
        self.q_audio_raw = queue.Queue(maxsize=q_size)
        self.q_audio_clean = queue.Queue(maxsize=q_size)
        self.q_text_src = queue.Queue(maxsize=q_size)
        self.q_text_tgt = queue.Queue(maxsize=q_size)

        site_pack = SitePackLoader.load(site_pack_path) if site_pack_path else None
        data_dir = self.cfg["pipeline"].get(
            "construction_data_dir", "data/onevoice_construction_v2"
        )
        safety_source_csv = Path(
            self.cfg["pipeline"].get("safety_source_csv")
            or Path(data_dir) / "safety_fast_path.csv"
        )
        self.context = ConstructionContextEngine.from_data_dir(
            data_dir,
            site_pack=site_pack,
            required_safety_review_status="approved" if self.profile == "edge" else None,
            safety_path=safety_source_csv,
        )
        self.committer = SemanticCommitController(
            safety_confirmations=2,
            normal_commit_policy=self.cfg["pipeline"].get("normal_commit_policy", "endpoint"),
        )
        self.aligner = StablePrefixAligner()
        self.hypothesis_assembler = RollingHypothesisAssembler()
        self.streaming_session = RollingUtteranceSession(
            self.cfg["audio"], self.cfg["pipeline"]
        )
        safety_audio_manifest = Path(
            self.cfg["pipeline"].get(
                "safety_audio_manifest", "artifacts/safety_audio/manifest.json"
            )
        )
        self.safety_audio = (
            SafetyAudioStore(
                safety_audio_manifest,
                source_csv=safety_source_csv,
            )
            if safety_audio_manifest.is_file()
            else None
        )

        self.capture = AudioCapture(self.q_audio_raw, self.cfg)
        self.denoiser = Denoiser(self.cfg.get("denoise", {}))
        self.asr = ASRManager(
            self.cfg,
            offline=self.offline,
            enforce_release=True,
            direction=self.direction,
        )
        self.translator = Translator(
            self.cfg,
            offline=self.offline,
            profile=self.profile,
            direction=self.direction,
        )
        self.tts = TTSEngine(
            self.cfg,
            profile=self.profile,
            offline=self.offline,
            backend=self.tts_backend,
            allow_online_tts=self.allow_online_tts,
        )
        self.srt = SRTGenerator(bilingual=True)
        self._latency_log: list[dict] = []
        self._translation_log: list[dict] = []
        self._playback_log: list[dict] = []
        self.last_file_result: dict | None = None
        self._preflight_complete = False
        self._denoiser_loaded = False
        self._asr_loaded = False
        self._translation_loaded = False
        self._tts_loaded = False
        self._worker_threads: list[threading.Thread] = []
        self._stream_playback_enabled = True
        self._stream_chunks: list[SynthesizedChunk] = []
        self._stream_trace: list[dict] = []

    def _emit_event(self, kind: str, **data) -> None:
        """Optional UI observer; never decides or changes model output."""
        callback = getattr(self, "event_callback", None)
        if callback is not None:
            try:
                callback({"kind": kind, "direction": self.direction,
                          "timestamp": time.time(), **data})
            except Exception as exc:
                print(f"[Live observer] Event delivery failed: {exc}")

    def _put(self, target: queue.Queue, item: object) -> bool:
        while not self.stop_event.is_set():
            try:
                target.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def _run_worker(self, name: str, fn) -> None:
        try:
            fn()
        except BaseException as exc:
            self._fatal_error = exc
            self.stop_event.set()
            print(f"[{name}] ❌ Fatal worker error: {exc}")
            self._emit_event("error", worker=name, message=str(exc))

    def _denoise_worker(self) -> None:
        print("[Denoise Worker] ✅ Started")
        while not self.stop_event.is_set():
            try:
                raw = self.q_audio_raw.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                started = time.perf_counter()
                clean_samples = self.denoiser.process(raw.samples, raw.sample_rate)
                clean = AudioFrame(
                    samples=clean_samples,
                    sample_rate=raw.sample_rate,
                    sequence=raw.sequence,
                    captured_at=raw.captured_at,
                )
                self._put(
                    self.q_audio_clean,
                    (clean, (time.perf_counter() - started) * 1000),
                )
            finally:
                self.q_audio_raw.task_done()

    def _asr_worker(self) -> None:
        print(f"[ASR Worker] ✅ Started (direction={self.direction})")
        while not self.stop_event.is_set():
            try:
                frame, denoise_ms = self.q_audio_clean.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                event = self.streaming_session.accept(frame, denoise_ms)
                if event is None:
                    continue
                started = time.perf_counter()
                result = self.asr.transcribe(event.audio, direction=self.direction)
                if event.endpoint:
                    result = self._refine_safety_asr(event.audio, result)
                asr_ms = (time.perf_counter() - started) * 1000
                if not result.get("text"):
                    self._emit_event("asr", text="", hypothesis="", endpoint=event.endpoint)
                    if event.endpoint:
                        self.aligner.reset()
                        self.hypothesis_assembler.reset()
                        self.committer.reset()
                    continue
                window_text = normalize_asr(result["text"], lang=result["lang"])
                text = self.hypothesis_assembler.update(
                    window_text, endpoint=event.endpoint
                )
                now = time.perf_counter()
                stable, unstable = self.aligner.update(text)
                if event.endpoint:
                    stable, unstable = text, ""
                hypothesis = ASRHypothesis(
                    text=text,
                    stable_prefix=stable,
                    unstable_tail=unstable,
                    direction=self.direction,
                    started_at=event.started_at,
                    updated_at=now,
                    endpoint=event.endpoint,
                    emotion=result.get("emotion", "neutral"),
                    event=result.get("event", "speech"),
                )
                context = self.context.analyze(text, self.direction)
                decision = self.committer.decide(hypothesis, context)
                self._emit_event(
                    "asr", text=result.get("primary_text", result["text"]),
                    hypothesis=hypothesis.text, endpoint=event.endpoint,
                    stable_prefix=stable, unstable_tail=unstable,
                    decision=decision.kind.value, reason=decision.reason,
                )
                if event.endpoint:
                    print(f"[ASR {result['lang'].upper()} FINAL] {text}")
                self._stream_trace.append(
                    {
                        "event_started_at": event.started_at,
                        "event_updated_at": event.updated_at,
                        "endpoint": event.endpoint,
                        "hypothesis": hypothesis.text,
                        "raw_asr_text": result.get("primary_text", result["text"]),
                        "decoded_asr_text": result["text"],
                        "primary_asr_text": result.get("primary_text", result["text"]),
                        "asr_retry": result.get("safety_retry"),
                        "stable_prefix": hypothesis.stable_prefix,
                        "unstable_tail": hypothesis.unstable_tail,
                        "decision": decision.kind.value,
                        "decision_text": decision.text,
                        "reason": decision.reason,
                        "safety_id": (
                            decision.safety_match.safety_id
                            if decision.safety_match is not None
                            else None
                        ),
                    }
                )
                if event.endpoint:
                    self.aligner.reset()
                    self.hypothesis_assembler.reset()
                    self.committer.reset()
                if decision.kind == CommitKind.WAIT:
                    continue
                output_context = (
                    context
                    if decision.kind == CommitKind.SAFETY
                    else self.context.analyze(decision.text, self.direction)
                )
                self._put(
                    self.q_text_src,
                    {
                        "text": decision.text,
                        "lang": result["lang"],
                        "direction": self.direction,
                        "emotion": hypothesis.emotion,
                        "event": hypothesis.event,
                        "context": output_context,
                        "decision": decision,
                        "speech_to_commit_ms": (
                            decision.decided_at - event.started_at
                        )
                        * 1000,
                        "denoise_ms": event.denoise_ms,
                        "asr_ms": asr_ms,
                    },
                )
            finally:
                self.q_audio_clean.task_done()

    def _refine_safety_asr(self, audio, result: dict) -> dict:
        primary = result.get("text", "")
        retry = getattr(self.asr, "transcribe_without_itn", None)
        if (self.direction != "en2vi" or not primary or not callable(retry)
                or self.context.analyze(primary, self.direction).safety_candidates):
            return result
        alternate = retry(audio, self.direction)
        accepted = self.context.accept_safety_alternative(
            primary, alternate.get("text", ""), self.direction
        )
        selected = {**(alternate if accepted else result)}
        selected.update(primary_text=primary, safety_retry={
            "textnorm": "woitn", "text": alternate.get("text", ""), "accepted": accepted,
        })
        return selected

    def _translate_validated(self, source: str, context, direction: str) -> tuple[str, list[str], list[dict]]:
        """Keep top one when valid; otherwise select the first valid model beam."""
        generate = getattr(self.translator, "translate_candidates", None)
        candidates = (generate(source, direction) if callable(generate)
                      else [self.translator.translate(source, direction)])
        checked = []
        for rank, candidate in enumerate(candidates, 1):
            errors = self.context.validate_translation(candidate, context, direction)
            if not candidate.strip():
                errors = [*errors, "empty_translation"]
            checked.append({"rank": rank, "translation": candidate, "validation_errors": errors})
            if not errors:
                return candidate, [], checked
        if checked:
            return candidates[0], checked[0]["validation_errors"], checked
        return "", ["empty_translation"], []

    def _mt_worker(self) -> None:
        print("[MT Worker] ✅ Started (VI↔EN + Context/Safety)")
        while not self.stop_event.is_set():
            try:
                item = self.q_text_src.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                started = time.perf_counter()
                decision = item["decision"]
                context = item["context"]
                candidate_trace = []
                errors = None
                if decision.kind == CommitKind.SAFETY:
                    translated = decision.safety_match.translated_text
                    route = "safety_fast_path"
                elif context.translation_memory:
                    translated = context.translation_memory
                    route = "translation_memory"
                else:
                    canonical_source = self.context.canonicalize_source(
                        item["text"], context, item["direction"]
                    )
                    translated, errors, candidate_trace = self._translate_validated(
                        canonical_source, context, item["direction"]
                    )
                    route = "mt"
                if errors is None:
                    errors = self.context.validate_translation(translated, context, item["direction"])
                item.update(
                    translated=translated,
                    translation_route=route,
                    validation_errors=errors,
                    mt_ms=(time.perf_counter() - started) * 1000,
                )
                unsafe_validation = errors and context.risk_level in {"high", "critical"}
                self._emit_event(
                    "translation", source=item["text"], text=translated,
                    route=route, suppressed=bool(unsafe_validation),
                    validation_errors=list(errors),
                )
                print(f"[Translation {item['direction']}] {item['text']!r} -> {translated!r}")
                self._translation_log.append({
                    "source": item["text"], "translation": translated,
                    "route": route, "validation_errors": errors,
                    "candidate_trace": candidate_trace,
                    "suppressed": bool(unsafe_validation),
                })
                if unsafe_validation:
                    print(
                        "[Safety Validator] Translation suppressed: " + ", ".join(errors)
                    )
                elif translated:
                    self._put(self.q_text_tgt, item)
            finally:
                self.q_text_src.task_done()

    def _tts_worker(self) -> None:
        print("[TTS Worker] ✅ Started")
        commit_id = 0
        while not self.stop_event.is_set():
            try:
                item = self.q_text_tgt.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                commit_id += 1
                started = time.perf_counter()
                safety_match = item["decision"].safety_match
                pre_generated = (
                    self.safety_audio.get(safety_match.safety_id, item["direction"])
                    if safety_match and self.safety_audio
                    else None
                )
                if pre_generated:
                    audio, sample_rate = pre_generated
                    item["translation_route"] = "safety_audio"
                elif safety_match and self.profile == "edge":
                    raise RuntimeError(
                        f"Missing pre-generated edge safety audio: {safety_match.safety_id}"
                    )
                else:
                    print(f"[TTS Worker] Synthesizing | direction={item['direction']} | text={item['translated']!r}")
                    audio, sample_rate = self.tts.synthesize(
                        item["translated"],
                        direction=item["direction"],
                        emotion=item.get("emotion", "neutral"),
                    )
                if self.stop_event.is_set():
                    # A native synthesis call may finish after Stop was clicked.
                    # Do not start new speaker output after cancellation.
                    continue
                first_audio_at = time.perf_counter()
                tts_ms = (first_audio_at - started) * 1000
                total_ms = sum(
                    (
                        item.get("denoise_ms", 0),
                        item.get("asr_ms", 0),
                        item.get("mt_ms", 0),
                        tts_ms,
                    )
                )
                commit_to_audio_ms = (
                    first_audio_at - item["decision"].decided_at
                ) * 1000
                self._log_latency(item, total_ms, commit_to_audio_ms, tts_ms)
                chunk = SynthesizedChunk(
                    audio=np.asarray(audio, dtype=np.float32),
                    sample_rate=int(sample_rate),
                    engine=(
                        "safety_audio"
                        if pre_generated
                        else self.tts.engine_name(item["direction"])
                    ),
                    commit_id=commit_id,
                    committed_at=item["decision"].decided_at,
                    first_audio_at=first_audio_at,
                )
                self._stream_chunks.append(chunk)
                if self.tts.is_silence(audio):
                    raise RuntimeError("TTS returned silence; commit was not played")
                if self._stream_playback_enabled:
                    pause_mic = bool(getattr(self, "pause_mic_during_playback", False))
                    if pause_mic:
                        self.capture.pause()
                    self._emit_event("playback", state="started", mic_paused=pause_mic)
                    try:
                        playback = self.tts.play(audio, sample_rate=sample_rate)
                        self._playback_log.append({"commit_id": commit_id, **playback})
                    finally:
                        if pause_mic:
                            self.stop_event.wait(0.2)  # Let room echo decay.
                            self.capture.resume()
                        self._emit_event("playback", state="finished", mic_paused=False)
                self.srt.add_entry(
                    item["text"], item["translated"], len(audio) / sample_rate
                )
            finally:
                self.q_text_tgt.task_done()

    def _log_latency(
        self, item: dict, total_ms: float, commit_to_audio_ms: float, tts_ms: float
    ) -> None:
        safety = item["decision"].kind == CommitKind.SAFETY
        target = self.cfg["pipeline"].get(
            "safety_target_latency_ms" if safety else "target_latency_ms",
            300 if safety else 1000,
        )
        record = {
            "direction": item["direction"],
            "route": item["translation_route"],
            "denoise_ms": item.get("denoise_ms", 0),
            "asr_ms": item.get("asr_ms", 0),
            "mt_ms": item.get("mt_ms", 0),
            "tts_ms": tts_ms,
            "compute_total_ms": total_ms,
            "commit_to_first_audio_ms": commit_to_audio_ms,
            "speech_to_commit_ms": item.get("speech_to_commit_ms"),
            "target_ms": target,
            "validation_errors": item.get("validation_errors", []),
        }
        self._latency_log.append(record)
        status = "✅" if commit_to_audio_ms < target else "⚠️"
        print(
            f"{status} [{item['direction']}] route={item['translation_route']} | "
            f"commit→audio={commit_to_audio_ms:.0f}ms | compute={total_ms:.0f}ms"
        )

    def load_models(
        self,
        *,
        load_translation: bool = True,
        load_tts: bool = True,
    ) -> None:
        """Load only the stages needed by the caller.

        File-mode safety smoke tests must be able to exercise local ASR and the
        pre-generated safety-audio store even if a normal TTS backend is not
        bundled. Translation and TTS remain mandatory for normal routes and
        for the microphone runtime.
        """
        if self.offline and not self._preflight_complete:
            self._emit_event("status", state="loading", message="Kiểm tra model local...")
            manifest = self.cfg["pipeline"].get("artifact_manifest", "artifacts/manifest.json")
            result = verify_artifacts(
                manifest,
                self.direction,
                self.profile,
                sample_rate=int(self.cfg["audio"]["sample_rate"]),
            )
            print(f"[Preflight] ✅ Local artifacts verified: {len(result['checked'])}")
            self._preflight_complete = True
        requested = (
            not self._denoiser_loaded
            or not self._asr_loaded
            or (load_translation and not self._translation_loaded)
            or (load_tts and not self._tts_loaded)
        )
        if not requested:
            return
        print(f"\n🔄 Loading models (profile={self.profile}, offline={self.offline})...")
        started = time.perf_counter()
        if not self._denoiser_loaded:
            self.denoiser.load()
            self._denoiser_loaded = True
        if not self._asr_loaded:
            self.asr.load(direction=self.direction)
            self._asr_loaded = True
        if load_translation and not self._translation_loaded:
            self._emit_event("status", state="loading", message="Nạp model dịch local...")
            self.translator.load()
            self._translation_loaded = True
        if load_tts and not self._tts_loaded:
            self._emit_event("status", state="loading", message="Nạp giọng đọc offline...")
            self.tts.load(direction=self.direction)
            self._tts_loaded = True
        print(f"\n✅ Models loaded in {time.perf_counter() - started:.1f}s\n")

    def start(self) -> None:
        self.load_models()
        if self.stop_event.is_set():
            self._save_reports()
            return
        self.capture.start()
        threads = self._start_workers()
        print(f"🎙️ OneVoice V2 LIVE — {self.direction} ({self.profile})")
        self._emit_event("status", state="listening", message="Đang nghe — bạn có thể nói.")
        try:
            while not self.stop_event.wait(0.2):
                if self.capture.error is not None:
                    self._fatal_error = self.capture.error
                    self.stop_event.set()
                elif not self.capture.is_alive():
                    self._fatal_error = RuntimeError("Microphone capture thread stopped")
                    self.stop_event.set()
        except KeyboardInterrupt:
            self.capture.stop()
            self._drain_queues(timeout_s=5.0)
            self.stop_event.set()
        finally:
            self._stop_workers(threads)
            self._save_reports()
            self._emit_event("status", state="stopped", message="Đã dừng thu micro.")
        if self._fatal_error:
            raise RuntimeError("OneVoice worker failed") from self._fatal_error

    def _drain_queues(self, timeout_s: float) -> None:
        deadline = time.perf_counter() + timeout_s
        queues = (self.q_audio_raw, self.q_audio_clean, self.q_text_src, self.q_text_tgt)
        while time.perf_counter() < deadline:
            if all(target.unfinished_tasks == 0 for target in queues):
                return
            time.sleep(0.05)
        remaining = sum(target.unfinished_tasks for target in queues)
        print(f"[Shutdown] Queue drain timeout; {remaining} items remain")

    def _start_workers(self) -> list[threading.Thread]:
        workers = [
            ("Denoise", self._denoise_worker),
            ("ASR", self._asr_worker),
            ("MT", self._mt_worker),
            ("TTS", self._tts_worker),
        ]
        threads = [
            threading.Thread(
                target=self._run_worker,
                args=(name, fn),
                daemon=True,
                name=name,
            )
            for name, fn in workers
        ]
        self._worker_threads = threads
        for thread in threads:
            thread.start()
        return threads

    def _stop_workers(self, threads: list[threading.Thread] | None = None) -> None:
        self.stop_event.set()
        self.capture.stop()
        for thread in threads or self._worker_threads:
            if thread is not threading.current_thread():
                thread.join(timeout=2)
        self._worker_threads = []

    def cancel(self) -> None:
        """Request cancellation; the active stream exits and flushes reports."""
        self.stop_event.set()
        self.capture.stop()

    def stream_file(self, input_path: str, realtime: bool = False) -> dict:
        """Replay a WAV as 32 ms frames through the real streaming workers.

        This deterministic P2 harness uses the configured ASR/MT/TTS models,
        bounded queues and the same VAD/commit path as microphone mode, while
        suppressing speaker playback. A silence tail forces the endpoint so the
        final utterance is not lost.
        """
        import soundfile as sf

        self.stop_event.clear()
        self._fatal_error = None
        self._stream_playback_enabled = False
        self._stream_chunks = []
        self._stream_trace = []
        self._latency_log = []
        self._translation_log = []
        self._playback_log = []
        self.srt = SRTGenerator(bilingual=True)
        self.streaming_session.reset(clear_sequence=True)
        self.aligner.reset()
        self.hypothesis_assembler.reset()
        self.committer.reset()
        self.load_models()

        audio, source_rate = sf.read(input_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if len(audio) == 0:
            raise ValueError(f"Input audio is empty: {input_path}")
        target_rate = int(self.cfg["audio"]["sample_rate"])
        if source_rate != target_rate:
            target_size = max(1, round(len(audio) * target_rate / source_rate))
            audio = np.interp(
                np.linspace(0, len(audio) - 1, target_size),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)
        frame_samples = self.streaming_session.frame_samples
        frame_ms = self.streaming_session.frame_ms
        frame_count = 0
        stream_started_at = time.perf_counter()
        stream_completed_at = stream_started_at
        threads = self._start_workers()
        stream_error: BaseException | None = None
        try:
            for offset in range(0, len(audio), frame_samples):
                if self.stop_event.is_set():
                    break
                block = np.asarray(audio[offset : offset + frame_samples], dtype=np.float32)
                if len(block) < frame_samples:
                    block = np.pad(block, (0, frame_samples - len(block)))
                frame_count += 1
                frame = AudioFrame(
                    samples=np.ascontiguousarray(block),
                    sample_rate=target_rate,
                    sequence=frame_count,
                    captured_at=time.perf_counter(),
                )
                if not self._put(self.q_audio_raw, frame):
                    break
                if realtime:
                    time.sleep(frame_ms / 1000.0)

            # Endpoint after configured silence. One extra frame avoids an
            # off-by-one when the duration is exactly a multiple of frame_ms.
            silence_frames = max(
                1, int(np.ceil(self.streaming_session.endpoint_ms / frame_ms)) + 1
            )
            for _ in range(silence_frames):
                if self.stop_event.is_set():
                    break
                frame_count += 1
                frame = AudioFrame(
                    samples=np.zeros(frame_samples, dtype=np.float32),
                    sample_rate=target_rate,
                    sequence=frame_count,
                    captured_at=time.perf_counter(),
                )
                if not self._put(self.q_audio_raw, frame):
                    break
                if realtime:
                    time.sleep(frame_ms / 1000.0)
            self._drain_queues(timeout_s=max(10.0, len(audio) / target_rate + 10.0))
            if self._fatal_error:
                raise RuntimeError("OneVoice streaming worker failed") from self._fatal_error
        except BaseException as exc:
            stream_error = exc
        finally:
            self._stop_workers(threads)
            self._stream_playback_enabled = True
            self._save_reports()
            stream_completed_at = time.perf_counter()

        commit_ids = [chunk.commit_id for chunk in self._stream_chunks]
        result = {
            "schema_version": 1,
            "direction": self.direction,
            "profile": self.profile,
            "offline": self.offline,
            "normal_commit_policy": self.committer.normal_commit_policy,
            "input_path": str(Path(input_path).resolve()),
            "frame_samples": frame_samples,
            "frame_ms": frame_ms,
            "frames_submitted": frame_count,
            "stream_started_at": stream_started_at,
            "stream_completed_at": stream_completed_at,
            "complete_turn_ms": (stream_completed_at - stream_started_at) * 1000,
            "commits": len(self._stream_chunks),
            "commit_ids": commit_ids,
            "hypothesis_trace": self._stream_trace,
            "translations": self._translation_log,
            "chunks": [
                {
                    "commit_id": chunk.commit_id,
                    "sample_rate": chunk.sample_rate,
                    "samples": int(len(chunk.audio)),
                    "engine": chunk.engine,
                    "committed_at": chunk.committed_at,
                    "first_audio_at": chunk.first_audio_at,
                    "commit_to_first_audio_ms": (
                        chunk.first_audio_at - chunk.committed_at
                    )
                    * 1000,
                }
                for chunk in self._stream_chunks
            ],
            "dropped_audio_frames": self.capture.dropped_frames,
            "fatal_error": repr(self._fatal_error) if self._fatal_error else None,
        }
        if self.report_dir:
            self.report_dir.mkdir(parents=True, exist_ok=True)
            temporary = self.report_dir / "stream_result.json.tmp"
            temporary.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary.replace(self.report_dir / "stream_result.json")
        if commit_ids != sorted(commit_ids) or len(commit_ids) != len(set(commit_ids)):
            raise RuntimeError("Streaming output commit order is not monotonic")
        if stream_error is not None:
            raise stream_error.with_traceback(stream_error.__traceback__)
        return result

    def process_file(self, input_path: str, output_path: str | None = None) -> str:
        import soundfile as sf

        # Safety audio has already been reviewed and generated locally. Delay
        # normal MT/TTS startup until ASR confirms that this input is not a
        # safety phrase; this makes the safety fast path testable offline.
        file_started = time.perf_counter()
        self.load_models(load_translation=False, load_tts=False)
        load_complete = time.perf_counter()
        audio, source_rate = sf.read(input_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if len(audio) == 0:
            raise ValueError(f"Input audio is empty: {input_path}")
        target_rate = int(self.cfg["audio"]["sample_rate"])
        if source_rate != target_rate:
            target_size = max(1, round(len(audio) * target_rate / source_rate))
            audio = np.interp(
                np.linspace(0, len(audio) - 1, target_size),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)
        denoise_started = time.perf_counter()
        clean = self.denoiser.process(audio, self.cfg["audio"]["sample_rate"])
        denoise_ms = (time.perf_counter() - denoise_started) * 1000
        asr_started = time.perf_counter()
        result = self.asr.transcribe(clean, self.direction)
        result = self._refine_safety_asr(clean, result)
        asr_ms = (time.perf_counter() - asr_started) * 1000
        if not result.get("text"):
            raise RuntimeError("ASR returned an empty transcript")
        text = normalize_asr(result["text"], result["lang"])
        context = self.context.analyze(text, self.direction)
        safety = context.safety_candidates[0] if context.safety_candidates else None
        canonical_source = text
        mt_started = time.perf_counter()
        candidate_trace = []
        if safety:
            translated = safety.translated_text
            pre_generated = (
                self.safety_audio.get(safety.safety_id, self.direction)
                if self.safety_audio
                else None
            )
            pre_generated_path = (
                self.safety_audio.path_for(safety.safety_id, self.direction)
                if self.safety_audio
                else None
            )
        else:
            canonical_source = self.context.canonicalize_source(
                text, context, self.direction
            )
            if context.translation_memory:
                translated = context.translation_memory
                translation_route = "translation_memory"
            else:
                self.load_models(load_translation=True, load_tts=False)
                translated, _, candidate_trace = self._translate_validated(
                    canonical_source, context, self.direction
                )
                translation_route = "mt"
            pre_generated = None
            pre_generated_path = None
        mt_ms = (time.perf_counter() - mt_started) * 1000
        errors = self.context.validate_translation(translated, context, self.direction)
        if errors and context.risk_level in {"high", "critical"}:
            raise RuntimeError("Unsafe translation: " + ", ".join(errors))
        if pre_generated:
            output_audio, sample_rate = pre_generated
            route = "safety_audio"
        elif safety and self.profile == "edge":
            raise RuntimeError(f"Missing pre-generated edge safety audio: {safety.safety_id}")
        else:
            tts_started = time.perf_counter()
            self.load_models(load_translation=False, load_tts=True)
            output_audio, sample_rate = self.tts.synthesize(
                translated, self.direction, result.get("emotion", "neutral")
            )
            route = "safety_tts" if safety else f"normal_{translation_route}_tts"
        tts_ms = 0.0 if pre_generated else (time.perf_counter() - tts_started) * 1000
        if self.tts.is_silence(output_audio):
            raise RuntimeError("TTS returned silence")
        destination = Path(output_path or (self.report_dir or Path(".")) / "output.wav")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if pre_generated_path is not None:
            shutil.copyfile(pre_generated_path, destination)
        else:
            sf.write(destination, output_audio, sample_rate)
        safety_id = safety.safety_id if safety else "-"
        output_sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()
        commit_id = hashlib.sha256(
            f"{self.direction}\0{Path(input_path).resolve()}\0{output_sha256}".encode("utf-8")
        ).hexdigest()[:16]
        self.last_file_result = {
            "schema_version": 1,
            "commit_id": commit_id,
            "direction": self.direction,
            "profile": self.profile,
            "offline": self.offline,
            "input_path": str(Path(input_path).resolve()),
            "output_path": str(destination.resolve()),
            "output_sha256": output_sha256,
            "output_sample_rate": int(sample_rate),
            "output_samples": int(len(output_audio)),
            "asr_text": text,
            "primary_asr_text": result.get("primary_text", result["text"]),
            "asr_retry": result.get("safety_retry"),
            "canonical_source": canonical_source,
            "translation": translated,
            "translation_candidates": candidate_trace,
            "route": route,
            "safety_id": None if safety is None else safety.safety_id,
            "domain": context.domain,
            "intent": context.intent,
            "risk_level": context.risk_level,
            "entities": context.entities,
            "validation_errors": errors,
            "model_reference": self.translator.model_reference,
            "artifacts": {
                "release_lock": self.cfg["pipeline"].get("release_lock")
                or self.cfg["pipeline"].get("artifact_manifest"),
                "asr_model_dir": (
                    self.cfg["asr"].get("gipformer_model_dir")
                    if self.direction == "vi2en"
                    else self.cfg["sensevoice"].get("model_path")
                ),
                "tts_engine": "safety_audio"
                if pre_generated_path is not None
                else self.tts.engine_name(self.direction),
            },
            "timings_ms": {
                "startup": (load_complete - file_started) * 1000,
                "denoise": denoise_ms,
                "asr": asr_ms,
                "mt_or_memory": mt_ms,
                "tts": tts_ms,
                "complete_turn": (time.perf_counter() - file_started) * 1000,
            },
        }
        if self.report_dir:
            self.report_dir.mkdir(parents=True, exist_ok=True)
            report_path = self.report_dir / "file_result.json"
            temporary = report_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(self.last_file_result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(report_path)
        print(
            f"[File pipeline] route={route} safety_id={safety_id} "
            f"source={text!r} target={translated!r}"
        )
        return str(destination.resolve())

    def _save_reports(self) -> None:
        if self.srt.entry_count:
            root = self.report_dir or Path(".")
            root.mkdir(parents=True, exist_ok=True)
            self.srt.save(str(root / f"output_{int(time.time())}.srt"))
        if self.report_dir:
            self.report_dir.mkdir(parents=True, exist_ok=True)
            (self.report_dir / "latency.json").write_text(
                json.dumps(self._latency_log, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            latency_summary = {}
            for route_name, safety in (("normal", False), ("safety", True)):
                rows = [
                    row
                    for row in self._latency_log
                    if (row["route"] in {"safety_fast_path", "safety_audio"}) == safety
                ]
                if not rows:
                    latency_summary[route_name] = {"samples": 0}
                    continue
                latency_summary[route_name] = {
                    "samples": len(rows),
                    "commit_to_first_audio_p50_ms": self._percentile(
                        [row["commit_to_first_audio_ms"] for row in rows], 0.50
                    ),
                    "commit_to_first_audio_p95_ms": self._percentile(
                        [row["commit_to_first_audio_ms"] for row in rows], 0.95
                    ),
                    "speech_to_commit_p50_ms": self._percentile(
                        [row["speech_to_commit_ms"] for row in rows if row["speech_to_commit_ms"] is not None],
                        0.50,
                    ),
                    "speech_to_commit_p95_ms": self._percentile(
                        [row["speech_to_commit_ms"] for row in rows if row["speech_to_commit_ms"] is not None],
                        0.95,
                    ),
                }
            (self.report_dir / "latency_summary.json").write_text(
                json.dumps(latency_summary, indent=2), encoding="utf-8"
            )
            (self.report_dir / "playback_events.json").write_text(
                json.dumps(self._playback_log, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (self.report_dir / "asr_events.json").write_text(
                json.dumps(getattr(self, "_stream_trace", []), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (self.report_dir / "translation_events.json").write_text(
                json.dumps(getattr(self, "_translation_log", []), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (self.report_dir / "runtime_summary.json").write_text(
                json.dumps(
                    {
                        "direction": self.direction,
                        "profile": self.profile,
                        "offline": self.offline,
                        "normal_commit_policy": self.committer.normal_commit_policy,
                        "output_device": self.cfg["audio"].get("output_device"),
                        "input_device": self.cfg["audio"].get("input_device"),
                        "tts_engine": self.tts.engine_name(self.direction),
                        "tts_online_network_enabled": self.tts.allow_online_tts and self.tts.backend == "gtts",
                        "device_playback_completed": len(self._playback_log),
                        "listener_confirmation": "not_recorded",
                        "vad_energy_threshold": self.cfg["audio"].get(
                            "vad_energy_threshold", 0.015
                        ),
                        "dropped_audio_frames": self.capture.dropped_frames,
                        "pause_mic_during_playback": getattr(self, "pause_mic_during_playback", False),
                        "paused_audio_frames": getattr(self.capture, "paused_frames", 0),
                        "fatal_error": repr(self._fatal_error) if self._fatal_error else None,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    @staticmethod
    def _percentile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        return float(ordered[round((len(ordered) - 1) * fraction)])


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="OneVoice V2 — VI↔EN speech translation")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--direction", choices=["vi2en", "en2vi"], default="vi2en")
    parser.add_argument("--profile", choices=["development", "edge", "premium"])
    parser.add_argument("--site-pack")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--input-file")
    parser.add_argument(
        "--stream-file",
        help="Replay a WAV through 32 ms streaming/VAD/commit workers (no speaker playback)",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Throttle --stream-file replay to capture frame rate",
    )
    parser.add_argument("--output-file")
    parser.add_argument("--report-dir")
    parser.add_argument("--output-device", type=parse_output_device,
                        help="Output index or name/host API, e.g. 'GZUT-MUSIC MME', from python -m sounddevice")
    parser.add_argument("--input-device", type=parse_input_device,
                        help="Microphone index or name/host API, e.g. 'Microphone Array Realtek MME'")
    parser.add_argument(
        "--tts-backend", choices=["auto", "sapi", "espeak", "gtts"],
        help="TTS voice backend; Windows auto prefers an installed offline voice in the output language",
    )
    parser.add_argument("--windows-en-voice", help="Installed Windows English voice name for this run, e.g. 'Microsoft Zira Desktop'")
    parser.add_argument("--windows-en-rate", type=int, help="Windows English speech rate from -10 to 10; overrides config for this run")
    parser.add_argument("--pause-mic-during-playback", action="store_true",
                        help="Half-duplex laptop demo: discard mic frames during speaker output to avoid self-translation (not AEC)")
    parser.add_argument(
        "--allow-online-tts", action="store_true",
        help="Allow gTTS network requests for translated text (requires --tts-backend gtts)",
    )
    parser.add_argument(
        "--vad-energy-threshold",
        type=float,
        help="Override the normalized RMS speech threshold for this run (0 < value <= 1)",
    )
    args = parser.parse_args()

    if args.input_file and args.stream_file:
        parser.error("--input-file and --stream-file are mutually exclusive")
    if args.vad_energy_threshold is not None and not 0 < args.vad_energy_threshold <= 1:
        parser.error("--vad-energy-threshold must be greater than 0 and at most 1")
    if args.tts_backend == "gtts" and not args.allow_online_tts:
        parser.error("--tts-backend gtts sends translated text online; also pass --allow-online-tts")
    if args.allow_online_tts and args.tts_backend != "gtts":
        parser.error("--allow-online-tts requires --tts-backend gtts")
    if args.windows_en_voice is not None and not args.windows_en_voice.strip():
        parser.error("--windows-en-voice cannot be empty")
    if args.windows_en_rate is not None and not -10 <= args.windows_en_rate <= 10:
        parser.error("--windows-en-rate must be between -10 and 10")

    pipeline = OneVoicePipeline(
        config_path=args.config,
        direction=args.direction,
        profile=args.profile,
        site_pack_path=args.site_pack,
        offline=args.offline,
        report_dir=args.report_dir,
        vad_energy_threshold=args.vad_energy_threshold,
        output_device=args.output_device,
        input_device=args.input_device,
        tts_backend=args.tts_backend,
        allow_online_tts=args.allow_online_tts,
        windows_en_voice=args.windows_en_voice,
        windows_en_rate=args.windows_en_rate,
        pause_mic_during_playback=args.pause_mic_during_playback,
    )
    if args.stream_file:
        result = pipeline.stream_file(args.stream_file, realtime=args.realtime)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.input_file:
        print(f"✅ Output saved: {pipeline.process_file(args.input_file, args.output_file)}")
    else:
        pipeline.start()


if __name__ == "__main__":
    main()
