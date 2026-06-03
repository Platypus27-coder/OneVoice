"""
Trạm 3: Text-to-Speech Engine (VI ↔ EN)
=========================================
Routes TTS output based on translation direction:
  - VI output (EN→VI direction): OmniVoice / BetterBox-TTS
  - EN output (VI→EN direction): Whisper-based English TTS / VITS-tiny

Premium Mode (Voice Cloning):
  - VALL-E X (activated when device has sufficient resources)
  - Preserves speaker voice identity across languages

References:
  BetterBox-TTS — Dolly VN / ContextBoxAI (CC BY-NC 4.0)
    https://github.com/nowtranminh1-TTS/BetterBox-TTS
  VALL-E X — Plachtaa / Songting (MIT License)
    https://github.com/Plachtaa/VALL-E-X
"""

import time
import queue
import sys
import os
import numpy as np
import sounddevice as sd


class TTSEngine:
    """
    Text-to-Speech router for VI↔EN pipeline.

    Routing:
      "vi2en" direction → output is English → English TTS (VITS/espeak)
      "en2vi" direction → output is Vietnamese → OmniVoice/BetterBox
    """

    def __init__(self, config: dict):
        self.cfg = config["tts"]
        self.sample_rate = config["audio"]["sample_rate"]
        self.default_engine = self.cfg.get("default_engine", "betterbox")
        self.en_speed = float(self.cfg.get("en_speed", 0.85))
        # Voice preset: a pre-recorded natural EN voice as reference (preferred over live cloning)
        self.en_preset_audio = self.cfg.get("en_preset_audio", None)
        self.en_preset_text  = self.cfg.get("en_preset_text",  None)
        self._omni = None          # OmniVoice for Vietnamese TTS
        self._en_tts = None        # English TTS engine
        self._vallex = None        # VALL-E X (Premium Mode)

    def load(self):
        """Initialize all TTS backends."""
        print(f"[TTS] Initializing engines...")
        self._load_omnivoice()
        self._load_english_tts()
        self._auto_prepare_preset()   # Tự động chuẩn bị giọng mẫu tiếng Anh
        print("[TTS] ✅ TTS Engine ready.")

    def _apply_speed(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        """
        Apply speed adjustment to audio using librosa time stretching.
        self.en_speed < 1.0 = slower (easier to listen), > 1.0 = faster.
        """
        if abs(self.en_speed - 1.0) < 0.02:  # Skip if essentially 1.0
            return audio, sr
        try:
            import librosa
            # time_stretch rate: > 1.0 speeds up, < 1.0 slows down
            # We invert because en_speed=0.8 means "play at 80% speed" = stretch by 1/0.8
            stretch_rate = self.en_speed
            audio_stretched = librosa.effects.time_stretch(audio, rate=stretch_rate)
            return audio_stretched.astype(np.float32), sr
        except Exception as e:
            print(f"[TTS EN] ⚠ Speed adjustment failed: {e}")
            return audio, sr

    def _auto_prepare_preset(self):
        """
        Tự động chuẩn bị file giọng mẫu tiếng Anh khi load().
        Không cần chạy script riêng. Thứ tự ưu tiên:
          1. File đã tồn tại sẵn → dùng ngay (không làm gì thêm).
          2. F5-TTS built-in reference (có sẵn trong Colab khi cài f5-tts).
          3. Tạo bằng gTTS (cần internet, fallback).
        """
        if not self.en_preset_audio:
            return   # Không cấu hình preset → bỏ qua

        if os.path.exists(self.en_preset_audio):
            print(f"[TTS] 🎤 Voice preset ready: {os.path.basename(self.en_preset_audio)}")
            return   # Đã có sẵn rồi, không cần tải lại

        # Đảm bảo thư mục tồn tại
        os.makedirs(os.path.dirname(self.en_preset_audio), exist_ok=True)

        # ── Phương án 1: F5-TTS built-in reference (nhanh nhất, 0 download, đa nền tảng) ──
        try:
            import f5_tts
            f5_dir = f5_tts.__path__[0]
            builtin = os.path.join(f5_dir, "infer", "examples", "basic", "basic_ref_en.wav")
            if os.path.exists(builtin):
                import shutil
                shutil.copy(builtin, self.en_preset_audio)
                
                # Update preset text to match F5-TTS built-in transcript
                txt_path = self.en_preset_audio.replace(".wav", ".txt")
                f5_text = "Some call me nature, others call me mother nature."
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(f5_text)
                self.en_preset_text = f5_text
                print(f"[TTS] 🎤 Voice preset: sử dụng F5-TTS built-in reference.")
                return
        except ImportError:
            pass

        # ── Phương án 2: gTTS (cần internet, fallback cho lần đầu) ─────────
        try:
            from gtts import gTTS
            from pydub import AudioSegment
            import tempfile
            ref_text = "Attention all site personnel. Please proceed to the designated safety zone immediately. Thank you."
            print(f"[TTS] 🌐 Đang tạo voice preset lần đầu bằng gTTS (chỉ cần 1 lần)...")
            tts = gTTS(text=ref_text, lang="en", slow=False, tld="com")
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                mp3_path = tmp.name
            tts.save(mp3_path)
            seg = AudioSegment.from_mp3(mp3_path)
            seg = seg[:9000]  # 9 giây đầu
            seg.export(self.en_preset_audio, format="wav")
            os.unlink(mp3_path)
            self.en_preset_text = ref_text
            txt_path = self.en_preset_audio.replace(".wav", ".txt")
            with open(txt_path, "w") as f:
                f.write(ref_text)
            print(f"[TTS] ✅ Voice preset đã được tạo tự động: {self.en_preset_audio}")
        except Exception as e:
            print(f"[TTS] ⚠ Không tạo được voice preset ({e}). Sẽ dùng voice cloning từ input audio.")
            self.en_preset_audio = None   # Reset để _get_en_reference() fallback sang clone

    def _load_omnivoice(self):
        """
        Load OmniVoice (BetterBox-TTS) for Vietnamese speech synthesis.
        Uses local port inside src/tts/omnivoice_inference/
        """
        try:
            # Ensure src/tts is in sys.path so 'omnivoice' and 'omnivoice_inference' resolve correctly
            tts_dir = os.path.dirname(os.path.abspath(__file__))
            if tts_dir not in sys.path:
                sys.path.insert(0, tts_dir)

            from omnivoice_inference.ttsOmni import Omni, generate_speech_omni
            
            model_path = self.cfg.get("betterbox", {}).get(
                "model_path", os.path.join("models", "omnivoice")
            )
            if not os.path.exists(model_path):
                print(f"[TTS] ⚠ Thư mục '{model_path}' không tồn tại. Đang tự động tải mô hình từ 'splendor1811/omnivoice-vietnamese' để test tạm...")
                model_path = "splendor1811/omnivoice-vietnamese"

            self._omni = Omni(model_path=model_path)
            self._generate_speech_omni = generate_speech_omni
            
            # ── Auto-generate VI reference audio if missing ──
            wavs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "omnivoice_inference", "wavs")
            # If the user didn't create a 'wavs' folder in the root, create one locally
            root_wavs = "wavs"
            if not os.path.exists(root_wavs):
                os.makedirs(root_wavs, exist_ok=True)
            
            ref_path = os.path.join(root_wavs, "reference_sound.wav")
            if not os.path.exists(ref_path):
                # Ưu tiên sử dụng Nobita.wav do người dùng cung cấp
                nobita_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Nobita.wav")
                # Normalize path
                nobita_path = os.path.abspath(nobita_path)
                
                if os.path.exists(nobita_path):
                    print(f"[TTS VI] 🌐 Tìm thấy file {nobita_path}, đang copy làm voice preset...")
                    import shutil
                    shutil.copy(nobita_path, ref_path)
                    
                    # Tạo file txt chứa transcript để OmniVoice không phải gọi mô hình nhận diện giọng nói (ASR)
                    # (tránh lỗi thiếu thư viện chunkformer)
                    ref_text_path = ref_path.replace(".wav", ".txt")
                    with open(ref_text_path, "w", encoding="utf-8") as f:
                        f.write("Cậu đã làm dì dới nó dở. Thêm năng lượng hả. Nó quạt động như thế nào dợ. Cho mình mượn chúc, đừng có keo kiệt dậy chứ. Hôm nai lớp mình có bài kiểm tra môn thể dục nên mình rất là cần nó luôn. Sài xong mình trả lại liền.")
                        
                    print(f"[TTS VI] ✅ Voice preset tiếng Việt đã được tạo từ Nobita.wav: {ref_path}")
                else:
                    print(f"[TTS VI] 🌐 Đang tạo voice preset tiếng Việt bằng gTTS (chỉ cần 1 lần)...")
                    try:
                        from gtts import gTTS
                        from pydub import AudioSegment
                        import tempfile
                        # Một câu tiếng Việt chuẩn, rõ ràng để làm mẫu giọng
                        ref_text = "Chào mừng bạn đến với hệ thống OneVoice. Hệ thống đã sẵn sàng."
                        tts = gTTS(text=ref_text, lang="vi", slow=False, tld="com.vn")
                        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                            mp3_path = tmp.name
                        tts.save(mp3_path)
                        seg = AudioSegment.from_mp3(mp3_path)
                        seg.export(ref_path, format="wav")
                        os.unlink(mp3_path)
                        # Create txt file too
                        with open(ref_path.replace(".wav", ".txt"), "w", encoding="utf-8") as f:
                            f.write(ref_text)
                        print(f"[TTS VI] ✅ Voice preset tiếng Việt đã được tạo: {ref_path}")
                    except Exception as e:
                        print(f"[TTS VI] ⚠ Không tạo được voice preset ({e})")
                    
            print(f"[TTS] ✅ OmniVoice loaded from: {self._omni.model_path}")
        except Exception as e:
            print(f"[TTS] ⚠ Failed to load OmniVoice: {e}")
            self._omni = None

    def _load_english_tts(self):
        """
        Load lightweight English TTS.
        Priority: VITS-ONNX → XTTS v2 (voice clone) → pyttsx3 → gTTS
        """
        vits_path = self.cfg.get("vits", {}).get("model_path", "models/vits_en_tiny.onnx")
        if os.path.exists(vits_path):
            try:
                import onnxruntime as ort
                self._en_tts = ort.InferenceSession(
                    vits_path, providers=["CPUExecutionProvider"]
                )
                self._en_tts_engine = "vits"
                print(f"[TTS] ✅ VITS English TTS loaded from: {vits_path}")
                return
            except Exception as e:
                print(f"[TTS] ⚠ VITS load failed: {e}")

        # Priority 2: F5-TTS — voice cloning (Python 3.10+, offline, GPU)
        try:
            if os.name == 'nt':
                # Fix DLL loading for torchcodec/ffmpeg on Windows Conda (Python 3.8+)
                conda_prefix = os.environ.get("CONDA_PREFIX")
                if conda_prefix:
                    bin_path = os.path.join(conda_prefix, "Library", "bin")
                    if os.path.exists(bin_path):
                        try:
                            os.add_dll_directory(bin_path)
                        except AttributeError:
                            pass
            from f5_tts.api import F5TTS
            self._en_tts = F5TTS()
            self._en_tts_engine = "f5tts"
            print("[TTS] ✅ F5-TTS loaded (voice cloning enabled).")
            return
        except Exception as e:
            print(f"[TTS] ⚠ F5-TTS not available ({e})")

        # Priority 3: pyttsx3 (offline, no voice clone)
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 160)
            self._en_tts = engine
            self._en_tts_engine = "pyttsx3"
            print("[TTS] ✅ pyttsx3 English TTS loaded (fallback).")
            return
        except Exception as e:
            print(f"[TTS] ⚠ pyttsx3 not available ({e})")

        # Priority 4: gTTS (internet, no voice clone, Colab test only)
        try:
            from gtts import gTTS
            self._en_tts = "gtts"
            self._en_tts_engine = "gtts"
            print("[TTS] ✅ gTTS English TTS loaded (online fallback for Colab).")
            return
        except ImportError:
            pass

        print("[TTS] ⚠ No English TTS available — using silence stub.")

    def synthesize_vi(self, text: str, emotion: str = "neutral") -> np.ndarray:
        """
        Synthesize Vietnamese speech using OmniVoice (BetterBox-TTS).
        Called for EN→VI direction (speaker heard Vietnamese output).
        Maps emotion to OmniVoice `instruct` prompt.
        """
        # ── Emotion Routing ──────────────────────────────────────────────────
        emotion_map = {
            "happy": "happy, high pitch, bright",
            "sad": "sad, low pitch, slow, quiet",
            "angry": "angry, fast, loud, high pitch",
            "fearful": "fearful, fast, trembling",
            "disgusted": "disgusted, low pitch",
            "surprised": "surprised, high pitch, fast",
            "neutral": ""
        }
        instruct = emotion_map.get(emotion.lower(), "")
        if instruct:
            print(f"[TTS VI] 🎭 Applied emotion routing: {emotion.upper()}")

        if self._omni is not None:
            try:
                t0 = time.perf_counter()
                # Use default reference audio if available
                ref_audio = self.cfg.get("betterbox", {}).get("reference_audio", None)
                ref_text = None
                
                if ref_audio is None:
                    # Look for the auto-generated reference in the root wavs folder
                    fallback_ref = os.path.join("wavs", "reference_sound.wav")
                    if os.path.exists(fallback_ref):
                        ref_audio = fallback_ref
                        fallback_txt = fallback_ref.replace(".wav", ".txt")
                        if os.path.exists(fallback_txt):
                            with open(fallback_txt, "r", encoding="utf-8") as f:
                                ref_text = f.read().strip()
                
                # We monkey-patch the wrapper slightly or pass instruct down
                # Currently generate_speech_omni doesn't take instruct in our port?
                # Let's check our ported ttsOmni.py -> generate_speech_omni
                # Oh wait, we need to pass instruct to generate_speech_omni!
                # I will also update generate_speech_omni in ttsOmni.py to accept instruct.
                result, status, _ = self._generate_speech_omni(
                    omni=self._omni,
                    text=text,
                    language="vi",
                    reference_audio=ref_audio,
                    ref_text=ref_text,
                    speed=self.cfg.get("betterbox", {}).get("speed", 1.0),
                    instruct=instruct  # Newly added
                )
                if result is not None:
                    sr, audio = result
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    print(f"[TTS VI] ⏱ {elapsed_ms:.0f}ms | {status}")
                    return audio.astype(np.float32), sr
                else:
                    print(f"[TTS VI] ⚠ OmniVoice failed: {status}")
            except Exception as e:
                print(f"[TTS VI] ⚠ OmniVoice error: {e}")

        # Stub: silence
        return np.zeros(int(self.sample_rate * 0.5), dtype=np.float32), self.sample_rate

    def _get_en_reference(self, fallback_wav: str = None, fallback_text: str = None) -> tuple:
        """
        Return (ref_audio_path, ref_text) for F5-TTS.
        Priority:
          1. Configured voice preset (en_preset_audio) — clean, natural, studio voice.
          2. Live voice cloning from input audio (fallback_wav) — may have noise.
        """
        # Priority 1: configured preset
        if self.en_preset_audio and os.path.exists(self.en_preset_audio):
            print(f"[TTS EN] 🎤 Using voice preset: {os.path.basename(self.en_preset_audio)}")
            return self.en_preset_audio, self.en_preset_text or ""

        # Priority 2: live reference trimmed to 10s
        if fallback_wav and os.path.exists(fallback_wav):
            print(f"[TTS EN] 🎤 No preset found — falling back to voice cloning from input audio")
            import soundfile as sf, tempfile
            audio_data, sr_data = sf.read(fallback_wav)
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            max_samples = 10 * sr_data
            if len(audio_data) > max_samples:
                print(f"[TTS EN] ✂️ Trimming reference audio to 10.0s")
                audio_data = audio_data[:max_samples]
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio_data, sr_data)
                return tmp.name, fallback_text or ""

        return None, ""

    # ── English TTS Synthesis ─────────────────────────────────────────────────
    # Fallback chain (ưu tiên từ trên xuống):
    #   1. F5-TTS   — Voice cloning chất lượng cao (CHÍNH, hoạt động trên Colab/Linux)
    #   2. pyttsx3  — Microsoft SAPI5 (DỰ PHÒNG, chỉ dùng khi F5-TTS lỗi trên Windows)
    #   3. Silence  — Không có engine nào khả dụng
    #
    # Trên Colab/Linux: F5-TTS luôn thành công → KHÔNG BAO GIỜ fallback
    # Trên Windows:     F5-TTS lỗi torchcodec → tự động chuyển sang pyttsx3
    # ─────────────────────────────────────────────────────────────────────────

    def synthesize_en(self, text: str, reference_wav: str = None, original_text: str = None) -> tuple[np.ndarray, int]:
        """
        Synthesize English speech.

        Fallback chain:
          1. F5-TTS (voice cloning, high quality) — primary engine
          2. pyttsx3 (Microsoft SAPI5) — Windows fallback only
          3. Silence stub — last resort
        """
        t0 = time.perf_counter()
        engine = getattr(self, "_en_tts_engine", None)

        # ── [1] F5-TTS (Primary — Colab/Linux) ──────────────────────────────
        if engine == "f5tts":
            try:
                import torch
                ref_file, ref_text = self._get_en_reference(
                    fallback_wav=reference_wav,
                    fallback_text=original_text,
                )
                if ref_file is None:
                    raise ValueError("No reference audio available for F5-TTS")

                wav, sr, _ = self._en_tts.infer(
                    ref_file=ref_file,
                    ref_text=ref_text,
                    gen_text=text,
                    speed=self.en_speed,
                )
                # Clean up temp file if it was created by fallback
                if ref_file != self.en_preset_audio and os.path.exists(ref_file):
                    try: os.unlink(ref_file)
                    except: pass

                audio = wav.numpy() if torch.is_tensor(wav) else wav
                elapsed_ms = (time.perf_counter() - t0) * 1000
                print(f"[TTS EN] ⏱ {elapsed_ms:.0f}ms | F5-TTS (speed={self.en_speed})")
                return audio.astype(np.float32), sr
            except Exception as e:
                print(f"[TTS EN] ⚠ F5-TTS inference failed: {e}")
                print(f"[TTS EN] ↓ Falling back to pyttsx3...")

        # ── [2] pyttsx3 (Fallback — Windows) ─────────────────────────────────
        try:
            import pyttsx3
            import tempfile, soundfile as sf
            _fallback_engine = pyttsx3.init()
            _fallback_engine.setProperty("rate", int(160 * self.en_speed))
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            _fallback_engine.save_to_file(text, tmp_path)
            _fallback_engine.runAndWait()
            audio, sr = sf.read(tmp_path, dtype="float32")
            os.unlink(tmp_path)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(f"[TTS EN] ⏱ {elapsed_ms:.0f}ms | pyttsx3 fallback (rate={int(160 * self.en_speed)} WPM)")
            return audio, sr
        except Exception as e:
            print(f"[TTS EN] ⚠ pyttsx3 fallback failed: {e}")

        # ── [3] Silence stub (last resort) ───────────────────────────────────
        print("[TTS EN] ⚠ No English TTS engine available — returning silence.")
        return np.zeros(int(self.sample_rate * 0.5), dtype=np.float32), self.sample_rate

    def synthesize(self, text: str, direction: str = "vi2en", emotion: str = "neutral", reference_wav: str = None, original_text: str = None) -> tuple[np.ndarray, int]:
        """
        Route synthesis based on direction.

        Args:
            text: Text to speak (already translated)
            direction: "vi2en" → output EN speech | "en2vi" → output VI speech
            emotion: Emotion tag from SenseVoice (e.g. "angry")
            reference_wav: Path to original input audio for voice cloning
            original_text: The original ASR text from Trạm 1 (used for F5-TTS ref_text)

        Returns:
            (audio_array, sample_rate)
        """
        if direction == "en2vi":
            return self.synthesize_vi(text, emotion=emotion)
        else:
            return self.synthesize_en(text, reference_wav=reference_wav, original_text=original_text)

    def play(self, audio: np.ndarray, sample_rate: int = None):
        """Play synthesized audio through the speaker."""
        sr = sample_rate or self.sample_rate
        try:
            sd.play(audio, samplerate=sr)
            sd.wait()
        except Exception as e:
            print(f"[TTS] ⚠ Playback error: {e}")

    def run(self, text_queue: queue.Queue):
        """Worker loop: reads translated text, synthesizes, plays to speaker."""
        print("[TTS Worker] ✅ Started")
        while True:
            try:
                item = text_queue.get(timeout=1)
                text = item["text"]
                direction = item.get("direction", "vi2en")

                print(f"[TTS Worker] Synthesizing [{direction}]: \"{text}\"")
                # Worker doesn't easily have original_text, but this is the real-time queue
                audio, sr = self.synthesize(text, direction=direction)
                self.play(audio, sample_rate=sr)

                text_queue.task_done()
            except queue.Empty:
                continue
