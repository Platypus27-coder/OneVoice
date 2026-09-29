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
import shutil
import subprocess
import base64
import json
import wave
from pathlib import Path
import numpy as np
from audio.playback import prepare_playback_audio
try:
    import sounddevice as sd
except ImportError:
    sd = None


class TTSEngine:
    """
    Text-to-Speech router for VI↔EN pipeline.

    Routing:
      "vi2en" direction → output is English → English TTS (VITS/espeak)
      "en2vi" direction → output is Vietnamese → OmniVoice/BetterBox
    """

    def __init__(self, config: dict, profile: str = "development", offline: bool = False,
                 backend: str | None = None, allow_online_tts: bool = False):
        self.cfg = config["tts"]
        self.profile = profile
        self.offline = offline
        self.backend = str(backend or self.cfg.get("backend", "auto")).lower()
        if self.backend not in {"auto", "espeak", "sapi", "gtts"}:
            raise ValueError("TTS backend must be auto, espeak, sapi, or gtts")
        # Non-offline development already supports its historical online
        # fallback. Edge/offline calls require the explicit CLI opt-in.
        self.allow_online_tts = bool(allow_online_tts or not offline)
        if self.backend == "gtts" and not self.allow_online_tts:
            raise ValueError(
                "gTTS sends translated text to Google. Use --allow-online-tts "
                "to enable this network request explicitly."
            )
        self.tts_tier = config.get("profiles", {}).get(profile, {}).get("tts_tier", "premium")
        self.sample_rate = config["audio"]["sample_rate"]
        self.default_engine = self.cfg.get("default_engine", "betterbox")
        self.en_speed = float(self.cfg.get("en_speed", 0.85))
        # Voice preset: a pre-recorded natural EN voice as reference (preferred over live cloning)
        self.en_preset_audio = self.cfg.get("en_preset_audio", None)
        self.en_preset_text  = self.cfg.get("en_preset_text",  None)
        self._omni = None          # OmniVoice for Vietnamese TTS
        self._en_tts = None        # English TTS engine
        self._en_tts_executable = None
        self._vallex = None        # VALL-E X (Premium Mode)
        self._vi_tts_engine = None
        self._vi_tts_engine_name = None
        self._sapi_voice_name = None
        self._sapi_vi_voice_name = None
        self._gtts_class = None
        self._audio_segment_class = None
        self.output_device = config["audio"].get("output_device")
        self.synthesis_timeout_s = float(self.cfg.get("synthesis_timeout_s", 15.0))
        if not 0 < self.synthesis_timeout_s < float("inf"):
            raise ValueError("tts.synthesis_timeout_s must be positive and finite")
        for language in ("en", "vi"):
            windows_rate = int(self.cfg.get(f"windows_{language}_rate", -1))
            if not -10 <= windows_rate <= 10:
                raise ValueError(f"tts.windows_{language}_rate must be between -10 and 10")

    @staticmethod
    def _find_espeak_executable() -> str | None:
        executable = shutil.which("espeak-ng") or shutil.which("espeak")
        if executable:
            return executable
        # Explicit Python paths do not necessarily execute Conda's PATH hooks.
        # Use the existing local runtime; never download a backend here.
        if sys.platform == "win32":
            candidate = Path(sys.prefix) / "espeak-ng-runtime" / "eSpeak NG" / "espeak-ng.exe"
            if candidate.is_file():
                return str(candidate)
        return None

    def _require_windows_espeak(self, executable: str | None) -> None:
        if sys.platform == "win32" and not executable:
            raise RuntimeError(
                "Windows offline TTS has no usable SAPI voice for the target language and no local eSpeak fallback. "
                "Activate the onevoice environment with its eSpeak NG hook, "
                "or install the local runtime under the environment's "
                "espeak-ng-runtime/eSpeak NG directory."
            )

    @staticmethod
    def _espeak_environment(executable: str) -> dict[str, str]:
        """Give a portable runtime its data path without mutating parent env."""
        child_env = os.environ.copy()
        data_dir = Path(executable).resolve().parent / "espeak-ng-data"
        if data_dir.is_dir():
            # eSpeak's --version/--help initialize without the CLI --path.
            # Missing Windows registry data can otherwise cause a native crash.
            child_env["ESPEAK_DATA_PATH"] = str(data_dir)
        return child_env

    @staticmethod
    def _powershell_executable() -> str | None:
        executable = shutil.which("powershell.exe") or shutil.which("powershell")
        if executable:
            return executable
        if sys.platform == "win32":
            system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
            candidate = system_root / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
            if candidate.is_file():
                return str(candidate)
        return None

    def _run_windows_sapi(self, request: dict) -> dict:
        """Use an isolated PowerShell/.NET process for Windows offline speech."""
        language = request.get("language", "en")
        if language not in {"en", "vi"}:
            raise ValueError("Windows SAPI language must be en or vi")
        request = {**request, "language": language}
        if sys.platform != "win32":
            raise RuntimeError("Windows SAPI is available only on Windows")
        executable = self._powershell_executable()
        if not executable:
            raise RuntimeError("Windows PowerShell was not found; cannot use the local SAPI voice")

        script = r"""
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
$synth = $null
$fileStream = $null
try {
    $request = [Console]::In.ReadToEnd() | ConvertFrom-Json
    Add-Type -AssemblyName System.Speech
    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $language = [string]$request.language
    $voices = @($synth.GetInstalledVoices() | Where-Object {
        $_.Enabled -and ($_.VoiceInfo.Culture.Name -eq $language -or $_.VoiceInfo.Culture.Name -like "$language-*")
    })
    # .NET Framework System.Speech can omit installed OneCore voices.
    # Read their SAPI token category directly; do NOT copy/edit registry keys.
    $useOneCore = $false
    if ($voices.Count -eq 0 -and $language -eq 'vi') {
        $category = New-Object -ComObject SAPI.SpObjectTokenCategory
        $category.SetId('HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech_OneCore\Voices', $false)
        $voices = @($category.EnumerateTokens() | ForEach-Object {
            $token = $_
            $culture = [System.Globalization.CultureInfo]::GetCultureInfo(
                [Convert]::ToInt32(($token.GetAttribute('Language') -split ';')[0], 16))
            if ($culture.Name -eq $language -or $culture.Name -like "$language-*") {
                [pscustomobject]@{ Token = $token; VoiceInfo = [pscustomobject]@{
                    Name = $token.GetDescription(); Culture = $culture
                } }
            }
        })
        $useOneCore = $true
    }
    if ($voices.Count -eq 0) { throw "No enabled '$language' Windows speech voice is installed." }
    $voice = $null
    if (-not [string]::IsNullOrWhiteSpace([string]$request.voice)) {
        $voice = $voices | Where-Object { $_.VoiceInfo.Name -eq [string]$request.voice } | Select-Object -First 1
        if (-not $voice) { throw "Requested '$language' voice '$($request.voice)' is not installed or enabled for this language." }
    } else {
        if ($language -eq 'en') {
            $voice = $voices | Where-Object { $_.VoiceInfo.Name -match 'David|Zira' } | Select-Object -First 1
            if (-not $voice) { $voice = $voices | Where-Object { $_.VoiceInfo.Culture.Name -eq 'en-US' } | Select-Object -First 1 }
        } else {
            $voice = $voices | Where-Object { $_.VoiceInfo.Culture.Name -eq 'vi-VN' } | Select-Object -First 1
        }
        if (-not $voice) { $voice = $voices | Select-Object -First 1 }
    }
    if (-not $useOneCore) { $synth.SelectVoice($voice.VoiceInfo.Name) }
    if ($request.probe) {
        $result = @{ voice = $voice.VoiceInfo.Name; culture = $voice.VoiceInfo.Culture.Name }
    } elseif ($useOneCore) {
        $oneCore = New-Object -ComObject SAPI.SpVoice
        $oneCore.Voice = $voice.Token
        $oneCore.Rate = [int]$request.rate
        $fileStream = New-Object -ComObject SAPI.SpFileStream
        $fileStream.Format.Type = 22 # SAFT22kHz16BitMono (PCM)
        $fileStream.Open([string]$request.path, 3, $false) # SSFMCreateForWrite
        $oneCore.AudioOutputStream = $fileStream
        [void]$oneCore.Speak([string]$request.text, 16) # SVSFIsNotXML: literal text only
        $fileStream.Close()
        $fileStream = $null
        $result = @{ voice = $voice.VoiceInfo.Name; culture = $voice.VoiceInfo.Culture.Name; written = $true }
    } else {
        $synth.Rate = [int]$request.rate
        $synth.SetOutputToWaveFile([string]$request.path)
        $synth.Speak([string]$request.text)
        $synth.SetOutputToNull()
        $result = @{ voice = $voice.VoiceInfo.Name; culture = $voice.VoiceInfo.Culture.Name; written = $true }
    }
    [Console]::Out.WriteLine(($result | ConvertTo-Json -Compress))
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
} finally {
    if ($fileStream) { $fileStream.Close() }
    if ($synth) { $synth.Dispose() }
}
"""
        encoded_script = base64.b64encode(script.encode("utf-16le")).decode("ascii")
        try:
            result = subprocess.run(
                [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded_script],
                input=json.dumps(request, ensure_ascii=True),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.synthesis_timeout_s,
                **({"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}),
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Windows SAPI exceeded {self.synthesis_timeout_s:g}s") from exc
        except subprocess.CalledProcessError as exc:
            detail = str(exc.stderr or exc.stdout or "").strip()[:500]
            raise RuntimeError(f"Windows SAPI failed: {detail or exc.returncode}") from exc
        try:
            return json.loads(result.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as exc:
            raise RuntimeError("Windows SAPI returned an invalid status response") from exc

    def _load_online_gtts(self) -> None:
        if not self.allow_online_tts:
            raise RuntimeError(
                "Online gTTS is disabled. Pass --allow-online-tts; synthesized text is sent to Google."
            )
        if self._gtts_class is not None and self._audio_segment_class is not None:
            return
        try:
            from gtts import gTTS
            from pydub import AudioSegment
            import imageio_ffmpeg
        except ImportError as exc:
            raise RuntimeError(
                "Online gTTS needs optional packages. Install requirements-online-tts.txt first."
            ) from exc
        AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
        self._gtts_class = gTTS
        self._audio_segment_class = AudioSegment
        print(
            "[TTS] ⚠ Online gTTS enabled: translated speech text will be sent to Google Translate TTS."
        )

    def _synthesize_sapi_en(self, text: str) -> tuple[np.ndarray, int]:
        return self._synthesize_windows_sapi(text, "en")

    def _synthesize_windows_sapi(self, text: str, language: str) -> tuple[np.ndarray, int]:
        import tempfile

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                tmp_path = handle.name
            self._run_windows_sapi({
                "probe": False,
                "language": language,
                "voice": (self._sapi_voice_name if language == "en" else self._sapi_vi_voice_name) or "",
                "rate": int(self.cfg.get(f"windows_{language}_rate", -1)),
                "text": text,
                "path": tmp_path,
            })
            with wave.open(tmp_path, "rb") as handle:
                channels = handle.getnchannels()
                sample_width = handle.getsampwidth()
                sample_rate = handle.getframerate()
                frames = handle.readframes(handle.getnframes())
            if sample_width != 2:
                raise RuntimeError(f"Windows SAPI produced {sample_width * 8}-bit audio; expected 16-bit PCM")
            audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)
            if self.is_silence(audio):
                raise RuntimeError("Windows SAPI produced an empty or silent WAV")
            return audio.astype(np.float32), int(sample_rate)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _synthesize_gtts(self, text: str, language: str) -> tuple[np.ndarray, int]:
        """Generate a new utterance online; never used without explicit opt-in."""
        if not self.allow_online_tts:
            raise RuntimeError("gTTS is unavailable without the explicit online TTS opt-in")
        if self._gtts_class is None or self._audio_segment_class is None:
            self._load_online_gtts()
        import tempfile

        mp3_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as handle:
                mp3_path = handle.name
            self._gtts_class(text=text, lang=language, slow=False,
                             timeout=self.synthesis_timeout_s).save(mp3_path)
            segment = self._audio_segment_class.from_mp3(mp3_path).set_channels(1).set_frame_rate(self.sample_rate).set_sample_width(2)
            samples = np.asarray(segment.get_array_of_samples(), dtype="<i2").astype(np.float32) / 32768.0
            return samples, int(segment.frame_rate)
        finally:
            if mp3_path and os.path.exists(mp3_path):
                os.unlink(mp3_path)

    def load(self, direction: str | None = None):
        """Initialize all TTS backends."""
        print(f"[TTS] Initializing engines...")
        if direction in (None, "en2vi"):
            # Explicit offline/demo mode uses a real local system voice and
            # avoids OmniVoice's large optional dependency stack.
            if self.backend in {"gtts", "sapi", "espeak"}:
                self._load_edge_vi_tts()
            elif self.offline and self.cfg.get("offline_engine") == "pyttsx3":
                self._load_edge_vi_tts()
            elif self.tts_tier == "premium":
                self._load_omnivoice()
            else:
                self._load_edge_vi_tts()
        if direction in (None, "vi2en"):
            self._load_english_tts()
            if getattr(self, "_en_tts_engine", None) == "f5tts":
                self._auto_prepare_preset()
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

        if self.offline:
            raise FileNotFoundError(
                f"Offline English voice preset not found: {self.en_preset_audio}"
            )

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
            if not os.path.exists(model_path) and self.offline:
                raise FileNotFoundError(f"Offline OmniVoice model not found: {model_path}")
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
                if self.offline:
                    raise FileNotFoundError(
                        f"Offline OmniVoice reference audio not found: {ref_path}"
                    )
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
            if self.offline:
                raise RuntimeError(f"Offline OmniVoice startup failed: {e}") from e
            print(f"[TTS] ⚠ Failed to load OmniVoice: {e}")
            self._omni = None

    def _load_english_tts(self):
        """
        Load English TTS.
        Priority: explicit backend → F5-TTS (premium) → Windows SAPI/eSpeak,
        or pyttsx3 on other systems → development-only gTTS.

        A bare VITS ONNX graph is deliberately not accepted: text phonemization,
        speaker/language metadata and output scaling are part of the deployable
        artifact contract. Loading a graph without its adapter previously caused
        the runtime to claim VITS while actually falling through to pyttsx3.
        """
        if self.backend == "gtts":
            self._load_online_gtts()
            self._en_tts_engine = "gtts-online"
            return

        # Keep the native eSpeak CLI available as a deterministic local escape
        # hatch. On headless Colab, pyttsx3 can occasionally return a valid WAV
        # container containing only zeros after many repeated calls.
        self._en_tts_executable = self._find_espeak_executable()

        if self.backend == "espeak":
            if not self._en_tts_executable:
                raise RuntimeError("The requested eSpeak backend is not installed")
            self._en_tts_engine = "espeak-ng-offline-demo"
            print("[TTS] ✅ Explicit local eSpeak backend selected (English voice=en-us).")
            return
        if self.backend == "sapi":
            if sys.platform != "win32":
                raise RuntimeError("Windows SAPI backend is available only on Windows")
            voice = self._run_windows_sapi({
                "probe": True,
                "voice": str(self.cfg.get("windows_en_voice", "")),
            })
            self._sapi_voice_name = voice["voice"]
            self._en_tts_engine = "windows-sapi-offline"
            print(f"[TTS] ✅ Windows offline English voice loaded: {self._sapi_voice_name}")
            return

        # Priority 1: F5-TTS — premium profile only
        try:
            if self.tts_tier != "premium":
                raise ImportError("F5-TTS disabled by edge profile")
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

        if sys.platform == "win32":
            try:
                voice = self._run_windows_sapi({
                    "probe": True,
                    "voice": str(self.cfg.get("windows_en_voice", "")),
                })
            except Exception as exc:
                print(f"[TTS] ⚠ Windows SAPI unavailable ({exc}); trying local eSpeak.")
                self._require_windows_espeak(self._en_tts_executable)
                self._en_tts_engine = "espeak-ng-offline-demo"
                print("[TTS] ✅ Native local English TTS loaded (voice=en-us).")
                return
            self._sapi_voice_name = voice["voice"]
            self._en_tts_engine = "windows-sapi-offline"
            print(f"[TTS] ✅ Windows offline English voice loaded: {self._sapi_voice_name}")
            return

        # Priority 2: pyttsx3 (offline, no voice clone)
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

        # Priority 3: gTTS (development-only online fallback)
        if self.offline:
            print("[TTS] ⚠ No offline English TTS backend available.")
            return
        try:
            from gtts import gTTS
            self._en_tts = "gtts"
            self._en_tts_engine = "gtts"
            print("[TTS] ✅ gTTS English TTS loaded (online fallback for Colab).")
            return
        except ImportError:
            pass

        print("[TTS] ⚠ No English TTS available — using silence stub.")
        if self.offline:
            raise RuntimeError("No offline English TTS backend available")

    def _load_edge_vi_tts(self):
        """Load a lightweight local VI fallback without remote model access."""
        if self.backend == "gtts":
            self._load_online_gtts()
            self._vi_tts_engine_name = "gtts-online"
            return
        if self.backend == "sapi" and sys.platform != "win32":
            raise RuntimeError("Windows SAPI backend is available only on Windows")
        # Linux Colab's pyttsx3/espeak driver can acknowledge save_to_file()
        # while leaving a non-RIFF placeholder behind.  Keep pyttsx3 as the
        # configured backend, but remember the native executable as a reliable
        # local fallback for writing a real PCM WAV.
        self._vi_tts_executable = self._find_espeak_executable()
        if self.backend == "espeak":
            if not self._vi_tts_executable:
                raise RuntimeError("The requested eSpeak backend is not installed")
            self._vi_tts_engine_name = "espeak-ng-offline-demo"
            return
        if sys.platform == "win32":
            try:
                voice = self._run_windows_sapi({
                    "probe": True,
                    "language": "vi",
                    "voice": str(self.cfg.get("windows_vi_voice", "")),
                })
            except Exception as exc:
                if self.backend == "sapi":
                    raise
                print(f"[TTS] ⚠ Vietnamese Windows SAPI unavailable ({exc}); trying local eSpeak demo voice.")
            else:
                self._sapi_vi_voice_name = voice["voice"]
                self._vi_tts_engine_name = "windows-sapi-offline"
                print(f"[TTS] ✅ Windows offline Vietnamese voice loaded: {self._sapi_vi_voice_name}")
                return
            self._require_windows_espeak(self._vi_tts_executable)
            self._vi_tts_engine_name = "espeak-ng-offline-demo"
            print("[TTS] ✅ Native local Vietnamese TTS loaded (voice=vi).")
            return
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
            self._vi_tts_engine = engine
            self._vi_tts_engine_name = "pyttsx3-offline-demo"
            print("[TTS] ✅ Local pyttsx3 VI fallback loaded for edge profile.")
        except Exception as exc:
            if self._vi_tts_executable:
                self._vi_tts_engine_name = "espeak-ng-offline-demo"
                print(f"[TTS] ✅ Local {os.path.basename(self._vi_tts_executable)} VI fallback loaded.")
                return
            raise RuntimeError(f"No local Vietnamese edge TTS available: {exc}") from exc

    def _synthesize_espeak_vi(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize a standards-compliant PCM WAV with the local eSpeak CLI."""
        executable = getattr(self, "_vi_tts_executable", None)
        return self._synthesize_espeak(text, executable, "vi", 165)

    def _synthesize_espeak_en(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize a non-silent English WAV via the native eSpeak CLI."""
        executable = getattr(self, "_en_tts_executable", None)
        return self._synthesize_espeak(text, executable, "en-us", max(80, int(160 * self.en_speed)))

    def _synthesize_espeak(self, text: str, executable: str | None,
                           voice: str, rate: int) -> tuple[np.ndarray, int]:
        if not executable:
            raise RuntimeError("espeak/espeak-ng executable is not installed")
        import tempfile

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            # Leave headroom in the synthesizer itself. Scaling a WAV after
            # synthesis would hide, rather than prevent, saturated PCM samples.
            command = [executable, "-v", voice, "-s", str(rate), "-a", "80", "-b", "1", "-w", tmp_path, "--stdin"]
            runtime_dir = Path(executable).resolve().parent
            if (runtime_dir / "espeak-ng-data").is_dir():
                command.append(f"--path={runtime_dir}")
            subprocess.run(
                command,
                input=text,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                timeout=self.synthesis_timeout_s,
                env=self._espeak_environment(executable),
                **({"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}),
            )
            with wave.open(tmp_path, "rb") as handle:
                channels = handle.getnchannels()
                sample_width = handle.getsampwidth()
                sample_rate = handle.getframerate()
                frames = handle.readframes(handle.getnframes())
            if sample_width == 2:
                audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
            elif sample_width == 1:
                audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            else:
                raise RuntimeError(f"Unsupported eSpeak sample width: {sample_width}")
            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)
            if self.is_silence(audio):
                raise RuntimeError("eSpeak produced an empty or silent WAV")
            return audio.astype(np.float32), int(sample_rate)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Local TTS exceeded {self.synthesis_timeout_s:g}s; no audio was played") from exc
        except subprocess.CalledProcessError as exc:
            detail = str(exc.stderr or "").strip()[:500]
            exit_hex = f"0x{exc.returncode & 0xffffffff:08X}"
            raise RuntimeError(
                f"Native eSpeak failed (exit={exc.returncode}, {exit_hex}); no audio was played. "
                f"Check the local runtime and espeak-ng-data path. {detail}"
            ) from exc
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def synthesize_vi(self, text: str, emotion: str = "neutral") -> tuple[np.ndarray, int]:
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

        if self._vi_tts_engine_name == "gtts-online":
            started = time.perf_counter()
            audio, sample_rate = self._synthesize_gtts(text, "vi")
            print(f"[TTS VI] {(time.perf_counter() - started) * 1000:.0f}ms | gTTS online")
            return audio, sample_rate

        if self._vi_tts_engine_name == "windows-sapi-offline":
            started = time.perf_counter()
            audio, sample_rate = self._synthesize_windows_sapi(text, "vi")
            print(f"[TTS VI] Windows SAPI voice={self._sapi_vi_voice_name} | {(time.perf_counter() - started) * 1000:.0f}ms")
            return audio, sample_rate

        if self._vi_tts_engine_name == "espeak-ng-offline-demo":
            started = time.perf_counter()
            audio, sample_rate = self._synthesize_espeak_vi(text)
            print(f"[TTS VI] Audio ready | native espeak voice=vi | {(time.perf_counter() - started) * 1000:.0f}ms")
            return audio, sample_rate

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

        if self._vi_tts_engine is not None:
            try:
                import tempfile
                import soundfile as sf
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                self._vi_tts_engine.save_to_file(text, tmp_path)
                self._vi_tts_engine.runAndWait()
                audio, sr = sf.read(tmp_path, dtype="float32")
                os.unlink(tmp_path)
                return audio, sr
            except Exception as exc:
                print(f"[TTS VI] ⚠ Local edge TTS failed: {exc}")

        # pyttsx3 may produce an invalid placeholder WAV under Colab's
        # headless eSpeak driver.  Retry through the native CLI before ever
        # returning the silence stub.
        if getattr(self, "_vi_tts_executable", None):
            try:
                audio, sr = self._synthesize_espeak_vi(text)
                self._vi_tts_engine_name = "espeak-ng-offline-demo"
                print(f"[TTS VI] ✅ {os.path.basename(self._vi_tts_executable)} fallback")
                return audio, sr
            except Exception as exc:
                print(f"[TTS VI] ⚠ eSpeak fallback failed: {exc}")

        if not self.offline:
            try:
                audio, sample_rate = self._synthesize_gtts(text, "vi")
                self._vi_tts_engine_name = "gtts-development"
                print("[TTS VI] Using gTTS development fallback; not valid for runtime edge.")
                return audio, sample_rate
            except Exception as exc:
                print(f"[TTS VI] gTTS fallback failed: {exc}")

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
    # Premium profiles may load F5-TTS. Windows system-voice mode uses native
    # eSpeak, while other systems retain pyttsx3 with a native eSpeak fallback.
    # None of these functional backend checks establishes voice intelligibility.
    # ─────────────────────────────────────────────────────────────────────────

    def synthesize_en(self, text: str, reference_wav: str = None, original_text: str = None) -> tuple[np.ndarray, int]:
        """
        Synthesize English speech.

        Fallback chain:
          1. F5-TTS (voice cloning, high quality) — primary engine
          2. Windows SAPI, then native eSpeak (Windows) or pyttsx3 (other systems)
          3. Silence stub — last resort
        """
        t0 = time.perf_counter()
        engine = getattr(self, "_en_tts_engine", None)

        if engine == "windows-sapi-offline":
            audio, sample_rate = self._synthesize_sapi_en(text)
            print(f"[TTS EN] Windows SAPI voice={self._sapi_voice_name} | {(time.perf_counter() - t0) * 1000:.0f}ms")
            return audio, sample_rate

        if engine == "gtts-online":
            audio, sample_rate = self._synthesize_gtts(text, "en")
            print(f"[TTS EN] {(time.perf_counter() - t0) * 1000:.0f}ms | gTTS online")
            return audio, sample_rate

        if engine == "espeak-ng-offline-demo":
            audio, sample_rate = self._synthesize_espeak_en(text)
            print(f"[TTS EN] Audio ready | native espeak voice=en-us | {(time.perf_counter() - t0) * 1000:.0f}ms")
            return audio, sample_rate

        if engine == "gtts":
            try:
                audio, sample_rate = self._synthesize_gtts(text, "en")
                elapsed_ms = (time.perf_counter() - t0) * 1000
                print(f"[TTS EN] {elapsed_ms:.0f}ms | gTTS development fallback")
                return audio, sample_rate
            except Exception as exc:
                print(f"[TTS EN] gTTS fallback failed: {exc}")

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
        tmp_path = None
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
            if self.is_silence(audio):
                raise RuntimeError("pyttsx3 produced silent audio")
            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(f"[TTS EN] ⏱ {elapsed_ms:.0f}ms | pyttsx3 fallback (rate={int(160 * self.en_speed)} WPM)")
            return audio, sr
        except Exception as e:
            print(f"[TTS EN] ⚠ pyttsx3 fallback failed: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        # Native eSpeak is more robust than pyttsx3's headless driver under a
        # long Colab soak. It remains a development/offline demo voice only.
        if getattr(self, "_en_tts_executable", None):
            try:
                audio, sr = self._synthesize_espeak_en(text)
                self._en_tts_engine = "espeak-ng-offline-demo"
                elapsed_ms = (time.perf_counter() - t0) * 1000
                print(f"[TTS EN] ⏱ {elapsed_ms:.0f}ms | {os.path.basename(self._en_tts_executable)} fallback")
                return audio, sr
            except Exception as e:
                print(f"[TTS EN] ⚠ eSpeak fallback failed: {e}")

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

    def play(self, audio: np.ndarray, sample_rate: int = None) -> dict:
        """Play to the selected output; raise on failure, not silent success."""
        if sd is None:
            raise RuntimeError("sounddevice is required for speaker playback")
        if self.is_silence(audio) or not np.isfinite(audio).all():
            raise RuntimeError("Cannot play silent or non-finite audio")
        sr = sample_rate or self.sample_rate
        try:
            info = sd.query_devices(self.output_device, kind="output")
            output_rate = float(info.get("default_samplerate", sr))
            played_audio = prepare_playback_audio(audio, sr, output_rate)
            output_rate = int(output_rate)
            host_api = (sd.query_hostapis(info["hostapi"])["name"]
                        if "hostapi" in info else None)
            channels = 1 if played_audio.ndim == 1 else played_audio.shape[1]
            sd.check_output_settings(device=info["index"], samplerate=output_rate,
                                     channels=channels, dtype="float32")
            print(f"[Playback] Started | output={info['name']} | api={host_api} | "
                  f"source_rate={sr} | playback_rate={output_rate}")
            # Resolve a name in this process and keep that exact device for
            # playback; never fall back to speakers if the headset is missing.
            sd.play(played_audio, samplerate=output_rate, device=info["index"])
            status = sd.wait(ignore_errors=False)
            if status:
                raise RuntimeError(f"Audio device reported an underrun/overflow: {status}")
            print(f"[Playback] Finished | output={info['name']}")
            return {"output_device": info["name"], "output_device_index": info["index"],
                    "output_host_api": host_api, "source_sample_rate": int(sr),
                    "source_samples": len(audio), "sample_rate": output_rate,
                    "samples": len(played_audio), "resampled": int(sr) != output_rate,
                    "source_duration_seconds": len(audio) / sr,
                    "playback_duration_seconds": len(played_audio) / output_rate,
                    "device_playback_completed": True}
        except Exception as e:
            raise RuntimeError(f"Audio playback failed: {e}") from e

    @staticmethod
    def is_silence(audio: np.ndarray, threshold: float = 1e-5) -> bool:
        value = np.asarray(audio)
        return value.size == 0 or float(np.max(np.abs(value), initial=0.0)) <= threshold

    def engine_name(self, direction: str) -> str:
        if direction == "vi2en":
            return str(getattr(self, "_en_tts_engine", None) or "unavailable")
        if self._omni is not None:
            return "omnivoice"
        if self._vi_tts_engine_name:
            return self._vi_tts_engine_name
        if self._vi_tts_engine is not None:
            return "pyttsx3"
        return "unavailable"

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
