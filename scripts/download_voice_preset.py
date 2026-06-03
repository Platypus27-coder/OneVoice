#!/usr/bin/env python3
"""
Download a high-quality English voice preset for F5-TTS.
Uses a LibriSpeech sample — natural, studio-recorded, NOT robotic.

Usage (run once on Colab or local):
    python scripts/download_voice_preset.py

Output:
    src/tts/presets/en_male_narrator.wav   (5-second clean male voice)
    src/tts/presets/en_male_narrator.txt   (transcript)
"""

import os
import sys
import urllib.request

PRESET_DIR = os.path.join(os.path.dirname(__file__), "../src/tts/presets")
os.makedirs(PRESET_DIR, exist_ok=True)

# ── LibriSpeech sample (public domain, studio quality, natural male narrator) ──
# Speaker 1088 from LibriSpeech test-clean — clear, mid-tempo American English
LIBRISPEECH_URL = (
    "https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy"
    "/resolve/main/data/validation/clean/1320-122617-0001.flac"
)

# Alternative: use the f5-tts built-in test reference (already installed with f5-tts)
F5TTS_BUILTIN_TEXT = "Some call me nature, others call me mother nature."

OUT_WAV = os.path.join(PRESET_DIR, "en_male_narrator.wav")
OUT_TXT = os.path.join(PRESET_DIR, "en_male_narrator.txt")


def _find_f5tts_builtin():
    """Dynamically find F5-TTS built-in reference audio (works on Windows, Linux, Colab)."""
    try:
        import f5_tts
        f5_dir = f5_tts.__path__[0]
        ref = os.path.join(f5_dir, "infer", "examples", "basic", "basic_ref_en.wav")
        if os.path.exists(ref):
            return ref
    except (ImportError, IndexError):
        pass
    return None


def download_preset():
    print("="*60)
    print("📥 Đang tải voice preset tiếng Anh chất lượng cao...")
    print("="*60)

    # Option 1: F5-TTS built-in reference (fastest, cross-platform)
    f5_ref = _find_f5tts_builtin()
    if f5_ref:
        import shutil
        shutil.copy(f5_ref, OUT_WAV)
        with open(OUT_TXT, "w", encoding="utf-8") as f:
            f.write(F5TTS_BUILTIN_TEXT)
        print(f"✅ Đã dùng giọng mẫu có sẵn từ F5-TTS: {OUT_WAV}")
        print(f"   📝 Transcript: '{F5TTS_BUILTIN_TEXT}'")
        return True

    # Option 2: Download from HuggingFace LibriSpeech
    try:
        print(f"🌐 Đang tải từ LibriSpeech (HuggingFace)...")
        urllib.request.urlretrieve(LIBRISPEECH_URL, OUT_WAV.replace(".wav", ".flac"))

        # Convert flac → wav
        import soundfile as sf
        data, sr = sf.read(OUT_WAV.replace(".wav", ".flac"))
        sf.write(OUT_WAV, data, sr)
        os.remove(OUT_WAV.replace(".wav", ".flac"))

        transcript = "He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered, flour-fattened sauce."
        with open(OUT_TXT, "w", encoding="utf-8") as f:
            f.write(transcript)

        print(f"✅ Đã tải xong: {OUT_WAV}")
        print(f"   📝 Transcript: '{transcript[:60]}...'")
        return True

    except Exception as e:
        print(f"❌ Không tải được từ LibriSpeech: {e}")

    # Option 3: Generate a short synthetic reference using gTTS (better than nothing)
    try:
        print("🔄 Fallback: Tạo giọng mẫu bằng gTTS...")
        from gtts import gTTS
        from pydub import AudioSegment
        import tempfile

        ref_text = "Attention all personnel. The hydraulic pump on excavator number three is leaking. Please proceed to the maintenance zone immediately."
        tts = gTTS(text=ref_text, lang="en", slow=False, tld="com")
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tts.save(tmp.name)
            seg = AudioSegment.from_mp3(tmp.name)
            # Take first 8 seconds
            seg = seg[:8000]
            seg.export(OUT_WAV, format="wav")
            os.unlink(tmp.name)

        with open(OUT_TXT, "w", encoding="utf-8") as f:
            f.write(ref_text[:80])

        print(f"⚠ Đã tạo giọng mẫu bằng gTTS (chất lượng thấp hơn): {OUT_WAV}")
        return True

    except Exception as e:
        print(f"❌ Tất cả phương án đều thất bại: {e}")
        return False


def print_config_snippet():
    print("\n" + "="*60)
    print("📋 Thêm đoạn này vào config/config.yaml:")
    print("="*60)
    print(f"""
tts:
  en_speed: 0.85
  en_preset_audio: "src/tts/presets/en_male_narrator.wav"
  en_preset_text:  "Some call me nature, others call me mother nature."
""")


if __name__ == "__main__":
    success = download_preset()
    if success:
        print_config_snippet()
    sys.exit(0 if success else 1)
