import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from translation.mt_engine import Translator
from tts.tts_engine import TTSEngine
import yaml

def test_vietnamese_to_english():
    print("\n" + "="*50)
    print("🇻🇳 TEST CHẾ ĐỘ: VIỆT NAM -> TIẾNG ANH (vi2en)")
    print("="*50)

    # 1. Load config
    cfg_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2. Khởi tạo Trạm 2 (Dịch thuật) và Trạm 3 (TTS)
    print("[Hệ thống] Đang khởi tạo các mô hình (Mất khoảng 10s)...")
    mt = Translator(cfg)
    mt.load()

    tts = TTSEngine(cfg)
    tts.load()

    # 3. Kịch bản test
    test_text_vi = "Máy xúc số 3 đang bị lỗi thủy lực, cần kỹ sư kiểm tra ngay lập tức."
    print(f"\n🎤 [Kỹ sư Việt Nam nói]: {test_text_vi}")

    # Bước 1: Dịch (VI -> EN)
    t0 = time.perf_counter()
    translated_en = mt.translate(test_text_vi, direction="vi2en")
    print(f"🧠 [MarianMT Dịch ({(time.perf_counter()-t0)*1000:.0f}ms)]: {translated_en}")

    # Bước 2: Đọc tiếng Anh
    print(f"🔊 [TTS Tiếng Anh] Đang tổng hợp giọng nói...")
    t0 = time.perf_counter()
    audio, sr = tts.synthesize(translated_en, direction="vi2en")
    print(f"✅ Đã tạo xong âm thanh tiếng Anh ({(time.perf_counter()-t0)*1000:.0f}ms)")
    
    print("\n" + "="*50)
    print("🇬🇧 TEST CHẾ ĐỘ: TIẾNG ANH -> VIỆT NAM (en2vi) VỚI CẢM XÚC")
    print("="*50)

    test_text_en = "This machine is completely broken! Fix it right now!"
    emotion_tag = "angry"
    
    print(f"\n🎤 [Chuyên gia Anh nói]: {test_text_en} (ASR bắt được cảm xúc: {emotion_tag.upper()})")

    # Bước 1: Dịch (EN -> VI)
    t0 = time.perf_counter()
    translated_vi = mt.translate(test_text_en, direction="en2vi")
    print(f"🧠 [MarianMT Dịch ({(time.perf_counter()-t0)*1000:.0f}ms)]: {translated_vi}")

    # Bước 2: Đọc tiếng Việt (OmniVoice) kèm cảm xúc
    print(f"🔊 [OmniVoice TTS] Đang tổng hợp giọng điệu {emotion_tag.upper()}...")
    t0 = time.perf_counter()
    audio_vi, sr_vi = tts.synthesize(translated_vi, direction="en2vi", emotion=emotion_tag)
    print(f"✅ Đã tạo xong âm thanh tiếng Việt gắt gỏng ({(time.perf_counter()-t0)*1000:.0f}ms)")
    print("\n🎉 BÀI TEST THÀNH CÔNG TỐT ĐẸP!")

if __name__ == "__main__":
    test_vietnamese_to_english()
