import sys
import os
import time
import argparse
import yaml
import soundfile as sf
import numpy as np
import librosa

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from asr.asr_manager import ASRManager
from translation.mt_engine import Translator
from tts.tts_engine import TTSEngine

def test_offline_file(input_wav: str, output_wav: str, direction: str):
    print("\n" + "="*50)
    print(f"🎙️ BẮT ĐẦU TEST OFFLINE PIPELINE ({direction.upper()})")
    print(f"📥 Đầu vào: {input_wav}")
    print(f"📤 Đầu ra : {output_wav}")
    print("="*50)

    if not os.path.exists(input_wav):
        print(f"❌ Không tìm thấy file âm thanh đầu vào: {input_wav}")
        return

    # 1. Load config
    cfg_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2. Khởi tạo các trạm
    print("\n[Hệ thống] Đang khởi tạo các mô hình...")
    
    asr = ASRManager(cfg)
    asr.load()

    mt = Translator(cfg)
    mt.load()

    # Truyền file input gốc làm reference audio để clone giọng người nói
    cfg["tts"]["betterbox"]["reference_audio"] = os.path.abspath(input_wav)
    tts = TTSEngine(cfg)
    tts.load()

    # 3. Đọc file âm thanh / video
    print(f"\n[Xử lý] Đang đọc file: {input_wav}")
    
    # Hỗ trợ tự động trích xuất âm thanh từ Video (mp4, mkv, avi, mov)
    if input_wav.lower().endswith((".mp4", ".mkv", ".avi", ".mov", ".webm")):
        print("🎥 Đã phát hiện định dạng Video. Đang trích xuất âm thanh bằng ffmpeg...")
        temp_wav = "temp_extracted_audio.wav"
        import subprocess
        try:
            # Dùng ffmpeg để lấy kênh audio ra file tạm
            subprocess.run([
                "ffmpeg", "-i", input_wav, "-q:a", "0", "-map", "a", temp_wav, "-y"
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            audio, sr = librosa.load(temp_wav, sr=16000)
            os.remove(temp_wav)
            print("✅ Trích xuất âm thanh thành công.")
        except Exception as e:
            print(f"❌ Lỗi trích xuất âm thanh từ Video (Bạn đã cài ffmpeg chưa?): {e}")
            return
    else:
        # Đọc file audio bình thường
        audio, sr = librosa.load(input_wav, sr=16000)
    
    # 4. Trạm 1: Nhận diện ASR
    asr_engine_name = "GIPFormer" if direction == "vi2en" else "SenseVoice"
    print(f"\n[Trạm 1: ASR - {asr_engine_name}] Bắt đầu nhận diện...")
    t0 = time.perf_counter()
    asr_result = asr.transcribe(audio, direction=direction)
    asr_ms = (time.perf_counter() - t0) * 1000
    
    text_src = asr_result["text"]
    emotion = asr_result.get("emotion", "neutral")
    src_lang = "VI" if direction == "vi2en" else "EN"
    tgt_lang = "EN" if direction == "vi2en" else "VI"
    
    print(f"   ⏱ Thời gian: {asr_ms:.0f}ms")
    print(f"   📝 Văn bản ({src_lang}): '{text_src}'")
    if direction == "en2vi":
        print(f"   🎭 Cảm xúc phát hiện: {emotion.upper()}")

    if not text_src:
        print("❌ Nhận diện không thành công hoặc file âm thanh trống.")
        return

    # 5. Trạm 2: Dịch thuật (MT)
    print(f"\n[Trạm 2: MT - VietAI/envit5] Bắt đầu dịch {src_lang} -> {tgt_lang}...")
    t0 = time.perf_counter()
    text_tgt = mt.translate(text_src, direction=direction)
    mt_ms = (time.perf_counter() - t0) * 1000
    print(f"   ⏱ Thời gian: {mt_ms:.0f}ms")
    print(f"   📝 Bản dịch ({tgt_lang}): '{text_tgt}'")

    # 6. Trạm 3: Phát âm (TTS)
    tts_engine_name = "F5-TTS/gTTS" if direction == "vi2en" else "OmniVoice (Voice Clone)"
    print(f"\n[Trạm 3: TTS - {tts_engine_name}] Tổng hợp giọng nói Tiếng {tgt_lang}...")
    if direction == "vi2en":
        print(f"   🎙️ Đang clone giọng nói từ: {input_wav}")
    t0 = time.perf_counter()
    out_audio, out_sr = tts.synthesize(
        text_tgt,
        direction=direction,
        emotion=emotion,
        reference_wav=os.path.abspath(input_wav),
        original_text=text_src,
    )
    tts_ms = (time.perf_counter() - t0) * 1000
    print(f"   ⏱ Thời gian: {tts_ms:.0f}ms")

    # 7. Lưu file kết quả
    if out_audio is not None and len(out_audio) > 0:
        sf.write(output_wav, out_audio, out_sr)
        print(f"\n✅ ĐÃ LƯU THÀNH CÔNG FILE ÂM THANH: {output_wav}")
    else:
        print("\n❌ Lỗi tạo file âm thanh.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Pipeline with Audio/Video File")
    parser.add_argument("--input", "-i", type=str, required=True, help="Đường dẫn file đầu vào (.wav, .mp3, .mp4)")
    parser.add_argument("--output", "-o", type=str, default="output_audio.wav", help="Tên file audio đầu ra")
    parser.add_argument("--direction", "-d", type=str, choices=["vi2en", "en2vi"], default="en2vi", help="Chiều dịch thuật (mặc định: en2vi)")
    
    args = parser.parse_args()
    test_offline_file(args.input, args.output, args.direction)
