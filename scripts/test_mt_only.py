import sys
import os
import yaml
import numpy as np

# Thêm src vào đường dẫn để import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from translation.mt_engine import Translator
from tts.tts_engine import TTSEngine
import soundfile as sf

def _tts_vi_gtts_fallback(text: str, output_path: str) -> bool:
    """Fallback: dùng gTTS (Google TTS online) để đọc tiếng Việt."""
    try:
        from gtts import gTTS
        from io import BytesIO
        from pydub import AudioSegment

        tts = gTTS(text=text, lang='vi', slow=False)
        mp3_buf = BytesIO()
        tts.write_to_fp(mp3_buf)
        mp3_buf.seek(0)

        audio_seg = AudioSegment.from_mp3(mp3_buf)
        audio_seg.export(output_path, format="wav")
        return True
    except Exception as e:
        print(f"[TTS VI Fallback] ⚠ gTTS cũng thất bại: {e}")
        return False

def test_translation():
    print("="*60)
    print("🚀 BẮT ĐẦU TEST NHANH TRẠM DẤU Câu (1.5) & DỊCH THUẬT (2)")
    print("="*60)

    # Load cấu hình
    cfg_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Khởi tạo Translator (Trạm 2)
    mt = Translator(cfg)
    mt.load()

    # Khởi tạo TTS (Trạm 3)
    tts = TTSEngine(cfg)
    tts.load()

    # Các câu test sử dụng từ vựng KHÔNG CÓ trong industrial_terms.csv
    # để đánh giá khả năng SUY LUẬN thực sự của mô hình (không dựa vào từ điển)
    test_cases = [
        # Test 1: Giao tiếp đời thường - văn nói tự nhiên (VI->EN)
        {
            "text": "Cậu đã làm gì với nó vậy thêm năng lượng hả nó hoạt động như thế nào vậy cho mình mượn chút đừng có keo kiệt vậy chứ hôm nay lớp mình có bài kiểm tra môn thể dục nên mình rất là cần nó luôn xài xong mình trả lại liền",
            "direction": "vi2en"
        },
        # Test 2: Công trường - từ vựng ĐÃ CÓ trong dictionary (VI->EN) (Baseline)
        {
            "text": "Ê bạn ơi cái máy xúc số ba nó bị xì nhớt thủy lực rồi bơm bê tông cũng kẹt luôn qua kiểm tra lẹ giùm mình đi chứ để vậy là cháy van an toàn nha",
            "direction": "vi2en"
        },
        # Test 3: Từ vựng NGOÀI dictionary - Kỹ thuật hàng hải/cảng (EN->VI)
        # "gantry crane", "container yard", "draft survey" KHÔNG có trong CSV
        {
            "text": "the gantry crane at berth seven is malfunctioning we cannot unload the containers the draft survey shows the vessel is listing to port side",
            "direction": "en2vi"
        },
        # Test 4: Từ vựng NGOÀI dictionary - Kỹ thuật điện năng lượng mặt trời (EN->VI)
        # "solar inverter", "photovoltaic panel", "string combiner box" KHÔNG có trong CSV
        {
            "text": "the solar inverter tripped again check the photovoltaic panels on the rooftop and make sure the string combiner box is not overheating",
            "direction": "en2vi"
        },
        # Test 5: Từ vựng NGOÀI dictionary - Cơ khí ô tô / Văn nói bình dân (VI->EN)
        # "hộp số", "bạc biên", "két nước", "ống bô" KHÔNG có trong CSV
        {
            "text": "anh ơi cái xe tải nó bị hộp số trục trặc rồi mà két nước cũng rỉ nước ra nữa bạc biên kêu to lắm chắc phải thay rồi mà ống bô cũng bị thủng luôn",
            "direction": "vi2en"
        },
        # Test 6: Từ vựng NGOÀI dictionary - Y tế công trường (EN->VI)
        # "heatstroke", "tourniquets", "defibrillator" KHÔNG có trong CSV
        {
            "text": "one worker collapsed from heatstroke bring the first aid kit and check if we have tourniquets and a portable defibrillator in the emergency cabinet",
            "direction": "en2vi"
        },
        # Test 7: Câu rất dài, nhiều mệnh đề - Stress test ngữ pháp (EN->VI)
        {
            "text": "the project manager said that if the geotechnical report confirms the soil bearing capacity is sufficient we can proceed with the shallow foundation design instead of using deep piles which would save us approximately thirty percent of the budget",
            "direction": "en2vi"
        }
    ]

    for i, case in enumerate(test_cases, 1):
        test_text = case["text"]
        direction = case["direction"]
        
        print(f"\n[{i}/{len(test_cases)}] 📝 VĂN BẢN ĐẦU VÀO ({direction.upper()}):")
        print(f"'{test_text}'")

        print("⏳ Đang xử lý chấm câu và dịch thuật...")
        
        # Thực hiện dịch (quá trình này sẽ in ra log của Punc và MT)
        result = mt.translate(test_text, direction=direction)

        print(f"\n✅ KẾT QUẢ DỊCH CUỐI CÙNG (TEST {i}):")
        print(f"'{result}'")
        
        output_wav = f"test_{i}_output_{direction}.wav"
        print("🔊 Đang tổng hợp giọng nói (TTS)...")
        
        if direction == "vi2en":
            # Chiều VI->EN: dùng F5-TTS (đã hoạt động tốt)
            audio, sr = tts.synthesize_en(result)
            if audio is not None and len(audio) > 0 and sr > 0:
                sf.write(output_wav, audio, sr)
                print(f"💾 Đã lưu file âm thanh: {output_wav}")
            else:
                print("⚠ Không thể tổng hợp giọng nói EN.")
        else:
            # Chiều EN->VI: thử OmniVoice trước, nếu fail thì dùng gTTS
            audio, sr = tts.synthesize_vi(result)
            
            # Kiểm tra xem audio có thực sự chứa dữ liệu không (không phải toàn số 0)
            if audio is not None and np.max(np.abs(audio)) > 0.001:
                sf.write(output_wav, audio, sr)
                print(f"💾 Đã lưu file âm thanh (OmniVoice): {output_wav}")
            else:
                print("[TTS VI] ⚠ OmniVoice trả về silence, chuyển sang gTTS...")
                if _tts_vi_gtts_fallback(result, output_wav):
                    print(f"💾 Đã lưu file âm thanh (gTTS fallback): {output_wav}")
                else:
                    print("⚠ Không thể tổng hợp giọng nói VI.")
            
        print("-" * 60)

    print("\n" + "="*60)

if __name__ == "__main__":
    test_translation()

