import sys
import os
import yaml

# Thêm src vào đường dẫn để import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from translation.mt_engine import Translator

def test_translation():
    print("="*60)
    print("🚀 BẮT ĐẦU TEST NHANH TRẠM DẤU Câu (1.5) & DỊCH THUẬT (2)")
    print("="*60)

    # Load cấu hình
    cfg_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Khởi tạo Translator (Trạm 2)
    # Lưu ý: Hàm translate() của Translator sẽ tự động gọi restore_punctuation (Trạm 1.5)
    mt = Translator(cfg)
    mt.load()

    # Các câu ASR thô để test (không dấu câu)
    test_cases = [
        # Test 1: Giao tiếp đời thường, nhiều từ lóng (Colloquial/Slang stress test)
        "Cậu đã làm gì với nó vậy thêm năng lượng hả nó hoạt động như thế nào vậy cho mình mượn chút đừng có keo kiệt vậy chứ hôm nay lớp mình có bài kiểm tra môn thể dục nên mình rất là cần nó luôn xài xong mình trả lại liền",
        
        # Test 2: Môi trường công trường (Hỗn hợp: Văn nói bình dân + Thuật ngữ kỹ thuật)
        "Ê bạn ơi cái máy xúc số ba nó bị xì nhớt thủy lực rồi bơm bê tông cũng kẹt luôn qua kiểm tra lẹ giùm mình đi chứ để vậy là cháy van an toàn nha"
    ]

    for i, test_text in enumerate(test_cases, 1):
        print(f"\n[{i}/2] 📝 VĂN BẢN ĐẦU VÀO (ASR THÔ):")
        print(f"'{test_text}'")

        print("⏳ Đang xử lý chấm câu và dịch thuật...")
        
        # Thực hiện dịch (quá trình này sẽ in ra log của Punc và MT)
        result = mt.translate(test_text, direction="vi2en")

        print(f"\n✅ KẾT QUẢ DỊCH CUỐI CÙNG (TEST {i}):")
        print(f"'{result}'")
        print("-" * 60)

    print("\n" + "="*60)

if __name__ == "__main__":
    test_translation()
