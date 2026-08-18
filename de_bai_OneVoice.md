# Đề Bài & Mục Tiêu Dự Án: OneVoice Edge (OneVoice AI Challenge 2026)

## 📌 1. Bối Cảnh & Vấn Đề Thực Tế
Trong các nhà máy, công trường xây dựng và môi trường công nghiệp hiện đại, rào cản ngôn ngữ giữa các chuyên gia/kỹ sư nước ngoài (nói tiếng Anh) và kỹ sư/công nhân bản địa (nói tiếng Việt) gây ra nhiều ảnh hưởng nghiêm trọng đến tiến độ công việc và an toàn lao động.

Các giải pháp truyền thống như Google Translate hay Microsoft Translator gặp các hạn chế lớn:
- **Phụ thuộc vào Internet:** Không thể hoạt động ở vùng sâu vùng xa, tầng hầm công trình hoặc khu vực cách ly mạng nội bộ.
- **Kém bền vũng với tiếng ồn:** Thất bại hoàn toàn khi có tiếng ồn máy móc, tiếng máy cắt, gió, công trường.
- **Độ trễ cao:** Không đáp ứng được nhu cầu giao tiếp thời gian thực (real-time).
- **Thiếu thuật ngữ chuyên ngành:** Thường dịch sai các từ lóng công trường, từ ngữ kỹ thuật cơ khí/xây dựng/điện.

---

## 🎯 2. Mục Tiêu Đề Bài & Yêu Cầu Kỹ Thuật
Dự án **OneVoice Edge** được thiết kế nhằm giải quyết triệt để các vấn đề trên với các chỉ tiêu kỹ thuật khắt khe:

1. **Chạy Offline 100% (Edge AI):** Không phụ thuộc vào kết nối Cloud hay Internet.
2. **Đường Ống Dịch Hai Chiều (Bi-directional Speech-to-Speech):**
   - **VI → EN:** Người Việt nói tiếng Việt ➔ Hệ thống dịch & đọc ra tiếng Anh (Voice Clone/Native voice).
   - **EN → VI:** Người nước ngoài nói tiếng Anh ➔ Hệ thống dịch & đọc ra tiếng Việt (mô phỏng cảm xúc/Nobita style).
3. **Độ Trễ Siêu Thấp (Low Latency):** Tổng độ trễ toàn tuyến (End-to-End Latency) **< 1 giây** (Mục tiêu tối ưu: **< 600ms**).
4. **Tối Ưu Phần Cứng (Hardware Constrained):** Chạy mượt mà trên thiết bị Edge/Chip **Qualcomm Snapdragon NPU**, dung lượng RAM chiếm dụng **< 200 MB** (hoặc chạy mượt trên CPU/GPU phổ thông với cơ chế fallback linh hoạt).
5. **Chuẩn Hóa Thuật Ngữ Kỹ Thuật & Tiếng Lóng:** Xử lý chính xác từ lóng công trường (bạc biên, két nước, ống bô, rỉ nhớt,...) sang thuật ngữ kỹ thuật chuẩn mực bằng từ điển chuẩn hóa (`colloquial_terms.csv`).

---

## 🏗️ 3. Kiến Trúc 4 Trạm Cục Bộ (4-Stage Pipeline)

```text
[Microphone] 
     │
     ▼
┌─────────────────────────────────┐
│  Trạm 0: Lọc Ồn (Denoise)       │  GIPFormer ONNX (INT8) — Khử tiếng ồn công nghiệp (~10ms)
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│  Trạm 1: Nhận Diện (ASR)        │  SenseVoice Small / GIPFormer — Chuyển giọng nói ➔ Văn bản + Cảm xúc
└─────────────────────────────────┘
     │ [Text + Emotion Metadata]
     ▼
┌─────────────────────────────────┐
│  Trạm 2: Dịch Thuật (MT)        │  VietAI/envit5-translation — Dịch thuật 2 chiều + Chuẩn hóa thuật ngữ
└─────────────────────────────────┘
     │ [Translated Text + Emotion]
     ▼
┌─────────────────────────────────┐
│  Trạm 3: Phát Âm (TTS)          │  F5-TTS (Voice Clone EN) / OmniVoice (VI) + pyttsx3 Fallback
└─────────────────────────────────┘
     │
     ▼
[Speaker / Earphone]
```

---

## 🛡️ 4. Cơ Chế Fallback & Độ Tin Cậy (Reliability & Robustness)
- **Cơ chế Fallback TTS:** `F5-TTS` (Primary GPU/Colab) ➔ `pyttsx3` (Windows Local / CPU) ➔ `Silence/Stub` đảm bảo chương trình không bao giờ bị crash giữa chừng.
- **Xử lý ngắt câu & Dấu câu:** Sử dụng `deepmultilingualpunctuation` kết hợp với bộ phân tách từ nối tiếng Việt (`_VI_SPLIT_MARKERS`) để tránh hiện tượng ảo giác (hallucination) khi dịch các câu dài không dấu.
