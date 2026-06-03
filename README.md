# OneVoice Edge — Hệ Thống Phiên Dịch Giọng Nói Thời Gian Thực
> **Cuộc thi OneVoice AI Challenge 2026 — Team Impact**

Hệ thống dịch thuật Speech-to-Speech chạy **100% Offline**, được thiết kế đặc biệt cho môi trường công nghiệp (nhà máy, công trường). Dự án được tối ưu hóa để chạy trên chip **Qualcomm Snapdragon NPU** với độ trễ (latency) dưới **1 giây** và mức ngốn RAM dưới **200 MB**.

---

## Vấn Đề Thực Tế
Rào cản ngôn ngữ giữa chuyên gia nước ngoài và kỹ sư bản địa gây giảm năng suất và nguy cơ mất an toàn. Các ứng dụng như Google Translate không thể dùng được vì:
- Bắt buộc phải có Internet (Cloud-based).
- Chết hoàn toàn khi gặp tiếng ồn máy móc công trường.

## Giải pháp — Kiến Trúc 4 Trạm Cục Bộ (Edge AI)

### Luồng 1: VI → EN

```text
Microphone 
    │
    ▼
┌─────────────────────────────────┐
│  Trạm 0: Lọc Ồn (Denoise)       │  GIPFormer ONNX (INT8)
│  Khử tiếng máy cắt, gió, ồn...  │  ~10ms
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Trạm 1: Nhận Diện (ASR)        │  GIPFormer
│  Giọng nói → Văn bản Tiếng Việt │  Chuyên dụng cho tiếng ồn công nghiệp
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Trạm 2: Dịch Thuật (MT)        │  VietAI/envit5-translation
│  Văn bản VI → EN                │  1 model cho cả 2 chiều (~600MB)
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Trạm 3: Phát Âm (TTS)          │  F5-TTS (voice clone) / OmniVoice
│  Văn bản EN → Giọng nói         │  Bảo toàn giọng nói qua ngôn ngữ
└─────────────────────────────────┘
    │
    ▼
Speaker / Earphone 
```

### Luồng 2: EN → VI

```text
Microphone
    │
    ▼
┌─────────────────────────────────┐
│  Trạm 0: Lọc Ồn (Denoise)       │  GIPFormer ONNX (INT8)
│  Khử tiếng máy cắt, gió, ồn...  │  ~10ms
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Trạm 1: Nhận Diện (ASR)        │  SenseVoice Small
│  Giọng nói → Văn bản Tiếng Anh  │  + Trích xuất Cảm Xúc (Emotion)
└─────────────────────────────────┘
    │ metadata: [text, emotion]
    ▼
┌─────────────────────────────────┐
│  Trạm 2: Dịch Thuật (MT)        │  VietAI/envit5-translation
│  Văn bản EN → VI                │  Luân chuyển metadata cảm xúc
└─────────────────────────────────┘
    │ metadata: [translated_text, emotion]
    ▼
┌─────────────────────────────────┐
│  Trạm 3: Phát Âm (TTS)          │  OmniVoice (Voice Design)
│  Văn bản VI → Giọng nói         │  Đọc Tiếng Việt mô phỏng cảm xúc
└─────────────────────────────────┘
    │
    ▼
Speaker / Earphone 
```

**Mục tiêu Độ trễ tổng: < 600ms (Vượt chỉ tiêu 1s của giải)**

---

## Các Chế Độ Hoạt Động (Translation Directions)
Hệ thống là một đường ống hai chiều, cho phép bạn chuyển đổi linh hoạt.

| Hướng (Direction) | Đầu vào (Người nói) | Đầu ra (Loa phát) | Lệnh chạy (Flag) |
|-------------------|---------------------|-------------------|------------------|
| **VI → EN** (Mặc định) | Người Việt | Người Anh | `--direction vi2en` |
| **EN → VI** | Người Anh | Người Việt | `--direction en2vi` |

---

## Demo Kết Quả Dịch Thuật & Voice Cloning
Dưới đây là 7 kịch bản kiểm thử (Test Scenarios) đầy rẫy các thuật ngữ chuyên ngành hóc búa, từ lóng thi công và các tình huống thực tế tại công trường. Hệ thống đã dịch chuẩn xác và trích xuất thành file âm thanh thành công vào thư mục `demo_outputs/`.

1. **Test 1 (VI→EN)**
   - **Đầu vào**: Cậu đã làm gì với nó vậy thêm năng lượng hả nó hoạt động như thế nào vậy cho mình mượn chút đừng có keo kiệt vậy chứ hôm nay lớp mình có bài kiểm tra môn thể dục nên mình rất là cần nó luôn xài xong mình trả lại liền
   - **Bản dịch**: *What did you do with it? More power, huh? How it works. Well, let me borrow some. don't be mean, because we have a gym test today, so... I really need it. I'll give it back when I'm done.*
<details>
  <summary><h5>🔗 Nghe Audio</h5></summary>

[test_1_output_vi2en.webm](https://github.com/user-attachments/assets/274cef99-a640-4cc5-8ed6-2c7836ec417b)

</details>

2. **Test 2 (VI→EN)**
   - **Đầu vào**: Ê bạn ơi cái máy xúc số ba nó bị xì nhớt thủy lực rồi bơm bê tông cũng kẹt luôn qua kiểm tra lẹ giùm mình đi chứ để vậy là cháy van an toàn nha
   - **Bản dịch**: *Hey, buddy, that excavator number three, it's leaking hydraulic fluid. The pump's jammed, too. please check it immediately. the safety valve will blow out.*
<details>
  <summary><h5>🔗 Nghe Audio</h5></summary>

[test_2_output_vi2en.webm](https://github.com/user-attachments/assets/9bad0263-e075-4f59-a08b-67be54f38863)

</details>

3. **Test 3 (EN→VI)**
   - **Đầu vào**: the gantry crane at berth seven is malfunctioning we cannot unload the containers the draft survey shows the vessel is listing to port side
   - **Bản dịch**: *Cần cẩu ở cầu cảng số 7 bị trục trặc. Chúng ta không thể dỡ các container. Cuộc giám định mớn nước cho thấy con tàu đang nghiêng sang mạn trái.*
<details>
  <summary><h5>🔗 Nghe Audio</h5></summary>

[test_3_output_en2vi.webm](https://github.com/user-attachments/assets/abb1cbe9-16e4-49b8-abe6-37017fae85c3)

</details>

4. **Test 4 (EN→VI)**
   - **Đầu vào**: the solar inverter tripped again check the photovoltaic panels on the rooftop and make sure the string combiner box is not overheating
   - **Bản dịch**: *Bộ đảo lưu năng lượng mặt trời lại bị hỏng. Kiểm tra các tấm pin quang điện trên mái nhà và đảm bảo bộ tổng hợp dây không bị quá nóng.*
<details>
  <summary><h5>🔗 Nghe Audio</h5></summary>

[test_4_output_en2vi.webm](https://github.com/user-attachments/assets/0111b682-b7c0-4d84-83cf-17091a52361a)

</details>

5. **Test 5 (VI→EN)**
   - **Đầu vào**: anh ơi cái xe tải nó bị hộp số trục trặc rồi mà két nước cũng rỉ nước ra nữa bạc biên kêu to lắm chắc phải thay rồi mà ống bô cũng bị thủng luôn
   - **Bản dịch**: *Hey, man, the truck's got a malfunctioning gearbox, and the cooling system's leaking water. Connecting rod ball bearings knocked. It's gotta be replaced, and the exhaust pipe's leaking too.*
<details>
  <summary><h5>🔗 Nghe Audio</h5></summary>

[test_5_output_vi2en.webm](https://github.com/user-attachments/assets/0ac1d9d2-23a3-4715-b085-d7b28689a677)

</details>

6. **Test 6 (EN→VI)**
   - **Đầu vào**: one worker collapsed from heatstroke bring the first aid kit and check if we have tourniquets and a portable defibrillator in the emergency cabinet
   - **Bản dịch**: *Một công nhân bị ngã do say nắng. Mang theo bộ sơ cứu và kiểm tra xem có ga-rô và máy khử rung cầm tay không trong tủ cấp cứu.*
<details>
  <summary><h5>🔗 Nghe Audio</h5></summary>

[test_6_output_en2vi.webm](https://github.com/user-attachments/assets/b8d8ee00-a9bc-4739-8419-e1bb333d9cad)

</details>

7. **Test 7 (EN→VI)**
   - **Đầu vào**: the project manager said that if the geotechnical report confirms the soil bearing capacity is sufficient we can proceed with the shallow foundation design instead of using deep piles which would save us approximately thirty percent of the budget
   - **Bản dịch**: *Giám đốc dự án nói rằng nếu báo cáo địa kỹ thuật xác nhận sức chịu tải của đất là đủ, chúng tôi có thể tiến hành thiết kế móng nông thay vì sử dụng cọc sâu, mà sẽ tiết kiệm cho chúng ta khoảng 30% ngân sách.*
<details>
  <summary><h5>🔗 Nghe Audio</h5></summary>

[test_7_output_en2vi.webm](https://github.com/user-attachments/assets/f24d1c5d-ec8b-4aab-bc2f-8668b2f1eb46)

</details>

*(Lưu ý: Các file âm thanh trên đã được áp dụng công nghệ Voice Cloning. Tiếng Việt lấy cảm hứng từ giọng của nhân vật Nobita, còn tiếng Anh sử dụng giọng mẫu F5-TTS).*

---

## Hướng dẫn Cài Đặt (Self-Contained)

```bash
# 1. Tạo môi trường Conda
conda create -n onevoice python=3.11.8
conda activate onevoice

# 2. Cài đặt FFmpeg (Bắt buộc cho F5-TTS)
# - Trên Windows (dùng terminal admin): winget install ffmpeg
# - Trên Linux/Colab: sudo apt-get install ffmpeg

# 3. Cài đặt các thư viện 
pip install -r requirements.txt

# 4. Tải các file âm thanh giọng mẫu (Voice Presets)
python scripts/download_voice_preset.py
```

*(Lưu ý: Lần chạy đầu tiên, hệ thống sẽ tự động tải các file weights của GIPFormer và SenseVoice từ HuggingFace/ModelScope về cache cục bộ. Để chạy 100% Offline không cần Wifi, hãy đảm bảo bạn đã chạy pipeline ít nhất 1 lần khi có mạng).*

---

## Cách Chạy Dự Án

### 1. Dịch từ Người Việt sang Tiếng Anh (VI → EN)
Đây là chế độ mặc định. Hệ thống sẽ bật mic, nghe bạn nói Tiếng Việt, khử ồn bằng GIPFormer, dịch sang Tiếng Anh và đọc ra loa.

```bash
python src/pipeline.py --direction vi2en
```

### 2. Dịch từ Người Anh sang Tiếng Việt (EN → VI)
Hệ thống sẽ nghe tiếng Anh. Đặc biệt, **SenseVoice** sẽ tự động trích xuất cảm xúc (Ví dụ: Giận dữ, Vui vẻ). Thái độ này sẽ được truyền thẳng xuống **OmniVoice** để đọc Tiếng Việt với đúng tông giọng gắt gỏng hoặc vui nhộn của người gốc.

```bash
python src/pipeline.py --direction en2vi
```

---

## Giấy phép & Tri ân tác giả
Dự án tuân thủ Giấy phép **CC BY-NC 4.0**.
Chúng tôi đã tích hợp trực tiếp, trích xuất và tinh chỉnh mã nguồn từ các tác giả:
- **BetterBox-TTS & OmniVoice**: Dolly VN / ContextBoxAI (CC BY-NC 4.0)
- **GIPFormer**: G-Group AI Lab (MIT)
- **SenseVoice**: FunAudioLLM / Alibaba (MIT)
- **VietAI/envit5**: VietAI (MIT)

> **Cảm ơn Ban Tổ Chức OneVoice AI Challenge 2026!**
