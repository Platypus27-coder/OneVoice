# OneVoice Edge — Implementation Plan

## 0. Mục tiêu

Xây dựng **OneVoice Edge** thành một hệ thống **dịch giọng nói Việt ↔ Anh chuyên biệt cho công trường**, thay vì một bộ dịch speech-to-speech dùng chung.

Các ràng buộc chính từ đề bài:

- Offline 100%.
- Dịch hai chiều VI ↔ EN.
- End-to-End latency < 1 giây.
- Mục tiêu tối ưu: < 600 ms.
- Hướng tới Qualcomm Snapdragon NPU.
- RAM mục tiêu < 200 MB.
- Chịu được tiếng ồn công nghiệp.
- Hiểu đúng thuật ngữ kỹ thuật và tiếng lóng công trường.
- Có fallback để pipeline không crash.

---

# 1. Chiến lược tổng thể

Không train lại toàn bộ hệ thống ngay từ đầu.

Chiến lược:

```text
Pretrained Models
      +
Construction Domain Layer
      +
Construction Data Factory
      ↓
Benchmark trên dữ liệu công trường
      ↓
Phân tích lỗi từng module
      ↓
Chỉ fine-tune module thực sự cần
```

Nguyên tắc:

> Fine-tune là bước cuối sau khi benchmark chứng minh model pretrained không đủ tốt.

---

# 2. Kiến trúc mục tiêu

```text
Microphone
   ↓
[0] Industrial Denoise
   ↓
[1] ASR
   ↓
[1.5] Construction Context Engine
   ├── Domain Detection
   ├── Terminology Normalization
   ├── Intent Detection
   ├── Entity Extraction
   ├── Safety Detection
   └── Site-specific Context
   ↓
[2] Construction-aware MT
   ↓
[3] TTS
   ↓
Speaker / Earphone
```

## Safety Fast Path

Các câu nguy hiểm không nên đi qua toàn bộ pipeline thông thường.

```text
ASR
 ├── Normal → Context → MT → TTS
 │
 └── Emergency Intent
          ↓
    Safety Fast Path
          ↓
    Fixed / constrained translation
          ↓
          TTS
```

Ví dụ intent:

- STOP_WORK
- DANGER
- FIRE
- ELECTRIC_SHOCK
- FALLING_OBJECT
- PERSON_BELOW
- EVACUATE

---

# 3. Construction Context Engine

Đây là thành phần giúp OneVoice thực sự chuyên biệt cho công trường.

## 3.1 Construction Domains

Bản đầu nên giới hạn domain:

```text
Construction
├── Safety
├── Civil
├── Mechanical
├── Electrical
├── Welding
├── Heavy Equipment
└── Plumbing
```

Không mở rộng quá nhiều domain ở giai đoạn đầu.

## 3.2 Construction Intents

Danh sách intent ban đầu:

```text
REPORT_PROBLEM
REQUEST_ACTION
GIVE_INSTRUCTION
REQUEST_INSPECTION
WARN_DANGER
STOP_WORK
ASK_LOCATION
ASK_MEASUREMENT
ASK_STATUS
REQUEST_TOOL
REPORT_DAMAGE
REPORT_MATERIAL
CONFIRM
```

## 3.3 Canonical Concepts

Không map trực tiếp slang → English.

Nên normalize về một concept trung gian.

Ví dụ:

```text
"CB"
"aptomat"
"cầu dao tự động"
        ↓
electrical.circuit_breaker
        ↓
"circuit breaker"
```

Ví dụ:

```text
"bạc biên"
"ổ trục thanh truyền"
        ↓
mechanical.connecting_rod_bearing
        ↓
"connecting rod bearing"
```

---

# 4. Construction Data Factory

Do không có sẵn dataset Việt–Anh chuyên biệt cho công trường, tự xây dữ liệu theo 3 tầng.

```text
Construction Text
      +
Synthetic Speech
      +
Small Real-site Dataset
      ↓
Construction Corpus
```

---

# 5. Tầng 1 — Construction Text Corpus

Đây là ưu tiên số 1.

## 5.1 `construction_terms.csv`

Schema:

```text
canonical_id
domain
vi_standard
vi_colloquial
en_standard
aliases
priority
notes
```

Ví dụ:

```text
M001
mechanical
ổ trục thanh truyền
bạc biên
connecting rod bearing
bạc dên|bạc tay biên
normal
```

## 5.2 `construction_utterances.csv`

Schema:

```text
id
vi_raw
vi_normalized
en
domain
intent
entities
priority
source
```

Ví dụ:

```text
vi_raw:
"Cẩu lên thêm hai chục phân"

vi_normalized:
"Nâng tải lên thêm 20 cm"

en:
"Raise the load another 20 centimeters."

domain:
heavy_equipment

intent:
GIVE_INSTRUCTION

entities:
action=raise;distance=20cm
```

## 5.3 `safety_phrases.csv`

Tạo riêng dữ liệu safety.

Schema:

```text
vi
en
intent
severity
fixed_translation
```

Ví dụ:

```text
"Dừng lại! Có người bên dưới!"
"Stop! There is someone below!"
PERSON_BELOW
critical
true
```

---

# 6. Tầng 2 — Synthetic Construction Speech

Biến Construction Text Corpus thành audio.

```text
Construction sentence
       ↓
Multiple TTS voices
       ↓
Speed / pitch / volume augmentation
       ↓
Construction noise mixing
       ↓
Reverberation
       ↓
Synthetic Construction Speech
```

## 6.1 Biến thiên speaker

Cần tạo:

- Nam / nữ.
- Giọng Bắc / Trung / Nam nếu có thể.
- Tốc độ nói khác nhau.
- Nói nhỏ.
- Nói nhanh.
- Câu bị ngắt.
- Câu mệnh lệnh ngắn.
- Tiếng hét cảnh báo.

## 6.2 Noise classes

Ưu tiên noise thực tế:

```text
excavator
drilling
hammer
grinder
truck
diesel_engine
metal_impact
welding
wind
alarm
crowd
generator
compressor
```

## 6.3 SNR

Sinh nhiều mức:

```text
20 dB
15 dB
10 dB
5 dB
0 dB
-5 dB (stress test)
```

Không nhất thiết dùng -5 dB để train; có thể dùng làm stress test.

---

# 7. Tầng 3 — Real-site Dataset

Không cần thu hàng trăm giờ ngay.

Mục tiêu của real data ban đầu:

> Đo synthetic-to-real gap và tìm hard cases.

Thu theo ma trận:

```text
Speaker
× Domain
× Intent
× Noise level
× Distance from microphone
```

Các tình huống cần ưu tiên:

- Ra lệnh vận hành.
- Báo lỗi máy.
- Electrical.
- Mechanical.
- Crane / lifting.
- Safety warning.
- Measurement.
- Tool request.

Gắn nhãn tối thiểu:

```text
audio
transcript
translation
domain
intent
entities
noise_type
approx_snr
speaker_id
```

---

# 8. Chiến lược cho từng model

## 8.1 Denoise

### Phase A

Giữ pretrained model.

Đánh giá trên:

- clean speech;
- synthetic construction noise;
- real-site noise.

### Chỉ fine-tune nếu

- ASR giảm mạnh khi gặp noise thực;
- denoise tạo artifact làm ASR tệ hơn;
- synthetic → real gap lớn.

---

## 8.2 ASR

Đây là module có khả năng cần adaptation cao nhất.

### Phase A — Không fine-tune

Dùng:

```text
Pretrained ASR
+
Construction vocabulary
+
Contextual biasing
+
Post-ASR correction
```

Ví dụ:

```text
ASR: "bạc biển"
Domain: mechanical
Candidate: "bạc biên"
→ sửa thành "bạc biên"
```

### Phase B — Fine-tune khi benchmark cho thấy lỗi hệ thống

Ví dụ:

```text
CB → xi bi
bạc biên → bạc biển
cốp pha → cốt pha
cẩu → cậu
MEP → ...
```

Nếu các lỗi này xảy ra thường xuyên dù đã contextual biasing thì fine-tune.

Ưu tiên:

```text
Adapter / LoRA / partial fine-tuning
```

Không train ASR từ đầu.

---

# 9. Machine Translation

## Phase A

Không fine-tune.

Sử dụng:

```text
ASR Text
  ↓
Terminology Normalization
  ↓
Canonical Concepts
  ↓
MT
  ↓
Terminology Enforcement
```

Ví dụ:

```text
"aptomat"
      ↓
electrical.circuit_breaker
      ↓
"circuit breaker"
```

## Translation Memory

Các câu lặp lại nhiều trong công trường nên bypass MT nếu confidence cao.

Ví dụ:

```text
"Dừng máy."
→ "Stop the machine."

"Ngắt điện."
→ "Disconnect the power."

"Có người bên dưới."
→ "There is someone below."
```

Lợi ích:

- nhanh;
- deterministic;
- giảm lỗi;
- tốt cho safety.

## Fine-tune MT khi

- syntax chuyên ngành vẫn sai;
- phrase-level context sai thường xuyên;
- terminology enforcement không đủ;
- benchmark cho thấy fine-tune tạo cải thiện rõ.

---

# 10. TTS

Không ưu tiên fine-tune domain.

TTS không cần hiểu kiến thức công trường.

Ưu tiên:

- latency;
- intelligibility trong môi trường ồn;
- streaming;
- fallback;
- voice consistency.

Pipeline:

```text
Primary TTS
    ↓ fail
Fallback TTS
    ↓ fail
Pre-generated Safety Audio / Stub
```

Các câu safety quan trọng có thể pre-generate audio để đạt latency cực thấp.

---

# 11. Site Pack

OneVoice không nên hard-code tất cả kiến thức vào model.

Mỗi công trường có một package riêng.

```text
site_pack/
├── metadata.json
├── terminology.csv
├── equipment.csv
├── worker_slang.csv
├── safety_phrases.csv
├── locations.csv
├── company_terms.csv
└── phrase_memory.csv
```

Ví dụ:

```text
site_pack_metro/
site_pack_factory/
site_pack_powerplant/
```

Khi triển khai:

```text
OneVoice Core
      +
Site Pack
      ↓
Site-specific OneVoice
```

Không cần train lại model khi chuyển site nếu vocabulary/context layer đủ tốt.

---

# 12. Benchmark bắt buộc trước khi fine-tune

Tạo một fixed benchmark set.

## ASR

Theo dõi:

```text
WER
CER
Construction Term Recall
Critical Term Recall
Safety Phrase Recall
```

Đặc biệt thêm:

### Construction Term Error Rate

```text
CTER =
số thuật ngữ công trường nhận sai
/
tổng số thuật ngữ công trường
```

WER thấp nhưng sai từ quan trọng vẫn không chấp nhận được.

Ví dụ:

```text
"Dừng máy xúc"
→ "Dừng máy chút"
```

WER có thể không quá xấu nhưng semantic risk rất lớn.

---

# 13. MT Metrics

Không chỉ dùng BLEU.

Theo dõi:

```text
Term Accuracy
Intent Preservation
Entity Preservation
Measurement Accuracy
Safety Meaning Accuracy
```

Ví dụ:

```text
VI:
"Nâng lên 20 cm"

Bad:
"Raise it 20 meters."

```

Đây phải được đánh lỗi critical dù câu nghe tự nhiên.

---

# 14. Safety Metric

Tạo riêng:

```text
Safety Critical Accuracy
```

Các field phải bảo toàn:

```text
action
negation
number
unit
direction
equipment
person
danger_type
```

Ví dụ:

```text
"Không bật máy"
```

không bao giờ được dịch thành:

```text
"Turn on the machine"
```

---

# 15. Latency Budget

Mục tiêu E2E:

```text
< 600 ms
```

Gợi ý budget ban đầu:

```text
Denoise            10–30 ms
ASR                150–250 ms
Context Engine      5–20 ms
MT                  50–120 ms
TTS                150–250 ms
--------------------------------
Target             < 600 ms
```

Đây là budget để engineering; phải benchmark trên hardware thực.

Safety Fast Path cần thấp hơn normal path.

---

# 16. Memory Budget

Target:

```text
< 200 MB RAM
```

Ưu tiên:

- INT8 quantization;
- ONNX;
- shared tokenizer;
- mmap nếu phù hợp;
- lazy loading;
- small lookup tables;
- Trie/FST thay vì LLM cho terminology;
- pre-generated safety speech;
- tránh load đồng thời model không cần thiết.

---

# 17. Experiment Matrix

Không thay nhiều biến cùng lúc.

## Experiment A

```text
Pretrained ASR
```

## Experiment B

```text
ASR + Construction Dictionary
```

## Experiment C

```text
ASR + Dictionary + Contextual Biasing
```

## Experiment D

```text
Fine-tuned ASR
```

So sánh:

```text
WER
CTER
Safety Recall
Latency
RAM
```

Chỉ giữ fine-tune nếu có cải thiện rõ và không phá constraint edge.

---

# 18. Decision Gate — Có fine-tune hay không?

```text
                  Benchmark
                      │
                      ▼
            Model đạt yêu cầu?
              /             \
            YES              NO
             │                │
        Không train       Error analysis
                              │
                    Context/rule sửa được?
                         /          \
                       YES           NO
                        │             │
                  Không train     Fine-tune
```

Quy tắc:

> Không fine-tune chỉ vì có data.

> Fine-tune vì đã xác định được một failure mode rõ ràng mà domain layer không giải quyết đủ tốt.

---

# 19. Thứ tự triển khai

## Phase 1 — Baseline

- [ ] Chạy được pipeline Denoise → ASR → MT → TTS.
- [ ] Đo latency từng stage.
- [ ] Đo RAM.
- [ ] Tạo baseline evaluation.

## Phase 2 — Construction Ontology

- [ ] Chốt domain taxonomy.
- [ ] Chốt intent taxonomy.
- [ ] Xây canonical concept IDs.
- [ ] Xây `construction_terms.csv`.
- [ ] Xây `safety_phrases.csv`.
- [ ] Xây `construction_utterances.csv`.

## Phase 3 — Construction Context Engine

- [ ] Terminology matcher.
- [ ] Alias normalization.
- [ ] Domain classifier.
- [ ] Intent classifier.
- [ ] Entity extractor.
- [ ] Safety detector.
- [ ] Translation memory.
- [ ] Post-ASR correction.

## Phase 4 — Synthetic Data Factory

- [ ] TTS generation.
- [ ] Multi-speaker generation.
- [ ] Noise library.
- [ ] Noise mixing.
- [ ] RIR/reverb.
- [ ] SNR augmentation.
- [ ] Dataset manifest.

## Phase 5 — Benchmark

- [ ] Clean benchmark.
- [ ] Construction terminology benchmark.
- [ ] Industrial noise benchmark.
- [ ] Safety benchmark.
- [ ] Synthetic benchmark.
- [ ] Real-site pilot benchmark.

## Phase 6 — Selective Fine-tuning

- [ ] Error analysis.
- [ ] Quyết định có cần ASR adaptation không.
- [ ] Quyết định có cần MT adaptation không.
- [ ] Fine-tune đúng module cần thiết.
- [ ] Quantize lại.
- [ ] Benchmark lại toàn pipeline.

## Phase 7 — Edge Optimization

- [ ] ONNX export.
- [ ] INT8 quantization.
- [ ] Snapdragon NPU test.
- [ ] Streaming inference.
- [ ] Memory profiling.
- [ ] Latency profiling.
- [ ] CPU/GPU/NPU fallback.

## Phase 8 — Demo

Chuẩn bị ít nhất 4 demo scenario.

### Demo 1 — Mechanical

```text
"Máy này bị rỉ nhớt ở bạc biên."
```

### Demo 2 — Electrical

```text
"Ngắt CB trước khi kiểm tra."
```

### Demo 3 — Heavy Equipment

```text
"Cẩu lên thêm hai chục phân."
```

### Demo 4 — Safety

```text
"Dừng lại! Có người bên dưới!"
```

Demo phải có:

```text
clean
+
industrial noise
+
offline
+
real-time
```

---

# 20. Repository Structure

```text
onevoice-edge/
│
├── app/
│
├── models/
│   ├── denoise/
│   ├── asr/
│   ├── mt/
│   └── tts/
│
├── construction/
│   ├── ontology/
│   │   ├── domains.json
│   │   ├── intents.json
│   │   └── concepts.json
│   │
│   ├── terminology/
│   │   └── construction_terms.csv
│   │
│   ├── safety/
│   │   └── safety_phrases.csv
│   │
│   ├── context_engine/
│   └── translation_memory/
│
├── data/
│   ├── text/
│   ├── synthetic/
│   ├── noise/
│   ├── real/
│   └── benchmark/
│
├── site_packs/
│
├── scripts/
│   ├── generate_sentences.py
│   ├── generate_tts.py
│   ├── mix_noise.py
│   ├── build_manifest.py
│   └── evaluate.py
│
├── evaluation/
│   ├── asr/
│   ├── mt/
│   ├── safety/
│   └── latency/
│
└── plan.md
```

---

# 21. MVP Definition

MVP chưa cần train model.

MVP được coi là đạt khi:

- [ ] Offline hoàn toàn.
- [ ] VI ↔ EN chạy end-to-end.
- [ ] Construction terminology được normalize.
- [ ] Có ít nhất 4 domain.
- [ ] Có safety fast path.
- [ ] Có contextual biasing hoặc post-ASR correction.
- [ ] Có Site Pack.
- [ ] Có industrial-noise evaluation.
- [ ] Có benchmark terminology riêng.
- [ ] Có latency report.
- [ ] Có RAM report.
- [ ] Demo được các câu chuyên ngành mà translator general dễ sai.

---

# 22. Definition of Success

OneVoice không cần thắng hệ thống general-purpose ở mọi câu.

Nó cần thắng trong distribution mục tiêu:

```text
Vietnamese / English
+
Construction domain
+
Industrial noise
+
Short operational speech
+
Safety commands
+
Technical terminology
+
Offline edge hardware
```

Thông điệp sản phẩm cuối cùng:

> **OneVoice Edge is not a general translator running at a construction site.  
> It is an offline construction communication system designed specifically for the language, noise, safety and latency constraints of real worksites.**

---

# 23. Ưu tiên ngay bây giờ

Thứ tự công việc đề xuất:

```text
1. Construction taxonomy
        ↓
2. Construction terminology
        ↓
3. Construction utterance dataset
        ↓
4. Baseline benchmark
        ↓
5. Context Engine
        ↓
6. Synthetic audio
        ↓
7. Benchmark again
        ↓
8. Decide whether ASR/MT needs fine-tuning
        ↓
9. Real-site pilot
        ↓
10. Edge optimization
```

**Không bắt đầu bằng fine-tuning.**

Bắt đầu bằng:

> **Domain definition → Data → Benchmark → Error analysis → Selective adaptation.**
