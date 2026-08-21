# OneVoice Edge — V1 Baseline → V2 Construction Upgrade Plan

> **Vai trò của file này:** roadmap nâng cấp OneVoice từ V1 đã chạy được lên V2 chuyên dụng cho công trường.
>
> **Nguyên tắc trung tâm:** Không viết lại V1 chỉ để khớp README cũ.  
> V1 được xem là **working baseline** cho tới khi test thực tế chứng minh ngược lại.
>
> **Development rule:** Measure → Change one thing → A/B test → Keep or Rollback.

---

# 1. Source of Truth

Từ bây giờ dùng thứ tự sau:

```text
1. Đề bài / Product Requirements
      ↓
   Hệ thống PHẢI đạt gì?

2. V1 Working Source + Reproduced Test
      ↓
   Hệ thống HIỆN ĐANG làm gì?

3. Architecture decisions đã chốt cho V2
      ↓
   Hệ thống SẼ làm như thế nào?

4. plan.md
      ↓
   Roadmap triển khai V1 → V2

5. README.md
      ↓
   Tài liệu mô tả implementation hiện tại
```

Không làm:

```text
README cũ
   ↓
ép source chạy theo README
```

Mà làm:

```text
Working Source
   ↓
test / verify
   ↓
README mô tả đúng current behavior
   ↓
plan.md chứa future upgrades
```

---

# 2. Vai trò tài liệu

## `de_bai_OneVoice.md`

Là **Product Requirement**:

- Offline Edge AI.
- VI ↔ EN.
- Speech-to-speech.
- Chịu tiếng ồn công trường/công nghiệp.
- Technical terminology.
- Low latency.
- Qualcomm / RAM constraints.
- Reliability / fallback.

## `README.md`

Là **Current Implementation Documentation**.

Chỉ ghi những thứ source hiện tại thật sự hỗ trợ:

- model nào đang dùng;
- pipeline hiện tại;
- cách cài/chạy;
- fallback;
- limitation.

Feature chưa implement phải ghi là `Planned` hoặc chuyển sang `plan.md`.

## `ONEVOICE_V1_TO_V2_PLAN.md`

Là **future roadmap**.

Các phần như:

- Construction Context Engine;
- Semantic Commit;
- Safety Fast Path;
- Site Pack;
- selective fine-tuning;
- real denoiser;
- Edge optimization;

nằm ở đây cho tới khi implement + test xong.

---

# 3. Chiến lược phát triển

Không:

```text
V1
 ↓
refactor toàn bộ
 ↓
thay ASR + MT + TTS + streaming
 ↓
fine-tune tất cả
 ↓
hy vọng tốt hơn
```

Mà:

```text
                 Working V1
                     │
                     ▼
               Baseline Test
                     │
                     ▼
                Upgrade #1
                     │
                  A/B Test
                  /      \
              Better     Worse
                │          │
               KEEP     ROLLBACK
                │
                ▼
                V1.1
```

Mỗi upgrade phải có:

- lý do;
- baseline;
- metric;
- expected gain;
- regression risk;
- rollback point.

---

# 4. V1 được giữ làm baseline

Không sửa runtime V1 chỉ vì:

- comment cũ;
- README cũ;
- tên module chưa đẹp;
- code có technical debt;
- architecture mới nhìn “đúng hơn”.

Chỉ sửa ngay khi:

```text
A. Có lỗi reproducible;
B. Có blocker reliability;
C. Benchmark đang tính sai;
D. Thay đổi là prerequisite tối thiểu để test đúng.
```

---

# 5. Kiến trúc V2 mục tiêu

V2 vẫn là **simultaneous translation**.

Không chuyển sang:

```text
Người nói nói xong hoàn toàn
→ mới bắt đầu dịch
```

Mục tiêu:

> **Translate as early as safely possible.**

## VI → EN

```text
Microphone
    ↓
Industrial Speech Enhancement
    ↓
VI ASR
    ↓
Rolling Source Context
    ↓
Construction Context Engine
    ├── terminology
    ├── domain
    ├── intent
    ├── entity
    ├── negation
    ├── number/unit
    └── safety
    ↓
Semantic Commit Controller
       ├── WAIT
       └── COMMIT
              ↓
Construction-aware MT
              ↓
Safety Fast Path / Normal TTS
              ↓
English Speech
```

## EN → VI

```text
Microphone
    ↓
Industrial Speech Enhancement
    ↓
EN ASR + Emotion
    ↓
Rolling Source Context
    ↓
Construction Context Engine
    ↓
Semantic Commit Controller
    ↓
MT EN→VI
    ↓
Vietnamese TTS + emotion
```

Đây là **target**, không phải claim rằng V1 đã có toàn bộ.

---

# PHASE 0 — Freeze & Reproduce V1

## 0.1 Freeze source

Tạo Git tag/branch:

```text
v1-working-baseline
backup/v1-working-baseline
```

Mục tiêu:

- rollback được;
- agent/Codex không phá baseline;
- có V1 để A/B với mọi upgrade.

## 0.2 Reproduce lần test cũ

Chạy lại:

```text
VI → EN
EN → VI
```

Xác nhận:

- microphone;
- ASR;
- MT;
- TTS;
- direction switching;
- fallback;
- output end-to-end.

## 0.3 Ghi trạng thái V1

Tạo:

```text
docs/V1_BASELINE_STATUS.md
```

Ghi:

- environment;
- command;
- model paths;
- models load được;
- VI→EN result;
- EN→VI result;
- latency quan sát;
- known limitations.

## 0.4 Fixed smoke-test set

Tạo:

```text
tests/baseline_audio/
```

Bao gồm một số mẫu:

- VI clean;
- EN clean;
- terminology;
- negation;
- number/unit;
- câu dài.

Mục tiêu: regression test, không phải benchmark cuối.

### Exit condition

```text
V1 reproducible
```

---

# PHASE 1 — Đồng nhất README với V1

Chỉ làm sau khi reproduce V1.

## 1.1 Đối chiếu

```text
README
vs
config
vs
source
vs
runtime logs
vs
actual test
```

## 1.2 Gắn trạng thái component

```text
IMPLEMENTED
PARTIAL
FALLBACK
PLANNED
```

## 1.3 Future feature không nằm trong Current Architecture

Nếu runtime chưa có:

```text
Semantic Commit
Safety Fast Path
Site Pack
```

thì README không viết như đã có.

### Exit condition

```text
README == verified V1 current truth
```

---

# PHASE 2 — Benchmark foundation

Chưa fine-tune.

Mục tiêu: biết V1 yếu ở đâu.

## 2.1 VI ASR benchmark

Dùng synthetic VI construction audio hiện có.

Đo:

- WER;
- CER;
- Construction Term Recall;
- Critical Term Recall;
- Number Accuracy;
- Unit Accuracy.

Theo:

```text
Clean
SNR 20
SNR 15
SNR 10
SNR 5
SNR 0
noise type
domain
risk
```

Dùng đúng:

```text
split == test
```

## 2.2 MT benchmark

Dùng:

```text
test.csv
minimal_pairs.csv
safety_fast_path.csv
```

Đo:

- terminology accuracy;
- negation preservation;
- intent preservation;
- number preservation;
- unit preservation;
- direction preservation;
- safety meaning accuracy.

## 2.3 Latency baseline

Không chỉ cộng model inference time.

Cần timestamp:

```text
speech/audio
→ ASR
→ commit/translation decision
→ MT
→ TTS ready
→ first translated audio
```

Report:

```text
p50
p95
max
```

## 2.4 RAM baseline

Đo:

```text
startup
after ASR
after MT
after TTS
peak runtime
```

### Exit condition

```text
Có baseline V1 đáng tin
```

---

# PHASE 3 — Fix benchmark/tooling, không đổi UX

Đây là nhóm thay đổi an toàn.

Ví dụ:

- prediction rỗng không được tính WER = 0;
- benchmark import phải dùng current modules;
- benchmark dùng test split;
- latency timestamps chính xác;
- audit manifest;
- unit tests cho normalizer/terminology.

Mục tiêu:

```text
runtime behavior V1 giữ nguyên
+
measurement reliability tăng
```

---

# PHASE 4 — Construction Context Engine (V1.1)

Đây là upgrade domain đầu tiên.

Chưa thay ASR/MT nếu chưa cần.

```text
ASR text
   ↓
Construction Context
   ↓
MT
```

## 4.1 Runtime ontology

Load:

```text
terminology_master.csv
term_aliases.csv
domains.json
```

Build lookup:

- hash map;
- trie;
- canonical IDs.

## 4.2 Canonical concept

Ví dụ:

```text
CB
aptomat
cầu dao tự động
       ↓
electrical.circuit_breaker
       ↓
circuit breaker
```

## 4.3 Intent/entity prototype

Ưu tiên lightweight:

```text
rule
+
small classifier nếu cần
```

Entities:

```text
action
equipment
number
unit
direction
location
condition
```

## Regression Gate

So sánh:

```text
V1
vs
V1 + Context Engine
```

Chỉ KEEP nếu:

- terminology tốt hơn;
- không phá general meaning;
- latency overhead nhỏ;
- RAM overhead nhỏ.

---

# PHASE 5 — Streaming Semantic Context (V1.2)

Mục tiêu giải vấn đề:

> “Hệ thống phát translation khi người nói chưa đưa đủ context.”

Nhưng vẫn phải realtime.

## 5.1 Rolling Context

Audio vẫn có thể đến theo frames/chunks.

Nhưng text context phải liên tục:

```text
chunk1
   ↓
context(chunk1 + chunk2)
   ↓
context(chunk2 + chunk3)
```

Không coi mỗi 1-second chunk là câu độc lập.

## 5.2 Stable / Unstable hypotheses

Nếu ASR hỗ trợ partial:

```text
stable prefix
unstable tail
```

Nếu chưa true streaming:

```text
overlapping window
+
rolling text alignment
```

làm prototype trước khi thay model.

## 5.3 Semantic Commit Controller

Không commit vội khi gặp:

```text
không
đừng
chưa
nếu
nhưng
trước khi
sau khi
không quá
ít nhất
nhiều nhất
```

Chờ pair:

```text
number + unit
negation + action
direction + amount
condition clause
```

Commit nhanh cho:

```text
Dừng lại!
Có cháy!
Tránh ra!
Dừng cẩu!
```

## 5.4 Simultaneous TTS

Mục tiêu:

```text
Speaker đang nói chunk tiếp
   │
   ├─ ASR tiếp tục
   ├─ MT chuẩn bị
   └─ TTS phát committed chunk trước
```

Không chờ full turn.

## Regression Gate

Đo:

- Premature Commit Rate;
- Unsafe Commit Rate;
- Average Controlled Lag;
- Time to First Stable Translation;
- translation quality;
- user-perceived latency.

---

# PHASE 6 — Safety Fast Path (V1.3)

## 6.1 Load

```text
safety_fast_path.csv
minimal_pairs.csv
```

## 6.2 Detector

MVP:

```text
phrase trie
+
intent rules / small classifier
```

Không cần LLM lớn.

## 6.3 Emergency path

```text
stable safety intent
      ↓
fixed/constrained translation
      ↓
pre-generated local audio
```

Critical cases có thể bỏ full MT/TTS generation.

Mục tiêu:

- nhanh;
- deterministic;
- không đảo nghĩa;
- offline.

---

# PHASE 7 — Selective Fine-tuning Decision

Không fine-tune vì “đã có data”.

Flow:

```text
Pretrained model
      ↓
benchmark
      ↓
error analysis
      ↓
context/rule fix
      ↓
vẫn fail systematic?
     /        \
   NO          YES
   │            │
  KEEP       FINE-TUNE
```

## 7.1 VI ASR / GIPFormer

Current VI noisy dataset phù hợp nhất cho VI ASR.

Fine-tune nếu baseline lỗi systematic về:

- construction terms;
- slang;
- acronym;
- noisy speech;
- safety commands.

Không train from scratch.

## 7.2 MT / EnViT5

8,064 VI↔EN pairs dùng được cả hai chiều.

Chỉ adapt nếu:

```text
Context Engine + terminology control
```

vẫn chưa đủ.

Ưu tiên adaptation nhẹ, tránh catastrophic forgetting.

## 7.3 EN ASR / SenseVoice

Không fine-tune bằng VI audio.

Trước:

```text
create EN Construction Speech Dataset
→ benchmark SenseVoice
→ fine-tune nếu cần
```

Nếu adapt SenseVoice phải regression-test emotion.

## 7.4 TTS

Không fine-tune bằng noisy synthetic construction audio hiện tại.

TTS adaptation cần:

```text
real clean target-speaker recordings
+
accurate transcript
```

---

# PHASE 8 — Denoise Upgrade

Không chọn denoiser theo README.

Đi từ requirement:

```text
industrial noise robustness
```

## 8.1 Baseline

```text
Noisy
→ ASR
```

## 8.2 Candidate

```text
Noisy
→ Real Denoiser
→ ASR
```

Đánh giá bằng downstream:

- WER;
- CTER;
- Safety Recall.

Không chỉ nghe waveform “sạch hơn”.

## 8.3 Fine-tune denoiser

Chỉ khi pretrained denoiser chưa đủ và clean↔noisy pairs hợp lệ.

---

# PHASE 9 — EN Construction Audio Dataset

Current audio dataset thiên về VI.

Tạo:

```text
3k–8k English construction utterances
```

Target diversity:

- ≥ 6 distinct speakers;
- male/female;
- different speaking rates;
- accent diversity nếu có thể;
- construction noise;
- SNR variation;
- clean/noisy pairs.

Sau đó mới benchmark EN-ASR đúng nghĩa.

---

# PHASE 10 — Site Pack

Implement khi Context Engine ổn.

```text
SitePackLoader
```

Load:

```text
project terminology
equipment IDs
zone names
company abbreviations
worker slang
safety overrides
translation memory
```

Mục tiêu:

> đổi site/project mà không phải retrain toàn bộ model.

---

# PHASE 11 — Edge / Offline Optimization

Chỉ tối ưu mạnh sau khi architecture ổn.

## 11.1 Production Offline Mode

Runtime production:

```text
NO runtime download
NO cloud fallback
NO gTTS
NO remote model fetch
```

Startup phải preflight local models.

## 11.2 Direction-specific loading

VI→EN chỉ load component cần cho VI→EN.

EN→VI cũng vậy.

## 11.3 Quantization

A/B:

```text
FP32
FP16
INT8
```

Theo:

- accuracy;
- latency;
- RAM;
- model size.

## 11.4 Qualcomm

Chỉ export model thật sự còn dùng trong V2.

Không tiếp tục legacy export nếu runtime đã đổi.

---

# PHASE 12 — Synthetic Data v2.1

Không regenerate toàn bộ ngay.

Sửa Data Factory trước:

- actual `speaker_id`;
- random noise crop offset;
- thêm VI speakers;
- thêm EN speakers;
- thêm RIR/acoustic diversity;
- giữ clean↔noisy pairs;
- target vs realized SNR;
- real-noise vs synthetic-fallback metadata.

---

# PHASE 13 — Real-site Validation

Synthetic data dùng cho development, không phải final proof.

Target pilot:

```text
500–2,000 real utterances
```

Nếu điều kiện cho phép.

Dùng để đo:

```text
Synthetic → Real Gap
```

Final real-site test set không dùng để fine-tune.

---

# 14. Milestones

## Milestone A — V1 Reproduced

- [ ] Git tag V1.
- [ ] VI→EN chạy.
- [ ] EN→VI chạy.
- [ ] Environment lưu lại.
- [ ] Baseline samples lưu lại.

## Milestone B — README Reconciled

- [ ] README khớp verified source.
- [ ] Future feature chuyển sang plan.

## Milestone C — Baseline Metrics

- [ ] VI ASR.
- [ ] MT.
- [ ] latency p50/p95.
- [ ] RAM.
- [ ] benchmark code đáng tin.

## Milestone D — V1.1 Construction Context

- [ ] terminology.
- [ ] canonical concepts.
- [ ] intent/entity.
- [ ] A/B test pass.

## Milestone E — V1.2 Streaming Semantics

- [ ] rolling context.
- [ ] stable/unstable hypothesis.
- [ ] semantic commit.
- [ ] simultaneous TTS scheduler.
- [ ] premature commit benchmark pass.

## Milestone F — V1.3 Safety

- [ ] detector.
- [ ] fast path.
- [ ] local critical audio.

## Milestone G — Model Adaptation

- [ ] GIPFormer decision.
- [ ] EnViT5 decision.
- [ ] EN dataset.
- [ ] SenseVoice decision.
- [ ] denoiser decision.

## Milestone H — V2 Edge Candidate

- [ ] offline preflight.
- [ ] direction-specific load.
- [ ] quantization.
- [ ] Qualcomm profiling.
- [ ] p95 latency.
- [ ] peak RAM.

## Milestone I — Real-site Validation

- [ ] real pilot.
- [ ] fixed final test.
- [ ] synthetic-real gap.

---

# 15. Upgrade Decision Template

Mỗi major change phải ghi:

```text
Change:
Why:
V1 Baseline:
Expected Gain:
Metric:
Regression Risk:
Rollback Point:
Measured Result:
Decision: KEEP / ROLLBACK
```

---

# 16. Những thứ không làm

- Không refactor toàn bộ V1 vì audit.
- Không ép source theo README cũ.
- Không fine-tune mọi model cùng lúc.
- Không thay ASR + MT + TTS + streaming trong một commit lớn.
- Không gọi synthetic audio là real-site audio.
- Không tối ưu latency bằng cách phát translation chưa ổn định.
- Không bỏ V1 baseline trước khi V2 chứng minh tốt hơn.

Target phải là:

> **Minimum safe latency**

không phải:

> **Minimum latency at all costs**

---

# 17. Definition of Done — OneVoice V2

- [ ] V1 baseline reproducible.
- [ ] README khớp current implementation.
- [ ] VI→EN simultaneous.
- [ ] EN→VI simultaneous.
- [ ] Premature safety-critical translation được kiểm soát.
- [ ] Construction terminology được xử lý có hệ thống.
- [ ] Safety Fast Path active.
- [ ] ASR có WER/CER/CTER benchmark.
- [ ] MT có critical-field benchmark.
- [ ] EN ASR có English construction test set.
- [ ] Latency có p50/p95 user-perceived.
- [ ] Peak RAM được đo.
- [ ] Runtime production không cần Internet.
- [ ] Edge optimization không gây regression lớn.
- [ ] Synthetic vs real-site evaluation được tách rõ.
- [ ] Có rollback point cho major upgrades.

---

# 18. Việc cần làm ngay

```text
1. Freeze source hiện tại thành V1 working baseline
       ↓
2. Reproduce lần test cũ
       ↓
3. Đồng nhất README với V1
       ↓
4. Fix benchmark/tooling, không đổi behavior
       ↓
5. Ghi baseline ASR / MT / latency / RAM
       ↓
6. Add Construction Context Engine
       ↓
7. Add Rolling Context + Semantic Commit
       ↓
8. Add Safety Fast Path
       ↓
9. Benchmark pretrained models trên data mới
       ↓
10. Fine-tune chỉ module thật sự cần
       ↓
11. Denoise / Edge / quantization optimization
       ↓
12. Real-site validation
```

---

# 19. Kết luận

OneVoice hiện được xem là:

```text
WORKING V1
   +
Construction Dataset V2
   +
Synthetic Noisy Speech Dataset
   +
Known UX limitations
   ↓
Controlled incremental upgrades
   ↓
ONEVOICE V2
```

Nguyên tắc xuyên suốt:

> **Preserve the working baseline. Measure before changing. Upgrade one variable at a time. Keep only proven improvements.**
