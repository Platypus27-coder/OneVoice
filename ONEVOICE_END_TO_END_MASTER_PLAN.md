# OneVoice — Master Plan End-to-End

> **Trạng thái:** DRAFT FOR REVIEW — chưa tự động triển khai các hạng mục trong file này  
> **Ngày lập:** 22/08/2026  
> **Phạm vi:** từ source hiện tại đến OneVoice V2 phát hành được, offline hai chiều, được kiểm thử trên dữ liệu thật và có reference deployment trên edge device.  
> **Source of truth:** source code + report đo được + artifact có checksum. README/slide/demo không thay thế bằng chứng chạy thật.

---

## 1. Cách đọc và cập nhật kế hoạch

- [x] DONE: đã có source/artifact và kiểm tra tương ứng.
- [-] PARTIAL: đã làm một phần nhưng còn thiếu benchmark, artifact hoặc acceptance gate.
- [ ] TODO: chưa triển khai.
- [!] BLOCKED: phụ thuộc data, device, quyền truy cập hoặc external approval.

Quy tắc quản lý:

1. Mỗi task có mã cố định để dùng trong issue, commit và report.
2. Chỉ đổi thành DONE khi exit gate của task đã đạt.
3. Mỗi benchmark phải lưu run manifest, predictions, aggregate và environment.
4. Mỗi thay đổi model phải có baseline, regression set và rollback point.
5. Synthetic, hosted-device và real-site metrics luôn báo riêng.

---

## 2. North Star và Definition of Done

OneVoice V2 hoàn thành khi cung cấp speech-to-speech Vietnamese ↔ English cho giao tiếp công trường với:

- production runtime không cần Internet;
- streaming theo semantic commit, không phát duplicate prefix;
- terminology, negation, number, unit và direction được bảo toàn;
- critical phrase đi qua deterministic safety fast path và audio local;
- headset/earpiece là interaction mặc định;
- artifact bundle theo direction, có license, version và SHA-256;
- peak RSS edge profile dưới 200 MB;
- normal commit-to-first-audio p95 dưới 1.000 ms;
- safety commit-to-first-audio p95 dưới 300 ms;
- synthetic benchmark và fixed real-site holdout đều đạt gate;
- có Android/reference-device build, rollback, release bundle và tài liệu vận hành.

Không coi dự án hoàn thành nếu chỉ có notebook, demo video, dev-machine result hoặc hosted Qualcomm profile.

---

## 3. Trạng thái hiện tại đã xác minh

### 3.1 Repository và runtime

- [x] GOV-001: tag rollback v1-working-baseline tại commit ce35ac5.
- [x] GOV-002: có V1 source/environment baseline report.
- [x] CORE-001: giữ CLI python src/pipeline.py --direction ...
- [x] CORE-002: có profile development, edge và premium.
- [x] CORE-003: có site-pack, offline, input/output file và report-dir CLI.
- [x] CORE-004: có AudioFrame, ASRHypothesis, ContextResult, CommitDecision, TranslationResult và SynthesizedChunk contracts.
- [x] CORE-005: bounded queues, worker error propagation, queue drain và runtime report đã có.
- [-] CORE-006: rolling window/stable prefix/semantic commit đã có source và unit test; chưa có full real-model integration benchmark.
- [x] CORE-007: 27 unit tests hiện tại pass.
- [-] CORE-008: README có status matrix nhưng cần cập nhật sau model fine-tune và các gate mới.

### 3.2 Context, safety và site pack

- [x] CTX-001: text corpus 8.064 pairs, 12 domains, 11 intents.
- [x] CTX-002: 216 canonical concepts và 426 surface forms.
- [x] CTX-003: longest-match trie, aliases, entity rules và critical-field validator.
- [x] CTX-004: Site Pack schema/precedence và alias collision checks.
- [x] CTX-005: safety detector và two-confirmation commit.
- [-] SAFE-001: có 196 safety rows và 240 minimal pairs; human review status chưa khóa toàn bộ.
- [ ] SAFE-002: chưa có approved safety audio bundle cho cả hai direction.

### 3.3 Data và benchmark

- [x] DATA-001: train=6.369, dev=716, test=979; frame-pattern leakage đã biết bằng 0.
- [-] DATA-002: VI audio V1 trên Drive theo cấu trúc 8.064 clean + 16.128 noisy; manifest phục hồi được.
- [ ] DATA-003: chưa lưu evidence full physical audit 24.192 WAV trong official reports.
- [-] DATA-004: V1 không phục hồi được speaker thật, noise source, realized SNR, RIR và reverb.
- [ ] DATA-005: chưa có English construction audio đủ điều kiện.
- [ ] DATA-006: chưa có real-site corpus/fixed holdout.
- [-] EVAL-001: VI-ASR passthrough clean/noisy từng chạy trên Colab; cần gom report và khóa official baseline.
- [ ] EVAL-002: MT base/fine-tuned full test/minimal/safety chưa hoàn tất.
- [ ] EVAL-003: chưa có official latency/RAM report trên edge hardware.

### 3.4 Models

- [x] ASR-VI-001: GIPFormer INT8 ONNX names và revision đã pin.
- [-] ASR-VI-002: GIPFormer là VI-ASR baseline; chưa có approved PyTorch/Icefall adaptation/export cycle.
- [-] ASR-EN-001: SenseVoice adapter có source; chưa benchmark do thiếu EN audio.
- [x] MT-001: EnViT5 fine-tune 3 epochs; dev loss 0.0386 → 0.0177 → 0.0169.
- [x] MT-002: checkpoint upload tại platypus123/onevoice-envit5-vi-en.
- [-] MT-003: training loader dùng cả vi: và en: examples; candidate chỉ được promote cho VI→EN sau held-out gate. EN→VI hiện giữ baseline.
- [-] DENOISE-001: passthrough là baseline; RNNoise/DeepFilterNet chưa pass downstream gate.
- [-] TTS-001: premium/local paths và silence failure handling đã có; edge TTS backend portable chưa khóa.

### 3.5 Edge, hardware và release

- [x] EDGE-001: artifact preflight kiểm tra hash, license, sample rate và backend.
- [x] EDGE-002: có no-network/RSS profiling script.
- [ ] EDGE-003: chưa có production artifact manifest thực.
- [ ] EDGE-004: chưa export/validate MT edge bundle.
- [ ] EDGE-005: chưa có FP32/FP16/INT8 A/B report.
- [ ] EDGE-006: chưa có Qualcomm hosted-device evidence.
- [ ] HW-001: chưa có Android build, field-device validation, power hoặc thermal report.
- [ ] RELEASE-001: chưa có signed bundle, SBOM, license inventory và reproducible release.

### 3.6 Cleanup bắt buộc

- [ ] CLEAN-001: deprecate/remove Whisper submission notebook khỏi workflow chính; không dùng làm OneVoice ASR architecture.
- [ ] CLEAN-002: review notebook ASR legacy đang untracked; archive hoặc loại bỏ sau khi kiểm tra evidence.
- [ ] CLEAN-003: reconcile denoiser notebook local với bản tracked; bảo toàn thay đổi người dùng.
- [ ] CLEAN-004: chuyển hồ sơ Phase 2 vào docs/submission/phase2 hoặc ignore theo quyết định repository.
- [ ] CLEAN-005: tạo tag phase2-submission-2026-08-21.

---

## 4. Quyết định kiến trúc đã khóa

1. Runtime public vẫn là src/pipeline.py; notebook chỉ orchestrate Colab/Drive.
2. Edge runtime không download model, gọi API, dùng gTTS hoặc cloud fallback.
3. VI→EN dùng GIPFormer VI-ASR, Context/Safety, candidate EnViT5 đã fine-tune và local EN TTS/safety audio.
4. EN→VI dùng SenseVoice sau data gate, Context/Safety, EnViT5 baseline và local VI TTS/safety audio.
5. GIPFormer không phải denoiser; denoiser baseline là passthrough.
6. Safety phrase không qua generative MT/TTS khi đã có approved fixed translation/audio.
7. Headset/earpiece là production interaction. External speaker bị khóa tới khi AEC pass.
8. Premium model không nằm trong edge memory gate.
9. Qualcomm hosted result không thay field-device validation.
10. Nếu EnViT5 INT8 làm tổng RSS vượt 200 MB, edge dùng direction-specific distilled student; premium giữ EnViT5.

---

## 5. Data inventory và data cần bổ sung

| Dataset / Artifact | Hiện có | Cần bổ sung | Gate hoàn thành |
|---|---:|---:|---|
| Construction text V2 | 8.064 pairs | Human safety/translation review | 100% critical rows reviewed |
| VI audio V1 | 8.064 clean + 16.128 noisy | Full physical audit evidence | 24.192 WAV decode/pair pass |
| VI audio V2.1 | Chưa có | 8.064 clean + 16.128 noisy | ≥6 voice IDs; metadata/SNR/RIR pass |
| EN audio V2.1 | Chưa có | 8.064 clean + 16.128 noisy | ≥6 voices/accents; split pass |
| Noise bank V2.1 | V1 metadata yếu | Licensed categorized bank | ≥60 phút, ≥10 classes |
| RIR bank | Chưa chuẩn hóa | Site-like RIRs | ≥20 RIR có metadata |
| Safety audio | Chưa có bundle | 196 phrases × 2 directions | 392 WAV + checksum + approval |
| MT general regression | Chưa khóa | General VI↔EN ngoài construction | ≥1.000 non-overlap pairs |
| TTS evaluation | Chưa có | Terms/numbers/safety prompts | ≥200 prompts/language |
| Real-site pilot | Chưa có | Consent-based recordings | Target 1.000; allowed 500–2.000 |

### 5.1 Text review

- [ ] DATA-TEXT-001: human-review 196 safety phrases và 240 minimal pairs.
- [ ] DATA-TEXT-002: review toàn bộ rows risk_level=critical.
- [ ] DATA-TEXT-003: lưu reviewer, status, timestamp và issue category.
- [ ] DATA-TEXT-004: khóa corpus version 2.0-reviewed.
- [ ] DATA-TEXT-005: tạo general MT regression set.

### 5.2 VI V1 audit

- [ ] DATA-VI1-001: full audit 8.064 clean + 16.128 noisy.
- [ ] DATA-VI1-002: decode/sample rate/duration/silence/clipping checks.
- [ ] DATA-VI1-003: clean↔noisy pairing và split consistency.
- [ ] DATA-VI1-004: lưu audit, recovery report và file hash inventory.
- [ ] DATA-VI1-005: đóng băng V1; không sửa audio để cải thiện metric.

### 5.3 Sinh VI audio V2.1

- [ ] DATA-VI21-001: dùng toàn bộ 8.064 VI utterances, phân phối qua ≥6 voice identities thật.
- [ ] DATA-VI21-002: sinh 8.064 clean và 2 noisy variants/clean = 16.128 noisy.
- [ ] DATA-VI21-003: lưu speaker_id, engine/version, rate, pitch và seed.
- [ ] DATA-VI21-004: random noise crop; lưu noise source/hash/offset.
- [ ] DATA-VI21-005: target SNR 20/15/10/5/0 dB; -5 dB chỉ stress.
- [ ] DATA-VI21-006: tính/lưu realized SNR.
- [ ] DATA-VI21-007: dùng ≥20 RIR và lưu rir_id.
- [ ] DATA-VI21-008: split theo frame_pattern_id; leakage audit pass.

### 5.4 Sinh EN audio V2.1

- [ ] DATA-EN21-001: dùng 8.064 English translations đã review.
- [ ] DATA-EN21-002: sinh 8.064 clean + 16.128 noisy.
- [ ] DATA-EN21-003: ≥6 voices, nam/nữ, nhiều rate và ≥3 accent groups nếu license cho phép.
- [ ] DATA-EN21-004: cùng pair/noise/RIR/SNR contract với VI V2.1.
- [ ] DATA-EN21-005: audit speaker identity, transcript và leakage.

### 5.5 Real-site pilot

- [ ] DATA-REAL-001: consent form và retention policy.
- [ ] DATA-REAL-002: thu target 1.000 utterances, ưu tiên safety/lifting/electrical/mechanical/measurement.
- [ ] DATA-REAL-003: metadata theo data/real_site_pilot/schema.json.
- [ ] DATA-REAL-004: pseudonymous speaker IDs; không lưu PII thừa.
- [ ] DATA-REAL-005: group split theo site_id + session_id + speaker_id.
- [ ] DATA-REAL-006: tạo holdout_lock trước tuning.
- [ ] DATA-REAL-007: real test không dùng cho model/threshold/prompt selection.

---

## 6. Interface và artifact contract cần hoàn thiện

### 6.1 Config theo direction

Target:

~~~yaml
translation:
  vi2en:
    development_model: platypus123/onevoice-envit5-vi-en
    local_model_dir: models/mt/vi2en
    edge_model_dir: models/mt/vi2en_ort
  en2vi:
    development_model: VietAI/envit5-translation
    local_model_dir: models/mt/en2vi
    edge_model_dir: models/mt/en2vi_ort
~~~

- [ ] API-001: direction-specific ASR/MT/TTS registry.
- [ ] API-002: production không nhận remote model override.
- [ ] API-003: report ghi exact model revision/hash.
- [ ] API-004: startup fail rõ nếu direction/profile thiếu artifact.

### 6.2 Dataset manifest V2.1

Required fields:

~~~text
utterance_id, frame_pattern_id, pair_id, language, text, translation,
clean_audio, noisy_audio, split, speaker_id, voice_engine, voice_revision,
noise_source, noise_hash, noise_crop_offset, target_snr_db, realized_snr_db,
rir_id, sample_rate, duration_s, domain, intent, risk_level, generation_seed
~~~

- [ ] API-005: JSON Schema manifest V2.1.
- [ ] API-006: physical/metadata audit trước benchmark.

### 6.3 Standard report

~~~text
run_manifest.json
predictions.csv
aggregate.json
breakdown.json
hardware.json
dependencies.json
logs.txt
~~~

- [ ] API-007: thống nhất schema cho ASR/MT/TTS/streaming/edge.
- [ ] API-008: experiment index so baseline/candidate theo run ID.

### 6.4 Production bundle

~~~text
onevoice-bundle/
├── manifest.json
├── licenses/
├── vi2en/{asr,mt,tts,safety_audio}/
├── en2vi/{asr,mt,tts,safety_audio}/
└── site_packs/
~~~

- [ ] API-009: manifest có revision, hash, sample rate, profile, direction, license và backend.
- [ ] API-010: ký bundle; runtime từ chối artifact sai hash/signature.

---

## 7. Milestones triển khai

### M0 — Snapshot và cleanup

**Thời lượng:** 1–2 ngày  
**Dependency:** không có

- [ ] M0-01: review dirty worktree và bảo toàn file người dùng.
- [ ] M0-02: archive hồ sơ Phase 2 theo quyết định.
- [ ] M0-03: deprecate/remove Whisper notebook khỏi workflow.
- [ ] M0-04: archive/remove legacy ASR notebook sai.
- [ ] M0-05: reconcile denoiser notebook; validate notebook syntax.
- [ ] M0-06: tag phase2-submission-2026-08-21.
- [ ] M0-07: tạo issue board theo task IDs.

**Exit:** main sạch, workflow không còn notebook sai kiến trúc, snapshot rollback được.

### M1 — Reproducibility và data audit

**Thời lượng:** 1 tuần  
**Dependency:** M0

- [ ] M1-01: pin Python/packages/model revisions cho Colab/local.
- [ ] M1-02: full physical audit VI V1.
- [ ] M1-03: archive official VI-ASR reports từ Drive.
- [ ] M1-04: process_file smoke cả hai direction với model thật.
- [ ] M1-05: startup/failure/fallback truth table.
- [ ] M1-06: artifact/license inventory đầu tiên.

**Exit:** người khác clone + mount đúng Drive tái lập được baseline/report.

### M2 — Benchmark foundation

**Thời lượng:** 1 tuần  
**Dependency:** M1

- [ ] M2-01: MT notebook stream stderr/stdout và ghi job thất bại.
- [ ] M2-02: benchmark CLI nhận explicit base/candidate source/revision.
- [ ] M2-03: VI-ASR breakdown domain/risk/clean/noisy.
- [ ] M2-04: MT base raw/context trên test/minimal/safety, cả hai direction.
- [ ] M2-05: candidate fine-tuned raw/context cho VI→EN.
- [ ] M2-06: general regression benchmark.
- [ ] M2-07: streaming hypothesis và queue-pressure suite.
- [ ] M2-08: tự sinh Markdown dashboard từ aggregate JSON.

**Exit:** official baseline/candidate numbers; không metric giả, empty pass hoặc silent success.

### M3 — Promote VI→EN MT và Context/Safety

**Thời lượng:** 2 tuần  
**Dependency:** M2

- [ ] M3-01: so base, base+context, fine-tuned, fine-tuned+context.
- [ ] M3-02: candidate tăng ≥3 điểm phần trăm metric lỗi mục tiêu.
- [ ] M3-03: general regression ≤1 điểm phần trăm.
- [ ] M3-04: terminology accuracy ≥95%.
- [ ] M3-05: critical-field preservation ≥99%.
- [ ] M3-06: zero safety meaning reversal.
- [ ] M3-07: pass thì promote exact HF revision cho VI→EN; fail thì rollback base+context.
- [ ] M3-08: EN→VI giữ base đến khi independent held-out pass.
- [ ] M3-09: human-review critical translations và khóa revision.

**Exit:** VI→EN MT được chọn bằng evidence; config không dùng nhầm cho EN→VI.

### M4 — VI-ASR và denoiser

**Thời lượng:** 2–3 tuần  
**Dependency:** M2 và DATA-VI1

- [ ] M4-01: official passthrough clean/noisy baseline.
- [ ] M4-02: A/B RNNoise.
- [ ] M4-03: A/B DeepFilterNet quality ceiling.
- [ ] M4-04: chỉ pass denoiser nếu WER/CTER gain ≥5% relative, Critical Term Recall không giảm, clean WER regression ≤1 điểm.
- [ ] M4-05: ASR error analysis theo terms/numbers/units/safety/noise.
- [ ] M4-06: Context/post-ASR correction trước fine-tune.
- [ ] M4-07: chỉ fine-tune GIPFormer nếu Critical Term Recall <95%.
- [ ] M4-08: dùng real PyTorch checkpoint, Icefall recipe và train/dev/test đúng.
- [ ] M4-09: export ONNX; quality regression ≤1 điểm.
- [ ] M4-10: nếu recipe/checkpoint không tương thích, giữ pretrained và ghi blocker; không adapter giả.

**Exit:** frozen VI-ASR/denoiser artifacts và official report.

### M5 — Safety audio và TTS

**Thời lượng:** 2 tuần  
**Dependency:** M3

- [ ] M5-01: chọn approved VI/EN safety voice/engine.
- [ ] M5-02: sinh 392 safety WAV.
- [ ] M5-03: human-listen review critical audio.
- [ ] M5-04: manifest có source hash, engine revision, checksum và review.
- [ ] M5-05: edge fail rõ nếu safety audio thiếu/corrupt.
- [ ] M5-06: chọn local edge TTS cho normal phrases.
- [ ] M5-07: test silence/corrupt/intelligibility/pronunciation/latency.
- [ ] M5-08: premium F5/OmniVoice ngoài edge memory gate.

**Exit:** critical output deterministic; normal TTS có verified local fallback.

### M6 — End-to-end streaming và reliability

**Thời lượng:** 3 tuần  
**Dependency:** M3, M4, M5

- [ ] M6-01: integration 32 ms frames → VAD → ASR → commit → MT → TTS.
- [ ] M6-02: stable prefix trên real ASR hypothesis.
- [ ] M6-03: zero duplicate spoken prefix.
- [ ] M6-04: ordered output dưới queue pressure.
- [ ] M6-05: cancellation khi đổi direction/profile/site pack.
- [ ] M6-06: worker exception propagates, không deadlock/silent exit.
- [ ] M6-07: graceful shutdown và report flush.
- [ ] M6-08: đo speech-to-commit, commit-to-first-audio, complete-turn.
- [ ] M6-09: normal p95 <1.000 ms; safety p95 <300 ms.
- [ ] M6-10: headset demo cho ≥20 fixed scenarios.

**Exit:** two-hour soak, zero unsafe/duplicate commit.

### M7 — EN→VI hoàn chỉnh

**Thời lượng:** 3–4 tuần sau khi EN data sẵn sàng  
**Dependency:** DATA-EN21 và M2

- [ ] M7-01: sinh/audit EN audio V2.1.
- [ ] M7-02: SenseVoice clean/noisy baseline và emotion regression.
- [ ] M7-03: error analysis terms/accents/numbers/safety.
- [ ] M7-04: fine-tune chỉ khi baseline/context fail.
- [ ] M7-05: không dùng VI audio cho EN-ASR.
- [ ] M7-06: EN→VI base EnViT5 + context benchmark.
- [ ] M7-07: chỉ adapt EN→VI nếu terminology <95% hoặc critical fields <99%.
- [ ] M7-08: full EN speech → VI speech và safety audio.

**Exit:** EN→VI đạt cùng safety/reliability gates với VI→EN.

### M8 — Edge/offline/quantization

**Thời lượng:** 4 tuần  
**Dependency:** frozen models từ M4, M5, M7

- [ ] M8-01: per-direction frozen bundle và real manifest.
- [ ] M8-02: no-network harness; zero request startup/runtime.
- [ ] M8-03: direction-specific lazy loading.
- [ ] M8-04: FP32/FP16/INT8 A/B.
- [ ] M8-05: MT export sang ONNX Runtime GenAI hoặc approved backend.
- [ ] M8-06: nếu RSS >200 MB, distill VI→EN teacher sang ≤100M direction-specific student; ưu tiên Marian/OPUS architecture.
- [ ] M8-07: chỉ promote student nếu MT gates pass.
- [ ] M8-08: Qualcomm AI Hub compile/numerical/load/p50/p95/memory.
- [ ] M8-09: ghi rõ hosted-device status.
- [ ] M8-10: profile physical Snapdragon device.

**Exit:** zero-network, RSS <200 MB, regression ≤1 điểm, latency SLA pass.

### M9 — Android/reference device

**Thời lượng:** 4 tuần  
**Dependency:** M8

- [ ] M9-01: Kotlin shell cho direction/profile/site pack/start-stop/status.
- [ ] M9-02: native 16 kHz headset audio path.
- [ ] M9-03: ONNX/QNN integration và signed bundle installer.
- [ ] M9-04: source/translation/safety/error UI.
- [ ] M9-05: raw audio logging off by default.
- [ ] M9-06: offline update/rollback.
- [ ] M9-07: power/thermal/battery/long-run measurement.
- [ ] M9-08: external speaker chỉ mở sau AEC pass.

**Exit:** installable build chạy full offline trên device thật.

### M10 — Real-site validation

**Thời lượng:** 4–8 tuần  
**Dependency:** M6, M9, DATA-REAL

- [ ] M10-01: pilot protocol, consent và safety supervision.
- [ ] M10-02: thu/audit real-site corpus.
- [ ] M10-03: lock final holdout.
- [ ] M10-04: tách synthetic/real reports.
- [ ] M10-05: tối đa một adaptation cycle dùng real train/dev.
- [ ] M10-06: không dùng real test trước final.
- [ ] M10-07: field usability với headset.
- [ ] M10-08: review false/missed safety và recovery workflow.

**Exit:** fixed real holdout đạt safety gates; lúc này mới claim production robustness.

### M11 — Release engineering

**Thời lượng:** 2 tuần  
**Dependency:** M8–M10

- [ ] M11-01: CI unit/notebook/schema/offline/artifact.
- [ ] M11-02: reproducible model/data/report manifests.
- [ ] M11-03: SBOM, notices và license inventory.
- [ ] M11-04: signed release + rollback bundle.
- [ ] M11-05: operator/troubleshooting/incident guide.
- [ ] M11-06: local-only telemetry; không log audio/text mặc định.
- [ ] M11-07: release-candidate soak.
- [ ] M11-08: tag onevoice-v2.0.0 sau khi mọi gate pass.

---

## 8. Acceptance gates

### Data Gate

- Files tồn tại, decode được, đúng sample rate/duration.
- Pair/split pass; không frame/speaker leakage.
- Speaker/noise/RIR/SNR metadata thật.
- Safety text/audio human-reviewed.

### ASR Gate

- Corpus WER/CER/CTER đúng.
- Critical Term Recall ≥95%.
- Safety Phrase Recall ≥99% synthetic fixed suite.
- Clean WER candidate regression ≤1 điểm.
- Breakdown domain/risk/noise/SNR/speaker.

### Denoiser Gate

- Noisy WER hoặc CTER gain ≥5% relative.
- Critical Term Recall không giảm.
- Clean WER regression ≤1 điểm.
- Latency/memory vẫn đạt.

### MT Gate

- Terminology accuracy ≥95%.
- Critical-field preservation ≥99%.
- Zero safety meaning reversal.
- Candidate gain ≥3 điểm phần trăm.
- General regression ≤1 điểm.

### Streaming/Safety Gate

- Zero unsafe commit.
- Zero duplicated spoken prefix.
- Correct order dưới queue pressure.
- Safety cần 2 confirmations hoặc endpoint.
- Missing/corrupt safety audio fail rõ.

### TTS Gate

- Silence/corrupt output không được pass.
- 100% safety audio approved.
- Numbers/units/equipment intelligible.
- Safety p95 <300 ms; normal commit p95 <1.000 ms.

### Edge Gate

- Zero production network.
- Peak RSS <200 MB.
- Quantization regression ≤1 điểm.
- License/hash/backend/sample rate pass.
- Hosted và physical results tách rõ.

### Real-site Gate

- Consent/schema/group split/holdout pass.
- Synthetic và real reports tách riêng.
- Fixed real holdout đạt safety gates.

---

## 9. Test matrix

### Unit

- Unicode/normalization.
- Longest-match/alias collision/site precedence.
- Negation/number/unit/direction/condition.
- Stable prefix/overlap/commit dedup.
- Safety confirmation.
- Artifact hash/license/sample rate.
- Real holdout isolation.

### Integration

- input-file VI→EN và EN→VI.
- Safety có/không có approved audio.
- Missing/corrupt/wrong-hash model.
- Empty ASR, invalid MT, silent TTS.
- Queue full, cancellation, worker exception, shutdown.
- Direction-specific loading.

### System

- Two-hour microphone/headset soak.
- Network-block.
- Clean/noisy suites.
- Quantized versus reference.
- Android install/start/translate/rollback.
- Power/thermal/battery.

### Human

- Safety translation/audio review.
- TTS intelligibility.
- Real-site recovery workflow.
- False alarm/missed warning review.

---

## 10. Dependency graph

~~~text
M0 Cleanup/Snapshot
        ↓
M1 Reproducibility + VI Audit
        ↓
M2 Benchmark Foundation
   ┌────┼───────────────┐
   ↓    ↓               ↓
 M3 MT/Context       M4 VI-ASR/Denoise       EN/VI Data V2.1
   ↓    ↓               ↓
   └── M5 Safety/TTS ───┘
             ↓
        M6 Streaming E2E
             ↓
        M7 EN→VI Complete
             ↓
        M8 Edge/Qualcomm
             ↓
        M9 Android Device
             ↓
       M10 Real-site Pilot
             ↓
        M11 V2 Release
~~~

Data V2.1 và real-site preparation có thể chạy song song; model promotion luôn chờ đúng data gate.

---

## 11. Timeline dự kiến

| Giai đoạn | Tuần | Kết quả |
|---|---:|---|
| M0–M1 | 1–2 | Repo sạch, baseline/data audit tái lập |
| M2–M3 | 3–5 | Official benchmark, VI→EN MT promotion |
| M4–M5 | 4–8 | VI-ASR/denoiser và safety/TTS bundle |
| M6 | 8–10 | Streaming reliability pass |
| EN/VI data + M7 | 3–12 song song | EN→VI hoàn chỉnh |
| M8 | 11–14 | Edge bundle, quantization, Qualcomm |
| M9 | 15–18 | Android/reference device |
| M10 | 15–24 | Real-site validation |
| M11 | 24–26 | Release candidate và V2.0.0 |

Không rút ngắn real-site gate bằng synthetic claims.

---

## 12. Phân công đề xuất

| Owner | Trách nhiệm |
|---|---|
| Ngô Gia Huy | Architecture, ASR/MT, Context/Safety, data/evaluation, release evidence |
| Trần Tấn Khải | Hardware, Android/device integration, power/thermal, deployment, field validation |
| Team Impact | Safety review coordination, demo, documentation, acceptance và release |

---

## 13. Risks và contingency

| Rủi ro | Trigger | Hành động |
|---|---|---|
| EnViT5 vượt RAM | RSS ≥200 MB | Distill direction-specific student; không nới gate |
| GIPFormer fine-tune không tái lập | Thiếu recipe/checkpoint | Giữ pretrained + context; không evidence giả |
| Denoiser làm ASR xấu | Gate fail | Giữ passthrough |
| SenseVoice fail EN gate | Sau baseline + một adaptation cycle | Benchmark alternative EN-only ONNX ASR; review trước switch |
| TTS edge không portable | Offline/intelligibility fail | Local system TTS normal + fixed safety audio |
| External speaker echo | Duplicate/false ASR | Headset/earpiece only |
| Không đủ real data | <500 compliant rows | Chỉ research/field-pilot release |
| Hosted Qualcomm pass, device fail | Physical p95/RSS/thermal fail | Không ship; optimize/distill module lớn nhất |

---

## 14. Final release checklist

- [ ] Main sạch, tag/rollback đủ.
- [ ] Data có version/schema/audit/license.
- [ ] VI→EN và EN→VI có held-out evidence.
- [ ] Safety translations/audio approved.
- [ ] Streaming/safety/reliability pass.
- [ ] Edge bundle signed, no-network, RSS <200 MB.
- [ ] Hosted và physical-device validation xong.
- [ ] Android/reference build offline.
- [ ] Real-site fixed holdout pass.
- [ ] SBOM/licenses/operator/incident docs xong.
- [ ] README chỉ claim VERIFIED.
- [ ] Tag onevoice-v2.0.0 và publish artifacts.

---

## 15. Việc bắt đầu ngay sau khi plan được duyệt

1. M0: snapshot và dọn notebook/workflow sai kiến trúc.
2. M1: full physical audit VI V1 và gom official Colab reports.
3. M2: sửa MT benchmark để chọn base/candidate rõ và stream log.
4. Chạy held-out MT matrix để quyết định promote HF checkpoint.
5. Song song chuẩn bị VI/EN V2.1 generator contract và safety review.

Không bắt đầu GIPFormer/SenseVoice fine-tune, quantization hoặc Qualcomm compile trước khi các gate phụ thuộc hoàn tất.

