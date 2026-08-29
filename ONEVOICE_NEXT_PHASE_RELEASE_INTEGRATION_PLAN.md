# OneVoice — Kế hoạch chặng tiếp theo: Release Integration & Edge Hardening

> **Trạng thái:** APPROVED / IN PROGRESS — P0 đã triển khai trong source; chờ materialize release lock trên Drive
> **Ngày lập:** 29/08/2026  
> **Mục tiêu:** biến các model/artifact đã được chọn thành một bản OneVoice V2 demo/release-candidate chạy end-to-end, offline và tái lập được trên Colab/Google Drive; không fine-tune lại GIPFormer trong chặng này.  
> **Nguồn sự thật:** source code, release benchmark, manifest SHA-256 và report chạy thật. README/demo không thay thế acceptance gate.

---

## 1. Quyết định đã khóa

1. Dừng toàn bộ thử nghiệm fine-tune GIPFormer trong chặng này.
2. VI→EN dùng **official GIPFormer ONNX baseline** tại runtime; các candidate `head_ft_v1`, `icefall_ft_v1`–`v4` không được load, export hoặc xuất hiện trong release benchmark.
3. EN→VI dùng **SenseVoice fine-tuned ONNX FP32**; INT8 đã fail và không được promote.
4. MT dùng hai model đã fine-tune:
   - VI→EN: `platypus123/onevoice-envit5-vi-en` + validator/context v4.
   - EN→VI: `platypus123/onevoice-envit5-en-vi` + validator/context v2.
5. Denoiser giữ `passthrough`; không chặn chặng tích hợp vì RNNoise/DeepFilterNet chưa qua downstream gate.
6. Safety phrase đi qua deterministic fast path và WAV local có checksum; không qua generative MT/TTS khi match thành công.
7. User chỉ chạy Colab; notebook phải clone/pull GitHub, lưu checkpoint/progress/report trên Drive và resume khi đổi tài khoản/runtime.
8. Không yêu cầu chạy lại data generation, fine-tune hoặc benchmark đã hoàn thành nếu model, data và evaluator không thay đổi.
9. Chặng này có thể kết thúc ở mức **demo/release-candidate**. Không được gọi `production-ready` cho tới khi GIPFormer, real-site, edge memory/latency và physical-device gates đều đạt.

---

## 2. Snapshot đã hoàn thành — không chạy lại

| Hạng mục | Trạng thái | Evidence/Quyết định |
|---|---|---|
| English audio V2.1 | DONE | 8.064 clean, 16.128 noisy, ≥6 voices, split audit pass |
| VI audio V1 | DONE cho synthetic baseline | 8.064 clean, 16.128 noisy; giữ nguyên dataset |
| VI→EN MT fine-tune | DONE | Hugging Face `platypus123/onevoice-envit5-vi-en` |
| EN→VI MT fine-tune | DONE | Hugging Face `platypus123/onevoice-envit5-en-vi` |
| VI→EN MT test/context | PASS | Critical 99,28%; terminology 99,41% |
| EN→VI MT test/context | PASS | Critical 98,98%; terminology 98,97% |
| SenseVoice fine-tune | DONE | Checkpoint đã publish; ONNX FP32 được chọn |
| SenseVoice ONNX FP32 clean/noisy | PASS quality | WER 0,37% / 0,54%; critical 99,52% / 99,17% |
| SenseVoice INT8 | REJECTED | Không dùng runtime |
| GIPFormer official baseline | SELECTED WITH DEBT | Chạy được; release test critical 85,76% clean / 76,29% noisy, chưa đạt 95% |
| GIPFormer fine-tune candidates | REJECTED | WER xấp xỉ 100%, critical recall 0%; đã rollback |
| Safety audio demo | DONE cho internal demo | 252 WAV local, manifest/checksum, approval `impact-safety-v1` |
| Offline safety E2E | PASS | Cả VI→EN và EN→VI đã route tới WAV local được xác minh |
| Runtime artifact preflight | PASS demo | 292 artifact VI→EN và 284 artifact EN→VI đã verify trong lần chạy đã lưu |
| Release benchmark | DONE | `summary.json`, `report.md`, `report.html`; profile `release` gồm 10 artifact |

### Không chạy lại các notebook sau trong chặng này

- English synthetic generation/audit, trừ khi manifest hoặc WAV bị thay đổi.
- MT fine-tune VI→EN và EN→VI.
- SenseVoice fine-tune/export/held-out benchmark.
- GIPFormer fine-tune notebook.
- Benchmark release hiện có, trừ khi code/model/runtime artifact thay đổi.

---

## 3. Mục tiêu và Definition of Done của chặng

Chặng Release Integration & Edge Hardening hoàn thành khi có:

- một release lock ghi chính xác model revision/hash cho hai direction;
- file-mode E2E cho normal và safety path, cả VI→EN lẫn EN→VI;
- streaming integration với model thật, zero duplicate prefix và output đúng thứ tự;
- safety bundle được đối chiếu đầy đủ với safety source hiện hành;
- normal TTS offline không trả silence giả;
- per-direction artifact bundle, license và SHA-256 manifest;
- zero network trong offline runtime;
- report latency/RAM theo từng stage và toàn pipeline;
- Qualcomm hosted-device report nếu có token/quota;
- release benchmark chỉ hiển thị artifact đang dùng;
- demo runbook một lần chạy từ GitHub + Drive, không cần local.

### Không nằm trong Definition of Done của chặng

- sửa/fine-tune lại GIPFormer;
- tuyên bố production robustness;
- external-speaker full duplex khi chưa có AEC;
- real-site final holdout;
- physical Snapdragon validation nếu chưa có thiết bị.

---

## 4. Đường găng triển khai

```text
P0 Release lock/config
→ P1 E2E file-mode hai chiều
→ P2 Streaming & reliability
→ P3 Offline TTS + safety reconciliation
→ P4 Per-direction offline bundle
→ P5 Latency/RAM optimization
→ P6 Qualcomm/Android reference
→ P7 Release-candidate report & demo

Deferred after project completion:
GIPFormer adaptation + real-site holdout + physical device + AEC
```

---

## 5. Milestone P0 — Khóa release model, config và provenance

**Ưu tiên:** cao nhất  
**GPU:** không cần  
**Thời lượng dự kiến:** 1–2 ngày

- [x] P0-01: đổi registry từ tên `candidate` sang `release` cho hai EnViT5 đã pass.
- [x] P0-02: config mặc định theo direction trỏ đúng local/HF artifact đã fine-tune; không còn vô tình dùng `VietAI/envit5-translation` khi release artifact tồn tại.
- [x] P0-03: GIPFormer runtime chỉ trỏ `models/gipformer`; từ chối đường dẫn chứa candidate fine-tune bị reject.
- [x] P0-04: SenseVoice runtime chỉ trỏ ONNX FP32 release bundle; INT8 không được auto-select.
- [ ] P0-05: sinh `artifacts/release_lock_v2.json` gồm source, revision, SHA-256, license, sample rate, direction và profile. **Code/notebook DONE; chờ chạy notebook trên Drive để materialize file thật.**
- [ ] P0-06: manifest ghi rõ safety source CSV/review revision và checksum. **Code/notebook DONE; chờ materialize cùng P0-05.**
- [x] P0-07: test startup fail rõ khi thiếu/corrupt/sai hash model.
- [x] P0-08: test production/offline config không chứa runtime download hoặc remote fallback.

**Exit gate:** một config release duy nhất, artifact provenance đầy đủ và startup không thể chọn nhầm model cũ.

---

## 6. Milestone P1 — E2E file-mode normal và safety

**Dependency:** P0  
**GPU:** Colab T4 hữu ích cho smoke; acceptance phải báo đúng hardware/backend  
**Thời lượng dự kiến:** 2–3 ngày

- [x] P1-01: tạo `scripts/run_release_e2e.py` dùng public CLI/runtime thật, không gọi model riêng lẻ.
- [x] P1-02: fixed suite tối thiểu 20 normal + 20 safety cases/direction.
- [x] P1-03: input VI→EN lấy từ VI corpus; input EN→VI lấy từ EN V2.1; không dùng output WAV làm reference sai direction.
- [x] P1-04: xác minh route `normal_mt_tts` và `safety_audio` tách biệt.
- [x] P1-05: output WAV decode được, non-silent, sample rate đúng và commit ID khớp report.
- [x] P1-06: validator giữ negation, number, unit, direction và equipment ID.
- [x] P1-07: safety output phải khớp checksum WAV trong manifest.
- [x] P1-08: lỗi worker/model/TTS phải làm process exit khác 0; không có silent success.
- [x] P1-09: report gồm ASR text, canonical/context text, route, translation, engine/artifact, timestamps và failure reason.

**Trạng thái P1:** runner và acceptance checks đã triển khai; exit gate vẫn chờ chạy suite thật trên Drive. Normal route có thể fail cho tới khi P3 khóa được TTS offline.

**Checkpoint/resume:** mỗi case có result JSON riêng trên Drive; rerun bỏ qua case đã có đủ WAV + checksum + report.  
**Exit gate:** cả hai direction chạy normal/safety end-to-end bằng artifact release hiện tại.

---

## 7. Milestone P2 — Streaming semantics và reliability với model thật

**Dependency:** P1  
**GPU:** không bắt buộc; model benchmark phải ghi CPU/GPU backend  
**Thời lượng dự kiến:** 4–6 ngày

- [ ] P2-01: feed WAV thành frame 32 ms qua `StreamingSession`; không dùng transcript giả.
- [ ] P2-02: rolling ASR hypothesis cập nhật stable/unstable prefix đúng timestamps.
- [ ] P2-03: WAIT khi negation/condition/number-unit/direction còn chưa ổn định.
- [ ] P2-04: safety chỉ commit sau hai match liên tiếp hoặc endpoint.
- [ ] P2-05: zero duplicate spoken prefix trên fixed suite.
- [ ] P2-06: ordered output dưới bounded-queue pressure.
- [ ] P2-07: cancellation khi đổi direction/profile/site pack.
- [ ] P2-08: worker exception propagate; graceful shutdown luôn flush report.
- [ ] P2-09: chạy 30 phút soak tự động trước; chỉ chạy 2 giờ sau khi short soak pass.
- [ ] P2-10: đo riêng speech-to-commit, commit-to-first-audio và complete-turn.

**Checkpoint/resume:** prediction/turn result append atomically; giữ `completed_turn_ids` trên Drive.  
**Exit gate:** zero unsafe commit, zero duplicate prefix, zero reorder/deadlock trên suite cố định.

---

## 8. Milestone P3 — Safety reconciliation và normal TTS offline

**Dependency:** P1  
**GPU:** không bắt buộc  
**Thời lượng dự kiến:** 3–5 ngày

- [ ] P3-01: đối chiếu số lượng safety benchmark (196 rows) với approved safety source (126 canonical rows/252 WAV); tạo mapping và giải thích duplicate/variant.
- [ ] P3-02: fail release nếu safety ID đang dùng không có đủ VI/EN WAV, review status và checksum.
- [ ] P3-03: giữ gTTS bundle ở nhãn `development/internal-demo`; không ghi là production voice.
- [ ] P3-04: chọn normal offline TTS riêng cho EN output và VI output; runtime không gọi gTTS.
- [ ] P3-05: benchmark tối thiểu 200 prompts/language gồm term, number, unit, acronym và tên thiết bị.
- [ ] P3-06: phát hiện silence, clipping, corrupt WAV và sample-rate mismatch.
- [ ] P3-07: đo synthesis-to-first-sample p50/p95 và peak RSS.
- [ ] P3-08: premium F5/OmniVoice nằm ngoài edge memory gate.

**Exit gate:** safety deterministic; normal path có TTS offline thật, không silence stub và không runtime network.

---

## 9. Milestone P4 — Per-direction offline release bundle

**Dependency:** P0, P1, P3  
**GPU:** không cần  
**Thời lượng dự kiến:** 2–4 ngày

Target:

```text
onevoice-v2-rc/
├── manifest.json
├── release_lock_v2.json
├── licenses/
├── vi2en/
│   ├── asr/gipformer/
│   ├── mt/envit5/
│   ├── tts/
│   └── safety_audio/
├── en2vi/
│   ├── asr/sensevoice_fp32/
│   ├── mt/envit5/
│   ├── tts/
│   └── safety_audio/
└── site_packs/
```

- [ ] P4-01: build bundle theo direction; không load model của direction còn lại.
- [ ] P4-02: SHA-256 mọi file; manifest atomically finalized.
- [ ] P4-03: license/notices cho GIPFormer, SenseVoice, EnViT5, TTS và safety audio.
- [ ] P4-04: no-network harness chặn socket/HF/ModelScope và chạy startup + E2E.
- [ ] P4-05: corrupt/missing file test cho mỗi component.
- [ ] P4-06: tạo bundle receipt và exact Git commit.
- [ ] P4-07: Drive artifact có resume; không copy lại file đúng size/hash.

**Exit gate:** clone source + mount bundle là chạy được offline, không phụ thuộc cache ngầm.

---

## 10. Milestone P5 — Latency, memory và edge optimization

**Dependency:** P4  
**GPU:** chỉ dùng khi backend cần; edge acceptance là CPU/NPU/backend mục tiêu  
**Thời lượng dự kiến:** 5–8 ngày

### Baseline release cần xử lý

| Component | Current p95 | Nhận định |
|---|---:|---|
| VI ASR clean/noisy | 524,8 / 426,0 ms | Chạy được; quality là debt riêng |
| EN ASR ONNX FP32 clean/noisy | 1.581,9 / 1.591,3 ms | Vượt normal latency budget |
| VI→EN MT test/context | 2.392,3 ms | Bottleneck lớn nhất của VI→EN |
| EN→VI MT test/context | 382,3 ms | Trong budget stage-level hiện tại |
| Safety local audio lookup | Đã chạy E2E | Phải đo lại commit→first-sample riêng |

- [ ] P5-01: profile load time, per-stage p50/p95, complete-turn và peak RSS theo direction.
- [ ] P5-02: lazy-load direction và loại premium TTS khỏi edge profile.
- [ ] P5-03: export EnViT5 sang approved local edge backend; A/B quality trước/sau export.
- [ ] P5-04: thử FP32/FP16; chỉ thử INT8 khi numerical/runtime contract đúng.
- [ ] P5-05: quantization regression không quá 1 điểm phần trăm trên release suites.
- [ ] P5-06: giảm VI→EN MT p95 trước khi tối ưu module nhỏ hơn.
- [ ] P5-07: tối ưu SenseVoice FP32 inference/load mà không quay lại INT8 lỗi.
- [ ] P5-08: đo toàn edge profile; nếu peak RSS >200 MB thì gate fail, không sửa README thành pass.
- [ ] P5-09: safety commit→first-audio p95 <300 ms.
- [ ] P5-10: normal commit→first-audio p95 <1.000 ms hoặc ghi rõ blocker/budget overrun.

**Exit gate:** có evidence zero-network, memory và latency; pass/fail trung thực theo từng direction.

---

## 11. Milestone P6 — Qualcomm hosted và Android reference

**Dependency:** P5  
**GPU:** không phải Colab GPU; cần Qualcomm AI Hub token/quota cho hosted profiling  
**Thời lượng dự kiến:** 5–10 ngày

- [ ] P6-01: compile/profile các ONNX artifact còn được promote trên Qualcomm AI Hub.
- [ ] P6-02: numerical equivalence với local reference.
- [ ] P6-03: ghi model load, p50/p95 và peak memory trên hosted device.
- [ ] P6-04: ghi rõ `hosted-device`, không gọi là field validation.
- [ ] P6-05: Android/Kotlin shell tối thiểu: direction, start/stop, status, site pack.
- [ ] P6-06: 16 kHz mono headset input và ordered audio output.
- [ ] P6-07: bundle installer + checksum + rollback.
- [ ] P6-08: raw audio/text logging tắt mặc định.
- [ ] P6-09: nếu chưa có điện thoại thật, dừng ở reference build và ghi blocker.

**Exit gate:** hosted profile hoặc reference build tái lập được; không giả định physical-device result.

---

## 12. Milestone P7 — Release candidate, tài liệu và demo

**Dependency:** P1–P6 theo phạm vi khả dụng  
**GPU:** không cần cho bước tổng hợp  
**Thời lượng dự kiến:** 2–3 ngày

- [ ] P7-01: tạo focused release benchmark từ đúng release lock.
- [ ] P7-02: bảng pass/partial/blocked theo quality, E2E, offline, latency, RAM, hosted và real-site.
- [ ] P7-03: README dùng đúng model fine-tuned/release; không còn tên model base ở runtime diagram nếu không được dùng.
- [ ] P7-04: Colab runbook một đường: clone → mount → preflight → smoke → report.
- [ ] P7-05: demo 4 case normal + 4 safety mỗi direction, kèm report JSON/WAV.
- [ ] P7-06: troubleshooting cho missing artifact, corrupt hash, no TTS, OOM và network block.
- [ ] P7-07: notebook active được rút gọn; notebook training đã hoàn thành chuyển sang `archive/training_evidence` hoặc đánh dấu không cần chạy lại.
- [ ] P7-08: tag `onevoice-v2.0.0-rc1` chỉ khi release-candidate gates pass.

**Exit gate:** người khác mở đúng một notebook Colab và tái lập demo/release report từ artifact đã khóa.

---

## 13. Notebook policy cho chặng mới

### Notebook active

1. `colab_release_e2e_v2.ipynb` — notebook mới, chạy E2E fixed suite và resume từng case.
2. `colab_edge_profile_v2.ipynb` — offline/RSS/latency/Qualcomm orchestration.
3. `colab_runtime_artifacts_v2.ipynb` — build/verify per-direction bundle.
4. `colab_benchmark_report_v2.ipynb` — chỉ tổng hợp release report, không benchmark lại.

### Notebook evidence/archive

- Data generation/audit, MT fine-tune, SenseVoice fine-tune/export/evaluate: giữ làm evidence, không nằm trong `Run next`.
- GIPFormer fine-tune: đánh dấu `EXPERIMENTAL/REJECTED`, không nằm trong release workflow.
- Notebook trùng chức năng hoặc chỉ chứa cell chữa lỗi tạm thời phải hợp nhất/xóa sau khi evidence cần thiết đã được giữ.

### Quy tắc resume/checkpoint

- Mọi output dài ghi vào `MyDrive/OneVoice/reports/<phase>/<run_id>/`.
- Mỗi case hoàn thành phải có atomic JSON + checksum; notebook chỉ skip khi đủ artifact bắt buộc.
- Không dùng riêng sự tồn tại của `aggregate.json` để skip nếu predictions/run manifest thiếu.
- Runtime mất GPU/đổi tài khoản được tiếp tục từ `completed_case_ids`; không chạy lại case đã xác minh.
- Cell setup luôn `git pull --ff-only origin main`; notebook không chứa implementation logic lớn.

---

## 14. Acceptance gates của chặng

| Gate | Điều kiện pass |
|---|---|
| Release selection | Runtime model/hash khớp `release_lock_v2.json`; không load rejected candidate |
| E2E correctness | Normal/safety hai direction sinh WAV hợp lệ và report đầy đủ |
| Streaming | Zero unsafe commit, duplicate prefix, reorder và deadlock |
| Safety | ID/source/translation/audio/checksum/review mapping đầy đủ; p95 <300 ms |
| MT | Dùng đúng hai model fine-tuned; terminology ≥95%, critical field ≥99% trên release test |
| TTS | Offline, non-silent, non-corrupt; term/number/unit intelligible |
| Offline | Zero network startup và runtime |
| Reliability | Missing/corrupt model fail startup; worker error không biến thành success |
| Latency | Normal commit→first-audio p95 <1.000 ms hoặc ghi BLOCKED với stage bottleneck |
| Edge memory | Peak RSS toàn direction <200 MB hoặc ghi BLOCKED |
| Reproducibility | Clone + mount Drive + một notebook tạo cùng release result |

### Ngoại lệ GIPFormer đã được chấp nhận tạm thời

- GIPFormer baseline được phép dùng để hoàn thiện integration.
- Quality gate VI-ASR ≥95% critical recall vẫn **không pass** và không được xóa khỏi risk register.
- Mọi báo cáo release phải ghi `VI-ASR QUALITY DEBT`.
- Không dùng Context/MT/safety fast path để che metric ASR gốc.
- Sau khi P7 hoàn tất mới mở lại một workstream độc lập cho GIPFormer.

---

## 15. Deferred workstream sau khi hoàn thành chặng

### D1 — GIPFormer recovery

- Xác định nguyên nhân fine-tune collapse: loss/tokenizer/blank ID, optimizer schedule, source checkpoint heads và decode contract.
- Thêm pre-training 32-sample overfit gate, then 500-sample dev gate trước full run.
- Không train full 19k records nếu overfit/compatibility gate chưa pass.
- Export ONNX và so PyTorch/ONNX ≤1 điểm phần trăm.
- Chỉ promote nếu critical recall ≥95% và clean regression ≤1 điểm.

### D2 — Real-site production gate

- Thu 500–2.000 utterances có consent; target 1.000.
- Group holdout theo site/session/speaker; không dùng final test để tune.
- Báo synthetic và real metrics riêng.

### D3 — Physical device và acoustic path

- Profile Snapdragon device thật, power/thermal/battery.
- Headset/earpiece là default.
- External speaker chỉ mở sau khi AEC pass.

**Production-ready chỉ được công bố sau D1/D2/D3 và mọi production gate liên quan đạt.**

---

## 16. Thứ tự thực thi đề xuất sau khi duyệt

1. P0 — release lock/config.
2. P1 — E2E file-mode fixed suite.
3. P2 và P3 — streaming/reliability cùng safety/TTS.
4. P4 — frozen offline bundle.
5. P5 — latency/RAM optimization.
6. P6 — Qualcomm/Android reference nếu có quyền truy cập.
7. P7 — release candidate/report/demo.
8. D1 — quay lại GIPFormer theo workstream độc lập.
9. D2/D3 — real-site và physical-device production validation.

Không yêu cầu người dùng chạy lại bất kỳ notebook training/data nào đã có evidence hợp lệ. Mỗi task triển khai phải cập nhật checkbox và ghi link commit/report ngay trong file này.
