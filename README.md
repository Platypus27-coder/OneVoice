# OneVoice Edge V2

OneVoice is a Vietnamese ↔ English speech-to-speech runtime for construction environments. V2 upgrades the existing
CLI in place while keeping the V1 tracked baseline at tag `v1-working-baseline` (commit `ce35ac5`).

This repository does **not** currently claim production readiness. Offline operation, p95 latency, quality under
construction noise, peak RSS below 200 MB and real-site safety must all be demonstrated by generated reports.

## Source-of-truth status

| Area | Status | Evidence / limitation |
|---|---|---|
| V1 rollback and source audit | `VERIFIED` | Annotated tag and [`docs/V1_BASELINE_STATUS.md`](docs/V1_BASELINE_STATUS.md) |
| V2 contracts, Context Engine, Site Pack, safety matching | `VERIFIED` | Deterministic implementation and unit tests |
| Semantic commit and 32 ms frame/VAD session | `PARTIAL` | Logic is tested; real model/audio p95 has not been measured |
| Denoising | `FALLBACK` | Passthrough is the baseline; RNNoise and DeepFilterNet remain quality-gated candidates |
| VI ASR / EN ASR / MT / TTS runtime | `PARTIAL` | Adapters exist; this checkout has no complete local artifact bundle for end-to-end reproduction |
| Offline edge startup | `PARTIAL` | Strict hash/license/sample-rate/backend preflight and network-guard profiler exist; model bundle is absent |
| Safety audio | `PLANNED` | Builder requires approved phrases and an approval record; current CSV rows still need safety-officer review |
| Physical 8,064 clean / 16,128 noisy audit | `PLANNED` | Must run against the Drive dataset; logical counts are not physical evidence |
| Qualcomm Snapdragon 8 Gen 3 | `PLANNED` | Tool submits frozen ONNX graphs to AI Hub; no hosted-device report has been produced |
| Real-site robustness | `PLANNED` | Import/audit/holdout-lock tooling exists; no fixed real holdout is present |

## V2 runtime

```text
32 ms AudioFrame
→ waveform denoiser
→ stateful energy-VAD utterance session
→ rolling ASR hypotheses
→ stable-prefix alignment
→ Construction Context Engine
→ semantic commit (WAIT / COMMIT_NORMAL / COMMIT_SAFETY)
→ safety audio, translation memory, or MT
→ critical-field validation
→ ordered single-worker TTS
→ first-audio and latency reports
```

Site Pack precedence is deterministic:

```text
site safety override
→ site translation memory
→ site terminology/equipment/zone/abbreviation/slang
→ canonical V2 terminology
→ base normalization
```

Safety phrases require two consecutive partial matches or one endpoint match. Edge startup rejects unreviewed safety
data, and edge safety commits require pre-generated checksum-verified audio. Silence is an error, never a successful
TTS result.

## CLI

The V1 entry point remains valid:

```bash
python src/pipeline.py --direction vi2en
python src/pipeline.py --direction en2vi
```

V2 options:

```bash
python src/pipeline.py --direction vi2en \
  --profile development \
  --site-pack path/to/site_pack.json \
  --input-file path/to/input.wav \
  --output-file reports/output.wav \
  --report-dir reports/runtime
```

Profiles:

- `development`: permits model preparation/downloads and premium fallbacks.
- `premium`: enables F5-TTS/OmniVoice when resources and local artifacts permit.
- `edge`: implies `--offline`, excludes F5/OmniVoice, requires an artifact manifest and approved local safety audio.

Production simultaneous use assumes a headset or earpiece. External-speaker full duplex is not a production mode
until acoustic echo cancellation is implemented and validated.

## Reproducible evaluation

The notebooks only mount Drive, configure paths, invoke checked-in modules and display reports:

- `notebooks/colab_data_audit_v2.ipynb`
- `notebooks/colab_vi_asr_v2.ipynb`
- `notebooks/colab_denoiser_v2.ipynb`
- `notebooks/colab_en_asr_v2.ipynb`
- `notebooks/colab_mt_v2.ipynb`
- `notebooks/colab_edge_profile_v2.ipynb`

Every measured benchmark writes `run_manifest.json`, `predictions.csv` and `aggregate.json`. Empty predictions count
as full errors. The removed notebook logic that copied baseline predictions and multiplied WER by a constant is not
accepted as evidence. See [`notebooks/README_V2.md`](notebooks/README_V2.md).

Useful commands:

```bash
python scripts/audit_audio_dataset.py /drive/onevoice_audio_v1/manifest.jsonl \
  --expected-clean 8064 --expected-noisy 16128 --report-dir reports/data_audit

python scripts/benchmark_asr_v2.py /drive/manifest.jsonl \
  --direction vi2en --split test --audio noisy --denoiser passthrough \
  --report-dir reports/asr/noisy

python scripts/benchmark_mt_v2.py --direction vi2en --suite minimal \
  --report-dir reports/mt/raw

python scripts/benchmark_mt_v2.py --direction vi2en --suite minimal --with-context \
  --report-dir reports/mt/context

python scripts/audit_real_site.py --manifest /pilot/manifest.jsonl \
  --report reports/real_site/audit.json --holdout-lock /pilot/holdout_lock.json --final-gate
```

Fine-tuning is permitted only when the context-corrected baseline misses its gate. MT training uses `train.csv`,
checkpoint selection uses `dev.csv`, and `test.csv` is evaluation-only. The legacy filename
`notebooks/finetune_marian.py` now trains EnViT5 with those split rules.

## Offline artifacts and edge profiling

Create a concrete manifest from local, licensed files:

```bash
python scripts/build_artifact_manifest.py \
  --spec artifacts/artifact_spec.json --output artifacts/manifest.json

python scripts/profile_edge_runtime.py --direction vi2en \
  --report reports/edge/vi2en.json
```

The profiler blocks network connections and samples native process RSS during full model loading. A missing/corrupt
artifact, unapproved safety row, missing backend, silence TTS or worker exception fails visibly.

Qualcomm tooling accepts only an already-frozen and locally validated ONNX graph; it does not pretend that exporting a
single seq2seq logits graph implements translation generation:

```bash
python notebooks/export_qai.py --model models/frozen/model.onnx \
  --inputs models/frozen/correctness_input.npz \
  --device "<exact AI Hub device name>" --report-dir reports/qai/model
```

AI Hub results must be labeled `qualcomm_ai_hub_hosted_device`. They are not field-device validation.

## Acceptance gates

- Zero unsafe commits and no duplicate spoken prefix on the fixed safety/minimal-pair suite.
- Normal commit→first-audio p95 below 1,000 ms; safety p95 below 300 ms. Speech→commit is reported separately.
- Denoiser must improve noisy WER/CTER by at least 5% relative, preserve Critical Term Recall, and degrade clean WER
  by no more than one percentage point.
- MT adaptation must improve the gated error metric by at least three percentage points and degrade the general set by
  no more than one point.
- Edge runtime performs no network access and full-profile peak RSS is below 200 MB.
- Quantized models regress numerical/quality metrics by no more than one percentage point.
- Production Definition of Done requires a separately reported, group-isolated fixed real-site holdout.

## Tests

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts notebooks
```

## License

Repository license and third-party model licenses are separate. Every production artifact must include concrete license
metadata in `artifacts/manifest.json`; the preflight rejects schema V2 entries without it.
