# OneVoice benchmark report (release)

Generated: `2026-08-29T13:42:43+00:00`

> Evidence is from checked benchmark artifacts. Synthetic audio and hosted/CPU measurements are not field-device or real-site validation.

## Scope

Current runtime only: official GIPFormer ONNX baseline for VI→EN, fine-tuned SenseVoice ONNX FP32 for EN→VI, and promoted EnViT5 translation validators. Historical and rejected fine-tune runs are excluded.

## Summary

- Aggregate artifacts in this report: **10**
- MT artifacts: **6**
- ASR artifacts: **4**

## Artifacts

| Family | Report | Direction/suite | Samples | WER/error | Critical | p95 ms | Gate |
|---|---|---|---:|---:|---:|---:|---|
| ASR | `denoiser/passthrough/clean` | vi2en / clean | 1958 | 8.93% | 85.76% | 524.8 | BELOW 95% |
| ASR | `denoiser/passthrough/noisy` | vi2en / noisy | 1958 | 11.32% | 76.29% | 426.0 | BELOW 95% |
| ASR | `en_asr_onnx_fp32_v1/full_clean` | en2vi / clean | 1273 | 0.37% | 99.52% | 1581.9 | PASS |
| ASR | `en_asr_onnx_fp32_v1/full_noisy` | en2vi / noisy | 2546 | 0.54% | 99.17% | 1591.3 | PASS |
| MT | `mt/candidate_en2vi_validator_v2/en2vi/minimal/context` | en2vi / minimal | 480 | 8.79% | 95.83% | 462.9 | PASS |
| MT | `mt/candidate_en2vi_validator_v2/en2vi/safety/context` | en2vi / safety | 196 | 22.80% | 100.00% | 318.6 | PASS |
| MT | `mt/candidate_en2vi_validator_v2/en2vi/test/context` | en2vi / test | 979 | 1.81% | 98.98% | 382.3 | PASS |
| MT | `mt/candidate_vi2en_validator_v4/vi2en/minimal/context` | vi2en / minimal | 480 | 15.36% | 97.92% | 2387.1 | PASS |
| MT | `mt/candidate_vi2en_validator_v4/vi2en/safety/context` | vi2en / safety | 196 | 0.11% | 100.00% | 1878.9 | PASS |
| MT | `mt/candidate_vi2en_validator_v4/vi2en/test/context` | vi2en / test | 979 | 1.47% | 99.28% | 2392.3 | PASS |
