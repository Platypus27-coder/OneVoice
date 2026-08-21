# OneVoice V1 baseline status

- Baseline tag: `v1-working-baseline`
- Baseline commit: `ce35ac5`
- Recorded: 2026-08-21
- Purpose: immutable tracked rollback point before the V2 upgrade
- Machine-readable environment report: [`baseline/v1_environment.json`](baseline/v1_environment.json)

## Reproduction status

Full model reproduction was not run during the source upgrade because the repository contains no local `models/`
artifacts and V1 downloads models at runtime. The tag records source truth, not a claim that current hardware metrics
were reproduced.

| Component | V1 status discovered from source |
|---|---|
| VI ASR | GIPFormer INT8, runtime download required when cache/local model is absent |
| EN ASR | Partial: configured remote ID was treated as a local path, so default EN→VI could return empty text |
| Denoise | Invalid claim: GIPFormer decoded audio but returned the original waveform |
| MT | EnViT5 for both directions; local checkpoint optional, remote fallback active |
| TTS | F5/OmniVoice with pyttsx3/gTTS/silence fallbacks; silence could be reported as success |
| Streaming | One-second independent VAD blocks, not rolling semantic streaming |
| Offline | Not production-offline; multiple runtime download/online paths existed |
| Qualcomm/RAM/latency | Not verified on target hardware |

## V2 rule

All improvements are measured against this tag. A missing model, empty ASR result, invalid translation or silence TTS
is a failure and must not be converted into a passing metric.
