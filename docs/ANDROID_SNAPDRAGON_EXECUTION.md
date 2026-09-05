# Android Snapdragon execution gate

This document turns the approved mobile/headset direction into executable
pre-device gates. It does not claim that the current Python runtime is an
Android application.

## What is already verified

- Current ASR/MT release artifacts are selected and hash-locked by direction.
- Offline normal/safety E2E and the fixed streaming suite passed in the
  recorded Colab release run.
- A 30-minute streaming soak completed 325/325 turns without a reported
  failure.
- The current Colab process footprint is too large for the legacy 200 MB
  profile. A physical phone must receive its own measured profile.

## P6-00: run before selecting the phone

On a machine that has the approved Drive artifacts, copy one direction into a
local bundle. `copy` is deliberately different from the old Drive inventory:
it writes files below the bundle and creates a local `runtime_config.yaml`.

```bash
python scripts/build_release_bundle.py \
  --artifact-manifest /path/to/runtime_demo_manifest.json \
  --output-dir /path/to/onevoice-phone/vi2en \
  --direction vi2en --mode copy \
  --runtime-config /path/to/runtime_demo_local.yaml

python scripts/audit_mobile_readiness.py \
  --bundle-dir /path/to/onevoice-phone/vi2en \
  --direction vi2en \
  --output /path/to/reports/mobile_readiness_vi2en.json
```

Repeat for `en2vi`. The audit must report:

- `artifact_portable: true`;
- `runtime_config_present: true`;
- no artifact/config blockers.

`android_app_ready` remains `false` by design: it records that Android native
adapters have not yet been written or benchmarked.

## P6-01: component RAM measurement

Run the profiler once per direction before choosing a phone. It measures the
cumulative RSS after context/safety, denoiser, ASR, MT and TTS load. The delta
is a host observation rather than a model file size.

```bash
python scripts/profile_component_memory.py \
  --config /path/to/runtime_demo_local.yaml \
  --direction vi2en --profile edge \
  --output /path/to/reports/component_memory_vi2en.json
```

The physical reference device should have enough RAM for the measured process
plus Android, audio buffers and thermal headroom. The report must be rerun on
the selected device.

## Device and audio sequence

1. Choose one Android Snapdragon reference phone after the two audits.
2. Build a CPU/GPU offline baseline using local artifacts only.
3. Validate phone microphone -> phone speaker/headphone output.
4. Validate Bluetooth playback.
5. Validate Bluetooth microphone plus playback as a separate duplex gate.
6. Run P5 and the streaming soak on the physical phone.
7. Investigate Qualcomm/QNN model acceleration only after the baseline has
   quality parity.

The release model selection does not change during these steps. Any FP16,
INT8, QNN or replacement candidate requires the same fixed quality gate and a
rollback path.
