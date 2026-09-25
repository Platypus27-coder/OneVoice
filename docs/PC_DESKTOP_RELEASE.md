# OneVoice V2 — Windows desktop release candidate

## Scope

The current delivery target is a **Windows PC offline demo** for a portfolio
and CV. Android, Snapdragon acceleration and Bluetooth headset integration are
future work; they are not prerequisites for this desktop release candidate.

Reference machine for the first acceptance run:

| Component | Value |
|---|---|
| Laptop | ASUS TUF Gaming F15 FX506HF |
| CPU | Intel Core i5-11400H |
| RAM | 24 GB |
| GPU | NVIDIA RTX 2050, 4 GB VRAM |
| Runtime mode | local, offline, `edge` profile |

The named P5 budget `desktop_tuf_f15_rtx2050` is deliberately different from
the portable `edge_200mb` budget. It records the actual desktop target rather
than making an unsupported 200 MB mobile claim.

## What “complete for CV” means

The release candidate is complete when the following evidence is checked in or
linked from the release report:

1. Current ASR/MT artifacts are hash-locked and copied into a local bundle.
2. The bundle loads without network access and passes one smoke per direction.
3. The fixed streaming suite passes normal and approved-safety routes in both
   directions.
4. A 30-minute offline streaming soak report has zero failed turns.
5. P5 is measured on this Windows PC using the named desktop hardware profile.
6. `report.md`, `report.html`, `summary.json`, and the README SVG summarize the
   selected runtime rather than historical candidates.

This is a demonstrable offline prototype, not a claim of site-ready safety
deployment. The data is synthetic/hosted, safety WAVs are internally approved
demo assets, the Vietnamese GIPFormer baseline remains below the 95% critical
term gate, and system-voice TTS is functional rather than a voice-quality
claim.

## Make a local bundle once

Run these commands on a machine that currently has the verified artifacts
(for example, after copying them off Drive). They never download model files.
`copy` can be re-run safely: hash-matching files are reused.

```powershell
$Repo = "D:\code\.vscode\OneVoice\onevoice-edge"
$Artifacts = "D:\OneVoiceArtifacts\runtime_demo_manifest.json"
$Lock = "D:\OneVoiceArtifacts\release_lock_v2.json"
$BundleRoot = "D:\OneVoiceDesktop\onevoice-v2-rc1"
$RuntimeConfig = "D:\OneVoiceArtifacts\runtime_demo_local.yaml"

Set-Location $Repo
python scripts/build_release_bundle.py `
  --artifact-manifest $Artifacts `
  --output-dir "$BundleRoot\vi2en" `
  --direction vi2en --mode copy --release-lock $Lock `
  --runtime-config $RuntimeConfig
python scripts/build_release_bundle.py `
  --artifact-manifest $Artifacts `
  --output-dir "$BundleRoot\en2vi" `
  --direction en2vi --mode copy --release-lock $Lock `
  --runtime-config $RuntimeConfig
```

Each copied bundle contains `manifest.json`, `receipt.json`, the selected
models, approved safety audio, construction data, and a `runtime_config.yaml`
whose paths are local to that bundle. A bundle is an immutable model/data
package; the Git repository provides the executable Python runtime.

## Offline smoke on Windows

Install the locked Python dependencies and the required local audio backends
once. In particular, the selected runtime needs `sherpa_onnx` for VI ASR,
`funasr_onnx` for EN ASR, ONNX Runtime/Transformers for MT, and an offline
system TTS backend. The preflight check fails clearly if one is absent.

For the local eSpeak NG demo fallback installed inside the `onevoice` Conda
environment, register its executable and voice-data paths once:

```powershell
$Repo = "D:\code\.vscode\OneVoice\onevoice-edge"
& "$Repo\scripts\install_espeak_conda_hooks.ps1" `
  -Prefix "D:\MINICONDA\envs\onevoice"
```

Then open a new PowerShell session and run `conda activate onevoice`. The
activation hook makes `espeak-ng` discoverable and sets `ESPEAK_DATA_PATH`; it
does not install eSpeak NG or change the voice model. This lightweight system
voice is a functional offline demo fallback, not a production voice-quality
claim.

Run each command from the corresponding bundle directory so
`runtime_config.yaml` resolves only local paths:

```powershell
$Repo = "D:\code\.vscode\OneVoice\onevoice-edge"
Set-Location "D:\OneVoiceDesktop\onevoice-v2-rc1\vi2en"
$env:PYTHONPATH = "$Repo\src"
python "$Repo\scripts\verify_release_bundle.py" `
  --bundle-dir . --direction vi2en --config runtime_config.yaml --profile edge `
  --input-file artifacts\safety_audio\SAFE2_0001_en2vi.wav `
  --output-file reports\vi2en_smoke.wav --report-dir reports

Set-Location "D:\OneVoiceDesktop\onevoice-v2-rc1\en2vi"
python "$Repo\scripts\verify_release_bundle.py" `
  --bundle-dir . --direction en2vi --config runtime_config.yaml --profile edge `
  --input-file artifacts\safety_audio\SAFE2_0001_vi2en.wav `
  --output-file reports\en2vi_smoke.wav --report-dir reports
```

Expected result: two `bundle_verify_*.json` reports with `passed: true` and
`network_blocked: true`.

## Measure the actual PC (P5)

P5 measures the **whole chain**—audio, ASR, translation, safety/context, TTS
and generated audio—not only a model benchmark. It should be run after the
offline smokes, from each bundle directory.

```powershell
$Repo = "D:\code\.vscode\OneVoice\onevoice-edge"
Set-Location "D:\OneVoiceDesktop\onevoice-v2-rc1\vi2en"
python "$Repo\scripts\profile_release_runtime.py" `
  --config runtime_config.yaml --direction vi2en --profile edge `
  --hardware-profile desktop_tuf_f15_rtx2050 --repeats 5 `
  --normal-input artifacts\safety_audio\SAFE2_0001_en2vi.wav `
  --safety-input artifacts\safety_audio\SAFE2_0001_en2vi.wav `
  --report-dir reports\p5_vi2en
```

For a meaningful normal-route result, replace `--normal-input` with a local
non-safety WAV. Do not use a safety WAV for both routes in the final report.
The output records the CPU/RAM target source, p50/p95 timing and any blocker;
it does not silently relax a budget.

## CV-safe project wording

> Built OneVoice V2, an offline Vietnamese–English speech-to-speech prototype
> for industrial communication. Integrated streaming ASR, domain-aware MT,
> deterministic approved-safety audio routing, hash-locked local artifacts and
> no-network release verification; evaluated component quality and a 30-minute
> streaming soak before packaging a Windows desktop release candidate.

Keep the accompanying report links and the limitations above. Do not claim
real-site validation, Android deployment, certified safety operation, or
sub-second desktop latency until a measurement on the target PC proves it.

## Future roadmap

The Android/Snapdragon/headset plan remains at
[Android Snapdragon execution gate](ANDROID_SNAPDRAGON_EXECUTION.md). It starts
only after this desktop candidate is packaged and demonstrated successfully.
