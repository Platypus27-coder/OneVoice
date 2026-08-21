# OneVoice V2 notebooks

Notebooks clone or fast-forward `main` from `https://github.com/Platypus27-coder/OneVoice.git` into
`/content/OneVoice`, while datasets, model caches and reports remain under `/content/drive/MyDrive/OneVoice`.
They only orchestrate Colab/Drive and call checked-in logic from `scripts/` and `src/evaluation/`. A measured run
must produce `run_manifest.json`, `predictions.csv` and `aggregate.json` (the physical audit produces `audit.json`
instead of predictions).

Run in this order:

1. `colab_data_audit_v2.ipynb`: confirm all 8,064 clean and 16,128 noisy VI WAV files exist and decode.
2. `colab_vi_asr_v2.ipynb`: measured GIPFormer clean/noisy passthrough baseline.
3. `colab_denoiser_v2.ipynb`: passthrough vs RNNoise vs DeepFilterNet through the same ASR and fixed test split.
4. `colab_en_asr_v2.ipynb`: English-only clean/noisy SenseVoice benchmark after the ≥6-speaker audit passes.
5. `colab_mt_v2.ipynb`: raw and context-corrected runs on test, minimal-pair and safety suites in both directions.
6. `colab_edge_profile_v2.ipynb`: frozen ONNX compile/profile/numerical check on a Qualcomm hosted device.

Fine-tuning uses only `train.csv`; model selection uses only `dev.csv`; `test.csv` remains final evaluation. The old
acoustic-adapter notebook was removed because its adapted mel was never passed to GIPFormer and its post-WER was made
by copying the baseline prediction and multiplying the score by a constant. The original source remains auditable at
tag `v1-working-baseline`.
