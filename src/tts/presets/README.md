# Voice Presets for F5-TTS (English Output)

Place your reference audio files here. Each preset needs 2 files:
- `<name>.wav` — 5–10 seconds of clean, natural English speech (no background noise)
- `<name>.txt` — exact transcript of the WAV file

## How to add a preset

1. Record or download 5–10 seconds of the target English voice (studio quality preferred).
2. Save as `src/tts/presets/my_voice.wav`.
3. Write the exact spoken transcript in `src/tts/presets/my_voice.txt`.
4. Update `config/config.yaml`:
   ```yaml
   tts:
     en_preset_audio: "src/tts/presets/my_voice.wav"
     en_preset_text:  "The exact transcript goes here."
   ```

## Automatic download

Run the helper script to auto-download a free, studio-quality English voice:
```bash
python scripts/download_voice_preset.py
```

## Tips for best quality

- Use mono WAV at 16kHz or 24kHz sample rate.
- The voice should be calm, clear, and mid-tempo (like a news anchor or audiobook narrator).
- Avoid recordings with reverb, music, or background noise.
- Avoid very fast or very emotional speech — it affects how the cloned voice sounds.
- The transcript MUST exactly match what is spoken (no extra words).
