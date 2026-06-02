"""
SenseVoice ASR Engine (English)
================================
Uses SenseVoice ONNX for high-speed English transcription and Emotion Detection.
Extracts emotion tags to pass to the TTS engine.
"""

import os
import re
import numpy as np

# Lazy load to avoid slowing down startup if SenseVoice is not used
_SenseVoiceSmall = None

class SenseVoiceASR:
    """Wrapper for SenseVoice ONNX ASR."""
    
    def __init__(self, config: dict):
        global _SenseVoiceSmall
        if _SenseVoiceSmall is None:
            try:
                from funasr_onnx import SenseVoiceSmall
                _SenseVoiceSmall = SenseVoiceSmall
            except ImportError:
                print("[ASR] ⚠ funasr_onnx not installed. Please `pip install funasr_onnx modelscope`")
                self.model = None
                return

        self.model_dir = config.get("sensevoice", {}).get("model_path", "models/sensevoice")
        
        if not os.path.exists(self.model_dir):
            # SenseVoice chỉ cần cho chiều EN→VI. Với VI→EN thì bỏ qua cảnh báo này.
            print(f"[ASR] ℹ SenseVoice không có tại '{self.model_dir}' — sẽ tải tự động khi cần (EN→VI).")
            self.model = None
        else:
            try:
                # Use quantize=True for edge deployment
                self.model = _SenseVoiceSmall(self.model_dir, batch_size=1, quantize=True)
                print(f"[ASR] ✅ SenseVoice ONNX loaded from {self.model_dir}")
            except Exception as e:
                print(f"[ASR] ⚠ SenseVoice load failed: {e}")
                self.model = None

    def _parse_output(self, raw_text: str) -> dict:
        """
        Parse SenseVoice output tags:
        e.g., "<|en|><|HAPPY|><|Speech|><|woitn|>Hello there."
        """
        # 1. Extract Emotion
        emo_match = re.search(r"<\|(HAPPY|SAD|ANGRY|NEUTRAL|FEARFUL|DISGUSTED|SURPRISED)\|>", raw_text, re.IGNORECASE)
        emotion = emo_match.group(1).lower() if emo_match else "neutral"
        
        # 2. Extract Event (Optional, for logging)
        event_match = re.search(r"<\|(BGM|Speech|Applause|Laughter|Cry|Sneeze|Breath|Cough)\|>", raw_text, re.IGNORECASE)
        event = event_match.group(1).lower() if event_match else "speech"

        # 3. Clean Text (remove all <|TAG|> tokens)
        clean_text = re.sub(r"<\|.*?\|>", "", raw_text).strip()
        
        return {
            "text": clean_text,
            "emotion": emotion,
            "event": event,
            "raw": raw_text
        }

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> dict:
        """
        Transcribe audio and return text with emotion metadata.
        """
        if self.model is None or len(audio) == 0:
            return {"text": "", "emotion": "neutral", "event": "speech"}
            
        try:
            # SenseVoice funasr_onnx expects 16kHz audio array or list of arrays
            audio_f32 = audio.astype(np.float32)
            
            # Ensure proper range [-1, 1] if not already
            if np.abs(audio_f32).max() > 1.0:
                audio_f32 /= 32768.0
                
            res = self.model([audio_f32], language="en", use_itn=True)
            
            if not res:
                return {"text": "", "emotion": "neutral", "event": "speech"}
                
            raw_text = res[0]
            parsed = self._parse_output(raw_text)
            
            # Log detected emotion if it's significant
            if parsed["emotion"] != "neutral" or parsed["event"] not in ["speech", "none"]:
                print(f"[SenseVoice] 🎭 Emotion: {parsed['emotion'].upper()} | Event: {parsed['event'].upper()}")
                
            return parsed
            
        except Exception as e:
            print(f"[ASR] ⚠ SenseVoice transcription failed: {e}")
            return {"text": "", "emotion": "neutral", "event": "speech"}
