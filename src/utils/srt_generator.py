"""
SRT Subtitle Generator
======================
Generates SRT subtitle files from translation timing data.
Ported from BetterBox-TTS/general/general_tool_audio.py (create_srt_file).

Reference:
  BetterBox-TTS — Dolly VN / ContextBoxAI (CC BY-NC 4.0)
    https://github.com/nowtranminh1-TTS/BetterBox-TTS
"""

import os
from pathlib import Path
from typing import List
from dataclasses import dataclass, field


@dataclass
class SRTEntry:
    """Single SRT subtitle entry."""
    index: int
    start_time: float    # seconds
    end_time: float      # seconds
    original_text: str   # source language text
    translated_text: str # target language text


def _format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


class SRTGenerator:
    """
    Accumulates translation entries with timing and writes SRT files.

    Features:
    - Bilingual SRT (original + translated on separate lines)
    - Single-language SRT (translated only)
    - Compatible with BetterBox-TTS create_srt_file format
    """

    def __init__(self, bilingual: bool = True):
        self.bilingual = bilingual
        self._entries: List[SRTEntry] = []
        self._current_time: float = 0.0

    def add_entry(self, original: str, translated: str, duration_s: float):
        """
        Add a new subtitle entry.

        Args:
            original: source language text
            translated: target language translated text
            duration_s: duration of this segment in seconds
        """
        start = self._current_time
        end = start + duration_s
        entry = SRTEntry(
            index=len(self._entries) + 1,
            start_time=start,
            end_time=end,
            original_text=original,
            translated_text=translated,
        )
        self._entries.append(entry)
        self._current_time = end

    def to_srt_string(self) -> str:
        """Generate SRT content as a string."""
        lines = []
        for entry in self._entries:
            lines.append(str(entry.index))
            lines.append(
                f"{_format_srt_time(entry.start_time)} --> {_format_srt_time(entry.end_time)}"
            )
            if self.bilingual:
                lines.append(entry.original_text)
                lines.append(entry.translated_text)
            else:
                lines.append(entry.translated_text)
            lines.append("")  # blank line between entries
        return "\n".join(lines)

    def save(self, output_path: str) -> str:
        """
        Save SRT file to disk.

        Args:
            output_path: full path to .srt file

        Returns:
            Absolute path to saved file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_srt_string(), encoding="utf-8")
        print(f"[SRT] ✅ Saved: {path}")
        return str(path.resolve())

    def reset(self):
        """Clear all entries and reset timer."""
        self._entries = []
        self._current_time = 0.0

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    # ── BetterBox-TTS compatible interface ────────────────────────────────────
    @staticmethod
    def from_timing_items(
        timing_items: List[dict], output_path: str
    ) -> str:
        """
        Create SRT file from BetterBox-TTS compatible timing_items format.

        Args:
            timing_items: List of {"startTime": float, "endTime": float, "text": str}
            output_path: path to write .srt file

        Returns:
            Path to saved file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        srt_lines = []
        for idx, item in enumerate(timing_items, start=1):
            start = _format_srt_time(item["startTime"])
            end = _format_srt_time(item["endTime"])
            text = item["text"]
            srt_lines.extend([str(idx), f"{start} --> {end}", text, ""])

        path.write_text("\n".join(srt_lines), encoding="utf-8")
        return str(path.resolve())


if __name__ == "__main__":
    gen = SRTGenerator(bilingual=True)
    gen.add_entry("Máy xúc số 3 bị lỗi thủy lực.", "Excavator 3 has a hydraulic failure.", 2.5)
    gen.add_entry("Cần kiểm tra ngay lập tức.", "Needs immediate inspection.", 1.8)
    gen.add_entry("Van an toàn đường ống số 5 bị rò rỉ.", "Safety valve on pipeline 5 is leaking.", 3.2)

    content = gen.to_srt_string()
    print("── Generated SRT ──")
    print(content)
    gen.save("output_test.srt")
