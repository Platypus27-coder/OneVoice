"""
Text Normalizer — Pre-MT processing for VI↔EN pipeline
=======================================================
Normalizes raw ASR output before feeding into the translation model.
Handles: numbers, units, abbreviations, symbols, industrial codes.

Ported and extended from:
  BetterBox-TTS/viterbox/tts_helper/tts_numberToken.py
  BetterBox-TTS/general/general_tool_audio.py (clearText, normalize_text)
"""

import re
import unicodedata

# ── Number-to-words mappings ──────────────────────────────────────────────────
_VI_DIGITS = {
    "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
    "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín",
}
_EN_DIGITS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}

# ── Industrial unit expansions ─────────────────────────────────────────────────
_UNIT_EXPAND_VI = {
    r"\bkw\b":   "ki-lô-oát",
    r"\bkwh\b":  "ki-lô-oát giờ",
    r"\bmpa\b":  "mê-ga-pa-scan",
    r"\bbar\b":  "ba",
    r"\bpsi\b":  "p-s-i",
    r"\brpm\b":  "vòng mỗi phút",
    r"\bmm\b":   "mi-li-mét",
    r"\bcm\b":   "xen-ti-mét",
    r"\bm\b":    "mét",
    r"\bkg\b":   "ki-lô-gam",
    r"\bton\b":  "tấn",
    r"\b°c\b":   "độ xê",
    r"\b%\b":    "phần trăm",
    r"\bv\b":    "vôn",
    r"\ba\b":    "am-pe",
    r"\bhz\b":   "héc",
}
_UNIT_EXPAND_EN = {
    r"\bkw\b":   "kilowatts",
    r"\bkwh\b":  "kilowatt-hours",
    r"\bmpa\b":  "megapascals",
    r"\brpm\b":  "RPM",
    r"\bmm\b":   "millimeters",
    r"\bcm\b":   "centimeters",
    r"\bkg\b":   "kilograms",
    r"\b°c\b":   "degrees Celsius",
    r"\b%\b":    "percent",
    r"\bv\b":    "volts",
    r"\ba\b":    "amperes",
    r"\bhz\b":   "hertz",
}

# ── Colloquial & Mechanics Slang ──────────────────────────────────────────────
_COLLOQUIAL_VI = {}
_COLLOQUIAL_LOADED = False

def _load_colloquial_dict():
    global _COLLOQUIAL_LOADED
    if _COLLOQUIAL_LOADED:
        return
        
    import os
    import csv
    csv_path = os.path.join(os.path.dirname(__file__), "../../data/colloquial_terms.csv")
    try:
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) >= 2:
                    slang = row[0].strip().lower()
                    formal = row[1].strip().lower()
                    if slang and formal:
                        pattern = r"\b" + re.escape(slang) + r"\b"
                        _COLLOQUIAL_VI[pattern] = formal
    except FileNotFoundError:
        pass # Optional file
    except Exception as e:
        print(f"[Normalizer] ⚠ Could not load colloquial terms: {e}")
        
    _COLLOQUIAL_LOADED = True

# ── Industrial code pattern (e.g. "V-001", "P-3B") ───────────────────────────
_INDUSTRIAL_CODE = re.compile(r'\b([A-Z]{1,3})-(\d{1,4}[A-Z]?)\b')


def _expand_number_vi(match: re.Match) -> str:
    """Convert a digit sequence to Vietnamese words."""
    digits = match.group(0)
    return " ".join(_VI_DIGITS.get(d, d) for d in digits)


def _expand_number_en(match: re.Match) -> str:
    """Convert a digit sequence to English words (simple, 0-9)."""
    digits = match.group(0)
    if len(digits) <= 2:
        return " ".join(_EN_DIGITS.get(d, d) for d in digits)
    return digits  # Keep multi-digit numbers as-is for MT


def normalize_vi(text: str) -> str:
    """
    Normalize Vietnamese ASR output before translation.

    Steps:
      1. Unicode normalization (NFC)
      2. Lowercase
      3. Expand industrial units
      4. Convert equipment codes to readable form
      5. Clean up whitespace
    """
    text = unicodedata.normalize("NFC", text)
    text = text.strip()

    # Expand units (case-insensitive)
    for pattern, replacement in _UNIT_EXPAND_VI.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Normalize colloquial terms (case-insensitive)
    _load_colloquial_dict()
    for pattern, replacement in _COLLOQUIAL_VI.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Expand industrial equipment codes: "Máy XC-03" → "Máy XC không ba"
    text = _INDUSTRIAL_CODE.sub(
        lambda m: f"{m.group(1)} {' '.join(_VI_DIGITS.get(c, c) for c in m.group(2) if c.isdigit())}",
        text
    )

    # Normalize whitespace
    text = " ".join(text.split())
    
    # Capitalize the first letter, lowercase the rest (essential for MarianMT to avoid garbage translation of ALL CAPS)
    if text.isupper():
        text = text.capitalize()
    
    return text


def normalize_en(text: str) -> str:
    """
    Normalize English ASR output before translation.

    Steps:
      1. Strip and clean
      2. Expand industrial units
      3. Clean up whitespace
    """
    text = text.strip()

    # Expand units (case-insensitive)
    for pattern, replacement in _UNIT_EXPAND_EN.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Normalize whitespace
    text = " ".join(text.split())
    return text


def normalize(text: str, lang: str = "vi") -> str:
    """
    Normalize text based on source language.

    Args:
        text: raw ASR output
        lang: "vi" for Vietnamese, "en" for English

    Returns:
        Normalized text ready for machine translation.
    """
    if lang == "vi":
        return normalize_vi(text)
    elif lang == "en":
        return normalize_en(text)
    return text.strip()


if __name__ == "__main__":
    tests_vi = [
        "Máy XC-03 bị lỗi, áp suất hiện tại là 3.5 MPa",
        "Nhiệt độ lò nung đang 850°C, vượt mức 120%",
        "Máy bơm P-001 đang hoạt động ở 1450 RPM",
    ]
    tests_en = [
        "Excavator XC-03 pressure reading is 3.5 MPa",
        "Furnace temperature is 850°C, exceeding limit by 20%",
        "Pump P-001 running at 1450 RPM",
    ]
    print("── Vietnamese normalization ──")
    for t in tests_vi:
        print(f"  IN:  {t}")
        print(f"  OUT: {normalize(t, 'vi')}\n")

    print("── English normalization ──")
    for t in tests_en:
        print(f"  IN:  {t}")
        print(f"  OUT: {normalize(t, 'en')}\n")
