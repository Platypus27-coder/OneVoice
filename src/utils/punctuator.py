"""
Trạm 1.5: Khôi phục Dấu câu (Punctuation Restoration)
=======================================================
Chuyển output ASR thô (không dấu câu) thành văn bản có cấu trúc
để MT model dịch từng câu hoàn chỉnh thay vì một dòng liên tục.

Priority:
  1. deepmultilingualpunctuation (kredor/punctuate-all — hỗ trợ tiếng Việt)
  2. Rule-based fallback (không cần GPU, không cần model)

Usage:
    from utils.punctuator import restore_punctuation
    punced = restore_punctuation("cậu đã làm gì với nó vậy thêm năng lượng hả")
    # → "Cậu đã làm gì với nó vậy? Thêm năng lượng hả?"
"""

import re

# ── Vietnamese question-ending particles ──────────────────────────────────────
_VI_QUESTION = {
    "hả", "không", "à", "ư", "nhỉ", "chứ", "nha",
    "phải không", "được không", "chưa", "ạ", "hả?",
}

# ── Words that signal end of a statement sentence ────────────────────────────
_VI_STATEMENT_END = {
    "vậy", "rồi", "luôn", "liền", "xong", "thôi",
    "được", "đó", "nhé", "ngay", "liền",
}

# ── Words that typically start a NEW sentence in speech ──────────────────────
# Note: Temporarily disabled because they cause too many false positive splits
_VI_SENTENCE_STARTERS = set()

# ── Lazy-loaded model ─────────────────────────────────────────────────────────
_punc_model = None
_punc_type: str = "unloaded"


def _load_model() -> tuple:
    global _punc_model, _punc_type
    if _punc_type != "unloaded":
        return _punc_model, _punc_type

    try:
        import transformers
        # Monkey-patch pipeline to fix grouped_entities error in transformers >= 4.38
        orig_pipeline = transformers.pipeline
        def patched_pipeline(*args, **kwargs):
            if "grouped_entities" in kwargs:
                kwargs["aggregation_strategy"] = "none" if not kwargs["grouped_entities"] else "simple"
                del kwargs["grouped_entities"]
            return orig_pipeline(*args, **kwargs)
        transformers.pipeline = patched_pipeline

        from deepmultilingualpunctuation import PunctuationModel
        # kredor/punctuate-all hỗ trợ tiếng Việt (vi) trong 17 ngôn ngữ
        _punc_model = PunctuationModel(model="kredor/punctuate-all")
        _punc_type = "deep"
        
        # Restore pipeline
        transformers.pipeline = orig_pipeline
        print("[Punc] ✅ deepmultilingualpunctuation (kredor/punctuate-all) loaded.")
    except Exception as e:
        _punc_model = None
        _punc_type = "rule"
        print(f"[Punc] ℹ Rule-based punctuation active (install 'deepmultilingualpunctuation' for ML quality). Reason: {e}")

    return _punc_model, _punc_type


# ── Rule-based fallback ───────────────────────────────────────────────────────

def _capitalize(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def _rule_based(text: str) -> str:
    """
    Heuristic punctuation for Vietnamese ASR output.
    Splits at question particles and statement-ending words,
    then capitalizes the first word of each sentence.
    """
    words = text.split()
    if not words:
        return text

    sentences: list[str] = []
    buf: list[str] = []
    min_sentence_len = 4   # Tối thiểu 4 từ trước khi ngắt câu

    for i, word in enumerate(words):
        buf.append(word)
        w_lower = word.lower()
        remaining = len(words) - i - 1

        is_question = w_lower in _VI_QUESTION
        is_stmt_end = w_lower in _VI_STATEMENT_END
        is_long_enough = len(buf) >= min_sentence_len
        is_last = (remaining == 0)

        # Kiểm tra từ tiếp theo xem có phải là khởi đầu câu mới không
        next_is_starter = (
            remaining > 0 and words[i + 1].lower() in _VI_SENTENCE_STARTERS
            and is_long_enough
        )

        if is_last:
            punct = "?" if is_question else "."
            sentences.append(_capitalize(" ".join(buf)) + punct)
            buf = []
        elif is_question and is_long_enough:
            sentences.append(_capitalize(" ".join(buf)) + "?")
            buf = []
        elif is_stmt_end and is_long_enough:
            sentences.append(_capitalize(" ".join(buf)) + ".")
            buf = []
        elif next_is_starter:
            sentences.append(_capitalize(" ".join(buf)) + ".")
            buf = []

    # Remaining words (nếu có)
    if buf:
        sentences.append(_capitalize(" ".join(buf)) + ".")

    return " ".join(sentences)


# ── Public API ────────────────────────────────────────────────────────────────

def restore_punctuation(text: str, lang: str = "vi") -> str:
    """
    Thêm dấu câu vào văn bản ASR thô (không dấu câu).

    Args:
        text: Văn bản thô từ ASR (vd: "cậu đã làm gì với nó vậy thêm năng lượng")
        lang: Mã ngôn ngữ nguồn ("vi" hoặc "en")

    Returns:
        Văn bản có dấu câu và viết hoa đầu câu.
    """
    if not text or not text.strip():
        return text

    # Nếu đã có dấu câu → bỏ qua
    if re.search(r"[.!?,;]", text):
        return text

    model, model_type = _load_model()
    
    # ML models fail on ALL CAPS text. If ASR outputs all caps, we must lowercase it first.
    original_is_upper = text.isupper()
    if original_is_upper:
        text = text.lower()

    if model_type == "deep":
        try:
            result = model.restore_punctuation(text)
            # Đảm bảo kết quả hợp lệ
            if result and len(result) > len(text) * 0.5:
                print(f"[Punc] ✅ Restored: \"{result[:80]}{'...' if len(result)>80 else ''}\"")
                return result
        except Exception as e:
            print(f"[Punc] ⚠ Model error: {e}. Falling back to rule-based.")

    # Rule-based fallback
    result = _rule_based(text)
    print(f"[Punc] ℹ Rule-based: \"{result[:80]}{'...' if len(result)>80 else ''}\"")
    return result
