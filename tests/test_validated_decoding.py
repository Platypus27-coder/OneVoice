from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asr.sensevoice_asr import SenseVoiceASR
from context.engine import ConstructionContextEngine
from pipeline import OneVoicePipeline
from translation.mt_engine import Translator


class ValidatedDecodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = ConstructionContextEngine.from_data_dir(ROOT / "data/onevoice_construction_v2")

    def pipeline(self, candidates):
        pipeline = OneVoicePipeline.__new__(OneVoicePipeline)
        pipeline.context = self.context
        pipeline.translator = SimpleNamespace(translate_candidates=lambda source, direction: candidates)
        return pipeline

    def test_lower_beam_recovers_a_valid_term_without_rewriting_the_output(self):
        context = self.context.analyze("Do not tighten the radiator", "en2vi")
        candidates = ["Không siết sàn làm mát.", "Không siết két làm mát."]
        text, errors, trace = self.pipeline(candidates)._translate_validated(context.source_text, context, "en2vi")
        self.assertEqual(text, candidates[1])
        self.assertEqual(errors, [])
        self.assertEqual(trace[-1]["rank"], 2)

    def test_valid_top_one_is_retained(self):
        context = self.context.analyze("Do not tighten the radiator", "en2vi")
        first = "Không siết két làm mát."
        text, _, trace = self.pipeline([first, "Siết két làm mát."])._translate_validated(context.source_text, context, "en2vi")
        self.assertEqual(text, first)
        self.assertEqual(len(trace), 1)

    def test_no_valid_beam_retains_failure_and_never_inserts_glossary_words(self):
        context = self.context.analyze("Do not tighten the radiator", "en2vi")
        candidates = ["Không siết sàn làm mát.", "Siết két làm mát."]
        text, errors, trace = self.pipeline(candidates)._translate_validated(context.source_text, context, "en2vi")
        self.assertEqual(text, candidates[0])
        self.assertTrue(errors)
        self.assertIn("missing_negation", trace[1]["validation_errors"])

    def test_empty_candidate_cannot_pass(self):
        context = self.context.analyze("Hello", "en2vi")
        self.assertIn("empty_translation", self.pipeline([""])._translate_validated("Hello", context, "en2vi")[1])

    def test_contextual_barricade_verb_is_not_a_missing_noun(self):
        context = self.context.analyze("Please barricade the edge of the full-body harness first", "en2vi")
        self.assertEqual(self.context.validate_translation("Rào mép dây đai toàn thân trước.", context, "en2vi"), [])
        noun = self.context.analyze("Inspect the barricade", "en2vi")
        self.assertTrue(self.context.validate_translation("Kiểm tra mép.", noun, "en2vi"))

    def test_negative_predicate_does_not_cover_another_prohibition(self):
        source = "Lối đi bộ xếp không ổn định."
        context = self.context.analyze(source, "vi2en")
        self.assertEqual(self.context.validate_translation("The pedestrian walkway is stacked unstably.", context, "vi2en"), [])
        both = self.context.analyze(source + " Đừng nâng tải.", "vi2en")
        self.assertIn("missing_negation", self.context.validate_translation("The pedestrian walkway is unstable; raise the load.", both, "vi2en"))

    def test_unrelated_safety_output_cannot_pass_an_empty_field_check(self):
        context = self.context.analyze("Smoke!", "en2vi")
        self.assertTrue(context.safety_candidates)
        self.assertEqual(self.context.validate_translation("Hàn.", context, "en2vi"), ["unverified_safety_translation"])
        self.assertEqual(self.context.validate_translation(context.safety_candidates[0].translated_text, context, "en2vi"), [])

    def test_quantity_substring_or_sign_change_is_rejected(self):
        context = self.context.analyze("Di chuyển 500 cm.", "vi2en")
        self.assertIn("missing_number:500", self.context.validate_translation("Move 1500 cm.", context, "vi2en"))
        self.assertEqual(self.context.validate_translation("Move 500.0 cm.", context, "vi2en"), [])
        negative = self.context.analyze("Đo ở -20 cm.", "vi2en")
        self.assertIn("missing_number:-20", self.context.validate_translation("Measure at 20 cm.", negative, "vi2en"))

    def test_no_itn_retry_accepts_exact_model_output_not_fuzzy_or_conflicting_commands(self):
        self.assertTrue(self.context.accept_safety_alternative("5re.", "fire", "en2vi"))
        self.assertFalse(self.context.accept_safety_alternative("S.", "s", "en2vi"))
        self.assertFalse(self.context.accept_safety_alternative("pling object", "folding object", "en2vi"))
        self.assertFalse(self.context.accept_safety_alternative("Start the crane", "Stop the crane", "en2vi"))
        self.assertFalse(self.context.accept_safety_alternative("Touch the electrical cable", "Do not touch the electrical cable", "en2vi"))
        self.assertFalse(self.context.accept_safety_alternative("Stop the crane XC-03", "Stop the crane", "en2vi"))
        self.assertFalse(self.context.accept_safety_alternative("Fire at BM1", "Fire", "en2vi"))
        self.assertFalse(self.context.accept_safety_alternative("Stop the crane XC03", "Stop the crane", "en2vi"))
        self.assertFalse(self.context.accept_safety_alternative("Stop the crane at 20 cm", "Stop the crane", "en2vi"))

    def test_asr_retry_leaves_normal_primary_when_alternate_is_not_reviewed(self):
        pipeline = self.pipeline([])
        pipeline.direction = "en2vi"
        retry = mock.Mock(return_value={"text": "folding object", "lang": "en"})
        pipeline.asr = SimpleNamespace(transcribe_without_itn=retry)
        result = pipeline._refine_safety_asr(np.ones(160), {"text": "pling object", "lang": "en"})
        self.assertEqual(result["text"], "pling object")
        self.assertFalse(result["safety_retry"]["accepted"])

    def test_recognized_primary_does_not_run_second_asr(self):
        pipeline = self.pipeline([])
        pipeline.direction = "en2vi"
        retry = mock.Mock(side_effect=AssertionError("not needed"))
        pipeline.asr = SimpleNamespace(transcribe_without_itn=retry)
        pipeline._refine_safety_asr(np.ones(160), {"text": "Stop the crane!", "lang": "en"})
        retry.assert_not_called()

    def test_numeric_and_legacy_asr_use_correct_no_itn_prompts(self):
        for numeric, expected in [(False, "woitn"), (True, [15])]:
            adapter = SenseVoiceASR({})
            adapter._numeric_tag_api = numeric
            adapter.model = mock.Mock(return_value=["<|en|>fire"])
            with contextlib.redirect_stdout(io.StringIO()):
                adapter.transcribe(np.ones(160, dtype=np.float32), 16000, textnorm="woitn")
            self.assertEqual(adapter.model.call_args.kwargs["textnorm"], expected)

    def test_beams_are_ranked_local_and_deterministic(self):
        model = Translator({"translation": {}}, direction="en2vi", profile="edge", offline=True)
        model._backend = "onnxruntime_seq2seq_cpu"
        tokenizer = mock.Mock(return_value={"input_ids": "stub"})
        tokenizer.decode.side_effect = lambda seq, **_: seq
        model._tokenizer = tokenizer
        model._model = SimpleNamespace(generate=mock.Mock(return_value=["vi: Không siết sàn.", "vi: Không siết két."]))
        with contextlib.redirect_stdout(io.StringIO()):
            candidates = model.translate_candidates("Do not tighten the radiator", "en2vi")
        self.assertEqual(candidates, ["Không siết sàn.", "Không siết két."])
        kwargs = model._model.generate.call_args.kwargs
        self.assertEqual(kwargs["num_return_sequences"], 5)
        self.assertEqual(kwargs["num_beams"], 5)
        self.assertNotIn("do_sample", kwargs)


if __name__ == "__main__":
    unittest.main()
