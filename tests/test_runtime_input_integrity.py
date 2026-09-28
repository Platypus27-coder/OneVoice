"""Regression coverage for information lost between ASR, safety lookup and MT."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context.engine import ConstructionContextEngine
from contracts import ASRHypothesis, CommitKind, ContextResult
from safety.fast_path import SafetyFastPath, SafetyMatch
from streaming.semantic_commit import SemanticCommitController
from utils.text_normalizer import normalize_asr, normalize_en


class InputIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ConstructionContextEngine.from_data_dir(ROOT / "data/onevoice_construction_v2")

    def test_english_article_is_preserved_but_numeric_amperes_can_be_spoken(self):
        text = "There is a gas leak!"
        self.assertEqual(normalize_asr(text, "en"), text)
        self.assertEqual(normalize_en(text), text)
        self.assertEqual(normalize_en("Current is 25 A."), "Current is 25 amperes.")

    def test_asr_cleanup_preserves_codes_units_and_numbers(self):
        self.assertEqual(normalize_asr("  MÁY XC-03B CÓ ÁP SUẤT 3.5 MPa  "),
                         "MÁY XC-03B CÓ ÁP SUẤT 3.5 MPa")
        self.assertEqual(normalize_asr("MÁY XC-03B KHÔNG HOẠT ĐỘNG"),
                         "Máy XC-03B không hoạt động")
        self.assertEqual(normalize_asr("25 A and 20 kg", "en"), "25 A and 20 kg")

    def test_scaffold_safety_phrase_survives_source_cleanup(self):
        for text in ("GIÀN GIÁO KHÔNG AN TOÀN", "NÀY KHÔNG LÊN GIÀN GIÁO"):
            normalized = normalize_asr(text)
            self.assertNotIn("hệ thống", normalized)
            self.assertTrue(self.engine.analyze(normalized, "vi2en").safety_candidates)

    def test_equivalent_unit_spellings_preserve_measurement(self):
        context = self.engine.analyze("Di chuyển thêm 300 km.", "vi2en")
        self.assertEqual(self.engine.validate_translation("Move another 300 kilometers.", context, "vi2en"), [])
        self.assertIn("missing_unit:km", self.engine.validate_translation("Move another 300 kilograms.", context, "vi2en"))

    def test_article_or_substring_cannot_satisfy_unit_validation(self):
        context = self.engine.analyze("Measure 25 A.", "en2vi")
        self.assertIn("missing_unit:A", self.engine.validate_translation("25 là a.", context, "en2vi"))
        self.assertNotIn("t", self.engine.analyze("Don't start a motor.", "en2vi").entities.get("units", []))

    def test_material_store_is_a_valid_terminology_realization(self):
        context = self.engine.analyze("Qua kho vật tư kiểm tra gạch.", "vi2en")
        self.assertEqual(self.engine.validate_translation("Go to the material store and check the brick.", context, "vi2en"), [])

    def test_small_grammatical_safety_variants_are_equivalent(self):
        for text in ("Wear the safety helmet!", "Do not climb on the scaffold!"):
            self.assertTrue(self.engine.analyze(text, "en2vi").safety_candidates)

    def test_safety_matching_preserves_negation_action_and_numbers(self):
        for text in ("Do now energize the system!", "The scaffold is safe!", "Tear your safety helmet!"):
            self.assertFalse(self.engine.analyze(text, "en2vi").safety_candidates)
        self.assertTrue(self.engine.analyze("disconck the power immediately", "en2vi").safety_candidates)

    def test_ambiguous_corruption_has_no_safety_match(self):
        lookup = SafetyFastPath.__new__(SafetyFastPath)
        lookup._vi = {}
        lookup._en = {
            "check valve": SafetyMatch("S1", "Check valve", "Kiểm tra van", "CHECK", "high"),
            "check value": SafetyMatch("S2", "Check value", "Kiểm tra giá trị", "CHECK", "high"),
        }
        self.assertIsNone(lookup.match("check valye", "en2vi"))


class SafetyCommitIntegrityTests(unittest.TestCase):
    def test_equivalent_ids_and_temporary_asr_revision_do_not_repeat_action(self):
        controller = SemanticCommitController(normal_commit_policy="endpoint")
        def decide(text, key, endpoint=False):
            safety = SimpleNamespace(safety_id=key, intent="STOP", translated_text="Stop the forklift!")
            context = ContextResult(text, text, safety_candidates=[safety])
            return controller.decide(ASRHypothesis(text, text, "", "vi2en", 1., 2., endpoint=endpoint), context)
        self.assertEqual(decide("Dừng xe nâng", "S1").kind, CommitKind.WAIT)
        self.assertEqual(decide("Dừng xe nâng", "S1").kind, CommitKind.SAFETY)
        changed = ASRHypothesis("Ê dừng xe", "", "", "vi2en", 1., 2.)
        controller.decide(changed, ContextResult(changed.text, changed.text))
        self.assertEqual(decide("Ê dừng xe nâng", "S2", endpoint=True).kind, CommitKind.WAIT)
        controller.reset()
        self.assertEqual(decide("Ê dừng xe nâng", "S2", endpoint=True).kind, CommitKind.SAFETY)


if __name__ == "__main__":
    unittest.main()
