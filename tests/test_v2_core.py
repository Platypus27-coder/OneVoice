from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from audio.denoise import Denoiser
from asr.asr_manager import GIPFORMER_INT8_FILES, GIPFORMER_REVISION
from asr.sensevoice_asr import SenseVoiceASR
from context.engine import ConstructionContextEngine
from context.site_pack import SitePackError, SitePackLoader
from contracts import ASRHypothesis, AudioFrame, CommitKind
from evaluation.dataset_audit import audit_audio_manifest
from evaluation.real_site import audit_real_site_manifest, write_holdout_lock
from evaluation.metrics import cer, corpus_error_rate, wer
from evaluation.reporting import create_run_manifest
from runtime.preflight import ArtifactPreflightError, verify_artifacts
from runtime.release_policy import ReleasePolicyError, validate_release_config
from scripts.recover_v1_manifest import UNRECOVERABLE, recover
from scripts.build_benchmark_dashboard import build_dashboard
from scripts.analyze_mt_errors import analyze_report
from scripts.reconcile_manifest_splits import reconcile_rows
from scripts.prepare_sensevoice_finetune_data import prepare
from scripts.manage_training_checkpoint import check_checkpoint, quarantine_checkpoint
from scripts.benchmark_sensevoice_checkpoint import _load_partial_predictions, _write_partial_predictions
from scripts.benchmark_asr_v2 import load_partial_predictions as load_asr_partial_predictions
from scripts.benchmark_asr_v2 import write_partial_predictions as write_asr_partial_predictions
from scripts.export_sensevoice_checkpoint_onnx import copy_runtime_bundle
from scripts.finetune_gipformer_rnnt import configure_trainable_parameters
from scripts.stage_gipformer_training_audio import cache_target, read_rows
from scripts.reconcile_safety_audio import reconcile, sha256 as safety_sha256
from scripts.build_release_bundle import build_bundle
from scripts.verify_release_bundle import verify_static
from streaming.semantic_commit import (
    RollingHypothesisAssembler,
    SemanticCommitController,
    StablePrefixAligner,
)
from streaming.session import RollingUtteranceSession
from translation.mt_engine import Translator
from tts.tts_engine import TTSEngine


DATA = ROOT / "data" / "onevoice_construction_v2"


class MetricsTests(unittest.TestCase):
    def test_empty_prediction_is_full_error(self):
        self.assertEqual(wer("dừng máy ngay", ""), 1.0)

    def test_corpus_error_rate_aggregates_tokens(self):
        score = corpus_error_rate(["a b", "c d"], ["a b", "c x"])
        self.assertEqual(score, 0.25)
        self.assertGreater(cer("máy xúc", "máy"), 0)


class ContextAndSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ConstructionContextEngine.from_data_dir(DATA)

    def test_canonical_term_and_entities(self):
        result = self.engine.analyze("Kiểm tra bạc biên ở 20 cm bên trái.", "vi2en")
        self.assertTrue(result.canonical_mentions)
        self.assertIn("20", result.entities["numbers"])
        self.assertIn("cm", [unit.casefold() for unit in result.entities["units"]])
        self.assertEqual(result.intent, "REQUEST_INSPECTION")

    def test_safety_match_is_deterministic(self):
        result = self.engine.analyze("Dừng lại ngay!", "vi2en")
        self.assertEqual(len(result.safety_candidates), 1)
        self.assertTrue(result.safety_candidates[0].translated_text)

    def test_safety_match_recovers_single_asr_token_slip(self):
        result = self.engine.analyze("disconck the power immediately", "en2vi")
        self.assertEqual(len(result.safety_candidates), 1)
        self.assertIn("Ngắt điện ngay", result.safety_candidates[0].translated_text)

    def test_safety_reconciliation_requires_both_direction_audio(self):
        root = ROOT / "tests" / ".tmp" / "safety-reconcile"
        root.mkdir(parents=True, exist_ok=True)
        try:
            try:
                import soundfile as sf
            except ImportError:
                self.skipTest("soundfile is optional in the lightweight unit-test environment")

            csv_path = root / "safety.csv"
            fields = [
                "safety_id", "vi", "en", "fixed_translation_candidate",
                "review_status", "reviewer", "reviewed_at",
            ]
            rows = []
            for index in range(126):
                rows.append({
                    "safety_id": f"SAFE_{index:04d}", "vi": "Dừng lại", "en": "Stop",
                    "fixed_translation_candidate": "True", "review_status": "approved",
                    "reviewer": "Impact", "reviewed_at": "2026-08-26",
                })
            csv_path.write_text(
                ",".join(fields) + "\n" + "\n".join(
                    ",".join(row[field] for field in fields) for row in rows
                ) + "\n",
                encoding="utf-8",
            )
            # The default expected benchmark size is intentionally overridden for
            # this compact fixture; one audio direction is omitted to exercise the gate.
            wav = root / "SAFE_0000_vi2en.wav"
            sf.write(wav, np.ones(160, dtype=np.float32) * 0.1, 16000)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "schema_version": 2,
                "source_sha256": safety_sha256(csv_path),
                "approval_id": "impact-safety-v1",
                "entries": [{
                    "safety_id": "SAFE_0000", "direction": "vi2en",
                    "path": wav.name, "sample_rate": 16000,
                    "sha256": safety_sha256(wav),
                }],
            }), encoding="utf-8")
            report = reconcile(csv_path, manifest, expected_benchmark_rows=126)
            self.assertFalse(report["passed"])
            self.assertIn("SAFE_0000/en2vi", report["missing_audio_entries"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_release_bundle_inventory_is_direction_scoped_and_hash_locked(self):
        root = ROOT / "tests" / ".tmp" / "release-bundle"
        root.mkdir(parents=True, exist_ok=True)
        try:
            source = root / "models" / "model.bin"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"onevoice-artifact")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            artifact_manifest = root / "artifact_manifest.json"
            artifact_manifest.write_text(json.dumps({
                "schema_version": 2,
                "artifacts": [
                    {"name": "gipformer/tokens.txt", "path": str(source), "sha256": digest, "license": "MIT", "directions": ["vi2en"], "profiles": ["development"]},
                    {"name": "mt_en2vi/config.json", "path": str(source), "sha256": digest, "license": "OpenRAIL", "directions": ["en2vi"], "profiles": ["development"]},
                ],
            }), encoding="utf-8")
            receipt = build_bundle(artifact_manifest, root / "bundle", "vi2en")
            self.assertEqual(receipt["artifact_count"], 1)
            self.assertFalse(receipt["portable"])
            manifest = json.loads((root / "bundle" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["direction"], "vi2en")
            self.assertEqual(manifest["artifacts"][0]["directions"], ["vi2en"])
            checked = verify_static(root / "bundle", "vi2en")
            self.assertEqual(len(checked["checked"]), 1)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_site_pack_validation(self):
        with self.assertRaises(SitePackError):
            SitePackLoader.from_dict({"site_id": "only-one-field"})

    def test_site_pack_precedence_and_translation_memory(self):
        pack = SitePackLoader.from_dict(
            {
                "site_id": "site-1",
                "site_name": "Pilot",
                "project_type": "building",
                "local_terms": [
                    {
                        "canonical_id": "SITE_HELMET",
                        "vi": "mũ bảo hộ",
                        "en": "site hard hat",
                        "aliases": ["nón công trường"],
                    }
                ],
                "translation_memory": [
                    {"vi": "kiểm tra khu alpha", "en": "inspect alpha zone"}
                ],
            }
        )
        engine = ConstructionContextEngine.from_data_dir(DATA, site_pack=pack)
        context = engine.analyze("mũ bảo hộ", "vi2en")
        self.assertEqual(context.canonical_mentions[0].canonical_id, "SITE_HELMET")
        self.assertEqual(context.canonical_mentions[0].en_standard, "site hard hat")
        self.assertEqual(
            engine.analyze("kiểm tra khu alpha", "vi2en").translation_memory,
            "inspect alpha zone",
        )

    def test_site_pack_rejects_alias_collision(self):
        with self.assertRaises(SitePackError):
            SitePackLoader.from_dict(
                {
                    "site_id": "site-1",
                    "site_name": "Pilot",
                    "project_type": "building",
                    "local_terms": [
                        {"canonical_id": "A", "vi": "máy một", "en": "one", "aliases": ["máy"]},
                        {"canonical_id": "B", "vi": "máy hai", "en": "two", "aliases": ["máy"]},
                    ],
                }
            )

    def test_site_pack_trie_uses_longest_match_at_same_priority(self):
        pack = SitePackLoader.from_dict(
            {
                "site_id": "site-1",
                "site_name": "Pilot",
                "project_type": "building",
                "local_terms": [
                    {"canonical_id": "SHORT", "vi": "máy", "en": "machine"},
                    {"canonical_id": "LONG", "vi": "máy xúc", "en": "excavator"},
                ],
            }
        )
        engine = ConstructionContextEngine.from_data_dir(DATA, site_pack=pack)
        mentions = engine.analyze("máy xúc", "vi2en").canonical_mentions
        self.assertEqual([mention.canonical_id for mention in mentions], ["LONG"])

    def test_critical_field_validator(self):
        source = "Không nâng sang trái quá 20 cm"
        context = self.engine.analyze(source, "vi2en")
        errors = self.engine.validate_translation("Raise 20", context, "vi2en")
        self.assertIn("missing_unit:cm", errors)
        self.assertTrue(any(error.startswith("missing_direction:") for error in errors))
        self.assertIn("missing_negation", errors)

    def test_validator_does_not_treat_temporal_markers_as_directions(self):
        before = self.engine.analyze("Kiểm tra mũ bảo hộ trước khi tiếp tục.", "vi2en")
        after = self.engine.analyze("Kiểm tra lại móc an toàn sau khi xử lý.", "vi2en")
        self.assertFalse(
            any(
                error.startswith("missing_direction:")
                for error in self.engine.validate_translation(
                    "Check the safety helmet before continuing.", before, "vi2en"
                )
            )
        )
        self.assertFalse(
            any(
                error.startswith("missing_direction:")
                for error in self.engine.validate_translation(
                    "Recheck the snap hook after corrective action.", after, "vi2en"
                )
            )
        )

    def test_validator_does_not_treat_trailing_temporal_marker_as_direction(self):
        context = self.engine.analyze("Kiểm tra đồng hồ áp suất trước.", "vi2en")
        errors = self.engine.validate_translation(
            "Check the pressure gauge first.", context, "vi2en"
        )
        self.assertFalse(any(error.startswith("missing_direction:") for error in errors))

    def test_validator_keeps_explicit_spatial_direction(self):
        context = self.engine.analyze("Di chuyển tải về phía trước 2 m.", "vi2en")
        errors = self.engine.validate_translation("Move the load forward 2 m.", context, "vi2en")
        self.assertFalse(any(error.startswith("missing_direction:") for error in errors))

    def test_validator_accepts_grounded_for_grounding(self):
        context = self.engine.analyze("Dây pha chưa tiếp địa.", "vi2en")
        self.assertNotIn(
            "missing_term:C0115:grounding",
            self.engine.validate_translation("The live conductor is not grounded.", context, "vi2en"),
        )

    def test_validator_accepts_standard_technical_inflections(self):
        oil = self.engine.analyze("Bu lông bị rò dầu.", "vi2en")
        rebar = self.engine.analyze("Đầm cốt thép ngay.", "vi2en")
        self.assertNotIn(
            "missing_term:C0140:oil leak",
            self.engine.validate_translation("The bolt is leaking oil.", oil, "vi2en"),
        )
        self.assertNotIn(
            "missing_term:C0057:reinforcing steel",
            self.engine.validate_translation("Vibrate the rebar now.", rebar, "vi2en"),
        )

    def test_validator_accepts_keep_clear_for_clearance_instruction(self):
        context = self.engine.analyze("Giữ nguyên khoảng cách, chưa được di chuyển.", "vi2en")
        self.assertNotIn(
            "missing_term:C0209:distance",
            self.engine.validate_translation("Keep clear of it; do not move it yet.", context, "vi2en"),
        )

    def test_validator_does_not_treat_order_after_reverse_alarm_as_direction(self):
        context = self.engine.analyze("Nhờ dừng còi lùi trước.", "vi2en")
        errors = self.engine.validate_translation("Please stop the reverse alarm first.", context, "vi2en")
        self.assertFalse(any(error.startswith("missing_direction:") for error in errors))

    def test_en_context_disambiguates_article_current_and_tie_down(self):
        generic = self.engine.analyze("What is the current weight of a pallet?", "en2vi")
        tie_down = self.engine.analyze("Use the cargo tie-down.", "en2vi")
        moved_down = self.engine.analyze("Move the crawler excavator down.", "en2vi")
        self.assertNotIn("a", generic.entities.get("units", []))
        self.assertNotIn("C0120", [item.canonical_id for item in generic.canonical_mentions])
        self.assertNotIn("down", tie_down.entities.get("directions", []))
        self.assertIn("down", moved_down.entities.get("directions", []))

    def test_en_validator_accepts_construction_term_aliases(self):
        context = self.engine.analyze("Barricade the equipment swing area.", "en2vi")
        errors = self.engine.validate_translation("Rào quanh khu quay máy.", context, "en2vi")
        self.assertNotIn("missing_term:C0012:rào chắn", errors)
        self.assertNotIn("missing_term:C0103:khu vực quay máy", errors)

    def test_validator_handles_completion_question_and_unsafe_predicate(self):
        question = self.engine.analyze("Khu vực cấm đã được kiểm tra chưa?", "vi2en")
        unsafe = self.engine.analyze("Giàn giáo không an toàn!", "vi2en")
        self.assertNotIn(
            "missing_negation",
            self.engine.validate_translation("Has the restricted area been checked?", question, "vi2en"),
        )
        punctuationless_question = self.engine.analyze("Khu v?c c?m ?? ???c ki?m tra ch?a", "vi2en")
        self.assertNotIn(
            "missing_negation",
            self.engine.validate_translation("Has the restricted area been checked?", punctuationless_question, "vi2en"),
        )
        self.assertNotIn(
            "missing_negation",
            self.engine.validate_translation("The scaffold is unsafe!", unsafe, "vi2en"),
        )

    def test_reviewed_safety_fast_path_is_validated_as_source_of_truth(self):
        context = self.engine.analyze("Dừng lại ngay!", "vi2en")
        self.assertTrue(context.safety_candidates)
        self.assertEqual(
            self.engine.validate_translation(
                context.safety_candidates[0].translated_text, context, "vi2en"
            ),
            [],
        )


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.engine = ConstructionContextEngine.from_data_dir(DATA)

    def _hypothesis(self, text, stable=None, endpoint=False):
        return ASRHypothesis(
            text=text,
            stable_prefix=stable if stable is not None else text,
            unstable_tail="",
            direction="vi2en",
            started_at=time.perf_counter(),
            updated_at=time.perf_counter(),
            endpoint=endpoint,
        )

    def test_stable_prefix(self):
        aligner = StablePrefixAligner()
        self.assertEqual(aligner.update("dừng máy")[0], "")
        stable, tail = aligner.update("dừng máy ngay")
        self.assertEqual(stable, "dừng máy")
        self.assertEqual(tail, "ngay")

    def test_rolling_hypothesis_overlap(self):
        assembler = RollingHypothesisAssembler()
        self.assertEqual(assembler.update("kiểm tra máy xúc"), "kiểm tra máy xúc")
        self.assertEqual(
            assembler.update("máy xúc số ba"), "kiểm tra máy xúc số ba"
        )

    def test_number_without_unit_waits(self):
        controller = SemanticCommitController()
        text = "nâng lên 20"
        decision = controller.decide(self._hypothesis(text), self.engine.analyze(text, "vi2en"))
        self.assertEqual(decision.kind, CommitKind.WAIT)

    def test_unstable_unit_does_not_release_stable_number(self):
        controller = SemanticCommitController()
        full = "nâng lên 20 cm"
        hypothesis = self._hypothesis(full, stable="nâng lên 20")
        decision = controller.decide(hypothesis, self.engine.analyze(full, "vi2en"))
        self.assertEqual(decision.kind, CommitKind.WAIT)

    def test_open_negation_waits(self):
        controller = SemanticCommitController()
        text = "không"
        decision = controller.decide(self._hypothesis(text), self.engine.analyze(text, "vi2en"))
        self.assertEqual(decision.kind, CommitKind.WAIT)

    def test_safety_requires_two_confirmations(self):
        controller = SemanticCommitController(safety_confirmations=2)
        text = "Dừng lại ngay!"
        context = self.engine.analyze(text, "vi2en")
        first = controller.decide(self._hypothesis(text), context)
        second = controller.decide(self._hypothesis(text), context)
        self.assertEqual(first.kind, CommitKind.WAIT)
        self.assertEqual(second.kind, CommitKind.SAFETY)

    def test_safety_endpoint_does_not_duplicate_confirmed_audio(self):
        controller = SemanticCommitController(safety_confirmations=2)
        text = "Dừng lại ngay!"
        context = self.engine.analyze(text, "vi2en")
        self.assertEqual(
            controller.decide(self._hypothesis(text), context).kind,
            CommitKind.WAIT,
        )
        self.assertEqual(
            controller.decide(self._hypothesis(text), context).kind,
            CommitKind.SAFETY,
        )
        endpoint = controller.decide(
            self._hypothesis(text, endpoint=True), context
        )
        self.assertEqual(endpoint.kind, CommitKind.WAIT)
        self.assertEqual(endpoint.reason, "safety_already_committed")

    def test_pending_safety_does_not_commit_normal_prefix(self):
        controller = SemanticCommitController(safety_confirmations=2)
        text = "Dừng lại ngay!"
        decision = controller.decide(
            self._hypothesis(text, stable="Dừng lại"),
            self.engine.analyze(text, "vi2en"),
        )
        self.assertEqual(decision.kind, CommitKind.WAIT)

    def test_endpoint_commits_once(self):
        controller = SemanticCommitController()
        text = "kiểm tra máy xúc"
        context = self.engine.analyze(text, "vi2en")
        first = controller.decide(self._hypothesis(text, endpoint=True), context)
        second = controller.decide(self._hypothesis(text, endpoint=True), context)
        self.assertEqual(first.kind, CommitKind.NORMAL)
        self.assertEqual(second.kind, CommitKind.WAIT)

    def test_progressive_commits_do_not_duplicate_prefix(self):
        controller = SemanticCommitController()
        emitted = []
        for text, stable, endpoint in (
            ("kiểm tra máy", "", False),
            ("kiểm tra máy xúc", "kiểm tra máy", False),
            ("kiểm tra máy xúc ngay", "kiểm tra máy xúc", False),
            ("kiểm tra máy xúc ngay", "kiểm tra máy xúc ngay", True),
        ):
            decision = controller.decide(
                self._hypothesis(text, stable=stable, endpoint=endpoint),
                self.engine.analyze(text, "vi2en"),
            )
            if decision.kind != CommitKind.WAIT:
                emitted.extend(decision.text.split())
        self.assertEqual(emitted, "kiểm tra máy xúc ngay".split())

    def test_rolling_session_emits_partial_and_endpoint(self):
        session = RollingUtteranceSession(
            {
                "sample_rate": 16000,
                "chunk_size": 512,
                "vad_energy_threshold": 0.01,
                "vad_min_speech_ms": 64,
                "vad_pre_roll_ms": 32,
                "vad_endpoint_ms": 64,
                "max_utterance_ms": 1000,
            },
            {"rolling_stride_ms": 64, "rolling_window_ms": 64},
        )
        events = []
        for sequence in range(1, 5):
            frame = AudioFrame(
                samples=np.ones(512, dtype=np.float32) * 0.1,
                sample_rate=16000,
                sequence=sequence,
                captured_at=float(sequence),
            )
            event = session.accept(frame, denoise_ms=1.0)
            if event:
                events.append(event)
        for sequence in range(5, 7):
            frame = AudioFrame(
                samples=np.zeros(512, dtype=np.float32),
                sample_rate=16000,
                sequence=sequence,
                captured_at=float(sequence),
            )
            event = session.accept(frame, denoise_ms=1.0)
            if event:
                events.append(event)
        self.assertTrue(any(not event.endpoint for event in events))
        partial = next(event for event in events if not event.endpoint)
        self.assertLessEqual(len(partial.audio), 2 * 512)
        self.assertTrue(events[-1].endpoint)

    def test_rolling_session_rejects_invalid_frame_contract(self):
        session = RollingUtteranceSession(
            {"sample_rate": 16000, "chunk_size": 512},
            {},
        )
        with self.assertRaisesRegex(ValueError, "sample rate"):
            session.accept(
                AudioFrame(np.zeros(512, dtype=np.float32), 8000, 1, 1.0)
            )
        with self.assertRaisesRegex(ValueError, "expected 512"):
            session.accept(
                AudioFrame(np.zeros(256, dtype=np.float32), 16000, 1, 1.0)
            )
        session.accept(AudioFrame(np.zeros(512, dtype=np.float32), 16000, 2, 2.0))
        with self.assertRaisesRegex(ValueError, "must increase"):
            session.accept(AudioFrame(np.zeros(512, dtype=np.float32), 16000, 2, 3.0))


class RuntimeSafetyTests(unittest.TestCase):
    def test_sensevoice_onnx_export_bundle_excludes_pytorch_weights(self):
        root = ROOT / "tests" / ".tmp" / "sensevoice-onnx-bundle"
        try:
            stage = root / "stage"
            output = root / "output"
            stage.mkdir(parents=True)
            for name in ("model.onnx", "config.yaml", "am.mvn", "chn_jpn_yue_eng_ko_spectok.bpe.model"):
                (stage / name).write_bytes(b"artifact")
            (stage / "model.pt").write_bytes(b"base weights")
            (stage / "model_quant.onnx").write_bytes(b"stale base quantization")
            copied = copy_runtime_bundle(stage, output)
            self.assertTrue((output / "model.onnx").is_file())
            self.assertFalse((output / "model.pt").exists())
            self.assertFalse((output / "model_quant.onnx").exists())
            self.assertEqual(len(copied), 4)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_sensevoice_partial_predictions_are_resumable(self):
        root = ROOT / "tests" / ".tmp" / "sensevoice-resume"
        try:
            root.mkdir(parents=True, exist_ok=True)
            partial = root / "predictions.partial.jsonl"
            expected = [
                {"audio": "example_clean.wav", "prediction": "secure the load"},
                {"audio": "example_noisy.wav", "prediction": "stop the machine"},
            ]
            _write_partial_predictions(partial, expected)
            self.assertEqual(_load_partial_predictions(partial), expected)
            self.assertFalse(partial.with_suffix(".jsonl.tmp").exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_asr_benchmark_partial_predictions_are_resumable(self):
        root = ROOT / "tests" / ".tmp" / "asr-resume"
        try:
            root.mkdir(parents=True, exist_ok=True)
            partial = root / "predictions.partial.jsonl"
            expected = [{"audio": "example.wav", "prediction": "secure the load"}]
            write_asr_partial_predictions(partial, expected)
            self.assertEqual(load_asr_partial_predictions(partial), expected)
            self.assertFalse(partial.with_suffix(".jsonl.tmp").exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_run_manifest_serializes_path_metadata(self):
        root = ROOT / "tests" / ".tmp" / "manifest-path-metadata"
        try:
            root.mkdir(parents=True, exist_ok=True)
            source = root / "source.txt"
            source.write_text("onevoice", encoding="utf-8")
            output = root / "run_manifest.json"
            create_run_manifest(
                output,
                "test",
                inputs=[source],
                metadata={"checkpoint": root / "checkpoint.pt", "report_dir": root},
            )
            recorded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(recorded["metadata"]["checkpoint"], str(root / "checkpoint.pt"))
            self.assertEqual(recorded["metadata"]["report_dir"], str(root))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_mt_error_analysis_counts_critical_validation_errors(self):
        report = ROOT / "tests" / ".tmp" / "mt-predictions.csv"
        try:
            report.write_text(
                "source,reference,prediction,route,critical_fields_valid,validation_errors\n"
                "Dừng máy,Stop machine,Stop machine,raw,True,\n"
                "Không nâng,Do not raise,Raise,raw,False,missing_negation\n",
                encoding="utf-8",
            )
            summary = analyze_report(report, top=1)
            self.assertEqual(summary["samples"], 2)
            self.assertEqual(summary["critical_invalid"], 1)
            self.assertEqual(summary["validation_error_counts"], {"missing_negation": 1})
            self.assertEqual(summary["review_examples"][0]["prediction"], "Raise")
        finally:
            report.unlink(missing_ok=True)

    def test_sensevoice_adapter_passes_pcm_array_not_path_list(self):
        class FakeSenseVoice:
            def __call__(self, waveform, **kwargs):
                self.waveform = waveform
                self.kwargs = kwargs
                return ["<|en|><|NEUTRAL|><|Speech|>secure the load"]

        adapter = SenseVoiceASR({})
        fake_model = FakeSenseVoice()
        adapter.model = fake_model

        result = adapter.transcribe(np.array([0.1, -0.2], dtype=np.float32), 16000)

        self.assertIsInstance(fake_model.waveform, np.ndarray)
        self.assertEqual(fake_model.waveform.dtype, np.float32)
        self.assertEqual(fake_model.kwargs, {"language": "en", "textnorm": "withitn"})
        self.assertEqual(result["text"], "secure the load")

    def test_sensevoice_quantization_is_explicitly_configurable(self):
        fp32 = SenseVoiceASR({"sensevoice": {"model_path": "candidate", "quantize": False}})
        default = SenseVoiceASR({"sensevoice": {"model_path": "baseline"}})
        self.assertFalse(fp32.quantize)
        self.assertFalse(default.quantize)

    def test_sensevoice_new_onnx_api_defaults_to_rank_one_prompt_tags(self):
        class FakeSenseVoice:
            def __call__(self, waveform, **kwargs):
                self.kwargs = kwargs
                return ["<|en|><|NEUTRAL|><|Speech|>secure the load"]

        adapter = SenseVoiceASR({"sensevoice": {"quantize": False}})
        adapter._numeric_tag_api = True
        fake_model = FakeSenseVoice()
        adapter.model = fake_model
        adapter.transcribe(np.array([0.1], dtype=np.float32), 16000)
        self.assertEqual(fake_model.kwargs, {"language": [4], "textnorm": [14]})

    def test_sensevoice_new_onnx_int8_api_uses_rank_one_prompt_tags(self):
        class FakeSenseVoice:
            def __call__(self, waveform, **kwargs):
                self.kwargs = kwargs
                return ["<|en|><|NEUTRAL|><|Speech|>secure the load"]

        adapter = SenseVoiceASR({"sensevoice": {"quantize": True}})
        adapter._numeric_tag_api = True
        fake_model = FakeSenseVoice()
        adapter.model = fake_model
        adapter.transcribe(np.array([0.1], dtype=np.float32), 16000)
        self.assertEqual(fake_model.kwargs, {"language": [4], "textnorm": [14]})

    def test_development_gtts_fallback_is_used_for_both_output_languages(self):
        config = {
            "audio": {"sample_rate": 16000},
            "tts": {"en_speed": 1.0},
            "profiles": {"development": {"tts_tier": "premium"}},
        }
        engine = TTSEngine(config, profile="development", offline=False)
        engine._en_tts_engine = "gtts"
        calls = []
        engine._synthesize_gtts = lambda text, language: (  # type: ignore[method-assign]
            calls.append((text, language)) or (np.array([0.25], dtype=np.float32), 22050)
        )
        english, english_sr = engine.synthesize("Stop now", "vi2en")
        vietnamese, vietnamese_sr = engine.synthesize("Dừng lại", "en2vi")
        self.assertEqual(calls, [("Stop now", "en"), ("Dừng lại", "vi")])
        self.assertEqual((english_sr, vietnamese_sr), (22050, 22050))
        self.assertFalse(engine.is_silence(english))
        self.assertFalse(engine.is_silence(vietnamese))

    def test_offline_demo_tts_selects_local_system_voice_instead_of_omnivoice(self):
        from unittest import mock

        config = {
            "audio": {"sample_rate": 16000},
            "tts": {"offline_engine": "pyttsx3"},
            "profiles": {"development": {"tts_tier": "premium"}},
        }
        engine = TTSEngine(config, profile="development", offline=True)
        calls = []
        engine._load_edge_vi_tts = lambda: calls.append("local")
        engine._load_omnivoice = lambda: calls.append("omnivoice")
        with mock.patch("builtins.print"):
            engine.load(direction="en2vi")
        self.assertEqual(calls, ["local"])

    def test_english_tts_uses_native_espeak_after_pyttsx3_silence(self):
        from types import SimpleNamespace
        from unittest import mock

        config = {
            "audio": {"sample_rate": 16000},
            "tts": {"en_speed": 1.0},
            "profiles": {"development": {"tts_tier": "edge"}},
        }
        engine = TTSEngine(config, profile="development", offline=True)
        engine._en_tts_executable = "espeak-ng"
        engine._synthesize_espeak_en = lambda text: (  # type: ignore[method-assign]
            np.array([0.2, -0.2], dtype=np.float32),
            22050,
        )
        fake_engine = SimpleNamespace(
            setProperty=lambda *args: None,
            save_to_file=lambda *args: None,
            runAndWait=lambda: None,
        )
        fake_soundfile = SimpleNamespace(
            read=lambda *args, **kwargs: (np.zeros(32, dtype=np.float32), 22050)
        )
        with mock.patch.dict(sys.modules, {"pyttsx3": SimpleNamespace(init=lambda: fake_engine), "soundfile": fake_soundfile}), \
             mock.patch("builtins.print"):
            audio, sample_rate = engine.synthesize_en("Check the load")
        self.assertEqual(sample_rate, 22050)
        self.assertFalse(engine.is_silence(audio))
        self.assertEqual(engine.engine_name("vi2en"), "espeak-ng-offline-demo")

    def test_split_reconciliation_preserves_test_holdout_and_pattern_groups(self):
        rows, report = reconcile_rows(
            [
                {"language": "en", "text": "Stop the machine", "frame_pattern_id": "A", "split": "train"},
                {"language": "en", "text": "check the machine", "frame_pattern_id": "A", "split": "train"},
                {"language": "en", "text": "stop   the machine", "frame_pattern_id": "B", "split": "test"},
                {"language": "vi", "text": "dừng máy", "split": "train"},
            ],
            "en",
        )
        self.assertEqual([row["split"] for row in rows[:3]], ["test", "test", "test"])
        self.assertEqual(rows[0]["source_split"], "train")
        self.assertEqual(report["components_reconciled"], 1)

    def test_benchmark_dashboard_keeps_missing_metrics_visible(self):
        report_root = ROOT / "tests" / "_tmp_dashboard"
        shutil.rmtree(report_root, ignore_errors=True)
        report_dir = report_root / "mt" / "candidate" / "test"
        report_dir.mkdir(parents=True)
        (report_dir / "aggregate.json").write_text(
            json.dumps({"samples": 2, "direction": "vi2en", "suite": "test", "reference_wer": 0.2}),
            encoding="utf-8",
        )
        (report_dir / "run_manifest.json").write_text(
            json.dumps({"command": "benchmark_mt_v2", "metadata": {"model_reference": {"source": "candidate"}}}),
            encoding="utf-8",
        )
        dashboard = report_root / "dashboard.md"
        rows = build_dashboard(report_root, dashboard)
        content = dashboard.read_text(encoding="utf-8")
        self.assertEqual(rows[0]["model"], "candidate")
        self.assertIn("0.2000", content)
        self.assertIn("—", content)
        shutil.rmtree(report_root)

    def test_mt_registry_is_direction_specific_and_release_is_default(self):
        config = {
            "translation": {
                "max_length": 128,
                "directions": {
                    "vi2en": {
                        "release_model": "release-vi-en",
                        "local_model_dir": "models/mt/vi2en",
                        "edge_model_dir": "models/mt/vi2en_ort",
                    },
                    "en2vi": {
                        "release_model": "release-en-vi",
                        "local_model_dir": "models/mt/en2vi",
                        "edge_model_dir": "models/mt/en2vi_ort",
                    },
                },
            }
        }
        release = Translator(config, direction="vi2en")
        explicit = Translator(
            config, direction="vi2en", model_source="diagnostic-vi-en", model_revision="abc123"
        )
        reverse = Translator(config, direction="en2vi")
        self.assertEqual(release.model_reference["source"], "release-vi-en")
        self.assertEqual(explicit.model_reference["source"], "diagnostic-vi-en")
        self.assertEqual(explicit.model_reference["revision"], "abc123")
        self.assertEqual(reverse.model_reference["source"], "release-en-vi")
        with self.assertRaises(ValueError):
            release.translate("xin chào", "en2vi")

    def test_release_policy_rejects_failed_asr_candidates_and_int8(self):
        valid = {
            "asr": {"gipformer_model_dir": "models/gipformer"},
            "sensevoice": {
                "model_path": "models/sensevoice_en_construction_v1_onnx_fp32",
                "quantize": False,
            },
        }
        validate_release_config(valid, "vi2en")
        validate_release_config(valid, "en2vi")

        rejected_gipformer = json.loads(json.dumps(valid))
        rejected_gipformer["asr"]["gipformer_model_dir"] = (
            "models/gipformer_vi_construction_icefall_ft_v4/best"
        )
        with self.assertRaisesRegex(ReleasePolicyError, "Rejected GIPFormer"):
            validate_release_config(rejected_gipformer, "vi2en")

        rejected_sensevoice = json.loads(json.dumps(valid))
        rejected_sensevoice["sensevoice"]["quantize"] = True
        with self.assertRaisesRegex(ReleasePolicyError, "INT8"):
            validate_release_config(rejected_sensevoice, "en2vi")

    def test_checked_in_config_selects_released_models_without_remote_fallback(self):
        import yaml

        config = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
        directions = config["translation"]["directions"]
        self.assertEqual(
            directions["vi2en"]["release_model"],
            "platypus123/onevoice-envit5-vi-en",
        )
        self.assertEqual(
            directions["en2vi"]["release_model"],
            "platypus123/onevoice-envit5-en-vi",
        )
        self.assertNotIn("candidate_model", directions["vi2en"])
        self.assertFalse(config["sensevoice"]["quantize"])
        self.assertFalse(config["sensevoice"]["allow_remote_fallback"])
        self.assertNotIn("remote_model", config["sensevoice"])
        validate_release_config(config, "vi2en")
        validate_release_config(config, "en2vi")

    def test_gipformer_download_uses_pinned_current_artifact_names(self):
        self.assertEqual(GIPFORMER_REVISION, "29621ec87ffec8fde06be25ed2150d4a1f41dbc9")
        self.assertEqual(
            GIPFORMER_INT8_FILES,
            {
                "encoder": "encoder.int8.onnx",
                "decoder": "decoder.int8.onnx",
                "joiner": "joiner.int8.onnx",
                "tokens": "tokens.txt",
            },
        )

    def test_parallel_physical_audio_audit(self):
        import types
        from unittest import mock

        root = ROOT / "tests" / ".tmp" / "physical-audit"
        manifest = root / "manifest.jsonl"
        try:
            (root / "clean").mkdir(parents=True, exist_ok=True)
            (root / "noisy").mkdir(parents=True, exist_ok=True)
            (root / "clean" / "u1_clean.wav").touch()
            (root / "noisy" / "u1_n01.wav").touch()
            manifest.write_text(
                json.dumps(
                    {
                        "audio": "u1_n01.wav",
                        "clean_audio": "u1_clean.wav",
                        "text": "dá»«ng láº¡i",
                        "translation": "stop",
                        "split": "test",
                        "speaker_id": "speaker-1",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            fake_soundfile = types.SimpleNamespace(
                info=lambda path: types.SimpleNamespace(
                    samplerate=16000, frames=1600, channels=1
                )
            )
            with mock.patch.dict(sys.modules, {"soundfile": fake_soundfile}):
                report = audit_audio_manifest(
                    manifest,
                    physical=True,
                    expected_clean=1,
                    expected_noisy=1,
                    workers=2,
                    progress_every=1,
                )
            self.assertTrue(report["passed"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_recover_v1_manifest_from_legacy_filenames(self):
        root = ROOT / "tests" / ".tmp" / "recover-v1"
        clean = root / "clean"
        noisy = root / "noisy"
        metadata = root / "utterances.csv"
        try:
            clean.mkdir(parents=True, exist_ok=True)
            noisy.mkdir(parents=True, exist_ok=True)
            (clean / "OV2_000001_clean.wav").touch()
            (noisy / "OV2_000001_n01.wav").touch()
            metadata.write_text(
                "utterance_id,pair_id,frame_pattern_id,split,domain,intent,risk_level,vi,en\n"
                "OV2_000001,P1,F1,test,safety,STOP,critical,Dá»«ng láº¡i,Stop\n",
                encoding="utf-8",
            )
            entries, report = recover(root, metadata)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["clean_audio"], "OV2_000001_clean.wav")
            self.assertEqual(entries[0]["speaker_id"], UNRECOVERABLE)
            self.assertEqual(report["status"], "PARTIAL")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_passthrough_denoiser(self):
        denoiser = Denoiser({"backend": "passthrough"})
        denoiser.load()
        audio = np.array([[0.1, 0.3], [-0.1, 0.1]], dtype=np.float32)
        result = denoiser.process(audio, 16000)
        np.testing.assert_allclose(result, [0.2, 0.0])

    def test_artifact_preflight_hash(self):
        root = ROOT / "tests" / ".tmp"
        artifact = root / "model.bin"
        manifest = root / "artifact_manifest.json"
        try:
            artifact.write_bytes(b"model")
            digest = hashlib.sha256(b"model").hexdigest()
            manifest.write_text(
                json.dumps(
                    {
                        "artifacts": [
                            {
                                "name": "model",
                                "path": "model.bin",
                                "sha256": digest,
                                "directions": ["vi2en"],
                                "profiles": ["edge"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            result = verify_artifacts(manifest, "vi2en", "edge")
            self.assertEqual(result["checked"], ["model"])
            artifact.write_bytes(b"changed")
            with self.assertRaises(ArtifactPreflightError):
                verify_artifacts(manifest, "vi2en", "edge")
        finally:
            artifact.unlink(missing_ok=True)
            manifest.unlink(missing_ok=True)

    def test_schema_v2_preflight_checks_sample_rate_and_license(self):
        root = ROOT / "tests" / ".tmp"
        artifact = root / "model-v2.bin"
        manifest = root / "artifact-v2.json"
        try:
            artifact.write_bytes(b"model-v2")
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "sample_rates": [16000],
                        "required_backends": [],
                        "artifacts": [
                            {
                                "name": "model",
                                "path": artifact.name,
                                "sha256": hashlib.sha256(b"model-v2").hexdigest(),
                                "license": "Apache-2.0",
                                "directions": ["vi2en"],
                                "profiles": ["edge"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(verify_artifacts(manifest, "vi2en", "edge", 16000)["checked"])
            with self.assertRaises(ArtifactPreflightError):
                verify_artifacts(manifest, "vi2en", "edge", 48000)
        finally:
            artifact.unlink(missing_ok=True)
            manifest.unlink(missing_ok=True)

    def test_release_lock_materializes_model_and_safety_provenance(self):
        root = ROOT / "tests" / ".tmp" / "release-lock"
        shutil.rmtree(root, ignore_errors=True)
        try:
            root.mkdir(parents=True)
            vi_model = root / "vi.bin"
            en_model = root / "en.bin"
            safety_csv = root / "safety.csv"
            safety_manifest = root / "safety-manifest.json"
            artifact_manifest = root / "artifact-manifest.json"
            release_lock = root / "release_lock_v2.json"
            vi_model.write_bytes(b"vi")
            en_model.write_bytes(b"en")
            safety_csv.write_text("safety_id,review_status\nS1,approved\n", encoding="utf-8")
            source_hash = hashlib.sha256(safety_csv.read_bytes()).hexdigest()
            safety_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "approval_id": "review-v1",
                        "source_sha256": source_hash,
                        "entries": [],
                    }
                ),
                encoding="utf-8",
            )
            artifact_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "sample_rates": [16000],
                        "required_backends": [],
                        "artifacts": [
                            {
                                "name": "vi/model.bin",
                                "path": vi_model.name,
                                "sha256": hashlib.sha256(vi_model.read_bytes()).hexdigest(),
                                "license": "MIT",
                                "directions": ["vi2en"],
                                "profiles": ["development"],
                            },
                            {
                                "name": "en/model.bin",
                                "path": en_model.name,
                                "sha256": hashlib.sha256(en_model.read_bytes()).hexdigest(),
                                "license": "Apache-2.0",
                                "directions": ["en2vi"],
                                "profiles": ["development"],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_release_lock.py"),
                    "--artifact-manifest",
                    str(artifact_manifest),
                    "--output",
                    str(release_lock),
                    "--model",
                    "vi",
                    "org/vi",
                    "a" * 40,
                    "MIT",
                    "vi2en",
                    "vi",
                    "--model",
                    "en",
                    "org/en",
                    "b" * 40,
                    "Apache-2.0",
                    "en2vi",
                    "en",
                    "--safety-source",
                    str(safety_csv),
                    "--safety-manifest",
                    str(safety_manifest),
                    "--safety-review-revision",
                    "review-v1",
                ],
                check=True,
            )
            payload = json.loads(release_lock.read_text(encoding="utf-8"))
            self.assertEqual(payload["manifest_kind"], "onevoice.release_lock")
            self.assertEqual(payload["safety_provenance"]["source_sha256"], source_hash)
            self.assertEqual(len(payload["models"]), 2)
            self.assertEqual(
                verify_artifacts(release_lock, "vi2en", "development", 16000)["checked"],
                ["vi/model.bin"],
            )
            safety_csv.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ArtifactPreflightError, "safety_provenance"):
                verify_artifacts(release_lock, "vi2en", "development", 16000)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_real_site_group_isolation_and_holdout_lock(self):
        root = ROOT / "tests" / ".tmp"
        manifest = root / "real-site.jsonl"
        lock = root / "holdout-lock.json"
        base = {
            "audio": "audio.wav",
            "language": "vi",
            "transcript": "dừng máy",
            "translation": "stop the machine",
            "site_id": "site-a",
            "domain": "safety_general",
            "intent": "STOP_WORK",
            "risk_level": "critical",
            "consent_recorded": True,
        }
        rows = [
            {**base, "utterance_id": "u1", "session_id": "s1", "speaker_id": "p1", "split": "train"},
            {**base, "utterance_id": "u2", "session_id": "s2", "speaker_id": "p2", "split": "test"},
        ]
        try:
            manifest.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )
            write_holdout_lock(manifest, lock)
            report = audit_real_site_manifest(manifest, physical=False, holdout_lock_path=lock)
            self.assertTrue(report["passed"])
            rows[1]["translation"] = "changed"
            manifest.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )
            self.assertFalse(
                audit_real_site_manifest(manifest, physical=False, holdout_lock_path=lock)["passed"]
            )
        finally:
            manifest.unlink(missing_ok=True)
            lock.unlink(missing_ok=True)

    def test_logical_manifest_audit(self):
        manifest = ROOT / "tests" / ".tmp" / "manifest.jsonl"
        try:
            row = {
                "audio": "a_n01.wav",
                "clean_audio": "a_clean.wav",
                "text": "dừng máy",
                "translation": "stop the machine",
                "split": "test",
                "speaker_id": "speaker-1",
                "noise_type": "wind",
                "snr_db": 10,
            }
            manifest.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            report = audit_audio_manifest(
                manifest, physical=False, expected_noisy=1, expected_clean=1
            )
            self.assertTrue(report["passed"])
        finally:
            manifest.unlink(missing_ok=True)

class SenseVoicePreparationTests(unittest.TestCase):
    def test_preparation_uses_only_train_and_dev_and_deduplicates_clean(self):
        root = ROOT / "tests" / ".tmp" / "sensevoice-preparation"
        root.mkdir(parents=True, exist_ok=True)
        manifest = root / "manifest.jsonl"
        try:
            rows = [
                {"utterance_id": "u1", "language": "en", "split": "train", "text": "Wear a helmet.", "clean_audio": "clean/u1.wav", "noisy_audio": "noisy/u1_n01.wav", "duration_s": 1.23},
                {"utterance_id": "u1", "language": "en", "split": "train", "text": "Wear a helmet.", "clean_audio": "clean/u1.wav", "noisy_audio": "noisy/u1_n02.wav", "duration_s": 1.23},
                {"utterance_id": "u2", "language": "en", "split": "dev", "text": "Stop the crane.", "clean_audio": "clean/u2.wav", "noisy_audio": "noisy/u2_n01.wav", "duration_s": 2.0},
                {"utterance_id": "u3", "language": "en", "split": "test", "text": "Never train on me.", "clean_audio": "clean/u3.wav", "noisy_audio": "noisy/u3_n01.wav", "duration_s": 1.0},
            ]
            manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            report = prepare(manifest, root / "prepared", lambda text: len(text.split()))
            self.assertEqual(report["splits"]["train"]["records"], 3)
            self.assertEqual(report["splits"]["dev"]["records"], 2)
            self.assertFalse(report["test_split_included"])
            train_records = [json.loads(line) for line in (root / "prepared/train.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(train_records[0]["source_len"], 123)
            self.assertEqual(train_records[0]["text_language"], "<|en|>")
            self.assertTrue(train_records[0]["source"].replace("\\", "/").endswith("clean/clean/u1.wav"))
            self.assertNotIn("Never train", (root / "prepared/train.jsonl").read_text(encoding="utf-8"))
        finally:
            shutil.rmtree(root, ignore_errors=True)


class TrainingCheckpointTests(unittest.TestCase):
    def test_corrupt_resume_checkpoint_is_quarantined_without_deleting_it(self):
        root = ROOT / "tests" / ".tmp" / "checkpoint-quarantine"
        root.mkdir(parents=True, exist_ok=True)
        checkpoint = root / "model.pt"
        try:
            checkpoint.write_bytes(b"partial checkpoint")
            result = check_checkpoint(root, lambda _: (_ for _ in ()).throw(RuntimeError("bad zip")))
            self.assertEqual(result["status"], "corrupt")
            quarantined = quarantine_checkpoint(root, result)
            self.assertFalse(checkpoint.exists())
            self.assertTrue(Path(quarantined["quarantined_to"]).is_file())
        finally:
            shutil.rmtree(root, ignore_errors=True)


class GIPFormerProductionAdaptationTests(unittest.TestCase):
    def test_head_only_freeze_selects_complete_module_prefixes(self):
        class Parameter:
            def __init__(self, size):
                self.size = size
                self.requires_grad = None

            def numel(self):
                return self.size

            def requires_grad_(self, enabled):
                self.requires_grad = enabled

        class Model:
            def __init__(self):
                self.params = {
                    "encoder.layers.0.weight": Parameter(10),
                    "decoder.embedding.weight": Parameter(4),
                    "joiner.output.weight": Parameter(6),
                }

            def named_parameters(self):
                return self.params.items()

        model = Model()
        selected, counts = configure_trainable_parameters(model, ["decoder", "joiner"])
        self.assertEqual(len(selected), 2)
        self.assertEqual(counts, {"total": 20, "trainable": 10, "frozen": 10})
        self.assertFalse(model.params["encoder.layers.0.weight"].requires_grad)
        self.assertTrue(model.params["decoder.embedding.weight"].requires_grad)
        with self.assertRaises(ValueError):
            configure_trainable_parameters(model, ["missing"])

    def test_local_audio_staging_rejects_test_and_uses_stable_names(self):
        root = ROOT / "tests" / ".tmp" / "gipformer-stage"
        root.mkdir(parents=True, exist_ok=True)
        audio = root / "sample.wav"
        manifest = root / "rows.jsonl"
        try:
            audio.write_bytes(b"not decoded in this unit test")
            manifest.write_text(
                json.dumps({"audio_path": str(audio), "text": "dung may", "split": "train"}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(len(read_rows(manifest)), 1)
            self.assertEqual(cache_target(audio, root / "cache"), cache_target(audio, root / "cache"))
            manifest.write_text(
                json.dumps({"audio_path": str(audio), "text": "must not train", "split": "test"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                read_rows(manifest)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
