from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from audio.denoise import Denoiser
from context.engine import ConstructionContextEngine
from context.site_pack import SitePackError, SitePackLoader
from contracts import ASRHypothesis, AudioFrame, CommitKind
from evaluation.dataset_audit import audit_audio_manifest
from evaluation.real_site import audit_real_site_manifest, write_holdout_lock
from evaluation.metrics import cer, corpus_error_rate, wer
from runtime.preflight import ArtifactPreflightError, verify_artifacts
from scripts.recover_v1_manifest import UNRECOVERABLE, recover
from streaming.semantic_commit import (
    RollingHypothesisAssembler,
    SemanticCommitController,
    StablePrefixAligner,
)
from streaming.session import RollingUtteranceSession


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


class RuntimeSafetyTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
