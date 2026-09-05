import hashlib
import json
import shutil
import unittest
from pathlib import Path

import yaml

from scripts.audit_mobile_readiness import audit
from scripts.build_release_bundle import build_bundle, write_portable_runtime_config
from scripts.profile_component_memory import _sample


ROOT = Path(__file__).resolve().parents[1]


class MobileReadinessTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / "tests" / ".tmp" / "mobile-readiness"
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _source(self, name: str) -> tuple[Path, str]:
        path = self.root / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode("utf-8"))
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def _runtime_config(self) -> Path:
        path = self.root / "source_config.yaml"
        path.write_text(yaml.safe_dump({
            "audio": {"sample_rate": 16000},
            "pipeline": {"offline": False},
            "asr": {}, "sensevoice": {},
            "translation": {"directions": {"vi2en": {}, "en2vi": {}}},
            "profiles": {"edge": {}},
        }), encoding="utf-8")
        return path

    def test_copied_bundle_has_local_runtime_contract_and_mobile_audit(self):
        entries = []
        for asset in ("gipformer", "mt_vi2en_ort", "safety_audio", "reviewed_safety_csv", "construction_data"):
            source, digest = self._source(f"{asset}.asset")
            entries.append({
                "name": f"{asset}/payload.bin", "path": str(source), "sha256": digest,
                "license": "test", "directions": ["vi2en"], "profiles": ["edge"],
            })
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({
            "schema_version": 2, "sample_rates": [16000],
            "required_backends": [{"name": "ASR", "python_module": "json", "directions": ["vi2en"], "profiles": ["edge"]}],
            "artifacts": entries,
        }), encoding="utf-8")
        bundle = self.root / "bundle"
        receipt = build_bundle(
            manifest, bundle, "vi2en", mode="copy", runtime_config=self._runtime_config()
        )
        self.assertTrue(receipt["portable"])
        config = yaml.safe_load((bundle / "runtime_config.yaml").read_text(encoding="utf-8"))
        self.assertTrue(config["pipeline"]["offline"])
        self.assertEqual(config["asr"]["gipformer_model_dir"], "models/gipformer")
        report = audit(bundle, "vi2en")
        self.assertTrue(report["artifact_portable"])
        self.assertTrue(report["runtime_config_present"])
        self.assertFalse(report["android_app_ready"])
        self.assertEqual(report["blockers"], [])

    def test_runtime_config_refuses_missing_mobile_context_assets(self):
        with self.assertRaisesRegex(ValueError, "construction_data"):
            write_portable_runtime_config(
                self._runtime_config(), self.root / "runtime_config.yaml", "en2vi",
                {"sensevoice_fp32", "mt_en2vi_ort", "safety_audio", "reviewed_safety_csv"},
            )

    def test_component_sample_reports_nonnegative_delta(self):
        class Sampler:
            peak_mb = 10.0

        sampler = Sampler()

        def loader():
            sampler.peak_mb = 14.25

        row = _sample("asr", loader, sampler, 10.0)
        self.assertEqual(row["stage"], "asr")
        self.assertEqual(row["rss_delta_mb"], 4.25)


if __name__ == "__main__":
    unittest.main()
