from __future__ import annotations

import unittest

from scripts.profile_release_runtime import (
    DEFAULT_HARDWARE_PROFILE,
    resolve_hardware_targets,
)


class HardwareProfileTargetTests(unittest.TestCase):
    def test_default_low_ram_profile_preserves_legacy_targets(self):
        targets, sources = resolve_hardware_targets({}, DEFAULT_HARDWARE_PROFILE)

        self.assertEqual(
            targets,
            {"max_rss_mb": 200.0, "normal_target_ms": 1000.0, "safety_target_ms": 300.0},
        )
        self.assertEqual(set(sources.values()), {"legacy_default"})

    def test_named_profile_and_cli_override_are_recorded_separately(self):
        config = {
            "pipeline": {
                "edge_hardware_profiles": {
                    "device-8gb": {
                        "max_rss_mb": 6000,
                        "normal_target_ms": 900,
                        "safety_target_ms": 250,
                    }
                }
            }
        }

        targets, sources = resolve_hardware_targets(
            config, "device-8gb", max_rss_mb=5500
        )

        self.assertEqual(targets["max_rss_mb"], 5500.0)
        self.assertEqual(targets["normal_target_ms"], 900.0)
        self.assertEqual(targets["safety_target_ms"], 250.0)
        self.assertEqual(sources["max_rss_mb"], "cli")
        self.assertEqual(sources["normal_target_ms"], "config:device-8gb")

    def test_unknown_profile_requires_explicit_memory_budget(self):
        with self.assertRaisesRegex(ValueError, "Unknown hardware profile"):
            resolve_hardware_targets({}, "unregistered-device")


if __name__ == "__main__":
    unittest.main()
