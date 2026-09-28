from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_desktop_regression import select_regression_cases


class RegressionSelectionTests(unittest.TestCase):
    def test_every_failure_and_safety_is_retained_with_fixed_passing_controls(self):
        cases = [{"case_id": str(i), "suite": "approved_safety" if i == 0 else "test_noisy"}
                 for i in range(20)]
        baseline = {row["case_id"]: {"status": "fail" if row["case_id"] in {"2", "5"} else "pass"}
                    for row in cases}
        chosen = select_regression_cases(cases, baseline, controls=3)
        keys = [row["case_id"] for row in chosen]
        self.assertEqual(len(keys), 6)
        self.assertEqual(len(set(keys)), 6)
        self.assertTrue({"0", "2", "5"} <= set(keys))
        self.assertEqual(chosen, select_regression_cases(cases, baseline, controls=3))


if __name__ == "__main__":
    unittest.main()
