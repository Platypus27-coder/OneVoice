from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from contracts import ASRHypothesis, CommitKind, ContextResult
from streaming.semantic_commit import SemanticCommitController


def hypothesis(text: str, endpoint: bool = False) -> ASRHypothesis:
    return ASRHypothesis(text, text, "", "en2vi", 1.0, 2.0, endpoint=endpoint)


class EndpointNormalPolicyTests(unittest.TestCase):
    def test_partial_phrases_are_not_translated_independently(self):
        controller = SemanticCommitController(normal_commit_policy="endpoint")
        fragments = ["Check", "Check the safety", "Check the safety helmet before continuing."]
        for text in fragments:
            decision = controller.decide(hypothesis(text), ContextResult(text, text))
            self.assertEqual(decision.kind, CommitKind.WAIT)
            self.assertEqual(decision.reason, "normal_waiting_for_endpoint")
        text = fragments[-1]
        decision = controller.decide(hypothesis(text, endpoint=True), ContextResult(text, text))
        self.assertEqual(decision.kind, CommitKind.NORMAL)
        self.assertEqual(decision.text, text)
        self.assertEqual(controller.decide(hypothesis(text, endpoint=True), ContextResult(text, text)).kind, CommitKind.WAIT)

    def test_confirmed_safety_phrase_still_commits_before_endpoint(self):
        controller = SemanticCommitController(normal_commit_policy="endpoint")
        text = "Stop immediately."
        context = ContextResult(text, text, safety_candidates=[SimpleNamespace(safety_id="SAFE2_0004")])
        self.assertEqual(controller.decide(hypothesis(text), context).kind, CommitKind.WAIT)
        self.assertEqual(controller.decide(hypothesis(text), context).kind, CommitKind.SAFETY)
        self.assertEqual(controller.decide(hypothesis(text, endpoint=True), context).kind, CommitKind.WAIT)

    def test_unknown_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            SemanticCommitController(normal_commit_policy="unknown")


if __name__ == "__main__":
    unittest.main()
