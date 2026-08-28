import unittest
from pathlib import Path

from scripts.benchmark_gipformer_pytorch import parse_upstream_output


class GIPFormerPyTorchBridgeTests(unittest.TestCase):
    def test_parser_keeps_audio_and_transcript_paired(self):
        output = """
  File: /tmp/a.wav
  Text: mọi người dừng lại ngay
  Time: 0.42s | Audio: 1.20s | RTF: 0.350

  File: /tmp/b.wav
  Text: kiểm tra dây an toàn
  Time: 0.50s | Audio: 1.00s | RTF: 0.500
"""
        parsed = parse_upstream_output(output)
        self.assertEqual(parsed[str(Path("/tmp/a.wav").resolve())], ("mọi người dừng lại ngay", 0.42))
        self.assertEqual(parsed[str(Path("/tmp/b.wav").resolve())], ("kiểm tra dây an toàn", 0.5))

    def test_parser_rejects_file_without_text(self):
        with self.assertRaises(ValueError):
            parse_upstream_output("File: /tmp/missing.wav\n")


if __name__ == "__main__":
    unittest.main()
