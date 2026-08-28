import unittest

from scripts.prepare_gipformer_finetune_data import is_vietnamese


class PrepareGIPFormerDataTests(unittest.TestCase):
    def test_legacy_blank_language_is_vietnamese(self):
        self.assertTrue(is_vietnamese({}))

    def test_explicit_other_language_is_excluded(self):
        self.assertFalse(is_vietnamese({"language": "en"}))


if __name__ == "__main__":
    unittest.main()
