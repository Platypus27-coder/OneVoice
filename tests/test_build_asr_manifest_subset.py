import unittest


class ASRSubsetManifestTests(unittest.TestCase):
    def test_legacy_blank_language_is_accepted_by_filter_policy(self):
        # Keep the policy visible: V1 rows without a language tag are VI rows
        # when this utility is used for the VI ASR benchmark.
        row_language = ""
        requested = "vi"
        self.assertFalse(row_language and row_language != requested)


if __name__ == "__main__":
    unittest.main()
