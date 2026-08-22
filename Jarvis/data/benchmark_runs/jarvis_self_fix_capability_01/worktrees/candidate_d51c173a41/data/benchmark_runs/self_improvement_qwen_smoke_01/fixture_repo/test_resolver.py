import unittest

from resolver import resolve


class ResolverTests(unittest.TestCase):
    def test_semantic_line_count_reuse(self):
        catalog = {"text.line_count": "Count non-empty lines in supplied text."}
        self.assertEqual(resolve("How many actual lines of content are in this string?", catalog), "text.line_count")


if __name__ == "__main__":
    unittest.main()
