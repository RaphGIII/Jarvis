import unittest

import main


PUBLIC_CASES = [
  {
    "expected_output": {
      "line_count": 3
    },
    "input": "line1\nline2\n\nline3"
  },
  {
    "expected_output": {
      "line_count": 0
    },
    "input": "   \n\n   \n"
  },
  {
    "expected_output": {
      "line_count": 0
    },
    "input": ""
  },
  {
    "expected_output": {
      "line_count": 3
    },
    "input": "a\nb\nc"
  }
]


class PublicSkillTests(unittest.TestCase):
    def test_public_cases(self):
        for case in PUBLIC_CASES:
            payload = case.get("input", {})
            if case.get("raises"):
                with self.assertRaises(Exception):
                    main.run(payload)
                continue
            result = main.run(payload)
            if "expected" in case:
                self.assertEqual(result, case["expected"], case.get("name", "public case"))
            for key in case.get("expected_keys", []):
                self.assertIsInstance(result, dict)
                self.assertIn(key, result)


if __name__ == "__main__":
    unittest.main()
