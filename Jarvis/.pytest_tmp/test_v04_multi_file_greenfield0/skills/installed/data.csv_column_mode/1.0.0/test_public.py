import unittest

import main


PUBLIC_CASES = [
  {
    "expected": {
      "frequency": 2,
      "value": "red"
    },
    "input": {
      "column": "color",
      "csv_text": "name,color\nAda,red\nBob,blue\nCy,red\n"
    },
    "name": "mode"
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
